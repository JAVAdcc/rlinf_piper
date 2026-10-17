# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    norm_stats_path_from_data_kwargs,
    select_openpi_norm_stats,
)


class OpenPISftDataLoader:
    """Expose one OpenPI data pass and its sampler to the SFT worker.

    OpenPI repeats its inner loader indefinitely. SFT workers instead advance
    the sampler epoch after ``StopIteration``. Bound each iterator to one pass
    without changing OpenPI's batch conversion or consuming the next pass.
    """

    def __init__(self, data_loader: Any) -> None:
        self._openpi_loader = data_loader
        self._torch_loader = _get_openpi_torch_loader(data_loader)
        if len(self._torch_loader) == 0:
            raise ValueError(
                "OpenPI SFT has no complete batches on this rank; check the "
                "dataset size, world size, and micro batch size."
            )

    @property
    def sampler(self) -> Any:
        """Return the sampler that receives the worker's epoch updates."""
        return self._torch_loader.sampler

    @property
    def dataset(self) -> Any:
        """Return the dataset used by the inner PyTorch loader."""
        return self._torch_loader.dataset

    def data_config(self) -> Any:
        """Return the original OpenPI data configuration."""
        return self._openpi_loader.data_config()

    def __len__(self) -> int:
        return len(self._torch_loader)

    def __iter__(self) -> Iterator[Any]:
        return islice(self._openpi_loader, len(self))

    def _checkpoint_contract(self) -> dict[str, Any]:
        """Describe the distributed batch stream that a cursor belongs to."""
        from torch.utils.data import DistributedSampler

        if not isinstance(self.sampler, DistributedSampler):
            raise TypeError(
                "OpenPI sampler checkpointing requires a DistributedSampler."
            )
        return {
            "dataset_size": len(self.dataset),
            "repo_id": str(self.data_config().repo_id),
            "batch_size": self._torch_loader.batch_size,
            "num_workers": self._torch_loader.num_workers,
            "batches_per_epoch": len(self),
            "drop_last": self._torch_loader.drop_last,
            "world_size": self.sampler.num_replicas,
            "rank": self.sampler.rank,
            "sampler_seed": self.sampler.seed,
            "sampler_shuffle": self.sampler.shuffle,
            "sampler_drop_last": self.sampler.drop_last,
        }

    def checkpoint_state(self, epoch: int, batch_offset: int) -> dict[str, Any]:
        """Capture the worker's consumed cursor, excluding prefetched batches.

        This supports deterministic dataset transforms. It does not snapshot
        RNG state inside persistent data workers; model RNG is saved by the
        SFT worker separately.
        """
        state = {
            "version": 1,
            "contract": self._checkpoint_contract(),
            "epoch": epoch,
            "batch_offset": batch_offset,
            "generator_state": self._torch_loader.generator.get_state().clone(),
        }
        self.validate_checkpoint_state(state)
        if self.sampler.epoch != epoch:
            raise ValueError("OpenPI sampler epoch and worker epoch disagree.")
        return state

    def validate_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Reject incompatible or invalid sampler state before restoring it."""
        import torch

        if state.get("version") != 1:
            raise ValueError("Unsupported OpenPI sampler checkpoint version.")
        if state.get("contract") != self._checkpoint_contract():
            raise ValueError("OpenPI sampler checkpoint does not match this loader.")
        epoch, offset = state.get("epoch"), state.get("batch_offset")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("OpenPI checkpoint epoch must be a nonnegative int.")
        if type(offset) is not int or not 0 <= offset <= len(self):
            raise ValueError("OpenPI checkpoint batch offset is outside the epoch.")
        # Validate without modifying the live loader's generator.
        try:
            torch.Generator().set_state(state["generator_state"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "OpenPI checkpoint contains invalid loader RNG state."
            ) from exc

    def restore_checkpoint_state(self, state: dict[str, Any]) -> Iterator[Any]:
        """Replay the consumed epoch prefix and return the next unread batch.

        Replay performs data reads only. Exact batch contents require
        deterministic dataset transforms; it does not restore arbitrary RNG
        inside persistent data workers. Restore the training RNG after replay.
        """
        self.validate_checkpoint_state(state)
        self.sampler.set_epoch(state["epoch"])
        iterator = iter(self)
        for _ in range(state["batch_offset"]):
            next(iterator)
        self._torch_loader.generator.set_state(state["generator_state"])
        return iterator


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    if not norm_stats_path_from_data_kwargs(data_kwargs):
        select_openpi_norm_stats(None, norm_stats_path=None)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=data_kwargs,
    )
    config = dataclasses.replace(
        config,
        num_workers=int(
            OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
        ),
        seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
    )
    _validate_openpi_model_shape(model_cfg, config)

    if Path(repo_id).is_dir():
        # Explicit local roots and a task mapping work for both LeRobot v2/v3.
        import openpi.transforms as transforms
        import torch

        from rlinf.data.datasets.openpi.lerobot_dataset import LeRobotSFTDataset
        from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch
        from rlinf.data.storage.lerobot.metadata import load_task_descriptions

        lerobot = openpi_data_loader._import_lerobot_dataset()
        root = Path(repo_id)
        metadata = lerobot.LeRobotDatasetMetadata(root.name, root=root)
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = LeRobotSFTDataset(
            root.name,
            root=root,
            download_videos=False,
            video_backend=OmegaConf.select(cfg, "data.video_backend", default=None),
            delta_timestamps={
                key: [t / metadata.fps for t in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        if data_config.prompt_from_task:
            dataset = openpi_data_loader.TransformedDataset(
                dataset,
                [transforms.PromptFromLeRobotTask(load_task_descriptions(root))],
            )
        dataset = openpi_data_loader.transform_dataset(dataset, data_config)
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not eval_dataset,
            drop_last=True,
        )
        loader = openpi_data_loader.TorchDataLoader(
            dataset,
            local_batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=config.num_workers,
            seed=config.seed,
            framework="pytorch",
        )
        data_loader = openpi_data_loader.DataLoaderImpl(data_config, loader)
        return OpenPISftDataLoader(data_loader), data_config

    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=not eval_dataset
    )
    return OpenPISftDataLoader(data_loader), data_loader.data_config()


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    if isinstance(data_loader, OpenPISftDataLoader):
        return len(data_loader)
    return len(_get_openpi_torch_loader(data_loader))


def _get_openpi_torch_loader(data_loader: Any) -> Any:
    """Resolve OpenPI's map-style PyTorch loader at the integration boundary."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "torch_loader", None)
    if torch_loader is None:
        torch_loader = getattr(openpi_loader, "_data_loader", None)
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return torch_loader


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return (
        isinstance(data_loader, OpenPISftDataLoader)
        or getattr(data_loader, "_data_loader", None) is not None
    )


def _validate_openpi_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config.

    The official loader sizes the SFT action window from
    ``TrainConfig.model.action_horizon``, not ``num_action_chunks`` (that field
    is the env-executed chunk). ``get_model`` already builds the network from
    the same TrainConfig unless YAML overrides ``openpi.action_horizon``.
    """
    yaml_horizon = OmegaConf.select(model_cfg, "openpi.action_horizon", default=None)
    official_horizon = int(openpi_config.model.action_horizon)
    if yaml_horizon is not None and int(yaml_horizon) != official_horizon:
        raise ValueError(
            "openpi SFT data uses TrainConfig.model.action_horizon="
            f"{official_horizon} from {model_cfg.openpi.config_name}; "
            f"openpi.action_horizon={int(yaml_horizon)} would build a different "
            "network. Unset it, or change the TrainConfig."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
