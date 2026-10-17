# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.utils import get_rng_state, seed_everything, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def init_worker(self):
        if self.cfg.actor.get("seed_training_rng", False):
            # Match the configured seed on every rank before random modules
            # are initialized. Checkpoint loading subsequently restores RNG.
            seed_everything(self.cfg.actor.seed)
        super().init_worker()

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi import (
                build_openpi_sft_dataloader,
            )

            return build_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.COSMOS3:
            from rlinf.data.datasets.cosmos3 import (
                build_cosmos3_sft_dataloader,
            )

            return build_cosmos3_sft_dataloader(self.cfg, data_paths, eval_dataset)
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.FASTWAM:
            from rlinf.data.datasets.fastwam import build_fastwam_sft_dataloader

            return build_fastwam_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        data_state = None
        if self.cfg.data.get("checkpoint_data_state", False):
            error = None
            try:
                data_state = self._openpi_data_state(step)
            except Exception as exc:
                error = exc
            self._sync_data_state_errors(error, data_state)
        super().save_checkpoint(save_path, step)

        if data_state is not None:
            error = None
            try:
                self._save_piper_inference_assets(save_path)
                path = Path(save_path) / f"openpi_data_rank_{self._rank}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(data_state, temporary)
                temporary.replace(path)
                # Checkpoint serialization must not change subsequent model noise.
                set_rng_state(data_state["rng"])
            except Exception as exc:
                error = exc
            self._sync_data_state_errors(error)
            return

        self._save_piper_inference_assets(save_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def _save_piper_inference_assets(self, save_path: str) -> None:
        if self._rank == 0 and self.cfg.actor.model.get("openpi", {}).get(
            "config_name"
        ) in ("pi05_piper", "pi05_piper_h50_state"):
            from rlinf.models.embodiment.piper_inference import (
                save_piper_inference_assets,
            )

            root = Path(self.cfg.data.train_data_paths)
            fps = json.loads((root / "meta/info.json").read_text())["fps"]
            save_piper_inference_assets(save_path, self.cfg.actor.model, fps=fps)

    def load_checkpoint(self, load_path: str) -> None:
        data_state = None
        if self.cfg.data.get("checkpoint_data_state", False):
            error = None
            try:
                path = Path(load_path) / f"openpi_data_rank_{self._rank}.pt"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {path}; cannot resume OpenPI sampling and RNG "
                        "from a checkpoint without data state."
                    )
                data_state = torch.load(path, map_location="cpu", weights_only=False)
                self._validate_openpi_data_state(data_state)
                # SftRunner resumes from global_step_N/actor. Bind the data
                # cursor to that same step before loading model/optimizer state.
                expected_step = parse_global_step_from_checkpoint_path(
                    Path(load_path).parent
                )
                if data_state["step"] != expected_step:
                    raise ValueError(
                        "OpenPI data checkpoint step differs from its path."
                    )
            except Exception as exc:
                error = exc
            self._sync_data_state_errors(error, data_state)
        super().load_checkpoint(load_path)

        if data_state is not None:
            error = None
            try:
                cursor = data_state["loader"]
                self.data_iter = self.data_loader.restore_checkpoint_state(cursor)
                self._data_epoch = cursor["epoch"]
                self._data_iter_offset = cursor["batch_offset"]
                set_rng_state(data_state["rng"])
            except Exception as exc:
                error = exc
            # A data read can still fail after model loading. All ranks must
            # fail the restore instead of letting peers begin training alone.
            self._sync_data_state_errors(error)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def _sync_data_state_errors(
        self, error: Exception | None, data_state: dict[str, Any] | None = None
    ) -> None:
        """Keep all ranks out of FSDP collectives when one data state is invalid."""
        if torch.distributed.is_initialized():
            results = [None] * torch.distributed.get_world_size()
            message = None if error is None else f"{type(error).__name__}: {error}"
            position = None
            if error is None and data_state is not None:
                cursor = data_state["loader"]
                position = (data_state["step"], cursor["epoch"], cursor["batch_offset"])
            torch.distributed.all_gather_object(results, (message, position))
            failures = [
                f"rank {rank}: {message}"
                for rank, (message, _) in enumerate(results)
                if message is not None
            ]
            if failures:
                raise RuntimeError(
                    "OpenPI checkpoint data-state validation failed; "
                    + "; ".join(failures)
                )
            if len({position for _, position in results}) != 1:
                raise RuntimeError(
                    "OpenPI checkpoint step/epoch/offset differ across ranks."
                )
        elif error is not None:
            raise error

    def _openpi_resume_contract(self) -> dict[str, Any]:
        """Pin inputs and batching to the configuration that produced a cursor."""
        return {
            "data": OmegaConf.to_container(self.cfg.data, resolve=True),
            "model": OmegaConf.to_container(self.cfg.actor.model, resolve=True),
            "seed": self.cfg.actor.seed,
            "seed_training_rng": self.cfg.actor.get("seed_training_rng", False),
            "micro_batch_size": self.cfg.actor.micro_batch_size,
            "global_batch_size": self.cfg.actor.global_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
        }

    def _openpi_data_state(self, step: int) -> dict[str, Any]:
        """Capture rank-local sampler position and training RNG at a step boundary."""
        from rlinf.data.datasets.openpi.official_sft_data_loader import (
            OpenPISftDataLoader,
        )

        if not isinstance(self.data_loader, OpenPISftDataLoader):
            raise TypeError("checkpoint_data_state requires an OpenPI SFT loader.")
        state = {
            "version": 1,
            "step": step,
            "contract": self._openpi_resume_contract(),
            "loader": self.data_loader.checkpoint_state(
                self._data_epoch, self._data_iter_offset
            ),
            "rng": get_rng_state(),
        }
        self._validate_openpi_data_state(state)
        return state

    def _validate_openpi_data_state(self, state: dict[str, Any]) -> None:
        """Fail before model mutation if checkpoint input state is incompatible."""
        from rlinf.data.datasets.openpi.official_sft_data_loader import (
            OpenPISftDataLoader,
        )

        if not isinstance(self.data_loader, OpenPISftDataLoader):
            raise TypeError("checkpoint_data_state requires an OpenPI SFT loader.")
        if state.get("version") != 1:
            raise ValueError("Unsupported OpenPI data checkpoint version.")
        if state.get("contract") != self._openpi_resume_contract():
            raise ValueError(
                "OpenPI checkpoint data/model/batch configuration changed."
            )
        self.data_loader.validate_checkpoint_state(state["loader"])
        step = state.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("OpenPI checkpoint step must be a nonnegative int.")
        cursor = state["loader"]
        consumed_batches = (
            cursor["epoch"] * len(self.data_loader) + cursor["batch_offset"]
        )
        if consumed_batches != step * self.gradient_accumulation:
            raise ValueError(
                "OpenPI checkpoint cursor and completed training step disagree."
            )
        required_rng_keys = {"cpu", "numpy", "random"}
        from rlinf.scheduler import Worker

        if Worker.torch_platform.is_available():
            required_rng_keys.add(Worker.torch_device_type)
        if not required_rng_keys.issubset(state.get("rng", {})):
            raise ValueError(
                "OpenPI checkpoint is missing required training RNG state."
            )
        # Validate every payload on isolated generators, without advancing or
        # replacing training RNG. This must precede model-load collectives.
        try:
            torch.Generator().set_state(state["rng"]["cpu"])
            np.random.RandomState().set_state(state["rng"]["numpy"])
            random.Random().setstate(state["rng"]["random"])
            if Worker.torch_platform.is_available():
                device = torch.device(
                    Worker.torch_device_type, Worker.torch_platform.current_device()
                )
                torch.Generator(device=device).set_state(
                    state["rng"][Worker.torch_device_type]
                )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "OpenPI checkpoint contains invalid training RNG state."
            ) from exc

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi import (
                get_official_openpi_sft_num_batches,
                is_official_openpi_sft_dataloader,
            )

            num_batches = (
                get_official_openpi_sft_num_batches(self.data_loader)
                if is_official_openpi_sft_dataloader(self.data_loader)
                else len(self.data_loader)
            )
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
