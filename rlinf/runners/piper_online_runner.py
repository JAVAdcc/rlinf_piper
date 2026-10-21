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

"""Drive native RLT workers at operator-confirmed Piper episode boundaries."""

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.datasets.rlt_warmup import file_sha256, json_fingerprint, write_json
from rlinf.models.embodiment.piper_inference import load_piper_inference_config
from rlinf.utils.logging import get_logger

logger = get_logger()


def online_config(
    config: Path,
    *,
    stage1: Path,
    actor: Path,
    hardware: dict,
    task: str,
    output: Path,
    allow_offline_actor: bool,
) -> DictConfig:
    """Bind a native learner configuration to explicit deployment contracts."""
    cfg = OmegaConf.load(config)
    if (
        cfg.actor.fsdp_config.strategy != "fsdp"
        or cfg.actor.fsdp_config.use_orig_params
    ):
        raise ValueError(
            "Use the supported native FSDP profile with use_orig_params=false."
        )
    if int(cfg.cluster.num_nodes) != 1 or any(
        int(v) != 0 for v in cfg.cluster.component_placement.values()
    ):
        raise ValueError("Piper online currently supports one local GPU (rank 0).")
    actor = actor.resolve()
    if not (actor / "piper_rlt_actor.json").exists():
        actor /= "actor"
    metadata = json.loads((actor / "piper_rlt_actor.json").read_text())
    model_cfg, manifest = load_piper_inference_config(stage1)
    if metadata.get("stage1_checkpoint_id") != manifest["checkpoint_id"]:
        raise ValueError("Warmup actor and Stage 1 checkpoint disagree.")
    if (
        metadata.get("reward_scope") != "critical_segment"
        or metadata.get("phase_filter") is not None
        or metadata.get("critic_action_projection") != "driver_limits"
    ):
        raise ValueError(
            "Online training requires shared segment heads with driver-limit Q projection."
        )
    if (
        metadata["format"] == "rlinf-piper-rlt-offline-local-actor-v1"
        and not allow_offline_actor
    ):
        raise ValueError(
            "The offline-local actor requires explicit experimental opt-in."
        )
    model = metadata["model"]
    if (
        model["model_type"] != "rlt_mlp_policy"
        or model["num_action_chunks"] != 10
        or model["ref_num_action_chunks"] != 50
    ):
        raise ValueError(
            "This online profile requires the deployed H50/C10 RLT AC heads."
        )
    if (
        model_cfg.openpi.action_chunk != 50
        or model_cfg.precision != "fp32"
        or model["precision"] != "fp32"
    ):
        raise ValueError("Piper online requires the FP32 H50 Stage 1 contract.")
    if (
        cfg.algorithm.loss_type != "rlt_ac"
        or cfg.algorithm.replay_buffer.critical_phase_only
        or not cfg.algorithm.replay_buffer.auto_save
    ):
        raise ValueError("Use native RLT AC with durable, already segmented replay.")
    if cfg.algorithm.get("rlt_schedule", {}).get("enable", False):
        raise ValueError(
            "Online initialization must not rerun the native warmup schedule."
        )
    if cfg.algorithm.get("demo_buffer") is not None and not cfg.piper.offline_cache:
        raise ValueError(
            "A demo pool requires an offline cache to avoid waiting on an empty pool."
        )
    if (
        cfg.actor.micro_batch_size < 1
        or cfg.actor.global_batch_size % cfg.actor.micro_batch_size
        or cfg.actor.global_batch_size < 2
    ):
        raise ValueError("Use a batch >= 2 divisible by the micro batch.")
    for value in (
        cfg.algorithm.update_epoch,
        cfg.algorithm.critic_actor_ratio,
        cfg.piper.keep_checkpoints,
    ):
        if int(value) < 1:
            raise ValueError("Update counts and checkpoint retention must be positive.")

    from rlinf.models.embodiment.openpi.policies.piper_policy import PiperActionCodec
    from rlinf.robotics.parts.arms.piper import PiperArm

    codec = PiperActionCodec(model_cfg.openpi_data.norm_stats_path)
    bounds = np.stack(
        (
            np.r_[np.rad2deg(PiperArm.JOINT_LIMITS_LOWER), 0.0],
            np.r_[
                np.rad2deg(PiperArm.JOINT_LIMITS_UPPER),
                hardware["execution"]["gripper_max_width_m"] * 1000,
            ],
        )
    )
    cfg.actor.model = model
    cfg.actor.model.is_lora = False
    cfg.actor.model.model_path = None
    cfg.runner.ckpt_path = str(actor / "model_state_dict/full_weights.pt")
    cfg.runner.logger.log_path = str(output / "native_logs")
    native_fsdp = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/hybrid_engines/fsdp.yaml"
    )
    cfg.actor.fsdp_config = OmegaConf.merge(
        OmegaConf.load(native_fsdp), cfg.actor.fsdp_config
    )
    cfg.piper.stage1 = str(stage1.resolve())
    cfg.piper.actor = str(actor)
    cfg.piper.actor_metadata = metadata
    cfg.piper.stage1_manifest = manifest
    cfg.piper.allow_offline_actor = allow_offline_actor
    cfg.piper.driver_action_bounds = codec.encode(bounds).tolist()
    cfg.piper.task = task
    cfg.algorithm.replay_buffer.auto_save_path = str(output / "replay")
    if cfg.piper.offline_cache:
        if cfg.actor.global_batch_size % 2:
            raise ValueError("Mixed offline/online replay requires an even batch size.")
        cfg.piper.offline_cache = str(
            Path(cfg.piper.offline_cache).expanduser().resolve()
        )
        cfg.algorithm.demo_buffer = OmegaConf.merge(
            cfg.algorithm.replay_buffer,
            {
                "auto_save_path": str(output / "demo_replay"),
                "min_buffer_size": 1,
            },
        )
    return cfg


class PiperOnlinePolicy:
    """Local controller API backed by a dedicated native rollout worker."""

    manual_critical = True

    def __init__(self, worker, info: dict) -> None:
        self.worker = worker
        self.manifest = info["manifest"]
        self.fps = info["fps"]
        self.action_horizon = info["action_horizon"]
        self.actor_horizon = info["actor_horizon"]
        self.version = 0

    def infer(self, observation: dict, task: str, *, actor_enabled: bool) -> np.ndarray:
        """Keep inference separate from local CAN, keyboard and human takeover."""
        return self.worker.infer(observation, task, actor_enabled=actor_enabled).wait()[
            0
        ]


class PiperOnlineRunner:
    """Own two native workers and atomically commit accepted episode updates.

    The terminal owns collection and calls ``accept_episode`` only after Right
    closes the undo window. Updates and weight broadcasts finish before another
    episode can start. Resume replays accepted but uncommitted episodes exactly
    once relative to the last complete learner checkpoint.
    """

    def __init__(self, cfg: DictConfig, output: Path, *, resume: bool = False) -> None:
        self.cfg = cfg
        self.output = output
        self.resume = resume
        self.actor = self.rollout = self.policy = None
        self.committed: list[str] = []
        self.generation = 0
        self.loaded_checkpoint = None
        self.root = output / "online"
        self.binding = json_fingerprint(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "initial_weights": file_sha256(cfg.runner.ckpt_path),
            }
        )

    def __enter__(self):
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.actor.piper_rlt_worker import PiperRLTActor
        from rlinf.workers.rollout.piper_rlt_worker import PiperRLTRollout

        self.root.mkdir(exist_ok=True)
        cluster = Cluster(
            cluster_cfg=self.cfg.cluster,
            distributed_log_dir=str(self.output / "native_logs"),
        )
        placement = HybridComponentPlacement(self.cfg, cluster)
        try:
            self.actor = PiperRLTActor.create_group(self.cfg).launch(
                cluster,
                name=self.cfg.actor.group_name,
                placement_strategy=placement.get_strategy("actor"),
            )
            self.rollout = PiperRLTRollout.create_group(self.cfg).launch(
                cluster,
                name=self.cfg.rollout.group_name,
                placement_strategy=placement.get_strategy("rollout"),
            )
            self.actor.init_worker().wait()
            info = self.rollout.init_worker().wait()[0]
            self.policy = PiperOnlinePolicy(self.rollout, info)
            latest = self.root / "latest.json"
            if latest.exists():
                if not self.resume:
                    raise ValueError("Online checkpoints already exist; use --resume.")
                saved = json.loads(latest.read_text())
                if saved["binding"] != self.binding:
                    raise ValueError(
                        "Online training contract changed; use a new run directory."
                    )
                checkpoint = self.root / saved["checkpoint"]
                self.loaded_checkpoint = checkpoint
                self.actor.load_checkpoint(str(checkpoint / "actor")).wait()
                self.rollout.restore_rng(
                    torch.load(checkpoint / "rollout_rng.pt", weights_only=True)
                ).wait()
                self.committed = saved["committed"]
                self.generation = saved["generation"]
            elif self.cfg.piper.offline_cache:
                self.actor.seed_offline_cache(self.cfg.piper.offline_cache).wait()
            self._sync()
            if not latest.exists():
                self._checkpoint({"initialization": True})
            # Recover the durable acceptance marker if the learner was interrupted.
            for path in sorted(self.output.glob("episode_*/result.json")):
                result = json.loads(path.read_text())
                if result.get("accepted") and path.parent.name not in self.committed:
                    self.accept_episode(path.parent, result)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _sync(self) -> dict:
        receive = self.rollout.sync_model_from_actor()
        send = self.actor.sync_model_to_rollout()
        send.wait()
        status = receive.wait()[0]
        self.policy.version = int(status["version"])
        return status

    def accept_episode(self, directory: Path, result: dict) -> None:
        """Ingest, update, checkpoint and install the next policy while stopped."""
        if directory.name in self.committed:
            return
        if not result.get("accepted") or result["outcome"] not in (
            "success",
            "failure",
        ):
            raise ValueError("Cannot train on an undoable or discarded episode.")
        started = time.monotonic()
        logger.info(
            "Accepted %s; caching segment features and updating RLT.", directory.name
        )
        trajectory = self.rollout.prepare_episode(str(directory), result).wait()[0]
        metrics = {"transitions": 0, "segments": 0}
        if trajectory is not None:
            metrics.update(self.actor.ingest(trajectory).wait()[0])
            metrics.update(self.actor.run_training().wait()[0])
        if any(not np.isfinite(value) for value in metrics.values()):
            raise RuntimeError(
                "Non-finite training metrics; refusing to publish this policy."
            )
        status = self._sync()
        self.committed.append(directory.name)
        metrics.update(
            status, episode=directory.name, training_s=time.monotonic() - started
        )
        self._checkpoint(metrics)
        metrics["elapsed_s"] = time.monotonic() - started
        write_json(directory / "online_update.json", metrics)
        logger.info(
            "Online update complete: version %s, %.2fs. Next episode is ready.",
            self.policy.version,
            metrics["elapsed_s"],
        )

    def _checkpoint(self, metrics: dict) -> None:
        self.generation += 1
        name = f"checkpoint_{self.generation:06d}"
        temporary = self.root / f".pending_{uuid.uuid4().hex}"
        temporary.mkdir()
        self.actor.save_checkpoint(str(temporary / "actor"), self.policy.version).wait()
        torch.save(self.rollout.rng_state().wait()[0], temporary / "rollout_rng.pt")
        state = {
            "binding": self.binding,
            "checkpoint": name,
            "generation": self.generation,
            "committed": self.committed.copy(),
            "version": self.policy.version,
            "metrics": metrics,
        }
        write_json(temporary / "state.json", state)
        final = self.root / name
        # An orphaned complete generation can exist after a crash before pointer publication.
        if final.exists():
            final.rename(self.root / f".orphan_{uuid.uuid4().hex}")
        temporary.rename(final)
        write_json(self.root / "latest.json", state)
        # Native replay now points only to its live store or the loaded checkpoint.
        # Keep that loaded generation for this process in addition to recent ones.
        checkpoints = sorted(self.root.glob("checkpoint_*"))
        for old in checkpoints[: -int(self.cfg.piper.keep_checkpoints)]:
            if old != self.loaded_checkpoint:
                shutil.rmtree(old)

    def __exit__(self, *_):
        import ray

        for name in ("actor", "rollout"):
            group = getattr(self, name)
            setattr(self, name, None)
            if group is None or not ray.is_initialized():
                continue
            try:
                if name == "actor":
                    group.close_buffers().wait()
            except Exception:
                logger.warning(
                    "Learner already unavailable during cleanup.", exc_info=True
                )
            finally:
                if ray.is_initialized():
                    try:
                        group._close()
                    except Exception:
                        logger.warning(
                            "Worker already stopped during cleanup.", exc_info=True
                        )
