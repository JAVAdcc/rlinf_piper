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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.datasets.rlt_warmup import file_sha256, json_fingerprint, write_json
from rlinf.models.embodiment.piper_inference import load_piper_inference_config
from rlinf.runners.piper_async import InferenceGate, PiperAsyncPipeline
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
    offline_cache: Path | None = None,
    resume: bool = False,
) -> DictConfig:
    """Bind a native learner configuration to explicit deployment contracts."""
    cfg = OmegaConf.load(config)
    if offline_cache is not None:
        cfg.piper.offline_cache = str(offline_cache.expanduser().resolve())
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
        or not 1 <= model["num_action_chunks"] <= 50
        or model["ref_num_action_chunks"] != 50
    ):
        raise ValueError("Online training requires RLT AC heads with 1 <= C <= H50.")
    if cfg.piper.get("require_critic_warmup", False):
        critic_steps = metadata.get("critic_warmup_steps", 0)
        for name in ("readiness.json", "original_readiness.json"):
            receipt = actor.parent / name
            if receipt.exists():
                report = json.loads(receipt.read_text())
                if report.get("weights_sha256") != file_sha256(
                    actor / "model_state_dict/full_weights.pt"
                ):
                    raise ValueError(
                        "Critic warmup receipt does not match actor weights."
                    )
                critic_steps = report.get("critic_steps", 0)
                break
        if critic_steps <= 0 and metadata.get("online_update_step", 0) <= 0:
            raise ValueError(
                "Online training requires a warmed-up critic; use eval_shared.sh "
                "for the C50 BC-only actor."
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
    schedule = cfg.algorithm.get("rlt_schedule", {})
    if cfg.piper.get("async_training", False) and not schedule.get("enable", False):
        raise ValueError("Async training requires transition-count update budgeting.")
    if schedule.get("enable", False) and (
        schedule.get("warmup_post_collect_updates", 0) != 0
        or schedule.get("train_every_transitions") != 1
        or schedule.get("train_every_episodes", 0) != 0
        or schedule.get("max_updates_per_train_step", 0) < 1
        or schedule.get("warmup_min_size", 0) < 1
    ):
        raise ValueError("Use transition-count scheduling without a second warmup.")
    if cfg.piper.get("require_offline_cache", False) and not cfg.piper.offline_cache:
        raise ValueError(
            "Online training requires --offline-cache with the warmup train split."
        )
    if (
        not 0
        < cfg.piper.get("replay_stride", model["num_action_chunks"])
        <= model["num_action_chunks"]
    ):
        raise ValueError("Replay stride must be in [1, actor horizon].")
    if cfg.piper.get("replay_feature_batch_size", 1) < 1:
        raise ValueError("Replay feature batch size must be positive.")
    if cfg.piper.get("replay_tail_policy", "pad") not in ("pad", "align"):
        raise ValueError("Replay tail policy must be pad or align.")
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
        if not resume:
            cfg.algorithm.demo_buffer = OmegaConf.merge(
                cfg.algorithm.replay_buffer,
                {
                    "auto_save_path": str(output / "demo_replay"),
                    "min_buffer_size": 1,
                    "sample_window_size": 0,
                    "add_interventions": False,
                },
            )
            manifest_path = Path(cfg.piper.offline_cache) / "manifest.json"
            cache_manifest = json.loads(manifest_path.read_text())
            if (
                cache_manifest["stage1"] != manifest
                or cache_manifest["annotations"]["task"] != task
            ):
                raise ValueError("Offline cache Stage 1 or task differs from this run.")
            for key in (
                "num_action_chunks",
                "ref_num_action_chunks",
                "action_dim",
                "z_dim",
            ):
                if cache_manifest["model"][key] != model[key]:
                    raise ValueError(f"Offline cache model mismatch: {key}.")
            cfg.piper.offline_cache_fingerprint = cache_manifest["content_fingerprint"]
    return cfg


class PiperOnlinePolicy:
    """Local controller API backed by a dedicated native rollout worker."""

    manual_critical = True
    record_replay_features = True

    def __init__(self, worker, info: dict, codec=None) -> None:
        self.worker = worker
        self.manifest = info["manifest"]
        self.fps = info["fps"]
        self.action_horizon = info["action_horizon"]
        self.actor_horizon = info["actor_horizon"]
        self.codec = codec
        self.version = 0
        self.gate = InferenceGate()
        self.cancel_backfill = lambda: False

    def infer(self, observation: dict, task: str, *, actor_enabled: bool) -> np.ndarray:
        """Return actions and track the version actually used by the rollout worker."""
        return self.infer_with_reference(
            observation, task, actor_enabled=actor_enabled
        )[0]

    def infer_with_reference(
        self,
        observation: dict,
        task: str,
        *,
        actor_enabled: bool,
    ) -> tuple[np.ndarray, dict]:
        """Carry paired reference diagnostics across the worker boundary."""
        queued = time.monotonic()
        with self.gate.request():
            wait_s = time.monotonic() - queued
            actions, diagnostics = self.worker.infer_with_reference(
                observation, task, actor_enabled=actor_enabled
            ).wait()[0]
            self.version = int(diagnostics["policy_version"])
        return actions, {**diagnostics, "background_wait_s": wait_s}

    def infer_for_replay(
        self,
        observation: dict,
        task: str,
        *,
        actor_enabled: bool,
        directory: Path,
        frame: int,
    ) -> tuple[np.ndarray, dict]:
        """Persist request-frame features and keep a stable version for the chunk."""
        queued = time.monotonic()
        with self.gate.request():
            wait_s = time.monotonic() - queued
            actions, diagnostics = self.worker.infer_for_replay(
                observation,
                task,
                actor_enabled=actor_enabled,
                directory=str(directory),
                frame=frame,
            ).wait()[0]
            self.version = int(diagnostics["policy_version"])
        return actions, {**diagnostics, "background_wait_s": wait_s}

    def extract_replay_features(
        self, observations: list[dict], task: str, seed: int
    ) -> list[dict]:
        """Yield between backfill batches so inference has priority on the shared VLA."""
        with self.gate.request(background=True, cancelled=self.cancel_backfill):
            return self.worker.extract_replay_features(observations, task, seed).wait()[
                0
            ]

    def rng_state(self) -> dict:
        """Capture a complete rollout RNG state between RPCs, without model copying."""
        with self.gate.request(background=True):
            return self.worker.rng_state().wait()[0]


class PiperOnlineRunner:
    """Own two native workers and atomically commit accepted episode updates.

    Right closes the undo window. Async mode prepares and learns in background,
    publishing complete checkpoints for rollout installation at chunk boundaries.
    Resume replays accepted but uncommitted episodes exactly once relative to the
    last complete learner checkpoint. Older run snapshots retain synchronous mode.
    """

    def __init__(self, cfg: DictConfig, output: Path, *, resume: bool = False) -> None:
        self.cfg = cfg
        self.output = output
        self.resume = resume
        self.actor = self.rollout = self.policy = None
        self.committed: list[str] = []
        self.generation = 0
        self.loaded_checkpoint = None
        self.async_training = bool(cfg.piper.get("async_training", False))
        self.background = None
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
            from rlinf.models.embodiment.openpi.policies.piper_policy import (
                PiperActionCodec,
            )

            feature_cfg, _ = load_piper_inference_config(self.cfg.piper.stage1)
            self.policy = PiperOnlinePolicy(
                self.rollout,
                info,
                PiperActionCodec(feature_cfg.openpi_data.norm_stats_path),
            )
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
            if self.async_training:
                self.rollout.enable_snapshots(str(self.root), self.binding).wait()
                self.background = PiperAsyncPipeline(
                    self._prepare_async, self._update_async
                )
                self.policy.cancel_backfill = self.background.stopped.is_set
                self.background.start()
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
        """Accept durable data, asynchronously when configured."""
        if directory.name in self.committed:
            return
        if not result.get("accepted") or result["outcome"] not in (
            "success",
            "failure",
        ):
            raise ValueError("Cannot train on an undoable or discarded episode.")
        if self.background is not None:
            self.background.submit(directory, result)
            logger.info(
                "Accepted %s; queued for background preparation and training.",
                directory.name,
            )
            return
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

    def check_health(self) -> None:
        """Report background errors at the next operator/control poll."""
        if self.background is not None:
            self.background.check_health()

    def finish_training(self, poll: Callable) -> None:
        """Drain final accepted data while the arm is stopped; q preserves pending work."""
        if self.background is None:
            return
        logger.info("Finishing background training; q saves pending work for resume.")
        while True:
            if "q" in poll():
                raise KeyboardInterrupt
            try:
                self.background.wait_idle(timeout=0.1)
                return
            except TimeoutError:
                continue

    def _prepare_async(self, directory: Path, result: dict):
        from rlinf.data.datasets.piper_online import cache_accepted_episode

        return cache_accepted_episode(
            self.policy,
            directory,
            result,
            self.cfg.piper.task,
            stride=self.cfg.piper.replay_stride,
            tail_policy=self.cfg.piper.replay_tail_policy,
            feature_batch_size=self.cfg.piper.replay_feature_batch_size,
        )

    def _update_async(self, item) -> bool:
        started = time.monotonic()
        metrics = {"transitions": 0, "segments": 0}
        if item is not None:
            directory, result, trajectory = item
            if trajectory is not None:
                metrics.update(self.actor.ingest(trajectory).wait()[0])
        learner_started_at = time.time()
        metrics.update(self.actor.run_training().wait()[0])
        learner_finished_at = time.time()
        if any(not np.isfinite(value) for value in metrics.values()):
            raise RuntimeError(
                "Non-finite background training metrics; no policy published."
            )
        updates = int(metrics["rlt/critic_updates_run"])
        if item is None and updates == 0:
            return False
        version = int(self.actor.get_rollout_sync_version().wait()[0])
        if item is not None:
            self.committed.append(directory.name)
            metrics["episode"] = directory.name
        metrics.update(
            version=version,
            training_s=time.monotonic() - started,
            learner_started_at=learner_started_at,
            learner_finished_at=learner_finished_at,
        )
        self._checkpoint(metrics, version=version)
        metrics["elapsed_s"] = time.monotonic() - started
        metrics["last_inferred_version"] = self.policy.version
        with (self.root / "async_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        write_json(self.root / "async_status.json", metrics)
        if item is not None:
            write_json(directory / "online_update.json", metrics)
        logger.info(
            "Background update published: version %d, pending budget %d.",
            version,
            int(metrics["rlt/pending_update_budget"]),
        )
        return updates > 0 and metrics["rlt/pending_update_budget"] > 0

    def _checkpoint(self, metrics: dict, *, version: int | None = None) -> None:
        version = self.policy.version if version is None else version
        self.generation += 1
        name = f"checkpoint_{self.generation:06d}"
        temporary = self.root / f".pending_{uuid.uuid4().hex}"
        temporary.mkdir()
        self.actor.save_checkpoint(str(temporary / "actor"), version).wait()
        torch.save(self.policy.rng_state(), temporary / "rollout_rng.pt")
        state = {
            "binding": self.binding,
            "checkpoint": name,
            "generation": self.generation,
            "committed": self.committed.copy(),
            "version": version,
            "snapshot_sha256": file_sha256(
                temporary / "actor/model_state_dict/full_weights.pt"
            ),
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

    def __exit__(self, exc_type, *_):
        import ray

        failure = None
        if self.background is not None:
            try:
                self.background.close()
            except BaseException as error:
                failure = error
            self.background = None

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

        if failure is not None and exc_type is None:
            raise failure
