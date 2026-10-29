# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Two independent native learners committed together at episode boundaries."""

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.rlt_warmup import file_sha256, json_fingerprint, write_json
from rlinf.models.embodiment.piper_experts import EXPERT_PHASES
from rlinf.runners.piper_online_runner import PiperOnlinePolicy
from rlinf.utils.logging import get_logger

logger = get_logger()


class PiperExpertOnlinePolicy(PiperOnlinePolicy):
    """Carry expert snapshots across the same inference gate as shared rollout."""

    expert_ids = tuple(EXPERT_PHASES)

    def __init__(self, worker, info):
        super().__init__(worker, info)
        self.expert_versions = dict.fromkeys(self.expert_ids, 0)

    def begin_episode(self):
        with self.gate.request():
            state = self.worker.begin_episode().wait()[0]
        self.version = state["version"]
        self.expert_versions = state["expert_versions"]
        return self.version

    def infer(self, observation, task, *, actor_enabled, expert_id=None):
        return self.infer_with_reference(
            observation, task, actor_enabled=actor_enabled, expert_id=expert_id
        )[0]

    def infer_with_reference(self, observation, task, *, actor_enabled, expert_id=None):
        with self.gate.request():
            return self.worker.infer_with_reference(
                observation, task, actor_enabled=actor_enabled, expert_id=expert_id
            ).wait()[0]

    def infer_for_replay(
        self, observation, task, *, actor_enabled, directory, frame, expert_id=None
    ):
        with self.gate.request():
            return self.worker.infer_for_replay(
                observation,
                task,
                actor_enabled=actor_enabled,
                expert_id=expert_id,
                directory=str(directory),
                frame=frame,
            ).wait()[0]


class PiperExpertRunner:
    """Own isolated learners and publish one durable pair before the next trial."""

    async_training = False

    def __init__(self, configs: dict, output: Path, *, resume=False):
        if set(configs) != set(EXPERT_PHASES):
            raise ValueError("Both expert learner configurations are required.")
        self.configs, self.output, self.resume = configs, output, resume
        self.root = output / "online_experts"
        self.actors = {}
        self.rollout = self.policy = None
        self.generation = 0
        self.committed = []
        self.loaded_checkpoint = None
        self.failed = False
        self.binding = json_fingerprint(
            {
                key: {
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "initial_weights": file_sha256(cfg.runner.ckpt_path),
                }
                for key, cfg in configs.items()
            }
        )

    def __enter__(self):
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from rlinf.workers.actor.piper_rlt_worker import PiperRLTActor
        from rlinf.workers.rollout.piper_expert_worker import PiperExpertRollout

        self.root.mkdir(exist_ok=True)
        cfg = self.configs["phase1"]
        cluster = Cluster(
            cluster_cfg=cfg.cluster,
            distributed_log_dir=str(self.output / "native_logs"),
        )
        try:
            for expert, config in self.configs.items():
                placement = HybridComponentPlacement(config, cluster)
                actor = PiperRLTActor.create_group(config).launch(
                    cluster,
                    name=config.actor.group_name,
                    placement_strategy=placement.get_strategy("actor"),
                )
                self.actors[expert] = actor
                actor.init_worker().wait()
            rollout_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            rollout_cfg.piper.experts = {
                key: value.piper.actor for key, value in self.configs.items()
            }
            self.rollout = PiperExpertRollout.create_group(rollout_cfg).launch(
                cluster,
                name=cfg.rollout.group_name,
                placement_strategy=HybridComponentPlacement(cfg, cluster).get_strategy(
                    "rollout"
                ),
            )
            self.policy = PiperExpertOnlinePolicy(
                self.rollout, self.rollout.init_worker().wait()[0]
            )
            latest = self.root / "latest.json"
            if latest.exists():
                if not self.resume:
                    raise ValueError("Expert run already exists; use --resume.")
                saved = json.loads(latest.read_text())
                if saved["binding"] != self.binding:
                    raise ValueError("Expert training contract changed; use a new run.")
                checkpoint = (self.root / saved["checkpoint"]).resolve()
                if not checkpoint.is_relative_to(self.root.resolve()):
                    raise ValueError("Checkpoint escapes the expert run.")
                if json.loads((checkpoint / "state.json").read_text()) != saved:
                    raise ValueError("Expert checkpoint publication is inconsistent.")
                self.loaded_checkpoint = checkpoint
                for expert, actor in self.actors.items():
                    actor.load_checkpoint(str(checkpoint / expert)).wait()
                self.rollout.restore_rng(
                    torch.load(checkpoint / "rollout_rng.pt", weights_only=True)
                ).wait()
                self.committed, self.generation = (
                    saved["committed"],
                    saved["generation"],
                )
                self.rollout.install_pair(str(checkpoint), self.binding).wait()
            else:
                for expert, actor in self.actors.items():
                    config = self.configs[expert]
                    actor.seed_offline_cache(
                        config.piper.offline_cache, EXPERT_PHASES[expert]
                    ).wait()
                self._checkpoint({"initialization": True})
            self.policy.begin_episode()
            for path in sorted(self.output.glob("episode_*/result.json")):
                result = json.loads(path.read_text())
                if result.get("accepted") and path.parent.name not in self.committed:
                    self.accept_episode(path.parent, result)
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def accept_episode(self, directory: Path, result: dict) -> None:
        """Update only experts with new data, then commit the whole pair once."""
        self.check_health()
        if directory.name in self.committed:
            return
        if not result.get("accepted") or result.get("outcome") not in (
            "success",
            "failure",
        ):
            raise ValueError("Only operator-accepted episodes may enter replay.")
        try:
            trajectories = self.rollout.prepare_expert_episode(
                str(directory), result
            ).wait()[0]
            metrics = {}
            for expert, actor in self.actors.items():
                row = {"transitions": 0, "critic_updates": 0, "actor_updates": 0}
                if expert in trajectories:
                    row.update(actor.ingest(trajectories[expert]).wait()[0])
                    while True:
                        update = actor.run_training().wait()[0]
                        if any(not np.isfinite(v) for v in update.values()):
                            raise RuntimeError(
                                "Non-finite expert metrics; no pair published."
                            )
                        row["critic_updates"] += int(update["rlt/critic_updates_run"])
                        row["actor_updates"] += int(update["rlt/actor_updates_run"])
                        row["last_update"] = update
                        with (self.root / f"{expert}_metrics.jsonl").open(
                            "a"
                        ) as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "episode": directory.name,
                                        "time": time.time(),
                                        **update,
                                    }
                                )
                                + "\n"
                            )
                        if (
                            update["rlt/pending_update_budget"] == 0
                            or update["rlt/critic_updates_run"] == 0
                        ):
                            break
                metrics[expert] = row
            self.committed.append(directory.name)
            self._checkpoint({"episode": directory.name, "experts": metrics})
            self.policy.begin_episode()
            write_json(
                directory / "online_update.json",
                {
                    "generation": self.generation,
                    "versions": self.policy.expert_versions,
                    "experts": metrics,
                    "checkpoint": str(self.root / f"checkpoint_{self.generation:06d}"),
                },
            )
        except BaseException:
            # Never continue with one updated learner after a partial failure.
            self.failed = True
            raise

    def _checkpoint(self, metrics: dict) -> None:
        generation = self.generation + 1
        name = f"checkpoint_{generation:06d}"
        temporary = self.root / f".pending_{uuid.uuid4().hex}"
        temporary.mkdir()
        versions, digests = {}, {}
        for expert, actor in self.actors.items():
            versions[expert] = int(actor.get_rollout_sync_version().wait()[0])
            actor.save_checkpoint(str(temporary / expert), versions[expert]).wait()
            digests[expert] = file_sha256(
                temporary / expert / "model_state_dict/full_weights.pt"
            )
        torch.save(self.policy.rng_state(), temporary / "rollout_rng.pt")
        state = {
            "binding": self.binding,
            "checkpoint": name,
            "generation": generation,
            "committed": self.committed.copy(),
            "versions": versions,
            "snapshot_sha256": digests,
            "metrics": metrics,
        }
        write_json(temporary / "state.json", state)
        final = self.root / name
        if final.exists():
            final.rename(self.root / f".orphan_{uuid.uuid4().hex}")
        temporary.rename(final)
        write_json(self.root / "latest.json", state)
        self.rollout.install_pair(str(final), self.binding).wait()
        self.generation = generation
        keep = max(int(cfg.piper.keep_checkpoints) for cfg in self.configs.values())
        for old in sorted(self.root.glob("checkpoint_*"))[:-keep]:
            if old.resolve() != self.loaded_checkpoint:
                shutil.rmtree(old)

    def check_health(self) -> None:
        if self.failed:
            raise RuntimeError(
                "Expert update failed; resume the last committed pair before collecting."
            )

    def finish_training(self, poll) -> None:
        self.check_health()

    def __exit__(self, exc_type, *_):
        failure = None
        for actor in self.actors.values():
            try:
                actor.close_buffers().wait()
            except BaseException as error:
                failure = error
                logger.warning("Expert replay cleanup failed.", exc_info=True)
            finally:
                try:
                    actor._close()
                except BaseException as error:
                    failure = error
                    logger.warning("Expert worker cleanup failed.", exc_info=True)
        self.actors.clear()
        if self.rollout is not None:
            try:
                self.rollout._close()
            except BaseException as error:
                failure = error
                logger.warning("Expert rollout cleanup failed.", exc_info=True)
            self.rollout = None
        if failure is not None and exc_type is None:
            raise failure
