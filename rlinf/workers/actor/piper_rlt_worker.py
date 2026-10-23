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

"""Piper segment semantics on the native RLT FSDP learner."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.segment import (
    action_mask,
    actor_training_observation,
    discounted_target,
    masked_behavior_loss,
)
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy


def segment_loss_batch(batch: dict, chunk: int, dim: int) -> dict:
    """Expose native replay tensors in the shared offline/online loss contract."""
    return {
        **batch["forward_inputs"],
        "curr_obs": batch["curr_obs"],
        "next_obs": batch["next_obs"],
        "actions": batch["actions"].reshape(-1, chunk, dim),
        "human": batch["intervene_flags"].reshape(-1, chunk, dim).any(-1),
        "rewards": batch["rewards"].sum(-1),
        "terminated": batch["terminations"].any(-1),
    }


class PiperRLTActor(RLTACFSDPPolicy):
    """Reuse RLT replay, optimizer steps, targets, checkpoints and weight sync.

    Only the local-terminal losses and additional resume state differ from the
    standard worker. There is no second optimizer loop or replay implementation.
    """

    def init_worker(self) -> None:
        """Start reproducibly before native model, optimizer and replay setup."""
        random.seed(self.cfg.actor.seed)
        np.random.seed(self.cfg.actor.seed)
        torch.manual_seed(self.cfg.actor.seed)
        super().init_worker()
        # These heads are already warmed up. Count the first accepted transitions
        # toward the native update budget instead of subtracting a collection phase.
        if self.use_rlt_schedule:
            self._warmup_ready_total_transitions = 0
            self._warmup_ready_total_episodes = 0

    def get_rollout_sync_version(self) -> int:
        """Version each published policy by the actual critic update count."""
        return int(self.update_step)

    def _critic_actions(
        self, actions: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        chunk, dim = self._chunk_shape()
        bounds = torch.as_tensor(
            self.cfg.piper.driver_action_bounds,
            device=actions.device,
            dtype=actions.dtype,
        )
        actions = actions.reshape(-1, chunk, dim).clamp(min=bounds[0], max=bounds[1])
        return actions * action_mask(lengths, chunk)[..., None]

    def forward_critic(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Use executed targets, tick discounts and segment-local terminals."""
        data = segment_loss_batch(batch, *self._chunk_shape())
        with torch.no_grad():
            next_actions, _, _ = self.model(
                forward_type=ForwardType.SAC, obs=data["next_obs"]
            )
            next_q = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=data["next_obs"],
                actions=self._critic_actions(next_actions, data["next_valid_steps"]),
            )
            target = discounted_target(
                data, self._min_twin_q(next_q).squeeze(-1), self.cfg.algorithm.gamma
            )
        q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=data["curr_obs"],
            actions=data["actions"],
        )
        loss = F.mse_loss(q, target[:, None].expand_as(q))
        return loss, {
            "q_data": q.mean().item(),
            "q_target": target.mean().item(),
            "td_abs": (q - target[:, None]).abs().mean().item(),
        }

    def forward_actor(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Preserve warmup's human conditioning, padding masks and Q projection."""
        data = segment_loss_batch(batch, *self._chunk_shape())
        predicted, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=actor_training_observation(data),
            apply_reference_dropout=True,
            reference_dropout_prob=self.cfg.algorithm.reference_dropout_prob,
        )
        q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=data["curr_obs"],
            actions=self._critic_actions(predicted, data["valid_steps"]),
            detach_encoder=True,
        )
        bc = masked_behavior_loss(predicted, data)
        bc_weight, q_weight, metrics = self._actor_objective_weights()
        q1 = self._q1(q).mean()
        loss = bc_weight * bc - q_weight * q1
        return (
            loss,
            -log_pi.mean(),
            {
                **metrics,
                "reference_dropout_prob": float(
                    self.cfg.algorithm.reference_dropout_prob
                ),
                "bc_loss": bc.item(),
                "q_pi": q1.item(),
                "weighted_bc": (bc_weight * bc).item(),
                "weighted_q": (q_weight * q1).item(),
            },
        )

    def ingest(self, trajectory: Trajectory) -> dict:
        """Commit one accepted episode through the native RLT ingestion path."""
        added, completed = self._ingest_rollout_trajectories([trajectory])
        self._update_rollout_ingest_counters(added, completed)
        return {"transitions": added, "segments": completed}

    def seed_offline_cache(self, path: str) -> int:
        """Populate the separate demo pool using only the warmup training split."""
        from rlinf.data.datasets.piper_online import segment_trajectory
        from rlinf.data.datasets.rlt_warmup import CachedRLTReplay

        replay = CachedRLTReplay(path)
        if (
            self.cfg.piper.get(
                "offline_cache_fingerprint", replay.manifest["content_fingerprint"]
            )
            != replay.manifest["content_fingerprint"]
        ):
            raise ValueError("Offline cache changed since the online run was bound.")
        if replay.manifest["annotations"]["task"] != self.cfg.piper.task:
            raise ValueError("Offline cache task differs from the online task.")
        if replay.reward_scope != "critical_segment" or replay.manifest[
            "stage1"
        ] != OmegaConf.to_container(self.cfg.piper.stage1_manifest, resolve=True):
            raise ValueError(
                "Offline cache must bind to this Stage 1 and local rewards."
            )
        for key in (
            "num_action_chunks",
            "ref_num_action_chunks",
            "action_dim",
            "z_dim",
        ):
            if replay.manifest["model"][key] != self.cfg.actor.model[key]:
                raise ValueError(f"Offline cache model mismatch: {key}.")
        if self.demo_buffer is None:
            raise ValueError("Configure a demo buffer before loading offline data.")
        for index in replay.splits["train"]:
            self.demo_buffer.add_trajectories(
                [segment_trajectory(replay.episodes[index], -1)]
            )
        if self.demo_buffer.total_samples == 0:
            raise ValueError("Offline cache contains no training transitions.")
        return self.demo_buffer.total_samples

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        """Extend native optimizer/target/replay checkpoints with counters and RNGs."""
        super().save_checkpoint(save_base_path, step)
        root = Path(save_base_path)
        if self.demo_buffer is not None:
            self.demo_buffer.save_checkpoint(str(root / "demo_buffer"))
        torch.save(
            {
                "update_step": self.update_step,
                "ingest_counters": {
                    key: getattr(self, key)
                    for key in (
                        "total_transitions_added",
                        "total_episodes_added",
                        "transitions_since_train",
                        "episodes_since_train",
                    )
                },
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "replay_rng": self.replay_buffer.random_generator.get_state(),
                "demo_rng": self.demo_buffer.random_generator.get_state()
                if self.demo_buffer is not None
                else None,
            },
            root / "piper_training.pt",
        )
        metadata = OmegaConf.to_container(self.cfg.piper.actor_metadata, resolve=True)
        metadata.update(
            format="rlinf-piper-rlt-online-segment-actor-v1",
            online_update_step=self.update_step,
        )
        (root / "piper_rlt_actor.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    def load_checkpoint(self, load_base_path: str) -> None:
        """Restore the complete online state; warmup exports use ckpt_path instead."""
        super().load_checkpoint(load_base_path)
        root = Path(load_base_path)
        state = torch.load(
            root / "piper_training.pt", map_location="cpu", weights_only=False
        )
        self.update_step = int(state["update_step"])
        for key, value in state.get("ingest_counters", {}).items():
            setattr(self, key, int(value))
        if self.demo_buffer is not None:
            self.demo_buffer.load_checkpoint(str(root / "demo_buffer"))
            self.demo_buffer.random_generator.set_state(state["demo_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.replay_buffer.random_generator.set_state(state["replay_rng"])

    def close_buffers(self) -> None:
        """Finish pending replay writes before the runner releases this worker."""
        if self.replay_buffer is not None:
            self.replay_buffer.close()
        if self.demo_buffer is not None:
            self.demo_buffer.close()
