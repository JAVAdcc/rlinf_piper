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

"""GPU inference and native weight reception for the local Piper controller."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.datasets.piper_online import cache_accepted_episode
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models.embodiment.piper_inference import PiperRLTPolicy
from rlinf.scheduler import Worker


class PiperRLTRollout(Worker):
    """Own one frozen VLA and one actor; hardware stays in the terminal process."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.version = 0
        self.policy = None
        self.weight_syncer = WeightSyncer.create(cfg.weight_syncer)

    def init_worker(self) -> dict:
        """Load the Stage 1 H50 model and its bound C10 actor exactly once."""
        torch.manual_seed(self.cfg.actor.seed)
        self.policy = PiperRLTPolicy(
            self.cfg.piper.stage1,
            actor=self.cfg.piper.actor,
            device="cuda:0",
            manual_critical=True,
            allow_offline_actor=self.cfg.piper.allow_offline_actor,
        )
        return {
            "manifest": self.policy.manifest,
            "fps": self.policy.fps,
            "action_horizon": self.policy.action_horizon,
            "actor_horizon": self.policy.actor_horizon,
        }

    def infer(self, observation: dict, task: str, *, actor_enabled: bool) -> np.ndarray:
        """Sample actor exploration only in explicitly selected segments."""
        return self.policy.infer(
            observation, task, actor_enabled=actor_enabled, mode="train"
        )

    def prepare_episode(self, directory: str, result: dict) -> Trajectory | None:
        """Build a CPU trajectory after collection has closed the undo window."""
        return cache_accepted_episode(
            self.policy, Path(directory), result, self.cfg.piper.task
        )

    async def sync_model_from_actor(self) -> dict:
        """Receive the standard RLT actor broadcast using RLinf's WeightSyncer."""
        options = self.weight_syncer.comm_options

        async def recv():
            return await self.broadcast(
                None,
                groups=[(self.cfg.actor.group_name, 0), (self._group_name, [0])],
                src=(self.cfg.actor.group_name, 0),
                async_op=True,
                options=options,
            ).async_wait()

        async def send(data):
            await self.send(
                data,
                dst_group_name=self.cfg.actor.group_name,
                dst_rank=0,
                async_op=True,
                options=options,
            ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.policy.actor.state_dict(), recv=recv, send=send
            )
        self.version = int(await self.weight_syncer.apply(self.policy.actor, recv))
        return self.status()

    def status(self) -> dict:
        """Report the policy version and digest actually installed for execution."""
        digest = hashlib.sha256()
        for name, tensor in self.policy.actor.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return {"version": self.version, "weights_sha256": digest.hexdigest()}

    def rng_state(self) -> dict:
        """Capture diffusion and exploration generators at an episode boundary."""
        return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

    def restore_rng(self, state: dict) -> None:
        """Restore the boundary generators before the next rollout."""
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])
