# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""One frozen VLA with two explicitly routed and atomically installed heads."""

import json
from pathlib import Path

import torch

from rlinf.data.datasets.piper_online import expert_trajectories, save_replay_anchor
from rlinf.data.datasets.rlt_warmup import file_sha256
from rlinf.models.embodiment.piper_experts import PiperExpertPolicy
from rlinf.workers.rollout.piper_rlt_worker import PiperRLTRollout


class PiperExpertRollout(PiperRLTRollout):
    """Keep both actor versions fixed until a complete episode update is installed."""

    def init_worker(self) -> dict:
        torch.manual_seed(self.cfg.actor.seed)
        self.policy = PiperExpertPolicy(
            self.cfg.piper.stage1,
            experts=dict(self.cfg.piper.experts),
            allow_offline_actor=self.cfg.piper.allow_offline_actor,
        )
        self.expert_versions = dict.fromkeys(self.policy.expert_ids, 0)
        return {
            "manifest": self.policy.manifest,
            "fps": self.policy.fps,
            "action_horizon": self.policy.action_horizon,
            "actor_horizon": self.policy.actor_horizon,
        }

    def begin_episode(self) -> dict:
        return {"version": self.version, "expert_versions": self.expert_versions.copy()}

    def infer_with_reference(self, observation, task, *, actor_enabled, expert_id=None):
        actions, diagnostics = self.policy.infer_with_reference(
            observation,
            task,
            actor_enabled=actor_enabled,
            expert_id=expert_id,
            mode="train",
        )
        return actions, {
            **diagnostics,
            "policy_version": self.expert_versions[expert_id] if actor_enabled else -1,
        }

    def infer_for_replay(
        self, observation, task, *, actor_enabled, directory, frame, expert_id=None
    ):
        actions, diagnostics = self.policy.infer_with_features(
            observation,
            task,
            actor_enabled=actor_enabled,
            expert_id=expert_id,
            mode="train",
        )
        save_replay_anchor(
            Path(directory),
            frame,
            self.policy.manifest,
            task,
            {
                **diagnostics.pop("replay_features"),
                "raw_state": observation["observation.state"],
            },
        )
        return actions, {
            **diagnostics,
            "policy_version": self.expert_versions[expert_id] if actor_enabled else -1,
        }

    def prepare_expert_episode(self, directory: str, result: dict) -> dict:
        self.prepare_episode(directory, result)
        return expert_trajectories(Path(directory), result)

    def install_pair(self, checkpoint: str, binding: str) -> dict:
        """Validate both saved models before installing either, between episodes."""
        root = Path(checkpoint)
        saved = json.loads((root / "state.json").read_text())
        if saved["binding"] != binding or set(saved["versions"]) != set(
            self.policy.expert_ids
        ):
            raise ValueError("Expert checkpoint belongs to another run.")
        states = {}
        for expert, model in self.policy.expert_actors.items():
            path = root / expert / "model_state_dict/full_weights.pt"
            if file_sha256(path) != saved["snapshot_sha256"][expert]:
                raise ValueError("Expert snapshot checksum mismatch.")
            state = torch.load(path, map_location="cpu", weights_only=True)
            current = model.state_dict()
            if state.keys() != current.keys() or any(
                state[key].shape != current[key].shape
                or state[key].dtype != current[key].dtype
                or not torch.isfinite(state[key]).all()
                for key in current
            ):
                raise ValueError("Expert snapshot tensor contract mismatch.")
            states[expert] = state
        for expert, state in states.items():
            self.policy.expert_actors[expert].load_state_dict(state, strict=True)
        self.version = int(saved["generation"])
        self.expert_versions = dict(saved["versions"])
        return self.begin_episode()
