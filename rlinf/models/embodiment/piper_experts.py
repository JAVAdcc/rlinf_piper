# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Explicit Piper stage experts sharing one frozen feature model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.piper_inference import (
    PiperRLTPolicy,
    load_piper_inference_config,
)

EXPERT_PHASES = {"phase1": 0, "phase2": 1}


def load_expert_paths(path: str | Path) -> dict[str, str]:
    """Resolve both actor directories relative to an explicit YAML/JSON registry."""
    path = Path(path).expanduser().resolve()
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict) or set(values) != set(EXPERT_PHASES):
        raise ValueError("Expert registry must contain exactly phase1 and phase2.")
    result = {}
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing actor directory for {key}.")
        root = Path(value).expanduser()
        if not root.is_absolute():
            root = path.parent / root
        if not (root / "piper_rlt_actor.json").is_file():
            root = root / "actor"
        if not (root / "piper_rlt_actor.json").is_file():
            raise ValueError(f"Missing expert metadata: {root}.")
        result[key] = str(root.resolve())
    return result


class PiperExpertPolicy(PiperRLTPolicy):
    """Choose a separately trained head explicitly on every critical request."""

    expert_ids = tuple(EXPERT_PHASES)

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        experts: dict[str, str],
        device: str = "cuda:0",
        allow_offline_actor: bool = False,
    ) -> None:
        from rlinf.models.embodiment.mlp_policy import get_model

        if set(experts) != set(EXPERT_PHASES):
            raise ValueError(
                "Exactly phase1 and phase2 actor directories are required."
            )
        feature_cfg, feature_manifest = load_piper_inference_config(checkpoint)
        if feature_cfg.precision != "fp32" or feature_cfg.openpi.action_chunk != 50:
            raise ValueError("Paired experts require an FP32 H50 Stage 1.")
        # Validate both heads before loading the large frozen model.
        heads = {}
        for expert, phase in EXPERT_PHASES.items():
            root = Path(experts[expert]).resolve()
            metadata = json.loads((root / "piper_rlt_actor.json").read_text())
            if (
                type(metadata.get("phase_filter")) is not int
                or metadata["phase_filter"] != phase
                or metadata.get("reward_scope") != "critical_segment"
                or metadata.get("critic_action_projection") != "driver_limits"
            ):
                raise ValueError(
                    f"Expert {expert} has an incompatible phase/reward contract."
                )
            if metadata.get("format") == "rlinf-piper-rlt-offline-local-actor-v1":
                if not allow_offline_actor:
                    raise ValueError(
                        "Offline experts require explicit experimental opt-in."
                    )
            elif metadata.get("format") != "rlinf-piper-rlt-online-segment-actor-v1":
                raise ValueError("Unsupported expert actor format.")
            if metadata["stage1_checkpoint_id"] != feature_manifest["checkpoint_id"]:
                raise ValueError(f"Expert {expert} belongs to another Stage 1.")
            cfg = metadata["model"]
            if (
                cfg["model_type"] != "rlt_mlp_policy"
                or cfg["precision"] != "fp32"
                or cfg["action_dim"] != 7
                or cfg["proprio_dim"] != 7
                or cfg["num_action_chunks"] != 50
                or cfg["ref_num_action_chunks"] != 50
                or not cfg["add_q_head"]
            ):
                raise ValueError(
                    "The paired deployment requires FP32 C50 actor/twin-Q heads."
                )
            heads[expert] = (root, metadata)
        super().__init__(checkpoint, device=device)
        self.manual_critical = True
        self.actor_horizon = 50
        self.expert_actors = {}
        contracts = {}
        for expert, (root, metadata) in heads.items():
            if metadata["stage1_checkpoint_id"] != self.manifest["checkpoint_id"]:
                raise ValueError(f"Expert {expert} belongs to another Stage 1.")
            actor = get_model(
                OmegaConf.create(metadata["model"]), torch_dtype=torch.float32
            )
            path = root / "model_state_dict/full_weights.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            if any(not torch.isfinite(t).all() for t in state.values()):
                raise ValueError(f"Non-finite expert weights: {expert}.")
            actor.load_state_dict(state, strict=True)
            self.expert_actors[expert] = actor.to(device).eval().requires_grad_(False)
            contracts[expert] = {
                "metadata": metadata,
                "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        if self.action_horizon != 50:
            raise ValueError("Paired experts require an H50 reference.")
        self.manifest = dict(
            self.manifest,
            stage2={
                "routing": "manual_experts",
                "experts": contracts,
                "allow_offline_actor": allow_offline_actor,
                "actor_horizon": 50,
                "reference_horizon": 50,
            },
        )

    def _selected_actor(
        self, actor_enabled: bool | None, expert_id: str | None
    ) -> torch.nn.Module | None:
        if actor_enabled is None:
            raise ValueError(
                "Expert inference requires an explicit actor_enabled snapshot."
            )
        if actor_enabled:
            if expert_id not in self.expert_actors:
                raise ValueError(
                    "Select phase1 or phase2 before entering a critical segment."
                )
            return self.expert_actors[expert_id]
        if expert_id is not None:
            raise ValueError("Reference requests must not name an active expert.")
        return None

    def infer(
        self, observation, task, *, actor_enabled=None, expert_id=None, mode="eval"
    ) -> np.ndarray:
        """Execute a deterministic mean by default, with an explicit expert identity."""
        return self.infer_with_reference(
            observation,
            task,
            actor_enabled=actor_enabled,
            expert_id=expert_id,
            mode=mode,
        )[0]

    def infer_with_reference(
        self, observation, task, *, actor_enabled=None, expert_id=None, mode="eval"
    ) -> tuple[np.ndarray, dict]:
        """Return actions and the reference from the same frozen-model pass."""
        actor = self._selected_actor(actor_enabled, expert_id)
        actions, diagnostics = self._infer(
            observation, task, actor_enabled, mode, True, actor_model=actor
        )
        return actions, {
            **diagnostics,
            "expert_id": expert_id,
            "policy_version": 0 if actor_enabled else -1,
        }

    def infer_with_features(
        self, observation, task, *, actor_enabled, expert_id=None, mode="train"
    ) -> tuple[np.ndarray, dict]:
        """Include frozen features for replay, preserving the request's expert."""
        actor = self._selected_actor(actor_enabled, expert_id)
        actions, diagnostics = self._infer(
            observation, task, actor_enabled, mode, True, True, actor_model=actor
        )
        return actions, {
            **diagnostics,
            "expert_id": expert_id,
            "policy_version": 0 if actor_enabled else -1,
        }
