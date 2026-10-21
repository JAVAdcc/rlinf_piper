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

"""Portable Piper checkpoints for RLinf RLT and optional RECAP/CFG inference."""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.openpi.policies.piper_policy import (
    PIPER_CAMERAS,
    PiperInputs,
)


def piper_batch_inputs(observations: dict) -> dict:
    """Keep batched LeRobot fields and add the CFG conditioning prompts."""
    prompts = list(observations["task_descriptions"])
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Each Piper observation needs a nonempty task description.")
    keys = {"observation.state"}
    for aliases in PIPER_CAMERAS.values():
        present = [key for key in aliases if key in observations]
        if len(present) != 1:
            raise ValueError(f"Expected exactly one camera from {aliases}.")
        keys.add(present[0])
    result = {key: observations[key] for key in keys}
    if any(value.shape[0] != len(prompts) for value in result.values()):
        raise ValueError(
            "Piper observation batch sizes do not match task descriptions."
        )
    result.update(
        prompt=prompts,
        positive_guidance_prompt=[f"{p}\nAdvantage: positive" for p in prompts],
        negative_guidance_prompt=[f"{p}\nAdvantage: negative" for p in prompts],
    )
    return result


def save_piper_inference_assets(
    checkpoint: str | Path, model_config: DictConfig, *, fps: float
) -> Path:
    """Save portable config and normalization beside a full training checkpoint.

    Call on rank zero after saving weights. Paths inside the saved manifest are
    relative to its directory, so copying the complete actor directory suffices.
    """
    if model_config.openpi.config_name not in ("pi05_piper", "pi05_piper_h50_state"):
        raise ValueError("Piper inference assets require a supported Piper config.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be positive and finite.")
    checkpoint = Path(checkpoint)
    if not (checkpoint / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(
            "Enable actor.fsdp_config.save_full_model_weights for Piper deployment."
        )
    source = Path(model_config.openpi_data.norm_stats_path)
    contents = source.read_bytes()
    stats = json.loads(contents)["norm_stats"]
    for key in ("state", "actions"):
        for name in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[key][name])
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid seven-dimensional {key}/{name} normalization."
                )
    assets = checkpoint / "assets/piper"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "norm_stats.json").write_bytes(contents)
    model = OmegaConf.to_container(model_config, resolve=True)
    model["model_path"] = "."
    model["openpi_data"]["norm_stats_path"] = "assets/piper/norm_stats.json"
    manifest = {
        "format": "rlinf-piper-policy-v1",
        "checkpoint_id": uuid.uuid4().hex,
        "model": model,
        "fps": float(fps),
        "action_units": "degrees_mm",
        "norm_stats_sha256": hashlib.sha256(contents).hexdigest(),
    }
    path = checkpoint / "piper_inference.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return path


def load_piper_inference_config(checkpoint: str | Path) -> tuple[DictConfig, dict]:
    """Validate and resolve a downloaded actor directory or global-step directory."""
    root = Path(checkpoint).expanduser().resolve()
    if not (root / "piper_inference.json").is_file():
        root = root / "actor"
    manifest_path = root / "piper_inference.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != "rlinf-piper-policy-v1"
        or manifest.get("action_units") != "degrees_mm"
    ):
        raise ValueError("Unsupported Piper inference manifest.")
    cfg = OmegaConf.create(manifest["model"])
    if cfg.model_type not in ("cfg_model", "openpi") or cfg.openpi.config_name not in (
        "pi05_piper",
        "pi05_piper_h50_state",
    ):
        raise ValueError("Expected a trained Piper model.")
    if not np.isfinite(manifest["fps"]) or manifest["fps"] <= 0:
        raise ValueError("Invalid dataset fps in inference manifest.")
    if not (root / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing full model weights in {root}.")
    stats = (root / cfg.openpi_data.norm_stats_path).resolve()
    if not stats.is_relative_to(root):
        raise ValueError("Normalization must be inside the downloaded actor directory.")
    if hashlib.sha256(stats.read_bytes()).hexdigest() != manifest["norm_stats_sha256"]:
        raise ValueError("Normalization does not match the saved policy checkpoint.")
    cfg.model_path = str(root)
    cfg.openpi_data.norm_stats_path = str(stats)
    return cfg, manifest


class PiperCFGPolicy:
    """Run single-observation CFG inference, returning absolute degrees/mm targets."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda:0") -> None:
        from rlinf.models.embodiment.openpi_cfg import get_model

        cfg, self.manifest = load_piper_inference_config(checkpoint)
        if cfg.model_type != "cfg_model":
            raise ValueError(
                "Select the rlt backend for an RLinf OpenPI/RLT checkpoint."
            )
        self.model = get_model(cfg, strict_weights=True).to(device).eval()
        self.action_horizon = int(self.model.config.action_horizon)
        self.fps = float(self.manifest["fps"])

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any], task: str) -> np.ndarray:
        """Predict one complete chunk from raw LeRobot RGB and degrees/mm state."""
        # Validate raw units, shape, and RGB before allocating model inputs.
        PiperInputs()(observation)
        batch = {
            key: np.asarray(value)[None]
            for key, value in observation.items()
            if key == "observation.state" or key.startswith("observation.images.")
        }
        batch["task_descriptions"] = [task]
        actions, _ = self.model.predict_action_batch(batch, mode="eval")
        actions = np.asarray(actions, dtype=np.float32)
        if (
            actions.shape != (1, self.action_horizon, 7)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid Piper action chunk: {actions.shape}.")
        return actions[0]


def piper_env_observation(observation: dict, task: str) -> dict:
    """Pack a raw Piper sample into RLinf's shared environment batch format."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A nonempty task description is required.")
    PiperInputs()(observation)
    images = []
    for aliases in PIPER_CAMERAS.values():
        key = next(key for key in aliases if key in observation)
        image = np.asarray(observation[key])
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        images.append(image)
    return {
        "states": np.asarray(observation["observation.state"])[None],
        "main_images": images[0][None],
        "extra_view_images": np.stack(images[1:])[None],
        "task_descriptions": [task],
    }


def save_piper_rlt_actor_assets(
    checkpoint: str | Path, actor_config: DictConfig, stage1: str | Path
) -> None:
    """Bind a Stage 2 actor's architecture to its frozen Stage 1 checkpoint."""
    cfg, manifest = load_piper_inference_config(stage1)
    if cfg.model_type != "openpi" or not cfg.openpi.use_rlt:
        raise ValueError("RLT Stage 2 requires an RLinf RLT Stage 1 checkpoint.")
    root = Path(checkpoint)
    metadata = {
        "format": "rlinf-piper-rlt-actor-v1",
        "stage1_checkpoint_id": manifest["checkpoint_id"],
        "model": OmegaConf.to_container(actor_config, resolve=True),
    }
    (root / "piper_rlt_actor.json").write_text(json.dumps(metadata, indent=2) + "\n")


class PiperRLTPolicy:
    """Use RLinf's native frozen RLT features and optional learned Stage 2 actor.

    Without an actor checkpoint, execute the Stage 1 VLA reference. With one,
    use the same RLT action-routing function as the online rollout worker.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        actor: str | Path | None = None,
        device: str = "cuda:0",
        manual_critical: bool = False,
        allow_offline_actor: bool = False,
    ) -> None:
        from rlinf.models.embodiment.openpi import get_model
        from rlinf.models.embodiment.openpi.policies.piper_policy import (
            PiperActionCodec,
        )

        cfg, self.manifest = load_piper_inference_config(checkpoint)
        if cfg.model_type != "openpi" or not cfg.openpi.get("use_rlt", False):
            raise ValueError(
                "The rlt backend requires RLinf Stage 1 trained with use_rlt=True."
            )
        self.manual_critical = manual_critical
        if (manual_critical or allow_offline_actor) and actor is None:
            raise ValueError("Manual critical routing requires an actor checkpoint.")
        self.actor = None
        if actor is not None:
            from rlinf.models.embodiment.mlp_policy import get_model as get_actor

            root = Path(actor).expanduser().resolve()
            if not (root / "piper_rlt_actor.json").is_file():
                root = root / "actor"
            metadata = json.loads((root / "piper_rlt_actor.json").read_text())
            local = metadata.get("format") == "rlinf-piper-rlt-offline-local-actor-v1"
            online_segment = (
                metadata.get("format") == "rlinf-piper-rlt-online-segment-actor-v1"
            )
            if online_segment and not manual_critical:
                raise ValueError(
                    "Segment-trained online actors require manual critical routing."
                )
            if local and not (manual_critical and allow_offline_actor):
                raise ValueError(
                    "Offline local actors require manual critical routing and explicit experimental opt-in."
                )
            if (local or online_segment) and (
                "phase_filter" not in metadata
                or metadata.get("phase_filter") is not None
                or metadata.get("reward_scope") != "critical_segment"
            ):
                raise ValueError(
                    "Manual routing currently supports only the shared local actor; specialist routing is not implemented."
                )
            if (
                metadata.get("format")
                not in (
                    "rlinf-piper-rlt-actor-v1",
                    "rlinf-piper-rlt-offline-local-actor-v1",
                    "rlinf-piper-rlt-online-segment-actor-v1",
                )
                or metadata.get("stage1_checkpoint_id")
                != self.manifest["checkpoint_id"]
            ):
                raise ValueError(
                    "Stage 2 actor was trained against a different Stage 1 checkpoint."
                )
            actor_cfg = OmegaConf.create(metadata["model"])
            if actor_cfg.model_type not in ("rlt_mlp_policy", "rlt_td3_mlp_policy"):
                raise ValueError("Unsupported RLT actor type.")
            if actor_cfg.action_dim != 7 or actor_cfg.proprio_dim != 7:
                raise ValueError(
                    "Piper RLT actor requires seven-dimensional actions and proprio."
                )
        cfg.openpi.task = "eval"
        cfg.openpi.torch_compile = False
        self.feature_model = get_model(cfg).to(device).eval().requires_grad_(False)
        self.codec = PiperActionCodec(cfg.openpi_data.norm_stats_path)
        self.fps = float(self.manifest["fps"])
        self.action_horizon = int(self.feature_model.action_chunk)
        if actor is not None:
            self.actor = get_actor(actor_cfg, torch_dtype=torch.float32)
            state = torch.load(
                root / "model_state_dict/full_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.actor.load_state_dict(state, strict=True)
            self.actor.to(device).eval().requires_grad_(False)
            self.actor_horizon = int(actor_cfg.num_action_chunks)
            if self.actor_horizon > self.feature_model.action_chunk:
                raise ValueError(
                    "Actor execution horizon exceeds the reference horizon."
                )
            if not manual_critical:
                self.action_horizon = self.actor_horizon
            # Bind resume to the actual actor bytes, not only its directory name.
            self.manifest = dict(
                self.manifest,
                stage2={
                    "metadata": metadata,
                    "weights_sha256": hashlib.sha256(
                        (root / "model_state_dict/full_weights.pt").read_bytes()
                    ).hexdigest(),
                    "routing": "manual_critical" if manual_critical else "always",
                    "allow_offline_actor": allow_offline_actor,
                    "actor_horizon": self.actor_horizon,
                    "reference_horizon": int(self.feature_model.action_chunk),
                },
            )

    @torch.inference_mode()
    def infer(
        self,
        observation: dict[str, Any],
        task: str,
        *,
        actor_enabled: bool | None = None,
        mode: str = "eval",
    ) -> np.ndarray:
        """Predict in normalized RLT action space, then decode for the executor."""
        from rlinf.algorithms.rlt import predict_rlt_actions
        from rlinf.algorithms.rlt.route import RealworldRLTRoute

        if mode not in ("train", "eval"):
            raise ValueError("RLT inference mode must be train or eval.")
        env_obs = piper_env_observation(observation, task)
        if getattr(self, "manual_critical", False) and actor_enabled is None:
            raise ValueError(
                "Manual inference requires an explicit actor_enabled snapshot."
            )
        use_actor = self.actor is not None and actor_enabled is not False
        if actor_enabled is True and self.actor is None:
            raise ValueError("No actor checkpoint is loaded.")
        if not use_actor:
            actions = self.feature_model.extract_rlt_obs(env_obs)["ref_chunk"]
        else:
            actions, _ = predict_rlt_actions(
                policy_model=self.actor,
                feature_model=self.feature_model,
                rlt_route=RealworldRLTRoute(),
                env_obs=env_obs,
                final_obs=None,
                mode=mode,
                rlt_switch_flags=torch.ones(1, dtype=torch.bool),
            )
        actions = actions.detach().float().cpu().numpy()
        expected = (
            getattr(self, "actor_horizon", self.action_horizon)
            if use_actor
            else int(getattr(self.feature_model, "action_chunk", self.action_horizon))
        )
        if actions.shape != (1, expected, 7):
            raise ValueError(f"Unexpected RLT action shape: {actions.shape}.")
        return self.codec.decode(actions[0])
