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

"""Evo-RL single-Piper observations and absolute joint targets for OpenPI.

Dataset joints are degrees and the gripper is millimetres. Models see radians
and metres; output transforms restore dataset units. Camera aliases follow
Evo-RL's USB task mapping, including its second external view in the left slot.
"""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

PIPER_SCALE = np.array([np.pi / 180] * 6 + [0.001], dtype=np.float32)
PIPER_CAMERAS = {
    "base_0_rgb": ("observation.images.base_0_rgb", "observation.images.front"),
    "left_wrist_0_rgb": (
        "observation.images.left_wrist_0_rgb",
        "observation.images.front_2",
    ),
    "right_wrist_0_rgb": (
        "observation.images.right_wrist_0_rgb",
        "observation.images.wrist",
    ),
}


def piper_to_model_units(values: Any) -> np.ndarray:
    """Convert seven absolute joint/gripper values to radians and metres."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != 7:
        raise ValueError(f"Single Piper requires 7 values, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Piper state/actions contain non-finite values.")
    return values * PIPER_SCALE


@dataclasses.dataclass(frozen=True)
class PiperInputs:
    """Map LeRobot Piper frames to the three OpenPI image slots."""

    def __call__(self, data: dict) -> dict:
        images = {}
        for slot, aliases in PIPER_CAMERAS.items():
            present = [key for key in aliases if key in data]
            if len(present) != 1:
                raise ValueError(
                    f"Piper camera {slot} needs exactly one of {aliases}; got {present}."
                )
            image = np.asarray(data[present[0]])
            if image.ndim != 3:
                raise ValueError(f"Expected a single RGB frame, got {image.shape}.")
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.shape[-1] != 3:
                raise ValueError(f"Expected RGB channels, got {image.shape}.")
            if np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError("Floating Piper images must be finite in [0, 1].")
                image = np.rint(image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 or floating RGB, got {image.dtype}.")
            images[slot] = image

        state = piper_to_model_units(data["observation.state"])
        if state.shape != (7,):
            raise ValueError(f"Expected one Piper state, got {state.shape}.")
        result = {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
        }
        if "action" in data:
            result["actions"] = piper_to_model_units(data["action"])
        prompt = data.get("prompt")
        if prompt is None:
            prompt = data.get("task")
        if prompt is not None:
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs:
    """Restore absolute targets in degrees/mm for an Evo-RL executor."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim < 1 or actions.shape[-1] < 7:
            raise ValueError(f"Expected at least 7 action values, got {actions.shape}.")
        return {"actions": actions[..., :7] / PIPER_SCALE}


class PiperActionCodec:
    """Convert RLT/replay normalized actions to absolute Piper dataset units."""

    def __init__(self, norm_stats_path: str | Path):
        import openpi.shared.normalize as normalize
        import openpi.transforms as transforms

        path = Path(norm_stats_path)
        stats = normalize.load(path.parent if path.is_file() else path)
        actions = {"actions": stats["actions"]}
        self._normalize = transforms.Normalize(actions, use_quantiles=True)
        self._unnormalize = transforms.Unnormalize(actions, use_quantiles=True)

    def decode(self, actions: np.ndarray) -> np.ndarray:
        """Decode finite normalized seven-dimensional actions without clipping."""
        values = np.asarray(actions, dtype=np.float32)
        if values.shape[-1:] != (7,) or not np.isfinite(values).all():
            raise ValueError(
                "Expected finite normalized Piper actions with last dimension 7."
            )
        return PiperOutputs()(self._unnormalize({"actions": values}))["actions"]

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """Encode absolute degree/mm targets, including a measured hold pose."""
        return self._normalize({"actions": piper_to_model_units(actions)})["actions"]
