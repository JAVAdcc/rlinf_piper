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

"""CPU-only experimental critical-phase probabilities, without control authority."""

from pathlib import Path

import cv2
import numpy as np


class PiperPhaseMonitor:
    """Evaluate a frozen linear baseline on current RGB/state observations.

    Call ``update`` at the calibration cadence (nominally 5 Hz), and ``reset``
    at each episode boundary. The returned alarm is advisory: this class never
    changes the actor/reference route or infers a phase's success or identity.
    """

    def __init__(self, weights: str | Path) -> None:
        with np.load(weights, allow_pickle=False) as data:
            self.feature = str(data["feature"])
            self.cameras = tuple(data["cameras"].tolist())
            self.mean = data["mean"].astype(np.float32)
            self.scale = data["scale"].astype(np.float32)
            self.weight = data["weight"].astype(np.float32)
            self.bias = float(data["bias"])
            if int(data["image_size"]) != 16:
                raise ValueError("Monitor requires the calibrated 16x16 RGB features.")
        dimensions = {"state": 7, "rgb": 2304, "state_rgb": 2311}
        if self.feature not in dimensions or self.cameras != (
            "front",
            "front_2",
            "wrist",
        ):
            raise ValueError("Unsupported monitor features or camera ordering.")
        shape = (dimensions[self.feature],)
        if (
            any(
                x.shape != shape or not np.isfinite(x).all()
                for x in (self.mean, self.scale, self.weight)
            )
            or not np.isfinite(self.bias)
            or np.any(self.scale <= 0)
        ):
            raise ValueError("Invalid monitor weights or normalization.")
        self.reset()

    def reset(self) -> None:
        """Clear the advisory alarm and consecutive-sample count."""
        self.active = False
        self.count = 0

    def predict(self, observation: dict[str, np.ndarray]) -> float:
        """Return the estimated current critical probability, using no future frames."""
        parts = []
        if self.feature in ("state", "state_rgb"):
            state = np.asarray(observation["observation.state"], dtype=np.float32)
            if state.shape != (7,) or not np.isfinite(state).all():
                raise ValueError("Monitor needs a finite seven-dimensional raw state.")
            parts.append(state)
        if self.feature in ("rgb", "state_rgb"):
            for name in self.cameras:
                image = np.asarray(observation[f"observation.images.{name}"])
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError("Monitor expects uint8 HWC RGB camera frames.")
                parts.append(
                    cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
                    .reshape(-1)
                    .astype(np.float32)
                    / 255.0
                )
        inputs = (np.concatenate(parts) - self.mean) / self.scale
        logit = np.clip(inputs @ self.weight + self.bias, -60, 60)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def update(self, observation: dict[str, np.ndarray]) -> dict[str, float | bool]:
        """Update the fixed .8/.3 two-sample hysteresis without switching policies."""
        probability = self.predict(observation)
        condition = probability <= 0.3 if self.active else probability >= 0.8
        self.count = self.count + 1 if condition else 0
        changed = self.count == 2
        if changed:
            self.active = not self.active
            self.count = 0
        return {
            "critical_probability": probability,
            "advisory_active": self.active,
            "changed": changed,
        }
