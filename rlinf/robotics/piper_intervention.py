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

"""Explicit Piper control ownership shared by local rollout and online envs."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rlinf.robotics.parts.teleop.piper_leader import PiperLeader

ACTION_SOURCES = {"reference": 0, "actor": 1, "human": 2, "hold": 3}


@dataclass(frozen=True)
class PiperInterventionConfig:
    """Keyboard and alignment settings independent of the leader SDK."""

    key: str = "i"
    alignment_duration_s: float = 3.0
    max_joint_gap_deg: float = 10.0
    max_gripper_gap_mm: float = 15.0
    toggle_cooldown_s: float = 0.3

    def __post_init__(self) -> None:
        if self.key not in ("i", "o"):
            raise ValueError(
                "Use i or o for intervention, independently of b and scoring keys."
            )
        for name, value in vars(self).items():
            if name != "key" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite.")


class PiperIntervention:
    """Arbitrate targets without owning cameras, inference, or the follower.

    Each takeover/release invalidates in-flight predictions via ``generation``.
    Release freezes the last human target until a fresh prediction is accepted.
    The release interval still belongs to the human correction, not the actor.
    """

    def __init__(
        self, leader: PiperLeader, config: PiperInterventionConfig | None = None
    ):
        self.leader = leader
        self.config = config or PiperInterventionConfig()
        self.mode = "policy"
        self.generation = 0
        self.takeovers = 0
        self.human_steps = 0
        self._hold = None
        self._last_toggle = -float("inf")

    def connect(self) -> None:
        """Connect the leader in position hold without aligning to the follower."""
        self.leader.connect()

    def align(
        self,
        follower: np.ndarray,
        fps: float,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> None:
        """Move only the leader to the follower pose before starting a trial."""
        self.stop()
        follower = self._target(follower)
        origin = self.leader.read_target()
        count = max(1, round(self.config.alignment_duration_s * fps))
        for step in range(1, count + 1):
            if abort is not None and abort():
                raise KeyboardInterrupt
            started = time.monotonic()
            self.leader.send_feedback(origin + (follower - origin) * (step / count))
            time.sleep(max(0, 1 / fps - (time.monotonic() - started)))
        self._check_alignment(follower)
        self._hold = follower.copy()
        self.takeovers = self.human_steps = 0
        self._last_toggle = -float("inf")

    @staticmethod
    def _target(value: np.ndarray) -> np.ndarray:
        target = np.asarray(value, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(
                "Piper control targets must be seven finite degree/mm values."
            )
        return target

    def _check_alignment(self, follower: np.ndarray) -> None:
        gap = np.abs(self.leader.read_target() - self._target(follower))
        if (
            gap[:6].max() > self.config.max_joint_gap_deg
            or gap[6] > self.config.max_gripper_gap_mm
        ):
            raise ValueError(
                f"Piper leader/follower are not aligned: joint gap {gap[:6].max():.2f} deg, "
                f"gripper gap {gap[6]:.2f} mm."
            )

    def toggle(self, follower: np.ndarray) -> bool:
        """Request takeover or release; return False for repeated held-key events."""
        now = time.monotonic()
        if now - self._last_toggle < self.config.toggle_cooldown_s:
            return False
        if self.mode == "policy":
            self._check_alignment(follower)
            self.leader.set_manual_control(True)
            self.mode = "human"
            self.takeovers += 1
        elif self.mode == "human":
            self._hold = self.leader.read_target().copy()
            self.leader.set_manual_control(False)
            self.mode = "release_pending"
        else:
            self.leader.set_manual_control(True)
            self.mode = "human"
        self.generation += 1
        self._last_toggle = now
        return True

    def accept_prediction(self, generation: int) -> bool:
        """Give control back only to predictions requested after the last event."""
        if generation != self.generation or self.mode == "human":
            return False
        self.mode = "policy"
        return True

    def choose(
        self, policy_target: np.ndarray | None, follower: np.ndarray, *, source: str
    ) -> tuple[np.ndarray, str]:
        """Choose a degree/mm command; human ownership always overrides policy."""
        if source not in ("reference", "actor"):
            raise ValueError("Policy source must be reference or actor.")
        if self.mode == "human":
            return self.leader.read_target(), "human"
        if self.mode == "release_pending":
            return self._hold.copy(), "human"
        if policy_target is not None:
            return self._target(policy_target), source
        return self._target(self._hold if self._hold is not None else follower), "hold"

    def executed(self, command: np.ndarray, source: str) -> None:
        """Mirror the confirmed command and account for the executed source."""
        if source not in ACTION_SOURCES:
            raise ValueError(f"Unknown Piper action source: {source}.")
        command = self._target(command)
        if not self.leader.manual:
            self.leader.send_feedback(command)
        self._hold = command.copy()
        self.human_steps += int(source == "human")

    def stop(self) -> None:
        """Freeze the leader and invalidate predictions before scoring or reset."""
        if self.leader.is_connected:
            self.leader.set_manual_control(False)
        self.mode = "policy"
        self.generation += 1

    def close(self) -> None:
        """Release the leader, whose driver restores position hold."""
        self.leader.disconnect()
