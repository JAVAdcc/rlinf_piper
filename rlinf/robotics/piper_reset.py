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

"""Evo-RL-style timed reset: prepare, open, interpolate joints, restore gripper.

Reference: Evo-RL/src/lerobot/scripts/lerobot_record.py::_move_robot_to_zero_pose.
The connection and low-level driver stay owned by PiperSession. Completion
means the trajectory was sent; measured final error is reported separately.
"""

import json
import time
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from rlinf.robotics.piper_runtime import PiperSession


@dataclass(frozen=True)
class PiperResetTiming:
    """Durations matching Evo-RL's default reset sequence."""

    joint_duration_s: float = 3.0
    gripper_duration_s: float = 0.5

    def __post_init__(self) -> None:
        for value in (self.joint_duration_s, self.gripper_duration_s):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Reset durations must be finite and positive.")


def reset_piper(
    session: PiperSession,
    target: np.ndarray,
    *,
    log: TextIO,
    timing: PiperResetTiming = PiperResetTiming(),
) -> dict:
    """Send a timed three-stage reset over the current connection.

    Unlike the old feedback-paced reset, a measured start outside nominal
    travel is not rejected. The session clips every sent target to legal
    travel and checks feedback freshness and controller status on every tick.
    No policy step-delta or arbitrary arrival threshold gates the trajectory.
    """
    target = np.asarray(target, dtype=float)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("Reset requires seven finite degree/mm targets.")
    fps = session.config.fps
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Reset requires a positive finite control frequency.")
    low, high = session.arm.joint_limits
    width = session.config.gripper_max_width_m * 1000
    lower, upper = np.r_[np.rad2deg(low), 0], np.r_[np.rad2deg(high), width]
    tolerance = np.r_[[0.5] * 6, 1.0]
    if np.any(target < lower - tolerance) or np.any(target > upper + tolerance):
        raise ValueError("Requested reset destination is outside configured travel.")
    target = np.clip(target, lower, upper)
    print("Reset: reasserting joint control and confirming motor enable.", flush=True)
    session.prepare_for_reset()
    start = np.asarray(session.read_state(), dtype=float)
    if start.shape != (7,) or not np.isfinite(start).all():
        raise ValueError("Invalid feedback before reset trajectory.")
    opened = start.copy()
    opened[6] = width
    returned = target.copy()
    returned[6] = width
    period = 1 / fps
    started = time.monotonic()
    log.write(
        json.dumps(
            {
                "event": "reset_start",
                "strategy": "evo_timed_v1",
                "state": start.tolist(),
                "target": target.tolist(),
                "joint_duration_s": timing.joint_duration_s,
                "gripper_duration_s": timing.gripper_duration_s,
            }
        )
        + "\n"
    )
    log.flush()
    for phase, origin, destination, duration in (
        ("open_gripper", start, opened, timing.gripper_duration_s),
        ("return_joints", opened, returned, timing.joint_duration_s),
        ("restore_gripper", returned, target, timing.gripper_duration_s),
    ):
        count = max(round(duration / period), 1)
        print(
            f"Reset {phase}: sending {count} targets over approximately {count * period:.2f}s.",
            flush=True,
        )
        for step in range(1, count + 1):
            tick = time.monotonic()
            command = origin + (destination - origin) * (step / count)
            receipt = session.execute_reset(command)
            log.write(
                json.dumps(
                    {
                        "event": "reset_action",
                        "phase": phase,
                        "step": step,
                        "steps": count,
                        "time": time.time(),
                        **{key: value.tolist() for key, value in receipt.items()},
                    }
                )
                + "\n"
            )
            log.flush()
            # No catch-up bursts after delayed I/O, matching Evo-RL's sender.
            time.sleep(max(0, period - (time.monotonic() - tick)))
    final = np.asarray(session.read_state(), dtype=float)
    if final.shape != (7,) or not np.isfinite(final).all():
        raise ValueError("Invalid feedback at reset completion.")
    result = {
        "completion": "trajectory_sent",
        "strategy": "evo_timed_v1",
        "target": target.tolist(),
        "final_state": final.tolist(),
        "final_error": (final - target).tolist(),
        "duration_s": time.monotonic() - started,
    }
    log.write(json.dumps({"event": "reset_complete", **result}) + "\n")
    log.flush()
    print(
        f"Reset trajectory sent. Final max joint error={np.max(np.abs(final[:6] - target[:6])):.3f} deg, "
        f"gripper error={abs(final[6] - target[6]):.3f} mm.",
        flush=True,
    )
    return result
