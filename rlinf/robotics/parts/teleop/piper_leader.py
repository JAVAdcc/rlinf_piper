# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piper leader modes adapted from Evo-RL (ee5c6b583071).

The hardware keeps role 0xFC. Explicit control ownership switches position
following to MIT gravity compensation; moving the arm never enables takeover.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rlinf.robotics.actions import ActionKind
from rlinf.robotics.parts.claims import DeviceClaim

from .base import TeleopAction, TeleopDevice
from .piper_gravity import PiperGravityCompensationLoop


@dataclass(frozen=True)
class PiperLeaderConfig:
    """Evo-RL control parameters, with degree/mm feedback at the SDK boundary."""

    command_speed_ratio: int = 100
    command_high_follow: bool = True
    gripper_effort_default: int = 1000
    gripper_status_code: int = 1
    gripper_width_mm: float = 70.0
    startup_sleep_s: float = 0.1
    enable_timeout_s: float = 3.0
    max_feedback_age_s: float = 0.5
    mode_refresh_interval_s: float = 1.0
    gravity_comp_control_hz: float = 200.0
    gravity_comp_tx_ratio: tuple[float, ...] = (0.0, 1.05, 1.1, 1.0, 1.4, 1.0)
    gravity_comp_torque_limit: float = 8.0
    gravity_comp_mit_kp: float = 0.0
    gravity_comp_mit_kd: float = 0.0
    gravity_comp_base_rpy_deg: tuple[float, ...] = (0.0, 0.0, 0.0)
    gravity_comp_payload_mass_kg: float = 0.5
    gravity_comp_payload_com_m: tuple[float, ...] = (-0.00017, 0.00007, 0.03784)

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite.")
        for name in (
            "enable_timeout_s",
            "max_feedback_age_s",
            "gripper_width_mm",
            "gravity_comp_control_hz",
            "gravity_comp_torque_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "startup_sleep_s",
            "mode_refresh_interval_s",
            "gravity_comp_mit_kp",
            "gravity_comp_mit_kd",
            "gravity_comp_payload_mass_kg",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name, size in (
            ("gravity_comp_tx_ratio", 6),
            ("gravity_comp_base_rpy_deg", 3),
            ("gravity_comp_payload_com_m", 3),
        ):
            if np.asarray(getattr(self, name)).shape != (size,):
                raise ValueError(f"{name} must contain {size} values.")
        if not 1 <= self.command_speed_ratio <= 100:
            raise ValueError("command_speed_ratio must be in [1,100].")
        if not 0 <= self.gripper_effort_default <= 5000:
            raise ValueError("gripper_effort_default must be in [0,5000].")
        if self.gripper_status_code not in (1, 3):
            raise ValueError("Use an enabled gripper status code, 1 or 3.")


@TeleopDevice.register("piper_leader")
class PiperLeader(TeleopDevice):
    """Own a leader CAN channel and switch its control mode explicitly.

    ``read_target`` and ``send_feedback`` use degree/mm targets, matching Piper
    recordings. The generic teleop observation uses radians/opening fraction.
    """

    SDK = "piper_sdk"
    PRODUCES = {"arm": ActionKind.JOINT_POSITION, "end_effector": ActionKind.GRIPPER}
    HOLD_WINDOW = 0.0

    def __init__(
        self,
        channel: str,
        *,
        config: PiperLeaderConfig | Mapping[str, Any] | None = None,
    ):
        self.channel = channel
        self.config = (
            PiperLeaderConfig(**config)
            if isinstance(config, Mapping)
            else config or PiperLeaderConfig()
        )
        self._arm = None
        self._gravity = None
        self._manual = False
        self._last_mode = 0.0
        self._last_feedback: np.ndarray | None = None
        self._claim = DeviceClaim(f"piper-arm:{channel}", type(self).__name__)

    @property
    def manual(self) -> bool:
        """Whether the leader currently runs gravity compensation."""
        return self._manual

    def _open(self) -> Any:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        cfg = self.config
        self._last_feedback = None
        arm = C_PiperInterface_V2(
            can_name=self.channel,
            judge_flag=False,
            can_auto_init=False,
            logger_level=LogLevel.WARNING,
        )
        # Load the model before connecting either controller mode.
        self._gravity = PiperGravityCompensationLoop(
            arm=arm,
            urdf_path=str(Path(__file__).parent / "assets/piper_no_gripper.urdf"),
            control_hz=cfg.gravity_comp_control_hz,
            tx_ratio=cfg.gravity_comp_tx_ratio,
            torque_limit=cfg.gravity_comp_torque_limit,
            mit_kp=cfg.gravity_comp_mit_kp,
            mit_kd=cfg.gravity_comp_mit_kd,
            base_rpy_deg=cfg.gravity_comp_base_rpy_deg,
            payload_mass_kg=cfg.gravity_comp_payload_mass_kg,
            payload_com_m=cfg.gravity_comp_payload_com_m,
            mode_refresh_interval_s=cfg.mode_refresh_interval_s,
            move_speed_ratio=cfg.command_speed_ratio,
            max_feedback_age_s=cfg.max_feedback_age_s,
        )
        self._claim.acquire()
        self._arm = arm
        try:
            # can_auto_init=False keeps CAN unopened until this owner has the claim.
            arm.CreateCanBus(self.channel, bustype="socketcan", judge_flag=False)
            arm.ConnectPort()
            time.sleep(cfg.startup_sleep_s)
            arm.MasterSlaveConfig(0xFC, 0, 0, 0)
            hold = self._wait_feedback()
            if not self._position_hold(hold):
                raise RuntimeError("Piper leader did not enable in position mode.")
            self._manual = False
            return arm
        except BaseException:
            try:
                self._release(arm)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Piper leader cleanup after connection failure."
                )
            raise

    def _mode(self) -> None:
        cfg = self.config
        self._arm.MotionCtrl_2(
            1,
            1,
            cfg.command_speed_ratio,
            0xAD if cfg.command_high_follow else 0,
        )
        self._last_mode = time.monotonic()

    def _gripper(self, position: float, *, enabled: bool) -> None:
        cfg = self.config
        self._arm.GripperCtrl(
            round(position * 1000),
            cfg.gripper_effort_default if enabled else 0,
            cfg.gripper_status_code if enabled else 0,
            0,
        )

    def _position(self, target: np.ndarray) -> None:
        self._arm.JointCtrl(*(int(round(value * 1000)) for value in target[:6]))
        self._gripper(float(target[6]), enabled=True)

    def _position_hold(self, hold: np.ndarray) -> bool:
        deadline = time.monotonic() + self.config.enable_timeout_s
        while True:
            self._mode()
            self._position(hold)
            self._arm.EnablePiper()
            time.sleep(0.01)
            status = self._arm.GetArmEnableStatus()
            if len(status) == 6 and all(status):
                return True
            if time.monotonic() >= deadline:
                return False

    def _feedback(self) -> np.ndarray:
        if self._arm is None:
            raise RuntimeError("Connect the Piper leader before reading it.")
        joints = self._arm.GetArmJointMsgs()
        gripper = self._arm.GetArmGripperMsgs()
        now = time.time()
        for message in (joints, gripper):
            if not 0 <= now - message.time_stamp <= self.config.max_feedback_age_s:
                raise TimeoutError("Piper leader feedback is missing or stale.")
        values = [
            getattr(joints.joint_state, f"joint_{i}") * 0.001 for i in range(1, 7)
        ]
        values.append(abs(gripper.gripper_state.grippers_angle) * 0.001)
        target = np.asarray(values, dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("Piper leader feedback is not finite.")
        self._last_feedback = target.copy()
        return target

    def _wait_feedback(self) -> np.ndarray:
        deadline = time.monotonic() + self.config.enable_timeout_s
        while True:
            try:
                return self._feedback()
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def set_manual_control(self, enabled: bool) -> None:
        """Change modes while retaining a measured position-hold fallback."""
        if not self.is_connected:
            raise RuntimeError("Connect the Piper leader before switching modes.")
        if enabled == self._manual:
            if enabled:
                self._gravity.check_health()
            return
        hold = self._wait_feedback()
        if enabled:
            if not self._position_hold(hold):
                raise RuntimeError("Piper leader did not enable before takeover.")
            self._gripper(hold[6], enabled=False)
            try:
                self._gravity.start()
            except BaseException:
                self._gravity.stop()
                self._position_hold(hold)
                raise
            self._manual = True
        else:
            self._gravity.stop()
            if not self._position_hold(hold):
                self._gripper(hold[6], enabled=False)
                self._gravity.start()
                raise RuntimeError("Piper leader could not return to position control.")
            self._manual = False

    def read_target(self) -> np.ndarray:
        """Read current leader positions in degrees/mm, rejecting stale feedback."""
        if self._manual:
            self._gravity.check_health()
        return self._feedback()

    def send_feedback(self, target: np.ndarray) -> None:
        """Mirror a degree/mm follower command while in position control."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Piper feedback targets must be seven finite values.")
        if not self.is_connected or self._manual:
            raise RuntimeError(
                "Mirroring requires a connected position-controlled leader."
            )
        interval = self.config.mode_refresh_interval_s
        if interval > 0 and time.monotonic() - self._last_mode >= interval:
            self._mode()
        self._position(target)

    @property
    def observation_features(self) -> dict:
        return {
            "joint_position": {"shape": (6,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> dict:
        """Expose canonical teleop units: radians and opening fraction."""
        target = self.read_target()
        return {
            "joint_position": np.deg2rad(target[:6]).astype(np.float32),
            "grip": np.array([target[6] / self.config.gripper_width_mm], np.float32),
        }

    def action(self, reading: Mapping, context: Mapping) -> TeleopAction:
        """Contribute absolute targets only during explicit manual control."""
        return TeleopAction(
            parts={"arm": reading["joint_position"], "end_effector": reading["grip"]},
            driving=self._manual,
        )

    def _release(self, device: Any) -> None:
        try:
            if self._gravity is not None:
                self._gravity.stop()
            if self._arm is not None:
                try:
                    hold = self._feedback()
                except (TimeoutError, ValueError):
                    # Never turn missing feedback into a zero-position command.
                    logging.getLogger(__name__).warning(
                        "No fresh leader pose available during disconnect."
                    )
                    hold = self._last_feedback
                if hold is not None:
                    if not self._position_hold(hold):
                        raise RuntimeError(
                            "Piper leader could not confirm position hold on exit."
                        )
        finally:
            try:
                device.DisconnectPort()
            finally:
                self._arm = None
                self._gravity = None
                self._manual = False
                self._claim.release()
