#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""MIT gravity compensation adapted from Evo-RL (ee5c6b583071)."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _rpy_deg_to_rot(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


class PiperGravityCompensationLoop:
    """Background MIT torque loop for gravity compensation on Piper leader arm."""

    def __init__(
        self,
        *,
        arm: Any,
        urdf_path: str,
        control_hz: float,
        tx_ratio: tuple[float, float, float, float, float, float],
        torque_limit: float,
        mit_kp: float,
        mit_kd: float,
        base_rpy_deg: tuple[float, float, float],
        payload_mass_kg: float,
        payload_com_m: tuple[float, float, float],
        mode_refresh_interval_s: float,
        move_speed_ratio: int,
        max_feedback_age_s: float = 0.5,
    ) -> None:
        import pinocchio as pin

        self._pin = pin
        self._arm = arm
        self._control_hz = control_hz
        self._tx_ratio = np.asarray(tx_ratio, dtype=np.float64)
        self._torque_limit = float(torque_limit)
        self._mit_kp = float(mit_kp)
        self._mit_kd = float(mit_kd)
        self._mode_refresh_interval_s = max(0.0, float(mode_refresh_interval_s))
        self._move_speed_ratio = int(move_speed_ratio)
        self._max_feedback_age_s = max_feedback_age_s

        urdf = Path(urdf_path).expanduser().resolve()
        if not urdf.is_file():
            raise FileNotFoundError(f"gravity compensation URDF does not exist: {urdf}")

        # Gravity needs inertial joints only; avoid loading visual meshes.
        self._model = pin.buildModelFromUrdf(str(urdf))
        self._data = self._model.createData()
        self._nq = 6

        if payload_mass_kg > 0:
            payload_com = np.asarray(payload_com_m, dtype=np.float64)
            # Rotational inertia does not affect g(q), but Pinocchio requires a valid spatial inertia.
            payload_inertia = np.diag([1e-3, 1e-3, 5e-4])
            self._model.inertias[self._nq] += pin.Inertia(
                float(payload_mass_kg), payload_com, payload_inertia
            )
            self._data = self._model.createData()

        rot_world_base = _rpy_deg_to_rot(*base_rpy_deg)
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        gravity_base = rot_world_base.T @ gravity_world
        self._model.gravity.linear = gravity_base

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_mode_refresh_t = 0.0
        self._error: Exception | None = None
        self._last_q_rad: np.ndarray | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.check_health()
            return
        self._stop_event.clear()
        self._error = None

        # Send a valid compensation frame before returning so intervention never starts unsupported.
        self._send_mit_mode()
        self._run_once()
        self._thread = threading.Thread(
            target=self._run,
            name="piper-gravity-comp-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is None:
            return
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError(
                "PiPER gravity compensation thread did not stop within timeout."
            )
        self._thread = None

    def check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "PiPER gravity compensation loop failed."
            ) from self._error
        if (
            self._thread is not None
            and not self._thread.is_alive()
            and not self._stop_event.is_set()
        ):
            raise RuntimeError("PiPER gravity compensation loop stopped unexpectedly.")

    def _send_mit_mode(self) -> None:
        # Use MOVE M + MIT to ensure torque pass-through mode is active.
        self._arm.MotionCtrl_2(0x01, 0x04, self._move_speed_ratio, 0xAD)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_mit_mode_if_needed(self) -> None:
        if self._mode_refresh_interval_s <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= self._mode_refresh_interval_s:
            self._send_mit_mode()

    def _read_q_v(self) -> tuple[np.ndarray, np.ndarray]:
        joint_msg = self._arm.GetArmJointMsgs()
        age = time.time() - joint_msg.time_stamp
        if not 0 <= age <= self._max_feedback_age_s:
            raise TimeoutError("Piper leader joint feedback is stale.")
        joint_state = joint_msg.joint_state
        q_deg = np.array(
            [
                float(joint_state.joint_1) * 1e-3,
                float(joint_state.joint_2) * 1e-3,
                float(joint_state.joint_3) * 1e-3,
                float(joint_state.joint_4) * 1e-3,
                float(joint_state.joint_5) * 1e-3,
                float(joint_state.joint_6) * 1e-3,
            ],
            dtype=np.float64,
        )
        q_rad = np.deg2rad(q_deg)
        self._last_q_rad = q_rad.copy()

        hs = self._arm.GetArmHighSpdInfoMsgs()
        v_rad = np.array(
            [
                float(hs.motor_1.motor_speed) * 1e-3,
                float(hs.motor_2.motor_speed) * 1e-3,
                float(hs.motor_3.motor_speed) * 1e-3,
                float(hs.motor_4.motor_speed) * 1e-3,
                float(hs.motor_5.motor_speed) * 1e-3,
                float(hs.motor_6.motor_speed) * 1e-3,
            ],
            dtype=np.float64,
        )
        return q_rad, v_rad

    def _compute_gravity_torque(
        self, q_rad: np.ndarray, v_rad: np.ndarray
    ) -> np.ndarray:
        del v_rad
        q_full = np.zeros(self._model.nq, dtype=np.float64)
        q_full[: self._nq] = q_rad
        tau_full = self._pin.computeGeneralizedGravity(
            self._model,
            self._data,
            q_full,
        )
        return np.asarray(tau_full[: self._nq], dtype=np.float64)

    def _run_once(self) -> None:
        self._refresh_mit_mode_if_needed()
        q_rad, v_rad = self._read_q_v()
        tau = self._compute_gravity_torque(q_rad, v_rad)
        if not np.isfinite(tau).all():
            raise ValueError("Piper gravity torque is not finite.")
        tau = np.clip(self._tx_ratio * tau, -self._torque_limit, self._torque_limit)

        for idx in range(6):
            self._arm.JointMitCtrl(
                idx + 1,
                0.0,
                0.0,
                self._mit_kp,
                self._mit_kd,
                float(tau[idx]),
            )

    def _restore_position_hold_after_failure(self) -> None:
        if self._last_q_rad is None:
            return
        try:
            q_millideg = np.rint(np.rad2deg(self._last_q_rad) * 1e3).astype(np.int64)
            self._arm.MotionCtrl_2(0x01, 0x01, self._move_speed_ratio, 0x00)
            self._arm.JointCtrl(*(int(value) for value in q_millideg))
            self._arm.EnablePiper()
        except Exception:
            logger.exception(
                "Failed to restore PiPER position hold after gravity compensation failure."
            )

    def _run(self) -> None:
        dt = 1.0 / self._control_hz
        try:
            while not self._stop_event.is_set():
                start_t = time.perf_counter()
                self._run_once()

                remain = dt - (time.perf_counter() - start_t)
                if remain > 0:
                    self._stop_event.wait(remain)
        except Exception as error:
            self._error = error
            self._stop_event.set()
            logger.exception(
                "PiPER gravity compensation loop failed; restoring position hold."
            )
            self._restore_position_hold_after_failure()
