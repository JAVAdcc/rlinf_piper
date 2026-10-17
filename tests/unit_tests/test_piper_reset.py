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

"""Evo-RL reset sequence checks using mock feedback and mock CAN SDKs."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.robotics import piper_reset
from rlinf.robotics.piper_reset import PiperResetTiming, reset_piper
from rlinf.robotics.piper_runtime import PiperExecutionConfig, PiperSession


class ResetSession:
    config = PiperExecutionConfig(fps=30)
    arm = SimpleNamespace(
        joint_limits=(
            np.deg2rad([-150, 0, -170, -100, -70, -120]),
            np.deg2rad([150, 180, 0, 100, 70, 120]),
        )
    )
    clip_reset_target = PiperSession.clip_reset_target

    def __init__(self, stalled=False):
        self.state = np.array([0.0, 10.0, -10.0, 0.0, 0.0, 0.0, 20.0])
        self.sent = []
        self.prepared = False
        self.stalled = stalled

    def prepare_for_reset(self):
        assert not self.sent
        self.prepared = True

    def read_state(self):
        return self.state.copy()

    def execute_reset(self, target):
        assert self.prepared
        before = self.state.copy()
        low, high = self.arm.joint_limits
        command = np.clip(
            target, np.r_[np.rad2deg(low), 0], np.r_[np.rad2deg(high), 70]
        )
        self.sent.append(command.copy())
        if not self.stalled:
            self.state = command.copy()
        return {"state": before, "requested": target.copy(), "command": command}


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        piper_reset,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now, time=lambda: clock.now, sleep=sleep
        ),
    )
    return clock


def test_three_phase_sequence_matches_evo_example(clock):
    session = ResetSession()
    session.config = PiperExecutionConfig(fps=10)
    reset_piper(
        session, np.zeros(7), log=io.StringIO(), timing=PiperResetTiming(0.1, 0.1)
    )
    np.testing.assert_allclose(
        session.sent,
        [
            [0, 10, -10, 0, 0, 0, 70],
            [0, 0, 0, 0, 0, 0, 70],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert clock.now == pytest.approx(0.3)


def test_default_timing_and_initial_gripper_pose(clock):
    session = ResetSession()
    target = np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 25.0])
    log = io.StringIO()
    result = reset_piper(session, target, log=log)
    assert len(session.sent) == 120
    assert clock.now == pytest.approx(4)
    np.testing.assert_allclose(session.sent[14], [0, 10, -10, 0, 0, 0, 70])
    np.testing.assert_allclose(session.sent[104], np.r_[target[:6], 70])
    np.testing.assert_allclose(session.sent[-1], target)
    assert result["completion"] == "trajectory_sent"
    assert result["final_error"] == [0.0] * 7
    rows = [json.loads(x) for x in log.getvalue().splitlines()]
    assert rows[0]["strategy"] == "evo_timed_v1"


def test_outside_feedback_is_a_start_not_an_illegal_destination(clock):
    session = ResetSession()
    session.state[4] = -75.871
    result = reset_piper(session, np.zeros(7), log=io.StringIO())
    assert np.all(np.array(session.sent)[:, 4] >= -70)
    assert np.all(np.diff(np.array(session.sent)[15:105, 4]) >= 0)
    assert result["final_state"] == pytest.approx([0.0] * 7)


def test_recorded_initial_pose_is_projected_before_reset_destination_check(clock):
    session = ResetSession()
    measured = np.array([-0.066, -0.511, 0.451, -4.303, 4.484, 0.474, 0.4])
    target = session.clip_reset_target(measured)
    np.testing.assert_allclose(target, [-0.066, 0, 0, -4.303, 4.484, 0.474, 0.4])
    assert measured[1] == -0.511 and measured[2] == 0.451
    result = reset_piper(session, target, log=io.StringIO())
    assert result["completion"] == "trajectory_sent"
    np.testing.assert_allclose(session.sent[-1], target)
    assert np.all(np.array(session.sent)[:, 1] >= 0)
    assert np.all(np.array(session.sent)[:, 2] <= 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_initial_target_projection_rejects_invalid_feedback(value):
    session = ResetSession()
    measured = np.zeros(7)
    measured[1] = value
    with pytest.raises(ValueError, match="finite"):
        session.clip_reset_target(measured)
    assert not session.prepared and not session.sent


def test_stationary_offset_is_reported_without_false_arrival_claim(clock):
    session = ResetSession(stalled=True)
    result = reset_piper(session, np.zeros(7), log=io.StringIO())
    assert result["completion"] == "trajectory_sent"
    assert result["final_error"][1] == 10
    assert clock.now == pytest.approx(4)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_rejected(value):
    with pytest.raises(ValueError):
        PiperResetTiming(value, 0.5)


def test_invalid_destination_never_prepares_or_moves(clock):
    session = ResetSession()
    target = np.zeros(7)
    target[1] = -10
    with pytest.raises(ValueError, match="destination"):
        reset_piper(session, target, log=io.StringIO())
    assert not session.prepared and not session.sent


def test_send_fault_stops_sequence_immediately(clock):
    session = ResetSession()

    def broken(target):
        raise RuntimeError("stale CAN feedback")

    session.execute_reset = broken
    with pytest.raises(RuntimeError, match="stale CAN"):
        reset_piper(session, np.zeros(7), log=io.StringIO())
    assert not session.sent and clock.now == 0


def test_preparation_reasserts_mode_without_reconnecting():
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.piper import PiperArm

    with mocked_sdks():
        arm = PiperArm.declare("can0", firmware="v189", high_follow=True)
        arm.connect()
        try:
            device = arm._robot
            device.mode_feedback = 255
            arm.prepare_for_reset()
            assert arm._robot is device and device.is_connected()
            assert device.motion_mode == "js" and device.enabled
            assert device.mode_feedback == 1
        finally:
            arm.disconnect()


def test_reset_session_uses_limits_even_when_policy_checks_disabled():
    from robot_mocks import mocked_sdks

    from rlinf.robotics import PiperRobot
    from rlinf.robotics.parts.arms.piper import PiperArm

    with mocked_sdks():
        robot = PiperRobot(
            arm=PiperArm.declare("can0", firmware="v189", high_follow=True)
        )
        robot.connect()
        try:
            session = PiperSession.__new__(PiperSession)
            session.robot = robot
            session.arm = robot.child("arm", PiperArm)
            session.config = PiperExecutionConfig(check_action_limits=False)
            session.intervention = None
            session._resources = object()
            session.arm._robot.joints[4] = np.deg2rad(-75.871)
            receipt = session.execute_reset(np.array([0, -5, 5, 0, -80, 0, 80]))
            assert receipt["state"][4] == pytest.approx(-75.871)
            assert receipt["command"][4] == pytest.approx(-70, abs=1e-3)
            assert receipt["command"][6] == 70
            assert session.arm._robot.sent[-1][4] == pytest.approx(
                np.deg2rad(-70), abs=1e-5
            )
        finally:
            robot.disconnect()


@pytest.mark.parametrize("status", [(2, 1, 0), (1, 1, 1)])
def test_reset_preparation_refuses_teaching_or_controller_fault(status):
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.piper import PiperArm

    with mocked_sdks():
        arm = PiperArm.declare("can0", firmware="v189", high_follow=True)
        arm.connect()
        try:
            count = len(arm._robot.sent)
            arm._robot.get_arm_status = lambda: SimpleNamespace(
                msg=SimpleNamespace(
                    ctrl_mode=status[0], mode_feedback=status[1], arm_status=status[2]
                ),
                timestamp=0,
            )
            with pytest.raises(RuntimeError):
                arm.prepare_for_reset()
            assert len(arm._robot.sent) == count
        finally:
            arm.disconnect()


def test_reset_control_rejects_stale_status_before_motion():
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.piper import PiperArm

    with mocked_sdks():
        arm = PiperArm.declare("can0", firmware="v189", high_follow=True)
        arm.connect()
        try:
            count = len(arm._robot.sent)
            arm._max_feedback_age_s = 0.5
            with pytest.raises(RuntimeError, match="Stale Piper arm status"):
                arm.check_reset_control()
            assert len(arm._robot.sent) == count
        finally:
            arm.disconnect()
