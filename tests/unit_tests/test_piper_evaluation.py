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

"""Hardware-free checks for operator-scored evaluation and terminal keys."""

import io
import json
import os
import pty
import termios
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.robotics.piper_announcements import PiperAnnouncer
from rlinf.robotics.piper_evaluation import TerminalKeys, evaluate_piper
from rlinf.robotics.piper_runtime import PiperSession, rollout_piper


class Session:
    config = SimpleNamespace(fps=1000, max_prediction_age_s=2)
    arm = SimpleNamespace(
        joint_limits=(
            np.deg2rad([-150, 0, -170, -100, -70, -120]),
            np.deg2rad([150, 180, 0, 100, 70, 120]),
        )
    )
    clip_reset_target = PiperSession.clip_reset_target

    def __init__(self):
        self.targets = []
        self.opens = 0
        self.closes = 0

    def __enter__(self):
        self.opens += 1
        return self

    def __exit__(self, *_):
        self.closes += 1

    def observe(self):
        return {"observation.state": np.zeros(7)}

    def execute(self, target):
        self.targets.append(target.copy())
        return np.zeros(7)


class Policy:
    fps = 1000
    action_horizon = 50

    def infer(self, obs, task):
        return np.zeros((50, 7))


class Keys:
    def __init__(self, episodes):
        self.episodes = iter(episodes)
        self.events = deque()

    def clear(self):
        self.events = deque(next(self.episodes, [["right"]]))

    def poll(self):
        event = self.events.popleft() if self.events else []
        if isinstance(event, BaseException):
            raise event
        return event


def test_human_controls_during_inference_and_release_discards_old_chunk(tmp_path):
    from threading import Event

    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.teleop.piper_leader import PiperLeader, PiperLeaderConfig
    from rlinf.robotics.piper_intervention import (
        PiperIntervention,
        PiperInterventionConfig,
    )

    inference_started, finish_prediction = Event(), Event()

    class SlowPolicy(Policy):
        calls = 0

        def infer(self, obs, task):
            self.calls += 1
            if self.calls == 1:
                inference_started.set()
                assert finish_prediction.wait(3)
                return np.full((50, 7), 77.0)  # Prediction from before takeover.
            return np.full((50, 7), 1.0)

    class Follower(Session):
        def execute(self, target):
            self.last_command = np.clip(target, 0, 2).astype(np.float32)
            return super().execute(target)

    with mocked_sdks() as sdk:
        control = PiperIntervention(
            PiperLeader("can-hil", config=PiperLeaderConfig(startup_sleep_s=0)),
            PiperInterventionConfig(toggle_cooldown_s=1e-9),
        )
        control.connect()
        session = Follower()
        session.intervention = control
        device = sdk["piper_sdk"].devices[-1]
        frames = []
        events = set()
        resumed = []

        def on_policy_resume():
            assert "release" in events
            assert frames[-1]["action_source"] == "reference"
            np.testing.assert_array_equal(session.last_command, np.ones(7))
            resumed.append(len(session.targets))

        def stop(step):
            if step >= 5 and "take" not in events:
                assert inference_started.is_set() and not finish_prediction.is_set()
                control.toggle(np.zeros(7))
                device.target[:] = 5
                events.add("take")
            if step >= 10 and "release" not in events:
                assert len([f for f in frames if f["is_intervention"]]) >= 5
                control.toggle(session.last_command)
                finish_prediction.set()
                events.add("release")
            return any(f["action_source"] == "reference" for f in frames)

        try:
            with (tmp_path / "actions.jsonl").open("w") as log:
                rollout_piper(
                    SlowPolicy(),
                    session,
                    "USB",
                    steps=200,
                    log=log,
                    stop_requested=stop,
                    record_step=lambda obs, action, **kw: frames.append(
                        dict(kw, action=action.copy())
                    ),
                    on_policy_resume=on_policy_resume,
                )
            assert resumed == [len(session.targets)]
            assert events == {"take", "release"}
            assert control.takeovers == 1 and control.human_steps >= 5
            assert frames[-1]["action_source"] == "reference"
            assert not any(np.any(target == 77) for target in session.targets)
            human = next(f for f in frames if f["is_intervention"])
            np.testing.assert_array_equal(human["requested"], np.full(7, 5))
            np.testing.assert_array_equal(human["action"], np.full(7, 2))
            assert human["policy_action"] is None
            assert not control.leader.manual
            predictions = [
                json.loads(line)
                for line in (tmp_path / "actions.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "prediction"
            ]
            assert predictions[0]["discarded"] and not predictions[-1]["discarded"]
        finally:
            finish_prediction.set()
            control.close()


def test_hil_recording_keeps_critical_labels_independent_and_scores_autonomy(tmp_path):
    from robot_mocks import mocked_sdks

    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from rlinf.robotics.parts.teleop.piper_leader import PiperLeader, PiperLeaderConfig
    from rlinf.robotics.piper_intervention import (
        PiperIntervention,
        PiperInterventionConfig,
    )

    class Follower(Session):
        config = SimpleNamespace(
            fps=30, max_prediction_age_s=3, gripper_max_width_m=0.07
        )

        def read_state(self):
            return np.zeros(7)

        def observe(self):
            return {
                "observation.state": self.read_state(),
                **{
                    f"observation.images.{name}": np.zeros((16, 32, 3), np.uint8)
                    for name in ("front", "front_2", "wrist")
                },
            }

        def execute(self, target):
            self.last_command = np.zeros(7, np.float32)
            return super().execute(target)

    class Events:
        clears = 0

        def clear(self):
            self.clears += 1
            if self.clears > 1:
                self.events = iter([["right"]])

        def poll(self):
            # Right starts, one alignment poll, then b and i together before tick 0.
            return next(self.events, [])

        events = iter([["right"], [], ["b", "i"], [], [], [], ["b"], [], ["s"]])

    with mocked_sdks():
        announcements = []
        session = Follower()
        session.intervention = PiperIntervention(
            PiperLeader("can-hil", config=PiperLeaderConfig(startup_sleep_s=0)),
            PiperInterventionConfig(alignment_duration_s=0.001),
        )
        session.intervention.connect()
        policy = Policy()
        policy.fps = 30
        try:
            summary = evaluate_piper(
                policy,
                session,
                "USB",
                output=tmp_path,
                num_episodes=1,
                steps=100,
                keys=Events(),
                record_dataset=True,
                announce=lambda event, number: announcements.append(event),
            )
        finally:
            session.intervention.close()
    assert announcements == [
        "ready",
        "aligning",
        "started",
        "critical_on",
        "human",
        "critical_off",
        "success",
        "saving",
        "saved",
        "review",
        "complete",
    ]
    assert summary["success_rate"] == 1 and summary["autonomous_success_rate"] == 0
    assert summary["intervention_rate"] == 1
    result = summary["episodes"][0]
    assert result["takeovers"] == 1 and result["human_steps"] == 3
    assert result["critical_phases"] == [{"start": 0, "end": 2}]
    root = tmp_path / "episode_001/dataset"
    dataset = lerobot.LeRobotDataset(
        "piper/rollout", root=root, video_backend="torchcodec"
    )
    for idx in range(3):
        assert dataset[idx]["complementary_info.is_intervention"].item()
        assert dataset[idx]["complementary_info.action_source"].item() == 2
        assert not dataset[idx]["complementary_info.policy_action_valid"].item()
        assert dataset[idx]["complementary_info.critical_phase"].item() == (idx < 2)
    assert not dataset[3]["complementary_info.action_valid"].item()
    assert not dataset[3]["complementary_info.is_intervention"].item()
    assert (
        json.loads((root / "meta/piper_control.json").read_text())["human_steps"] == 3
    )


@pytest.mark.parametrize("num_episodes", [20, 100])
def test_trials_score_once_and_keep_one_session(tmp_path, num_episodes):
    session = Session()
    keys = Keys([
        [["s"], ["right"], [], [], ["s" if i % 2 else "f", "right", "s"]]
        for i in range(num_episodes)
    ])
    result = evaluate_piper(
        Policy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=num_episodes,
        steps=300,
        keys=keys,
        save_observations=True,
    )
    assert result["status"] == "complete"
    assert result["completed_episodes"] == num_episodes
    assert result["successes"] == result["failures"] == num_episodes // 2
    assert result["success_rate"] == 0.5
    assert len(session.targets) == num_episodes
    assert session.opens == session.closes == 1
    assert (
        len(list(tmp_path.glob("episode_*/observations/step_000000.npz")))
        == num_episodes
    )
    assert json.loads((tmp_path / "summary.json").read_text()) == result


def test_label_during_inference_prevents_first_action(tmp_path):
    session = Session()
    result = evaluate_piper(
        Policy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=300,
        keys=Keys([[["right"], [], ["s"]]]),
    )
    assert not session.targets
    assert result["successes"] == 1
    assert result["episodes"][0]["completed_steps"] == 0


def test_step_limit_waits_for_label_and_ignores_right(tmp_path):
    session = Session()
    announcements = []
    result = evaluate_piper(
        Policy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=1,
        keys=Keys([[["right"], [], [], ["right"], ["f"]]]),
        announce=lambda event, number: announcements.append(event),
    )
    assert len(session.targets) == 1
    assert result["failures"] == 1
    assert announcements == [
        "ready",
        "started",
        "waiting_score",
        "failure",
        "review",
        "complete",
    ]


@pytest.mark.parametrize(
    "failure,status",
    [(KeyboardInterrupt(), "interrupted"), (RuntimeError("fault"), "error")],
)
def test_abort_saves_partial_results_and_closes_session(tmp_path, failure, status):
    session = Session()
    announcements = []

    def announce(event, number):
        if event in ("stopped", "fault"):
            assert session.closes == 1
        announcements.append(event)

    keys = Keys([
        [["right"], [], [], ["s"]],
        [["right"], failure],
    ])
    with pytest.raises(type(failure)):
        evaluate_piper(
            Policy(),
            session,
            "USB",
            output=tmp_path,
            num_episodes=20,
            steps=300,
            keys=keys,
            announce=announce,
        )
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["status"] == status
    assert result["completed_episodes"] == 1
    assert result["success_rate"] == 1
    assert result["episodes"][1]["outcome"] == status
    assert session.closes == 1
    assert len(session.targets) == 1
    assert announcements[-1] == ("stopped" if status == "interrupted" else "fault")


def test_inference_fault_sends_no_target_and_preserves_error(tmp_path):
    class BrokenPolicy(Policy):
        def infer(self, obs, task):
            raise ValueError("bad prediction")

    session = Session()
    with pytest.raises(ValueError, match="bad prediction"):
        evaluate_piper(
            BrokenPolicy(),
            session,
            "USB",
            output=tmp_path,
            num_episodes=20,
            steps=300,
            keys=Keys([[["right"], []]]),
        )
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["completed_episodes"] == 0
    assert result["success_rate"] is None
    assert not session.targets and session.closes == 1


def test_legacy_rollout_still_owns_session_and_runs_to_limit():
    session = Session()
    assert rollout_piper(Policy(), session, "USB", steps=2, log=io.StringIO()) == 2
    assert session.opens == session.closes == 1
    assert len(session.targets) == 2


def test_terminal_reads_fragmented_arrows_and_restores_settings():
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, "rb", closefd=False) as stream:
            original = termios.tcgetattr(slave)
            with TerminalKeys(stream) as keys:
                os.write(master, b"\x1b[")
                assert keys.poll() == []
                os.write(master, b"CBsF\x1bOC")
                assert keys.poll() == ["right", "b", "s", "f", "right"]
                os.write(master, b"s\x1b[C")
                keys.clear()
                assert keys.poll() == []
                os.write(master, b"\x1b[")
                assert keys.poll() == []
                os.write(master, b"D\x1bODs\x1b[C")
                keys.clear()
                assert keys.poll() == ["left", "left"]
                os.write(master, b"\x1b[")
                keys.clear()
                os.write(master, b"D")
                assert keys.poll() == ["left"]
            assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize("mode", ["initial", "zero"])
@pytest.mark.parametrize(
    "initial, bounded",
    [
        ([1, 10, -20, 1, 2, 1, 25], [1, 10, -20, 1, 2, 1, 25]),
        (
            [-0.066, -0.511, 0.451, -4.303, 4.484, 0.474, 0.4],
            [-0.066, 0, 0, -4.303, 4.484, 0.474, 0.4],
        ),
    ],
)
def test_score_is_saved_before_reset_and_reset_reuses_connection(
    tmp_path, monkeypatch, mode, initial, bounded
):
    from rlinf.robotics import piper_evaluation

    session = Session()
    session.config = SimpleNamespace(
        fps=1000, max_prediction_age_s=2, gripper_max_width_m=0.07
    )
    initial = np.array(initial, dtype=float)
    session.read_state = lambda: initial.copy()
    resets = []

    def reset(session_arg, target, *, log, timing, on_tick=None):
        assert session_arg is session and session.opens == 1 and session.closes == 0
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["completed_episodes"] >= 1
        resets.append(target.copy())
        return {"completion": "trajectory_sent"}

    monkeypatch.setattr(piper_evaluation, "reset_piper", reset)
    summary = evaluate_piper(
        Policy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=2,
        steps=300,
        keys=Keys([[["right"], [], ["s"]], [["right"], [], ["f"]]]),
        reset_after_episode=mode,
    )
    target = bounded if mode == "initial" else np.zeros(7)
    assert len(resets) == 2
    np.testing.assert_array_equal(resets[0], target)
    np.testing.assert_array_equal(resets[1], target)
    assert all(x["reset"]["status"] == "complete" for x in summary["episodes"])
    assert session.closes == 1 and summary["success_rate"] == 0.5
    metadata = json.loads((tmp_path / "reset_pose.json").read_text())
    assert metadata["target_degrees_mm"] == list(target)
    if mode == "initial":
        assert metadata["measured_initial_degrees_mm"] == initial.tolist()


def test_reset_fault_preserves_score_and_aborts_next_trial(tmp_path, monkeypatch):
    from rlinf.robotics import piper_evaluation

    session = Session()
    session.config = SimpleNamespace(
        fps=1000, max_prediction_age_s=2, gripper_max_width_m=0.07
    )
    session.read_state = lambda: np.zeros(7)

    def reset(*args, **kwargs):
        raise TimeoutError("stalled")

    monkeypatch.setattr(piper_evaluation, "reset_piper", reset)
    with pytest.raises(TimeoutError):
        evaluate_piper(
            Policy(),
            session,
            "USB",
            output=tmp_path,
            num_episodes=20,
            steps=300,
            keys=Keys([[["right"], [], ["s"]], [["right"], ["q"]]]),
            reset_after_episode="zero",
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["successes"] == 1
    assert summary["episodes"][0]["outcome"] == "success"
    assert summary["episodes"][0]["reset"]["status"] == "error"
    assert session.closes == 1


def test_reset_retry_keeps_score_and_connection_without_starting_next_trial(
    tmp_path, monkeypatch
):
    from rlinf.robotics import piper_evaluation

    session = Session()
    session.read_state = lambda: np.zeros(7)
    attempts = []
    announcements = []

    def reset(session_arg, target, *, log, timing, on_tick=None):
        assert session_arg.opens == 1 and session_arg.closes == 0
        assert announcements[-1] == "reset_start"
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary controller fault")
        return {"completion": "trajectory_sent"}

    monkeypatch.setattr(piper_evaluation, "reset_piper", reset)
    summary = evaluate_piper(
        Policy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=300,
        reset_after_episode="zero",
        keys=Keys([[["right"], [], ["s"]], [["right"], ["s"], ["r"]]]),
        announce=lambda event, number: announcements.append(event),
    )
    assert summary["successes"] == summary["completed_episodes"] == 1
    assert len(attempts) == 2 and session.opens == session.closes == 1
    assert len(summary["episodes"][0]["reset"]["attempts"]) == 2
    assert (tmp_path / "episode_001/reset_retry_02.jsonl").exists()
    assert announcements == [
        "ready",
        "started",
        "success",
        "reset_start",
        "reset_failed",
        "reset_start",
        "reset_done",
        "review",
        "complete",
    ]


@pytest.mark.parametrize("streaming_video", [True, False])
def test_recorded_trials_have_aligned_actions_terminal_views_and_phase_labels(
    tmp_path, streaming_video
):
    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from rlinf.data.storage.lerobot.metadata import load_episode_outcomes

    class RecordedSession(Session):
        config = SimpleNamespace(fps=30, max_prediction_age_s=3)

        def observe(self):
            return {
                "observation.state": np.full(7, len(self.targets), dtype=np.float32),
                **{
                    f"observation.images.{view}": np.full(
                        (32, 48, 3), 16 * len(self.targets), dtype=np.uint8
                    )
                    for view in ("front", "front_2", "wrist")
                },
            }

        def execute(self, target):
            self.last_command = np.full(7, 2, dtype=np.float32)
            return super().execute(target)

    class RecordedPolicy(Policy):
        fps = 30

        def infer(self, obs, task):
            return np.ones((50, 7), dtype=np.float32)

    session = RecordedSession()
    announcements = []

    def announce(event, number):
        if event == "saved":
            labels = json.loads((tmp_path / "annotations.json").read_text())
            assert len(labels["episodes"]) == number
            saved = json.loads(
                (tmp_path / f"episode_{number:03d}/result.json").read_text()
            )
            assert saved["recording"]["status"] == "complete"
        announcements.append((event, number))

    summary = evaluate_piper(
        RecordedPolicy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=2,
        steps=100,
        keys=Keys([
            [
                ["right"],
                [],
                [],
                ["b"],
                [],
                [],
                [],
                ["b"],
                [],
                [],
                [],
                ["b"],
                [],
                ["b"],
                [],
                ["s"],
            ],
            [["right"], ["b"], [], ["f"]],
        ]),
        record_dataset=True,
        streaming_video=streaming_video,
        announce=announce,
    )
    assert announcements == [
        ("ready", 1),
        ("started", 1),
        ("critical_on", 1),
        ("critical_off", 1),
        ("critical_on", 2),
        ("critical_off", 2),
        ("success", 1),
        ("saving", 1),
        ("saved", 1),
        ("ready", 2),
        ("started", 2),
        ("critical_on", 1),
        ("critical_closed", 0),
        ("failure", 2),
        ("saving", 2),
        ("saved", 2),
        ("review", 2),
        ("complete", 0),
    ]
    assert summary["successes"] == summary["failures"] == 1
    assert summary["recorded_episodes"] == summary["phase_labeled_episodes"] == 2
    assert summary["episodes"][0]["recording"]["video_encoding"] == (
        "streaming" if streaming_video else "png"
    )
    labels = json.loads((tmp_path / "annotations.json").read_text())
    assert labels["format"] == "piper-rlt-episodes-v2"
    assert labels["episodes"][0]["critical_phases"] == [
        {"start": 1, "end": 3},
        {"start": 5, "end": 6},
    ]
    assert labels["episodes"][0]["terminal_frame"] == 7
    assert labels["episodes"][1]["critical_phases"] == [{"start": 0, "end": 1}]
    assert labels["episodes"][1]["terminal_frame"] == 1
    root = tmp_path / "episode_001/dataset"
    dataset = lerobot.LeRobotDataset(
        "piper/rollout", root=root, video_backend="torchcodec"
    )
    assert len(dataset) == 8 and load_episode_outcomes(root) == {0: True}
    for i in range(8):
        sample = dataset[i]
        np.testing.assert_array_equal(sample["observation.state"], np.full(7, i))
        assert bool(sample["complementary_info.action_valid"].item()) == (i < 7)
        assert bool(sample["complementary_info.critical_phase"].item()) == (
            i in (1, 2, 5)
        )
        for view in ("front", "front_2", "wrist"):
            assert sample[f"observation.images.{view}"].shape == (3, 32, 48)
            assert float(sample[f"observation.images.{view}"].mean()) == pytest.approx(
                i * 16 / 255, abs=2 / 255
            )
        if i < 7:
            np.testing.assert_array_equal(sample["action"], np.full(7, 2))
            np.testing.assert_array_equal(
                sample["complementary_info.requested_action"], np.ones(7)
            )
    extended = lerobot.LeRobotDataset(
        "piper/rollout",
        root=tmp_path / "episode_002/dataset",
        video_backend="torchcodec",
    )
    assert bool(extended[0]["complementary_info.critical_phase"].item())
    assert not bool(extended[1]["complementary_info.critical_phase"].item())
    assert not bool(extended[1]["complementary_info.action_valid"].item())


@pytest.mark.parametrize("score", ["s", "f"])
@pytest.mark.parametrize("has_prior_pair", [False, True])
@pytest.mark.parametrize("steps", [3, 100])
def test_scoring_extends_only_the_unpaired_final_phase_to_terminal(
    tmp_path, score, has_prior_pair, steps
):
    summary = evaluate_piper(
        Policy(),
        Session(),
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=steps,
        keys=Keys([
            [
                ["right"],
                ["b"],
                [],
                ["b"] if has_prior_pair else [],
                [],
                ["b"] if has_prior_pair else [],
                [],
                [score],
            ]
        ]),
    )
    saved = json.loads((tmp_path / "episode_001/result.json").read_text())
    assert saved["critical_phases"] == (
        [{"start": 0, "end": 1}, {"start": 2, "end": 3}]
        if has_prior_pair
        else [{"start": 0, "end": 3}]
    )
    assert saved["completed_steps"] == 3
    assert saved["outcome"] == ("success" if score == "s" else "failure")
    assert summary["episodes"][0] == saved


def test_scoring_ignores_a_new_phase_with_no_executed_actions(tmp_path):
    summary = evaluate_piper(
        Policy(),
        Session(),
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=100,
        keys=Keys([[["right"], [], [], ["b", "s"]]]),
    )
    assert summary["episodes"][0]["critical_phases"] == []
    assert summary["episodes"][0]["completed_steps"] == 1


def test_phase_can_close_at_step_limit_but_cannot_start_there(tmp_path):
    summary = evaluate_piper(
        Policy(),
        Session(),
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=1,
        keys=Keys([[["right"], ["b"], [], ["b", "b", "s"]]]),
    )
    assert summary["episodes"][0]["critical_phases"] == [{"start": 0, "end": 1}]


def test_duplicate_b_at_same_frame_cannot_create_an_empty_phase(tmp_path):
    summary = evaluate_piper(
        Policy(),
        Session(),
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=100,
        keys=Keys([[["right"], ["b", "b"], [], ["b", "s"]]]),
    )
    assert summary["episodes"][0]["critical_phases"] == [{"start": 0, "end": 1}]


def test_quit_at_start_saves_completed_trials_without_executing(tmp_path):
    session = Session()
    with pytest.raises(KeyboardInterrupt):
        evaluate_piper(
            Policy(),
            session,
            "USB",
            output=tmp_path,
            num_episodes=100,
            steps=100,
            keys=Keys([[["q"]]]),
        )
    assert not session.targets and session.closes == 1
    assert (
        json.loads((tmp_path / "summary.json").read_text())["status"] == "interrupted"
    )


def test_recording_failure_preserves_operator_score_and_excludes_invalid_data(
    tmp_path, monkeypatch
):
    from rlinf.data.storage.lerobot.piper import PiperEpisodeRecorder

    class RecordedSession(Session):
        config = SimpleNamespace(fps=30, max_prediction_age_s=3)

        def observe(self):
            return {
                "observation.state": np.zeros(7, dtype=np.float32),
                **{
                    f"observation.images.{view}": np.zeros((32, 48, 3), dtype=np.uint8)
                    for view in ("front", "front_2", "wrist")
                },
            }

        def execute(self, target):
            self.last_command = target.copy()
            return super().execute(target)

    class RecordedPolicy(Policy):
        fps = 30

    def fail_encoding(self, *args, **kwargs):
        saved = json.loads((tmp_path / "episode_001/result.json").read_text())
        assert saved["outcome"] == "success"
        assert saved["critical_phases"] == [{"start": 0, "end": 1}]
        raise OSError("video write failed")

    monkeypatch.setattr(PiperEpisodeRecorder, "finish", fail_encoding)
    announcements = []
    with pytest.raises(OSError, match="video write failed"):
        evaluate_piper(
            RecordedPolicy(),
            RecordedSession(),
            "USB",
            output=tmp_path,
            num_episodes=100,
            steps=100,
            keys=Keys([[["right"], ["b"], [], ["s"]]]),
            record_dataset=True,
            announce=lambda event, number: announcements.append(event),
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "error" and summary["successes"] == 1
    assert summary["recorded_episodes"] == 0
    assert not json.loads((tmp_path / "annotations.json").read_text())["episodes"]
    assert announcements[-3:] == ["success", "saving", "fault"]
    assert "saved" not in announcements


@pytest.mark.parametrize(
    "language, voice, expected",
    [
        (
            "en",
            "en-US",
            [
                "Ready for episode 1. Press right.",
                "Policy control resumed.",
                "Critical phase 2 off.",
            ],
        ),
        (
            "zh",
            "cmn",
            [
                "准备采集第1回合，按右方向键开始。",
                "模型控制已恢复。",
                "退出第2段关键阶段。",
            ],
        ),
    ],
)
def test_speech_runs_in_background_and_replaces_stale_control_prompts(
    tmp_path, monkeypatch, language, voice, expected
):
    from robot_mocks.audio import SpeechDispatcher

    backend = SpeechDispatcher(tmp_path, blocked=True)
    monkeypatch.setenv("PATH", str(tmp_path))
    with PiperAnnouncer(language=language) as announce:
        announce("ready", 1)
        backend.wait_for_calls(1)
        started = time.monotonic()
        for _ in range(30):
            announce("human")
            announce("policy_wait")
        announce("policy_resumed")
        announce("critical_on", 2)
        announce("critical_off", 2)
        announce("critical_off", 2)  # An unchanged state is not repeated.
        assert time.monotonic() - started < 0.5
        assert len(backend.calls) == 1  # Playback still blocked; enqueue never waited.
        backend.release()
        calls = backend.wait_for_calls(3)
    assert [call["args"][-1] for call in calls] == expected
    assert all(call["args"][:3] == ["--wait", "--language", voice] for call in calls)
    for call in calls:
        with pytest.raises(ChildProcessError):
            os.waitpid(call["pid"], os.WNOHANG)


@pytest.mark.parametrize("enabled", [False, True])
def test_speech_disabled_or_unavailable_does_not_break_collection(
    tmp_path, monkeypatch, caplog, enabled
):
    monkeypatch.setenv("PATH", str(tmp_path))  # No speech executable installed.
    session = Session()
    with PiperAnnouncer(enabled=enabled) as announce:
        result = evaluate_piper(
            Policy(),
            session,
            "USB",
            output=tmp_path,
            num_episodes=1,
            steps=10,
            keys=Keys([[["right"], [], [], ["s"]]]),
            announce=announce,
        )
    assert result["successes"] == 1 and len(session.targets) == 1
    assert session.closes == 1
    assert caplog.text.count("Piper speech disabled") == int(enabled)


@pytest.mark.parametrize("hung", [False, True])
def test_speech_failure_or_timeout_disables_backend_and_reaps_child(
    tmp_path, monkeypatch, caplog, hung
):
    from robot_mocks.audio import SpeechDispatcher

    backend = SpeechDispatcher(tmp_path, blocked=hung, exit_code=7)
    monkeypatch.setenv("PATH", str(tmp_path))
    with PiperAnnouncer() as announce:
        announce("ready", 1)
        calls = backend.wait_for_calls(1)
        deadline = time.monotonic() + 10
        while (
            "Piper speech disabled" not in caplog.text and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert caplog.text.count("Piper speech disabled") == 1
        assert ("exceeded eight seconds" if hung else "status 7") in caplog.text
        announce("human")
        announce("complete")
    assert len(backend.calls) == 1
    with pytest.raises(ChildProcessError):
        os.waitpid(calls[0]["pid"], os.WNOHANG)


@pytest.mark.parametrize("terminal", [None, "fault"])
def test_speech_close_cancels_pending_work_and_reaps_a_stalled_process(
    tmp_path, monkeypatch, terminal
):
    from robot_mocks.audio import SpeechDispatcher

    backend = SpeechDispatcher(tmp_path, blocked=True)
    monkeypatch.setenv("PATH", str(tmp_path))
    with PiperAnnouncer() as announce:
        announce("ready", 1)
        calls = backend.wait_for_calls(1)
        announce("human")
        if terminal:
            announce(terminal)
        started = time.monotonic()
    assert time.monotonic() - started < (4.5 if terminal else 1.5)
    announce.close()  # Idempotent, with no late subprocess or speech.
    announce("success", 1)
    assert len(backend.calls) == 1
    with pytest.raises(ChildProcessError):
        os.waitpid(calls[0]["pid"], os.WNOHANG)


def test_terminal_speech_is_delivered_after_hardware_closes(tmp_path, monkeypatch):
    from robot_mocks.audio import SpeechDispatcher

    backend = SpeechDispatcher(tmp_path / "speech", blocked=True)
    monkeypatch.setenv("PATH", str(backend.root))
    session = Session()
    (tmp_path / "collection").mkdir()
    with PiperAnnouncer() as speaker:

        def announce(event, number):
            if event == "complete":
                assert session.closes == 1
            speaker(event, number)
            if event == "ready":
                backend.wait_for_calls(1)
            if event == "complete":
                backend.release()

        evaluate_piper(
            Policy(),
            session,
            "USB",
            output=tmp_path / "collection",
            num_episodes=1,
            steps=10,
            keys=Keys([[["right"], ["b"], [], ["s"]]]),
            announce=announce,
        )
    assert [call["args"][-1] for call in backend.calls] == [
        "Ready for episode 1. Press right.",
        "Last critical phase closed at episode end.",
        "Episode 1 marked successful.",
        "Collection complete.",
    ]


def test_failed_announcement_callback_preserves_scoring_and_cleanup(tmp_path, caplog):
    calls = []

    def announce(event, number):
        calls.append(event)
        raise OSError("audio service unavailable")

    session = Session()
    summary = evaluate_piper(
        Policy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=10,
        keys=Keys([[["right"], [], [], ["s"]]]),
        announce=announce,
    )
    assert summary["successes"] == 1 and len(session.targets) == 1
    assert session.closes == 1 and calls == ["ready"]
    assert caplog.text.count("Piper announcement delivery failed") == 1


class VideoSession(Session):
    config = SimpleNamespace(fps=30, max_prediction_age_s=3, gripper_max_width_m=0.07)

    def observe(self):
        return {
            "observation.state": np.full(7, len(self.targets), dtype=np.float32),
            **{
                f"observation.images.{key}": np.full(
                    (32, 48, 3), len(self.targets), dtype=np.uint8
                )
                for key in ("front", "front_2", "wrist")
            },
        }

    def execute(self, target):
        self.last_command = target.copy()
        return super().execute(target)


class VideoPolicy(Policy):
    fps = 30


@pytest.mark.parametrize("streaming_video", [True, False])
def test_left_during_encoding_cancels_before_publication_and_replaces_trial(
    tmp_path, streaming_video
):
    from robot_mocks.video import video_output

    session = VideoSession()
    events = []
    with video_output() as (entered, release):

        def announce(event, number):
            if event == "discarded":
                assert entered.wait(2)
                assert not release.is_set()
                assert (tmp_path / "episode_001").is_dir()
                assert not (tmp_path / "discarded/episode_001").exists()
                assert (
                    json.loads((tmp_path / "annotations.json").read_text())["episodes"]
                    == []
                )
                release.set()
            events.append(event)

        summary = evaluate_piper(
            VideoPolicy(),
            session,
            "USB",
            output=tmp_path,
            num_episodes=1,
            steps=100,
            keys=Keys([
                [["right"], ["b"], [], ["s"], ["left"]],
                [["right"], ["b"], [], ["f"]],
            ]),
            record_dataset=True,
            streaming_video=streaming_video,
            announce=announce,
        )
    assert summary["completed_episodes"] == summary["recorded_episodes"] == 1
    assert summary["successes"] == 0 and summary["failures"] == 1
    assert summary["discarded_episodes"] == 1 and len(session.targets) == 2
    assert session.opens == session.closes == 1
    assert events.count("discarded") == events.count("saved") == 1
    discarded = json.loads((tmp_path / "discarded/episode_001/result.json").read_text())
    assert (
        discarded["outcome"] == "discarded"
        and discarded["original_outcome"] == "success"
    )
    assert discarded["recording"]["status"] == "discarded"
    assert not (tmp_path / "episode_001").exists()
    labels = json.loads((tmp_path / "annotations.json").read_text())
    assert list(labels["datasets"]) == ["episode_002"]
    assert labels["episodes"][0]["critical_phases"] == [{"start": 0, "end": 1}]


@pytest.mark.parametrize("record_dataset", [False, True])
def test_left_at_final_review_discards_only_last_trial_and_keeps_target_count(
    tmp_path, record_dataset
):
    session = VideoSession()
    summary = evaluate_piper(
        VideoPolicy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=2,
        steps=100,
        keys=Keys([
            [["right"], ["b"], [], ["s"]],
            [["right"], ["b"], [], ["s"]],
            [["left", "right"], ["left"], ["right"], ["b"], [], ["f"]],
        ]),
        record_dataset=record_dataset,
    )
    assert summary["completed_episodes"] == 2 and summary["discarded_episodes"] == 1
    assert summary["success_rate"] == summary["autonomous_success_rate"] == 0.5
    assert [episode["outcome"] for episode in summary["episodes"]] == [
        "success",
        "discarded",
        "failure",
    ]
    assert [episode["episode"] for episode in summary["episodes"]] == [1, 2, 2]
    assert (tmp_path / "episode_001/result.json").exists()
    assert (tmp_path / "discarded/episode_002/result.json").exists()
    assert (tmp_path / "episode_003/result.json").exists()
    if record_dataset:
        labels = json.loads((tmp_path / "annotations.json").read_text())
        assert list(labels["datasets"]) == ["episode_001", "episode_003"]
        assert summary["phase_labeled_episodes"] == summary["recorded_episodes"] == 2


def test_left_in_score_batch_and_quit_never_publishes_discarded_recording(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        evaluate_piper(
            VideoPolicy(),
            VideoSession(),
            "USB",
            output=tmp_path,
            num_episodes=1,
            steps=100,
            keys=Keys([[["right"], ["b"], [], ["s", "left", "q"]]]),
            record_dataset=True,
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_episodes"] == summary["recorded_episodes"] == 0
    assert summary["discarded_episodes"] == 1
    assert not json.loads((tmp_path / "annotations.json").read_text())["episodes"]
    assert (tmp_path / "discarded/episode_001/result.json").exists()


def test_left_during_reset_keeps_reset_and_excludes_trial(tmp_path, monkeypatch):
    from rlinf.robotics import piper_evaluation

    session = VideoSession()
    session.read_state = lambda: np.zeros(7)
    keys = Keys([[["right"], ["b"], [], ["s"]], [["right"], ["b"], [], ["f"]]])
    resets = []

    def reset(session_arg, target, *, log, timing, on_tick):
        if not resets:
            keys.events.append(["left"])
        on_tick()
        resets.append(target.copy())
        return {"completion": "trajectory_sent"}

    monkeypatch.setattr(piper_evaluation, "reset_piper", reset)
    summary = evaluate_piper(
        VideoPolicy(),
        session,
        "USB",
        output=tmp_path,
        num_episodes=1,
        steps=100,
        keys=keys,
        record_dataset=True,
        reset_after_episode="initial",
    )
    assert len(resets) == 2 and summary["completed_episodes"] == 1
    assert summary["episodes"][0]["reset"]["status"] == "complete"
    assert (tmp_path / "discarded/episode_001/reset.jsonl").exists()
    assert summary["episodes"][0]["outcome"] == "discarded"
    assert summary["failures"] == 1


@pytest.mark.parametrize("steps", [0, 45])
def test_streaming_video_copies_frames_and_preserves_timestamps_without_pngs(
    tmp_path, steps
):
    import av

    from rlinf.data.storage.lerobot.piper import PiperEpisodeRecorder

    observation = VideoSession().observe()
    recorder = PiperEpisodeRecorder(tmp_path / "dataset", fps=30, task="USB")
    try:
        recorder.prepare(observation)
        for index in range(steps):
            for key, image in observation.items():
                if key.startswith("observation.images."):
                    image.fill(index)
            recorder.add_step(
                observation,
                np.ones(7),
                requested=np.ones(7),
                measured=np.zeros(7),
                critical=index >= 20,
                wall_time=float(index),
            )
        assert not list(tmp_path.rglob("*.png"))
        for key, image in observation.items():
            if key.startswith("observation.images."):
                image.fill(steps)
        result = recorder.finish(observation, critical=False, outcome="success")
    finally:
        recorder.close()
    assert result["frames"] == steps + 1 and result["video_encoding"] == "streaming"
    videos = list(tmp_path.rglob("*.mp4"))
    assert len(videos) == 3
    for path in videos:
        with av.open(str(path)) as video:
            frames = list(video.decode(video=0))
            assert len(frames) == steps + 1
            for index, frame in enumerate(frames):
                assert float(frame.pts * frame.time_base) == pytest.approx(index / 30)
                assert float(frame.to_ndarray(format="rgb24").mean()) == pytest.approx(
                    index, abs=2
                )
    assert not list(tmp_path.rglob("streaming-video-*"))


@pytest.mark.parametrize("fail", [False, True])
def test_video_backpressure_or_writer_failure_cannot_silently_drop_frames(
    tmp_path, fail
):
    import threading

    from robot_mocks.video import video_output

    from rlinf.data.storage.lerobot.piper import PiperEpisodeRecorder

    observation = VideoSession().observe()
    recorder = PiperEpisodeRecorder(tmp_path / "dataset", fps=30, task="USB")
    with video_output(fail=fail) as (_, release):
        if fail:
            release.set()
        try:
            recorder.prepare(observation)
            with pytest.raises(RuntimeError if fail else TimeoutError):
                for _ in range(100):
                    recorder.add_step(
                        observation,
                        np.zeros(7),
                        requested=np.zeros(7),
                        measured=np.zeros(7),
                        critical=True,
                        wall_time=0,
                    )
        finally:
            release.set()
            recorder.close()
    assert not (tmp_path / "dataset/meta/episodes.jsonl").exists()
    assert not list(tmp_path.rglob("streaming-video-*"))
    assert not any(thread.name == "piper-video" for thread in threading.enumerate())
