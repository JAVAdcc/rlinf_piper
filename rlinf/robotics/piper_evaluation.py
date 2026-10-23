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

"""Operator-scored Piper episodes with terminal keys and durable results."""

import json
import logging
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TextIO

import numpy as np

from rlinf.data.schema.critical_phase import (
    SEGMENT_LABEL_FORMAT,
    SEGMENT_PHASE_DEFINITION,
    SEGMENT_SCORING_PROTOCOL,
)
from rlinf.robotics.piper_reset import PiperResetTiming, reset_piper
from rlinf.robotics.piper_runtime import (
    ChunkPolicy,
    PiperPolicySelection,
    PiperSession,
    rollout_piper,
)


class EvaluationKeys(Protocol):
    """Keyboard event boundary for an operator-scored trial."""

    def clear(self) -> None:
        """Discard stale start/score input at a stopped episode boundary."""
        ...

    def poll(self) -> list[str]:
        """Return ordered events without blocking."""
        ...


class TerminalKeys:
    """Read episode labels and arrows from the foreground terminal without Enter."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdin if stream is None else stream
        self.pending = b""

    def __enter__(self) -> "TerminalKeys":
        if not self.stream.isatty():
            raise ValueError("Interactive evaluation requires a foreground terminal.")
        self.fd = self.stream.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)

    def clear(self) -> None:
        """Discard stale start/score keys, retaining late discard/quit requests."""
        pressed = self.poll()
        fragment = self.pending
        self.pending = (
            b"".join(
                b"\x1b[D" if key == "left" else b"q"
                for key in pressed
                if key in ("left", "q")
            )
            + fragment
        )

    def poll(self) -> list[str]:
        """Return ordered key events, retaining incomplete arrow sequences."""
        while select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 4096)
            if not data:
                raise EOFError("Evaluation terminal closed.")
            self.pending += data
        keys = []
        while self.pending:
            if self.pending.startswith(b"\x1b"):
                if len(self.pending) < 3:
                    break
                if self.pending[:3] in (b"\x1b[C", b"\x1bOC"):
                    keys.append("right")
                elif self.pending[:3] in (b"\x1b[D", b"\x1bOD"):
                    keys.append("left")
                self.pending = self.pending[3:]
            else:
                char = chr(self.pending[0]).lower()
                self.pending = self.pending[1:]
                if char in ("b", "s", "f", "r", "q", "i", "o"):
                    keys.append(char)
        return keys


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def evaluate_piper(
    policy: ChunkPolicy,
    session: PiperSession,
    task: str,
    *,
    output: Path,
    num_episodes: int,
    steps: int,
    keys: EvaluationKeys,
    replan_steps: int | None = None,
    save_observations: bool = False,
    save_rlt_reference: bool = False,
    record_dataset: bool = False,
    streaming_video: bool = True,
    phase_definition: str = SEGMENT_PHASE_DEFINITION,
    session_id: str | None = None,
    reset_after_episode: str = "none",
    reset_timing: PiperResetTiming = PiperResetTiming(),
    policy_source: str = "reference",
    announce: Callable[[str, int], None] | None = None,
    resume: bool = False,
    on_accept: Callable[[Path, dict], None] | None = None,
    on_poll: Callable[[], None] | None = None,
    on_finish: Callable | None = None,
    async_training: bool = False,
) -> dict[str, Any]:
    """Keep one session open; only send policy targets in a started episode.

    Right starts a trial; b starts a segment; s/f closes and scores that segment
    while the episode continues. Right during execution or the step limit stops
    actions. Score any still-open segment first, then score the episode with a
    separate s/f input batch. Errors abort the run; unscored trials are excluded
    from the success rate. Local results never determine the episode score.
    Left stops and discards the active trial, including while waiting for a score.
    After stopping, Left can discard that trial until Right starts another one.
    After the requested retained count, Right confirms completion; Left retries.
    Optional announcement delivery is best-effort and cannot abort collection.
    Resume restores attempts and the reset target from this exclusively owned
    output directory. ``num_episodes`` remains the total retained target.
    With ``on_accept``, Right commits the previous trial before starting another
    one. The callback runs while stopped, after recording/reset and the undo
    window. Accepted trials cannot be discarded, including after resume.
    """

    def poll_keys() -> list[str]:
        if on_poll is not None:
            on_poll()
        return keys.poll()

    if num_episodes <= 0 or steps <= 0:
        raise ValueError("Episode count and step limit must be positive.")
    if reset_after_episode not in ("none", "initial", "zero"):
        raise ValueError("Unknown reset pose selection.")
    if not phase_definition.strip():
        raise ValueError("Specify the meaning of the critical phase.")
    reset_pose = None
    control = getattr(session, "intervention", None)
    if policy_source not in ("reference", "actor"):
        raise ValueError("Policy source must be reference or actor.")
    episodes = []
    if resume:
        from rlinf.robotics.piper_collection import load_collection

        episodes, reset_pose = load_collection(
            output,
            record_dataset=record_dataset,
            reset_after_episode=reset_after_episode,
        )
    elif (output / "summary.json").exists() or any(output.glob("episode_*")):
        raise ValueError(
            "Collection already exists; use resume to preserve its history."
        )
    retained = sum(x["outcome"] in ("success", "failure") for x in episodes)
    if num_episodes < retained:
        raise ValueError(
            f"Episode target {num_episodes} is below the {retained} retained episodes."
        )

    def notify(event: str, number: int = 0) -> None:
        nonlocal announce
        if announce is not None:
            try:
                announce(event, number)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Piper announcement delivery failed; continuing without speech.",
                    exc_info=True,
                )
                announce = None

    def summarize(status: str) -> dict[str, Any]:
        scored = [x for x in episodes if x["outcome"] in ("success", "failure")]
        successes = sum(x["outcome"] == "success" for x in scored)
        autonomous = sum(
            x["outcome"] == "success" and not x.get("takeovers", 0) for x in scored
        )
        summary = {
            "scoring_protocol": SEGMENT_SCORING_PROTOCOL,
            "status": status,
            "requested_episodes": num_episodes,
            "completed_episodes": len(scored),
            "successes": successes,
            "failures": len(scored) - successes,
            "success_rate": successes / len(scored) if scored else None,
            "autonomous_successes": autonomous,
            "autonomous_success_rate": autonomous / len(scored) if scored else None,
            "intervention_rate": sum(bool(x.get("takeovers", 0)) for x in scored)
            / len(scored)
            if scored
            else None,
            "episodes": episodes,
            "discarded_episodes": sum(x["outcome"] == "discarded" for x in episodes),
        }
        if record_dataset:
            collected = [
                x for x in scored if x.get("recording", {}).get("status") == "complete"
            ]
            summary["recorded_episodes"] = len(collected)
            summary["phase_labeled_episodes"] = sum(
                bool(x["critical_phases"])
                and all(
                    phase["end"] is not None
                    and phase.get("outcome") in ("success", "failure")
                    for phase in x["critical_phases"]
                )
                for x in collected
            )
            _write_json(
                output / "annotations.json",
                {
                    "format": SEGMENT_LABEL_FORMAT,
                    "scoring_protocol": SEGMENT_SCORING_PROTOCOL,
                    "reward_scope": "critical_segment",
                    "task": task,
                    "phase_definition": phase_definition,
                    "test_only": False,
                    "datasets": {
                        x["directory"]: f"{x['directory']}/dataset" for x in collected
                    },
                    "episodes": [
                        {
                            "dataset": x["directory"],
                            "episode": 0,
                            "critical_phases": x["critical_phases"],
                            "terminal_frame": x["recording"]["terminal_frame"],
                            "outcome": x["outcome"],
                            "behavior": policy_source,
                            "split": None,
                            "session": session_id,
                            "collection_run": str(output.resolve()),
                        }
                        for x in collected
                    ],
                },
            )
        _write_json(output / "summary.json", summary)
        return summary

    def discard_trial(trial: dict) -> None:
        if trial.get("accepted", False):
            raise ValueError("An accepted training episode cannot be discarded.")
        if trial["outcome"] == "discarded":
            return
        trial["original_outcome"] = trial["outcome"]
        trial["outcome"] = "discarded"
        trial["discarded_at"] = time.time()
        _write_json(output / trial["directory"] / "result.json", trial)
        summarize("discarding")
        notify("discarded", trial["episode"])
        print(
            f"Episode {trial['episode']} discarded; excluded from scores and training. Press Right to collect a replacement after reset.",
            flush=True,
        )

    def archive_discarded(trial: dict) -> None:
        # Call only after the recorder and reset have released their files.
        if trial.get("recording_cleanup_error"):
            return
        source = output / trial["directory"]
        if source.parent == output / "discarded":
            return
        destination = output / "discarded" / source.name
        destination.parent.mkdir(exist_ok=True)
        source.rename(destination)
        trial["directory"] = str(destination.relative_to(output))
        if "recording" in trial:
            trial["recording"]["status"] = "discarded"
            trial["recording"]["dataset"] = str((destination / "dataset").resolve())
        _write_json(destination / "result.json", trial)
        summarize("running")

    def reset_trial(trial: dict, on_tick: Callable[[], None]) -> None:
        episode_dir = output / trial["directory"]
        trial.setdefault("reset", {"status": "pending", "attempts": []})
        trial["reset"].setdefault("attempts", [])
        # A crash may leave a log before the attempt was appended to result.json.
        attempt = 0
        while True:
            attempt += 1
            log_name = (
                "reset.jsonl" if attempt == 1 else f"reset_retry_{attempt:02d}.jsonl"
            )
            if (episode_dir / log_name).exists():
                continue
            trial["reset"].pop("error", None)
            trial["reset"]["status"] = "running"
            _write_json(episode_dir / "result.json", trial)
            summarize("resetting")
            try:
                notify("reset_start")
                with (episode_dir / log_name).open("x") as reset_log:
                    completed_reset = reset_piper(
                        session,
                        reset_pose,
                        log=reset_log,
                        timing=reset_timing,
                        on_tick=on_tick,
                    )
                trial["reset"].update(completed_reset, status="complete")
                trial["reset"]["attempts"].append(
                    {
                        "log": log_name,
                        "status": "complete",
                    }
                )
                notify("reset_done")
                print(
                    "Reset trajectory complete. Check the pose, restore the scene, then press Right.",
                    flush=True,
                )
                return
            except KeyboardInterrupt:
                trial["reset"]["status"] = "interrupted"
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                trial["reset"].update(status="error", error=error)
                trial["reset"]["attempts"].append(
                    {
                        "log": log_name,
                        "status": "error",
                        "error": error,
                    }
                )
                _write_json(episode_dir / "result.json", trial)
                summarize("reset_failed")
                notify("reset_failed")
                print(
                    f"Reset stopped: {error}\nScore saved; connection remains open. "
                    "After resolving the fault, press r to retry reset, or q / Ctrl-C to exit. "
                    "Right will not start another trial until reset finishes.",
                    flush=True,
                )
                keys.clear()
                while True:
                    pressed = poll_keys()
                    if "left" in pressed:
                        discard_trial(trial)
                    if "q" in pressed:
                        raise
                    if "r" in pressed:
                        break
                    time.sleep(0.02)
            finally:
                _write_json(episode_dir / "result.json", trial)
                summarize(
                    "reset_failed" if trial["reset"]["status"] == "error" else "running"
                )

    attempt_number = max((x["attempt"] for x in episodes), default=0)
    undoable = (
        episodes[-1]
        if episodes
        and episodes[-1]["outcome"] in ("success", "failure")
        and not episodes[-1].get("accepted", False)
        else None
    )
    if resume:
        print(
            f"Resuming {output}: {retained}/{num_episodes} retained; "
            f"next attempt directory is episode_{attempt_number + 1:03d}.",
            flush=True,
        )
        notify("resumed", retained)
    for trial in episodes:
        if trial["outcome"] == "discarded":
            archive_discarded(trial)
    summarize("running")
    try:
        with session:
            if (
                resume
                and reset_pose is not None
                and episodes
                and episodes[-1].get("reset", {}).get("status") != "complete"
            ):
                last_trial = episodes[-1]
                summarize("waiting_for_reset")
                notify("resume_reset")
                print(
                    "The last attempt did not finish reset. Check the workspace, "
                    "then press r to return to the saved reset pose, or q to exit. "
                    "Right cannot start an episode until reset finishes.",
                    flush=True,
                )

                def poll_resumed_reset() -> list[str]:
                    pressed = poll_keys()
                    if "left" in pressed:
                        if undoable is not None:
                            discard_trial(last_trial)
                        else:
                            notify("nothing_to_discard")
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    return pressed

                keys.clear()
                while "r" not in poll_resumed_reset():
                    time.sleep(0.02)

                def resumed_reset_tick() -> None:
                    poll_resumed_reset()

                reset_trial(last_trial, resumed_reset_tick)
                if last_trial["outcome"] == "discarded":
                    archive_discarded(last_trial)
                    undoable = None
            while True:
                number = (
                    sum(x["outcome"] in ("success", "failure") for x in episodes) + 1
                )
                final_review = number > num_episodes
                summarize(
                    "waiting_for_confirmation" if final_review else "waiting_for_start"
                )
                keys.clear()
                print(
                    "Requested episodes collected. Right confirms completion; Left discards the last episode; q exits keeping it."
                    if final_review
                    else (
                        f"[{number}/{num_episodes}] Restore the scene; press Right to start. "
                        f"Left discards the last episode. Reset after trial: {reset_after_episode}."
                    ),
                    flush=True,
                )
                notify("review" if final_review else "ready", min(number, num_episodes))
                while True:
                    pressed = poll_keys()
                    if "left" in pressed:
                        if undoable is not None:
                            discard_trial(undoable)
                            archive_discarded(undoable)
                            undoable = None
                            number -= 1
                            final_review = False
                            summarize("waiting_for_start")
                            notify("ready", number)
                        else:
                            notify("nothing_to_discard")
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    if "left" in pressed:
                        continue  # A same-batch Right cannot start after discarding.
                    if "right" in pressed:
                        break
                    time.sleep(0.02)
                if undoable is not None and on_accept is not None:
                    # Persist acceptance first. A failed update is retried from the
                    # last committed learner checkpoint on resume, never undone.
                    undoable["accepted"] = True
                    _write_json(
                        output / undoable["directory"] / "result.json", undoable
                    )
                    summarize("training")
                    notify("queued" if async_training else "training")
                    on_accept(output / undoable["directory"], undoable)
                    if not async_training:
                        notify("updated")
                    if "q" in poll_keys():
                        raise KeyboardInterrupt
                    keys.clear()
                if final_review:
                    if on_finish is not None:
                        on_finish(poll_keys)
                    break
                undoable = None
                attempt_number += 1
                if reset_pose is None and reset_after_episode != "none":
                    reset_metadata = {"mode": reset_after_episode}
                    if reset_after_episode == "initial":
                        measured_initial = session.read_state().copy()
                        reset_pose = session.clip_reset_target(measured_initial)
                        reset_metadata["measured_initial_degrees_mm"] = (
                            measured_initial.tolist()
                        )
                        if not np.array_equal(reset_pose, measured_initial):
                            print(
                                "Initial reset target clipped to configured travel; original feedback and target saved in reset_pose.json.",
                                flush=True,
                            )
                    else:
                        reset_pose = np.zeros(7)
                    reset_metadata["target_degrees_mm"] = reset_pose.tolist()
                    _write_json(
                        output / "reset_pose.json",
                        reset_metadata,
                    )
                # Any s/f or additional Right in the start batch is stale.
                if control is not None:
                    notify("aligning")
                    print(
                        "Aligning the leader to the follower before the trial.",
                        flush=True,
                    )
                    control.align(
                        session.clip_reset_target(session.read_state()),
                        session.config.fps,
                        abort=lambda: "q" in poll_keys(),
                    )
                episode_dir = output / f"episode_{attempt_number:03d}"
                episode_dir.mkdir(exist_ok=False)
                result = {
                    "scoring_protocol": SEGMENT_SCORING_PROTOCOL,
                    "episode": number,
                    "attempt": attempt_number,
                    "outcome": "unscored",
                    "completed_steps": 0,
                    "started_at": time.time(),
                    "directory": episode_dir.name,
                    "critical_phases": [],
                    "takeovers": 0,
                    "human_steps": 0,
                }
                if on_accept is not None:
                    result["policy_version"] = int(policy.version)
                _write_json(episode_dir / "result.json", result)
                summarize("running_episode")
                started = time.monotonic()
                print(
                    f"[{number}/{num_episodes}] Running: b starts a segment; s/f ends and scores only that segment. Right stops the episode for separate scoring; Left discards; q exits.",
                    flush=True,
                )
                notify("started", number)
                post_episode_keys = []
                selection = (
                    PiperPolicySelection()
                    if getattr(policy, "manual_critical", False)
                    else None
                )

                def critical_active() -> bool:
                    phases = result["critical_phases"]
                    return bool(phases) and phases[-1]["end"] is None

                def score_segment(completed: int, key: str) -> None:
                    if selection is not None:
                        selection.set_critical(False)
                    phase = result["critical_phases"][-1]
                    if completed == phase["start"]:
                        # An observation with no action cannot define a training segment.
                        result["critical_phases"].pop()
                        _write_json(episode_dir / "result.json", result)
                        notify("critical_empty")
                        print(
                            "No segment action executed; removing the empty marker.",
                            flush=True,
                        )
                    else:
                        phase.update(
                            end=completed,
                            outcome="success" if key == "s" else "failure",
                        )
                        _write_json(episode_dir / "result.json", result)
                        notify(
                            "critical_" + phase["outcome"],
                            len(result["critical_phases"]),
                        )
                        print(
                            f"Critical segment {len(result['critical_phases'])}: {phase['outcome']}, "
                            f"actions [{phase['start']}, {completed}); episode remains unscored.",
                            flush=True,
                        )

                def discard_pressed(pressed: list[str]) -> bool:
                    if "left" not in pressed:
                        return False
                    # Discard wins over labels, takeover and Right in the same batch.
                    if recorder is not None:
                        recorder.request_discard()
                    discard_trial(result)
                    post_episode_keys.extend(key for key in pressed if key == "q")
                    return True

                def stop(completed: int) -> bool:
                    result["completed_steps"] = completed
                    pressed = poll_keys()
                    if discard_pressed(pressed):
                        return True
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    if "right" in pressed:
                        # Same-batch scores cannot also score the stopped episode.
                        result["stop_reason"] = "operator"
                        return True
                    for key in pressed:
                        if (
                            control is not None
                            and key == control.config.key
                            and completed < steps
                        ):
                            if control.toggle(session.read_state()):
                                result["takeovers"] = control.takeovers
                                notify(
                                    "human"
                                    if control.mode == "human"
                                    else "policy_wait"
                                )
                                print(
                                    f"Control: {control.mode}; critical phase is unchanged.",
                                    flush=True,
                                )
                        if key == "b":
                            phases = result["critical_phases"]
                            if critical_active():
                                notify("waiting_segment_score")
                                print(
                                    "Segment already active; use s/f to end and score it.",
                                    flush=True,
                                )
                                continue
                            elif completed < steps:
                                phases.append(
                                    {
                                        "start": completed,
                                        "end": None,
                                        "outcome": "unscored",
                                    }
                                )
                            else:
                                notify("phase_limit")
                                print(
                                    "Step limit reached; cannot start another phase.",
                                    flush=True,
                                )
                                continue
                            _write_json(episode_dir / "result.json", result)
                            notify("critical_on", len(phases))
                            if selection is not None:
                                selection.set_critical(True)
                            print(
                                f"Critical segment {len(phases)} started at frame {completed}.",
                                flush=True,
                            )
                        if key in ("s", "f"):
                            if critical_active():
                                score_segment(completed, key)
                            else:
                                notify("no_active_segment")
                                print(
                                    "No active segment. Right stops the episode for separate s/f scoring.",
                                    flush=True,
                                )
                            # No repeated score or b in this input batch creates another label.
                            return False
                    return False

                def score_stopped_episode() -> None:
                    """Score frozen end observations without executing another action."""
                    episode_score_armed = (
                        result.get("stop_reason") == "operator"
                        and not critical_active()
                    )
                    while result["outcome"] == "unscored":
                        segment_pending = critical_active()
                        summarize(
                            "waiting_for_segment_score"
                            if segment_pending
                            else "waiting_for_score"
                        )
                        notify(
                            "waiting_segment_score"
                            if segment_pending
                            else (
                                "waiting_score"
                                if episode_score_armed
                                else "confirm_episode_score"
                            )
                        )
                        print(
                            "Episode stopped; s/f scores the open SEGMENT first; Left discards."
                            if segment_pending
                            else (
                                "Episode stopped; s/f scores the WHOLE EPISODE; Left discards."
                                if episode_score_armed
                                else "Episode stopped; press Right to enter WHOLE EPISODE scoring; repeated s/f is ignored."
                            ),
                            flush=True,
                        )
                        while True:
                            pressed = poll_keys()
                            if discard_pressed(pressed):
                                return
                            if "q" in pressed:
                                raise KeyboardInterrupt
                            if "right" in pressed:
                                if not segment_pending and not episode_score_armed:
                                    episode_score_armed = True
                                    break
                                # Right never doubles as a local/global score.
                                time.sleep(0.02)
                                continue
                            if not segment_pending and not episode_score_armed:
                                time.sleep(0.02)
                                continue
                            score = next((k for k in pressed if k in ("s", "f")), None)
                            if score is not None:
                                if segment_pending:
                                    score_segment(result["completed_steps"], score)
                                else:
                                    result["outcome"] = (
                                        "success" if score == "s" else "failure"
                                    )
                                    _write_json(episode_dir / "result.json", result)
                                    notify(result["outcome"], number)
                                # A segment score and episode score require separate polls.
                                break
                            time.sleep(0.02)

                recorder = None
                quit_requested = False

                def poll_after_episode() -> None:
                    nonlocal quit_requested
                    pressed = post_episode_keys + poll_keys()
                    post_episode_keys.clear()
                    if "left" in pressed:
                        discard_trial(result)
                        if recorder is not None:
                            recorder.request_discard()
                    quit_requested |= "q" in pressed

                def poll_reset() -> None:
                    poll_after_episode()
                    if quit_requested:
                        raise KeyboardInterrupt

                try:
                    if record_dataset:
                        from rlinf.data.storage.lerobot.piper import (
                            PiperEpisodeRecorder,
                        )

                        recorder = PiperEpisodeRecorder(
                            episode_dir / "dataset",
                            fps=session.config.fps,
                            task=task,
                            streaming_video=streaming_video,
                        )
                        recorder.prepare(session.observe())

                    def record_step(
                        observation: dict, action: np.ndarray, **kwargs: Any
                    ) -> None:
                        kwargs.setdefault("action_source", policy_source)
                        recorder.add_step(
                            observation,
                            action,
                            critical=critical_active(),
                            **kwargs,
                        )

                    with (episode_dir / "trace.jsonl").open("x") as log:
                        result["completed_steps"] = rollout_piper(
                            policy,
                            session,
                            task,
                            steps=steps,
                            log=log,
                            replan_steps=replan_steps,
                            save_rlt_reference=save_rlt_reference,
                            observation_dir=episode_dir / "observations"
                            if save_observations
                            else None,
                            stop_requested=stop,
                            record_step=record_step if recorder is not None else None,
                            manage_session=False,
                            policy_source=policy_source,
                            on_policy_resume=lambda: notify("policy_resumed"),
                            policy_selection=selection,
                        )
                    result["action_duration_s"] = time.monotonic() - started
                    if on_accept is not None:
                        result["policy_version_end"] = int(policy.version)
                    # Capture before scoring waits or reset can change the scene.
                    terminal_wall_time = time.time()
                    terminal_observation = (
                        session.observe()
                        if recorder is not None and result["outcome"] != "discarded"
                        else None
                    )
                    if terminal_observation is not None:
                        result["terminal_observation_wall_time"] = terminal_wall_time
                    if result["outcome"] == "unscored":
                        result.setdefault("stop_reason", "step_limit")
                        score_stopped_episode()
                    if recorder is not None and result["outcome"] == "discarded":
                        result["recording"] = {"status": "discarded"}
                    elif recorder is not None:
                        notify("saving", number)
                        print(
                            "Trial stopped; finishing videos and saving data. Left discards this trial; q exits after saving.",
                            flush=True,
                        )
                        save_started = time.monotonic()
                        result["recording"] = recorder.finish(
                            terminal_observation,
                            critical=critical_active(),
                            outcome=result["outcome"],
                            wall_time=terminal_wall_time,
                            poll=poll_after_episode,
                        )
                        result["save_duration_s"] = time.monotonic() - save_started
                        if (
                            result["outcome"] != "discarded"
                            and not result["critical_phases"]
                        ):
                            notify("no_critical")
                            print(
                                "No critical actions marked: full recording saved; annotate a critical interval before warmup.",
                                flush=True,
                            )
                except KeyboardInterrupt:
                    if result["outcome"] == "unscored":
                        result["outcome"] = "interrupted"
                    if recorder is not None:
                        result.setdefault("recording", {"status": "interrupted"})
                    raise
                except Exception as exc:
                    if result["outcome"] == "unscored":
                        result["outcome"] = "error"
                    if recorder is not None:
                        result["recording"] = {"status": "error"}
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    try:
                        if control is not None:
                            result["takeovers"] = control.takeovers
                            result["human_steps"] = control.human_steps
                        if recorder is not None:
                            try:
                                recorder.close()
                            except Exception as exc:
                                result["recording_cleanup_error"] = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                                raise
                    finally:
                        if result["outcome"] == "discarded" and "recording" in result:
                            result["recording"]["status"] = "discarded"
                        result["ended_at"] = time.time()
                        _write_json(episode_dir / "result.json", result)
                        episodes.append(result)
                        summary = summarize("running")
                poll_after_episode()
                summary = summarize("running")
                if recorder is not None and result["outcome"] != "discarded":
                    notify("saved", number)
                rate = (
                    f"{summary['success_rate']:.1%}"
                    if summary["success_rate"] is not None
                    else "n/a"
                )
                print(
                    f"[{number}/{num_episodes}] {result['outcome']}; "
                    f"success rate {summary['successes']}/{summary['completed_episodes']} "
                    f"= {rate}",
                    flush=True,
                )
                if quit_requested:
                    if result["outcome"] == "discarded":
                        archive_discarded(result)
                    raise KeyboardInterrupt
                if reset_pose is not None:
                    reset_trial(result, poll_reset)
                if result["outcome"] == "discarded":
                    archive_discarded(result)
                else:
                    undoable = result
    except KeyboardInterrupt:
        for trial in episodes:
            if trial["outcome"] == "discarded":
                archive_discarded(trial)
        summarize("interrupted")
        notify("stopped")
        raise
    except Exception:
        for trial in episodes:
            if trial["outcome"] == "discarded":
                archive_discarded(trial)
        summarize("error")
        notify("fault")
        raise
    summary = summarize("complete")
    notify("complete")
    return summary
