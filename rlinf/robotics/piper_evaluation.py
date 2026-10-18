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

from rlinf.robotics.piper_reset import PiperResetTiming, reset_piper
from rlinf.robotics.piper_runtime import ChunkPolicy, PiperSession, rollout_piper


class EvaluationKeys(Protocol):
    """Keyboard event boundary for an operator-scored trial."""

    def clear(self) -> None:
        """Discard pending events."""
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
        """Discard keys from the previous episode or model startup."""
        self.pending = b""
        termios.tcflush(self.fd, termios.TCIFLUSH)

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
    record_dataset: bool = False,
    phase_definition: str = "Each pair of b presses marks a critical interval; s/f ends the task.",
    session_id: str | None = None,
    reset_after_episode: str = "none",
    reset_timing: PiperResetTiming = PiperResetTiming(),
    policy_source: str = "reference",
    announce: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Keep one session open; only send policy targets in a started episode.

    Right starts a trial; b alternates phase start/end; s/f scores and stops it.
    At the step limit, wait for a label without sending further actions. Errors
    abort the whole run. Unscored trials are excluded from the success rate.
    Scoring closes an active final phase at the terminal observation frame.
    Optional announcement delivery is best-effort and cannot abort collection.
    """
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
        }
        if record_dataset:
            collected = [
                x for x in scored if x.get("recording", {}).get("status") == "complete"
            ]
            summary["recorded_episodes"] = len(collected)
            summary["phase_labeled_episodes"] = sum(
                bool(x["critical_phases"])
                and all(phase["end"] is not None for phase in x["critical_phases"])
                for x in collected
            )
            _write_json(
                output / "annotations.json",
                {
                    "format": "piper-rlt-episodes-v2",
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

    summarize("running")
    try:
        with session:
            for number in range(1, num_episodes + 1):
                summarize("waiting_for_start")
                keys.clear()
                print(
                    f"[{number}/{num_episodes}] Restore the scene; press Right to start. "
                    f"Reset after scoring: {reset_after_episode}.",
                    flush=True,
                )
                notify("ready", number)
                while True:
                    pressed = keys.poll()
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    if "right" in pressed:
                        break
                    time.sleep(0.02)
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
                        abort=lambda: "q" in keys.poll(),
                    )
                episode_dir = output / f"episode_{number:03d}"
                episode_dir.mkdir(exist_ok=False)
                result = {
                    "episode": number,
                    "outcome": "unscored",
                    "completed_steps": 0,
                    "started_at": time.time(),
                    "directory": episode_dir.name,
                    "critical_phases": [],
                    "takeovers": 0,
                    "human_steps": 0,
                }
                summarize("running_episode")
                started = time.monotonic()
                print(
                    f"[{number}/{num_episodes}] Running: b = enter/exit critical phase, s = success, f = failure; q exits.",
                    flush=True,
                )
                notify("started", number)

                def critical_active() -> bool:
                    phases = result["critical_phases"]
                    return bool(phases) and phases[-1]["end"] is None

                def stop(completed: int) -> bool:
                    result["completed_steps"] = completed
                    for key in keys.poll():
                        if key == "q":
                            raise KeyboardInterrupt
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
                                if completed == phases[-1]["start"]:
                                    notify("critical_empty")
                                    print(
                                        "No action since phase start; ignoring an empty end marker.",
                                        flush=True,
                                    )
                                    continue
                                phases[-1]["end"] = completed
                                event = "ended"
                            elif completed < steps:
                                phases.append({"start": completed, "end": None})
                                event = "started"
                            else:
                                notify("phase_limit")
                                print(
                                    "Step limit reached; cannot start another phase.",
                                    flush=True,
                                )
                                continue
                            _write_json(episode_dir / "result.json", result)
                            notify(
                                "critical_on" if critical_active() else "critical_off",
                                len(phases),
                            )
                            print(
                                f"Critical phase {len(phases)} {event} at frame {completed}; "
                                f"critical={'ON' if critical_active() else 'OFF'}.",
                                flush=True,
                            )
                        if key in ("s", "f"):
                            result["outcome"] = "success" if key == "s" else "failure"
                            phase_event = None
                            if critical_active():
                                phases = result["critical_phases"]
                                if phases[-1]["start"] < completed:
                                    phases[-1]["end"] = completed
                                    phase_event = "critical_closed"
                                    print(
                                        f"Critical phase {len(phases)} automatically ended at terminal frame {completed}.",
                                        flush=True,
                                    )
                                else:
                                    phases.pop()
                                    phase_event = "critical_empty"
                                    print(
                                        "No action after the last b marker; discarding the empty phase.",
                                        flush=True,
                                    )
                            _write_json(episode_dir / "result.json", result)
                            if phase_event is not None:
                                notify(phase_event)
                            notify(result["outcome"], number)
                            return True
                    return False

                recorder = None
                try:
                    if record_dataset:
                        from rlinf.data.storage.lerobot.piper import (
                            PiperEpisodeRecorder,
                        )

                        recorder = PiperEpisodeRecorder(
                            episode_dir / "dataset",
                            fps=session.config.fps,
                            task=task,
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
                            observation_dir=episode_dir / "observations"
                            if save_observations
                            else None,
                            stop_requested=stop,
                            record_step=record_step if recorder is not None else None,
                            manage_session=False,
                            policy_source=policy_source,
                            on_policy_resume=lambda: notify("policy_resumed"),
                        )
                    result["action_duration_s"] = time.monotonic() - started
                    # Capture before scoring waits or reset can change the scene.
                    terminal_observation = (
                        session.observe() if recorder is not None else None
                    )
                    if result["outcome"] == "unscored":
                        summarize("waiting_for_score")
                        notify("waiting_score")
                        print(
                            "Step limit reached; holding. Press s or f to score.",
                            flush=True,
                        )
                        while not stop(result["completed_steps"]):
                            time.sleep(0.02)
                    if recorder is not None:
                        notify("saving", number)
                        print(
                            "Trial stopped; saving its dataset and encoding videos.",
                            flush=True,
                        )
                        result["recording"] = recorder.finish(
                            terminal_observation,
                            critical=critical_active(),
                            outcome=result["outcome"],
                        )
                        if not result["critical_phases"]:
                            notify("no_critical")
                            print(
                                "No critical actions marked: full recording saved; annotate a critical interval before warmup.",
                                flush=True,
                            )
                except KeyboardInterrupt:
                    if result["outcome"] == "unscored":
                        result["outcome"] = "interrupted"
                    if recorder is not None:
                        result["recording"] = {"status": "interrupted"}
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
                            recorder.close()
                    finally:
                        result["ended_at"] = time.time()
                        _write_json(episode_dir / "result.json", result)
                        episodes.append(result)
                        summary = summarize("running")
                if recorder is not None:
                    notify("saved", number)
                print(
                    f"[{number}/{num_episodes}] {result['outcome']}; "
                    f"success rate {summary['successes']}/{summary['completed_episodes']} "
                    f"= {summary['success_rate']:.1%}",
                    flush=True,
                )
                if reset_pose is not None:
                    result["reset"] = {"status": "pending", "attempts": []}
                    attempt = 0
                    while True:
                        attempt += 1
                        result["reset"].pop("error", None)
                        result["reset"]["status"] = "running"
                        _write_json(episode_dir / "result.json", result)
                        summarize("resetting")
                        log_name = (
                            "reset.jsonl"
                            if attempt == 1
                            else f"reset_retry_{attempt:02d}.jsonl"
                        )
                        try:
                            notify("reset_start")
                            with (episode_dir / log_name).open("x") as reset_log:
                                completed_reset = reset_piper(
                                    session,
                                    reset_pose,
                                    log=reset_log,
                                    timing=reset_timing,
                                )
                            result["reset"].update(completed_reset, status="complete")
                            result["reset"]["attempts"].append({
                                "log": log_name,
                                "status": "complete",
                            })
                            notify("reset_done")
                            print(
                                "Reset trajectory complete. Check the pose, restore the scene, then press Right.",
                                flush=True,
                            )
                            break
                        except KeyboardInterrupt:
                            result["reset"]["status"] = "interrupted"
                            raise
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            result["reset"].update(status="error", error=error)
                            result["reset"]["attempts"].append({
                                "log": log_name,
                                "status": "error",
                                "error": error,
                            })
                            _write_json(episode_dir / "result.json", result)
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
                                pressed = keys.poll()
                                if "q" in pressed:
                                    raise
                                if "r" in pressed:
                                    break
                                time.sleep(0.02)
                        finally:
                            _write_json(episode_dir / "result.json", result)
                            summarize(
                                "reset_failed"
                                if result["reset"]["status"] == "error"
                                else "running"
                            )
    except KeyboardInterrupt:
        summarize("interrupted")
        notify("stopped")
        raise
    except Exception:
        summarize("error")
        notify("fault")
        raise
    summary = summarize("complete")
    notify("complete")
    return summary
