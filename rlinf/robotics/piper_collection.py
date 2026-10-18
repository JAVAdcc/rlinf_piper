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

"""Exclusive ownership and recovery of a local Piper collection directory."""

import fcntl
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


@contextmanager
def collection_lock(output: Path) -> Iterator[None]:
    """Hold the run lock until model loading, collection and cleanup finish.

    The lock file remains on disk; the OS releases ownership even after a crash.
    Callers must create a new run directory before acquiring its lock.
    """
    with (output / ".collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Collection is already running: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_collection_json(path: Path) -> dict[str, Any]:
    """Read metadata without changing it; report the failing file on corruption."""
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read collection metadata {path}: {exc}") from exc


def _validate_recording(directory: Path, trial: dict[str, Any]) -> None:
    """Check published frame counts and file headers without decoding RGB."""
    import av
    import pyarrow.parquet as pq

    root = directory / "dataset"
    recording = trial["recording"]
    frames = trial["completed_steps"] + 1
    info = read_collection_json(root / "meta/info.json")
    if (
        recording.get("terminal_frame") != frames - 1
        or recording.get("frames") != frames
        or info.get("total_frames") != frames
        or info.get("total_episodes") != 1
    ):
        raise ValueError(f"Recording frame counts disagree: {directory}")
    try:
        data_files = list((root / "data").rglob("*.parquet"))
        if sum(pq.read_metadata(path).num_rows for path in data_files) != frames:
            raise ValueError("action table frame count disagrees")
        video_keys = [
            key
            for key, feature in info["features"].items()
            if feature["dtype"] == "video"
        ]
        if not video_keys:
            raise ValueError("no RGB video features")
        for key in video_keys:
            video_frames = 0
            for path in (root / "videos" / key).rglob("*.mp4"):
                with av.open(str(path)) as video:
                    video_frames += video.streams.video[0].frames
            if video_frames != frames:
                raise ValueError(f"{key} video frame count disagrees")
        if not (root / "meta/episodes.jsonl").is_file():
            raise ValueError("missing terminal outcome metadata")
    except Exception as exc:
        raise ValueError(f"Cannot resume damaged recording {directory}: {exc}") from exc


def load_collection(
    output: Path, *, record_dataset: bool, reset_after_episode: str
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """Recover stopped attempts and the original reset target without writing.

    Per-attempt results take precedence over the aggregate summary, which may
    lag behind a crash. Directory names reserve attempt numbers, including
    discarded and interrupted attempts. Unfinished recordings remain on disk
    but cannot contribute to the retained count or training annotations.
    Published recordings that are damaged cause an error instead of data loss.
    The caller must hold ``collection_lock`` while applying recovered state.
    """
    summary = (
        read_collection_json(output / "summary.json")
        if (output / "summary.json").exists()
        else {}
    )
    directories: dict[int, Path] = {}
    for parent in (output, output / "discarded"):
        for directory in parent.glob("episode_*"):
            match = re.fullmatch(r"episode_(\d+)", directory.name)
            if match is None or not directory.is_dir() or directory.is_symlink():
                raise ValueError(f"Unexpected episode directory: {directory}")
            number = int(match[1])
            if number < 1 or number in directories:
                raise ValueError(f"Duplicate or invalid attempt number: {directory}")
            directories[number] = directory
    previous = summary.get("episodes", [])
    if not isinstance(previous, list) or any(not isinstance(x, dict) for x in previous):
        raise ValueError(f"Invalid episode list: {output / 'summary.json'}")
    indexed = {}
    for trial in previous:
        name = trial.get("directory", "")
        match = re.fullmatch(r"(?:discarded/)?episode_(\d+)", name)
        if match is None or int(match[1]) not in directories:
            raise ValueError(f"Episode from summary is missing: {name!r}")
        indexed[int(match[1])] = trial

    episodes = []
    retained = 0
    for attempt, directory in sorted(directories.items()):
        result_path = directory / "result.json"
        if not result_path.exists() and indexed.get(attempt, {}).get("outcome") in (
            "success",
            "failure",
        ):
            raise ValueError(f"Saved episode result is missing: {result_path}")
        trial = (
            read_collection_json(result_path)
            if result_path.exists()
            else {
                "episode": retained + 1,
                "outcome": "unscored",
                "completed_steps": 0,
                "critical_phases": [],
            }
        )
        if trial.get("attempt", attempt) != attempt:
            raise ValueError(f"Attempt number disagrees with directory: {directory}")
        trial["attempt"] = attempt
        trial["directory"] = str(directory.relative_to(output))
        trial.setdefault("critical_phases", [])
        if not isinstance(trial.get("episode"), int) or trial["episode"] <= 0:
            raise ValueError(f"Invalid episode number: {result_path}")
        if directory.parent == output / "discarded":
            if trial["outcome"] != "discarded":
                trial["original_outcome"] = trial["outcome"]
            trial["outcome"] = "discarded"
        outcome = trial.get("outcome")
        if outcome not in (
            "success",
            "failure",
            "discarded",
            "interrupted",
            "error",
            "unscored",
        ):
            raise ValueError(f"Unknown episode outcome: {result_path}")
        if outcome == "unscored" or (
            outcome in ("success", "failure")
            and record_dataset
            and trial.get("recording", {}).get("status") != "complete"
        ):
            trial["original_outcome"] = outcome
            trial["outcome"] = "interrupted"
            trial["recovery_note"] = "Interrupted before scoring or recording commit."
        if trial["outcome"] in ("success", "failure"):
            steps = trial.get("completed_steps")
            if not isinstance(steps, int) or steps < 0:
                raise ValueError(f"Invalid episode length: {result_path}")
            if not isinstance(trial["critical_phases"], list):
                raise ValueError(f"Invalid critical phase list: {result_path}")
            previous_end = 0
            for phase in trial.get("critical_phases", []):
                start, end = phase.get("start"), phase.get("end")
                if not (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and previous_end <= start < end <= steps
                ):
                    raise ValueError(f"Invalid critical phase: {result_path}")
                previous_end = end
            if record_dataset:
                _validate_recording(directory, trial)
            retained += 1
        if "recording" in trial:
            trial["recording"]["dataset"] = str((directory / "dataset").resolve())
            if trial["outcome"] == "discarded":
                trial["recording"]["status"] = "discarded"
        episodes.append(trial)

    reset_pose = None
    reset_path = output / "reset_pose.json"
    if reset_after_episode != "none":
        if reset_path.exists():
            metadata = read_collection_json(reset_path)
            reset_pose = np.asarray(metadata.get("target_degrees_mm"), dtype=float)
            if (
                metadata.get("mode") != reset_after_episode
                or reset_pose.shape != (7,)
                or not np.isfinite(reset_pose).all()
            ):
                raise ValueError(f"Invalid saved reset target: {reset_path}")
        elif episodes:
            raise ValueError(f"Original reset target is missing: {reset_path}")
    return episodes, reset_pose
