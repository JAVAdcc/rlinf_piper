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

"""Read task text and episode outcomes from LeRobot v2/v3 metadata."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def parse_success(value: Any) -> bool:
    """Parse explicit outcome labels, rejecting missing or ambiguous values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        label = value.strip().lower()
        if label in ("success", "true", "1"):
            return True
        if label in ("failure", "false", "0"):
            return False
    raise ValueError(f"Unknown episode success label: {value!r}")


def load_episode_outcomes(dataset_path: str | Path) -> dict[int, bool]:
    """Load Evo-RL's episode_success field when recorded in metadata."""
    meta = Path(dataset_path) / "meta"
    records = []
    for path in sorted((meta / "episodes").rglob("*.parquet")):
        columns = pq.ParquetFile(path).schema_arrow.names
        if "episode_success" in columns:
            records.extend(
                pq.read_table(
                    path, columns=["episode_index", "episode_success"]
                ).to_pylist()
            )
    jsonl = meta / "episodes.jsonl"
    if jsonl.exists():
        records.extend(
            json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
        )
    outcomes = {}
    for record in records:
        if "episode_success" not in record:
            continue
        episode = int(record["episode_index"])
        success = parse_success(record["episode_success"])
        if episode in outcomes and outcomes[episode] != success:
            raise ValueError(f"Conflicting success labels for episode {episode}.")
        outcomes[episode] = success
    return outcomes


def load_task_descriptions(dataset_path: str | Path) -> dict[int, str]:
    """Read v2 JSONL or v3 Parquet tasks, including a pandas task index."""
    meta = Path(dataset_path) / "meta"
    jsonl = meta / "tasks.jsonl"
    if jsonl.exists():
        records = [
            json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
        ]
        return {
            int(row.get("task_index", i)): str(row["task"])
            for i, row in enumerate(records)
        }
    parquet = meta / "tasks.parquet"
    if parquet.exists():
        frame = pq.read_table(parquet).to_pandas()
        if "task_index" not in frame:
            raise ValueError(f"Missing task_index in {parquet}")
        if "task" not in frame and not all(
            isinstance(index, str) for index in frame.index
        ):
            raise ValueError(f"Missing task descriptions in {parquet}")
        return {
            int(row["task_index"]): str(row["task"] if "task" in frame else index)
            for index, row in frame.iterrows()
        }
    return {}
