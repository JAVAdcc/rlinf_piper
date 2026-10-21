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

"""Make repository and shared test modules importable by unit tests."""

import sys
from pathlib import Path

import pytest

for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture
def piper_warmup_cache(tmp_path):
    """Small immutable feature pool with held-out episodes and partial chunks."""
    import numpy as np

    from rlinf.data.datasets.rlt_warmup import (
        CACHE_FORMAT,
        LABEL_FORMAT,
        cache_episode,
        episode_transitions,
        file_sha256,
        json_fingerprint,
        write_json,
    )

    root = tmp_path / "cache"
    root.mkdir()
    stats = {
        "mean": [0.0] * 7,
        "std": [1.0] * 7,
        "q01": [-1.0] * 6 + [0.0],
        "q99": [1.0] * 6 + [0.07],
    }
    write_json(
        root / "norm_stats.json", {"norm_stats": {"actions": stats, "state": stats}}
    )
    annotations = {
        "format": LABEL_FORMAT,
        "task": "TEST ONLY",
        "phase_definition": "fixture phase",
        "test_only": True,
        "datasets": {"synthetic": str(tmp_path / "source")},
        "episodes": [],
    }
    entries = []
    for index, terminal in enumerate((13, 21, 12, 16)):
        record = {
            "dataset": "synthetic",
            "episode": index,
            "critical_phases": (
                [{"start": 1, "end": 4}, {"start": 7, "end": terminal - 2}]
                if index % 2 == 0
                else [{"start": 1, "end": terminal}]
            ),
            "terminal_frame": terminal,
            "outcome": "success" if index % 2 == 0 else "failure",
            "behavior": "reference",
            "split": "train" if index < 2 else "validation",
            "session": None,
        }
        annotations["episodes"].append(record)
        transitions = episode_transitions(
            np.zeros((terminal + 1, 7), dtype=np.float32),
            phases=record["critical_phases"],
            terminal=terminal,
            chunk_length=10,
            stride=10,
            success=index % 2 == 0,
            human=np.zeros(terminal + 1, dtype=bool),
        )

        def features(frame):
            return {
                "z_rl": np.full(2048, index / 10, dtype=np.float32),
                "proprio": np.full(7, frame / 100, dtype=np.float32),
                "ref_chunk": np.full((50, 7), 0.1, dtype=np.float32),
                "raw_state": np.array([0, 20, -30, 0, 0, 0, 35], dtype=np.float32),
            }

        path = root / f"episode_{index:06d}.npz"
        digest = cache_episode(path, transitions, features)
        entries.append(
            {
                **record,
                "file": path.name,
                "sha256": digest,
                "states": len(transitions["frames"]),
                "transitions": len(transitions["current"]),
            }
        )
    manifest = {
        "format": CACHE_FORMAT,
        "annotations": annotations,
        "chunk_length": 10,
        "stride": 10,
        "stage1": {
            "checkpoint_id": "test-only",
            "fps": 30,
            "norm_stats_sha256": file_sha256(root / "norm_stats.json"),
        },
        "model": {
            "model_type": "rlt_mlp_policy",
            "precision": "fp32",
            "z_dim": 2048,
            "proprio_dim": 7,
            "action_dim": 7,
            "num_action_chunks": 10,
            "ref_num_action_chunks": 50,
            "add_q_head": True,
            "q_head_type": "default",
            "fixed_std": 0.002,
        },
        "episodes": entries,
    }
    manifest["content_fingerprint"] = json_fingerprint(manifest)
    write_json(root / "manifest.json", manifest)
    return root


@pytest.fixture
def piper_local_warmup_cache(piper_warmup_cache):
    """Scored local segments whose outcomes deliberately differ from parents."""
    import json

    import numpy as np

    from rlinf.data.datasets.rlt_warmup import (
        cache_episode,
        episode_transitions,
        file_sha256,
        json_fingerprint,
        write_json,
    )

    root = piper_warmup_cache
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["format"] = "piper-rlt-cache-v3"
    manifest["annotations"].update(
        format="piper-rlt-episodes-v3",
        reward_scope="critical_segment",
        scoring_protocol="critical-segment-v1",
    )
    for entry, label in zip(
        manifest["episodes"], manifest["annotations"]["episodes"], strict=True
    ):
        for record in (entry, label):
            for ordinal, phase in enumerate(record["critical_phases"]):
                phase["outcome"] = "success" if ordinal == 0 else "failure"
        with np.load(root / entry["file"], allow_pickle=False) as shard:
            old = {key: shard[key].copy() for key in shard.files}
        end = entry["terminal_frame"]
        transitions = episode_transitions(
            np.zeros((end + 1, 7), dtype=np.float32),
            phases=entry["critical_phases"],
            terminal=end,
            chunk_length=10,
            stride=10,
            success=entry["outcome"] == "success",
            human=np.zeros(end + 1, dtype=bool),
            reward_scope="critical_segment",
        )

        def features(frame):
            row = np.flatnonzero(old["frames"] == frame).item()
            return {
                key: old[key][row]
                for key in ("z_rl", "proprio", "ref_chunk", "raw_state")
            }

        cache_episode(root / entry["file"], transitions, features)
        entry.update(
            sha256=file_sha256(root / entry["file"]),
            states=len(transitions["frames"]),
            transitions=len(transitions["current"]),
        )
    manifest.pop("content_fingerprint")
    manifest["content_fingerprint"] = json_fingerprint(manifest)
    write_json(root / "manifest.json", manifest)
    return root
