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

"""Prepare Evo-RL Piper data and normalization for OpenPI SFT, RLT, or RECAP."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rlinf.models.embodiment.openpi.policies.piper_policy import (
    PIPER_CAMERAS,
    piper_to_model_units,
)


def prepare_dataset(
    source: Path,
    output: Path,
    failure_reward: float = -300.0,
    *,
    with_returns: bool = False,
) -> dict:
    """Copy metadata, link immutable media/data, and compute RECAP sidecars.

    Existing destinations are rejected. A failed preparation keeps its output
    for diagnosis; use a fresh destination after correcting the source.
    """
    source = source.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the source dataset.")
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("robot_type") != "piper_follower":
        raise ValueError("Expected an Evo-RL single-arm piper_follower dataset.")
    expected_names = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    for key in ("action", "observation.state"):
        feature = info["features"][key]
        if feature["shape"] != [7] or feature.get("names") != expected_names:
            raise ValueError(f"Unexpected Piper joint layout in {key}: {feature}")
    for slot, aliases in PIPER_CAMERAS.items():
        if sum(key in info["features"] for key in aliases) != 1:
            raise ValueError(f"Missing or ambiguous {slot}; expected one of {aliases}.")
    files = sorted((source / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No trajectory data in {source}")
    columns = ["episode_index", "frame_index", "action", "observation.state"]
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in files])
    table = table.sort_by(
        [("episode_index", "ascending"), ("frame_index", "ascending")]
    )
    if len(table) != info["total_frames"]:
        raise ValueError("Parquet frame count does not match meta/info.json.")
    state = piper_to_model_units(table["observation.state"].to_pylist())
    action = piper_to_model_units(table["action"].to_pylist())
    # Match the policy loader's ten-step windows and terminal padding.
    episode_indices = table["episode_index"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(episode_indices)) + 1]
    ends = np.r_[starts[1:], len(episode_indices)]
    boundaries = list(zip(episode_indices[starts], starts, ends))
    frames = table["frame_index"].to_numpy()
    for episode, start, end in boundaries:
        if not np.array_equal(frames[start:end], np.arange(end - start)):
            raise ValueError(f"Incomplete or duplicate frames in episode {episode}.")
    if len(boundaries) != info["total_episodes"]:
        raise ValueError("Episode count does not match meta/info.json.")
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        lengths = pq.read_table(path, columns=["episode_index", "length"]).to_pylist()
        actual = {episode: end - start for episode, start, end in boundaries}
        for row in lengths:
            if actual.get(row["episode_index"]) != row["length"]:
                raise ValueError(f"Incomplete episode {row['episode_index']}.")
    chunks = []
    for _, start, end in boundaries:
        indices = np.minimum(np.arange(start, end)[:, None] + np.arange(10), end - 1)
        chunks.append(action[indices].reshape(-1, 7))
    windows = np.concatenate(chunks)

    def statistics(values: np.ndarray) -> dict:
        return {
            "mean": values.mean(axis=0, dtype=np.float64).tolist(),
            "std": values.std(axis=0, dtype=np.float64).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    for directory in ("data", "videos", "images"):
        if (source / directory).exists():
            (output / directory).symlink_to(
                source / directory, target_is_directory=True
            )
    stats = None
    if with_returns:
        from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
            process_dataset,
        )

        stats = process_dataset(
            output, None, "rollout", 1.0, failure_reward, tag="piper"
        )
    norm_path = output / "meta/norm_stats.json"
    norm_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": statistics(state),
                    "actions": statistics(windows),
                }
            },
            indent=2,
        )
        + "\n"
    )
    report = {
        "source": str(source),
        "dataset": str(output),
        "frames": len(table),
        "episodes": info["total_episodes"],
        "fps": info["fps"],
        "model_units": "joint radians; gripper metres",
        "action_horizon": 10,
        "gamma": 1.0,
        "failure_reward": failure_reward,
        "norm_stats_path": str(norm_path),
        "return_stats": stats,
    }
    (
        output
        / ("meta/piper_recap.json" if with_returns else "meta/piper_dataset.json")
    ).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(*, default_with_returns: bool = False) -> None:
    """Prepare one local dataset without changing the original recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-reward", type=float, default=-300.0)
    parser.add_argument(
        "--with-returns", action="store_true", default=default_with_returns
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_dataset(
                args.source,
                args.output,
                args.failure_reward,
                with_returns=args.with_returns,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
