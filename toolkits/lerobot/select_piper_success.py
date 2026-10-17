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

"""Build a success-only LeRobot v3 Piper RGB view without copying videos."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def select_success(manifest: Path, output: Path) -> dict:
    """Materialize selected core columns, preserving video time offsets.

    The manifest contains ``selected_episodes`` with ``source_dataset``,
    ``source_episode_index`` and ``frames``. Success is rechecked against source
    metadata. Depth and complementary columns are omitted because Piper's
    OpenPI input transform uses RGB, state and actions only. Source recordings
    remain untouched. Existing outputs are rejected; incomplete outputs are
    retained for diagnosis and must not be used for training.
    """
    selection = json.loads(manifest.read_text())["selected_episodes"]
    if not selection:
        raise ValueError("Selection must contain at least one episode.")
    identities = [
        (str(Path(r["source_dataset"]).resolve()), r["source_episode_index"])
        for r in selection
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate source episodes in selection.")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    columns = [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    sources = {}
    reference = None
    task_rows = None
    for row in selection:
        root = Path(row["source_dataset"]).resolve()
        if output.resolve().is_relative_to(root):
            raise ValueError("Output must be outside source datasets.")
        if str(root) in sources:
            continue
        info = json.loads((root / "meta/info.json").read_text())
        if info["codebase_version"] != "v3.0" or info["robot_type"] != "piper_follower":
            raise ValueError(f"Expected LeRobot v3 single-arm Piper: {root}")
        cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        features = {k: info["features"][k] for k in columns + cameras}
        signature = (info["fps"], features)
        if reference is None:
            reference = signature
            result_info = copy.deepcopy(info)
            result_info["features"] = features
            for key in (
                "camera_calibrations",
                "camera_calibration_format_version",
                "uncalibrated_depth_cameras",
            ):
                result_info.pop(key, None)
            task_rows = pq.read_table(root / "meta/tasks.parquet")
        elif signature != reference or not task_rows.equals(
            pq.read_table(root / "meta/tasks.parquet")
        ):
            raise ValueError(
                "Source FPS, RGB/core features and task mappings must match."
            )
        episodes = [
            ep
            for path in sorted((root / "meta/episodes").rglob("*.parquet"))
            for ep in pq.read_table(path).to_pylist()
        ]
        mapping = {ep["episode_index"]: ep for ep in episodes}
        if len(mapping) != len(episodes):
            raise ValueError(f"Duplicate episode metadata in {root}")
        sources[str(root)] = (root, info, mapping, cameras)
    for row in selection:
        root, info, episodes, cameras = sources[
            str(Path(row["source_dataset"]).resolve())
        ]
        ep = episodes[row["source_episode_index"]]
        if ep["episode_success"] != "success" or ep["length"] != row["frames"]:
            raise ValueError(f"Selection disagrees with source success/length: {row}")

    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta/episodes/chunk-000").mkdir(parents=True)
    result_info.update(
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        chunks_size=1000,
        total_episodes=len(selection),
        splits={"train": f"0:{len(selection)}"},
    )
    episode_rows, provenance, states, actions = [], [], [], []
    links = {}
    offset = 0
    for new_index, row in enumerate(selection):
        root, info, episodes, cameras = sources[
            str(Path(row["source_dataset"]).resolve())
        ]
        ep = episodes[row["source_episode_index"]]
        source_path = root / info["data_path"].format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )
        table = pq.read_table(source_path, columns=columns)
        table = table.filter(
            pa.compute.equal(table["episode_index"], ep["episode_index"])
        )
        n = len(table)
        if n != row["frames"] or not np.array_equal(
            table["frame_index"].to_numpy(), np.arange(n)
        ):
            raise ValueError(f"Incomplete/unordered episode: {source_path}")
        for key, destination in (("action", actions), ("observation.state", states)):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (n, 7) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in {source_path}")
            destination.append(values)
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(np.full(n, new_index, dtype=np.int64)),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(np.arange(offset, offset + n, dtype=np.int64)),
        )
        # Drop the source HuggingFace schema, which still describes depth columns.
        table = table.replace_schema_metadata(None)
        chunk, file = divmod(new_index, 1000)
        path = output / result_info["data_path"].format(
            chunk_index=chunk, file_index=file
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        metadata = {
            "episode_index": new_index,
            "tasks": ep["tasks"],
            "length": n,
            "episode_success": "success",
            "data/chunk_index": chunk,
            "data/file_index": file,
            "dataset_from_index": offset,
            "dataset_to_index": offset + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera in cameras:
            prefix = f"videos/{camera}"
            video = root / info["video_path"].format(
                video_key=camera,
                chunk_index=ep[f"{prefix}/chunk_index"],
                file_index=ep[f"{prefix}/file_index"],
            )
            if not video.is_file():
                raise FileNotFoundError(video)
            identity = (camera, str(video.resolve()))
            if identity not in links:
                video_index = sum(key[0] == camera for key in links)
                links[identity] = divmod(video_index, 1000)
                vc, vf = links[identity]
                target = output / result_info["video_path"].format(
                    video_key=camera, chunk_index=vc, file_index=vf
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(video.resolve())
            vc, vf = links[identity]
            metadata.update(
                {
                    f"{prefix}/chunk_index": vc,
                    f"{prefix}/file_index": vf,
                    f"{prefix}/from_timestamp": ep[f"{prefix}/from_timestamp"],
                    f"{prefix}/to_timestamp": ep[f"{prefix}/to_timestamp"],
                }
            )
        episode_rows.append(metadata)
        provenance.append({**row, "episode_index": new_index})
        offset += n

    result_info["total_frames"] = offset
    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        output / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(task_rows, output / "meta/tasks.parquet")
    stats = {}
    for name, arrays in (("action", actions), ("observation.state", states)):
        values = np.concatenate(arrays).astype(np.float64)
        stats[name] = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "count": [len(values)],
            **{
                f"q{q:02}": np.quantile(values, q / 100, axis=0).tolist()
                for q in (1, 10, 50, 90, 99)
            },
        }
    report = {
        "episodes": len(selection),
        "frames": offset,
        "selected_episodes": provenance,
        "video_storage": "symlinks with original episode time offsets; no re-encoding",
        "omitted_features": sorted(
            set(info["features"]) - set(result_info["features"])
        ),
        "stats_scope": "Exact selected-frame raw action/state statistics only. OpenPI RGB preprocessing uses fixed scaling, not dataset image statistics. Run prepare_piper for model-unit action-window normalization.",
        "status": "complete",
    }
    for name, value in (
        ("info.json", result_info),
        ("stats.json", stats),
        ("selection.json", report),
    ):
        (output / "meta" / name).write_text(json.dumps(value, indent=2) + "\n")
    return {"dataset": str(output), "episodes": len(selection), "frames": offset}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_success(args.manifest, args.output), indent=2))
