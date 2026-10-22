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

"""Convert accepted Piper segments to native RLT replay trajectories."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from rlinf.data.datasets.rlt_warmup import (
    cache_episode,
    episode_transitions,
    file_sha256,
    json_fingerprint,
    write_json,
)
from rlinf.data.schema.embodied_types import Trajectory

if TYPE_CHECKING:
    from rlinf.models.embodiment.piper_inference import PiperRLTPolicy


def segment_trajectory(arrays: dict[str, torch.Tensor], version: int) -> Trajectory:
    """Pack warmup-compatible, variable-length transitions into native replay.

    T is the number of chunk decisions and B is one physical robot. Local
    termination never crosses a segment boundary; reward sits on the last
    executed tick. Padding carries neither reward nor human supervision.
    """
    actions = arrays["actions"]
    count, chunk, dim = actions.shape
    lengths = arrays["valid_steps"]
    if count == 0 or not ((lengths > 0) & (lengths <= chunk)).all():
        raise ValueError("Online replay requires nonempty executed chunks.")
    if not torch.equal(arrays["discount_steps"], lengths):
        raise ValueError("Segment replay cannot bootstrap across reference phases.")
    mask = torch.arange(chunk)[None, :] < lengths[:, None]
    rewards = torch.zeros(count, chunk, dtype=torch.float32)
    rewards[torch.arange(count), lengths - 1] = arrays["rewards"]
    done = torch.zeros(count, chunk, dtype=torch.bool)
    done[torch.arange(count), lengths - 1] = arrays["terminated"]
    human = arrays["human"] & mask
    return Trajectory(
        max_episode_length=int(lengths.sum()),
        model_weights_id=str(version),
        actions=(actions * mask[..., None]).reshape(count, 1, chunk * dim),
        intervene_flags=human[..., None].expand(-1, -1, dim).reshape(count, 1, -1),
        rewards=rewards[:, None],
        terminations=done[:, None],
        dones=done[:, None],
        truncations=torch.zeros_like(done[:, None]),
        versions=torch.full((count, 1, 1), version, dtype=torch.int64),
        forward_inputs={
            key: arrays[key][:, None]
            for key in ("valid_steps", "discount_steps", "next_valid_steps")
        },
        curr_obs={
            key: arrays[key][arrays["current"]][:, None]
            for key in ("z_rl", "proprio", "ref_chunk")
        },
        next_obs={
            key: arrays[key][arrays["next"]][:, None]
            for key in ("z_rl", "proprio", "ref_chunk")
        },
    )


def save_replay_anchor(
    directory: Path, frame: int, manifest: dict, task: str, features: dict
) -> None:
    """Persist the frozen features of a rollout observation in normalized units."""
    root = directory / "replay_anchors"
    root.mkdir(exist_ok=True)
    path = root / f"frame_{frame:06d}.npz"
    binding = json_fingerprint(
        [manifest["checkpoint_id"], manifest["norm_stats_sha256"], task, frame]
    )
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            binding=np.asarray(binding),
            **{
                key: np.asarray(features[key], dtype=np.float32)
                for key in ("z_rl", "proprio", "ref_chunk", "raw_state")
            },
        )
    temporary.replace(path)


def backfill_replay_features(
    policy: "PiperRLTPolicy",
    frames: np.ndarray,
    sample_at: Callable[[int], dict],
    *,
    directory: Path,
    task: str,
    fingerprint: str,
    batch_size: int,
) -> tuple[dict, dict]:
    """Reuse rollout anchors; batch only missing states and preserve rollout RNG."""
    if batch_size < 1:
        raise ValueError("Replay feature batch size must be positive.")
    started = time.monotonic()
    features = {}
    for frame in map(int, frames):
        path = directory / "replay_anchors" / f"frame_{frame:06d}.npz"
        if not path.exists():
            continue
        sample = sample_at(frame)
        # A request cancelled before execute/record may share the terminal frame
        # index. Its observation is not the recorder's later terminal observation.
        if not np.asarray(sample.get("complementary_info.action_valid", True)).all():
            continue
        with np.load(path, allow_pickle=False) as saved:
            binding = json_fingerprint(
                [
                    policy.manifest["checkpoint_id"],
                    policy.manifest["norm_stats_sha256"],
                    task,
                    frame,
                ]
            )
            if str(saved["binding"]) != binding:
                raise ValueError(
                    "Rollout feature anchor belongs to another data contract."
                )
            values = {
                key: saved[key].copy()
                for key in ("z_rl", "proprio", "ref_chunk", "raw_state")
            }
        if not np.allclose(
            values["raw_state"],
            sample["observation.state"],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError(
                "Rollout feature anchor is not aligned to the recorded frame."
            )
        features[frame] = values
    reused = len(features)
    missing = [int(frame) for frame in frames if int(frame) not in features]
    device = next(policy.feature_model.parameters()).device
    for start in range(0, len(missing), batch_size):
        batch_frames = missing[start : start + batch_size]
        samples = [sample_at(frame) for frame in batch_frames]
        if any(sample["task"] != task for sample in samples):
            raise ValueError("Recorded task text changed.")
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(
                int(json_fingerprint([fingerprint, batch_frames])[:15], 16)
            )
            values = policy.extract_features_batch(samples, task)
        for frame, value in zip(batch_frames, values, strict=True):
            features[frame] = value
    return features, {
        "reused_anchors": reused,
        "backfilled_anchors": len(missing),
        "feature_batches": (len(missing) + batch_size - 1) // batch_size,
        "feature_seconds": time.monotonic() - started,
    }


def cache_accepted_episode(
    policy: "PiperRLTPolicy",
    directory: Path,
    result: dict,
    task: str,
    *,
    stride: int | None = None,
    tail_policy: str = "pad",
    feature_batch_size: int = 1,
) -> Trajectory | None:
    """Extract frozen features once, only after the operator commits an episode.

    This runs between episodes on the rollout worker's existing VLA. Decoding,
    frame selection, action normalization and local terminals match warmup.
    A durable shard avoids repeated extraction after a learner interruption.
    """
    if not result.get("accepted") or result["outcome"] not in ("success", "failure"):
        raise ValueError("Only accepted, scored episodes may enter online replay.")
    if result.get("recording", {}).get("status") != "complete":
        raise ValueError("Online training requires a complete LeRobot recording.")
    if not result["critical_phases"]:
        return None
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch

    root = directory / "dataset"
    binding = {
        "format": "piper-online-segments-v1",
        "stage1": policy.manifest["checkpoint_id"],
        "norm": policy.manifest["norm_stats_sha256"],
        "task": task,
        "phases": result["critical_phases"],
        "terminal": result["recording"]["terminal_frame"],
        "chunk": policy.actor_horizon,
        "stride": policy.actor_horizon if stride is None else stride,
        "tail_policy": tail_policy,
        "feature_batch_size": feature_batch_size,
        "anchors": {
            path.name: file_sha256(path)
            for path in sorted((directory / "replay_anchors").glob("*.npz"))
        },
        "sources": {
            str(path.relative_to(root)): file_sha256(path)
            for name in ("meta", "data", "videos")
            for path in sorted((root / name).rglob("*"))
            if path.is_file()
        },
    }
    shard = directory / "online_features.npz"
    manifest = directory / "online_features.json"
    fingerprint = json_fingerprint(binding)
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != fingerprint or saved["sha256"] != file_sha256(shard):
            raise ValueError("Accepted episode or feature shard changed.")
    else:
        dataset = LeRobotDataset(
            root.name, root=root, download_videos=False, video_backend="torchcodec"
        )
        if float(dataset.fps) != policy.fps:
            raise ValueError("Recorded FPS differs from the Stage 1 contract.")
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        rows = dataset.hf_dataset.with_format(None)
        terminal = binding["terminal"]
        if len(rows) != terminal + 1:
            raise ValueError("Episode recording must include one terminal observation.")
        valid = (
            np.asarray(rows["complementary_info.action_valid"]).reshape(-1).astype(bool)
        )
        for phase in result["critical_phases"]:
            if not valid[phase["start"] : phase["end"]].all():
                raise ValueError("Critical segment contains an unexecuted action.")
        transitions = episode_transitions(
            policy.codec.encode(np.asarray(rows["action"], dtype=np.float32)),
            phases=result["critical_phases"],
            terminal=terminal,
            chunk_length=policy.actor_horizon,
            stride=binding["stride"],
            tail_policy=tail_policy,
            success=result["outcome"] == "success",
            human=np.asarray(rows["complementary_info.is_intervention"])
            .reshape(-1)
            .astype(bool),
            reward_scope="critical_segment",
        )

        features, feature_stats = backfill_replay_features(
            policy,
            transitions["frames"],
            dataset.__getitem__,
            directory=directory,
            task=task,
            fingerprint=fingerprint,
            batch_size=feature_batch_size,
        )
        digest = cache_episode(shard, transitions, features.__getitem__)
        write_json(
            manifest,
            {
                "binding": binding,
                "fingerprint": fingerprint,
                "sha256": digest,
                "feature_stats": feature_stats,
            },
        )
    with np.load(shard, allow_pickle=False) as saved:
        arrays = {key: torch.from_numpy(saved[key].copy()) for key in saved.files}
    return segment_trajectory(arrays, int(result["policy_version"]))
