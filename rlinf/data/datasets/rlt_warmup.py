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

"""Episode annotations and frozen Piper features for offline RLT training."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.schema.critical_phase import (
    SEGMENT_LABEL_FORMAT,
    SEGMENT_SCORING_PROTOCOL,
    critical_phase_intervals,
    scored_critical_phase_intervals,
)
from rlinf.utils.logging import get_logger

logger = get_logger()
LABEL_FORMAT = "piper-rlt-episodes-v2"
CACHE_FORMAT = "piper-rlt-cache-v2"
SEGMENT_CACHE_FORMAT = "piper-rlt-cache-v3"


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    """Identify a JSON-compatible configuration or data manifest."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def implementation_fingerprints(paths: list[str]) -> dict[str, str]:
    """Bind repository files and Python source trees used by cache or training."""
    root = Path(__file__).resolve().parents[3]
    result = {}
    for name in paths:
        path = root / name
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for source in files:
            result[str(source.relative_to(root))] = file_sha256(source)
    return result


def runtime_versions() -> dict[str, str | None]:
    """Record installed decoder and numerical dependency versions without imports."""
    result = {}
    for package in (
        "torch",
        "torchvision",
        "torchcodec",
        "lerobot",
        "av",
        "numpy",
        "rlinf-openpi",
        "openpi",
        "transformers",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def runtime_source_fingerprints() -> dict[str, str]:
    """Bind external preprocessing code even for editable or dual installations."""
    result = {}
    for package, names in (
        ("openpi", ("transforms.py", "models/tokenizer.py", "shared/normalize.py")),
        (
            "lerobot",
            (
                "datasets/video_utils.py",
                "datasets/lerobot_dataset.py",
                "datasets/utils.py",
            ),
        ),
    ):
        spec = find_spec(package)
        if spec is None or spec.origin is None:
            raise ImportError(f"Cannot fingerprint {package} preprocessing.")
        root = Path(spec.origin).parent
        for name in names:
            result[f"{package}/{name}"] = file_sha256(root / name)
    return result


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a finite JSON document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


@contextmanager
def directory_lock(directory: Path):
    """Allow one writer per cache or run; release the lock after a crash."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer owns {directory}.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _reject_local_outcomes(manifest: dict) -> None:
    """Keep segment labels out of the legacy global-episode reward pipeline."""
    if (
        manifest.get("format") == "piper-rlt-episodes-v3"
        or manifest.get("reward_scope") == "critical_segment"
        or any(
            "outcome" in phase
            for record in manifest.get("episodes", [])
            for phase in record.get("critical_phases", [])
        )
    ):
        raise ValueError(
            "Local critical-segment outcomes require a segment-terminal trainer. "
            "This legacy warmup uses global episode rewards; refusing local labels."
        )


def annotation_reward_scope(manifest: dict) -> str:
    """Require an explicit versioned contract before using segment outcomes."""
    if manifest.get("format") == SEGMENT_LABEL_FORMAT:
        if (
            manifest.get("reward_scope") != "critical_segment"
            or manifest.get("scoring_protocol") != SEGMENT_SCORING_PROTOCOL
        ):
            raise ValueError(
                "Local critical-segment outcomes require the v3 scoring contract."
            )
        return "critical_segment"
    _reject_local_outcomes(manifest)
    return "global_episode"


def _read_annotations(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    annotation_reward_scope(manifest)
    if manifest.get("format") == "piper-rlt-episodes-v1":
        # The old schema explicitly defined one interval ending at the episode.
        # Preserve that historical meaning; v2 never fills missing end markers.
        for record in manifest["episodes"]:
            start = record.pop("critical_start")
            record["critical_phases"] = (
                []
                if start is None
                else [{"start": start, "end": record["terminal_frame"]}]
            )
        manifest["format"] = LABEL_FORMAT
    if manifest.get("format") not in (LABEL_FORMAT, SEGMENT_LABEL_FORMAT):
        raise ValueError(
            f"Expected annotation format {LABEL_FORMAT} or {SEGMENT_LABEL_FORMAT}."
        )
    return manifest


def load_episode_annotations(path: str | Path) -> dict:
    """Read explicit phase boundaries and episode-level train/validation splits.

    Each ``critical_phases`` entry is an explicit [start,end) action interval.
    ``terminal_frame`` independently identifies the episode's final observation.
    Dataset paths are relative to the annotation file. A session must belong
    to a single split; repeated physical episodes are rejected across aliases.
    """
    path = Path(path).resolve()
    manifest = _read_annotations(path)
    for field in ("task", "phase_definition"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Annotations require a nonempty {field}.")
    if not isinstance(manifest.get("test_only", False), bool):
        raise ValueError("test_only must be a boolean.")
    datasets = manifest.get("datasets", {})
    if not datasets or not manifest.get("episodes"):
        raise ValueError("Annotations require datasets and episodes.")
    manifest["datasets"] = {
        name: str((path.parent / Path(root).expanduser()).resolve())
        for name, root in datasets.items()
    }
    seen, sessions = set(), {}
    for record in manifest["episodes"]:
        dataset = record.get("dataset")
        if dataset not in datasets:
            raise ValueError(f"Unknown dataset {dataset!r}.")
        for field in ("episode", "terminal_frame"):
            if type(record.get(field)) is not int or record[field] < 0:
                raise ValueError(f"Episode annotation needs an integer {field}.")
        validator = (
            scored_critical_phase_intervals
            if annotation_reward_scope(manifest) == "critical_segment"
            else critical_phase_intervals
        )
        validator(record.get("critical_phases"), record["terminal_frame"])
        for field, choices in (
            ("split", ("train", "validation")),
            ("outcome", ("success", "failure")),
            ("behavior", ("human", "reference", "actor")),
        ):
            if record.get(field) not in choices:
                raise ValueError(f"Episode {field} must be one of {choices}.")
        key = (manifest["datasets"][dataset], record["episode"])
        if key in seen:
            raise ValueError(f"Duplicate episode across dataset aliases: {key}.")
        seen.add(key)
        session = record.get("session")
        if session is not None:
            if not isinstance(session, str) or not session.strip():
                raise ValueError("Session labels must be nonempty strings or null.")
            if session in sessions and sessions[session] != record["split"]:
                raise ValueError(f"Session {session!r} crosses train/validation.")
            sessions[session] = record["split"]
    return manifest


def create_annotation_template(
    dataset: str | Path, output: str | Path, *, seed: int = 1234
) -> Path:
    """Draft metadata and a stratified split; leave phase/behavior labels empty."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from rlinf.data.storage.lerobot.metadata import (
        load_episode_outcomes,
        load_task_descriptions,
    )

    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing annotations: {output}")
    meta = LeRobotDatasetMetadata(dataset.name, root=dataset)
    tasks = load_task_descriptions(dataset)
    outcomes = load_episode_outcomes(dataset)
    rng = np.random.default_rng(seed)
    validation = set()
    for success in (False, True):
        ids = sorted(ep for ep, result in outcomes.items() if result == success)
        rng.shuffle(ids)
        count = min(max(10, round(len(ids) * 0.2)), max(1, len(ids) - 20))
        validation.update(ids[:count])
    payload = {
        "format": LABEL_FORMAT,
        "task": next(iter(tasks.values())) if len(tasks) == 1 else "",
        "phase_definition": "",
        "test_only": False,
        "datasets": {"piper": str(dataset)},
        "episodes": [
            {
                "dataset": "piper",
                "episode": ep,
                "critical_phases": [],
                "terminal_frame": None,
                "outcome": ("success" if outcomes[ep] else "failure")
                if ep in outcomes
                else None,
                "behavior": None,
                "split": "validation" if ep in validation else "train",
                "session": None,
            }
            for ep in range(meta.total_episodes)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return output


def split_collection_annotations(
    sources: list[str | Path],
    output: str | Path,
    *,
    seed: int = 1234,
    skip_unlabeled: bool = False,
) -> Path:
    """Merge recorded trials and fix a reproducible episode/session split.

    Sessions, when supplied by the operator, stay together. Without session
    groups the split is stratified by episode outcome. Missing phase labels
    require correction or explicit exclusion before this command succeeds.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing annotations: {output}")
    merged = None
    for source in sources:
        source = Path(source).resolve()
        value = _read_annotations(source)
        if merged is None:
            merged = {**value, "datasets": {}, "episodes": [], "excluded_episodes": []}
        for key in (
            "format",
            "reward_scope",
            "scoring_protocol",
            "task",
            "phase_definition",
            "test_only",
        ):
            if value.get(key, False) != merged.get(key, False):
                raise ValueError(f"Collection {key} differs across recordings.")
        for record in value["episodes"]:
            try:
                validator = (
                    scored_critical_phase_intervals
                    if annotation_reward_scope(value) == "critical_segment"
                    else critical_phase_intervals
                )
                validator(record.get("critical_phases"), record["terminal_frame"])
            except ValueError as exc:
                if not skip_unlabeled:
                    raise ValueError(
                        f"Invalid critical phases in {source}: {exc}; correct them or use --skip-unlabeled."
                    ) from exc
                merged["excluded_episodes"].append({"source": str(source), **record})
                continue
            name = f"source_{len(merged['episodes']):04d}"
            merged["datasets"][name] = str(
                (source.parent / value["datasets"][record["dataset"]]).resolve()
            )
            merged["episodes"].append({**record, "dataset": name, "split": "train"})
    if not merged or len(merged["episodes"]) < 2:
        raise ValueError("At least two labeled episodes are required for splitting.")
    rng = np.random.default_rng(seed)
    groups = {}
    for index, record in enumerate(merged["episodes"]):
        group = (
            ("session", record["session"])
            if record.get("session")
            else ("episode", index)
        )
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise ValueError(
            "One session cannot span train/validation; collect an independent session."
        )
    strata = {}
    for group, indices in groups.items():
        outcome = tuple(sorted({merged["episodes"][i]["outcome"] for i in indices}))
        strata.setdefault(outcome, []).append(group)
    for members in strata.values():
        rng.shuffle(members)
        count = (
            min(max(10, round(len(members) * 0.2)), max(1, len(members) - 20))
            if len(members) > 1
            else 0
        )
        for group in members[:count]:
            for index in groups[group]:
                merged["episodes"][index]["split"] = "validation"
    if not any(r["split"] == "validation" for r in merged["episodes"]):
        raise ValueError(
            "Too few independent groups for a stratified validation split."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, merged)
    try:
        load_episode_annotations(output)
    except Exception:
        output.unlink()
        raise
    return output


def episode_transitions(
    actions: np.ndarray,
    *,
    phases: list[dict],
    terminal: int,
    chunk_length: int,
    stride: int,
    success: bool,
    human: np.ndarray,
    reward_scope: str = "global_episode",
) -> dict[str, np.ndarray]:
    """Build phase-local actions and discounted transitions between actor decisions.

    At a phase exit the frozen reference continues to the next critical entry
    or the episode terminal. These observed gaps contribute discount steps,
    but never actor actions. Only the actual episode outcome supplies reward.
    ``action_end`` retains the immediate post-action observation separately
    from ``next``, the next actor decision or terminal observation.
    """
    if not 0 < stride <= chunk_length:
        raise ValueError("Use 0 < stride <= chunk_length to cover terminal rewards.")
    if type(terminal) is not int or not 0 < terminal < len(actions):
        raise ValueError("Phase bounds must leave a recorded terminal observation.")
    if reward_scope == "critical_segment":
        intervals = scored_critical_phase_intervals(phases, terminal)
    elif reward_scope == "global_episode":
        _reject_local_outcomes({"episodes": [{"critical_phases": phases}]})
        intervals = critical_phase_intervals(phases, terminal)
    else:
        raise ValueError("Unknown reward_scope.")
    if actions.shape[1:] != (7,) or human.shape != (len(actions),):
        raise ValueError("Expected actions [frames,7] and human flags [frames].")
    if not np.isfinite(actions).all():
        raise ValueError("Recorded actions contain non-finite values.")
    phase_ids, start_rows, length_rows, next_rows, next_lengths = [], [], [], [], []
    for phase_id, (start, end) in enumerate(intervals):
        for first in range(start, end, stride):
            length = min(chunk_length, end - first)
            last = first + length
            next_frame, next_length = last, min(chunk_length, end - last)
            if last == end:
                if reward_scope == "critical_segment":
                    next_frame, next_length = end, 0
                elif phase_id + 1 < len(intervals):
                    next_frame, next_end = intervals[phase_id + 1]
                    next_length = min(chunk_length, next_end - next_frame)
                else:
                    next_frame, next_length = terminal, 0
            phase_ids.append(phase_id)
            start_rows.append(first)
            length_rows.append(length)
            next_rows.append(next_frame)
            next_lengths.append(next_length)
    starts = np.asarray(start_rows, dtype=np.int64)
    lengths = np.asarray(length_rows, dtype=np.int64)
    ends = starts + lengths
    next_frames = np.asarray(next_rows, dtype=np.int64)
    states = np.unique(np.concatenate([starts, ends, next_frames]))
    count = len(starts)
    chunk_actions = np.zeros((count, chunk_length, 7), dtype=np.float32)
    terminated = next_frames == terminal
    rewards = (terminated & success).astype(np.float32)
    if reward_scope == "critical_segment":
        terminated = ends == np.asarray(intervals)[phase_ids, 1]
        successes = np.asarray([p["outcome"] == "success" for p in phases])
        rewards = (terminated & successes[phase_ids]).astype(np.float32)
    intervention = np.zeros((count, chunk_length), dtype=np.bool_)
    for row, (first, length) in enumerate(zip(starts, lengths, strict=True)):
        chunk_actions[row, :length] = actions[first : first + length]
        intervention[row, :length] = human[first : first + length]
    return {
        "frames": states,
        "current": np.searchsorted(states, starts),
        "action_end": np.searchsorted(states, ends),
        "next": np.searchsorted(states, next_frames),
        "phase_id": np.asarray(phase_ids, dtype=np.int64),
        "actions": chunk_actions,
        "rewards": rewards,
        "valid_steps": lengths,
        "discount_steps": next_frames - starts,
        "next_valid_steps": np.asarray(next_lengths, dtype=np.int64),
        "terminated": terminated,
        "human": intervention,
    }


def cache_episode(
    output: Path,
    transitions: dict[str, np.ndarray],
    extract: Callable[[int], dict[str, np.ndarray]],
) -> str:
    """Extract each required state once and atomically save an episode shard."""
    features: dict[str, list[np.ndarray]] = {
        "z_rl": [],
        "proprio": [],
        "ref_chunk": [],
        "raw_state": [],
    }
    for frame in transitions["frames"]:
        values = extract(int(frame))
        for key in features:
            value = np.asarray(values[key], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {key} at frame {frame}.")
            features[key].append(value)
    arrays = {key: np.stack(values) for key, values in features.items()}
    if (
        arrays["z_rl"].shape[1:] != (2048,)
        or arrays["proprio"].shape[1:] != (7,)
        or arrays["raw_state"].shape[1:] != (7,)
    ):
        raise ValueError("Piper RLT cache requires z_rl[2048] and proprio[7].")
    if arrays["ref_chunk"].ndim != 3 or arrays["ref_chunk"].shape[-1] != 7:
        raise ValueError("Reference actions must have shape [states,horizon,7].")
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **transitions, **arrays)
    temporary.replace(output)
    return file_sha256(output)


def _source_fingerprints(datasets: dict[str, str]) -> dict:
    result = {}
    for name, root in datasets.items():
        files = {}
        for directory in ("meta", "data", "videos"):
            for path in sorted((Path(root) / directory).rglob("*")):
                if path.is_file():
                    files[str(path.relative_to(root))] = file_sha256(path)
        if not files:
            raise ValueError(f"No LeRobot source files in {root}.")
        result[name] = json_fingerprint(files)
    return result


def prepare_piper_cache(
    stage1: str | Path,
    annotations: str | Path,
    output: str | Path,
    *,
    chunk_length: int = 10,
    stride: int = 10,
    seed: int = 1234,
    device: str = "cuda:0",
    video_backend: str = "torchcodec",
) -> dict:
    """Cache annotated LeRobot episodes; resume only an identical data contract."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot import episode_boundaries
    from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch
    from rlinf.data.storage.lerobot.metadata import (
        load_episode_outcomes,
        load_task_descriptions,
    )
    from rlinf.models.embodiment.piper_inference import (
        PiperRLTPolicy,
        load_piper_inference_config,
        piper_env_observation,
    )

    labels = load_episode_annotations(annotations)
    if not 0 < stride <= chunk_length:
        raise ValueError("Use 0 < stride <= chunk_length.")
    cfg, policy_manifest = load_piper_inference_config(stage1)
    if cfg.model_type != "openpi" or not cfg.openpi.get("use_rlt", False):
        raise ValueError("Warmup requires an OpenPI RLT Stage 1 checkpoint.")
    horizon = int(cfg.openpi.action_chunk)
    if chunk_length > horizon:
        raise ValueError("Actor chunk exceeds the Stage 1 reference horizon.")
    datasets, boundaries, outcomes = {}, {}, {}
    for name, root in labels["datasets"].items():
        tasks = load_task_descriptions(root)
        if not tasks or set(tasks.values()) != {labels["task"]}:
            raise ValueError(f"Task text mismatch for {root}; verify source metadata.")
        dataset = LeRobotDataset(
            Path(root).name,
            root=Path(root),
            download_videos=False,
            video_backend=video_backend,
        )
        if float(dataset.fps) != float(policy_manifest["fps"]):
            raise ValueError(f"FPS mismatch for {root}.")
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        datasets[name] = dataset
        boundaries[name] = episode_boundaries(dataset)
        outcomes[name] = load_episode_outcomes(root)
    for record in labels["episodes"]:
        name, ep = record["dataset"], record["episode"]
        starts, ends = boundaries[name]
        if ep >= len(starts) or record["terminal_frame"] >= int(ends[ep] - starts[ep]):
            raise ValueError(f"Episode or terminal frame outside dataset: {record}.")
        if ep in outcomes[name] and outcomes[name][ep] != (
            record["outcome"] == "success"
        ):
            raise ValueError(f"Outcome disagrees with dataset episode {name}/{ep}.")

    output = Path(output).resolve()
    with directory_lock(output):
        binding = {
            "format": SEGMENT_CACHE_FORMAT
            if annotation_reward_scope(labels) == "critical_segment"
            else CACHE_FORMAT,
            "annotations": labels,
            "source_fingerprints": _source_fingerprints(labels["datasets"]),
            "stage1": policy_manifest,
            "stage1_weights_sha256": file_sha256(
                Path(cfg.model_path) / "model_state_dict/full_weights.pt"
            ),
            "chunk_length": chunk_length,
            "stride": stride,
            "seed": seed,
            "preprocessing": f"PiperRLTPolicy.extract_rlt_obs/{video_backend}/float32/paired-phases-v2",
            "runtime_versions": runtime_versions(),
            "runtime_source_sha256": runtime_source_fingerprints(),
            "preprocessing_sha256": implementation_fingerprints(
                [
                    "rlinf/data/datasets/rlt_warmup.py",
                    "rlinf/data/schema/critical_phase.py",
                    "rlinf/data/storage/lerobot",
                    "rlinf/models/embodiment/piper_inference.py",
                    "rlinf/models/embodiment/base_policy.py",
                    "rlinf/models/embodiment/openpi",
                    "rlinf/models/embodiment/modules/rlt_token_transformer.py",
                ]
            ),
        }
        fingerprint = json_fingerprint(binding)
        progress_path = output / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["fingerprint"] != fingerprint:
                raise ValueError("Cache binding changed; choose a new cache directory.")
        else:
            progress = {"fingerprint": fingerprint, "episodes": {}}
            write_json(progress_path, progress)
        (output / "norm_stats.json").write_bytes(
            Path(cfg.openpi_data.norm_stats_path).read_bytes()
        )
        policy = None
        entries = []
        for index, record in enumerate(labels["episodes"]):
            filename = f"episode_{index:06d}.npz"
            shard = output / filename
            previous = progress["episodes"].get(filename)
            if previous and shard.exists() and file_sha256(shard) == previous["sha256"]:
                entries.append(previous)
                continue
            if policy is None:
                policy = PiperRLTPolicy(cfg.model_path, device=device)
            name, episode = record["dataset"], record["episode"]
            dataset = datasets[name]
            first = int(boundaries[name][0][episode])
            end = first + record["terminal_frame"] + 1
            rows = dataset.hf_dataset.with_format(None).select(range(first, end))
            raw_actions = np.asarray(rows["action"], dtype=np.float32)
            intervals = critical_phase_intervals(
                record["critical_phases"], record["terminal_frame"]
            )
            if "complementary_info.action_valid" in rows.column_names:
                valid = np.asarray(rows["complementary_info.action_valid"]).reshape(-1)
                if not valid[intervals[0][0] : record["terminal_frame"]].all():
                    raise ValueError(
                        "Phase includes a terminal placeholder instead of an executed action."
                    )
            human = np.full(len(rows), record["behavior"] == "human", dtype=bool)
            if "complementary_info.is_intervention" in rows.column_names:
                human |= (
                    np.asarray(rows["complementary_info.is_intervention"])
                    .reshape(-1)
                    .astype(bool)
                )
            transitions = episode_transitions(
                policy.codec.encode(raw_actions),
                phases=record["critical_phases"],
                terminal=record["terminal_frame"],
                chunk_length=chunk_length,
                stride=stride,
                success=record["outcome"] == "success",
                human=human,
                reward_scope=annotation_reward_scope(labels),
            )

            def extract(frame: int) -> dict[str, np.ndarray]:
                # Frame-specific seeds make episode-level resume reproducible.
                frame_seed = int(
                    json_fingerprint([seed, name, episode, frame])[:15], 16
                )
                with torch.random.fork_rng(
                    devices=[torch.device(device)]
                    if str(device).startswith("cuda")
                    else []
                ):
                    torch.manual_seed(frame_seed)
                    sample = dataset[first + frame]
                    features = policy.feature_model.extract_rlt_obs(
                        piper_env_observation(sample, labels["task"])
                    )
                return {
                    **{
                        key: value[0].detach().float().cpu().numpy()
                        for key, value in features.items()
                    },
                    "raw_state": np.asarray(
                        sample["observation.state"], dtype=np.float32
                    ),
                }

            digest = cache_episode(shard, transitions, extract)
            entry = {
                **record,
                "file": filename,
                "sha256": digest,
                "transitions": len(transitions["current"]),
                "states": len(transitions["frames"]),
            }
            entries.append(entry)
            progress["episodes"][filename] = entry
            write_json(progress_path, progress)
            logger.info(
                "Cached %s/%s: %s states, %s chunks",
                name,
                episode,
                entry["states"],
                entry["transitions"],
            )
        manifest = {
            **binding,
            "fingerprint": fingerprint,
            "episodes": entries,
            "model": {
                "model_type": "rlt_mlp_policy",
                "precision": "fp32",
                "z_dim": 2048,
                "proprio_dim": 7,
                "action_dim": 7,
                "num_action_chunks": chunk_length,
                "ref_num_action_chunks": horizon,
                "add_q_head": True,
                "q_head_type": "default",
                "fixed_std": 0.002,
            },
        }
        manifest["content_fingerprint"] = json_fingerprint(manifest)
        write_json(output / "manifest.json", manifest)
    return manifest


def relabel_cached_segments(
    source: str | Path,
    annotations: str | Path,
    output: str | Path,
    *,
    stage1: str | Path,
) -> dict:
    """Reuse verified frozen features in a new cache with local terminal targets.

    This CPU operation accepts identical physical episodes, splits and boundaries.
    It hashes original LeRobot sources and Stage 1 weights again. No features,
    actions or human flags are inferred or regenerated; only TD edges change.
    """
    from rlinf.models.embodiment.piper_inference import load_piper_inference_config

    replay = CachedRLTReplay(source)
    if replay.reward_scope != "global_episode":
        raise ValueError("Feature relabeling requires a legacy global cache.")
    labels = load_episode_annotations(annotations)
    if annotation_reward_scope(labels) != "critical_segment":
        raise ValueError("Feature relabeling requires scored local annotations.")
    old = replay.manifest
    if (
        labels["datasets"] != old["annotations"]["datasets"]
        or labels["task"] != old["annotations"]["task"]
    ):
        raise ValueError("Dataset aliases/paths or task changed; extract a new cache.")
    if labels.get("test_only", False) != old["annotations"].get("test_only", False):
        raise ValueError("Cannot change the test-only data designation.")
    if len(labels["episodes"]) != len(old["episodes"]):
        raise ValueError("Parent episodes changed; extract a new cache.")
    for record, entry in zip(labels["episodes"], old["episodes"], strict=True):
        for key in (
            "dataset",
            "episode",
            "terminal_frame",
            "outcome",
            "split",
            "session",
            "behavior",
        ):
            if record.get(key) != entry.get(key):
                raise ValueError(
                    f"Parent episode {key} changed; cannot reuse features."
                )
        if critical_phase_intervals(
            record["critical_phases"], record["terminal_frame"]
        ) != critical_phase_intervals(
            entry["critical_phases"], entry["terminal_frame"]
        ):
            raise ValueError("Phase boundaries changed; cannot reuse feature rows.")
    source_fingerprints = _source_fingerprints(labels["datasets"])
    if source_fingerprints != old["source_fingerprints"]:
        raise ValueError("Source LeRobot data changed since feature extraction.")
    cfg, policy = load_piper_inference_config(stage1)
    if (
        policy != old["stage1"]
        or file_sha256(Path(cfg.model_path) / "model_state_dict/full_weights.pt")
        != old["stage1_weights_sha256"]
    ):
        raise ValueError(
            "Stage 1 identity or weights changed since feature extraction."
        )
    if (
        file_sha256(cfg.openpi_data.norm_stats_path)
        != old["stage1"]["norm_stats_sha256"]
    ):
        raise ValueError("Stage 1 normalization changed.")
    model_code = implementation_fingerprints(
        [
            "rlinf/models/embodiment/piper_inference.py",
            "rlinf/models/embodiment/base_policy.py",
            "rlinf/models/embodiment/openpi",
            "rlinf/models/embodiment/modules/rlt_token_transformer.py",
        ]
    )
    if any(old["preprocessing_sha256"].get(k) != v for k, v in model_code.items()):
        raise ValueError("Frozen feature model implementation changed.")
    if (
        runtime_versions() != old["runtime_versions"]
        or runtime_source_fingerprints() != old["runtime_source_sha256"]
    ):
        raise ValueError("Feature runtime changed; use the recorded environment.")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing cache: {output}")
    with directory_lock(output):
        if any(p.name != ".lock" for p in output.iterdir()):
            raise FileExistsError(f"Preserving existing cache: {output}")
        binding = {
            **{
                k: v
                for k, v in old.items()
                if k not in ("episodes", "fingerprint", "content_fingerprint")
            },
            "format": SEGMENT_CACHE_FORMAT,
            "annotations": labels,
            "feature_reuse": {
                "source": str(replay.path),
                "manifest_sha256": file_sha256(replay.path / "manifest.json"),
                "content_fingerprint": old["content_fingerprint"],
                "annotations_sha256": file_sha256(annotations),
                "verified_model_code": model_code,
                "transition_implementation": implementation_fingerprints(
                    [
                        "rlinf/data/datasets/rlt_warmup.py",
                        "rlinf/data/schema/critical_phase.py",
                    ]
                ),
            },
        }
        (output / "norm_stats.json").write_bytes(
            (replay.path / "norm_stats.json").read_bytes()
        )
        entries = []
        for record, entry, data in zip(
            labels["episodes"], old["episodes"], replay.episodes, strict=True
        ):
            arrays = {k: v.numpy().copy() for k, v in data.items()}
            phases = np.asarray(
                critical_phase_intervals(
                    record["critical_phases"], record["terminal_frame"]
                )
            )
            arrays["next"] = arrays["action_end"].copy()
            last = arrays["frames"][arrays["next"]]
            phase_end = phases[arrays["phase_id"], 1]
            arrays["terminated"] = last == phase_end
            arrays["discount_steps"] = arrays["valid_steps"].copy()
            arrays["next_valid_steps"] = np.minimum(
                old["chunk_length"], phase_end - last
            )
            successful = np.asarray(
                [p["outcome"] == "success" for p in record["critical_phases"]]
            )
            arrays["rewards"] = (
                arrays["terminated"] & successful[arrays["phase_id"]]
            ).astype(np.float32)
            filename = output / entry["file"]
            with filename.open("xb") as stream:
                np.savez(stream, **arrays)
            entries.append({**entry, **record, "sha256": file_sha256(filename)})
        manifest = {
            **binding,
            "fingerprint": json_fingerprint(binding),
            "episodes": entries,
        }
        manifest["content_fingerprint"] = json_fingerprint(manifest)
        write_json(output / "manifest.json", manifest)
        CachedRLTReplay(output)
    return manifest


class CachedRLTReplay:
    """Immutable CPU episode pool, with uniform episode then chunk sampling."""

    def __init__(
        self, path: str | Path, *, seed: int = 1234, phase_filter: int | None = None
    ):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.reward_scope = annotation_reward_scope(
            self.manifest.get("annotations", {})
        )
        expected_format = (
            SEGMENT_CACHE_FORMAT
            if self.reward_scope == "critical_segment"
            else CACHE_FORMAT
        )
        if self.manifest.get("format") != expected_format:
            raise ValueError(
                "Local critical-segment outcomes and cache version disagree."
            )
        if self.reward_scope == "global_episode":
            _reject_local_outcomes(self.manifest)
        if phase_filter is not None and (
            type(phase_filter) is not int
            or phase_filter < 0
            or self.reward_scope != "critical_segment"
        ):
            raise ValueError(
                "phase_filter requires a nonnegative ordinal and local rewards."
            )
        self.phase_filter = phase_filter
        content = dict(self.manifest)
        fingerprint = content.pop("content_fingerprint")
        if json_fingerprint(content) != fingerprint:
            raise ValueError("Cache manifest content fingerprint mismatch.")
        if (
            file_sha256(self.path / "norm_stats.json")
            != self.manifest["stage1"]["norm_stats_sha256"]
        ):
            raise ValueError("Cache normalization fingerprint mismatch.")
        self.rng = np.random.default_rng(seed)
        if len(self.manifest["episodes"]) != len(
            self.manifest["annotations"]["episodes"]
        ):
            raise ValueError("Cache parent episode count disagrees with annotations.")
        self.episodes: list[dict[str, torch.Tensor]] = []
        self.selected_rows: list[np.ndarray] = []
        self.splits: dict[str, list[int]] = {"train": [], "validation": []}
        identities, sessions = set(), {}
        for index, entry in enumerate(self.manifest["episodes"]):
            label = self.manifest["annotations"]["episodes"][index]
            if any(entry.get(key) != value for key, value in label.items()):
                raise ValueError(
                    "Cache shard labels disagree with embedded annotations."
                )
            filename = (self.path / entry["file"]).resolve()
            if (
                not filename.is_relative_to(self.path)
                or file_sha256(filename) != entry["sha256"]
            ):
                raise ValueError(f"Cache shard changed: {entry['file']}")
            identity = (
                self.manifest["annotations"]["datasets"][entry["dataset"]],
                entry["episode"],
            )
            if identity in identities:
                raise ValueError("Duplicate episode in cached replay.")
            identities.add(identity)
            split = entry["split"]
            if split not in self.splits:
                raise ValueError("Unknown cached episode split.")
            session = entry.get("session")
            if session is not None and sessions.get(session, split) != split:
                raise ValueError("Session crosses cached train/validation splits.")
            if session is not None:
                sessions[session] = split
            with np.load(filename, allow_pickle=False) as shard:
                arrays = {
                    key: torch.from_numpy(shard[key].copy()) for key in shard.files
                }
            self._validate_arrays(arrays, entry)
            self.episodes.append(arrays)
            selected = (
                np.arange(len(arrays["current"]))
                if phase_filter is None
                else np.flatnonzero(arrays["phase_id"].numpy() == phase_filter)
            )
            self.selected_rows.append(selected)
            if len(selected):
                self.splits[split].append(index)
        if not all(self.splits.values()):
            raise ValueError("Cache requires disjoint train and validation episodes.")

    def _validate_arrays(self, arrays: dict, entry: dict) -> None:
        count, states = entry["transitions"], entry["states"]
        chunk = self.manifest["model"]["num_action_chunks"]
        horizon = self.manifest["model"]["ref_num_action_chunks"]
        shapes = {
            "frames": (states,),
            "current": (count,),
            "action_end": (count,),
            "next": (count,),
            "phase_id": (count,),
            "actions": (count, chunk, 7),
            "rewards": (count,),
            "valid_steps": (count,),
            "discount_steps": (count,),
            "next_valid_steps": (count,),
            "terminated": (count,),
            "human": (count, chunk),
            "z_rl": (states, 2048),
            "proprio": (states, 7),
            "ref_chunk": (states, horizon, 7),
            "raw_state": (states, 7),
        }
        if count < 1 or set(arrays) != set(shapes):
            raise ValueError("Invalid cached episode fields or empty phase.")
        for key, shape in shapes.items():
            if (
                tuple(arrays[key].shape) != shape
                or not torch.isfinite(arrays[key]).all()
            ):
                raise ValueError(f"Invalid cached {key} in {entry['file']}.")
        for key in ("current", "action_end", "next"):
            if (
                arrays[key].dtype != torch.int64
                or not ((arrays[key] >= 0) & (arrays[key] < states)).all()
            ):
                raise ValueError("Invalid cached state indices.")
        lengths = arrays["valid_steps"]
        if (
            lengths.dtype != torch.int64
            or not ((lengths > 0) & (lengths <= chunk)).all()
        ):
            raise ValueError("Invalid cached chunk lengths.")
        intervals = critical_phase_intervals(
            entry["critical_phases"], entry["terminal_frame"]
        )
        phase_ids = arrays["phase_id"]
        if (
            phase_ids.dtype != torch.int64
            or not ((phase_ids >= 0) & (phase_ids < len(intervals))).all()
        ):
            raise ValueError("Invalid cached phase indices.")
        phases = torch.tensor(intervals, dtype=torch.int64)
        if arrays["frames"].dtype != torch.int64 or not (
            (arrays["frames"] >= 0).all()
            and (arrays["frames"] <= entry["terminal_frame"]).all()
            and (arrays["frames"][1:] > arrays["frames"][:-1]).all()
        ):
            raise ValueError(
                "Cached feature frames must be ordered episode observations."
            )
        first = arrays["frames"][arrays["current"]]
        last = arrays["frames"][arrays["action_end"]]
        following = arrays["frames"][arrays["next"]]
        if not torch.equal(last - first, lengths):
            raise ValueError("Cached action end is not aligned with executed actions.")
        if not torch.equal(lengths, (phases[phase_ids, 1] - first).clamp(max=chunk)):
            raise ValueError("Cached action window length disagrees with its phase.")
        if not ((first >= phases[phase_ids, 0]) & (last <= phases[phase_ids, 1])).all():
            raise ValueError("Cached action window crosses phase boundaries.")
        next_phases = torch.cat(
            (
                phases[1:],
                torch.tensor([[entry["terminal_frame"], entry["terminal_frame"]]]),
            )
        )
        at_exit = last == phases[phase_ids, 1]
        expected_next = torch.where(at_exit, next_phases[phase_ids, 0], last)
        next_lengths = torch.where(
            at_exit,
            next_phases[phase_ids, 1] - expected_next,
            phases[phase_ids, 1] - expected_next,
        ).clamp(max=chunk)
        if self.reward_scope == "critical_segment":
            scored_critical_phase_intervals(
                entry["critical_phases"], entry["terminal_frame"]
            )
            expected_next = last
            next_lengths = (phases[phase_ids, 1] - last).clamp(max=chunk)
        expected_starts = torch.tensor(
            [
                first
                for start, end in intervals
                for first in range(start, end, self.manifest["stride"])
            ],
            dtype=torch.int64,
        )
        if not torch.equal(first, expected_starts):
            raise ValueError(
                "Cached windows omit or duplicate annotated decision starts."
            )
        if not torch.equal(following, expected_next):
            raise ValueError(
                "Cached next decision is not the next critical state or episode terminal."
            )
        if arrays["discount_steps"].dtype != torch.int64 or not torch.equal(
            arrays["discount_steps"], following - first
        ):
            raise ValueError("Cached discount omits actual reference-controlled ticks.")
        if arrays["next_valid_steps"].dtype != torch.int64 or not torch.equal(
            arrays["next_valid_steps"], next_lengths
        ):
            raise ValueError("Cached next action window crosses phase boundaries.")
        if (
            arrays["terminated"].dtype != torch.bool
            or arrays["human"].dtype != torch.bool
        ):
            raise ValueError("Cached outcome and intervention flags must be boolean.")
        if not arrays["terminated"].any():
            raise ValueError("Cached phase is missing its terminal transition.")
        expected_done = (
            at_exit
            if self.reward_scope == "critical_segment"
            else following == entry["terminal_frame"]
        )
        if not torch.equal(arrays["terminated"], expected_done):
            raise ValueError("Cached terminal flags disagree with annotated endpoint.")
        mask = torch.arange(chunk)[None, :] < lengths[:, None]
        if arrays["actions"][~mask].count_nonzero() or arrays["human"][~mask].any():
            raise ValueError("Cached padding must contain no actions or interventions.")
        rewards = torch.zeros_like(arrays["rewards"])
        if self.reward_scope == "critical_segment":
            successes = torch.tensor(
                [p["outcome"] == "success" for p in entry["critical_phases"]]
            )
            rewards[arrays["terminated"] & successes[phase_ids]] = 1
        elif entry["outcome"] == "success":
            rewards[arrays["terminated"]] = 1
        if not torch.equal(rewards, arrays["rewards"]):
            raise ValueError("Cached rewards disagree with the terminal outcome.")

    def counts(self, split: str) -> dict[str, int]:
        """Return independent episode outcome counts and transition count."""
        entries = [self.manifest["episodes"][i] for i in self.splits[split]]
        counts = Counter(entry["outcome"] for entry in entries)
        return {
            "success": counts["success"],
            "failure": counts["failure"],
            "episodes": len(entries),
            "transitions": sum(len(self.selected_rows[i]) for i in self.splits[split]),
        }

    def metric_groups(self, split: str) -> list[tuple[int, int | None]]:
        """Keep parent splits while scoring local results by phase ordinal."""
        if self.reward_scope != "critical_segment":
            return [(i, None) for i in self.splits[split]]
        return [
            (i, int(phase))
            for i in self.splits[split]
            for phase in torch.unique(
                self.episodes[i]["phase_id"][self.selected_rows[i]]
            ).tolist()
        ]

    def rng_state(self) -> dict:
        """Return a checkpointable sampler RNG state."""
        return copy.deepcopy(self.rng.bit_generator.state)

    def restore_rng(self, state: dict) -> None:
        """Restore the exact next episode/chunk draw."""
        self.rng.bit_generator.state = copy.deepcopy(state)

    def batch(
        self, episode_ids: list[int], rows: list[int], device: str | torch.device
    ) -> dict:
        """Gather transition rows and their shared current/next feature states."""
        samples = []
        for episode, row in zip(episode_ids, rows, strict=True):
            data = self.episodes[episode]
            sample = {
                key: data[key][row]
                for key in (
                    "actions",
                    "rewards",
                    "valid_steps",
                    "discount_steps",
                    "next_valid_steps",
                    "terminated",
                    "human",
                )
            }
            sample["raw_state"] = data["raw_state"][data["current"][row]]
            for prefix, column in (("curr_obs", "current"), ("next_obs", "next")):
                sample[prefix] = {
                    key: data[key][data[column][row]]
                    for key in ("z_rl", "proprio", "ref_chunk")
                }
            samples.append(sample)

        def stack(values):
            if isinstance(values[0], dict):
                return {key: stack([v[key] for v in values]) for key in values[0]}
            return torch.stack(values).to(device)

        return stack(samples)

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> dict:
        """Draw only training episodes, giving each episode equal probability."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        episodes = self.rng.choice(self.splits["train"], size=batch_size).tolist()
        rows = [int(self.rng.choice(self.selected_rows[i])) for i in episodes]
        return self.batch(episodes, rows, device)
