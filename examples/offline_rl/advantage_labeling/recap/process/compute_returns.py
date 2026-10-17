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

"""
Compute returns for LeRobot datasets.

Writes `return`, `reward`, and `prompt` as a sidecar parquet at
``meta/returns_{tag}.parquet``. Updates meta/stats.json and meta/info.json.
Does not modify original per-episode parquet files.

Return computation:
- reward=-1 per step; last step=0 (success) or failure_reward (failure)
- Returns via backward iteration: G_t = r_t + gamma * G_{t+1}

Usage:
    python compute_returns.py --config-name compute_returns
"""

import json
import logging
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Make the rlinf package importable regardless of the cwd the user launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from rlinf.data.storage.lerobot.metadata import (
    load_episode_outcomes,
    load_task_descriptions,
    parse_success,
)

logger = logging.getLogger(__name__)

# Columns needed for computation (tiny — no images)
_READ_COLUMNS = ["episode_index", "frame_index", "is_success", "task_index", "task"]


def compute_returns_for_episode(
    episode_length: int,
    is_success: bool,
    gamma: float,
    failure_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute returns and rewards for a single episode.

    Uses the DynamicReturn approach:
    - Each step reward = -1
    - Success at episode end: last step reward = 0
    - Failure at episode end: last step reward = failure_reward

    Returns are computed via backward iteration: G_t = r_t + gamma * G_{t+1}

    Args:
        episode_length: Total length of episode
        is_success: Whether episode was successful
        gamma: Discount factor (default 1.0 for undiscounted)
        failure_reward: Penalty for failure (must be specified in config)

    Returns:
        Tuple of (returns array, rewards array) for all steps
    """
    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if is_success else failure_reward

    returns = np.zeros(episode_length, dtype=np.float32)
    returns[-1] = rewards[-1]
    for t in range(episode_length - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns, rewards


def get_episode_boundaries(episode_indices: np.ndarray) -> list[tuple[int, int, int]]:
    """Extract episode boundaries from episode_index array.

    Args:
        episode_indices: 1-D numpy array of episode indices.

    Returns:
        List of (episode_index, start_idx, end_idx) tuples
    """
    if len(episode_indices) == 0:
        return []

    change_mask = np.diff(episode_indices) != 0
    change_positions = np.where(change_mask)[0] + 1  # +1: index of new ep

    starts = np.concatenate([[0], change_positions])
    ends = np.concatenate([change_positions, [len(episode_indices)]])
    ep_ids = episode_indices[starts]

    return list(zip(ep_ids.tolist(), starts.tolist(), ends.tolist()))


def _read_metadata_columns(pq_file: str) -> pa.Table:
    """Read only the scalar columns needed for return computation."""
    pf = pq.ParquetFile(pq_file)
    available = set(pf.schema_arrow.names)
    return pf.read(columns=[c for c in _READ_COLUMNS if c in available])


def compute_returns_table(
    table: pa.Table,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    tasks: dict[int, str],
    episode_outcomes: dict[int, bool] | None = None,
) -> pa.Table:
    """Compute complete episodes, including those split across v3 files.

    Explicit outcome labels take precedence over the success-only SFT default.
    Rollout episodes must have either terminal is_success or episode metadata.
    """
    if dataset_type not in ("sft", "rollout"):
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1].")
    if not np.isfinite(failure_reward):
        raise ValueError("failure_reward must be finite.")
    table = table.sort_by(
        [("episode_index", "ascending"), ("frame_index", "ascending")]
    )
    ep_indices = table["episode_index"].to_numpy().astype(np.int64)
    frame_indices = table["frame_index"].to_numpy().astype(np.int64)
    returns_arr = np.empty(len(table), dtype=np.float32)
    rewards_arr = np.empty(len(table), dtype=np.float32)
    outcomes = episode_outcomes or {}
    success_values = (
        table["is_success"].to_pylist() if "is_success" in table.column_names else None
    )
    for episode, start, end in get_episode_boundaries(ep_indices):
        if not np.array_equal(frame_indices[start:end], np.arange(end - start)):
            raise ValueError(
                f"Episode {episode} has missing, duplicate or non-zero-start frame indices."
            )
        terminal = success_values[end - 1] if success_values is not None else None
        if terminal is not None:
            success = parse_success(terminal)
            if episode in outcomes and outcomes[episode] != success:
                raise ValueError(f"Conflicting success labels for episode {episode}.")
        elif episode in outcomes:
            success = outcomes[episode]
        elif dataset_type == "sft":
            success = True
        else:
            raise ValueError(
                f"Episode {episode} needs is_success or metadata episode_success."
            )
        returns_arr[start:end], rewards_arr[start:end] = compute_returns_for_episode(
            end - start, success, gamma, failure_reward
        )

    if "task" in table.column_names:
        prompts = [str(value) for value in table["task"].to_pylist()]
    elif "task_index" in table.column_names:
        prompts = [
            tasks.get(int(index), "perform the task")
            for index in table["task_index"].to_pylist()
        ]
    else:
        prompts = ["perform the task"] * len(table)
    return pa.table(
        {
            "episode_index": ep_indices,
            "frame_index": frame_indices,
            "return": returns_arr,
            "reward": rewards_arr,
            "prompt": prompts,
        }
    )


def process_dataset(
    dataset_path: Path,
    output_path: Path | None,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    num_workers: int = 8,
    tag: str | None = None,
) -> dict:
    """Process a LeRobot dataset and compute return/reward/prompt.

    Writes results as a sidecar ``meta/returns_{tag}.parquet`` (or
    ``meta/returns.parquet`` when *tag* is None) without modifying the
    original per-episode parquet files (no image I/O).

    Args:
        dataset_path: Path to input dataset
        output_path: Path to output dataset (or None to modify in-place)
        dataset_type: "sft" or "rollout"
        gamma: Discount factor
        failure_reward: Penalty for failed episodes
        num_workers: Number of parallel workers for parquet processing
        tag: Optional tag for the sidecar filename

    Returns:
        Statistics dict with return/reward stats
    """
    logger.info(f"Processing dataset: {dataset_path}")
    logger.info(
        f"  Type: {dataset_type}, Gamma: {gamma}, Failure reward: {failure_reward}"
    )

    if output_path is None:
        output_path = dataset_path
    else:
        if output_path.exists():
            logger.warning(f"Removing existing output: {output_path}")
            shutil.rmtree(output_path)
        shutil.copytree(dataset_path, output_path)
        logger.info(f"Copied dataset to: {output_path}")

    data_dir = output_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(str(p) for p in data_dir.rglob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    tasks = load_task_descriptions(output_path)
    outcomes = load_episode_outcomes(output_path)
    if not parquet_files:
        raise ValueError(f"No parquet data found: {data_dir}")
    # Read tiny metadata columns concurrently, then group globally: v3 file
    # boundaries need not coincide with episode boundaries.
    with ThreadPoolExecutor(
        max_workers=max(1, min(num_workers, len(parquet_files)))
    ) as pool:
        tables = list(
            tqdm(
                pool.map(_read_metadata_columns, parquet_files),
                total=len(parquet_files),
            )
        )
    table = pa.concat_tables(tables, promote_options="default")
    if len(table) == 0:
        raise ValueError(f"No data found in dataset: {output_path}")
    combined = compute_returns_table(
        table, dataset_type, gamma, failure_reward, tasks, outcomes
    )
    sidecar_filename = f"returns_{tag}.parquet" if tag else "returns.parquet"
    sidecar_path = output_path / "meta" / sidecar_filename
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, str(sidecar_path))
    logger.info(
        f"Wrote sidecar: {sidecar_path} "
        f"({combined.num_rows} rows, {sidecar_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    returns_arr = combined.column("return").to_numpy()
    rewards_arr = combined.column("reward").to_numpy()

    count = len(returns_arr)
    ret_mean = float(returns_arr.mean())
    ret_var = float(returns_arr.astype(np.float64).var())
    rew_mean = float(rewards_arr.mean())
    rew_var = float(rewards_arr.astype(np.float64).var())

    stats = {
        "return": {
            "mean": ret_mean,
            "std": math.sqrt(max(ret_var, 0.0)),
            "min": float(returns_arr.min()),
            "max": float(returns_arr.max()),
        },
        "reward": {
            "mean": rew_mean,
            "std": math.sqrt(max(rew_var, 0.0)),
            "min": float(rewards_arr.min()),
            "max": float(rewards_arr.max()),
        },
    }

    logger.info("\nStatistics:")
    logger.info(
        f"  Return: mean={stats['return']['mean']:.4f}, std={stats['return']['std']:.4f}, "
        f"min={stats['return']['min']:.4f}, max={stats['return']['max']:.4f}"
    )
    logger.info(
        f"  Reward: mean={stats['reward']['mean']:.4f}, std={stats['reward']['std']:.4f}, "
        f"min={stats['reward']['min']:.4f}, max={stats['reward']['max']:.4f}"
    )
    logger.info(f"  Total rows: {count}")

    stats_path = output_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            existing_stats = json.load(f)
    else:
        existing_stats = {}

    existing_stats["return"] = stats["return"]
    existing_stats["reward"] = stats["reward"]

    with open(stats_path, "w") as f:
        json.dump(existing_stats, f, indent=2)
    logger.info("Updated stats.json")

    info_path = output_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

        info["features"]["return"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        info["features"]["reward"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        info["features"]["prompt"] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info("Updated info.json with new features")

    return stats


def compute_returns(cfg: DictConfig) -> None:
    """Main entry point for return computation."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting return computation...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    data_root = cfg.data.get("data_root", None)
    default_type = cfg.data.get("dataset_type", "sft")
    default_gamma = cfg.data.get("gamma", 1.0)
    default_failure_reward = cfg.data.get("failure_reward", None)
    if default_failure_reward is None:
        raise ValueError(
            "data.failure_reward must be specified. "
            "This is the reward assigned to the last step of failed episodes "
            "(e.g., failure_reward=-300.0)."
        )
    num_workers = cfg.data.get("num_workers", 8)
    tag = cfg.data.get("tag", None)

    datasets_list = cfg.data.get("train_data_paths", None)

    datasets_to_process = []

    if datasets_list is not None and len(datasets_list) > 0:
        for entry in datasets_list:
            entry = dict(entry)
            ds_path = entry.get("dataset_path")
            if ds_path is None:
                raise ValueError("Each dataset entry must have 'dataset_path'")

            if data_root and not Path(ds_path).is_absolute():
                ds_path = str(Path(data_root) / ds_path)

            output_path = entry.get("output_path", None)
            if output_path and data_root and not Path(output_path).is_absolute():
                output_path = str(Path(data_root) / output_path)

            datasets_to_process.append(
                {
                    "dataset_path": ds_path,
                    "output_path": output_path,
                    "dataset_type": entry.get("type", default_type),
                    "gamma": entry.get("gamma", default_gamma),
                    "failure_reward": entry.get(
                        "failure_reward", default_failure_reward
                    ),
                }
            )
    else:
        dataset_path = cfg.data.get("dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "No datasets specified. Either set data.train_data_paths list or data.dataset_path"
            )

        if data_root and not Path(dataset_path).is_absolute():
            dataset_path = str(Path(data_root) / dataset_path)

        output_path = cfg.data.get("output_path", None)
        if output_path and data_root and not Path(output_path).is_absolute():
            output_path = str(Path(data_root) / output_path)

        datasets_to_process.append(
            {
                "dataset_path": dataset_path,
                "output_path": output_path,
                "dataset_type": default_type,
                "gamma": default_gamma,
                "failure_reward": default_failure_reward,
            }
        )

    logger.info(f"Processing {len(datasets_to_process)} dataset(s)...")

    all_stats = []
    for i, ds_config in enumerate(datasets_to_process):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset {i + 1}/{len(datasets_to_process)}")
        logger.info(f"{'=' * 60}")

        dataset_path = Path(ds_config["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        output_path = ds_config["output_path"]
        if output_path is not None:
            output_path = Path(output_path)

        stats = process_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            dataset_type=ds_config["dataset_type"],
            gamma=ds_config["gamma"],
            failure_reward=ds_config["failure_reward"],
            num_workers=num_workers,
            tag=tag,
        )
        all_stats.append(
            {
                "path": str(output_path if output_path else dataset_path),
                "stats": stats,
            }
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("Return computation complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Processed {len(all_stats)} dataset(s):")
    for ds_stats in all_stats:
        ret_stats = ds_stats["stats"]["return"]
        logger.info(f"  {ds_stats['path']}")
        logger.info(
            f"    return: min={ret_stats['min']:.2f}, max={ret_stats['max']:.2f}"
        )


@hydra.main(version_base=None, config_path=None, config_name="recap_compute_returns")
def main(cfg: DictConfig) -> None:
    compute_returns(cfg)


if __name__ == "__main__":
    main()
