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

"""Datasets, batching, and the on-the-wire shapes they produce."""

import asyncio
import copy
import inspect
import json
import random
import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import rlinf.data.datasets.d4rl as d4rl_dataset_module
import rlinf.utils.obs_compression as obs_compression
from rlinf.data.datasets.d4rl import D4RLDataset
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.schema.embodied_trajectory import (
    LeRobotEpisodeAccumulator,
    RolloutGeometry,
    TrajectoryCollector,
    TrajectoryMode,
    TrajectoryPlan,
    select_trajectory_collector,
    select_trajectory_dispatcher,
)
from rlinf.data.schema.embodied_types import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    LeRobotFrame,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    Trajectory,
    TrajectoryKey,
    TrajectorySource,
    TrajectoryStep,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_episode_data,
)
from rlinf.data.storage.lerobot import add_frame_to_dataset, episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.channel.hooks import ChannelContext
from rlinf.scheduler.cluster.utils import (
    TensorPlaceholder,
    pack_dataclass_tensors,
    unpack_dataclass_tensors,
)
from rlinf.utils.env_helpers import SmoothInterveneController
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.obs_compression import (
    _CODEC_KEY,
    compress_obs,
    decompress_obs,
    infer_obs_batch_size,
    is_compressed_image,
    is_compression_enabled,
)
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_piper_online_segment_replay_keeps_partial_human_targets(tmp_path):
    from rlinf.data.datasets.piper_online import segment_trajectory
    from rlinf.data.datasets.rlt_warmup import episode_transitions
    from rlinf.data.storage.replay import TrajectoryReplayBuffer

    actions = np.arange(30 * 7, dtype=np.float32).reshape(30, 7) / 100
    human = np.zeros(30, dtype=bool)
    human[4:7] = True
    values = episode_transitions(
        actions,
        phases=[
            {"start": 2, "end": 15, "outcome": "success"},
            {"start": 20, "end": 24, "outcome": "failure"},
        ],
        terminal=29,
        chunk_length=10,
        stride=10,
        success=True,
        human=human,
        reward_scope="critical_segment",
    )
    arrays = {key: torch.from_numpy(value) for key, value in values.items()}
    states = len(arrays["frames"])
    arrays.update(
        z_rl=torch.zeros(states, 8),
        proprio=torch.zeros(states, 7),
        ref_chunk=torch.ones(states, 50, 7),
    )
    trajectory = segment_trajectory(arrays, 32)
    assert trajectory.actions.shape == (3, 1, 70)
    assert trajectory.forward_inputs["valid_steps"].flatten().tolist() == [10, 3, 4]
    assert trajectory.forward_inputs["next_valid_steps"].flatten().tolist() == [3, 0, 0]
    assert trajectory.rewards[1, 0, 2] == 1
    assert trajectory.rewards[2].sum() == 0
    assert trajectory.terminations.any(-1).flatten().tolist() == [False, True, True]
    torch.testing.assert_close(
        trajectory.actions[0, 0].reshape(10, 7), torch.from_numpy(actions[2:12])
    )
    assert not trajectory.actions[1, 0].reshape(10, 7)[3:].any()
    assert trajectory.intervene_flags[0, 0].reshape(10, 7).any(-1).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    intervention = trajectory.extract_intervene_traj()[0]
    assert intervention.versions.flatten().tolist() == [32]
    assert intervention.forward_inputs["valid_steps"].flatten().tolist() == [10]

    # Save before any sampling, then save a restored pool into a new checkpoint.
    # Both snapshots must contain all indexed trajectories, including evicted cache entries.
    first = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(tmp_path / "live"),
        cache_size=1,
        sample_window_size=2,
    )
    second = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(tmp_path / "live2"),
        cache_size=1,
        sample_window_size=2,
    )
    third = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(tmp_path / "live3"),
        cache_size=1,
        sample_window_size=2,
    )
    try:
        first.add_trajectories([trajectory, trajectory])
        first.save_checkpoint(str(tmp_path / "checkpoint1"))
        assert len(list((tmp_path / "checkpoint1").glob("trajectory_*.pt"))) == 2
        second.load_checkpoint(str(tmp_path / "checkpoint1"))
        second.add_trajectories([trajectory])
        second.save_checkpoint(str(tmp_path / "checkpoint2"))
        assert len(list((tmp_path / "checkpoint2").glob("trajectory_*.pt"))) == 3
        third.load_checkpoint(str(tmp_path / "checkpoint2"))
        assert third.total_samples == 9
        batch = third.sample(16)
        assert batch["actions"].shape == (6, 70)
        assert set(batch["forward_inputs"]) == {
            "valid_steps",
            "discount_steps",
            "next_valid_steps",
        }
    finally:
        first.close()
        second.close()
        third.close()


def test_piper_online_demo_pool_excludes_validation(piper_local_warmup_cache):
    from rlinf.data.datasets.rlt_warmup import CachedRLTReplay
    from rlinf.data.storage.replay import TrajectoryReplayBuffer
    from rlinf.workers.actor.piper_rlt_worker import PiperRLTActor

    cache = CachedRLTReplay(piper_local_warmup_cache)
    worker = object.__new__(PiperRLTActor)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": cache.manifest["model"]},
            "piper": {
                "stage1_manifest": cache.manifest["stage1"],
                "task": cache.manifest["annotations"]["task"],
            },
        }
    )
    worker.demo_buffer = TrajectoryReplayBuffer(sample_window_size=10)
    try:
        expected = sum(len(cache.episodes[i]["current"]) for i in cache.splits["train"])
        assert worker.seed_offline_cache(str(piper_local_warmup_cache)) == expected
        assert len(worker.demo_buffer) == 2
        batch = worker.demo_buffer.sample(100)
        assert (
            batch["curr_obs"]["z_rl"].max() <= 0.1
        )  # Held-out parents use 0.2 and 0.3.
        assert batch["versions"].eq(-1).all()
        worker.cfg.piper.task = "other task"
        with pytest.raises(ValueError, match="task differs"):
            worker.seed_offline_cache(str(piper_local_warmup_cache))
    finally:
        worker.demo_buffer.close()


@pytest.mark.parametrize("chunk", [10, 50])
def test_piper_dense_replay_uses_overlapping_executed_actions_and_one_terminal(chunk):
    from rlinf.data.datasets.rlt_warmup import episode_transitions

    actions = np.arange((chunk + 30) * 7, dtype=np.float32).reshape(-1, 7)
    human = np.zeros(len(actions), dtype=bool)
    human[5:9] = True
    data = episode_transitions(
        actions,
        phases=[
            {"start": 1, "end": chunk + 8, "outcome": "success"},
            {"start": chunk + 12, "end": chunk + 15, "outcome": "failure"},
        ],
        terminal=len(actions) - 1,
        chunk_length=chunk,
        stride=2,
        success=False,
        human=human,
        reward_scope="critical_segment",
        tail_policy="align",
    )
    starts = data["frames"][data["current"]]
    assert starts.tolist() == [1, 3, 5, 7, 8, chunk + 12]
    assert data["valid_steps"].tolist() == [chunk] * 5 + [3]
    assert data["terminated"].tolist() == [False] * 4 + [True, True]
    assert data["rewards"].tolist() == [0] * 4 + [1, 0]
    for row, (first, length) in enumerate(
        zip(starts, data["valid_steps"], strict=True)
    ):
        np.testing.assert_array_equal(
            data["actions"][row, :length], actions[first : first + length]
        )
        np.testing.assert_array_equal(
            data["human"][row, :length], human[first : first + length]
        )
        assert data["frames"][data["next"][row]] == first + length


def test_piper_replay_backfill_reuses_aligned_anchors_and_preserves_rng(tmp_path):
    from rlinf.data.datasets.piper_online import (
        backfill_replay_features,
        save_replay_anchor,
    )

    class FeatureProvider:
        manifest = {"checkpoint_id": "stage1", "norm_stats_sha256": "norm"}
        feature_model = torch.nn.Linear(1, 1)

        def __init__(self):
            self.calls = []

        def extract_features_batch(self, observations, task):
            self.calls.append(
                [int(obs["observation.state"][0]) for obs in observations]
            )
            return [
                {
                    "z_rl": torch.rand(2048).numpy(),
                    "proprio": obs["observation.state"],
                    "ref_chunk": np.full((50, 7), obs["observation.state"][0]),
                    "raw_state": obs["observation.state"],
                }
                for obs in observations
            ]

    provider = FeatureProvider()

    def sample(frame):
        return {"observation.state": np.full(7, frame, dtype=np.float32), "task": "USB"}

    saved = {
        "z_rl": np.ones(2048),
        "proprio": np.zeros(7),
        "ref_chunk": np.ones((50, 7)),
        "raw_state": np.zeros(7),
    }
    save_replay_anchor(tmp_path, 0, provider.manifest, "USB", saved)
    rng = torch.get_rng_state()
    values, stats = backfill_replay_features(
        provider,
        np.array([0, 2, 4, 6]),
        sample,
        directory=tmp_path,
        task="USB",
        fingerprint="fixed",
        batch_size=2,
    )
    assert stats["reused_anchors"] == 1 and stats["backfilled_anchors"] == 3
    assert provider.calls == [[2, 4], [6]]
    torch.testing.assert_close(torch.get_rng_state(), rng)
    np.testing.assert_array_equal(values[0]["ref_chunk"], saved["ref_chunk"])
    assert (values[2]["ref_chunk"] == 2).all()
    repeated, _ = backfill_replay_features(
        provider,
        np.array([0, 2, 4, 6]),
        sample,
        directory=tmp_path,
        task="USB",
        fingerprint="fixed",
        batch_size=2,
    )
    np.testing.assert_array_equal(values[2]["z_rl"], repeated[2]["z_rl"])
    saved["raw_state"] = np.ones(7)
    save_replay_anchor(tmp_path, 0, provider.manifest, "USB", saved)
    with pytest.raises(ValueError, match="aligned"):
        backfill_replay_features(
            provider,
            np.array([0]),
            sample,
            directory=tmp_path,
            task="USB",
            fingerprint="fixed",
            batch_size=2,
        )
    terminal_features, terminal_stats = backfill_replay_features(
        provider,
        np.array([0]),
        lambda frame: {**sample(frame), "complementary_info.action_valid": False},
        directory=tmp_path,
        task="USB",
        fingerprint="fixed",
        batch_size=2,
    )
    assert terminal_stats["reused_anchors"] == 0
    assert (terminal_features[0]["raw_state"] == 0).all()


def test_piper_mixed_replay_keeps_full_batch_and_all_offline_episodes(tmp_path):
    from rlinf.data.schema.embodied_types import Trajectory
    from rlinf.data.storage.replay import TrajectoryReplayBuffer
    from rlinf.data.storage.replay.dataset import ReplayBufferDataset

    def trajectory(version):
        return Trajectory(
            max_episode_length=1,
            model_weights_id=str(version),
            rewards=torch.ones(1, 1, 1),
            actions=torch.full((1, 1, 7), float(version)),
        )

    demo = TrajectoryReplayBuffer(
        full_batch=True,
        sample_window_size=0,
        cache_size=1,
        auto_save=True,
        auto_save_path=str(tmp_path / "demo"),
    )
    online = TrajectoryReplayBuffer(full_batch=True, sample_window_size=200)
    try:
        demo.add_trajectories([trajectory(i) for i in range(205)])
        demo.flush()
        online.add_trajectories([trajectory(-1)])
        dataset = ReplayBufferDataset(
            online,
            demo,
            batch_size=64,
            min_replay_buffer_size=1,
            min_demo_buffer_size=1,
        )
        batch = next(iter(dataset))
        assert batch["actions"].shape == (64, 7)
        assert (batch["actions"][:, 0] == -1).sum() == 32
        assert demo.sample(5000)["actions"][:, 0].min() == 0
    finally:
        online.close()
        demo.close()


class TestD4RLDataset:
    """Tests for loading D4RL transition datasets."""

    @pytest.mark.parametrize("has_next_observations", [False, True])
    def test_from_path_converts_standard_hdf5_dataset(
        self, tmp_path, monkeypatch, has_next_observations
    ):
        """Standard D4RL files go through ``qlearning_dataset``.

        AntMaze files have no ``next_observations``; MuJoCo v2 files have both
        ``next_observations`` and ``timeouts``.
        """
        dataset_path = tmp_path / "standard-d4rl.hdf5"
        dataset_path.touch()
        raw = {
            "observations": np.array([[0.0], [1.0], [10.0], [11.0], [12.0]]),
            "actions": np.array([[-1.5], [0.25], [0.5], [0.75], [1.5]]),
            "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "terminals": np.array([False, False, False, False, True]),
            "timeouts": np.array([False, True, False, False, False]),
        }
        if has_next_observations:
            raw["next_observations"] = np.array([[1.0], [2.0], [11.0], [12.0], [13.0]])
        converted = {
            "observations": raw["observations"][[0, 2, 3]],
            "actions": raw["actions"][[0, 2, 3]],
            "rewards": raw["rewards"][[0, 2, 3]],
            "terminals": raw["terminals"][[0, 2, 3]],
            "next_observations": raw["observations"][[1, 3, 4]],
        }

        env = mock.Mock()
        env.get_dataset.return_value = raw
        gym_api = mock.Mock()
        gym_api.make.return_value = env

        def qlearning_dataset(actual_env, *, dataset):
            assert actual_env is env
            assert dataset is raw
            assert "timeouts" in dataset
            return converted

        d4rl_api = mock.Mock(qlearning_dataset=qlearning_dataset)
        monkeypatch.setattr(d4rl_dataset_module, "gym", gym_api)
        monkeypatch.setattr(d4rl_dataset_module, "d4rl", d4rl_api)

        dataset = D4RLDataset.from_path(dataset_path, task_name="antmaze-test-v0")

        np.testing.assert_allclose(dataset.observations[:, 0], [0.0, 10.0, 11.0])
        np.testing.assert_allclose(dataset.next_observations[:, 0], [1.0, 11.0, 12.0])
        np.testing.assert_allclose(dataset.actions[:, 0], [-0.99999, 0.5, 0.75])
        np.testing.assert_allclose(dataset.rewards, [0.0, 2.0, 3.0])
        np.testing.assert_allclose(dataset.dones_float, [1.0, 0.0, 1.0])
        env.get_dataset.assert_called_once_with(h5path=str(dataset_path))
        env.close.assert_called_once_with()

    def test_from_path_preserves_materialized_transition_dataset(
        self, tmp_path, monkeypatch
    ):
        """Existing files with next observations remain supported."""
        dataset_path = tmp_path / "materialized-d4rl.hdf5"
        dataset_path.touch()
        raw = {
            "observations": np.array([[1.0], [2.0]]),
            "actions": np.array([[-0.25], [0.25]]),
            "rewards": np.array([3.0, 4.0]),
            "terminals": np.array([False, True]),
            "next_observations": np.array([[2.0], [3.0]]),
        }

        env = mock.Mock()
        env.get_dataset.return_value = raw
        gym_api = mock.Mock()
        gym_api.make.return_value = env
        d4rl_api = mock.Mock()
        monkeypatch.setattr(d4rl_dataset_module, "gym", gym_api)
        monkeypatch.setattr(d4rl_dataset_module, "d4rl", d4rl_api)

        dataset = D4RLDataset.from_path(dataset_path, task_name="custom-test-v0")

        np.testing.assert_array_equal(dataset.observations, raw["observations"])
        np.testing.assert_array_equal(
            dataset.next_observations, raw["next_observations"]
        )
        d4rl_api.qlearning_dataset.assert_not_called()
        env.close.assert_called_once_with()


class TestMathDatasetMultithread:
    """Tests for ReasoningDataset multithread processing consistency."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = mock.Mock()
        tokenizer.is_fast = True
        tokenizer.eos_token_id = 2

        def apply_chat_template_side_effect(
            prompts, tokenize=False, add_generation_prompt=True
        ):
            """Mock apply_chat_template that handles generator input."""
            # Convert generator to list if needed
            prompts_list = list(prompts) if not isinstance(prompts, list) else prompts
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n"
                if isinstance(prompt, str)
                else prompt
                for prompt in prompts_list
            ]

        tokenizer.apply_chat_template = mock.Mock(
            side_effect=apply_chat_template_side_effect
        )
        tokenizer.batch_encode_plus = mock.Mock(
            side_effect=lambda texts: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            }
        )
        tokenizer.encode = mock.Mock(side_effect=lambda text: [1] * len(text.split()))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = DictConfig(
            {
                "data": {
                    "max_prompt_length": 1000,
                    "prompt_key": "question",
                    "answer_key": "answer",
                    "apply_chat_template": True,
                    "filter_prompt_by_length": False,
                    "process_workers": 4,
                    "process_batch_size": 32,
                }
            }
        )
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing (at least 10000 items)."""
        # Generate at least 10000 math problems
        data = []
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a // b if b != 0 else 0),
        ]

        for i in range(10000):
            op_symbol, op_func = random.choice(operations)
            a = random.randint(1, 1000)
            b = random.randint(1, 1000) if op_symbol != "/" else random.randint(1, 100)
            if op_symbol == "/" and b == 0:
                b = 1
            result = op_func(a, b)
            question = f"What is {a} {op_symbol} {b}?"
            data.append({"question": question, "answer": str(result)})

        return data

    def test_multithread_vs_singlethread_consistency(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test that multithread processing produces identical results to single-thread processing.

        This test verifies that:
        1. Results from multi-worker processing match single-worker processing
        2. All keys are the same
        3. All values are the same
        """
        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        # We need to reload the raw data to avoid double processing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        time_start = time.time()
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )
        time_elapse_multithread = time.time() - time_start

        # Test with single thread
        time_start = time.time()
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)
        time_elapse_singlethread = time.time() - time_start

        # Verify lengths are equal
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify all items have the same keys and values
        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            keys_mt, keys_st = item_mt.keys(), item_st.keys()
            assert keys_mt == keys_st, (
                f"Keys mismatch at index {idx}: "
                f"multithread={list(keys_mt)}, singlethread={list(keys_st)}"
            )

            # Check all values are equal
            unequal_keys = [key for key in keys_mt if item_mt[key] != item_st[key]]
            assert len(unequal_keys) == 0, (
                f"Values mismatch at index {idx} for keys: {unequal_keys}"
            )

        # Log timing information (for debugging)
        print(
            f"Data count: {len(data_multithread)}, "
            f"Multithread processing time: {time_elapse_multithread:.2f}s, "
            f"Singlethread processing time: {time_elapse_singlethread:.2f}s"
        )

    def test_multithread_consistency_with_filter(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test multithread processing consistency when filter_prompt_by_length is enabled.
        """
        # Update config to enable filtering
        mock_config.data.filter_prompt_by_length = True
        mock_config.data.max_prompt_length = 50  # Reasonable limit to test filtering

        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )

        # Test with single thread
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)

        # Verify consistency
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify that some data was filtered (not all data passed)
        assert len(data_multithread) <= len(raw_data), (
            f"Filtering should reduce data size, but got {len(data_multithread)} >= {len(raw_data)}"
        )

        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            assert item_mt.keys() == item_st.keys(), f"Keys mismatch at index {idx}"
            for key in item_mt.keys():
                assert item_mt[key] == item_st[key], (
                    f"Mismatch at index {idx}, key {key}"
                )

        print(
            f"Filtered data count: {len(data_multithread)}/{len(raw_data)} "
            f"(max_prompt_length={mock_config.data.max_prompt_length})"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def test_split_dict_to_chunk_keeps_mixed_fields_aligned():
    batch = {
        "values": torch.arange(10),
        "sample_ids": list(range(10)),
        "nested": {"values": torch.arange(10) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3)

    assert [chunk["values"].tolist() for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["sample_ids"] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["nested"]["values"].tolist() for chunk in chunks] == [
        [100, 101, 102, 103],
        [104, 105, 106],
        [107, 108, 109],
    ]


def test_split_dict_to_chunk_returns_requested_number_of_chunks():
    batch = {"values": torch.arange(2), "sample_ids": ["a", "b"]}

    chunks = split_dict_to_chunk(batch, 4)

    assert len(chunks) == 4
    assert [chunk["values"].tolist() for chunk in chunks] == [[0], [1], [], []]
    assert [chunk["sample_ids"] for chunk in chunks] == [["a"], ["b"], [], []]


class _LegacyDataset:
    """Mimics lerobot < 0.2: the task lives inside the frame dict."""

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class _PostRevertDataset(_LegacyDataset):
    """Mimics lerobot >= 0.4: back to ``add_frame(frame)``, but it pops the task.

    0.4 reverted the 0.3.x signature, so a version-number check would dispatch
    this one wrongly. It also mutates the caller's dict.
    """

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append({**frame, "task": frame.pop("task")})


class _CurrentDataset:
    """Mimics lerobot >= 0.3: the task is a separate argument.

    Like the real implementation, a ``task`` key inside *frame* is rejected
    because it is not part of the feature schema.
    """

    def __init__(self):
        self.frames = []
        self.tasks = []
        self.saved_episodes = 0

    def add_frame(self, frame, task, timestamp=None):
        if "task" in frame:
            raise ValueError("Extra features: {'task'}")
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        self.saved_episodes += 1


def _make_writer(dataset):
    # ``create()`` needs a real lerobot install, so attach the dataset the way
    # ``create()`` would.
    writer = LeRobotDatasetWriter()
    writer.dataset = dataset
    return writer


def _episode(n=2):
    return [{"state": i, "actions": i, "task": "pick up the cube"} for i in range(n)]


def test_legacy_dataset_keeps_task_in_frame():
    dataset = _LegacyDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


def test_current_dataset_gets_task_as_argument():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode(_episode())

    assert dataset.tasks == ["pick up the cube"] * 2
    assert all("task" not in f for f in dataset.frames)
    assert dataset.frames[0]["state"] == 0
    assert dataset.saved_episodes == 1


ALL_SHAPES = [_LegacyDataset, _CurrentDataset, _PostRevertDataset]


def test_post_revert_dataset_keeps_task_in_frame():
    # lerobot >= 0.4 took the 0.3.x signature back out again.
    dataset = _PostRevertDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_caller_frames_are_not_mutated(dataset_cls):
    # The DAgger worker shares these dicts with the in-memory training store,
    # and lerobot >= 0.4 pops "task" out of whatever frame it is handed.
    episode = _episode()
    before = [dict(f) for f in episode]
    _make_writer(dataset_cls()).add_episode(episode)

    assert episode == before


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_frame_without_task_is_rejected(dataset_cls):
    with pytest.raises(ValueError, match="missing the required 'task' field"):
        add_frame_to_dataset(dataset_cls(), {"state": 0, "actions": 0})


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_add_frame_to_dataset_is_usable_standalone(dataset_cls):
    # The toolkit collectors drive LeRobotDataset directly, without the writer.
    dataset = dataset_cls()
    add_frame_to_dataset(dataset, {"state": 0, "task": "wipe the table"})

    assert len(dataset.frames) == 1


def test_empty_episode_is_skipped():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode([])

    assert dataset.frames == []
    assert dataset.saved_episodes == 0


# --------------------------------------------------------------------------
# episode_boundaries: dataset format v2.1 vs v3.0
# --------------------------------------------------------------------------


class _V21Dataset:
    """Dataset format v2.1: a dict of two tensors on the dataset itself."""

    def __init__(self, starts, ends):
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }


class _V30Meta:
    def __init__(self, starts, ends):
        self.episodes = {"dataset_from_index": starts, "dataset_to_index": ends}


class _V30Dataset:
    """Dataset format v3.0 (lerobot >= 0.4): columns on ``meta.episodes``."""

    def __init__(self, starts, ends):
        self.episode_data_index = None
        self.meta = _V30Meta(starts, ends)


@pytest.mark.parametrize("dataset_cls", [_V21Dataset, _V30Dataset])
def test_episode_boundaries_agree_across_formats(dataset_cls):
    starts, ends = episode_boundaries(dataset_cls([0, 3, 7], [3, 7, 9]))

    assert starts == [0, 3, 7]
    assert ends == [3, 7, 9]
    assert all(isinstance(x, int) for x in starts + ends)


def test_episode_boundaries_reports_an_unknown_layout():
    class _Alien:
        episode_data_index = None
        meta = None

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Alien())


def test_episode_boundaries_rejects_v30_meta_without_the_columns():
    class _Partial:
        episode_data_index = None
        meta = _V30Meta([0], [1])

    _Partial.meta.episodes = {"length": [1]}

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Partial())


# Skip codec round-trip tests when the optional backends are not installed.
_CODECS = []
try:
    import lz4.frame  # noqa: F401

    _CODECS.append("lz4")
except ImportError:
    pass
try:
    import zstandard  # noqa: F401

    _CODECS.append("zstd")
except ImportError:
    pass

requires_codec = pytest.mark.skipif(
    not _CODECS, reason="no observation compression codec (lz4/zstd) installed"
)


def _make_payload(num_envs: int = 4) -> dict:
    """A payload shaped like EnvWorker._build_rollout_input_data output."""
    obs = {
        "main_images": torch.randint(0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (num_envs, 6, 6, 3), dtype=torch.uint8
        ),
        "states": torch.randn(num_envs, 7, dtype=torch.float32),
        "task_descriptions": ["put carrot on plate"] * num_envs,
    }
    return {
        "obs": obs,
        "final_obs": {
            "main_images": torch.randint(
                0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8
            ),
            "states": torch.randn(num_envs, 7, dtype=torch.float32),
        },
        "rlt_switch_flags": None,
    }


def _assert_payload_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_payload_equal(va, vb)
        elif isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"tensor mismatch for {key!r}"
        else:
            assert va == vb, f"value mismatch for {key!r}"


def _cfg(**overrides):
    # A plain dict is sufficient: the codec only calls ``config.get(...)``,
    # which both ``dict`` and OmegaConf's ``DictConfig`` support identically.
    base = {"enable": True, "codec": "lz4", "level": 1, "xor_delta": True}
    base.update(overrides)
    return base


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
@pytest.mark.parametrize("xor_delta", [True, False])
def test_compress_decompress_is_lossless(codec, xor_delta):
    payload = _make_payload()
    config = _cfg(codec=codec, xor_delta=xor_delta)

    compressed = compress_obs(payload, config)
    # Image tensors are replaced by self-describing marker dicts...
    assert _CODEC_KEY in compressed["obs"]["main_images"]
    assert _CODEC_KEY in compressed["obs"]["extra_view_images"]
    # ...while non-image fields are passed through untouched.
    assert isinstance(compressed["obs"]["states"], torch.Tensor)
    assert compressed["obs"]["task_descriptions"] == payload["obs"]["task_descriptions"]

    restored = decompress_obs(compressed)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_single_env_batch_roundtrip():
    # XOR-delta is skipped when there is only one frame; must still be lossless.
    payload = _make_payload(num_envs=1)
    restored = decompress_obs(compress_obs(payload, _cfg(xor_delta=True)))
    _assert_payload_equal(payload, restored)


def test_disabled_config_is_passthrough():
    payload = _make_payload()
    assert compress_obs(payload, _cfg(enable=False)) is payload
    assert compress_obs(payload, None) is payload
    assert not is_compression_enabled(None)
    assert not is_compression_enabled(_cfg(enable=False))
    assert is_compression_enabled(_cfg(enable=True))


def test_decompress_on_uncompressed_payload_is_noop():
    # The rollout worker always routes received data through decompress_obs, so
    # it must be a no-op on payloads sent without compression.
    payload = _make_payload()
    restored = decompress_obs(payload)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_only_uint8_images_are_compressed():
    # A float image-shaped tensor is not a uint8 observation and must be left
    # untouched, as must low-rank uint8 tensors (e.g. flags).
    payload = {
        "obs": {
            "float_map": torch.randn(4, 8, 8, 3),
            "uint8_flags": torch.ones(4, dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert isinstance(compressed["obs"]["float_map"], torch.Tensor)
    assert isinstance(compressed["obs"]["uint8_flags"], torch.Tensor)


def test_unknown_codec_raises():
    payload = _make_payload()
    with pytest.raises(ValueError, match="Unknown observation compression codec"):
        compress_obs(payload, _cfg(codec="bogus"))


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
def test_routing_split_then_compress_roundtrip(codec):
    """Compression must be compatible with the Env->Rollout channel routing.

    The env worker installs compression as a ``split_fn`` so it runs *after*
    the scheduler splits the batch: ``infer_batch_size`` and ``split_batch``
    see plain tensors, and each shard is compressed independently. This test
    reproduces that flow with the real routing helpers and asserts the payload
    survives split -> compress -> decompress -> merge unchanged.
    """
    routing = pytest.importorskip("rlinf.scheduler.worker.routing")

    payload = _make_payload(num_envs=6)
    # The scheduler infers the batch size from the *uncompressed* payload.
    assert routing.infer_batch_size(payload) == 6

    # split_fn = split_batch first, then compress each shard (env send path).
    split_sizes = [2, 1, 3]
    shards = routing.split_batch(payload, split_sizes)
    compressed_shards = [compress_obs(shard, _cfg(codec=codec)) for shard in shards]

    # Rollout side: decompress each shard, then merge (merge_obs path).
    restored_shards = [decompress_obs(shard) for shard in compressed_shards]
    merged = routing.merge_batches(restored_shards)
    _assert_payload_equal(payload, merged)


def test_infer_obs_batch_size_uncompressed():
    payload = _make_payload(num_envs=5)
    assert infer_obs_batch_size(payload) == 5
    # Also accepts a bare obs dict (no "obs" wrapper).
    assert infer_obs_batch_size(payload["obs"]) == 5


@requires_codec
def test_infer_obs_batch_size_with_compressed_images():
    # The rollout worker infers batch size on the receive path, before
    # decompression, so a compressed image must still report its batch size.
    payload = _make_payload(num_envs=5)
    compressed = compress_obs(payload, _cfg())
    assert is_compressed_image(compressed["obs"]["main_images"])
    assert infer_obs_batch_size(compressed) == 5


@requires_codec
def test_infer_obs_batch_size_images_only():
    # Regression: a batch whose only batched field is a (compressed) image,
    # with no states/task_descriptions, must not break batch-size inference.
    payload = {
        "obs": {
            "main_images": torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert infer_obs_batch_size(compressed) == 3


def test_infer_obs_batch_size_raises_when_unbatched():
    with pytest.raises(ValueError, match="Cannot infer batch size"):
        infer_obs_batch_size({"obs": {}})


def test_env_output_composes_one_transition_object():
    transition = EnvTransition(
        rewards=torch.ones(2, 1),
        dones=torch.zeros(2, 1, dtype=torch.bool),
    )
    output = EnvOutput(obs={"states": torch.zeros(2, 3)}, transition=transition)

    assert output.transition is transition
    assert output.rewards is transition.rewards
    assert output.dones is transition.dones
    assert set(output.__dataclass_fields__) == {
        "obs",
        "transition",
        "final_obs",
        "env_infos",
    }


def test_removed_duplicate_types_are_not_schema_api():
    import rlinf.data.schema as schema

    for name in (
        "ChunkStepResult",
        "DummyPolicyInput",
        "EmbodiedRolloutResult",
        "EnvResult",
        "PolicyCompletion",
    ):
        assert not hasattr(schema, name)


def test_env_part_completion_reuses_the_environment_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    transition = EnvTransition(rewards=torch.ones(2, 1))
    part = EnvPart(
        sources=[TrajectorySource(key, 2)],
        transition=transition,
        next_obs={"states": torch.zeros(2, 3)},
        requires_inference=True,
    )

    completed = part.complete(
        next_obs=part.next_obs,
        next_rlt_obs={"states": torch.ones(2, 3)},
        final_prev_values=torch.tensor([[2.0], [3.0]]),
    )

    assert completed.transition is transition
    assert not completed.requires_inference
    assert torch.equal(completed.bootstrap_values, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(completed.final_prev_values, completed.bootstrap_values)


def test_transport_results_store_contiguous_cpu_tensors():
    non_contiguous = torch.arange(12).reshape(3, 4).T

    env_transition = EnvTransition(rewards=non_contiguous)
    policy_output = PolicyOutput(
        forward_inputs={"states": non_contiguous},
    )

    assert env_transition.rewards.device.type == "cpu"
    assert env_transition.rewards.is_contiguous()
    assert not hasattr(policy_output, "actions")
    assert policy_output.forward_inputs["states"].device.type == "cpu"
    assert policy_output.forward_inputs["states"].is_contiguous()


def test_policy_output_detaches_nested_model_tensors_for_transport():
    source = torch.arange(12.0, requires_grad=True)
    model_output = (source * 2).reshape(3, 4).T
    policy_output = PolicyOutput(
        forward_inputs={
            "nested": {
                "states": model_output,
                "features": [model_output[:, :2]],
            }
        },
        prev_logprobs=model_output,
        prev_values=model_output[:, :1],
        versions=model_output,
    )
    part = PolicyPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 4)],
        obs={},
        output=policy_output,
    )

    _, tensors = pack_dataclass_tensors(part)
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(not tensor.requires_grad for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_nested_dataclass_transport_separates_tensors_from_skeleton():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    shared = torch.arange(4).reshape(2, 2)
    event = PolicyPart(
        sources=[TrajectorySource(key, 2)],
        obs={"states": shared, "task_descriptions": ["a", "b"]},
        output=PolicyOutput(
            forward_inputs={"nested": {"states": shared}},
            prev_values=torch.ones(2, 1),
        ),
    )

    skeleton, tensors = pack_dataclass_tensors(event)
    restored = unpack_dataclass_tensors(skeleton, tensors)

    assert isinstance(skeleton.obs["states"], TensorPlaceholder)
    assert isinstance(
        skeleton.output.forward_inputs["nested"]["states"],
        TensorPlaceholder,
    )
    assert len(tensors) == 2
    assert restored.obs["states"] is restored.output.forward_inputs["nested"]["states"]
    assert torch.equal(restored.output.prev_values, torch.ones(2, 1))


def test_policy_input_split_merge_preserves_sources_and_nested_payloads():
    keys = [TrajectoryKey(0, 0, rank, 0, 2) for rank in range(2)]
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(12).reshape(4, 3),
            "labels": np.arange(4),
        },
        rlt_switch_flags=torch.tensor([False, True, False, True]),
        sources=[TrajectorySource(key, 2) for key in keys],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[0].sources == [TrajectorySource(keys[0], 1)]
    assert shards[1].sources == [
        TrajectorySource(keys[0], 1, offset=1),
        TrajectorySource(keys[1], 2),
    ]
    assert merged.sources == policy_input.sources
    assert torch.equal(merged.obs["states"], policy_input.obs["states"])
    assert np.array_equal(merged.obs["labels"], policy_input.obs["labels"])
    assert torch.equal(merged.rlt_switch_flags, policy_input.rlt_switch_flags)


def test_external_policy_input_split_merge_preserves_actions():
    key = TrajectoryKey(0, 0, 0, 0, 1)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        external_actions=torch.arange(24).reshape(4, 2, 3),
        sources=[TrajectorySource(key, 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert all(not shard.requires_inference for shard in shards)
    assert not merged.requires_inference
    assert torch.equal(merged.external_actions, policy_input.external_actions)
    assert merged.sources == policy_input.sources


def test_online_lerobot_payload_survives_source_routing():
    episode_data = {
        "chunk_actions": torch.arange(12).reshape(3, 2, 2),
        "obs_list": [
            {"images": torch.arange(6).reshape(3, 2)},
            {"images": torch.arange(6, 12).reshape(3, 2)},
        ],
        "terminations": torch.tensor([False, True, False]),
        "truncations": torch.tensor([False, False, True]),
        "infos_list": [
            {"score": np.arange(3)},
            {"score": np.arange(3, 6)},
        ],
    }
    split_data = split_episode_data(episode_data, [1, 2])
    merged_data = merge_episode_data(split_data)

    assert merged_data is not None
    assert torch.equal(merged_data["chunk_actions"], episode_data["chunk_actions"])
    assert torch.equal(
        merged_data["obs_list"][1]["images"],
        episode_data["obs_list"][1]["images"],
    )
    assert np.array_equal(
        merged_data["infos_list"][0]["score"],
        episode_data["infos_list"][0]["score"],
    )


def test_env_part_split_merge_preserves_offsets():
    current_key = TrajectoryKey(1, 2, 3, 0, 4)
    previous_key = TrajectoryKey(1, 2, 3, 0, 3)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        sources=[TrajectorySource(current_key, 4)],
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(previous_key, 4)],
                transition=EnvTransition(rewards=torch.arange(4).reshape(4, 1)),
                next_obs={"states": torch.arange(12).reshape(4, 3)},
                requires_inference=False,
                initial_transition=EnvTransition(
                    dones=torch.zeros(4, 1, dtype=torch.bool)
                ),
            )
        ],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[1].sources == [TrajectorySource(current_key, 3, offset=1)]
    assert shards[1].env_parts[0].sources == [
        TrajectorySource(previous_key, 3, offset=1)
    ]
    assert merged.request_sizes == [1, 3]
    assert torch.equal(
        shards[1].env_parts[0].initial_transition.dones,
        torch.zeros(3, 1, dtype=torch.bool),
    )
    assert torch.equal(
        merged.env_parts[1].next_obs["states"],
        torch.arange(12).reshape(4, 3)[1:],
    )


def test_scalar_batch_leaves_survive_a_split_merge_round_trip():
    """Scalars broadcast by a split must collapse back to the original value.

    Source fragments carry scalar info flags (e.g. ``record_reset``) unchanged,
    so reassembling them must not turn one flag into a list of per-shard copies.
    """
    for value in (True, 3, 1.5, "reset"):
        shards = split_batch_value(value, [2, 2])
        assert shards == [value, value]
        assert merge_batch_values(shards) == value

    nested = {"flag": True, "name": "abc", "states": torch.arange(4).reshape(4, 1)}
    shards = split_batch_value(nested, [3, 1])
    merged = merge_batch_values(shards)
    assert merged["flag"] is True
    assert merged["name"] == "abc"
    assert torch.equal(merged["states"], nested["states"])


def test_merging_conflicting_scalar_batch_values_is_rejected():
    with pytest.raises(ValueError, match="conflicting scalar"):
        merge_batch_values([True, False])


def test_episode_data_round_trip_keeps_scalar_info_flags_intact():
    episode_data = {
        "chunk_actions": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "obs_list": [{"states": torch.arange(4).reshape(4, 1)}],
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
        "truncations": torch.zeros(4, 1, dtype=torch.bool),
        "infos_list": [{"record_reset": True, "segment_advance": False}],
    }

    merged = merge_episode_data(split_episode_data(episode_data, [3, 1]))

    assert merged["infos_list"][0]["record_reset"] is True
    assert merged["infos_list"][0]["segment_advance"] is False
    assert torch.equal(merged["chunk_actions"], episode_data["chunk_actions"])


def test_policy_part_owns_routed_fragment_split_and_merge():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    actions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    part = PolicyPart(
        sources=[TrajectorySource(key, 1), TrajectorySource(key, 3, offset=1)],
        obs={"states": torch.arange(12).reshape(4, 3)},
        output=PolicyOutput(
            forward_inputs={"action": actions},
            prev_values=torch.arange(4).reshape(4, 1),
        ),
    )

    fragments = part.split()
    merged = PolicyPart.merge(fragments)

    assert len(fragments) == 2
    assert fragments[1].sources == [TrajectorySource(key, 3, offset=1)]
    assert torch.equal(fragments[0].output.forward_inputs["action"], actions[:1])
    assert torch.equal(merged.obs["states"], part.obs["states"])


def test_trajectory_step_owns_intervention_and_transition_conversion():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    model_actions = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    policy = PolicyPart(
        sources=[TrajectorySource(key, 1)],
        obs={"states": torch.zeros(1, 2), "task_descriptions": ["pick"]},
        output=PolicyOutput(
            forward_inputs={"action": model_actions, "model_action": model_actions},
            prev_values=torch.ones(1, 1),
        ),
    )
    env = EnvPart(
        sources=[TrajectorySource(key, 1)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            intervene_actions=torch.tensor([[9.0, 8.0, 7.0, 6.0]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
        next_obs={"states": torch.ones(1, 2), "task_descriptions": ["pick"]},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=torch.full((1, 1), 5.0),
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )

    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=True,
        collect_transitions=True,
        enable_rlt=False,
        include_final_value=True,
    )

    assert torch.equal(step.actions, torch.tensor([[1.0, 2.0, 7.0, 6.0]]))
    assert torch.equal(step.forward_inputs["action"], step.actions)
    assert "model_action" not in step.forward_inputs
    assert "task_descriptions" not in step.curr_obs
    assert "task_descriptions" not in step.next_obs
    assert torch.equal(step.final_prev_values, torch.full((1, 1), 5.0))


def test_trajectory_owns_step_materialization_splitting_and_batching():
    steps = [
        TrajectoryStep(
            actions=torch.tensor([[1.0], [2.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            initial_dones=torch.zeros(2, 1, dtype=torch.bool),
            forward_inputs={"action": torch.tensor([[1.0], [2.0]])},
            versions=torch.ones(2, 1),
        ),
        TrajectoryStep(
            actions=torch.tensor([[3.0], [4.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            final_prev_values=torch.zeros(2, 1),
            forward_inputs={"action": torch.tensor([[3.0], [4.0]])},
            versions=torch.ones(2, 1),
        ),
    ]

    trajectory = Trajectory.from_steps(steps, max_episode_length=8)
    shards = trajectory.split(2)
    batch = Trajectory.to_batch(shards)

    assert trajectory.actions.shape == (2, 2, 1)
    assert trajectory.dones.shape == (3, 2, 1)
    assert [shard.actions.shape for shard in shards] == [(2, 1, 1)] * 2
    assert torch.equal(batch["actions"], trajectory.actions)
    assert torch.equal(batch["forward_inputs"]["action"], trajectory.actions)


def test_policy_input_methods_replace_legacy_routing_helpers():
    policy_input = PolicyInput(
        obs={"states": torch.arange(8).reshape(4, 2)},
        sources=[TrajectorySource(TrajectoryKey(1, 0, 0, 0, 0), 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert torch.equal(merged.obs["states"], policy_input.obs["states"])


def test_accumulators_are_not_public_schema_api():
    import rlinf.data.schema as schema

    assert not hasattr(schema, "TrajectoryAccumulator")
    assert not hasattr(schema, "EmbodiedTrajectoryBuilder")
    assert not hasattr(schema, "EmbodiedLerobotTrajectoryBuilder")


def _config(**overrides):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "max_episode_steps": 4,
                    "max_steps_per_rollout_epoch": 1,
                    "rollout_epoch": 1,
                    "total_num_envs": 1,
                }
            },
            "rollout": {
                "collect_prev_infos": True,
                "collect_transitions": False,
                "pipeline_stage_num": 1,
            },
            "actor": {
                "micro_batch_size": 1,
                "seed": 1,
                "model": {"action_dim": 1, "num_action_chunks": 1},
            },
            "algorithm": {
                "adv_type": "gae",
                "dagger": {"online_lerobot": {"enabled": False}},
                "gae_lambda": 1.0,
                "gamma": 0.5,
                "group_size": 1,
                "loss_type": "actor_critic",
                "normalize_advantages": False,
                "reward_type": "chunk_level",
                "shuffle_rollout": False,
            },
            "reward": {"env_reward_weight": 1.0, "reward_weight": 1.0},
            "runner": {
                "enable_decoupled_mode": False,
                "task_type": "embodied",
                "use_training_pipeline": False,
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _source(key: TrajectoryKey, size: int = 1, offset: int = 0):
    return TrajectorySource(key, size, offset)


def _policy(
    key: TrajectoryKey,
    value: float = 1.0,
    *,
    external: bool = False,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    kwargs = (
        {"external_actions": value_tensor.unsqueeze(1)}
        if external
        else {
            "output": PolicyOutput(
                forward_inputs={"action": value_tensor},
                prev_logprobs=torch.zeros(1, 1),
                prev_values=torch.zeros(1, 1),
                versions=torch.full((1, 1), 7.0),
            )
        }
    )
    return PolicyPart(
        sources=[_source(key, **source_kwargs)],
        obs={"states": value_tensor},
        **kwargs,
    )


def _env_part(
    key: TrajectoryKey,
    *,
    episode_data=None,
    initial_transition=None,
    value: float = 2.0,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    return EnvPart(
        sources=[_source(key, **source_kwargs)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.ones(1, 1, dtype=torch.bool),
            truncations=torch.ones(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            episode_data=episode_data,
        ),
        next_obs={"states": value_tensor},
        next_rlt_obs={"states": value_tensor},
        bootstrap_values=torch.tensor([[4.0]]),
        final_prev_values=torch.tensor([[5.0]]),
        initial_transition=initial_transition,
    )


def _episode_data():
    return {
        "chunk_actions": torch.zeros(1, 1, 1),
        "obs_list": [{"states": torch.zeros(1, 1)}],
        "terminations": torch.zeros(1, 1, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, dtype=torch.bool),
        "infos_list": [{}],
    }


def _make(cfg=None, **geometry):
    """Build the public collector with a deterministic cluster geometry."""
    shape = {
        "source_count": 1,
        "chunk_count": 1,
        "shards_per_source": 1,
        "actor_world_size": 1,
    }
    shape.update(geometry)
    collector = TrajectoryCollector()
    with patch.object(
        RolloutGeometry, "from_cfg", return_value=RolloutGeometry(**shape)
    ):
        collector.setup(
            ChannelContext(name="Actor", cfg=cfg if cfg is not None else _config())
        )
    return collector


def _collect(collector, *parts):
    outputs = []
    for part in parts:
        outputs.extend(collector.collect(part, DEFAULT_KEY))
    return outputs


def test_policy_part_requires_exactly_one_policy_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    common = {"sources": [_source(key)], "obs": {"states": torch.zeros(1, 1)}}

    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(**common)
    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(
            **common,
            output=PolicyOutput(forward_inputs={}),
            external_actions=torch.zeros(1, 1),
        )


def test_setup_requires_the_run_config():
    with pytest.raises(ValueError, match="needs the run config"):
        TrajectoryCollector().setup(ChannelContext(name="Actor", cfg=None))


@pytest.mark.parametrize(
    ("overrides", "mode", "dispatcher"),
    [
        ({}, TrajectoryMode.ROLLOUT, "least_loaded"),
        (
            {"runner": {"enable_decoupled_mode": False}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"enable_decoupled_mode": True}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"use_training_pipeline": True}},
            TrajectoryMode.PIPELINE,
            None,
        ),
        (
            {"algorithm": {"dagger": {"online_lerobot": {"enabled": True}}}},
            TrajectoryMode.LEROBOT,
            "least_loaded",
        ),
    ],
    ids=["sync", "async", "decoupled", "pipeline", "lerobot"],
)
def test_plan_is_the_single_mode_and_dispatcher_source(overrides, mode, dispatcher):
    cfg = _config(**overrides)

    assert TrajectoryPlan.mode_from_cfg(cfg) is mode
    assert select_trajectory_collector(cfg) is TrajectoryCollector
    assert select_trajectory_dispatcher(cfg) == dispatcher


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "runner": {
                "enable_decoupled_mode": True,
                "use_training_pipeline": True,
            }
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"dagger": {"online_lerobot": {"enabled": True}}},
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"adv_type": "opd"},
        },
    ],
    ids=["pipeline-decoupled", "pipeline-lerobot", "pipeline-opd"],
)
def test_plan_rejects_unsupported_mode_combinations(overrides):
    with pytest.raises(ValueError, match="does not support"):
        TrajectoryPlan.mode_from_cfg(_config(**overrides))


@pytest.mark.parametrize(
    ("decoupled", "part_order"),
    [
        (False, ("policy", "env")),
        (False, ("env", "policy")),
        (True, ("policy", "env")),
        (True, ("env", "policy")),
    ],
    ids=[
        "sync-policy-first",
        "sync-env-first",
        "decoupled-policy-first",
        "decoupled-env-first",
    ],
)
def test_rollout_output_matches_main_fields_for_sync_and_decoupled(
    decoupled, part_order
):
    """Assert the fields built by main's EnvWorker remain byte-for-byte equal."""
    cfg = _config(runner={"enable_decoupled_mode": decoupled})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    parts = {
        "policy": _policy(key),
        "env": _env_part(key, initial_transition=initial),
    }

    [(_, trajectory)] = _collect(collector, *(parts[name] for name in part_order))

    # These are the exact values main appends in EnvWorker._run_interact_once:
    # one initial boundary, one policy step, and gamma * terminal bootstrap.
    assert torch.equal(trajectory.actions, torch.tensor([[[1.0]]]))
    assert torch.equal(trajectory.prev_logprobs, torch.tensor([[[0.0]]]))
    assert torch.equal(trajectory.prev_values, torch.tensor([[[0.0]], [[5.0]]]))
    assert torch.equal(trajectory.rewards, torch.tensor([[[3.0]]]))
    assert torch.equal(trajectory.dones, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.truncations, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.terminations, torch.tensor([[[False]], [[False]]]))
    assert torch.equal(trajectory.versions, torch.tensor([[[7.0]]]))
    assert torch.equal(trajectory.forward_inputs["action"], torch.tensor([[[1.0]]]))


@pytest.mark.parametrize("decoupled", [False, True], ids=["async", "decoupled"])
def test_rollout_sources_flush_independently_like_main_env_workers(decoupled):
    """A slow env source must not add a global barrier outside pipeline mode."""
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"enable_decoupled_mode": decoupled},
    )
    collector = _make(cfg, source_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    first_key = TrajectoryKey(4, 0, 0, 0, 0)
    first_outputs = _collect(
        collector,
        _policy(first_key),
        _env_part(first_key, initial_transition=initial),
    )
    assert len(first_outputs) == 1

    second_key = TrajectoryKey(4, 0, 1, 0, 0)
    second_outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )
    assert len(second_outputs) == 1


def test_collector_joins_out_of_order_routed_fragments_and_initial_state():
    collector = _make(source_count=1, chunk_count=1, shards_per_source=1)
    collector._joiner._source_batch_size = 2
    key = TrajectoryKey(0, 0, 0, 0, 0)
    initial = EnvTransition(dones=torch.zeros(2, 1, dtype=torch.bool))
    policy_tail = _policy(key, 11.0, size=1, offset=1)
    policy_head = _policy(key, 10.0, size=1, offset=0)
    env = EnvPart(
        sources=[_source(key, size=2)],
        transition=EnvTransition(
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            truncations=torch.zeros(2, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.tensor([[20], [21]])},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=initial,
    )

    assert _collect(collector, policy_tail, policy_head) == []
    [(_, trajectory)] = _collect(collector, env)

    assert torch.equal(trajectory.actions, torch.tensor([[[10.0], [11.0]]]))
    assert torch.equal(trajectory.dones[0], initial.dones)


def test_collector_rejects_initial_state_after_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 1)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="Only chunk zero"):
        _collect(collector, _env_part(key, initial_transition=EnvTransition()))


def test_collector_requires_initial_state_for_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="missing its initial state"):
        _collect(collector, _env_part(key))


def test_collector_keeps_joined_parts_when_output_materialization_fails():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy = _policy(key)
    env = _env_part(key, initial_transition=EnvTransition())
    original_emit = collector._output.emit
    collector._output.emit = Mock(side_effect=RuntimeError("invalid output"))

    _collect(collector, policy)
    with pytest.raises(RuntimeError, match="invalid output"):
        _collect(collector, env)

    collector._output.emit = original_emit
    assert len(_collect(collector, policy)) == 1


def test_rollout_collector_rejects_duplicate_completed_key():
    cfg = _config(env={"train": {"rollout_epoch": 2}})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )
        == []
    )
    with pytest.raises(ValueError, match="duplicate trajectory event"):
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )


def test_rollout_materializes_each_epoch_boundary_and_final_value_in_order():
    collector = _make(_config(env={"train": {"rollout_epoch": 2}}))
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 1, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    [(_, trajectory)] = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert torch.equal(
        trajectory.dones,
        torch.tensor([[[False]], [[True]], [[False]], [[True]]]),
    )
    assert torch.equal(
        trajectory.prev_values,
        torch.tensor([[[0.0]], [[5.0]], [[0.0]], [[5.0]]]),
    )


def test_rollout_omits_policy_statistics_when_collection_is_disabled():
    collector = _make(_config(rollout={"collect_prev_infos": False}))
    key = TrajectoryKey(3, 0, 0, 0, 0)

    [(_, trajectory)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=EnvTransition()),
    )

    assert trajectory.prev_logprobs is None
    assert trajectory.prev_values is None


def test_history_reward_assignment_matches_main_across_chunks():
    cfg = _config(
        env={
            "train": {
                "auto_reset": False,
                "max_steps_per_rollout_epoch": 2,
            }
        },
        reward={
            "env_reward_weight": 1.0,
            "history_reward_assign": True,
            "reward_mode": "history_buffer",
            "reward_weight": 2.0,
        },
    )
    collector = _make(cfg, chunk_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 0, 0, 0, 1)
    first_env = _env_part(first_key, initial_transition=initial)
    first_env.transition.dones = torch.zeros(1, 1, dtype=torch.bool)
    first_env.transition.truncations = torch.zeros(1, 1, dtype=torch.bool)
    second_env = _env_part(second_key)
    second_env.transition.reward_model_output = torch.tensor([[3.0]])
    second_env.transition.reward_assign_lengths = [2]

    assert _collect(collector, _policy(first_key), first_env) == []
    [(_, trajectory)] = _collect(
        collector,
        second_env,
        _policy(second_key),
    )

    # Current reward: 1 + 2 * 3. History assignment adds 2 * 3 to step 0.
    assert torch.equal(trajectory.rewards, torch.tensor([[[7.0]], [[7.0]]]))


@pytest.mark.parametrize("loss_type", ["rlt_ac", "rlt_td3"])
def test_rlt_interventions_change_actions_but_preserve_reference_inputs(loss_type):
    collector = _make(_config(algorithm={"loss_type": loss_type}))
    key = TrajectoryKey(3, 0, 0, 0, 0)
    current_ref_chunk = torch.tensor([[1.0, 2.0]])
    intervened_chunk = torch.tensor([[9.0, 8.0]])
    policy = PolicyPart(
        sources=[_source(key)],
        obs={"states": torch.zeros(1, 1)},
        output=PolicyOutput(
            forward_inputs={
                "action": current_ref_chunk,
                "z_rl": torch.tensor([[3.0, 4.0]]),
                "proprio": torch.tensor([[5.0, 6.0, 7.0]]),
                "ref_chunk": current_ref_chunk,
            },
            prev_logprobs=torch.zeros(1, 1),
            prev_values=torch.zeros(1, 1),
            intervene_flags=torch.zeros(1, 1, dtype=torch.bool),
        ),
    )
    env = EnvPart(
        sources=[_source(key)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
            intervene_actions=intervened_chunk,
            intervene_flags=torch.ones(1, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.ones(1, 1)},
        next_rlt_obs={
            "z_rl": torch.tensor([[13.0, 14.0]]),
            "proprio": torch.tensor([[15.0, 16.0, 17.0]]),
            "ref_chunk": torch.tensor([[11.0, 12.0]]),
        },
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=EnvTransition(),
    )

    [(_, trajectory)] = _collect(collector, env, policy)

    assert set(trajectory.curr_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert set(trajectory.next_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert torch.equal(trajectory.curr_obs["ref_chunk"][0], current_ref_chunk)
    assert torch.equal(trajectory.actions[0], intervened_chunk)
    assert trajectory.intervene_flags.all()
    assert torch.equal(policy.output.forward_inputs["ref_chunk"], current_ref_chunk)
    assert torch.equal(
        trajectory.next_obs["ref_chunk"][0], torch.tensor([[11.0, 12.0]])
    )


def test_confirmed_commands_and_human_flags_survive_transport_and_replay():
    key = TrajectoryKey(3, 0, 0, 0, 0)
    policy = _policy(key)
    policy.output.forward_inputs = {
        "action": torch.tensor([[10.0, 20.0]]),
        "model_action": torch.tensor([[10.0, 20.0]]),
    }
    policy.output.intervene_flags = torch.zeros(1, 2, dtype=torch.bool)
    env = _env_part(key)
    env.transition = EnvTransition(
        executed_actions=torch.tensor([[1.0, 2.0]]),
        intervene_actions=torch.tensor([[float("nan"), 3.0]]),
        intervene_flags=torch.tensor([[False, True]]),
    )
    # Transport must carry confirmed nonhuman commands independently of labels.
    env.transition = EnvTransition.merge(env.transition.split([1]))
    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=None,
        collect_prev_infos=False,
        collect_transitions=False,
        enable_rlt=False,
        include_final_value=False,
    )
    torch.testing.assert_close(step.actions, torch.tensor([[1.0, 3.0]]))
    assert step.intervene_flags.tolist() == [[False, True]]
    torch.testing.assert_close(step.forward_inputs["action"], step.actions)
    assert "model_action" not in step.forward_inputs
    torch.testing.assert_close(
        policy.output.forward_inputs["action"], torch.tensor([[10.0, 20.0]])
    )


def test_pipeline_output_contains_main_actor_training_fields():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    key = TrajectoryKey(2, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    [(queue_key, batch)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=initial),
    )

    assert queue_key == "0_0_pipeline_actor"
    assert {
        "actions",
        "advantages",
        "dones",
        "prev_logprobs",
        "prev_values",
        "returns",
        "rewards",
    } <= batch.keys()
    assert batch["actions"].shape[0] == 1
    assert batch["advantages"].shape == batch["returns"].shape


def test_pipeline_waits_for_all_sources_and_routes_each_actor_like_main():
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"use_training_pipeline": True},
    )
    collector = _make(
        cfg,
        source_count=2,
        actor_world_size=2,
    )
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(2, 0, 0, 0, 0)
    second_key = TrajectoryKey(2, 0, 1, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert {queue_key for queue_key, _ in outputs} == {
        "0_0_pipeline_actor",
        "1_1_pipeline_actor",
    }
    assert all("advantages" in batch for _, batch in outputs)


def test_pipeline_flushes_each_epoch_to_the_actor_specific_key():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    collector._output._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    collector._output._pipeline_micro_batches = lambda batch, actor_rank: [batch]

    for epoch_id in (0, 1):
        key = TrajectoryKey(2, epoch_id, 0, 0, 0)
        outputs = _collect(
            collector,
            _env_part(key, initial_transition=EnvTransition()),
            _policy(key),
        )
        assert outputs[0][0] == "0_0_pipeline_actor"
        assert (2, epoch_id) not in collector._output._accumulators


def test_online_lerobot_accepts_external_policy_actions_and_emits_shards():
    cfg = _config(
        algorithm={"dagger": {"online_lerobot": {"enabled": True}}},
    )
    collector = _make(cfg, shards_per_source=2)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    outputs = _collect(
        collector,
        _policy(key, external=True),
        _env_part(
            key,
            episode_data=_episode_data(),
            initial_transition=EnvTransition(),
        ),
    )

    assert len(outputs) == 2
    assert all(key == DEFAULT_KEY for key, _ in outputs)


def test_online_lerobot_accumulator_records_external_intervened_action():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=2,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0, 2.0]]]),
        obs_list=[
            {
                "states": torch.tensor([[0.0, 0.0]]),
                "task_descriptions": ["pick"],
            }
        ],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "intervene_action": torch.tensor([[9.0, 8.0]]),
                "intervene_flag": torch.tensor([True]),
            }
        ],
    )

    [episode] = accumulator.drain_episodes()
    assert torch.equal(
        torch.from_numpy(episode[0]["actions"]), torch.tensor([9.0, 8.0])
    )


def test_online_lerobot_accumulator_preserves_auto_reset_observation():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0]]]),
        obs_list=[{"states": torch.tensor([[100.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "final_observation": {"states": torch.tensor([[10.0]])},
                "final_info": {"success_once": torch.tensor([True])},
            }
        ],
    )
    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[2.0]]]),
        obs_list=[{"states": torch.tensor([[20.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[{}],
    )

    first, second = accumulator.drain_episodes()
    assert first[0]["state"].item() == 10.0
    assert first[0]["is_success"].item()
    assert second[0]["state"].item() == 100.0


def test_online_lerobot_accumulator_filters_unsuccessful_episodes():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        only_success=True,
        num_action_chunks=1,
        action_dim=1,
    )

    for terminated, truncated, success in (
        (True, False, False),
        (False, True, True),
        (True, False, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[1.0]]]),
            obs_list=[{"states": torch.tensor([[1.0]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[truncated]]),
            infos_list=[{"success_once": torch.tensor([success])}],
        )

    [episode] = accumulator.drain_episodes()
    assert episode[-1]["is_success"].item()
    assert episode[-1]["done"].item()


def test_online_lerobot_accumulator_expands_vectorized_action_chunks():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=2,
        num_action_chunks=2,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        obs_list=[
            {"states": torch.tensor([[10.0], [30.0]])},
            {"states": torch.tensor([[20.0], [40.0]])},
        ],
        terminations=torch.tensor([[False, True], [False, True]]),
        truncations=torch.zeros(2, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    episodes = accumulator.drain_episodes()
    assert len(episodes) == 2
    assert [frame["actions"].item() for frame in episodes[0]] == [1.0, 2.0]
    assert [frame["actions"].item() for frame in episodes[1]] == [3.0, 4.0]


def test_online_lerobot_accumulator_uses_policy_intervention_metadata():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=2,
        action_dim=1,
    )
    policy_output = PolicyOutput(
        forward_inputs={"action": torch.tensor([[[9.0], [8.0]]])},
        intervene_flags=torch.tensor([[False, True]]),
    )
    chunk_actions = torch.tensor([[[1.0], [2.0]]])

    accumulator.append_chunk_episode_data(
        policy_output=policy_output,
        chunk_actions=chunk_actions,
        obs_list=[
            {"states": torch.tensor([[10.0]])},
            {"states": torch.tensor([[20.0]])},
        ],
        terminations=torch.tensor([[False, True]]),
        truncations=torch.zeros(1, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    [episode] = accumulator.drain_episodes()
    assert [frame["actions"].item() for frame in episode] == [1.0, 8.0]
    assert [frame["intervene_flag"].item() for frame in episode] == [False, True]


def test_online_lerobot_accumulator_applies_recording_controls():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    for state, action, info, terminated in (
        (10.0, 1.0, {"record_reset": True}, False),
        (20.0, 2.0, {"pre_record": True}, False),
        (30.0, 3.0, {"segment_advance": True}, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[action]]]),
            obs_list=[{"states": torch.tensor([[state]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[False]]),
            infos_list=[info],
        )

    [episode] = accumulator.drain_episodes()
    assert len(episode) == 1
    assert episode[0]["state"].item() == 10.0
    assert episode[0]["actions"].item() == 3.0
    assert episode[0]["segment_id"].item() == 1


def test_lerobot_frame_owns_observation_and_image_conversion():
    frame = LeRobotFrame.from_step(
        observation={
            "main_images": torch.full((2, 2, 3), 0.5),
            "wrist_images": torch.zeros(2, 2, 2, 3),
            "states": torch.tensor([1.0, 2.0]),
            "task_descriptions": ["pick"],
        },
        action=torch.tensor([1.0, 2.0]).numpy(),
        info={
            "intervene_action": torch.tensor([9.0, 8.0]),
            "intervene_flag": torch.tensor(True),
            "success_once": torch.tensor(True),
        },
        segment_id=3,
        action_dim=2,
    )

    assert frame is not None
    output = frame.to_dict(episode_success=True, done=True)
    assert output["actions"].tolist() == [9.0, 8.0]
    assert output["image"].dtype.name == "uint8"
    assert {"wrist_image-0", "wrist_image-1"} <= output.keys()
    assert output["task"] == "pick"
    assert output["segment_id"].item() == 3


def test_offline_lerobot_export_reuses_canonical_frame_conversion():
    collector = object.__new__(CollectEpisode)
    collector.num_envs = 1
    buffer = {
        "observations": [
            {
                "main_images": torch.full((2, 2, 3), 0.5),
                "states": torch.tensor([1.0, 2.0]),
                "task_descriptions": ["pick"],
            }
        ],
        "actions": [torch.tensor([1.0, 2.0])],
        "terminated": [True],
        "infos": [
            {},
            {
                "intervene_action": torch.tensor([9.0, 8.0]),
                "intervene_flag": torch.tensor([True]),
            },
        ],
        "segment_ids": [4],
    }

    [frame] = collector._buffer_to_lerobot_ep(buffer, env_idx=0, is_success=True)
    assert frame["actions"].tolist() == [9.0, 8.0]
    assert frame["image"].dtype.name == "uint8"
    assert frame["task"] == "pick"
    assert frame["segment_id"].item() == 4
    assert frame["done"].item()


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_policy_input_compression_preserves_attached_env_parts(monkeypatch):
    """Trajectory requests remain lossless across split/compress/merge routing."""
    monkeypatch.setattr(
        obs_compression,
        "_get_backend",
        lambda _codec: (
            lambda raw, _level: raw,
            lambda raw: raw,
        ),
    )
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key=key, size=4)
    current_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    next_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    policy_input = PolicyInput(
        obs={"main_images": current_images},
        sources=[source],
        env_parts=[
            EnvPart(
                sources=[source],
                transition=EnvTransition(rewards=torch.ones(4, 2)),
                next_obs={"main_images": next_images},
            )
        ],
    )
    env_worker = object.__new__(EnvWorker)
    env_worker.obs_compression_cfg = {
        "enable": True,
        "codec": "lz4",
        "level": 1,
        "xor_delta": True,
    }

    shards = env_worker._split_and_compress_policy_input(policy_input, [1, 3])

    assert is_compressed_image(shards[0].obs["main_images"])
    assert is_compressed_image(shards[0].env_parts[0].next_obs["main_images"])
    merged = MultiStepRolloutWorker._merge_policy_inputs(shards)
    assert torch.equal(merged.obs["main_images"], current_images)
    assert torch.equal(
        torch.cat([part.next_obs["main_images"] for part in merged.env_parts], dim=0),
        next_images,
    )


def test_sync_and_async_workers_share_the_part_channel_contract():
    """Only Rollout publishes parts; Env communicates exclusively with Rollout."""
    assert "actor_channel" not in inspect.signature(EnvWorker.interact).parameters
    assert "actor_channel" not in inspect.signature(AsyncEnvWorker.interact).parameters
    assert (
        "actor_channel" in inspect.signature(MultiStepRolloutWorker.generate).parameters
    )
    assert (
        "actor_channel"
        in inspect.signature(AsyncMultiStepRolloutWorker.generate).parameters
    )


def test_decoupled_evaluation_service_is_reused_and_cancelled_on_stop():
    async def run():
        worker = object.__new__(AsyncMultiStepRolloutWorker)
        worker.env_decoupled_mode = True
        worker._generate_task = None
        worker._evaluate_task = None
        blocker = asyncio.Event()
        calls = 0

        async def serve(_input_channel, _output_channel):
            nonlocal calls
            calls += 1
            await blocker.wait()

        worker._run_evaluate_service = serve
        await worker.ensure_evaluate_service(Mock(), Mock())
        first_task = worker._evaluate_task
        await worker.ensure_evaluate_service(Mock(), Mock())

        assert worker._evaluate_task is first_task
        assert calls == 1

        worker.stop()
        await asyncio.sleep(0)
        assert first_task.cancelled()

    asyncio.run(run())


def test_async_runner_reuses_decoupled_evaluation_service_across_validations():
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"enable_decoupled_mode": True}})
    runner.env_channel = Mock()
    runner.rollout_channel = Mock()
    runner.rollout = Mock()
    runner.rollout.ensure_evaluate_service.return_value = Mock(
        wait=Mock(return_value=None)
    )
    runner.env = Mock()
    runner.env.evaluate.return_value = Mock(wait=Mock(return_value=[{}]))

    with patch(
        "rlinf.runners.async_embodied_runner.compute_evaluate_metrics",
        return_value={},
    ):
        runner.evaluate()
        runner.evaluate()

    assert runner.rollout.ensure_evaluate_service.call_count == 2
    runner.rollout.evaluate.assert_not_called()


def test_decoupled_policy_route_round_trip():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"policy": ["stale"]}
    rollout.recv_from_and_record_batch_routes_with_timeout = AsyncMock(
        return_value=(_policy_input(), [8, 8])
    )
    rollout.send_to_recorded_batch_routes = Mock()

    policy_input, split_sizes = asyncio.run(
        rollout._receive_policy_input(None, "policy", 0)
    )
    source = torch.zeros(16, 4, requires_grad=True)
    output = source * 2
    rollout._send_actions(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=rollout._merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    sent_output = rollout.send_to_recorded_batch_routes.call_args.kwargs["data"]
    assert torch.equal(sent_output, output)
    assert not sent_output.requires_grad
    assert sent_output.grad_fn is None
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=sent_output,
        tag="policy",
        split_fn=rollout._split_actions,
        split_sizes=[8, 8],
    )


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=torch.zeros(8, 4))

    env._recv_actions(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_action_batch_size,
        decoupled_mode=True,
    )


def test_smooth_intervention_builds_external_input_without_policy_history():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    env = SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0, 3.0]])

    policy_input = controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    assert isinstance(policy_input, PolicyInput)
    assert not policy_input.requires_inference
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )


def test_smooth_intervention_requires_online_lerobot_dagger():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 2, "action_dim": 3}},
            "algorithm": {
                "loss_type": "ppo",
                "dagger": {"online_lerobot": {"enabled": True}},
            },
            "env": {
                "train": {
                    "smooth_intervene": True,
                    "env_type": "realworld",
                    "use_pico": True,
                    "use_spacemouse": False,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="loss_type=embodied_dagger"):
        SmoothInterveneController.from_cfg(
            cfg,
            stage_num=1,
            enable_train=True,
            train_num_envs_per_stage=1,
        )


def test_env_sends_external_request_after_intervention_continues():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = True
    env.env_decoupled_mode = False
    env.env_list = [
        SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0]])
    ]
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 2, enabled=True)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(
        obs={"states": torch.zeros(1, 4)},
        transition=EnvTransition(
            dones=torch.tensor([[False, False]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, {}, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert not policy_input.requires_inference
    assert policy_input.env_parts[0] is not None
    assert policy_input.env_parts[0].next_obs is None
    assert torch.equal(policy_input.obs["states"], env_output.obs["states"])
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
    )


def _publish_step_env(**overrides):
    """Build the minimal EnvWorker used by _publish_step tests."""
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = False
    env.env_decoupled_mode = False
    env.collect_final_values = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    for name, value in overrides.items():
        setattr(env, name, value)
    return env


def _terminal_env_output(reset_obs, terminal_obs, **transition):
    flags = {"dones": torch.ones(1, 1, dtype=torch.bool), **transition}
    return EnvOutput(
        obs={"states": reset_obs},
        final_obs={"states": terminal_obs},
        transition=EnvTransition(**flags),
    )


def test_env_sends_terminal_observation_only_as_a_boundary_override():
    env = _publish_step_env()
    reset_obs = torch.zeros(1, 4)
    terminal_obs = torch.ones(1, 4)
    env_output = _terminal_env_output(
        reset_obs,
        terminal_obs,
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, None, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert torch.equal(policy_input.obs["states"], reset_obs)
    assert torch.equal(policy_input.env_parts[0].next_obs["states"], terminal_obs)
    assert policy_input.env_parts[0].requires_inference


def test_rollout_completes_each_epoch_through_policy_inputs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        side_effect=[
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[1]])},
            ),
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[2]])},
            ),
        ]
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[2]])},
        requires_inference=False,
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[3]])},
        requires_inference=True,
    )
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.tensor([[1]])},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[2]])},
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[3]])},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        side_effect=[
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[1]])}}),
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[2]])}}),
            (
                torch.zeros(1, 1),
                {
                    "forward_inputs": {"states": torch.tensor([[3]])},
                    "prev_values": torch.tensor([[4.0, 5.0]]),
                },
            ),
        ]
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(
        generate_one_epoch(
            rollout,
            input_channel=Mock(),
            output_channel=Mock(),
            actor_channel=actor_channel,
        )
    )

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    assert sum(isinstance(part, PolicyPart) for part in parts) == 2
    env_parts = [part for part in parts if isinstance(part, EnvPart)]
    assert len(env_parts) == 2
    assert env_parts[0].initial_transition is completion0.initial_transition
    assert env_parts[0].next_rlt_obs is None
    assert env_parts[0].next_obs is None
    assert torch.equal(env_parts[1].bootstrap_values, torch.tensor([[4.0]]))
    assert torch.equal(env_parts[1].final_prev_values, torch.tensor([[4.0, 5.0]]))
    assert rollout._predict_rollout_actions.call_count == 3


def test_rollout_routes_external_input_without_model_inference():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        return_value=PolicyOutput(
            forward_inputs={"action": torch.ones(1, 6)},
        )
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    dummy_actions = torch.full((1, 2, 3), 2.0)
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.zeros(1, 4)},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    external_actions=dummy_actions,
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.ones(1, 2, 3),
            {"forward_inputs": {"action": torch.ones(1, 6)}},
        )
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(generate_one_epoch(rollout, Mock(), Mock(), actor_channel))

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    inferred_parts = [
        part for part in parts if isinstance(part, PolicyPart) and part.inferred
    ]
    assert len(inferred_parts) == 1
    external_part = next(
        part for part in parts if isinstance(part, PolicyPart) and not part.inferred
    )
    assert torch.equal(external_part.external_actions, dummy_actions)
    assert rollout._predict_rollout_actions.call_count == 1
    assert torch.equal(rollout._send_actions.call_args_list[1].args[1], dummy_actions)


def test_decoupled_final_completion_uses_next_bootstrap():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.stage_num = 1
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 1
    env.enable_online_lerobot = False
    env.env_decoupled_mode = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(obs={"states": torch.zeros(1, 4)})

    completion = env._publish_step(
        rollout_channel=Mock(),
        env_output=env_output,
        initial_transition=EnvTransition(),
        reward_model_output=None,
        chunk_step_data=None,
        epoch_id=0,
        chunk_id=0,
        stage_id=0,
    )

    assert completion is not None
    assert completion.initial_transition is not None
    assert torch.equal(completion.next_obs["states"], env_output.obs["states"])
    env._send_policy_input.assert_not_called()

    env._send_train_bootstrap(
        rollout_channel=Mock(),
        env_outputs=[env_output],
        step_id=1,
        epoch_id=0,
        previous_env_parts={0: completion},
    )

    policy_input = env._send_policy_input.call_args.args[1]
    assert policy_input.env_parts == [completion]


def test_env_closes_reward_stream_with_final_chunk():
    env = object.__new__(EnvWorker)
    env.rollout_epoch = 2
    env.stage_num = 1
    env.n_train_chunk_steps = 2
    env._prefetched_train_bootstrap = None
    env._trajectory_step = 0
    env._rank = 0
    env.env_list = [SimpleNamespace()]
    env.use_training_pipeline = False
    env.env_decoupled_mode = False
    env.enable_online_lerobot = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_actions = Mock(return_value=torch.zeros(1, 4))
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 4)
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                transition=EnvTransition(rewards=torch.zeros(1, 1)),
            ),
            {},
            {},
        )
    )
    env.get_reward_model_output = Mock(return_value=torch.zeros(1, 1))
    env._publish_step = Mock()
    env.record_env_metrics = Mock()
    env.store_last_obs_and_intervened_info = Mock()
    env.finish_rollout = Mock()
    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]


def test_smooth_intervention_holds_at_the_last_commanded_step():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    seen = []

    def get_hold_actions(fallback=None):
        seen.append(fallback)
        return [[1.0, 2.0, 3.0]]

    env = SimpleNamespace(get_hold_actions=get_hold_actions)
    controller.remember_actions(0, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))

    controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    # The wrapper receives the final step of the chunk that was just executed,
    # so it can hold that pose instead of snapping back to its own default.
    assert len(seen) == 1
    assert seen[0].tolist() == [[4.0, 5.0, 6.0]]


def test_rollout_runs_requested_terminal_inference_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "prev_values": torch.zeros(1, 1),
                "forward_inputs": {"states": torch.ones(1, 1)},
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once_with(completion.next_obs)
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.bootstrap_values, torch.zeros(1, 1))
    assert torch.equal(part.final_prev_values, torch.zeros(1, 1))
    assert part.next_rlt_obs is None
    assert part.next_obs is None


def test_rollout_reuses_policy_obs_for_transition_next_obs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = True
    actor_channel = Mock()
    policy_obs = {"states": torch.ones(1, 4)}
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
    )

    rollout._publish_env_part(
        completion,
        policy_obs,
        {"unused": torch.ones(1, 1)},
        actor_channel,
    )

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is policy_obs
    assert part.next_rlt_obs is None


def test_rollout_runs_terminal_inference_for_rlt_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    rollout.hf_model = SimpleNamespace()
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "forward_inputs": {
                    "rlt_transition_z_rl": torch.ones(1, 2),
                    "rlt_transition_proprio": torch.ones(1, 3),
                    "rlt_transition_ref_chunk": torch.ones(1, 4),
                }
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once()
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.next_rlt_obs["z_rl"], torch.ones(1, 2))


def test_rollout_keeps_next_forward_inputs_only_for_rlt():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    actor_channel = Mock()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy_input = PolicyInput(
        obs={"states": torch.zeros(1, 4)},
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(key, 1)],
                transition=EnvTransition(),
            )
        ],
        request_sizes=[1],
    )
    forward_inputs = {
        "rlt_transition_z_rl": torch.ones(1, 2),
        "rlt_transition_proprio": torch.ones(1, 3),
        "rlt_transition_ref_chunk": torch.ones(1, 4),
        "unrelated": torch.ones(1, 8),
    }

    rollout._publish_env_parts(policy_input, forward_inputs, actor_channel)

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is None
    assert torch.equal(
        part.next_rlt_obs["z_rl"],
        forward_inputs["rlt_transition_z_rl"],
    )
    assert set(part.next_rlt_obs) == {"z_rl", "proprio", "ref_chunk"}


def test_vlm_trend_batch_video_metadata_stays_nested_per_sample():
    pytest.importorskip("transformers.video_utils")
    from transformers.video_utils import VideoMetadata

    from rlinf.data.datasets.vlm.vlm_trend_reward import VLMTrendRewardSFTDataset

    captured = {}

    class _Processor:
        video_token = "<|video_pad|>"

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured["videos_kwargs"] = kwargs["videos_kwargs"]
            batch = len(kwargs["text"])
            return {
                "input_ids": torch.zeros(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    VLMTrendRewardSFTDataset.process_inputs(
        processor=_Processor(),
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[["task a"], ["task b"]],
        videos=[
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        ],
        answer_text=None,
        video_fps=24.0,
    )

    metadata = captured["videos_kwargs"]["video_metadata"]
    assert len(metadata) == 2
    assert all(len(sample) == 2 for sample in metadata)
    assert all(
        isinstance(item, VideoMetadata) for sample in metadata for item in sample
    )
    assert metadata[0][0].total_num_frames == 5
    assert metadata[0][0].frames_indices == [0, 1, 2, 3, 4]
    assert metadata[1][1].total_num_frames == 5


@pytest.mark.parametrize("legacy_names", [False, True])
def test_piper_data_transform_preserves_views_and_absolute_actions(legacy_names):
    from rlinf.models.embodiment.openpi.policies.piper_policy import (
        PiperInputs,
        PiperOutputs,
    )

    names = (
        ("front", "front_2", "wrist")
        if legacy_names
        else ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    )
    frame = {
        "observation.state": np.array([180, 90, -90, 0, 45, -45, 70]),
        "action": np.array([[90, 45, -45, 0, 30, -30, 35]] * 10),
        "task": "insert the USB cable",
    }
    for i, name in enumerate(names):
        frame[f"observation.images.{name}"] = np.full(
            (3, 8, 9), (i + 1) / 255, np.float32
        )
    original = copy.deepcopy(frame)
    sample = PiperInputs()(frame)
    np.testing.assert_allclose(
        sample["state"],
        [np.pi, np.pi / 2, -np.pi / 2, 0, np.pi / 4, -np.pi / 4, 0.07],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        sample["actions"][0, [0, 6]], [np.pi / 2, 0.035], rtol=1e-6
    )
    for i, image in enumerate(sample["image"].values()):
        assert image.shape == (8, 9, 3)
        assert image.dtype == np.uint8
        assert np.all(image == i + 1)
    assert all(sample["image_mask"].values())
    assert sample["prompt"] == frame["task"]
    padded = np.pad(sample["actions"], ((0, 0), (0, 25)))
    np.testing.assert_allclose(
        PiperOutputs()({"actions": padded})["actions"], frame["action"], atol=1e-5
    )
    np.testing.assert_array_equal(frame["action"], original["action"])


@pytest.mark.parametrize("values", [np.zeros(14), np.zeros(6), [float("nan")] * 7])
def test_piper_data_rejects_wrong_embodiment_and_nonfinite_values(values):
    from rlinf.models.embodiment.openpi.policies.piper_policy import (
        piper_to_model_units,
    )

    with pytest.raises(ValueError):
        piper_to_model_units(values)


@pytest.mark.parametrize(
    "label,expected",
    [(True, True), (0, False), ("success", True), ("failure", False), ("false", False)],
)
def test_lerobot_episode_success_is_explicit(label, expected):
    from rlinf.data.storage.lerobot.metadata import parse_success

    assert parse_success(label) is expected


@pytest.mark.parametrize("label", [None, "unknown", "", 2, float("nan")])
def test_lerobot_unknown_outcome_is_not_assumed_successful(label):
    from rlinf.data.storage.lerobot.metadata import parse_success

    with pytest.raises(ValueError, match="Unknown"):
        parse_success(label)


def test_recap_returns_cover_v3_split_episodes_and_metadata_labels(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        process_dataset,
    )
    from rlinf.data.storage.lerobot.metadata import load_task_descriptions

    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    # Episode 0 crosses a file boundary; file ordering is deliberately reversed.
    for name, episodes, frames in [
        ("file-001", [0, 0], [0, 1]),
        ("file-000", [0, 1, 1], [2, 0, 1]),
    ]:
        pq.write_table(
            pa.table(
                {
                    "episode_index": episodes,
                    "frame_index": frames,
                    "task_index": [0] * len(frames),
                }
            ),
            tmp_path / f"data/chunk-000/{name}.parquet",
        )
    pq.write_table(
        pa.table({"episode_index": [0, 1], "episode_success": ["success", "failure"]}),
        tmp_path / "meta/episodes/chunk-000/file-000.parquet",
    )
    pd.DataFrame({"task_index": [0]}, index=["insert USB"]).to_parquet(
        tmp_path / "meta/tasks.parquet"
    )
    assert load_task_descriptions(tmp_path) == {0: "insert USB"}
    process_dataset(tmp_path, None, "rollout", 1, -10, tag="test")
    table = pq.read_table(tmp_path / "meta/returns_test.parquet")
    assert table["return"].to_pylist() == [-2, -1, 0, -11, -10]
    assert table["reward"].to_pylist() == [-1, -1, 0, -1, -10]
    assert table["prompt"].to_pylist() == ["insert USB"] * 5
    # Explicit failures must not become successes even under an SFT declaration.
    process_dataset(tmp_path, None, "sft", 1, -10, tag="sft")
    assert (
        pq.read_table(tmp_path / "meta/returns_sft.parquet")["reward"].to_pylist()[-1]
        == -10
    )


@pytest.mark.parametrize("frames", [[0, 0], [0, 2], [1, 2]])
def test_recap_returns_refuse_duplicate_or_missing_frames(frames):
    import pyarrow as pa

    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        compute_returns_table,
    )

    table = pa.table({"episode_index": [0, 0], "frame_index": frames})
    with pytest.raises(ValueError, match="frame indices"):
        compute_returns_table(table, "sft", 1, -10, {})


def test_recap_returns_reject_missing_and_conflicting_outcomes():
    import pyarrow as pa

    from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
        compute_returns_table,
    )

    table = pa.table({"episode_index": [0], "frame_index": [0]})
    with pytest.raises(ValueError, match="needs is_success"):
        compute_returns_table(table, "rollout", 1, -10, {})
    table = table.append_column("is_success", pa.array([False]))
    with pytest.raises(ValueError, match="Conflicting"):
        compute_returns_table(table, "rollout", 1, -10, {}, {0: True})
    assert compute_returns_table(table, "rollout", 0.9, -10, {})[
        "reward"
    ].to_pylist() == [-10]


def test_piper_value_training_and_inference_use_same_transform():
    pytest.importorskip("openpi")
    from openpi.transforms import compose

    from rlinf.data.datasets.recap.value_dataset import ValueDataset
    from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
        build_input_transforms,
    )

    sample = {
        "observation.state": np.array([10, 20, 30, 40, 50, 60, 35]),
        "task": "insert USB",
    }
    for name in ("front", "front_2", "wrist"):
        sample[f"observation.images.{name}"] = np.zeros((8, 9, 3), np.uint8)
    train = ValueDataset._build_transform("piper", "pi05", 7, None)(
        copy.deepcopy(sample)
    )
    infer = compose(build_input_transforms("piper", "pi05", 7, None, None, True))(
        copy.deepcopy(sample)
    )
    np.testing.assert_array_equal(train["state"], infer["state"])
    assert train["prompt"] == infer["prompt"]
    for key in train["image"]:
        np.testing.assert_array_equal(train["image"][key], infer["image"][key])


def test_piper_camera_aliases_are_unambiguous():
    from rlinf.models.embodiment.openpi.policies.piper_policy import PiperInputs

    image = np.zeros((8, 9, 3), np.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        PiperInputs()(
            {"observation.images.front": image, "observation.images.base_0_rgb": image}
        )
    with pytest.raises(ValueError, match="exactly one"):
        PiperInputs()({})


def test_piper_prompt_falls_back_from_null_sidecar_feature():
    from rlinf.models.embodiment.openpi.policies.piper_policy import PiperInputs

    sample = {"observation.state": np.zeros(7), "prompt": None, "task": "insert USB"}
    for name in ("front", "front_2", "wrist"):
        sample[f"observation.images.{name}"] = np.zeros((8, 9, 3), np.uint8)
    assert PiperInputs()(sample)["prompt"] == "insert USB"


def test_piper_recap_configs_resolve_dataset_and_normalization(monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    for key, value in {
        "REPO_PATH": str(root),
        "PIPER_DATA_PATH": "/test/piper",
        "SIGLIP_MODEL_PATH": "/test/siglip",
        "GEMMA_MODEL_PATH": "/test/gemma",
        "PI05_MODEL_PATH": "/test/pi05",
        "PIPER_VALUE_CHECKPOINT": "/test/value",
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "examples/offline_rl/config")
    ):
        for name in ("piper_recap_value", "piper_recap_advantages", "piper_recap_cfg"):
            cfg = compose(config_name=name)
            OmegaConf.to_container(cfg, resolve=True)
            assert cfg.data.train_data_paths[0].dataset_path == "/test/piper"
            assert cfg.data.train_data_paths[0].type == "rollout"
            if name == "piper_recap_cfg":
                assert cfg.actor.model.openpi.config_name == "pi05_piper"
                assert (
                    cfg.actor.model.openpi_data.norm_stats_path
                    == "/test/piper/meta/norm_stats.json"
                )


def test_piper_preparation_preserves_source_and_refuses_existing_output(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from toolkits.lerobot.prepare_piper_recap import prepare_dataset

    source = tmp_path / "source"
    output = tmp_path / "prepared"
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "videos").mkdir()
    names = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    features = {
        key: {"shape": [7], "names": names} for key in ("action", "observation.state")
    }
    for name in ("front", "front_2", "wrist"):
        features[f"observation.images.{name}"] = {"shape": [8, 9, 3]}
    info = json.dumps(
        {
            "robot_type": "piper_follower",
            "features": features,
            "total_frames": 4,
            "total_episodes": 2,
            "fps": 30,
        }
    )
    (source / "meta/info.json").write_text(info)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
                "action": [[90.0] * 6 + [35.0]] * 4,
                "observation.state": [[0.0] * 7] * 4,
            }
        ),
        source / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "length": [2, 2],
                "episode_success": ["success", "failure"],
            }
        ),
        source / "meta/episodes/chunk-000/file-000.parquet",
    )
    report = prepare_dataset(source, output, failure_reward=-10)
    assert report["frames"] == 4
    assert (source / "meta/info.json").read_text() == info
    assert not (source / "meta/returns_piper.parquet").exists()
    assert (output / "data").is_symlink()
    assert (output / "videos").resolve() == source / "videos"
    stats = json.loads((output / "meta/norm_stats.json").read_text())["norm_stats"]
    np.testing.assert_allclose(
        stats["actions"]["mean"], [np.pi / 2] * 6 + [0.035], rtol=1e-6
    )
    assert pq.read_table(output / "meta/returns_piper.parquet")[
        "return"
    ].to_pylist() == [-1, 0, -11, -10]
    with pytest.raises(FileExistsError):
        prepare_dataset(source, output)
    with pytest.raises(ValueError, match="outside"):
        prepare_dataset(source, source / "nested")


def test_piper_success_selection_preserves_episode_and_video_boundaries(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from toolkits.lerobot.select_piper_success import select_success

    selected = []
    originals = {}
    for source_number in range(2):
        root = tmp_path / f"source_{source_number}"
        (root / "meta/episodes").mkdir(parents=True)
        (root / "data").mkdir()
        features = {
            key: {"dtype": "float32", "shape": [7]}
            for key in ("action", "observation.state")
        }
        features.update(
            {
                key: {"dtype": "int64", "shape": [1]}
                for key in ("frame_index", "episode_index", "index", "task_index")
            }
        )
        features["timestamp"] = {"dtype": "float32", "shape": [1]}
        features["observation.depths.front"] = {"dtype": "uint16", "shape": [1, 1]}
        episodes = []
        for camera in ("front", "front_2", "wrist"):
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": [2, 2, 3],
            }
            # The selector links external media; decoding is covered by the real-data probe.
            (root / f"{camera}.mp4").write_bytes(bytes([source_number]))
        info = {
            "codebase_version": "v3.0",
            "robot_type": "piper_follower",
            "fps": 30,
            "features": features,
            "data_path": "data/file-{file_index}.parquet",
            "video_path": "{video_key}.mp4",
        }
        for camera in ("front", "front_2", "wrist"):
            (root / f"{camera}.mp4").rename(root / f"observation.images.{camera}.mp4")
        (root / "meta/info.json").write_text(json.dumps(info))
        pq.write_table(
            pa.table({"task_index": [0], "__index_level_0__": ["insert USB"]}),
            root / "meta/tasks.parquet",
        )
        for ep_index in range(2):
            ep = {
                "episode_index": ep_index,
                "episode_success": "success" if ep_index else "failure",
                "length": 2,
                "tasks": ["insert USB"],
                "data/chunk_index": 0,
                "data/file_index": ep_index,
            }
            for camera in ("front", "front_2", "wrist"):
                prefix = f"videos/observation.images.{camera}"
                ep.update(
                    {
                        f"{prefix}/chunk_index": 0,
                        f"{prefix}/file_index": 0,
                        f"{prefix}/from_timestamp": ep_index / 15,
                        f"{prefix}/to_timestamp": (ep_index + 1) / 15,
                    }
                )
            episodes.append(ep)
            pq.write_table(
                pa.table(
                    {
                        "action": [[source_number + ep_index] * 7] * 2,
                        "observation.state": [[source_number] * 7] * 2,
                        "timestamp": [0.0, 1 / 30],
                        "frame_index": [0, 1],
                        "episode_index": [ep_index] * 2,
                        "index": [ep_index * 2, ep_index * 2 + 1],
                        "task_index": [0, 0],
                        "observation.depths.front": [[[42]], [[42]]],
                    }
                ),
                root / f"data/file-{ep_index}.parquet",
            )
        pq.write_table(
            pa.Table.from_pylist(episodes), root / "meta/episodes/file.parquet"
        )
        selected.append(
            {"source_dataset": str(root), "source_episode_index": 1, "frames": 2}
        )
        originals.update({p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"selected_episodes": selected}))
    output = tmp_path / "output"
    report = select_success(manifest, output)
    assert report["episodes"] == 2 and report["frames"] == 4
    rows = pq.read_table(
        output / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    assert [r["dataset_from_index"] for r in rows] == [0, 2]
    for index, row in enumerate(rows):
        assert row["episode_success"] == "success"
        assert row["videos/observation.images.front/from_timestamp"] == 1 / 15
        video = (
            output / f"videos/observation.images.front/chunk-000/file-{index:03}.mp4"
        )
        assert (
            video.resolve() == tmp_path / f"source_{index}/observation.images.front.mp4"
        )
        table = pq.read_table(output / f"data/chunk-000/file-{index:03}.parquet")
        assert table["episode_index"].to_pylist() == [index, index]
        assert table["index"].to_pylist() == [index * 2, index * 2 + 1]
        assert table["action"].to_pylist() == [[index + 1] * 7] * 2
        assert "observation.depths.front" not in table.column_names
    assert all(p.read_bytes() == data for p, data in originals.items())
    with pytest.raises(FileExistsError):
        select_success(manifest, output)
    manifest.write_text(json.dumps({"selected_episodes": [selected[0], selected[0]]}))
    with pytest.raises(ValueError, match="Duplicate"):
        select_success(manifest, tmp_path / "duplicate")
    alias = tmp_path / "source_alias"
    alias.symlink_to(tmp_path / "source_0", target_is_directory=True)
    manifest.write_text(
        json.dumps(
            {
                "selected_episodes": [
                    selected[0],
                    {**selected[0], "source_dataset": str(alias)},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Duplicate"):
        select_success(manifest, tmp_path / "duplicate_alias")
    selected[0]["source_episode_index"] = 0
    manifest.write_text(json.dumps({"selected_episodes": selected}))
    with pytest.raises(ValueError, match="success/length"):
        select_success(manifest, tmp_path / "failure")
    assert not (tmp_path / "failure").exists()


@pytest.mark.parametrize(
    "stage1_name", ["piper_rlt_stage1", "piper_usb_rlt_stage1_success97"]
)
def test_piper_native_rlt_configs_share_horizon_stats_and_units(
    monkeypatch, stage1_name
):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    for name, value in {
        "REPO_PATH": str(root),
        "EMBODIED_PATH": str(root / "examples/sft"),
        "PIPER_DATA_PATH": "/test/data",
        "PI05_MODEL_PATH": "/test/base",
        "PIPER_STAGE1_PATH": "/test/stage1",
    }.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "examples/sft/config")
    ):
        stage1 = compose(config_name=stage1_name)
        OmegaConf.to_container(stage1, resolve=True)
    if stage1_name == "piper_usb_rlt_stage1_success97":
        assert stage1.actor.global_batch_size == 64
        assert stage1.actor.micro_batch_size == 8
        assert stage1.runner.max_steps == 60000
        assert stage1.cluster.component_placement.actor == "0-7"
    assert stage1.actor.model.openpi.train_expert_only == (
        stage1_name == "piper_usb_rlt_stage1_success97"
    )
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "examples/embodiment/config")
    ):
        stage2 = compose(config_name="piper_rlt_stage2")
        OmegaConf.to_container(stage2, resolve=True)
    assert stage1.actor.model.openpi.use_rlt
    assert stage1.actor.model.model_type == "openpi"
    assert stage2.algorithm.loss_type == "rlt_ac"
    assert stage2.actor.model.proprio_dim == 7
    assert stage2.actor.model.action_dim == 7
    assert (
        stage1.actor.model.openpi.action_horizon
        == stage2.actor.model.ref_num_action_chunks
        == 10
    )
    assert stage2.env.train.init_params.id == "PiperPolicyEnv-v1"
    assert (
        stage2.rollout.rlt_feature_model.openpi_data.norm_stats_path
        == stage2.env.train.override_cfg.norm_stats_path
    )
    assert not stage2.env.train.keyboard_reward_wrapper


@pytest.mark.parametrize("episodes", [None, [1, 2], [0, 2]])
def test_openpi_lerobot_sequence_reads_only_requested_rows(tmp_path, episodes):
    from openpi.training.data_loader import _import_lerobot_dataset

    from rlinf.data.datasets.openpi.lerobot_dataset import LeRobotSFTDataset
    from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch

    lerobot = _import_lerobot_dataset()
    root = tmp_path / "sequence_data"
    writer = lerobot.LeRobotDataset.create(
        "sequence_data",
        root=root,
        fps=30,
        robot_type="piper_follower",
        features={
            key: {"dtype": "float32", "shape": (7,), "names": None}
            for key in ("action", "observation.state")
        },
        use_videos=False,
    )
    for episode in range(3):
        for frame in range(12):
            value = np.full(7, episode * 100 + frame, dtype=np.float32)
            writer.add_frame(
                {
                    "action": value,
                    "observation.state": -value,
                    "task": "test sequence boundaries",
                }
            )
        writer.save_episode()
    writer.finalize()

    deltas = {
        "action": [-1 / 30, 0.0, 1 / 30, 2 / 30],
        "observation.state": [0.0, 1 / 30],
    }
    reader = LeRobotSFTDataset(
        "sequence_data",
        root=root,
        episodes=episodes,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="pyav",
    )
    reference = lerobot.LeRobotDataset(
        "sequence_data",
        root=root,
        episodes=episodes,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="pyav",
    )
    formatted_rows = []

    def count_formatted_rows(batch):
        formatted_rows.append(len(next(iter(batch.values()))))
        return decode_image_struct_batch(batch)

    reader.hf_dataset.set_transform(count_formatted_rows)
    reference.hf_dataset.set_transform(decode_image_struct_batch)
    for index in range(len(reader)):
        actual = reader[index]
        expected = reference[index]
        assert actual.keys() == expected.keys()
        for key in actual:
            if torch.is_tensor(actual[key]):
                assert torch.equal(actual[key], expected[key]), (index, key)
            else:
                assert actual[key] == expected[key]
        frame = int(actual["frame_index"])
        episode = int(actual["episode_index"])
        expected_actions = [
            episode * 100 + min(11, max(0, frame + offset)) for offset in (-1, 0, 1, 2)
        ]
        torch.testing.assert_close(
            actual["action"],
            torch.tensor(expected_actions).float()[:, None].expand(-1, 7),
        )
        assert actual["action_is_pad"].tolist() == [
            frame + offset < 0 or frame + offset > 11 for offset in (-1, 0, 1, 2)
        ]
    assert max(formatted_rows) == 4
    with pytest.raises(IndexError):
        reader[len(reader)]


class _OpenPISamplingObservations:
    """Carry row indices through the real OpenPI observation/batch wrappers."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        value = np.array([index], dtype=np.float32)
        return {
            "image": {},
            "image_mask": {},
            "state": value,
            "actions": value[None, :],
        }


def _make_openpi_sampling_loaders(
    *, size=1035, world_size=8, rank=0, shuffle=True, num_workers=0
):
    from openpi.training.config import DataConfig
    from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader

    from rlinf.data.datasets.openpi.official_sft_data_loader import OpenPISftDataLoader

    dataset = _OpenPISamplingObservations(size)
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )
    raw = DataLoaderImpl(
        DataConfig(repo_id="indices-only"),
        TorchDataLoader(
            dataset,
            local_batch_size=8,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            seed=42,
            framework="pytorch",
        ),
    )
    return raw, OpenPISftDataLoader(raw)


def _openpi_sampling_indices(loader):
    batches = []
    iterator = iter(loader)
    for _ in range(len(loader)):
        observation, actions = next(iterator)
        torch.testing.assert_close(observation.state, actions[:, 0])
        assert observation.state.dtype == torch.float32
        batches.append(observation.state[:, 0].to(torch.int64))
    with pytest.raises(StopIteration):
        next(iterator)
    return torch.cat(batches)


def test_openpi_sft_epochs_reshuffle_without_changing_first_pass_or_shards():
    from itertools import islice

    from rlinf.data.datasets.common.epoch import forward_set_epoch
    from rlinf.data.datasets.openpi import (
        get_official_openpi_sft_num_batches,
        is_official_openpi_sft_dataloader,
    )

    all_epochs = [[], []]
    for rank in range(8):
        raw, loader = _make_openpi_sampling_loaders(rank=rank)
        assert len(loader) == 16
        assert get_official_openpi_sft_num_batches(loader) == len(loader)
        assert get_official_openpi_sft_num_batches(raw) == len(loader)
        assert is_official_openpi_sft_dataloader(loader)
        assert loader.data_config() is raw.data_config()
        reference = torch.cat(
            [obs.state[:, 0].to(torch.int64) for obs, _ in islice(raw, len(loader))]
        )
        for epoch in range(2):
            forward_set_epoch(loader, epoch)
            all_epochs[epoch].append(_openpi_sampling_indices(loader))
            assert loader.sampler.epoch == epoch
        assert torch.equal(all_epochs[0][-1], reference)
        assert not torch.equal(all_epochs[0][-1], all_epochs[1][-1])
        forward_set_epoch(loader, 0)
        assert torch.equal(_openpi_sampling_indices(loader), reference)

    for shards in all_epochs:
        indices = torch.cat(shards)
        assert len(indices) == len(torch.unique(indices)) == 1024
        assert int(indices.min()) >= 0 and int(indices.max()) < 1035


@pytest.mark.parametrize("num_workers", [0, 2])
def test_openpi_sft_rollover_uses_next_epoch_with_real_loader_workers(num_workers):
    from rlinf.data.datasets.common.epoch import forward_set_epoch

    _, loader = _make_openpi_sampling_loaders(size=139, num_workers=num_workers)
    data_iter = iter(loader)
    epochs = []
    epoch = 0
    for _ in range(2):
        batches = []
        for _ in range(len(loader)):
            try:
                observation, _ = next(data_iter)
            except StopIteration:
                epoch += 1
                forward_set_epoch(loader, epoch)
                data_iter = iter(loader)
                observation, _ = next(data_iter)
            batches.append(observation.state[:, 0].to(torch.int64))
        epochs.append(torch.cat(batches))
    assert epoch == loader.sampler.epoch == 1
    assert not torch.equal(*epochs)
    _, fresh = _make_openpi_sampling_loaders(size=139)
    forward_set_epoch(fresh, 1)
    assert torch.equal(epochs[1], _openpi_sampling_indices(fresh))


def test_openpi_sft_eval_order_stays_fixed_across_epochs():
    from rlinf.data.datasets.common.epoch import forward_set_epoch

    _, loader = _make_openpi_sampling_loaders(shuffle=False)
    first = _openpi_sampling_indices(loader)
    forward_set_epoch(loader, 1)
    assert torch.equal(first, _openpi_sampling_indices(loader))


def test_openpi_sft_rejects_a_rank_without_complete_batches():
    with pytest.raises(ValueError, match="no complete batches on this rank"):
        _make_openpi_sampling_loaders(size=8, world_size=16)


@pytest.mark.parametrize("num_workers,batch_offset", [(0, 0), (0, 3), (0, 8), (2, 3)])
def test_openpi_sft_cursor_restores_inside_epoch_and_across_next_boundary(
    num_workers, batch_offset
):
    from rlinf.data.datasets.common.epoch import forward_set_epoch

    _, original = _make_openpi_sampling_loaders(size=515, num_workers=num_workers)
    for epoch in range(3):
        forward_set_epoch(original, epoch)
        _openpi_sampling_indices(original)
    forward_set_epoch(original, 3)
    iterator = iter(original)
    for _ in range(batch_offset):
        next(iterator)
    saved = original.checkpoint_state(3, batch_offset)
    expected = [obs.state.clone() for obs, _ in iterator]
    forward_set_epoch(original, 4)
    expected.extend(obs.state.clone() for obs, _ in original)

    _, restored = _make_openpi_sampling_loaders(size=515, num_workers=num_workers)
    iterator = restored.restore_checkpoint_state(saved)
    actual = [obs.state.clone() for obs, _ in iterator]
    forward_set_epoch(restored, 4)
    actual.extend(obs.state.clone() for obs, _ in restored)
    assert len(actual) == len(expected) == 16 - batch_offset
    for first, second in zip(actual, expected, strict=True):
        assert torch.equal(first, second)


def _make_openpi_checkpoint_worker(monkeypatch):
    from contextlib import nullcontext

    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    # Model/FSDP serialization is checked separately on the execution node.
    monkeypatch.setattr(
        FSDPSftWorker, "save_checkpoint", lambda *a, **kw: torch.rand(5)
    )
    monkeypatch.setattr(
        FSDPSftWorker, "load_checkpoint", lambda *a, **kw: torch.rand(7)
    )
    monkeypatch.setattr(
        "rlinf.workers.sft.fsdp_sft_worker.all_reduce_dict", lambda value, **kw: value
    )
    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create(
        {
            "data": {"train_data_paths": "indices-only", "checkpoint_data_state": True},
            "actor": {
                "seed": 42,
                "micro_batch_size": 8,
                "global_batch_size": 192,
                "model": {
                    "precision": "fp32",
                    "num_action_chunks": 50,
                    "openpi": {
                        "config_name": "test",
                        "action_chunk": "${actor.model.num_action_chunks}",
                    },
                },
            },
        }
    )
    worker._rank, worker._world_size = 0, 8
    _, worker.data_loader = _make_openpi_sampling_loaders(size=515)
    worker.data_iter = iter(worker.data_loader)
    worker._data_epoch = worker._data_iter_offset = 0
    worker.gradient_accumulation = 3
    worker.global_step = 0
    worker.model = Mock()
    worker.worker_timer = nullcontext
    worker.before_micro_batch = lambda *a, **kw: nullcontext()
    worker.grad_scaler = Mock()
    worker.optimizer_step = lambda: (0.0, [1e-4])
    worker.optimizer = Mock(param_groups=[{"lr": 1e-4}])
    worker.lr_scheduler = Mock()
    worker.observed_batches = []

    def receive(batch):
        worker.observed_batches.append(
            (batch[0].state.clone(), torch.rand(4), np.random.rand(4), random.random())
        )
        return torch.tensor(0.0), {"loss": 0.0}

    worker.get_train_model_output = receive
    return worker


def test_openpi_sft_worker_checkpoint_restores_consumed_batches_and_training_rng(
    tmp_path, monkeypatch
):
    tmp_path = tmp_path / "global_step_6" / "actor"
    tmp_path.mkdir(parents=True)
    original = _make_openpi_checkpoint_worker(monkeypatch)
    for _ in range(6):
        original.run_training()
    assert (original._data_epoch, original._data_iter_offset) == (2, 2)
    assert original._openpi_resume_contract()["model"]["openpi"]["action_chunk"] == 50
    original.save_checkpoint(str(tmp_path), step=6)
    assert (tmp_path / "openpi_data_rank_0.pt").is_file()
    original.observed_batches.clear()
    for _ in range(4):
        original.run_training()

    restored = _make_openpi_checkpoint_worker(monkeypatch)
    restored.load_checkpoint(str(tmp_path))
    assert (restored._data_epoch, restored._data_iter_offset) == (2, 2)
    for _ in range(4):
        restored.run_training()
    assert len(restored.observed_batches) == len(original.observed_batches) == 12
    for actual, expected in zip(
        restored.observed_batches, original.observed_batches, strict=True
    ):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        assert actual[3] == expected[3]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "batch",
        "offset",
        "rng",
        "rank",
        "cpu_rng",
        "numpy_rng",
        "python_rng",
        "loader_rng",
        "cursor_step",
        "path_step",
    ],
)
def test_openpi_sft_resume_rejects_missing_or_incompatible_state_before_model_load(
    tmp_path, monkeypatch, damage
):
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

    tmp_path = tmp_path / "global_step_1" / "actor"
    tmp_path.mkdir(parents=True)
    original = _make_openpi_checkpoint_worker(monkeypatch)
    original.run_training()
    original.save_checkpoint(str(tmp_path), step=1)
    restored = _make_openpi_checkpoint_worker(monkeypatch)
    path = tmp_path / "openpi_data_rank_0.pt"
    state = torch.load(path, weights_only=False)
    if damage == "missing":
        path.unlink()
    else:
        if damage == "batch":
            restored.cfg.actor.global_batch_size = 128
        elif damage == "offset":
            state["loader"]["batch_offset"] = 999
        elif damage == "rng":
            del state["rng"]["cpu"]
        elif damage == "rank":
            state["loader"]["contract"]["rank"] = 1
        elif damage == "cpu_rng":
            state["rng"]["cpu"] = torch.zeros(3, dtype=torch.uint8)
        elif damage == "numpy_rng":
            state["rng"]["numpy"] = ("invalid algorithm",)
        elif damage == "python_rng":
            state["rng"]["random"] = (3, (1,), None)
        elif damage == "loader_rng":
            state["loader"]["generator_state"] = torch.zeros(3, dtype=torch.uint8)
        elif damage == "cursor_step":
            state["loader"]["epoch"] = 2
        elif damage == "path_step":
            state["step"] = 2
            state["loader"]["batch_offset"] = 6
        torch.save(state, path)
    model_load = Mock()
    monkeypatch.setattr(FSDPSftWorker, "load_checkpoint", model_load)
    with pytest.raises((ValueError, FileNotFoundError)):
        restored.load_checkpoint(str(tmp_path))
    model_load.assert_not_called()


def test_openpi_sft_resume_rejects_another_ranks_error_before_model_load(
    tmp_path, monkeypatch
):
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

    tmp_path = tmp_path / "global_step_1" / "actor"
    tmp_path.mkdir(parents=True)
    worker = _make_openpi_checkpoint_worker(monkeypatch)
    worker.run_training()
    worker.save_checkpoint(str(tmp_path), step=1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather_errors(output, local_result):
        assert local_result == (None, (1, 0, 3))
        output[:] = [
            local_result,
            ("FileNotFoundError: missing rank 1 data state", None),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_errors)
    model_load = Mock()
    monkeypatch.setattr(FSDPSftWorker, "load_checkpoint", model_load)
    with pytest.raises(RuntimeError, match="rank 1: FileNotFoundError"):
        worker.load_checkpoint(str(tmp_path))
    model_load.assert_not_called()


def test_openpi_sft_checkpoint_rejects_cursor_step_mismatch_before_model_save(
    tmp_path, monkeypatch
):
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

    worker = _make_openpi_checkpoint_worker(monkeypatch)
    worker.run_training()
    model_save = Mock()
    monkeypatch.setattr(FSDPSftWorker, "save_checkpoint", model_save)
    with pytest.raises(ValueError, match="cursor and completed training step"):
        worker.save_checkpoint(str(tmp_path), step=2)
    model_save.assert_not_called()


def test_openpi_sft_resume_rejects_mismatched_rank_positions_before_model_load(
    tmp_path, monkeypatch
):
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

    tmp_path = tmp_path / "global_step_1" / "actor"
    tmp_path.mkdir(parents=True)
    worker = _make_openpi_checkpoint_worker(monkeypatch)
    worker.run_training()
    worker.save_checkpoint(str(tmp_path), step=1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather_positions(output, local_result):
        output[:] = [local_result, (None, (2, 0, 6))]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_positions)
    model_load = Mock()
    monkeypatch.setattr(FSDPSftWorker, "load_checkpoint", model_load)
    with pytest.raises(RuntimeError, match="differ across ranks"):
        worker.load_checkpoint(str(tmp_path))
    model_load.assert_not_called()


def test_openpi_sft_seeds_initialization_before_parent_init_when_enabled(monkeypatch):
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create({"actor": {"seed": 42, "seed_training_rng": True}})
    seen = []
    monkeypatch.setattr(
        FSDPSftWorker,
        "init_worker",
        lambda self: seen.append((torch.rand(4), np.random.rand(4), random.random())),
    )
    worker.init_worker()
    torch.rand(11)
    np.random.rand(11)
    random.random()
    worker.init_worker()
    assert torch.equal(seen[0][0], seen[1][0])
    np.testing.assert_array_equal(seen[0][1], seen[1][1])
    assert seen[0][2] == seen[1][2]
    worker.cfg.actor.seed_training_rng = False
    worker.init_worker()
    assert not torch.equal(seen[1][0], seen[2][0])


def test_piper_critical_replay_filters_prefix_and_preserves_epoch_terminals(
    monkeypatch,
):
    from rlinf.data.storage.replay import TrajectoryReplayBuffer
    from rlinf.workers.actor import fsdp_rlt_ac_policy_worker as rlt

    monkeypatch.setattr(rlt, "all_reduce_dict", lambda values, **kwargs: values)

    class Learner(rlt.RLTACReplayMixin):
        pass

    learner = Learner()
    learner.cfg = OmegaConf.create(
        {
            "algorithm": {"replay_buffer": {"critical_phase_only": True}},
            "env": {"train": {"env_type": "real", "auto_reset": True}},
        }
    )
    learner.replay_buffer = TrajectoryReplayBuffer(seed=123, sample_window_size=4)
    learner.demo_buffer = None
    values = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
    flags = torch.tensor([False, True, False, True]).reshape(4, 1, 1)
    # Two epochs, each with an initial slot followed by two action chunks.
    dones = torch.tensor([False, False, False, False, False, True]).reshape(6, 1, 1)
    trajectory = Trajectory(
        max_episode_length=4,
        actions=values.clone(),
        rewards=values.clone(),
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        curr_obs={"z_rl": values.clone()},
        next_obs={"z_rl": values + 10},
        forward_inputs={
            "record_transition": flags,
            "actor_switch": torch.zeros_like(flags),
        },
    )
    added, completed = learner._ingest_rollout_trajectories([trajectory])
    assert (added, completed) == (2, 1)
    assert learner.replay_buffer.total_samples == 2
    seen = set()
    for _ in range(10):
        batch = learner.replay_buffer.sample(2)
        seen.update(batch["actions"].flatten().tolist())
        assert batch["forward_inputs"]["record_transition"].all()
        assert not batch["forward_inputs"]["actor_switch"].any()
        torch.testing.assert_close(
            batch["next_obs"]["z_rl"], batch["curr_obs"]["z_rl"] + 10
        )
        assert torch.equal(batch["terminations"], batch["actions"] == 3)
    assert seen == {1.0, 3.0}
    assert trajectory.actions.shape[0] == 4
    trajectory.forward_inputs["record_transition"] = torch.zeros_like(flags)
    assert learner._ingest_rollout_trajectories([trajectory]) == (0, 0)
    assert learner.replay_buffer.total_samples == 2


def test_piper_critical_replay_rejects_missing_phase_labels():
    from rlinf.data.storage.replay import TrajectoryReplayBuffer
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin

    learner = RLTACReplayMixin()
    learner.cfg = OmegaConf.create(
        {
            "algorithm": {"replay_buffer": {"critical_phase_only": True}},
            "env": {"train": {"env_type": "real"}},
        }
    )
    learner.replay_buffer = TrajectoryReplayBuffer(sample_window_size=4)
    learner.demo_buffer = None
    trajectory = Trajectory(actions=torch.zeros(1, 1, 7), rewards=torch.zeros(1, 1, 1))
    with pytest.raises(ValueError, match="record_transition"):
        learner._ingest_rollout_trajectories([trajectory])
    assert learner.replay_buffer.total_samples == 0


def test_piper_warmup_chunks_keep_terminal_rewards_and_do_not_cross_phase():
    from rlinf.data.datasets.rlt_warmup import episode_transitions

    actions = np.arange(140, dtype=np.float32).reshape(20, 7)
    data = episode_transitions(
        actions,
        phases=[{"start": 4, "end": 17}],
        terminal=17,
        chunk_length=10,
        stride=10,
        success=True,
        human=np.zeros(20, dtype=bool),
    )
    np.testing.assert_array_equal(data["frames"], [4, 14, 17])
    np.testing.assert_array_equal(data["valid_steps"], [10, 3])
    np.testing.assert_array_equal(data["terminated"], [False, True])
    np.testing.assert_array_equal(data["actions"][1, :3], actions[14:17])
    assert not data["actions"][1, 3:].any()
    assert data["rewards"].sum() == data["rewards"][1] == 1
    with pytest.raises(ValueError, match="terminal observation"):
        episode_transitions(
            actions,
            phases=[{"start": 4, "end": 20}],
            terminal=20,
            chunk_length=10,
            stride=10,
            success=True,
            human=np.zeros(20, dtype=bool),
        )


def test_piper_warmup_separate_phases_keep_action_windows_and_reference_gaps():
    from rlinf.data.datasets.rlt_warmup import episode_transitions

    actions = np.arange(35 * 7, dtype=np.float32).reshape(35, 7)
    data = episode_transitions(
        actions,
        phases=[{"start": 2, "end": 17}, {"start": 24, "end": 27}],
        terminal=30,
        chunk_length=10,
        stride=10,
        success=True,
        human=np.zeros(35, dtype=bool),
    )
    for key, expected in {
        "current": [2, 12, 24],
        "action_end": [12, 17, 27],
        "next": [12, 24, 30],
    }.items():
        np.testing.assert_array_equal(data["frames"][data[key]], expected)
    for key, expected in {
        "valid_steps": [10, 5, 3],
        "discount_steps": [10, 12, 6],
        "next_valid_steps": [5, 3, 0],
        "phase_id": [0, 0, 1],
        "terminated": [False, False, True],
        "rewards": [0, 0, 1],
    }.items():
        np.testing.assert_array_equal(data[key], expected)
    for row, (start, end) in enumerate(((2, 12), (12, 17), (24, 27))):
        np.testing.assert_array_equal(
            data["actions"][row, : end - start], actions[start:end]
        )
        assert not data["actions"][row, end - start :].any()


@pytest.mark.parametrize(
    "phases",
    [
        [],
        [{"start": 1, "end": None}],
        [{"start": 2, "end": 2}],
        [{"start": -1, "end": 5}],
        [{"start": 2, "end": 14}],
        [{"start": 1, "end": 5}, {"start": 4, "end": 8}],
        [{"start": True, "end": 5}],
    ],
)
def test_piper_warmup_rejects_incomplete_or_overlapping_phase_labels(
    piper_warmup_cache,
    tmp_path,
    phases,
):
    from rlinf.data.datasets.rlt_warmup import load_episode_annotations

    labels = json.loads((piper_warmup_cache / "manifest.json").read_text())[
        "annotations"
    ]
    labels["episodes"][0]["critical_phases"] = phases
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="critical phase|Critical phase"):
        load_episode_annotations(path)


def test_piper_warmup_legacy_annotations_preserve_the_explicit_old_interval(
    piper_warmup_cache,
    tmp_path,
):
    from rlinf.data.datasets.rlt_warmup import LABEL_FORMAT, load_episode_annotations

    labels = json.loads((piper_warmup_cache / "manifest.json").read_text())[
        "annotations"
    ]
    labels["format"] = "piper-rlt-episodes-v1"
    for record in labels["episodes"]:
        record.pop("critical_phases")
        record["critical_start"] = 1
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(labels))
    loaded = load_episode_annotations(path)
    assert loaded["format"] == LABEL_FORMAT
    for record in loaded["episodes"]:
        assert record["critical_phases"] == [
            {"start": 1, "end": record["terminal_frame"]}
        ]


@pytest.mark.parametrize(
    "column, message",
    [
        ("discount_steps", "actual reference-controlled ticks"),
        ("next_valid_steps", "next action window crosses phase boundaries"),
    ],
)
def test_piper_warmup_rejects_inconsistent_phase_timing_in_cache(
    piper_warmup_cache,
    column,
    message,
):
    from rlinf.data.datasets.rlt_warmup import (
        CachedRLTReplay,
        file_sha256,
        json_fingerprint,
    )

    path = piper_warmup_cache / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["episodes"][0]
    shard = piper_warmup_cache / entry["file"]
    with np.load(shard) as original:
        arrays = {key: original[key].copy() for key in original.files}
    arrays[column][0] += 1
    np.savez(shard, **arrays)
    entry["sha256"] = file_sha256(shard)
    manifest.pop("content_fingerprint")
    manifest["content_fingerprint"] = json_fingerprint(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        CachedRLTReplay(piper_warmup_cache)


def test_piper_warmup_cache_only_samples_training_and_restores_rng(piper_warmup_cache):
    from rlinf.data.datasets.rlt_warmup import CachedRLTReplay

    replay = CachedRLTReplay(piper_warmup_cache)
    state = replay.rng_state()
    first = replay.sample(4096)
    # Each episode's feature identity is known; validation identities are 0.2/0.3.
    identities = first["curr_obs"]["z_rl"][:, 0]
    assert identities.min() >= 0 and identities.max() < 0.2
    assert 0.45 < float((identities == 0).float().mean()) < 0.55
    replay.restore_rng(state)
    repeated = replay.sample(4096)
    for key in first["curr_obs"]:
        torch.testing.assert_close(
            first["curr_obs"][key], repeated["curr_obs"][key], rtol=0, atol=0
        )
    assert (
        replay.counts("validation")["success"]
        == replay.counts("validation")["failure"]
        == 1
    )


def test_piper_warmup_rejects_changed_shards_and_leaking_annotations(
    piper_warmup_cache, tmp_path
):
    from rlinf.data.datasets.rlt_warmup import CachedRLTReplay, load_episode_annotations

    manifest = json.loads((piper_warmup_cache / "manifest.json").read_text())
    labels = manifest["annotations"]
    labels["episodes"][0]["session"] = labels["episodes"][2]["session"] = "same scene"
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="crosses train/validation"):
        load_episode_annotations(path)
    shard = piper_warmup_cache / manifest["episodes"][0]["file"]
    with shard.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="shard changed"):
        CachedRLTReplay(piper_warmup_cache)


def test_piper_warmup_collection_split_rejects_missing_labels_and_keeps_sessions(
    tmp_path,
):
    from rlinf.data.datasets.rlt_warmup import (
        LABEL_FORMAT,
        load_episode_annotations,
        split_collection_annotations,
    )

    labels = {
        "format": LABEL_FORMAT,
        "task": "USB",
        "phase_definition": "approach to insertion",
        "datasets": {"piper": "dataset"},
        "episodes": [],
    }
    for index in range(20):
        labels["episodes"].append(
            {
                "dataset": "piper",
                "episode": index,
                "critical_phases": [{"start": 1, "end": 5}, {"start": 10, "end": 15}],
                "terminal_frame": 20,
                "outcome": "success" if index % 2 else "failure",
                "behavior": "reference",
                "session": f"scene{index // 2}",
                "split": None,
            }
        )
    source = tmp_path / "collected.json"
    source.write_text(json.dumps(labels))
    output = split_collection_annotations([source], tmp_path / "labels.json")
    result = load_episode_annotations(output)
    assert len(result["episodes"]) == 20
    assert all(len(r["critical_phases"]) == 2 for r in result["episodes"])
    assert {r["split"] for r in result["episodes"]} == {"train", "validation"}
    labels["episodes"][0]["critical_phases"][0]["end"] = None
    source.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="Unpaired critical phase"):
        split_collection_annotations([source], tmp_path / "invalid.json")


@pytest.mark.parametrize("variant", ["v3", "local_outcome_in_v2", "local_scope_in_v2"])
def test_piper_warmup_refuses_local_rewards_in_global_episode_pipeline(
    tmp_path, variant
):
    from rlinf.data.datasets.rlt_warmup import (
        load_episode_annotations,
        split_collection_annotations,
    )

    labels = {
        "format": "piper-rlt-episodes-v3"
        if variant == "v3"
        else "piper-rlt-episodes-v2",
        "episodes": [{"critical_phases": [{"start": 1, "end": 2}]}],
    }
    if variant == "local_outcome_in_v2":
        labels["episodes"][0]["critical_phases"][0]["outcome"] = "success"
    if variant == "local_scope_in_v2":
        labels["reward_scope"] = "critical_segment"
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="Local critical-segment"):
        load_episode_annotations(path)
    with pytest.raises(ValueError, match="Local critical-segment"):
        split_collection_annotations([path], tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("location", ["annotations", "shard_entry"])
def test_piper_warmup_cache_refuses_local_labels_even_with_valid_fingerprint(
    piper_warmup_cache, location
):
    from rlinf.data.datasets.rlt_warmup import CachedRLTReplay, json_fingerprint

    path = piper_warmup_cache / "manifest.json"
    manifest = json.loads(path.read_text())
    records = (
        manifest["annotations"]["episodes"]
        if location == "annotations"
        else manifest["episodes"]
    )
    records[0]["critical_phases"][0]["outcome"] = "failure"
    content = {k: v for k, v in manifest.items() if k != "content_fingerprint"}
    manifest["content_fingerprint"] = json_fingerprint(content)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="segment-terminal trainer"):
        CachedRLTReplay(piper_warmup_cache)


@pytest.mark.parametrize("second_start", [4, 7])
@pytest.mark.parametrize("stride", [2, 10])
def test_piper_warmup_local_targets_end_each_scored_segment(second_start, stride):
    from rlinf.data.datasets.rlt_warmup import episode_transitions
    from rlinf.runners.rlt_warmup_runner import discounted_target

    phases = [
        {"start": 1, "end": 4, "outcome": "success"},
        {"start": second_start, "end": second_start + 13, "outcome": "failure"},
    ]
    end = second_start + 17
    data = episode_transitions(
        np.ones((end + 1, 7), dtype=np.float32),
        phases=phases,
        terminal=end,
        chunk_length=10,
        stride=stride,
        success=False,
        human=np.ones(end + 1, dtype=bool),
        reward_scope="critical_segment",
    )
    np.testing.assert_array_equal(data["next"], data["action_end"])
    np.testing.assert_array_equal(data["discount_steps"], data["valid_steps"])
    ends = np.array([p["end"] for p in phases])[data["phase_id"]]
    np.testing.assert_array_equal(
        data["terminated"], data["frames"][data["next"]] == ends
    )
    assert (data["next_valid_steps"][data["terminated"]] == 0).all()
    np.testing.assert_array_equal(
        data["rewards"],
        (data["terminated"] & (data["phase_id"] == 0)).astype(np.float32),
    )
    batch = {key: torch.from_numpy(value) for key, value in data.items()}
    target = discounted_target(batch, torch.full_like(batch["rewards"], 123.0), 0.5)
    done = batch["terminated"]
    torch.testing.assert_close(
        target[done], batch["rewards"][done] * 0.5 ** (batch["valid_steps"][done] - 1)
    )
    mask = np.arange(10)[None] < data["valid_steps"][:, None]
    assert not data["human"][~mask].any()
    assert not data["actions"][~mask].any()


def test_piper_warmup_local_phase_filter_preserves_parent_splits_and_rng(
    piper_local_warmup_cache,
):
    from rlinf.data.datasets.rlt_warmup import CachedRLTReplay

    replay = CachedRLTReplay(piper_local_warmup_cache, phase_filter=1)
    assert replay.splits == {"train": [0], "validation": [2]}
    assert replay.metric_groups("train") == [(0, 1)]
    state = replay.rng_state()
    batch = replay.sample(100)
    replay.restore_rng(state)
    repeated = replay.sample(100)
    assert not batch["rewards"].any()
    assert batch["terminated"].all()
    assert (batch["curr_obs"]["z_rl"] == 0).all()
    torch.testing.assert_close(batch["actions"], repeated["actions"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="disjoint train and validation"):
        CachedRLTReplay(piper_local_warmup_cache, phase_filter=4)


@pytest.mark.parametrize("missing", ["interval", "outcome"])
def test_piper_warmup_local_split_explicitly_excludes_unlabeled(
    piper_local_warmup_cache, tmp_path, missing
):
    from rlinf.data.datasets.rlt_warmup import (
        load_episode_annotations,
        split_collection_annotations,
    )

    labels = json.loads((piper_local_warmup_cache / "manifest.json").read_text())[
        "annotations"
    ]
    if missing == "interval":
        labels["episodes"][0]["critical_phases"] = []
    else:
        labels["episodes"][0]["critical_phases"][0].pop("outcome")
    source = tmp_path / "local.json"
    source.write_text(json.dumps(labels))
    with pytest.raises(ValueError):
        load_episode_annotations(source)
    with pytest.raises(ValueError):
        split_collection_annotations([source], tmp_path / "strict.json")
    result = split_collection_annotations(
        [source], tmp_path / "excluded.json", skip_unlabeled=True
    )
    kept = load_episode_annotations(result)
    assert len(kept["episodes"]) == 3
    assert len(kept["excluded_episodes"]) == 1
    assert kept["excluded_episodes"][0]["episode"] == 0


@pytest.mark.parametrize("field", ["next", "rewards", "terminated", "discount_steps"])
def test_piper_warmup_local_cache_rejects_global_edges_and_rewards(
    piper_local_warmup_cache, field
):
    from rlinf.data.datasets.rlt_warmup import (
        CachedRLTReplay,
        file_sha256,
        json_fingerprint,
    )

    path = piper_local_warmup_cache / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["episodes"][0]
    shard = piper_local_warmup_cache / entry["file"]
    with np.load(shard, allow_pickle=False) as src:
        data = {k: src[k].copy() for k in src.files}
    data[field][0] = data[field][0] + 1 if field in ("next", "discount_steps") else 0
    with shard.open("wb") as f:
        np.savez(f, **data)
    entry["sha256"] = file_sha256(shard)
    manifest.pop("content_fingerprint")
    manifest["content_fingerprint"] = json_fingerprint(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        CachedRLTReplay(piper_local_warmup_cache)
