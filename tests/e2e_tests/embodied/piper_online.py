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

"""GPU integration check using synthetic or recorded images; never opens hardware.

Synthetic local outcomes exercise success/failure terminals and partial chunks.
These labels and trained test weights are not task-performance measurements.
Run once normally, then twice with --resume to verify deterministic continuation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.runners.piper_online_runner import PiperOnlineRunner, online_config


def synthetic_episode(directory: Path, task: str, chunk: int) -> None:
    """Create fresh control-tick data without using any previous robot rollout."""
    from rlinf.data.storage.lerobot.piper import PiperEpisodeRecorder

    directory.mkdir()
    observation = {
        "observation.state": np.array([0, 30, -40, 0, 10, 0, 35], dtype=np.float32)
    }
    for camera in ("front", "front_2", "wrist"):
        observation[f"observation.images.{camera}"] = np.zeros((240, 320, 3), np.uint8)
    recorder = PiperEpisodeRecorder(directory / "dataset", fps=30, task=task)
    try:
        for step in range(chunk + 14):
            recorder.add_step(
                observation,
                observation["observation.state"],
                requested=observation["observation.state"],
                measured=observation["observation.state"],
                critical=True,
                wall_time=step / 30,
                is_intervention=step % 3 == 0,
                action_source="human" if step % 3 == 0 else "actor",
            )
        recording = recorder.finish(observation, critical=False, outcome="success")
    finally:
        recorder.close()
    (directory / "result.json").write_text(
        json.dumps({"recording": recording, "outcome": "success"})
    )


def synthetic_cache(path: Path, task: str, stage1: Path, actor: Path) -> None:
    """Use a small labelled cache to exercise mixing and validation exclusion."""
    from rlinf.data.datasets.rlt_warmup import (
        cache_episode,
        episode_transitions,
        json_fingerprint,
        write_json,
    )
    from rlinf.models.embodiment.piper_inference import load_piper_inference_config

    cfg, manifest = load_piper_inference_config(stage1)
    metadata = json.loads((actor / "piper_rlt_actor.json").read_text())
    chunk = metadata["model"]["num_action_chunks"]
    path.mkdir()
    (path / "norm_stats.json").write_bytes(
        Path(cfg.openpi_data.norm_stats_path).read_bytes()
    )
    labels = {
        "format": "piper-rlt-episodes-v3",
        "reward_scope": "critical_segment",
        "scoring_protocol": "critical-segment-v1",
        "task": task,
        "datasets": {"synthetic": "test-only"},
        "episodes": [],
    }
    entries = []
    for index in range(4):
        record = {
            "dataset": "synthetic",
            "episode": index,
            "critical_phases": [
                {
                    "start": 0,
                    "end": chunk + 5,
                    "outcome": "success" if index % 2 == 0 else "failure",
                }
            ],
            "terminal_frame": chunk + 5,
            "outcome": "success" if index % 2 == 0 else "failure",
            "behavior": "human",
            "split": "train" if index < 2 else "validation",
            "session": None,
        }
        labels["episodes"].append(record)
        transitions = episode_transitions(
            np.zeros((chunk + 6, 7), np.float32),
            phases=record["critical_phases"],
            terminal=chunk + 5,
            chunk_length=chunk,
            stride=2,
            success=index % 2 == 0,
            human=np.ones(chunk + 6, bool),
            reward_scope="critical_segment",
            tail_policy="align",
        )
        shard = path / f"episode_{index:06d}.npz"
        digest = cache_episode(
            shard,
            transitions,
            lambda frame: {
                "z_rl": np.zeros(2048, np.float32),
                "proprio": np.zeros(7, np.float32),
                "ref_chunk": np.zeros((50, 7), np.float32),
                "raw_state": np.array([0, 30, -40, 0, 10, 0, 35], np.float32),
            },
        )
        entries.append(
            {
                **record,
                "file": shard.name,
                "sha256": digest,
                "transitions": len(transitions["current"]),
                "states": len(transitions["frames"]),
            }
        )
    content = {
        "format": "piper-rlt-cache-v3",
        "annotations": labels,
        "stage1": manifest,
        "model": metadata["model"],
        "chunk_length": chunk,
        "stride": 2,
        "tail_policy": "align",
        "episodes": entries,
    }
    write_json(
        path / "manifest.json",
        {**content, "content_fingerprint": json_fingerprint(content)},
    )
    from rlinf.data.datasets.rlt_warmup import CachedRLTReplay

    CachedRLTReplay(path)


def asynchronous_check(cfg, args, output, task):
    """Exercise concurrent real GPU inference/updates and pending-budget recovery."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch

    source = output / "synthetic_episode"
    episode = output / "episode_001"
    if not args.resume:
        synthetic_episode(source, task, int(cfg.actor.model.num_action_chunks))
        episode.mkdir()
        (episode / "dataset").symlink_to(source / "dataset", target_is_directory=True)
    dataset = LeRobotDataset(
        "piper-smoke",
        root=source / "dataset",
        download_videos=False,
        video_backend="torchcodec",
    )
    dataset.hf_dataset.set_transform(decode_image_struct_batch)
    sample = dataset[0]
    calls = []
    with PiperOnlineRunner(cfg, output, resume=args.resume) as runner:
        if not args.resume:
            runner.policy.infer_for_replay(
                sample, task, actor_enabled=True, directory=episode, frame=0
            )
            chunk = int(cfg.actor.model.num_action_chunks)
            result = json.loads((source / "result.json").read_text())
            result.update(
                accepted=True,
                policy_version=0,
                directory=episode.name,
                test_only=True,
                critical_phases=[
                    {"start": 0, "end": chunk + 5, "outcome": "success"},
                    {"start": chunk + 7, "end": chunk + 11, "outcome": "failure"},
                ],
            )
            (episode / "result.json").write_text(json.dumps(result))
        result = json.loads((episode / "result.json").read_text())
        started = time.monotonic()
        runner.accept_episode(episode, result)
        acceptance_s = time.monotonic() - started
        assert acceptance_s < 1.0, acceptance_s
        deadline = time.monotonic() + 240
        while True:
            runner.check_health()
            # A new synthetic episode adopts the latest published snapshot.
            fixed_version = runner.policy.begin_episode()
            began = time.time()
            actions, diagnostics = runner.policy.infer_with_reference(
                sample, task, actor_enabled=True
            )
            assert diagnostics["policy_version"] == fixed_version
            # A subsequent chunk in the same episode must keep its version even
            # while the learner publishes checkpoints.
            _, next_diagnostics = runner.policy.infer_with_reference(
                sample, task, actor_enabled=True
            )
            assert next_diagnostics["policy_version"] == fixed_version
            calls.append(
                {
                    "start": began,
                    "end": time.time(),
                    "version": diagnostics["policy_version"],
                }
            )
            assert actions.shape == (50, 7) and np.isfinite(actions).all()
            latest = json.loads((output / "online/latest.json").read_text())
            progress = output / "online/async_status.json"
            if not args.resume and progress.exists():
                live = json.loads(progress.read_text())
                if live["version"] > 0:
                    # Training progresses without writing intermediate checkpoints.
                    assert latest["version"] == 0 and latest["generation"] == 1
                    break
            if args.resume and latest["metrics"].get("rlt/pending_update_budget") == 0:
                runner.background.wait_idle()
                break
            assert time.monotonic() < deadline, "Async learner did not progress"
        if args.resume:
            assert runner.committed == ["episode_001"]
    latest = json.loads((output / "online/latest.json").read_text())
    metrics = [
        json.loads(line)
        for line in (output / "online/async_metrics.jsonl").read_text().splitlines()
    ]
    overlaps = sum(
        any(
            c["start"] < m["learner_finished_at"] and c["end"] > m["learner_started_at"]
            for m in metrics
        )
        for c in calls
    )
    assert overlaps > 0, "No inference request overlapped a learner update RPC"
    assert latest["committed"] == ["episode_001"]
    assert latest["metrics"]["demo_buffer/total_samples"] == 8
    assert latest["generation"] == (3 if args.resume else 2), latest
    if args.resume:
        assert latest["version"] == 200, latest
        assert latest["metrics"]["rlt/pending_update_budget"] == 0
    else:
        assert latest["metrics"]["rlt/pending_update_budget"] > 0
    stats = json.loads((episode / "online_features.json").read_text())["feature_stats"]
    assert stats["reused_anchors"] == 1 and stats["backfilled_anchors"] > 0
    report = {
        "acceptance_s": acceptance_s,
        "calls": calls,
        "overlapping_calls": overlaps,
        "version": latest["version"],
        "pending": latest["metrics"]["rlt/pending_update_budget"],
    }
    (output / ("async_resume.json" if args.resume else "async_first.json")).write_text(
        json.dumps(report, indent=2)
    )
    print("Async inference, snapshot installation and durable budget passed:", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument(
        "--source-episode", type=Path, help="Omit to generate synthetic observations"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(exist_ok=args.resume, parents=True)
    task = "pick up the two USB cables and insert them into the USB port on the outlet."
    profile = OmegaConf.load(repo / "examples/embodiment/config/piper_rlt_online.yaml")
    # This is an isolated optimizer/shape test, never a deployable training run.
    profile.piper.async_training = args.asynchronous
    profile.piper.require_critic_warmup = False
    profile.algorithm.rlt_schedule.warmup_min_size = 1
    profile.algorithm.rlt_schedule.max_updates_per_train_step = 4
    profile_path = output / "test_profile.yaml"
    OmegaConf.save(profile, profile_path)
    cache = output / "synthetic_cache"
    if not args.resume:
        synthetic_cache(cache, task, args.stage1, args.actor)
    cfg = online_config(
        output / "online.yaml" if args.resume else profile_path,
        stage1=args.stage1,
        actor=args.actor,
        hardware=OmegaConf.to_container(
            OmegaConf.load(repo / "examples/embodiment/config/piper_usb_hardware.yaml")
        ),
        task=task,
        output=output,
        allow_offline_actor=True,
        offline_cache=cache,
        resume=args.resume,
    )
    cfg.algorithm.update_epoch = 40 if args.asynchronous else 4
    cfg.actor.micro_batch_size = cfg.actor.global_batch_size = 4
    OmegaConf.save(cfg, output / "online.yaml")
    if args.asynchronous:
        asynchronous_check(cfg, args, output, task)
        return
    with PiperOnlineRunner(cfg, output, resume=args.resume) as runner:
        initial = runner.rollout.status().wait()[0]
        if args.resume:
            expected = json.loads((output / "expected.json").read_text())
            assert initial == expected, (initial, expected)
            assert runner.committed == ["episode_001"]
            runner.accept_episode(
                output / "episode_001",
                json.loads((output / "episode_001/result.json").read_text()),
            )
            assert runner.rollout.status().wait()[0] == expected
            # Leave this next update uncommitted. A second invocation must reproduce
            # it from the same optimizer, target, replay and RNG checkpoint.
            runner.actor.run_training().wait()
            receive = runner.rollout.sync_model_from_actor()
            runner.actor.sync_model_to_rollout().wait()
            continued = receive.wait()[0]
            path = output / "expected_continuation.json"
            if path.exists():
                assert continued == json.loads(path.read_text())
            else:
                path.write_text(json.dumps(continued))
            assert continued["version"] == 8
            print(
                "Resume, duplicate-accept protection and continuation passed:",
                continued,
            )
            return

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch

        chunk = int(cfg.actor.model.num_action_chunks)
        source = (
            args.source_episode.resolve()
            if args.source_episode
            else output / "synthetic_episode"
        )
        if args.source_episode is None:
            synthetic_episode(source, task, chunk)
        dataset = LeRobotDataset(
            "piper-smoke",
            root=source / "dataset",
            download_videos=False,
            video_backend="torchcodec",
        )
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        sample = dataset[0]
        assert runner.policy.infer(sample, task, actor_enabled=False).shape == (50, 7)
        assert runner.policy.infer(sample, task, actor_enabled=True).shape == (chunk, 7)
        episode = output / "episode_001"
        episode.mkdir()
        (episode / "dataset").symlink_to(source / "dataset", target_is_directory=True)
        runner.policy.infer_for_replay(
            sample, task, actor_enabled=True, directory=episode, frame=0
        )
        result = json.loads((source / "result.json").read_text())
        result.update(
            accepted=True,
            policy_version=0,
            directory=episode.name,
            test_only=True,
            critical_phases=[
                {"start": 0, "end": chunk + 5, "outcome": "success"},
                {"start": chunk + 7, "end": chunk + 11, "outcome": "failure"},
            ],
        )
        (episode / "result.json").write_text(json.dumps(result))
        runner.accept_episode(episode, result)
        cache_stats = json.loads((episode / "online_features.json").read_text())[
            "feature_stats"
        ]
        assert cache_stats["reused_anchors"] == 1
        assert cache_stats["backfilled_anchors"] > 0
        metrics = json.loads((episode / "online_update.json").read_text())
        assert (
            metrics["rlt/critic_updates_run"] == 4
            and metrics["rlt/actor_updates_run"] == 2
        )
        assert metrics["rlt/pending_update_budget"] > 0
        assert metrics["demo_buffer/total_samples"] == 8
        final = runner.rollout.status().wait()[0]
        assert (
            final["version"] == 4
            and final["weights_sha256"] != initial["weights_sha256"]
        )
        (output / "expected.json").write_text(json.dumps(final))
        root = output / "online"
        before = torch.load(
            root / "checkpoint_000001/actor/model_state_dict/full_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        after = torch.load(
            root / "checkpoint_000002/actor/model_state_dict/full_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        for prefix in ("backbone.", "actor_mean.", "q_head."):
            assert any(
                not torch.equal(before[key], after[key])
                for key in before
                if key.startswith(prefix)
            ), prefix
        print(
            "Recorded features, native replay, actor/critic updates and weight sync passed:",
            final,
        )


if __name__ == "__main__":
    main()
