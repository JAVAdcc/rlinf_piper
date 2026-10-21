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

"""GPU integration check using recorded images; never opens hardware.

Synthetic local outcomes exercise success/failure terminals and partial chunks.
These labels and trained test weights are not task-performance measurements.
Run once normally, then twice with --resume to verify deterministic continuation.
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.runners.piper_online_runner import PiperOnlineRunner, online_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(exist_ok=args.resume, parents=True)
    task = "pick up the two USB cables and insert them into the USB port on the outlet."
    cfg = online_config(
        repo / "examples/embodiment/config/piper_rlt_online.yaml",
        stage1=args.stage1,
        actor=args.actor,
        hardware=OmegaConf.to_container(
            OmegaConf.load(repo / "examples/embodiment/config/piper_usb_hardware.yaml")
        ),
        task=task,
        output=output,
        allow_offline_actor=True,
    )
    cfg.algorithm.update_epoch = 4
    cfg.actor.micro_batch_size = cfg.actor.global_batch_size = 4
    OmegaConf.save(cfg, output / "online.yaml")
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

        source = args.source_episode.resolve()
        dataset = LeRobotDataset(
            "piper-smoke",
            root=source / "dataset",
            download_videos=False,
            video_backend="torchcodec",
        )
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        sample = dataset[0]
        assert runner.policy.infer(sample, task, actor_enabled=False).shape == (50, 7)
        assert runner.policy.infer(sample, task, actor_enabled=True).shape == (10, 7)
        episode = output / "episode_001"
        episode.mkdir()
        (episode / "dataset").symlink_to(source / "dataset", target_is_directory=True)
        result = json.loads((source / "result.json").read_text())
        result.update(
            accepted=True,
            policy_version=0,
            directory=episode.name,
            test_only=True,
            critical_phases=[
                {"start": 0, "end": 3, "outcome": "success"},
                {"start": 5, "end": 9, "outcome": "failure"},
            ],
        )
        (episode / "result.json").write_text(json.dumps(result))
        runner.accept_episode(episode, result)
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
