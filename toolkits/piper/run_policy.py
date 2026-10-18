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

"""Infer on recorded Piper observations or execute a bounded local rollout."""

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.piper_inference import PiperCFGPolicy, PiperRLTPolicy


def main() -> None:
    """Load a downloaded checkpoint before opening any robot or camera."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("infer", "rollout"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage 1 actor directory, or CFG actor directory",
    )
    parser.add_argument("--backend", choices=("rlt", "cfg"), default="rlt")
    parser.add_argument(
        "--actor", type=Path, help="Optional RLT Stage 2 actor checkpoint"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--task", help="Training task text; infer defaults to the dataset task"
    )
    parser.add_argument("--dataset", type=Path, help="Local LeRobot dataset for infer")
    parser.add_argument(
        "--frame", type=int, default=0, help="Global dataset sample index"
    )
    parser.add_argument("--hardware", type=Path, help="Piper hardware YAML for rollout")
    parser.add_argument(
        "--teleop",
        choices=("none", "piper_leader"),
        help="Optional human takeover device",
    )
    parser.add_argument(
        "--leader-usb-serial",
        help="Leader USB-CAN serial; requires --teleop piper_leader",
    )
    parser.add_argument(
        "--intervention-key",
        choices=("i", "o"),
        help="Toggle human takeover, independently of b",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Right starts/confirms completion; s/f scores; Left discards the last stopped episode",
    )
    parser.add_argument(
        "--reset-after-episode",
        choices=("none", "initial", "zero"),
        default="none",
        help="After s/f: open gripper, return to the first start pose or zero, restore gripper",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of scored episodes to retain; discarded attempts do not count",
    )
    parser.add_argument(
        "--record-dataset",
        action="store_true",
        help="Record every RGB/state/command plus terminal observation; b alternates critical phase start/end",
    )
    parser.add_argument(
        "--streaming-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode RGB during collection; --no-streaming-video uses post-episode PNG encoding",
    )
    parser.add_argument(
        "--phase-definition",
        default="Each pair of b presses marks a critical interval; s/f ends the task.",
        help="Shared annotation rule for these trials",
    )
    parser.add_argument(
        "--session-id", help="Optional scene/session group kept within one warmup split"
    )
    parser.add_argument("--reset-duration-s", type=float, default=3.0)
    parser.add_argument("--reset-gripper-duration-s", type=float, default=0.5)
    parser.add_argument(
        "--replan-steps",
        type=int,
        help="Targets to execute before replanning; defaults to the policy's full horizon",
    )
    parser.add_argument(
        "--save-observations",
        action="store_true",
        help="Save each prediction's RGB/state/task as an NPZ for offline diagnosis",
    )
    parser.add_argument(
        "--play-sounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speak interactive collection events through local Speech Dispatcher",
    )
    parser.add_argument(
        "--speech-language",
        choices=("en", "zh"),
        default="en",
        help="Language of interactive voice prompts (default: en, as in Evo-RL)",
    )
    parser.add_argument(
        "--no-action-limits",
        action="store_true",
        help="Disable extra rollout action bounds/step checks; driver travel limits remain",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; defaults to a unique outputs/piper run",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from rlinf.robotics.piper_reset import PiperResetTiming

    try:
        reset_timing = PiperResetTiming(
            args.reset_duration_s, args.reset_gripper_duration_s
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.reset_after_episode != "none" and not args.interactive:
        parser.error("--reset-after-episode requires --interactive.")
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive.")
    if args.num_episodes != 1 and not args.interactive:
        parser.error("Multiple episodes require --interactive.")
    if args.interactive and args.command != "rollout":
        parser.error("--interactive only applies to rollout.")
    if args.backend == "cfg" and args.actor is not None:
        parser.error("--actor is only used by the rlt backend.")
    if args.record_dataset and (not args.interactive or args.command != "rollout"):
        parser.error("--record-dataset requires interactive rollout.")
    if args.record_dataset and args.backend != "rlt":
        parser.error("The warmup data recorder requires the RLT backend.")
    if any(
        value is not None
        for value in (args.teleop, args.leader_usb_serial, args.intervention_key)
    ):
        if args.command != "rollout" or not args.interactive:
            parser.error("Teleop requires interactive rollout.")
    if args.leader_usb_serial and args.teleop != "piper_leader":
        parser.error("--leader-usb-serial requires --teleop piper_leader.")
    if args.command == "infer" and args.dataset is None:
        parser.error("infer requires --dataset.")
    if args.command == "infer" and args.no_action_limits:
        parser.error("--no-action-limits only applies to rollout.")
    if args.command == "infer" and args.save_observations:
        parser.error(
            "infer already saves its observation; --save-observations is for rollout."
        )
    if args.command == "rollout" and (args.hardware is None or not args.task):
        parser.error("rollout requires --hardware and --task.")
    if (
        args.steps <= 0
        or (args.replan_steps is not None and args.replan_steps <= 0)
        or args.frame < 0
    ):
        parser.error("Use positive steps/replan-steps and a nonnegative frame.")
    output = args.output_dir or Path("outputs/piper") / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Results: {output.resolve()}", flush=True)
    if args.command == "rollout":
        from rlinf.robotics.piper_runtime import build_piper_session

        hardware = OmegaConf.load(args.hardware)
        if args.teleop is not None:
            if "teleop" not in hardware:
                hardware.teleop = {}
            hardware.teleop.type = args.teleop
        if args.leader_usb_serial is not None:
            hardware.teleop.usb_serial = args.leader_usb_serial
        if args.intervention_key is not None:
            if hardware.get("teleop", {}).get("type") != "piper_leader":
                parser.error("--intervention-key requires a configured Piper leader.")
            if "intervention" not in hardware.teleop:
                hardware.teleop.intervention = {}
            hardware.teleop.intervention.key = args.intervention_key
        if (
            hardware.get("teleop", {}).get("type", "none") != "none"
            and not args.interactive
        ):
            parser.error("Configured teleop requires --interactive.")
        if args.no_action_limits:
            hardware.execution.check_action_limits = False
        OmegaConf.save(hardware, output / "hardware.yaml")
        session = build_piper_session(output / "hardware.yaml")
        if session.intervention is not None:
            import importlib.util

            for module in ("piper_sdk", "pinocchio"):
                if importlib.util.find_spec(module) is None:
                    parser.error(
                        f"Missing {module}; install requirements/embodied/piper-hil.txt."
                    )
    torch.manual_seed(args.seed)
    policy = (
        PiperRLTPolicy(args.checkpoint, actor=args.actor, device=args.device)
        if args.backend == "rlt"
        else PiperCFGPolicy(args.checkpoint, device=args.device)
    )
    (output / "policy.json").write_text(json.dumps(policy.manifest, indent=2) + "\n")
    if args.command == "infer":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch

        root = args.dataset.expanduser().resolve()
        dataset = LeRobotDataset(root.name, root=root, download_videos=False)
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        sample = dataset[args.frame]
        task = args.task or sample["task"]
        started = time.monotonic()
        actions = policy.infer(sample, task)
        result = {
            "task": task,
            "frame": args.frame,
            "action_units": "degrees_mm",
            "actions": actions.tolist(),
            "inference_s": time.monotonic() - started,
        }
        (output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
        # This sample can also be inspected without decoding the source videos again.
        np.savez_compressed(
            output / "observation.npz",
            **{
                key: np.asarray(value)
                for key, value in sample.items()
                if key.startswith("observation.")
            },
        )
    else:
        from rlinf.robotics.piper_runtime import rollout_piper

        if args.interactive:
            from rlinf.robotics.piper_announcements import PiperAnnouncer
            from rlinf.robotics.piper_evaluation import TerminalKeys, evaluate_piper

            if session.intervention is not None:
                print(
                    f"Human takeover: {session.intervention.config.key}; b marks critical phase. "
                    "The leader aligns after Right. Assisted success is reported separately.",
                    flush=True,
                )

            print(
                "Interactive evaluation enables the follower after cameras connect, "
                f"then waits for Right. Reset after scoring: {args.reset_after_episode}. "
                "Keep this terminal focused; b alternates critical phase start/end; s/f stops and scores a trial.",
                flush=True,
            )
            try:
                with (
                    PiperAnnouncer(
                        enabled=args.play_sounds, language=args.speech_language
                    ) as announcements,
                    TerminalKeys() as keys,
                ):
                    evaluate_piper(
                        policy,
                        session,
                        args.task,
                        output=output,
                        num_episodes=args.num_episodes,
                        steps=args.steps,
                        keys=keys,
                        replan_steps=args.replan_steps,
                        save_observations=args.save_observations,
                        record_dataset=args.record_dataset,
                        streaming_video=args.streaming_video,
                        phase_definition=args.phase_definition,
                        session_id=args.session_id,
                        reset_after_episode=args.reset_after_episode,
                        reset_timing=reset_timing,
                        policy_source="actor"
                        if args.actor is not None
                        else "reference",
                        announce=announcements,
                    )
            except KeyboardInterrupt:
                print(
                    "Interrupted; partial results saved. Arm holds its last target.",
                    flush=True,
                )
            except Exception:
                print(
                    f"Evaluation stopped after a fault; keys are no longer active. "
                    f"Saved results: {output.resolve()}. Restart the command after resolving the fault.",
                    flush=True,
                )
                raise
            print(f"Results: {output.resolve()}")
            return
        print(
            f"Opening cameras from {args.hardware}; the follower enables only after every camera is ready, then executes policy targets without a reset trajectory. "
            f"high_follow={hardware.get('high_follow', False)}, speed_percent={hardware.get('speed_percent', 10)}, "
            f"replan_steps={args.replan_steps if args.replan_steps is not None else policy.action_horizon}.",
            flush=True,
        )
        with (output / "trace.jsonl").open("x") as log:
            try:
                completed = rollout_piper(
                    policy,
                    session,
                    args.task,
                    steps=args.steps,
                    replan_steps=args.replan_steps,
                    log=log,
                    observation_dir=output / "observations"
                    if args.save_observations
                    else None,
                )
            except KeyboardInterrupt:
                print(
                    "Interrupted; robot disconnected and left holding its last target.",
                    flush=True,
                )
                return
        (output / "result.json").write_text(
            json.dumps({"completed_steps": completed}) + "\n"
        )
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
