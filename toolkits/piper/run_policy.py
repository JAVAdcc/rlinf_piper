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
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.critical_phase import (
    SEGMENT_PHASE_DEFINITION,
    SEGMENT_SCORING_PROTOCOL,
)
from rlinf.robotics.piper_collection import collection_lock, read_collection_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a new run or restore collection settings with explicit overrides."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.set_defaults(scoring_protocol=SEGMENT_SCORING_PROTOCOL)
    parser.add_argument("command", choices=("infer", "rollout"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Stage 1 actor directory, or CFG actor directory; restored by --resume",
    )
    parser.add_argument("--backend", choices=("rlt", "cfg"), default="rlt")
    parser.add_argument(
        "--actor", type=Path, help="Optional RLT Stage 2 actor checkpoint"
    )
    parser.add_argument(
        "--actor-mode",
        choices=("always", "manual-critical"),
        default="always",
        help="manual-critical: b selects actor, s/f returns to reference; reference and actor keep their own horizons",
    )
    parser.add_argument(
        "--allow-offline-actor",
        action="store_true",
        help="Explicitly test a shared offline-local actor whose readiness has not passed; requires manual-critical mode",
    )
    parser.add_argument(
        "--infer-actor",
        action="store_true",
        help="With manual-critical mode, select actor for an offline infer command (default reference)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--online-config",
        type=Path,
        help="Train native RLT workers after Right accepts an episode; requires recorded manual-critical rollout",
    )
    parser.add_argument(
        "--task", help="Training task text; infer defaults to the dataset task"
    )
    parser.add_argument(
        "--offline-cache",
        type=Path,
        help="Warmup cache for mixed online training; train split only",
    )
    parser.add_argument("--dataset", type=Path, help="Local LeRobot dataset for infer")
    parser.add_argument(
        "--video-backend",
        default="torchcodec",
        choices=("pyav", "torchcodec"),
        help="Decoder for hardware-free dataset inference",
    )
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
        help="Right starts/stops an episode or confirms completion; b starts a segment; s/f scores the segment, then separately the stopped episode; Left discards",
    )
    parser.add_argument(
        "--reset-after-episode",
        choices=("none", "initial", "zero"),
        default="none",
        help="After scoring or discarding: open gripper, return to the first start pose or zero, restore gripper",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Total scored episodes to retain, including previous collection when resuming",
    )
    parser.add_argument(
        "--record-dataset",
        action="store_true",
        help="Record every RGB/state/command plus terminal observation; b starts a critical segment, s/f ends and scores it",
    )
    parser.add_argument(
        "--streaming-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode RGB during collection; --no-streaming-video uses post-episode PNG encoding",
    )
    parser.add_argument(
        "--phase-definition",
        default=SEGMENT_PHASE_DEFINITION,
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
        "--save-rlt-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log the full same-pass VLA reference alongside each rollout prediction",
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
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; defaults to a unique outputs/piper run",
    )
    destination.add_argument(
        "--resume",
        type=Path,
        help="Continue an existing interactive run, restoring its settings and reset pose",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.resume is not None:
        allowed = {
            "--resume",
            "--num-episodes",
            "--device",
            "--play-sounds",
            "--no-play-sounds",
            "--speech-language",
            "--streaming-video",
            "--no-streaming-video",
            "--save-rlt-reference",
            "--no-save-rlt-reference",
        }
        supplied = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
        if supplied - allowed:
            parser.error(
                "--resume restores collection settings; permitted overrides: "
                + ", ".join(sorted(allowed - {"--resume"}))
            )
        try:
            output = args.resume.expanduser().resolve()
            saved = read_collection_json(output / "run.json")
            if (
                args.command != "rollout"
                or saved.get("command") != "rollout"
                or saved.get("interactive") is not True
            ):
                raise ValueError("--resume requires an existing interactive rollout.")
            if saved.get("scoring_protocol") != SEGMENT_SCORING_PROTOCOL:
                raise ValueError(
                    "Legacy episode-only collection cannot resume with segment scoring. "
                    "Preserve it and use a new --output-dir."
                )
            if not (output / "hardware.yaml").is_file():
                raise ValueError(
                    f"Missing hardware snapshot: {output / 'hardware.yaml'}"
                )
            defaults = {
                key: value
                for key, value in saved.items()
                if key in vars(args) and key not in ("command", "resume", "output_dir")
            }
            # Historical wrapper runs used the repository as their working directory.
            for key in (
                "checkpoint",
                "actor",
                "hardware",
                "dataset",
                "online_config",
                "offline_cache",
            ):
                if defaults.get(key) is not None:
                    path = Path(defaults[key]).expanduser()
                    if not path.is_absolute():
                        path = Path(__file__).resolve().parents[2] / path
                    defaults[key] = path
            if (output / "summary.json").exists():
                defaults["num_episodes"] = read_collection_json(
                    output / "summary.json"
                )["requested_episodes"]
            parser.set_defaults(**defaults)
            args = parser.parse_args(argv)
            args.resume = output
        except (ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    from rlinf.robotics.piper_reset import PiperResetTiming

    try:
        PiperResetTiming(args.reset_duration_s, args.reset_gripper_duration_s)
    except ValueError as exc:
        parser.error(str(exc))
    if args.checkpoint is None:
        parser.error("--checkpoint is required for a new run.")
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
    if args.actor_mode == "manual-critical":
        if args.backend != "rlt" or args.actor is None:
            parser.error(
                "--actor-mode manual-critical requires --actor with the rlt backend."
            )
        if args.command == "rollout" and not args.interactive:
            parser.error("Manual critical rollout requires --interactive.")
    if args.allow_offline_actor and args.actor_mode != "manual-critical":
        parser.error("--allow-offline-actor requires --actor-mode manual-critical.")
    if args.infer_actor and (
        args.command != "infer" or args.actor_mode != "manual-critical"
    ):
        parser.error("--infer-actor requires infer with --actor-mode manual-critical.")
    if args.record_dataset and (not args.interactive or args.command != "rollout"):
        parser.error("--record-dataset requires interactive rollout.")
    if args.record_dataset and args.backend != "rlt":
        parser.error("The warmup data recorder requires the RLT backend.")
    if args.online_config is not None:
        if (
            args.command != "rollout"
            or not args.record_dataset
            or args.actor_mode != "manual-critical"
        ):
            parser.error(
                "--online-config requires --record-dataset and --actor-mode manual-critical rollout."
            )
        if args.device != "cuda:0":
            parser.error("The current online profile uses the single local GPU cuda:0.")
    elif args.offline_cache is not None:
        parser.error("--offline-cache requires --online-config.")
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
    if args.save_rlt_reference and (args.backend != "rlt" or args.command != "rollout"):
        parser.error("--save-rlt-reference requires rollout with the RLT backend.")
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
    return args


def main() -> None:
    """Own the output directory before loading a model or connecting hardware."""
    args = parse_args()
    output = (
        args.resume
        or args.output_dir
        or Path("outputs/piper")
        / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8])
    )
    output = output.expanduser().resolve()
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
    with collection_lock(output):
        _run(args, output)


def _run(args: argparse.Namespace, output: Path) -> None:
    with ExitStack() as stack:
        _run_with_resources(args, output, stack)


def _run_with_resources(
    args: argparse.Namespace, output: Path, stack: ExitStack
) -> None:
    """Execute under the run lock, keeping original launch snapshots immutable."""
    from rlinf.models.embodiment.piper_inference import PiperCFGPolicy, PiperRLTPolicy
    from rlinf.robotics.piper_reset import PiperResetTiming

    reset_timing = PiperResetTiming(
        args.reset_duration_s, args.reset_gripper_duration_s
    )
    metadata_output = output
    if args.resume is not None:
        metadata_output = (
            output
            / "resumes"
            / (
                time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                + "_"
                + uuid.uuid4().hex[:8]
            )
        )
        metadata_output.mkdir(parents=True, exist_ok=False)
    (metadata_output / "run.json").write_text(
        json.dumps(
            {
                key: str(value.expanduser().resolve())
                if isinstance(value, Path)
                else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Results: {output.resolve()}", flush=True)
    if args.command == "rollout":
        from rlinf.robotics.piper_runtime import build_piper_session

        hardware = OmegaConf.load(
            output / "hardware.yaml" if args.resume is not None else args.hardware
        )
        if args.teleop is not None:
            if "teleop" not in hardware:
                hardware.teleop = {}
            hardware.teleop.type = args.teleop
        if args.leader_usb_serial is not None:
            hardware.teleop.usb_serial = args.leader_usb_serial
        if args.intervention_key is not None:
            if hardware.get("teleop", {}).get("type") != "piper_leader":
                raise ValueError(
                    "--intervention-key requires a configured Piper leader."
                )
            if "intervention" not in hardware.teleop:
                hardware.teleop.intervention = {}
            hardware.teleop.intervention.key = args.intervention_key
        if (
            hardware.get("teleop", {}).get("type", "none") != "none"
            and not args.interactive
        ):
            raise ValueError("Configured teleop requires --interactive.")
        if args.no_action_limits:
            hardware.execution.check_action_limits = False
        OmegaConf.save(hardware, metadata_output / "hardware.yaml")
        session = build_piper_session(metadata_output / "hardware.yaml")
        if session.intervention is not None:
            import importlib.util

            for module in ("piper_sdk", "pinocchio"):
                if importlib.util.find_spec(module) is None:
                    raise ValueError(
                        f"Missing {module}; install requirements/embodied/piper-hil.txt."
                    )
    torch.manual_seed(args.seed)
    online = None
    if args.online_config is not None:
        from rlinf.runners.piper_online_runner import PiperOnlineRunner, online_config

        cfg = online_config(
            output / "online.yaml" if args.resume is not None else args.online_config,
            stage1=args.checkpoint,
            actor=args.actor,
            hardware=OmegaConf.to_container(hardware, resolve=True),
            task=args.task,
            output=output,
            allow_offline_actor=args.allow_offline_actor,
            offline_cache=args.offline_cache,
            resume=args.resume is not None,
        )
        if (
            args.replan_steps is not None
            and args.replan_steps < cfg.actor.model.num_action_chunks
        ):
            raise ValueError("Online execution must not truncate the actor horizon.")
        OmegaConf.save(cfg, metadata_output / "online.yaml")
        online = stack.enter_context(
            PiperOnlineRunner(cfg, output, resume=args.resume is not None)
        )
    if online is not None:
        policy = online.policy
    elif args.backend == "rlt":
        policy = PiperRLTPolicy(
            args.checkpoint,
            actor=args.actor,
            device=args.device,
            manual_critical=args.actor_mode == "manual-critical",
            allow_offline_actor=args.allow_offline_actor,
        )
    else:
        policy = PiperCFGPolicy(args.checkpoint, device=args.device)
    if args.resume is not None and (output / "policy.json").exists():
        if policy.manifest != read_collection_json(output / "policy.json"):
            raise ValueError(
                "Policy manifest changed; collect this policy in a new directory."
            )
    (metadata_output / "policy.json").write_text(
        json.dumps(policy.manifest, indent=2) + "\n"
    )
    if args.command == "infer":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from rlinf.data.storage.lerobot.decoding import decode_image_struct_batch

        root = args.dataset.expanduser().resolve()
        dataset = LeRobotDataset(
            root.name,
            root=root,
            download_videos=False,
            video_backend=args.video_backend,
        )
        dataset.hf_dataset.set_transform(decode_image_struct_batch)
        sample = dataset[args.frame]
        task = args.task or sample["task"]
        started = time.monotonic()
        actions = (
            policy.infer(sample, task, actor_enabled=args.infer_actor)
            if args.actor_mode == "manual-critical"
            else policy.infer(sample, task)
        )
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
                f"Interactive {'online training' if online is not None else 'evaluation'} enables the follower after cameras connect, "
                f"then waits for Right. Reset after trial: {args.reset_after_episode}. "
                "Keep this terminal focused: b starts a segment; s/f ends and scores only that segment while actions continue. Right stops the episode for a separate s/f score; Left stops and discards it.",
                flush=True,
            )
            if online is not None:
                print(
                    f"Online policy version: {policy.version}. After recording and reset, "
                    + (
                        "Right accepts and queues the previous episode; training runs in background. "
                        "Each episode uses one policy version, installed before it starts. "
                        if online.async_training
                        else "Right accepts the previous episode, trains, and installs the next policy. "
                    )
                    + "Accepted episodes cannot be discarded; q leaves an unaccepted episode pending for resume.",
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
                        save_rlt_reference=args.save_rlt_reference,
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
                        resume=args.resume is not None,
                        on_accept=online.accept_episode if online is not None else None,
                        on_poll=online.check_health if online is not None else None,
                        on_episode_start=policy.begin_episode
                        if online is not None
                        else None,
                        on_finish=online.finish_training
                        if online is not None
                        else None,
                        async_training=online.async_training
                        if online is not None
                        else False,
                    )
            except KeyboardInterrupt:
                print(
                    "Interrupted; partial results saved. Arm holds its last target. "
                    "Use --resume with this results directory to continue collection.",
                    flush=True,
                )
            except Exception:
                print(
                    f"Evaluation stopped after a fault; keys are no longer active. "
                    f"Saved results: {output.resolve()}. After resolving the fault, "
                    "use --resume with this directory to continue collection.",
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
                    save_rlt_reference=args.save_rlt_reference,
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
