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

"""Local Piper observation and action execution, independent of the RL algorithm."""

import contextlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TextIO

import numpy as np

from rlinf.models.embodiment.openpi.policies.piper_policy import PIPER_CAMERAS
from rlinf.robotics import PiperRobot
from rlinf.robotics.parts.arms.piper import PiperArm
from rlinf.robotics.piper_intervention import (
    PiperIntervention,
    PiperInterventionConfig,
)


class ChunkPolicy(Protocol):
    """Policy boundary shared by local execution and algorithm-specific clients."""

    action_horizon: int
    fps: float

    def infer(self, observation: dict[str, Any], task: str) -> np.ndarray:
        """Return absolute targets of shape (horizon, 7) in degrees/mm."""
        ...


@dataclass
class PiperPolicySelection:
    """Latch manual routing changes; each change invalidates queued predictions."""

    actor_enabled: bool = False
    generation: int = 0

    def set_critical(self, active: bool) -> None:
        if self.actor_enabled != active:
            self.actor_enabled = active
            self.generation += 1


@dataclass(frozen=True)
class _PolicyRequest:
    generation: int
    actor_enabled: bool | None
    source: str
    horizon: int
    execute_steps: int

    def infer(
        self,
        policy: ChunkPolicy,
        observation: dict,
        task: str,
        save_rlt_reference: bool,
        replay_directory: Path | None = None,
        frame: int = 0,
    ) -> tuple[np.ndarray, dict]:
        kwargs = (
            {} if self.actor_enabled is None else {"actor_enabled": self.actor_enabled}
        )
        if replay_directory is not None:
            started = time.monotonic()
            actions, diagnostics = policy.infer_for_replay(
                observation, task, **kwargs, directory=replay_directory, frame=frame
            )
            return actions, {
                **diagnostics,
                "inference_call_s": time.monotonic() - started,
            }
        if save_rlt_reference:
            started = time.monotonic()
            actions, diagnostics = policy.infer_with_reference(
                observation, task, **kwargs
            )
            return actions, {
                **diagnostics,
                "inference_call_s": time.monotonic() - started,
            }
        return policy.infer(observation, task, **kwargs), {}


def _policy_request(
    policy: ChunkPolicy,
    selection: PiperPolicySelection | None,
    source: str,
    replan_steps: int,
) -> _PolicyRequest:
    enabled = selection.actor_enabled if selection is not None else None
    horizon = policy.actor_horizon if enabled else policy.action_horizon
    return _PolicyRequest(
        selection.generation if selection is not None else 0,
        enabled,
        ("actor" if enabled else "reference") if selection is not None else source,
        horizon,
        min(replan_steps, horizon),
    )


@dataclass(frozen=True)
class PiperExecutionConfig:
    """Control units and maximum changes checked against measured feedback."""

    fps: float = 30.0
    gripper_max_width_m: float = 0.07
    max_joint_step_deg: float = 5.0
    max_gripper_step_mm: float = 20.0
    max_camera_age_s: float = 0.5
    max_prediction_age_s: float = 2.0
    check_action_limits: bool = True

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "check_action_limits":
                if not isinstance(value, bool):
                    raise ValueError("check_action_limits must be a boolean.")
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")


def resolve_piper_channel(
    usb_serial: str, *, sysfs: Path = Path("/sys/class/net")
) -> str:
    """Resolve one SocketCAN channel by its USB adapter serial, rejecting ambiguity."""
    if not usb_serial:
        raise ValueError("Specify the follower USB CAN serial.")
    matches = []
    for interface in sysfs.glob("can*"):
        device = (interface / "device").resolve()
        for parent in (device, *device.parents):
            serial = parent / "serial"
            if serial.is_file() and serial.read_text().strip() == usb_serial:
                matches.append(interface.name)
                break
    if len(matches) != 1:
        raise ValueError(f"Expected one CAN channel for {usb_serial}, found {matches}.")
    return matches[0]


def piper_state_in_dataset_units(
    reading: dict, gripper_max_width_m: float
) -> np.ndarray:
    """Convert the robotics observation's radians/opening fraction to degrees/mm."""
    joints = np.asarray(reading["arm_joint_position"], dtype=np.float64)
    opening = np.asarray(reading["end_effector"]["state"], dtype=np.float64)
    if joints.shape != (6,) or opening.shape != (1,):
        raise ValueError("Expected six measured joints and one gripper opening.")
    state = np.concatenate([np.rad2deg(joints), opening * gripper_max_width_m * 1000])
    if not np.isfinite(state).all() or not 0 <= opening[0] <= 1:
        raise ValueError("Invalid Piper feedback.")
    return state.astype(np.float32)


def piper_action_for_robot(
    target: np.ndarray,
    measured: np.ndarray,
    joint_limits: tuple[np.ndarray, np.ndarray],
    config: PiperExecutionConfig,
) -> dict:
    """Validate an absolute degrees/mm target and convert to robotics units.

    With check_action_limits enabled, reject out-of-range targets except for
    normalization roundoff and bound changes relative to measured position.
    Disabling it preserves shape/finite checks and the underlying driver limits.
    """
    target = np.array(target, dtype=np.float64, copy=True)
    measured = np.asarray(measured, dtype=np.float64)
    if target.shape != (7,) or measured.shape != (7,):
        raise ValueError("Piper targets and measured states must have shape (7,).")
    if not np.isfinite(target).all() or not np.isfinite(measured).all():
        raise ValueError("Piper targets and measured states must be finite.")
    joints = np.deg2rad(target[:6])
    width_mm = config.gripper_max_width_m * 1000
    if not config.check_action_limits:
        # This bypasses the policy executor's extra bounds and step checks.
        # PiperArm still applies the physical travel supported by its driver.
        return {
            "arm": {
                "joint_position": joints,
                "end_effector": {"target": np.array([target[6] / width_mm])},
            }
        }
    low, high = joint_limits
    if np.any(joints < low - 1e-5) or np.any(joints > high + 1e-5):
        raise ValueError("Policy target is outside the Piper joint limits.")
    if not -0.01 <= target[6] <= width_mm + 0.01:
        raise ValueError("Policy gripper target is outside its calibrated stroke.")
    # OpenPI's quantile inverse includes an epsilon in model units. Tolerate
    # less than 0.001 degree / 0.01 mm, then enforce the exact driver bounds.
    joints = np.clip(joints, low, high)
    target[:6] = np.rad2deg(joints)
    target[6] = np.clip(target[6], 0, width_mm)
    if np.any(np.abs(target[:6] - measured[:6]) > config.max_joint_step_deg):
        raise ValueError("Policy joint target exceeds max_joint_step_deg.")
    if abs(target[6] - measured[6]) > config.max_gripper_step_mm:
        raise ValueError("Policy gripper target exceeds max_gripper_step_mm.")
    return {
        "arm": {
            "joint_position": joints,
            "end_effector": {"target": np.array([target[6] / width_mm])},
        }
    }


class PiperSession:
    """Own one local robot and three RGB cameras for explicit observation/action I/O.

    Camera objects use LeRobot's connect/read_latest/disconnect interface.
    Connecting the robot enables motors; neither opening nor closing resets it.
    An online environment can use observe/execute directly and own its reward,
    episode reset, replay, and learner lifecycle separately.
    """

    def __init__(
        self,
        robot: PiperRobot,
        cameras: dict[str, Any],
        config: PiperExecutionConfig,
        intervention: PiperIntervention | None = None,
    ):
        keys = {f"observation.images.{name}" for name in cameras}
        for aliases in PIPER_CAMERAS.values():
            if len(keys.intersection(aliases)) != 1:
                raise ValueError(f"Expected one RGB camera for {aliases}.")
        if len(cameras) != 3:
            raise ValueError("A Piper session requires exactly three RGB cameras.")
        self.robot = robot
        self.cameras = cameras
        self.config = config
        self.intervention = intervention
        self.arm = robot.child("arm", PiperArm)
        self.last_command: np.ndarray | None = None
        self._resources: contextlib.ExitStack | None = None

    def __enter__(self) -> "PiperSession":
        if self._resources is not None:
            raise RuntimeError("Piper session is already connected.")
        resources = contextlib.ExitStack()
        try:
            for camera in self.cameras.values():
                resources.callback(self._close_camera, camera)
                camera.connect()
                camera.read_latest(max_age_ms=int(self.config.max_camera_age_s * 1000))
            if self.intervention is not None:
                resources.callback(self.intervention.close)
                self.intervention.connect()
            resources.callback(self.robot.disconnect)
            self.robot.connect()
        except BaseException:
            resources.close()
            raise
        self._resources = resources
        return self

    @property
    def is_connected(self) -> bool:
        """Whether this session owns open robot and camera connections."""
        return self._resources is not None

    @staticmethod
    def _close_camera(camera: Any) -> None:
        if camera.is_connected:
            camera.disconnect()

    def close(self) -> None:
        """Release all opened resources, leaving the arm holding its last target."""
        resources, self._resources = self._resources, None
        if resources is not None:
            try:
                if self.intervention is not None:
                    self.intervention.stop()
            finally:
                resources.close()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def observe(self) -> dict[str, np.ndarray]:
        """Read raw RGB views and the seven-value state in dataset units."""
        if self._resources is None:
            raise RuntimeError("Connect the Piper session before observing.")
        result = {}
        for name, camera in self.cameras.items():
            frame = np.asarray(
                camera.read_latest(max_age_ms=int(self.config.max_camera_age_s * 1000))
            )
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"Camera {name} must supply uint8 HWC RGB.")
            result[f"observation.images.{name}"] = frame.copy()
        result["observation.state"] = piper_state_in_dataset_units(
            self.robot.get_observation()["arm"], self.config.gripper_max_width_m
        )
        return result

    def read_state(self) -> np.ndarray:
        """Read fresh joint/gripper feedback without depending on camera capture."""
        if self._resources is None:
            raise RuntimeError("Connect the Piper session before reading feedback.")
        return piper_state_in_dataset_units(
            self.robot.get_observation()["arm"], self.config.gripper_max_width_m
        )

    def prepare_for_reset(self) -> None:
        """Keep cameras/CAN open while restoring position-control readiness."""
        if not self.is_connected:
            raise RuntimeError("Connect the Piper session before resetting.")
        if self.intervention is not None:
            self.intervention.stop()
        self.arm.prepare_for_reset()

    def clip_reset_target(self, target: np.ndarray) -> np.ndarray:
        """Project finite degree/mm targets onto this arm's configured travel."""
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Reset targets must contain seven finite values.")
        low, high = self.arm.joint_limits
        return np.clip(
            target,
            np.r_[np.rad2deg(low), 0],
            np.r_[np.rad2deg(high), self.config.gripper_max_width_m * 1000],
        )

    def execute_reset(self, target: np.ndarray) -> dict[str, np.ndarray]:
        """Send one paced reset target; retain travel, finite and status checks.

        Reset trajectories own pacing instead of the policy's per-step limits.
        Feedback outside nominal software travel is logged, not used as a new
        destination. Commands are clipped to the driver's unchanged travel.
        """
        self.arm.check_reset_control()
        measured = self.read_state()
        requested = np.asarray(target, dtype=float)
        command = self.clip_reset_target(requested)
        low, high = self.arm.joint_limits
        action = piper_action_for_robot(
            command,
            measured,
            (low, high),
            replace(self.config, check_action_limits=False),
        )
        self.robot.send_action(action)
        if self.intervention is not None:
            self.intervention.executed(command, "hold")
        return {"state": measured, "requested": requested.copy(), "command": command}

    def execute(self, target: np.ndarray) -> np.ndarray:
        """Check fresh feedback and send one target, returning the pre-action state."""
        if self._resources is None:
            raise RuntimeError("Connect the Piper session before executing.")
        measured = piper_state_in_dataset_units(
            self.robot.get_observation()["arm"], self.config.gripper_max_width_m
        )
        action = piper_action_for_robot(
            target, measured, self.arm.joint_limits, self.config
        )
        applied = self.robot.send_action(action)["arm"]
        self.last_command = np.r_[
            np.rad2deg(applied["joint_position"]),
            np.clip(applied["end_effector"]["target"], 0, 1)
            * self.config.gripper_max_width_m
            * 1000,
        ].astype(np.float32)
        return measured


def build_piper_session(hardware_path: str | Path) -> PiperSession:
    """Construct a disconnected session from explicit USB serial and RGB devices."""
    from omegaconf import OmegaConf

    from rlinf.robotics.piper_camera import build_piper_cameras

    hardware = OmegaConf.to_container(OmegaConf.load(hardware_path), resolve=True)
    config = PiperExecutionConfig(**hardware["execution"])
    channel = resolve_piper_channel(hardware["usb_serial"])
    intervention = None
    teleop = dict(hardware.get("teleop") or {})
    kind = teleop.pop("type", "none")
    if kind != "none":
        if kind != "piper_leader":
            raise ValueError(f"Unsupported Piper teleop type: {kind}.")
        from rlinf.robotics.parts.teleop.piper_leader import (
            PiperLeader,
            PiperLeaderConfig,
        )

        leader_channel = resolve_piper_channel(teleop.pop("usb_serial", ""))
        if leader_channel == channel:
            raise ValueError(
                "Piper leader and follower must use different CAN channels."
            )
        control = PiperInterventionConfig(**teleop.pop("intervention", {}))
        leader_config = PiperLeaderConfig(**teleop)
        if not np.isclose(
            leader_config.gripper_width_mm, config.gripper_max_width_m * 1000
        ):
            raise ValueError("Leader and follower gripper strokes must match.")
        intervention = PiperIntervention(
            PiperLeader(leader_channel, config=leader_config),
            control,
        )
    cameras = build_piper_cameras(hardware["cameras"])
    speed = int(hardware.get("speed_percent", 10))
    if not 1 <= speed <= 100:
        raise ValueError("speed_percent must be in [1, 100].")
    robot = PiperRobot(
        arm=PiperArm.declare(
            channel,
            firmware=hardware.get("firmware", "v189"),
            speed_percent=speed,
            high_follow=hardware.get("high_follow", False),
            gripper_force=hardware.get("gripper_force", 1.0),
            gripper_max_width=config.gripper_max_width_m,
            max_feedback_age_s=0.5,
        )
    )
    return PiperSession(robot, cameras, config, intervention)


def rollout_piper(
    policy: ChunkPolicy,
    session: PiperSession,
    task: str,
    *,
    steps: int,
    log: TextIO,
    replan_steps: int | None = None,
    observation_dir: Path | None = None,
    save_rlt_reference: bool = False,
    stop_requested: Callable[[int], bool] | None = None,
    record_step: Callable[..., None] | None = None,
    manage_session: bool = True,
    policy_source: str = "reference",
    on_policy_resume: Callable[[], None] | None = None,
    policy_selection: PiperPolicySelection | None = None,
) -> int:
    """Execute bounded chunks, with concurrent inference when HIL is enabled.

    As in Evo-RL's select_action/record_loop, refill an empty action queue and
    send one target per tick. Observation, inference, logging and submission all
    count toward that tick's period. A slow tick does not create timing debt.
    By default, consume the policy's full horizon before predicting again.
    The JSONL output is a diagnostic trace, not an online RL replay dataset.
    HIL keeps processing input and human commands while inference runs, holds
    between predictions, and invalidates queued predictions on ownership changes.
    on_policy_resume runs after the first fresh policy command following release.
    """
    if replan_steps is None:
        replan_steps = policy.action_horizon
    if getattr(policy, "manual_critical", False) != (policy_selection is not None):
        raise ValueError(
            "Manual critical policy and execution selection must be supplied together."
        )
    if steps <= 0 or not 1 <= replan_steps <= policy.action_horizon:
        raise ValueError(
            "Use positive steps and replan_steps within the policy horizon."
        )
    if not task.strip():
        raise ValueError("Specify the task used for policy training.")
    if not np.isclose(session.config.fps, policy.fps):
        raise ValueError("Execution fps must match the policy's training data.")
    if save_rlt_reference and not callable(
        getattr(policy, "infer_with_reference", None)
    ):
        raise ValueError("Reference logging requires an RLT policy.")
    completed = 0
    period = 1 / session.config.fps
    if observation_dir is not None:
        observation_dir.mkdir(parents=True, exist_ok=False)
    actions = np.empty((0, 7))
    chunk_index = 0
    request_index = 0
    request = _policy_request(policy, policy_selection, policy_source, replan_steps)
    with session if manage_session else contextlib.nullcontext(session):
        if getattr(session, "intervention", None) is not None:
            return _rollout_with_intervention(
                policy,
                session,
                task,
                steps=steps,
                log=log,
                replan_steps=replan_steps,
                observation_dir=observation_dir,
                save_rlt_reference=save_rlt_reference,
                stop_requested=stop_requested,
                record_step=record_step,
                policy_source=policy_source,
                on_policy_resume=on_policy_resume,
                policy_selection=policy_selection,
            )
        while completed < steps:
            if stop_requested is not None and stop_requested(completed):
                break
            started = time.monotonic()
            wall_time = time.time()
            observation = session.observe()
            if (
                policy_selection is not None
                and actions.size
                and request.generation != policy_selection.generation
            ):
                actions, chunk_index = np.empty((0, 7)), 0
            if chunk_index == len(actions):
                request_index += 1
                observed_at = started
                request = _policy_request(
                    policy, policy_selection, policy_source, replan_steps
                )
                actions, diagnostics = request.infer(
                    policy,
                    observation,
                    task,
                    save_rlt_reference,
                    Path(log.name).parent
                    if getattr(policy, "record_replay_features", False)
                    else None,
                    completed,
                )
                actions = np.asarray(actions)
                if (
                    actions.shape != (request.horizon, 7)
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError("Policy returned an invalid action chunk.")
                prediction = {
                    "event": "prediction",
                    "step": completed,
                    "inference_s": time.monotonic() - observed_at,
                    "actions": actions.tolist(),
                    "replan_steps": request.execute_steps,
                    "action_source": request.source,
                    "route_generation": request.generation,
                    "policy_version": diagnostics.get("policy_version", -1),
                    "background_wait_s": diagnostics.get("background_wait_s", 0.0),
                    **({"rlt_reference": diagnostics} if save_rlt_reference else {}),
                }
                if observation_dir is not None:
                    name = f"step_{completed:06d}"
                    if policy_selection is not None:
                        name += f"_request_{request_index:06d}"
                    path = observation_dir / f"{name}.npz"
                    np.savez(path, **observation, task=np.asarray(task))
                    prediction["observation_file"] = str(path.resolve())
                log.write(json.dumps(prediction) + "\n")
                log.flush()
                actions = actions[: request.execute_steps]
                chunk_index = 0
            # Consume labels entered during inference before sending another target.
            if stop_requested is not None and stop_requested(completed):
                break
            if (
                policy_selection is not None
                and request.generation != policy_selection.generation
            ):
                log.write(
                    json.dumps(
                        {
                            "event": "route_discard",
                            "step": completed,
                            "route_generation": request.generation,
                        }
                    )
                    + "\n"
                )
                actions, chunk_index = np.empty((0, 7)), 0
                continue
            prediction_age = time.monotonic() - observed_at
            if prediction_age > session.config.max_prediction_age_s:
                raise TimeoutError(
                    f"Policy prediction age {prediction_age:.3f}s exceeded "
                    f"max_prediction_age_s={session.config.max_prediction_age_s:.3f}s "
                    f"before step {completed}, chunk target {chunk_index + 1}/{len(actions)} "
                    f"(observation/inference={prediction['inference_s']:.3f}s). "
                    "The age includes observation, inference, recording and chunk playback."
                )
            target = actions[chunk_index]
            measured = session.execute(target)
            log.write(
                json.dumps(
                    {
                        "event": "action",
                        "step": completed,
                        "state": measured.tolist(),
                        "target": target.tolist(),
                        "command": session.last_command.tolist()
                        if record_step is not None
                        else None,
                        "time": time.time(),
                        "prediction_age_s": time.monotonic() - observed_at,
                        "action_source": request.source,
                        "route_generation": request.generation,
                        "policy_version": diagnostics.get("policy_version", -1),
                    }
                )
                + "\n"
            )
            log.flush()
            if record_step is not None:
                record_step(
                    observation,
                    session.last_command,
                    requested=target,
                    measured=measured,
                    wall_time=wall_time,
                    action_source=request.source,
                    policy_action=target,
                    policy_version=diagnostics.get("policy_version", -1),
                )
            chunk_index += 1
            completed += 1
            time.sleep(max(0, period - (time.monotonic() - started)))
    return completed


def _rollout_with_intervention(
    policy: ChunkPolicy,
    session: PiperSession,
    task: str,
    *,
    steps: int,
    log: TextIO,
    replan_steps: int,
    observation_dir: Path | None,
    save_rlt_reference: bool,
    stop_requested: Callable[[int], bool] | None,
    record_step: Callable[..., None] | None,
    policy_source: str,
    on_policy_resume: Callable[[], None] | None,
    policy_selection: PiperPolicySelection | None,
) -> int:
    """Keep control and recording responsive while one inference job runs."""
    control = session.intervention
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-policy")
    request_generation = request_step = 0
    requested_at = 0.0
    future = None
    actions = np.empty((0, 7))
    index = completed = 0
    generation = control.generation
    prediction_time = 0.0
    resume_generation = None
    route_generation = (
        policy_selection.generation if policy_selection is not None else 0
    )
    action_source = policy_source
    chunk_version = -1
    request = _policy_request(policy, policy_selection, policy_source, replan_steps)
    period = 1 / session.config.fps
    try:
        while completed < steps:
            if stop_requested is not None and stop_requested(completed):
                break
            started, wall_time = time.monotonic(), time.time()
            observation = session.observe()
            if generation != control.generation:
                actions, index = np.empty((0, 7)), 0
                generation = control.generation
            if (
                policy_selection is not None
                and route_generation != policy_selection.generation
            ):
                actions, index = np.empty((0, 7)), 0
                route_generation = policy_selection.generation
            if future is not None and future.done():
                prediction, diagnostics = future.result()
                prediction = np.asarray(prediction)
                future = None
                if (
                    prediction.shape != (request.horizon, 7)
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError("Policy returned an invalid action chunk.")
                accepted = (
                    (
                        policy_selection is None
                        or request.generation == policy_selection.generation
                    )
                    and request_generation == control.generation
                    and control.mode != "human"
                )
                log.write(
                    json.dumps(
                        {
                            "event": "prediction",
                            "step": request_step,
                            "accepted_at_step": completed if accepted else None,
                            "generation": request_generation,
                            "discarded": not accepted,
                            "inference_s": time.monotonic() - requested_at,
                            "actions": prediction.tolist(),
                            "replan_steps": request.execute_steps,
                            "action_source": request.source,
                            "route_generation": request.generation,
                            "policy_version": diagnostics.get("policy_version", -1),
                            "background_wait_s": diagnostics.get(
                                "background_wait_s", 0.0
                            ),
                            **(
                                {"rlt_reference": diagnostics}
                                if save_rlt_reference
                                else {}
                            ),
                        }
                    )
                    + "\n"
                )
                log.flush()
                if accepted:
                    actions, index = prediction[: request.execute_steps], 0
                    action_source = request.source
                    chunk_version = int(diagnostics.get("policy_version", -1))
                    prediction_time = requested_at
            if future is None and index == len(actions) and control.mode != "human":
                request_generation = control.generation
                request = _policy_request(
                    policy, policy_selection, policy_source, replan_steps
                )
                request_step, requested_at = completed, started
                snapshot = {key: value.copy() for key, value in observation.items()}
                if observation_dir is not None:
                    np.savez(
                        observation_dir / f"step_{completed:06d}.npz",
                        **snapshot,
                        task=np.asarray(task),
                    )
                future = executor.submit(
                    request.infer,
                    policy,
                    snapshot,
                    task,
                    save_rlt_reference,
                    Path(log.name).parent
                    if getattr(policy, "record_replay_features", False)
                    else None,
                    completed,
                )
            # An input arriving during observation/result handling still wins.
            if stop_requested is not None and stop_requested(completed):
                break
            if generation != control.generation:
                actions, index = np.empty((0, 7)), 0
                generation = control.generation
            if (
                policy_selection is not None
                and route_generation != policy_selection.generation
            ):
                actions, index = np.empty((0, 7)), 0
                route_generation = policy_selection.generation
            policy_target = actions[index] if index < len(actions) else None
            if (
                policy_target is not None
                and time.monotonic() - prediction_time
                > session.config.max_prediction_age_s
            ):
                raise TimeoutError("Piper policy prediction expired before execution.")
            if policy_target is not None and control.mode == "release_pending":
                # Accept only after the final input poll and route invalidation.
                # A discarded result must not end human-owned release waiting.
                if control.accept_prediction(generation):
                    resume_generation = control.generation
            target, source = control.choose(
                policy_target,
                observation["observation.state"],
                source=action_source,
            )
            measured = session.execute(target)
            log.write(
                json.dumps(
                    {
                        "event": "action",
                        "step": completed,
                        "state": measured.tolist(),
                        "target": target.tolist(),
                        "command": session.last_command.tolist(),
                        "time": time.time(),
                        "action_source": source,
                        "policy_version": chunk_version
                        if source in ("reference", "actor")
                        else -1,
                        "is_intervention": source == "human",
                        "control_generation": control.generation,
                        "route_generation": route_generation,
                        "prediction_age_s": time.monotonic() - prediction_time
                        if source in ("reference", "actor")
                        else None,
                    }
                )
                + "\n"
            )
            log.flush()
            if record_step is not None:
                record_step(
                    observation,
                    session.last_command,
                    requested=target,
                    measured=measured,
                    wall_time=wall_time,
                    is_intervention=source == "human",
                    action_source=source,
                    policy_version=chunk_version
                    if source in ("reference", "actor")
                    else -1,
                    policy_action=policy_target
                    if source in ("reference", "actor")
                    else None,
                )
            completed += 1
            control.executed(session.last_command, source)
            if source in ("reference", "actor"):
                if resume_generation == control.generation:
                    resume_generation = None
                    if on_policy_resume is not None:
                        on_policy_resume()
                index += 1
            time.sleep(max(0, period - (time.monotonic() - started)))
    finally:
        try:
            control.stop()
        finally:
            # The job owns only observation copies and the model, never hardware.
            executor.shutdown(wait=True, cancel_futures=True)
    return completed
