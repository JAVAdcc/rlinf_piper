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

"""Three-view Piper task for RLinf's native RLT online actor-critic loop."""

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.models.embodiment.openpi.policies.piper_policy import (
    PIPER_CAMERAS,
    PiperActionCodec,
)
from rlinf.robotics.piper_intervention import PiperIntervention
from rlinf.robotics.piper_runtime import build_piper_session
from rlinf.utils.logging import get_logger


@dataclass
class PiperPolicyConfig:
    """Task and asset paths for a manually reset, operator-scored Piper task."""

    norm_stats_path: str
    hardware_path: str
    task: str
    is_dummy: bool = False
    max_num_steps: int = 300
    step_frequency: float = 30.0
    image_width: int = 640
    image_height: int = 480


class PiperPolicyEnv(gym.Env):
    """Execute normalized RLT targets; expose raw RGB and degree/mm state.

    The operator arranges the scene between episodes. Keys: a starts, b toggles
    the next chunk's critical phase, i toggles human control when configured,
    c reports success, q reports failure. Human ownership overrides either route.
    Connecting enables motors. Reset sends no arm trajectory.
    """

    TELEOP = ()
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()
    REFUSE_FLAGS = ("keyboard_reward_wrapper", "keyboard_intervention_wrapper")

    def __init__(
        self,
        override_cfg: dict,
        worker_info: Any = None,
        robot_info: Any = None,
        env_idx: int = 0,
    ):
        del worker_info, robot_info
        if env_idx != 0:
            raise ValueError("Use exactly one Piper task environment per workstation.")
        self.config = PiperPolicyConfig(**override_cfg)
        if (
            self.config.max_num_steps <= 0
            or not np.isfinite(self.config.step_frequency)
            or self.config.step_frequency <= 0
        ):
            raise ValueError("Use a positive episode length and control frequency.")
        if not self.config.task.strip():
            raise ValueError("Specify the Piper task description.")
        self._logger = get_logger()
        self.codec = PiperActionCodec(self.config.norm_stats_path)
        self.session = None
        self.listener = None
        self._steps = 0
        self._actor = False
        self._outcome = None
        self._running = False
        self._terminal = None
        self._chunk_generation = 0
        self._chunk_actor = False
        self._dummy_state = np.zeros(7, dtype=np.float32)
        frame = gym.spaces.Box(
            0,
            255,
            shape=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {"joint_gripper": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32)}
                ),
                "frames": gym.spaces.Dict(
                    dict.fromkeys(("front", "front_2", "wrist"), frame)
                ),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        if not self.config.is_dummy:
            from rlinf.envs.real.wrappers.episode.keyboard import KeyboardListener

            self.session = build_piper_session(self.config.hardware_path)
            if not np.isclose(self.session.config.fps, self.config.step_frequency):
                raise ValueError("Hardware and task control frequencies must match.")
            self.listener = KeyboardListener()

    @property
    def task_description(self) -> str:
        """Return the instruction supplied to the frozen feature model."""
        return self.config.task

    def set_actor_enabled(self, enabled: bool) -> None:
        """Select the next chunk's reference/actor route without changing replay units."""
        self._actor = bool(enabled)

    @property
    def intervention(self) -> PiperIntervention | None:
        """Return the session-owned controller, if HIL is configured."""
        return getattr(self.session, "intervention", None)

    def on_action_chunk_begin(self) -> None:
        """Accept a newly inferred chunk; never release into an old chunk."""
        self._chunk_actor = self._actor
        if self.intervention is not None:
            self._chunk_generation = self.intervention.generation
            self.intervention.accept_prediction(self._chunk_generation)

    def finish_episode(self, success: bool) -> None:
        """Set the terminal outcome before the next control tick."""
        self._outcome = bool(success)

    def _observation(self) -> dict[str, Any]:
        if self.config.is_dummy:
            return {
                "state": {"joint_gripper": self._dummy_state.copy()},
                "frames": {
                    name: np.zeros(
                        (self.config.image_height, self.config.image_width, 3), np.uint8
                    )
                    for name in ("front", "front_2", "wrist")
                },
            }
        raw = self.session.observe()
        return {
            "state": {"joint_gripper": raw["observation.state"]},
            "frames": {
                name: raw[next(key for key in aliases if key in raw)]
                for name, aliases in zip(
                    ("front", "front_2", "wrist"), PIPER_CAMERAS.values(), strict=True
                )
            },
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Wait for the operator to prepare the scene and start an episode."""
        super().reset(seed=seed)
        self._steps = 0
        self._actor = False
        self._outcome = None
        self._running = False
        self._terminal = None
        if not self.config.is_dummy:
            if not self.session.is_connected:
                self.session.__enter__()
            self.listener.pop_pressed_keys()
            self._logger.info(
                "Piper holds its pose. Arrange the scene; a starts, b toggles the next chunk's critical phase, c succeeds, q fails."
            )
            while "a" not in self.listener.pop_pressed_keys():
                time.sleep(0.05)
            if self.intervention is not None:
                self.intervention.align(
                    self.session.clip_reset_target(self.session.read_state()),
                    self.config.step_frequency,
                )
        self._running = True
        self.on_action_chunk_begin()
        return self._observation(), {"rlt_switch_flags": False}

    def get_hold_action(self, fallback: np.ndarray | None = None) -> np.ndarray:
        """Encode the measured pose for the runner's chunk pause/hold path."""
        del fallback
        return self.codec.encode(self._observation()["state"]["joint_gripper"])

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one normalized target and return the operator-scored transition."""
        try:
            return self._step(action)
        except BaseException:
            if self.intervention is not None:
                self.intervention.stop()
            self._running = False
            raise

    def _step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if not self._running:
            if self._terminal is not None:
                obs, _, terminated, truncated, info = self._terminal
                padding = dict(info, action_valid=False, action_source="hold")
                padding.pop("intervene_action", None)
                return obs, 0.0, terminated, truncated, padding
            raise RuntimeError("Reset the episode before sending Piper actions.")
        started = time.monotonic()
        if self.listener is not None:
            for key in self.listener.pop_pressed_keys():
                if key == "b":
                    self.set_actor_enabled(not self._actor)
                elif (
                    self.intervention is not None
                    and key == self.intervention.config.key
                ):
                    self.intervention.toggle(self.session.read_state())
                elif key in ("c", "q"):
                    self.finish_episode(key == "c")
        terminated = self._outcome is not None
        source = "hold"
        command = None
        if terminated:
            command = getattr(self.session, "last_command", None)
            if command is None:
                command = self._observation()["state"]["joint_gripper"]
        if not terminated:
            control = self.intervention
            uses_policy = control is None or (
                control.mode == "policy"
                and self._chunk_generation == control.generation
            )
            target = self.codec.decode(np.asarray(action)) if uses_policy else None
            if target is not None and target.shape != (7,):
                raise ValueError(
                    "PiperPolicyEnv.step requires a seven-dimensional action."
                )
            if self.config.is_dummy:
                self._dummy_state = target.astype(np.float32)
                command = self._dummy_state.copy()
                source = "actor" if self._chunk_actor else "reference"
            else:
                source = "actor" if self._chunk_actor else "reference"
                if self.intervention is not None:
                    target, source = self.intervention.choose(
                        target
                        if self._chunk_generation == self.intervention.generation
                        else None,
                        self.session.read_state(),
                        source=source,
                    )
                self.session.execute(target)
                command = self.session.last_command.copy()
                if self.intervention is not None:
                    self.intervention.executed(command, source)
            self._steps += 1
            time.sleep(
                max(0, 1 / self.config.step_frequency - (time.monotonic() - started))
            )
        truncated = not terminated and self._steps >= self.config.max_num_steps
        self._running = not (terminated or truncated)
        if not self._running and self.intervention is not None:
            self.intervention.stop()
        info = {
            "rlt_switch_flags": self._actor,
            "success": self._outcome is True,
            "executed_action": self.codec.encode(command),
            # The upstream route owns actor-vs-reference identity (warmup can
            # override the requested actor route). Only it can label that split.
            "action_source": "policy" if source in ("reference", "actor") else source,
            "action_valid": not terminated,
        }
        if source == "human":
            info["intervene_action"] = info["executed_action"].copy()
        result = (
            self._observation(),
            float(self._outcome is True),
            terminated,
            truncated,
            info,
        )
        if not self._running:
            self._terminal = result
        return result

    def close(self) -> None:
        """Release camera, robot, and keyboard resources without a reset motion."""
        try:
            if self.session is not None:
                self.session.close()
        finally:
            if self.listener is not None:
                self.listener.close()
            self._running = False
