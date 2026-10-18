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

"""Record one Piper trial as an independently readable LeRobot dataset."""

import inspect
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rlinf.data.storage.lerobot.compat import add_frame_to_dataset
from rlinf.data.storage.lerobot.video import PiperVideoEncoder, RecordingDiscarded
from rlinf.models.embodiment.openpi.policies.piper_policy import PiperInputs
from rlinf.robotics.piper_intervention import ACTION_SOURCES


class PiperEpisodeRecorder:
    """Own image writers and one episode, finalized before the next robot reset.

    Each action row contains the observation before the command. The final
    row is a measured terminal observation with ``action_valid=False``; its
    placeholder action must never become a training target. Video frames are
    encoded during collection, with exact frame delivery and bounded queues.
    """

    def __init__(
        self, root: Path, *, fps: float, task: str, streaming_video: bool = True
    ) -> None:
        if fps <= 0 or not float(fps).is_integer() or not task.strip():
            raise ValueError("Recording requires integer FPS and a task.")
        if root.exists():
            raise FileExistsError(root)
        self.root, self.fps, self.task = root, int(fps), task
        self.dataset = None
        self.steps = 0
        self.human_steps = 0
        self.closed = False
        self.streaming_video = streaming_video
        self.cancelled = threading.Event()

    def _create(self, observation: dict) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        PiperInputs()(observation)
        features = {
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "complementary_info.requested_action": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            "complementary_info.command_state": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            "complementary_info.action_valid": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            "complementary_info.critical_phase": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            "complementary_info.is_intervention": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            "complementary_info.wall_time": {
                "dtype": "float64",
                "shape": (1,),
                "names": None,
            },
            "complementary_info.action_source": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "complementary_info.policy_action": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            "complementary_info.policy_action_valid": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
        }
        for key, value in observation.items():
            if key.startswith("observation.images."):
                features[key] = {
                    "dtype": "video",
                    "shape": tuple(value.shape),
                    "names": ["height", "width", "channels"],
                }
        options = {}
        if self.streaming_video:
            if (
                "streaming_encoding"
                not in inspect.signature(LeRobotDataset.create).parameters
            ):
                raise RuntimeError(
                    "This LeRobot release lacks streaming encoding; update LeRobot or use --no-streaming-video."
                )
            options["streaming_encoding"] = True
        self.dataset = LeRobotDataset.create(
            repo_id="piper/rollout",
            root=self.root,
            robot_type="piper",
            fps=self.fps,
            features=features,
            vcodec="h264",
            image_writer_threads=0 if self.streaming_video else 6,
            image_writer_processes=0,
            **options,
        )
        if self.streaming_video:
            # LeRobot owns publication and metadata. Replace its dormant encoder
            # because the upstream queue drops frames when it cannot keep up.
            self.dataset._streaming_encoder = PiperVideoEncoder(
                self.fps, self.cancelled
            )

    def prepare(self, observation: dict) -> None:
        """Open writers before the first policy action or prediction timer."""
        if self.dataset is not None or self.closed:
            raise RuntimeError("Episode recorder already prepared or closed.")
        self._create(observation)

    def _append(
        self,
        observation: dict,
        action: np.ndarray,
        *,
        requested: np.ndarray,
        measured: np.ndarray,
        critical: bool,
        valid: bool,
        wall_time: float,
        is_intervention: bool = False,
        action_source: str | None = None,
        policy_action: np.ndarray | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("Episode recording is closed.")
        if self.dataset is None:
            self._create(observation)
        frame = {key: np.asarray(value).copy() for key, value in observation.items()}
        source = action_source or ("human" if is_intervention else "reference")
        if source not in ACTION_SOURCES or is_intervention != (source == "human"):
            raise ValueError("Piper action source and intervention flag disagree.")
        if valid and source in ("reference", "actor") and policy_action is None:
            policy_action = requested
        policy_valid = valid and policy_action is not None
        frame["observation.state"] = frame["observation.state"].astype(np.float32)
        frame.update({
            "action": np.asarray(action, dtype=np.float32),
            "task": self.task,
            "complementary_info.requested_action": np.asarray(
                requested, dtype=np.float32
            ),
            "complementary_info.command_state": np.asarray(measured, dtype=np.float32),
            "complementary_info.action_valid": np.array([valid], dtype=bool),
            "complementary_info.critical_phase": np.array([critical], dtype=bool),
            "complementary_info.is_intervention": np.array(
                [is_intervention], dtype=bool
            ),
            "complementary_info.action_source": np.array(
                [ACTION_SOURCES[source]], dtype=np.int64
            ),
            "complementary_info.policy_action": np.asarray(
                policy_action if policy_valid else np.zeros(7),
                dtype=np.float32,
            ),
            "complementary_info.policy_action_valid": np.array(
                [policy_valid], dtype=bool
            ),
            "complementary_info.wall_time": np.array([wall_time], dtype=np.float64),
        })
        add_frame_to_dataset(self.dataset, frame)

    def add_step(
        self,
        observation: dict,
        action: np.ndarray,
        *,
        requested: np.ndarray,
        measured: np.ndarray,
        critical: bool,
        wall_time: float,
        is_intervention: bool = False,
        action_source: str | None = None,
        policy_action: np.ndarray | None = None,
    ) -> None:
        """Record a successfully submitted command and its pre-command observation."""
        self._append(
            observation,
            action,
            requested=requested,
            measured=measured,
            critical=critical,
            valid=True,
            wall_time=wall_time,
            is_intervention=is_intervention,
            action_source=action_source,
            policy_action=policy_action,
        )
        self.steps += 1
        self.human_steps += int(is_intervention)

    def request_discard(self) -> None:
        """Request cancellation without mutating the dataset from the input thread."""
        self.cancelled.set()

    def finish(
        self,
        observation: dict,
        *,
        critical: bool,
        outcome: str,
        poll: Callable[[], None] | None = None,
    ) -> dict:
        """Finish on a worker while the caller optionally services terminal input."""
        if poll is None:
            return self._finish(observation, critical=critical, outcome=outcome)
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="piper-save"
        ) as worker:
            future = worker.submit(
                self._finish, observation, critical=critical, outcome=outcome
            )
            try:
                while not future.done():
                    poll()
                    time.sleep(0.02)
                poll()  # A final Left still applies when saving completed quickly.
                return future.result()
            except BaseException:
                self.request_discard()
                raise

    def _finish(self, observation: dict, *, critical: bool, outcome: str) -> dict:
        """Append the terminal observation and publish a readable dataset."""
        try:
            if self.cancelled.is_set():
                raise RecordingDiscarded
            state = np.asarray(observation["observation.state"], dtype=np.float32)
            self._append(
                observation,
                state,
                requested=state,
                measured=state,
                critical=critical,
                valid=False,
                wall_time=time.time(),
                action_source="hold",
            )
            self.dataset.save_episode(parallel_encoding=False)
            self.dataset.finalize()
            if self.cancelled.is_set():
                raise RecordingDiscarded
            (self.root / "meta/piper_control.json").write_text(
                json.dumps(
                    {"action_sources": ACTION_SOURCES, "human_steps": self.human_steps},
                    indent=2,
                )
                + "\n"
            )
            # Preserve explicit outcomes for the existing LeRobot metadata reader.
            with (self.root / "meta/episodes.jsonl").open("x") as stream:
                stream.write(
                    json.dumps({
                        "episode_index": 0,
                        "length": self.steps + 1,
                        "episode_success": outcome == "success",
                    })
                    + "\n"
                )
            return {
                "status": "complete",
                "frames": self.steps + 1,
                "terminal_frame": self.steps,
                "dataset": str(self.root.resolve()),
                "human_steps": self.human_steps,
                "video_encoding": "streaming" if self.streaming_video else "png",
            }
        except RecordingDiscarded:
            return {"status": "discarded"}
        finally:
            self.close()

    def close(self) -> None:
        """Release image writers; incomplete datasets remain outside annotations."""
        if self.closed:
            return
        if self.dataset is not None:
            try:
                self.dataset.stop_image_writer()
            finally:
                self.dataset.finalize()
        self.closed = True
