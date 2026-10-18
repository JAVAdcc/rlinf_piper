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

"""Lossless frame delivery to background H.264 encoders for LeRobot collection.

Like Evo-RL, each camera encodes directly from RGB arrays. Queue saturation
fails the recording instead of dropping a frame and shifting action labels.
"""

import queue
import shutil
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class RecordingDiscarded(Exception):
    """The operator cancelled the current recording."""


class _VideoStream:
    def __init__(self, root: Path, fps: int, cancelled: threading.Event) -> None:
        self.path = (
            Path(tempfile.mkdtemp(prefix="streaming-video-", dir=root)) / "video.mp4"
        )
        self.fps = fps
        self.cancelled = cancelled
        self.stop = threading.Event()
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=32)
        self.error: BaseException | None = None
        self.frames = 0
        self.accepted = 0
        self.stats: dict | None = None
        self.thread = threading.Thread(
            target=self._run, name="piper-video", daemon=True
        )
        try:
            self.thread.start()
        except BaseException:
            shutil.rmtree(self.path.parent)
            raise

    def _check(self) -> None:
        if self.cancelled.is_set():
            raise RecordingDiscarded
        if self.error is not None:
            raise RuntimeError(f"Video encoding failed: {self.path}") from self.error

    def put(self, frame: np.ndarray | None, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self._check()
            if not self.thread.is_alive():
                raise RuntimeError("Video encoder stopped before accepting a frame.")
            try:
                self.queue.put(frame, timeout=0.05)
                self.accepted += int(frame is not None)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Video encoder cannot keep up; recording stopped without dropping frames."
                    ) from None

    def _run(self) -> None:
        container = None
        stream = None
        shape = None
        try:
            from lerobot.datasets.compute_stats import RunningQuantileStats

            tracker = RunningQuantileStats()
            while not self.stop.is_set() and not self.cancelled.is_set():
                try:
                    rgb = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if rgb is None:
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                        self.stats = tracker.get_statistics()
                    break
                if container is None:
                    shape = rgb.shape
                    container = av.open(str(self.path), "w")
                    stream = container.add_stream(
                        "libx264",
                        rate=self.fps,
                        options={
                            "preset": "veryfast",
                            "crf": "30",
                            "g": "2",
                            "threads": "2",
                        },
                    )
                    stream.width, stream.height = rgb.shape[1], rgb.shape[0]
                    stream.pix_fmt = "yuv420p"
                if rgb.shape != shape:
                    raise ValueError("Camera resolution changed during recording.")
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = self.frames, Fraction(1, self.fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
                stride = max(1, (max(rgb.shape[:2]) + 149) // 150)
                tracker.update(rgb[::stride, ::stride].reshape(-1, 3))
                self.frames += 1
        except BaseException as exc:
            self.error = exc
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception as exc:
                    self.error = self.error or exc

    def finish(self) -> tuple[Path, dict]:
        self.put(None, timeout=10)
        self.thread.join(timeout=10)
        self._check()
        if (
            self.thread.is_alive()
            or not self.frames
            or self.frames != self.accepted
            or self.stats is None
        ):
            raise RuntimeError("Video encoder did not finish every submitted frame.")
        with av.open(str(self.path)) as video:
            if video.streams.video[0].frames != self.accepted:
                raise RuntimeError(
                    "Encoded video frame count differs from recorded actions."
                )
        return self.path, self.stats

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError(f"Video encoder has not stopped; preserving {self.path}")
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.path.parent.exists():
            shutil.rmtree(self.path.parent)


class PiperVideoEncoder:
    """Implement LeRobot's streaming encoder contract with bounded, exact delivery.

    The dataset owns this encoder and calls start/feed/finish/close. Only
    setting ``cancelled`` is allowed from another thread; dataset mutation
    and cleanup remain with the recording owner.
    """

    def __init__(self, fps: int, cancelled: threading.Event) -> None:
        self.fps, self.cancelled = fps, cancelled
        self.streams: dict[str, _VideoStream] = {}
        self.finished = False

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        """Own one independent encoder and temporary file per camera."""
        if self.streams or self.finished:
            raise RuntimeError("Piper encoders record exactly one episode.")
        try:
            for key in video_keys:
                self.streams[key] = _VideoStream(temp_dir, self.fps, self.cancelled)
        except BaseException:
            self.close()
            raise

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """Copy a frame into the bounded queue, failing instead of dropping it."""
        if self.finished:
            raise RuntimeError("Video encoding is finished.")
        rgb = np.asarray(image)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("Video requires HWC or CHW RGB frames.")
        if rgb.dtype != np.uint8:
            if (
                not np.issubdtype(rgb.dtype, np.floating)
                or not np.isfinite(rgb).all()
                or rgb.min() < 0
                or rgb.max() > 1
            ):
                raise ValueError("Video requires uint8 RGB or finite RGB in [0, 1].")
            rgb = np.rint(rgb * 255).astype(np.uint8)
        self.streams[video_key].put(np.ascontiguousarray(rgb).copy(), timeout=1)

    def finish_episode(self) -> dict[str, tuple[Path, dict]]:
        """Flush all videos and return their validated paths and RGB statistics."""
        if self.finished or not self.streams:
            raise RuntimeError("No unfinished video episode.")
        results = {key: stream.finish() for key, stream in self.streams.items()}
        self.finished = True
        return results

    def cancel_episode(self) -> None:
        """Stop writers before removing their unpublished temporary files."""
        error = None
        for stream in self.streams.values():
            try:
                stream.close()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error
        self.streams.clear()

    def close(self) -> None:
        """Release all remaining workers and files; safe to call repeatedly."""
        self.cancel_episode()
