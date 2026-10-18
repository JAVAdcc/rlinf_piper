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

"""Bounded background speech for operator-scored Piper collection."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import TracebackType

_LOG = logging.getLogger(__name__)

# Use the same local Speech Dispatcher backend as Evo-RL's log_say.
# Each topic retains only its latest pending state; results remain independent.
_MESSAGES = {
    "resumed": (
        "resume",
        "Collection resumed with {number} saved episodes.",
        "继续采集，已保留{number}条数据。",
    ),
    "resume_reset": (
        "reset",
        "Previous reset unfinished. Check the workspace, then press R to reset.",
        "上次复位未完成，确认工作区后按 R 复位。",
    ),
    "ready": (
        "episode",
        "Ready for episode {number}. Press right.",
        "准备采集第{number}回合，按右方向键开始。",
    ),
    "aligning": ("episode", "Aligning the leader arm.", "正在对齐主臂。"),
    "started": ("episode", "Starting episode {number}.", "开始第{number}回合。"),
    "review": (
        "episode",
        "Collection target reached. Right to finish. Left to discard the last episode.",
        "已采够回合数，右键结束，左键丢弃重采。",
    ),
    "discarded": (
        "discard",
        "Episode {number} discarded.",
        "第{number}回合已丢弃。",
    ),
    "nothing_to_discard": (
        "discard",
        "No episode available to discard.",
        "当前没有可以丢弃的回合。",
    ),
    "critical_on": (
        "phase",
        "Critical phase {number} on.",
        "进入第{number}段关键阶段。",
    ),
    "critical_off": (
        "phase",
        "Critical phase {number} off.",
        "退出第{number}段关键阶段。",
    ),
    "critical_closed": (
        "phase_result",
        "Last critical phase closed at episode end.",
        "最后一段关键阶段已自动延长到回合结束。",
    ),
    "critical_empty": (
        "phase",
        "Empty critical phase ignored.",
        "忽略空的关键阶段标记。",
    ),
    "phase_limit": (
        "phase",
        "Step limit. Cannot start a critical phase.",
        "已到步数上限，不能开始新的关键阶段。",
    ),
    "human": ("control", "Human control.", "人工接管。"),
    "policy_wait": ("control", "Waiting for a new prediction.", "等待模型的新预测。"),
    "policy_resumed": ("control", "Policy control resumed.", "模型控制已恢复。"),
    "waiting_score": (
        "episode",
        "Step limit. Press S or F to score.",
        "已到步数上限，请按 S 或 F 评分。",
    ),
    "success": (
        "outcome",
        "Episode {number} marked successful.",
        "第{number}回合已标记成功。",
    ),
    "failure": (
        "outcome",
        "Episode {number} marked failed.",
        "第{number}回合已标记失败。",
    ),
    "saving": ("recording", "Saving episode {number}.", "正在保存第{number}回合。"),
    "saved": ("recording", "Episode {number} saved.", "第{number}回合数据已保存。"),
    "no_critical": (
        "annotation",
        "No critical phase marked.",
        "本回合没有关键阶段标注。",
    ),
    "reset_start": ("reset", "Resetting the arm.", "开始复位。"),
    "reset_done": (
        "reset_result",
        "Reset trajectory finished. Check the pose.",
        "复位轨迹已执行，请检查姿态。",
    ),
    "reset_failed": (
        "reset",
        "Reset failed. Press R to retry, or Q to exit.",
        "复位失败，按 R 重试，或按 Q 退出。",
    ),
    "stopped": ("terminal", "Collection stopped.", "采集已停止。"),
    "complete": ("terminal", "Collection complete.", "采集完成。"),
    "fault": (
        "terminal",
        "Collection stopped after an error.",
        "发生故障，采集已停止。",
    ),
}
_TRANSIENT = {"episode", "phase", "control", "reset"}


@dataclass(frozen=True)
class _Cue:
    event: str
    topic: str
    text: str
    expires: float


class PiperAnnouncer:
    """Speak collection events without doing process I/O in the control thread.

    Enter once before opening hardware, then call with an event and optional
    episode/phase number. Close after hardware cleanup: allow eight seconds for
    final results on completion, or three seconds for a stop/fault announcement.
    Missing or failed speech disables audio for this instance and logs once.
    An utterance already submitted to Speech Dispatcher may finish playing;
    global stop/cancel commands are never sent to other desktop applications.
    """

    def __init__(self, *, enabled: bool = True, language: str = "en") -> None:
        if language not in ("en", "zh"):
            raise ValueError("Speech language must be en or zh.")
        self.enabled = enabled
        self.language = language
        self._condition = threading.Condition()
        self._pending: deque[_Cue] = deque(maxlen=8)
        self._abort = threading.Event()
        self._thread: threading.Thread | None = None
        self._entered = False
        self._closing = False
        self._failed = False
        self._last: tuple[str, int] | None = None
        self._terminal: str | None = None
        self._executable: str | None = None

    def __enter__(self) -> PiperAnnouncer:
        if self._entered or self._closing:
            raise RuntimeError("A PiperAnnouncer can only be entered once.")
        self._entered = True
        if not self.enabled:
            return self
        try:
            self._executable = shutil.which("spd-say")
            if self._executable is None:
                self._disable(
                    "spd-say is unavailable; install speech-dispatcher or use --no-play-sounds"
                )
                return self
            thread = threading.Thread(
                target=self._run, name="piper-speech", daemon=True
            )
            thread.start()
            self._thread = thread
        except Exception as exc:
            self._disable(f"{type(exc).__name__}: {exc}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self("stopped" if issubclass(exc_type, KeyboardInterrupt) else "fault")
        self.close()

    def __call__(self, event: str, number: int = 0) -> None:
        """Enqueue the latest state for a topic, discarding superseded cues."""
        if not self.enabled or self._failed:
            return
        topic, english, chinese = _MESSAGES[event]
        text = (english if self.language == "en" else chinese).format(number=number)
        with self._condition:
            if self._closing or self._last == (event, number):
                return
            self._last = (event, number)
            if topic == "terminal":
                # Completion must not erase the last score, saved-data status,
                # or reset result. Stop/fault takes priority over queued history.
                self._pending = deque(
                    (
                        cue
                        for cue in self._pending
                        if event == "complete" and cue.topic not in _TRANSIENT
                    ),
                    maxlen=8,
                )
                self._terminal = event
            else:
                if self._terminal:
                    return
                discard = {topic}
                if event == "discarded":
                    discard |= {
                        "outcome",
                        "recording",
                        "annotation",
                        "phase_result",
                    } | _TRANSIENT
                if event in (
                    "ready",
                    "success",
                    "failure",
                    "reset_start",
                    "reset_failed",
                ):
                    discard |= _TRANSIENT
                self._pending = deque(
                    (cue for cue in self._pending if cue.topic not in discard),
                    maxlen=8,
                )
            lifetime = 3.0 if topic in _TRANSIENT else 10.0
            self._pending.append(_Cue(event, topic, text, time.monotonic() + lifetime))
            self._condition.notify()

    def _disable(self, reason: str) -> None:
        with self._condition:
            first_failure = not self._failed
            self._failed = True
            self._pending.clear()
        if first_failure:
            _LOG.warning(
                "Piper speech disabled: %s. Collection continues with terminal messages.",
                reason,
            )

    def _speak(self, cue: _Cue) -> None:
        # --wait serializes playback only in this worker. The wait is bounded,
        # and the worker owns and reaps every child even when collection exits.
        assert self._executable is not None
        process = subprocess.Popen(
            [
                self._executable,
                "--wait",
                "--language",
                "en-US" if self.language == "en" else "cmn",
                cue.text,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 8.0
        try:
            while not self._abort.is_set():
                try:
                    code = process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "speech playback exceeded eight seconds"
                        ) from None
                else:
                    if code:
                        raise RuntimeError(f"spd-say exited with status {code}")
                    return
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=0.5)

    def _run(self) -> None:
        try:
            while not self._abort.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closing)
                    if not self._pending:
                        return
                    cue = self._pending.popleft()
                if time.monotonic() < cue.expires:
                    self._speak(cue)
        except Exception as exc:
            self._disable(f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        """Discard pending states and stop the owned worker after hardware closes."""
        with self._condition:
            if self._closing:
                return
            self._closing = True
            if self._terminal != "complete":
                self._pending = deque(
                    (cue for cue in self._pending if cue.topic == "terminal"), maxlen=8
                )
            if not self._terminal:
                self._abort.set()
            self._condition.notify_all()
        if self._thread is not None:
            timeout = 0.0
            if self._terminal == "complete":
                timeout = 8.0
            elif self._terminal:
                timeout = 3.0
            try:
                self._thread.join(timeout=timeout)
            finally:
                self._abort.set()
                self._thread.join(timeout=0.75)
