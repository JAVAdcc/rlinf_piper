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

"""Bounded background preparation and learning for accepted Piper episodes."""

import copy
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager


class BackgroundStopped(Exception):
    """Cancel unfinished feature extraction; the accepted recording stays durable."""


class InferenceGate:
    """Serialize rollout RPCs, giving waiting inference priority over backfill."""

    def __init__(self):
        self._condition = threading.Condition()
        self._busy = False
        self._inference_waiters = 0

    @contextmanager
    def request(self, *, background: bool = False, cancelled: Callable = lambda: False):
        """Allow at most one bounded GPU request; never preempt an active forward."""
        with self._condition:
            if not background:
                self._inference_waiters += 1
            try:
                while self._busy or (background and self._inference_waiters):
                    if cancelled():
                        raise BackgroundStopped()
                    self._condition.wait(0.05)
                if cancelled():
                    raise BackgroundStopped()
                self._busy = True
            finally:
                if not background:
                    self._inference_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()


class PiperAsyncPipeline:
    """Own one preparation thread and one learner thread with one ready slot.

    Accepted recordings are the durable input journal. ``update`` publishes a
    complete checkpoint before returning whether more update budget remains.
    Closing asks the active round to checkpoint after its bounded update batch,
    leaving unprocessed inputs and remaining budget for resume. Background failures
    propagate to the operator.
    """

    def __init__(self, prepare: Callable, update: Callable):
        self._prepare = prepare
        self._update = update
        self._condition = threading.Condition()
        self._pending = deque()
        self._ready = deque()
        self._known = set()
        self._preparing = self._updating = False
        self._budget = True  # Check restored learner budget on startup.
        self._error = None
        self.stopped = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._prepare_loop, name="piper-cache", daemon=True
            ),
            threading.Thread(
                target=self._learn_loop, name="piper-learner", daemon=True
            ),
        ]

    def start(self) -> None:
        """Start after worker initialization and the initial durable checkpoint."""
        for thread in self._threads:
            thread.start()

    def submit(self, directory, result: dict) -> None:
        """Enqueue a committed-to-keep episode without waiting for extraction."""
        with self._condition:
            self.check_health()
            if self.stopped.is_set():
                raise RuntimeError("Async learner is closed.")
            if directory.name not in self._known:
                self._known.add(directory.name)
                self._pending.append((directory, copy.deepcopy(result)))
                self._condition.notify_all()

    def check_health(self) -> None:
        """Surface a background fault before another control tick or confirmation."""
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    "Piper background learning failed; resume after fixing the fault."
                ) from self._error

    def wait_idle(self, timeout: float = 120.0) -> None:
        """Wait for accepted inputs and earned budget, primarily for validation."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._pending
                or self._ready
                or self._preparing
                or self._updating
                or self._budget
            ):
                self.check_health()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Background learning did not become idle.")
                self._condition.wait(min(remaining, 0.1))
            self.check_health()

    def close(self) -> None:
        """Stop after the active learner cycle; do not drain an unlimited budget."""
        self.stopped.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=60)
                if thread.is_alive():
                    raise TimeoutError(f"{thread.name} did not stop.")
        self.check_health()

    def _failed(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            self.stopped.set()
            self._condition.notify_all()

    def _prepare_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stopped.is_set()
                        or (self._pending and not self._ready)
                    )
                    if self.stopped.is_set():
                        return
                    directory, result = self._pending.popleft()
                    self._preparing = True
                trajectory = self._prepare(directory, result)
                with self._condition:
                    self._preparing = False
                    if self.stopped.is_set():
                        return
                    self._ready.append((directory, result, trajectory))
                    self._condition.notify_all()
        except BackgroundStopped:
            if not self.stopped.is_set():
                self._failed(
                    RuntimeError("Feature preparation unexpectedly cancelled.")
                )
        except BaseException as error:
            self._failed(error)

    def _learn_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.stopped.is_set() or self._ready or self._budget
                    )
                    if self.stopped.is_set():
                        return
                    item = self._ready.popleft() if self._ready else None
                    self._updating = True
                    self._condition.notify_all()
                budget = self._update(item)
                with self._condition:
                    self._budget = bool(budget)
                    self._updating = False
                    self._condition.notify_all()
        except BaseException as error:
            self._failed(error)
