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

"""Delay or fail the external video file boundary while using real codecs."""

from contextlib import contextmanager
from threading import Event
from typing import Iterator
from unittest.mock import patch

import av


@contextmanager
def video_output(*, fail: bool = False) -> Iterator[tuple[Event, Event]]:
    entered, release = Event(), Event()
    original = av.open

    def open_video(path, mode="r", *args, **kwargs):
        if mode == "w":
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the video writer.")
            if fail:
                raise OSError("Video disk unavailable")
        return original(path, mode, *args, **kwargs)

    with patch.object(av, "open", open_video):
        try:
            yield entered, release
        finally:
            release.set()
