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

"""Controllable external speech process; no desktop audio or hardware is used."""

import json
import sys
import time
from pathlib import Path
from typing import Any


class SpeechDispatcher:
    """Expose a fake spd-say executable while keeping real process lifecycle tests."""

    def __init__(
        self, root: Path, *, blocked: bool = False, exit_code: int = 0
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (root / "exit-code").write_text(str(exit_code))
        if blocked:
            (root / "block").touch()
        executable = root / "spd-say"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, time\n"
            "from pathlib import Path\n"
            "root = Path(sys.argv[0]).parent\n"
            "with (root / 'calls.jsonl').open('a') as stream:\n"
            "    stream.write(json.dumps({'pid': os.getpid(), 'args': sys.argv[1:]}) + '\\n')\n"
            "deadline = time.monotonic() + 30\n"
            "while (root / 'block').exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "sys.exit(int((root / 'exit-code').read_text()))\n"
        )
        executable.chmod(0o700)

    @property
    def calls(self) -> list[dict[str, Any]]:
        path = self.root / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().split("\n")[:-1]]

    def wait_for_calls(self, count: int, *, timeout: float = 3) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            calls = self.calls
            if len(calls) >= count:
                return calls
            time.sleep(0.01)
        raise AssertionError(f"Expected {count} speech calls, got {self.calls}")

    def release(self) -> None:
        (self.root / "block").unlink(missing_ok=True)
