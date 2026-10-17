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

"""LeRobot camera lifecycle with an ordered shutdown for blocked V4L2 reads."""

import os
import subprocess
import time
from pathlib import Path

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from rlinf.utils.logging import get_logger


def reset_piper_camera(
    usb_serial: str,
    *,
    usb_sysfs: Path = Path("/sys/bus/usb/devices"),
    video_sysfs: Path = Path("/sys/class/video4linux"),
    device_root: Path = Path("/dev"),
) -> None:
    """Reset exactly one idle USB camera, leaving CAN and other devices alone."""
    matches = [
        p
        for p in usb_sysfs.iterdir()
        if (p / "idVendor").is_file()
        and (p / "serial").is_file()
        and (p / "serial").read_text().strip() == usb_serial
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one USB device for camera {usb_serial}; found {matches}."
        )
    usb = matches[0].resolve()
    target = f"{int((usb / 'busnum').read_text()):03d}/{int((usb / 'devnum').read_text()):03d}"
    node = device_root / "bus/usb" / target
    video_nodes = [
        device_root / p.name
        for p in video_sysfs.glob("video*")
        if (p / "device").resolve().is_relative_to(usb)
    ]
    owners = subprocess.run(
        ["fuser", str(node), *(str(p) for p in video_nodes)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if owners.returncode != 1:
        raise RuntimeError(
            f"Camera {usb_serial} is in use or ownership could not be checked; refusing USB reset. "
            f"{owners.stdout.strip()} {owners.stderr.strip()}"
        )
    if not os.access(node, os.W_OK):
        raise PermissionError(
            f"Camera {usb_serial} requires USB write access to {node}."
        )
    get_logger().info(
        "Resetting camera %s at USB %s before capture", usb_serial, target
    )
    subprocess.run(["usbreset", target], check=True, timeout=10, capture_output=True)
    subprocess.run(["udevadm", "settle", "--timeout=10"], check=True, timeout=12)
    time.sleep(2)


def resolve_piper_camera(
    usb_serial: str,
    usb_interface: str,
    *,
    sysfs: Path = Path("/sys/class/video4linux"),
    device_root: Path = Path("/dev"),
) -> str:
    """Select one capture node by USB serial, interface, and capture index zero."""
    if not usb_serial or not usb_interface:
        raise ValueError("Camera selection requires a USB serial and interface.")
    matches = []
    for entry in sysfs.glob("video*"):
        if (entry / "index").read_text().strip() != "0":
            continue
        device = (entry / "device").resolve()
        parents = (device, *device.parents)
        interface = next(
            (
                p / "bInterfaceNumber"
                for p in parents
                if (p / "bInterfaceNumber").is_file()
            ),
            None,
        )
        serial = next(
            (p / "serial" for p in parents if (p / "idVendor").is_file()), None
        )
        if (
            interface is not None
            and interface.read_text().strip() == usb_interface
            and serial is not None
            and serial.is_file()
            and serial.read_text().strip() == usb_serial
        ):
            matches.append(str(device_root / entry.name))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one capture node for camera {usb_serial}, interface {usb_interface}; found {matches}."
        )
    return matches[0]


def build_piper_cameras(specifications: dict) -> dict[str, "PiperOpenCVCamera"]:
    """Construct disconnected RGB cameras using stable USB identities or paths."""
    cameras = {}
    paths = set()
    for name, values in specifications.items():
        spec = dict(values)
        serial = spec.pop("usb_serial", None)
        interface = spec.pop("usb_interface", None)
        reset = spec.pop("reset_on_connect", False)
        if not isinstance(reset, bool) or (reset and serial is None):
            raise ValueError(
                "reset_on_connect requires a boolean and a camera USB identity."
            )
        if serial is not None or interface is not None:
            if "index_or_path" in spec:
                raise ValueError("Specify either camera USB identity or index_or_path.")
            spec["index_or_path"] = resolve_piper_camera(serial, interface)
        path = Path(spec["index_or_path"]).resolve(strict=True)
        if path in paths:
            raise ValueError("Each camera must identify a different physical device.")
        paths.add(path)
        cameras[name] = PiperOpenCVCamera(
            OpenCVCameraConfig(**spec, color_mode="rgb"),
            usb_serial=serial,
            usb_interface=interface,
            reset_on_connect=reset,
        )
    return cameras


class PiperOpenCVCamera(OpenCVCamera):
    """Keep the reader's stop event and capture alive until the reader exits.

    LeRobot 0.4.4 clears these after a two-second join even when a V4L2 read
    is still blocked. OpenCV's default select timeout is ten seconds.
    """

    def __init__(
        self,
        config: OpenCVCameraConfig,
        *,
        usb_serial: str | None = None,
        usb_interface: str | None = None,
        reset_on_connect: bool = False,
    ) -> None:
        super().__init__(config)
        if bool(usb_serial) != bool(usb_interface) or (
            reset_on_connect and not usb_serial
        ):
            raise ValueError("USB camera recovery requires both serial and interface.")
        self.usb_serial = usb_serial
        self.usb_interface = usb_interface
        self.reset_on_connect = reset_on_connect

    def connect(self, warmup: bool = True) -> None:
        """Recover the selected idle camera if configured, then open its RGB node."""
        if not self.is_connected and self.usb_serial is not None:
            if self.reset_on_connect:
                reset_piper_camera(self.usb_serial)
            self.index_or_path = resolve_piper_camera(
                self.usb_serial, self.usb_interface
            )
        super().connect(warmup=warmup)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=12.0)
            if self.thread.is_alive():
                raise RuntimeError(
                    f"{self}: reader did not stop; keeping its capture and stop event alive."
                )
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()
