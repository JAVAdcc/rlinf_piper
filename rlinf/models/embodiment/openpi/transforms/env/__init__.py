# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dispatch env-worker dicts onto openpi ``observation/*`` keys."""

from __future__ import annotations

from typing import Callable

from rlinf.models.embodiment.openpi.transforms.env import calvin, default


def repack_env_obs(
    config_name: str,
    env_obs: dict,
    *,
    select_state: Callable,
) -> dict:
    """Map an env observation dict to ``observation/*`` keys for ``config_name``."""
    if config_name in ("pi05_piper", "pi05_piper_h50_state"):
        extra = env_obs["extra_view_images"]
        if extra.shape[1] != 2:
            raise ValueError("Piper requires front_2 and wrist as its two extra views.")
        return {
            "observation.state": select_state(env_obs["states"]),
            "observation.images.front": env_obs["main_images"],
            "observation.images.front_2": extra[:, 0],
            "observation.images.wrist": extra[:, 1],
            "prompt": env_obs["task_descriptions"],
        }
    if "calvin" in config_name:
        return calvin.repack_env_obs(env_obs, select_state=select_state)
    return default.repack_env_obs(env_obs, select_state=select_state)
