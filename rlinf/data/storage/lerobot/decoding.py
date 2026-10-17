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

"""Decode LeRobot image structs before converting dataset columns to tensors."""

import io

from PIL import Image as PILImage

try:
    from lerobot.datasets.utils import hf_transform_to_torch
except ModuleNotFoundError:
    from lerobot.common.datasets.utils import hf_transform_to_torch


def decode_image_struct_batch(batch: dict) -> dict:
    """Decode Image-feature struct dicts before ``hf_transform_to_torch``."""
    for key in list(batch.keys()):
        vals = batch[key]
        if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
            batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
    return hf_transform_to_torch(batch)
