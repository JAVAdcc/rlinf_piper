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

"""LeRobot sequence reads bounded by the requested action window."""

import torch
from openpi.training.data_loader import _import_lerobot_dataset


class LeRobotSFTDataset(_import_lerobot_dataset().LeRobotDataset):
    """Read local SFT data with LeRobot's constructor and sample interface.

    Select sequence rows before formatting columns, so datasets versions with
    eager column indexing only convert the requested window. Episode selection,
    padding and video decoding retain the installed LeRobot implementation.
    """

    def _query_hf_dataset(
        self, query_indices: dict[str, list[int]]
    ) -> dict[str, torch.Tensor]:
        index_mapping = getattr(self, "_absolute_to_relative_idx", None)
        result = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                indices
                if index_mapping is None
                else [index_mapping[index] for index in indices]
            )
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result
