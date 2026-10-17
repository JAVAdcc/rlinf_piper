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

"""OpenPI data configuration for Evo-RL single-arm Piper recordings."""

import dataclasses
import pathlib

import openpi.models.model as model
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies.piper_policy import (
    PiperInputs,
    PiperOutputs,
)


@dataclasses.dataclass(frozen=True)
class PiperDataConfig(DataConfigFactory):
    """Train on absolute six-joint targets plus continuous gripper width."""

    def create(
        self, assets_dirs: pathlib.Path, model_config: model.BaseModelConfig
    ) -> DataConfig:
        """Use the same physical-unit conversion for policy and value inputs."""
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_sequence_keys=("action",),
            data_transforms=transforms.Group(
                inputs=[PiperInputs()], outputs=[PiperOutputs()]
            ),
            model_transforms=ModelTransformFactory()(model_config),
        )
