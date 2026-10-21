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

"""Loss contracts shared by offline warmup and online segment replay."""

import torch


def action_mask(lengths: torch.Tensor, chunk: int) -> torch.Tensor:
    """Select executed ticks, excluding terminal padding."""
    return torch.arange(chunk, device=lengths.device)[None, :] < lengths[:, None]


def discounted_target(batch: dict, next_q: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discount actual ticks to the reward endpoint specified by the cache."""
    ticks = batch["discount_steps"]
    returns = batch["rewards"] * gamma ** (ticks - 1)
    return returns + (~batch["terminated"]) * gamma**ticks * next_q


def actor_training_observation(batch: dict) -> dict:
    """Condition actor learning on executed human corrections at valid ticks.

    Keep cached VLA references intact for next-state policy evaluation and
    deployment-style validation. Reference dropout is applied by the actor
    after this replacement.
    """
    chunk = batch["actions"].shape[1]
    reference = batch["curr_obs"]["ref_chunk"].clone()
    human = batch["human"] & action_mask(batch["valid_steps"], chunk)
    reference[:, :chunk] = torch.where(
        human[..., None], batch["actions"], reference[:, :chunk]
    )
    return {**batch["curr_obs"], "ref_chunk": reference}


def masked_behavior_loss(predicted: torch.Tensor, batch: dict) -> torch.Tensor:
    """Imitate human actions where labeled; otherwise imitate frozen references."""
    chunk = batch["actions"].shape[1]
    prediction = predicted.reshape_as(batch["actions"])
    target = torch.where(
        batch["human"][..., None],
        batch["actions"],
        batch["curr_obs"]["ref_chunk"][:, :chunk],
    )
    mask = action_mask(batch["valid_steps"], chunk)
    return ((prediction - target).square().mean(-1) * mask).sum() / mask.sum()
