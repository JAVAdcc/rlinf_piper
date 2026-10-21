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

"""Critical action intervals and independently scored segment outcomes."""

SEGMENT_LABEL_FORMAT = "piper-rlt-episodes-v3"
SEGMENT_SCORING_PROTOCOL = "critical-segment-v1"
SEGMENT_PHASE_DEFINITION = (
    "b starts a critical segment; s/f ends and scores that segment. "
    "Right stops the episode; a separate s/f scores the whole episode."
)


def critical_phase_intervals(
    phases: object, terminal_frame: int
) -> list[tuple[int, int]]:
    """Validate ordered [start,end) action intervals with recorded end observations.

    Saved annotations need explicit endpoints. The collector closes the final
    active interval when scoring; an empty list or null saved endpoint remains
    an incomplete annotation.
    """
    if type(terminal_frame) is not int or terminal_frame <= 0:
        raise ValueError("terminal_frame must be a positive observation frame index.")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Missing critical phases; mark start/end pairs.")
    intervals = []
    previous_end = 0
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("Each critical phase needs start and end frame indices.")
        start, end = phase.get("start"), phase.get("end")
        if type(start) is not int or type(end) is not int:
            raise ValueError("Unpaired critical phase: start and end must be integers.")
        if not previous_end <= start < end <= terminal_frame:
            raise ValueError(
                "Critical phases must be nonempty, ordered, nonoverlapping and within the episode."
            )
        intervals.append((start, end))
        previous_end = end
    return intervals


def scored_critical_phase_intervals(
    phases: object, terminal_frame: int
) -> list[tuple[int, int]]:
    """Validate nonempty segments with their own explicit success/failure labels.

    Segment results do not imply an episode result, or vice versa. The end
    observation may also be the start observation of a later segment.
    """
    intervals = critical_phase_intervals(phases, terminal_frame)
    for phase in phases:
        if phase.get("outcome") not in ("success", "failure"):
            raise ValueError(
                "Each critical segment needs its own success/failure outcome."
            )
    return intervals
