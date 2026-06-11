"""useful type definitions for highway env"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TypedDict
from typing_extensions import Self

from jaxtyping import Float
from numpy import ndarray


__all__ = ["Position", "Polygon", "Vec2D", "Color", "NewLaneIndex", "Vector", "Matrix", "Interval", "ActionDict", "LineType"]


Position = Float[ndarray, "2"]
"""A class representing a position in 2D space,
as a numpy array of shape (2,) and dtype float."""

Polygon = Float[ndarray, "*, 2"]
"""A class representing a polygon in 2D space,as a numpy array of shape (n, 2) and dtype float."""

Vec2D = Float[ndarray, "2"]
"""A class representing a 2D vector, as a numpy array of shape (2,) and dtype float.

impact, velocity, heading, etc. can be represented as Vec2D."""

Color = tuple[int, int, int] | tuple[int, int, int, int]
"""A class representing a color, as a tuple of 3 or 4 integers in [0, 255]."""


class NewLaneIndex(tuple[str, str, int]):
    """A class representing the index of a lane, as a tuple of (from_node, to_node, lane_id).

    if lane_id is None, it will be set to -1, which means the lane_id is not specified.
    """
    EMPTY: "NewLaneIndex"
    """an empty lane index, used for type checking,

    value is ``("", "", -1)``
    """

    def __new__(cls, _from: str, to: str, lane_id: int | None) -> Self:
        lane_id = lane_id if lane_id is not None else -1
        return super().__new__(cls, (_from, to, lane_id))

    def __bool__(self):
        return self is not self.EMPTY


NewLaneIndex.EMPTY = NewLaneIndex("", "", None)


Vector = Float[ndarray, "*"]
"""An 1D ndarray, shape (n,)"""
Matrix = Float[ndarray, "*, *"]
"""An 2D ndarray, shape (m, n)"""
Interval = Float[ndarray, "2"] | Float[ndarray, "2, *"] | Float[ndarray, "2, *, *"]
"""An 1D or 2D or 3D ndarray, shape (2,) or (2, n) or (2, m, n)"""


class ActionDict(TypedDict):
    """A dictionary representation of an action, for use in MultiAgentAction."""

    acceleration: float
    """the acceleration to apply, range in [-1,1],

    mapped to the acceleration range defined in `ContinuousAction.acceleration_range`
    """
    steering: float
    """the steering angle to apply, range in [-1,1],

    mapped to the steering range defined in `ContinuousAction.steering_range`
    """


@dataclass
class Action:
    """A dictionary representation of an action, for use in MultiAgentAction."""
    acceleration: float
    """the acceleration to apply, range in [-1,1],

    mapped to the acceleration range defined in `ContinuousAction.acceleration_range`
    """
    steering: float
    """the steering angle to apply, range in [-1,1],

    mapped to the steering range defined in `ContinuousAction.steering_range`
    """


class LineType(enum.Enum):
    """A lane side line type."""

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3
