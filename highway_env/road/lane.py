from __future__ import annotations

import enum
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from highway_env import utils
from highway_env.road.spline import LinearSpline2D
from highway_env.utils import (
    Position,
    Vec2D,
    class_from_path,
    get_class_path,
    wrap_to_pi,
)


class AbstractLane:
    """A lane on the road, described by its central curve."""

    EMPTY: "AbstractLane"

    def __bool__(self):
        return False if self is AbstractLane.EMPTY else True

    if TYPE_CHECKING:
        forbidden: bool
        """is changing to this lane forbidden.

        This field only exist in implementations of AbstractLane, but not in the abstract class itself.
        """
        speed_limit: float
        """the lane speed limit [m/s].

        This field only exist in implementations of AbstractLane, but not in the abstract class itself.
        """

        priority: int
        """priority level of the lane, for determining who has right of way.

        This field only exist in implementations of AbstractLane, but not in the abstract class itself.
        """

        heading: float
        """the lane heading [rad].

            This field only exist in implementations of AbstractLane, but not in the abstract class itself.
        """

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: tuple[LineType, LineType]

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> Position:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: Position) -> tuple[float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: dict):
        """
        Create lane instance from config

        :param config: json dict with lane parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def to_config(self) -> dict:
        """
        Write lane parameters to dict which can be serialized to json

        :return: dict of lane parameters
        """
        raise NotImplementedError()

    def on_lane(
        self,
        position: Position,
        longitudinal: float | None = None,
        lateral: float | None = None,
        margin: float = 0,
    ) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = (
            np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin
            and -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_on

    def is_reachable_from(self, position: Position) -> bool:
        """
        Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = (
            np.abs(lateral) <= 2 * self.width_at(longitudinal)
            and 0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_close

    def after_end(
        self, position: Position, longitudinal: float | None = None, lateral: float | None = None
    ) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position: Position) -> float:
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(
        self,
        position: Position,
        heading: float | None,
        heading_weight: float = 1.0,
    ) -> float:
        """Compute a weighted distance in position and heading to the lane."""
        if heading is None:
            return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(self.local_angle(heading, s))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

    def local_angle(self, heading: float, long_offset: float) -> float:
        """Compute non-normalised angle of heading to the lane."""
        return wrap_to_pi(heading - self.heading_at(long_offset))


AbstractLane.EMPTY = AbstractLane()


class LineType(enum.Enum):
    """A lane side line type."""

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class StraightLane(AbstractLane):
    """A lane going in straight line."""

    def __init__(
        self,
        start: Position,
        end: Position,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] | None = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane, default to STRIPED.
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for determining who has right of way
        """
        self.start: Position = start
        """the lane starting position [m]"""
        self.end: Position = end
        """the lane ending position [m]"""
        self.width: float = width
        """the lane width [m]"""
        self.heading: float = np.arctan2(
            self.end[1] - self.start[1], self.end[0] - self.start[0]
        )
        """the lane heading [rad]"""
        self.length: float = float(np.linalg.norm(self.end - self.start))
        """the lane length [m]"""
        self.line_types: tuple[LineType, LineType] = line_types if line_types else (LineType.STRIPED, LineType.STRIPED)
        """the type of lines on both sides of the lane"""
        self.direction: Vec2D = (self.end - self.start) / self.length
        """the lane direction vector"""
        self.direction_lateral: Vec2D = np.array([-self.direction[1], self.direction[0]])
        """the lane lateral direction vector"""
        self.forbidden: bool = forbidden
        """is changing to this lane forbidden"""
        self.priority: int = priority
        """priority level of the lane, for determining who has right of way"""
        self.speed_limit: float = speed_limit
        """the lane speed limit [m/s]"""

    def position(self, longitudinal: float, lateral: float) -> Position:
        return (
            self.start
            + longitudinal * self.direction
            + lateral * self.direction_lateral
        )

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: Position) -> tuple[float, float]:
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "start": _to_serializable(self.start),
                "end": _to_serializable(self.end),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class SineLane(StraightLane):
    """A sinusoidal lane."""

    def __init__(
        self,
        start: Position,
        end: Position,
        amplitude: float,
        pulsation: float,
        phase: float,
        width: float = StraightLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] | None = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        """
        New sinusoidal lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        """
        super().__init__(
            start, end, width, line_types, forbidden, speed_limit, priority
        )
        self.amplitude: float = amplitude
        self.pulsation: float = pulsation
        self.phase: float = phase

    def position(self, longitudinal: float, lateral: float) -> Position:
        return super().position(
            longitudinal,
            lateral
            + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase),
        )

    def heading_at(self, longitudinal: float) -> float:
        return super().heading_at(longitudinal) + np.arctan(
            self.amplitude
            * self.pulsation
            * np.cos(self.pulsation * longitudinal + self.phase)
        )

    def local_coordinates(self, position: Position) -> tuple[float, float]:
        longitudinal, lateral = super().local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(
            self.pulsation * longitudinal + self.phase
        )

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        config = super().to_config()
        config.update(
            {
                "class_path": get_class_path(self.__class__),
            }
        )
        config["config"].update(
            {
                "amplitude": self.amplitude,
                "pulsation": self.pulsation,
                "phase": self.phase,
            }
        )
        return config


class CircularLane(AbstractLane):
    """A lane going in circle arc."""

    def __init__(
        self,
        center: Position,
        radius: float,
        start_phase: float,
        end_phase: float,
        clockwise: bool = True,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] | None = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        super().__init__()
        self.center: Position = center
        self.radius: float = radius
        self.start_phase: float = start_phase
        self.end_phase: float = end_phase
        self.clockwise: bool = clockwise
        self.direction: int = 1 if clockwise else -1
        self.width: float = width
        self.line_types: tuple[LineType, LineType] = line_types if line_types else (LineType.STRIPED, LineType.STRIPED)
        self.forbidden: bool = forbidden
        self.length: float = radius * (end_phase - start_phase) * self.direction
        self.priority: int = priority
        self.speed_limit: float = speed_limit

    def position(self, longitudinal: float, lateral: float) -> Position:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction) * np.array(
            [np.cos(phi), np.sin(phi)]
        )

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi / 2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: Position) -> tuple[float, float]:
        delta = position - self.center
        phi: float = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + utils.wrap_to_pi(phi - self.start_phase)
        r: float = np.linalg.norm(delta)  # type: ignore
        longitudinal = self.direction * (phi - self.start_phase) * self.radius
        lateral = self.direction * (self.radius - r)
        return longitudinal, lateral  # type: ignore

    @classmethod
    def from_config(cls, config: dict):
        config["center"] = np.array(config["center"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "center": _to_serializable(self.center),
                "radius": self.radius,
                "start_phase": self.start_phase,
                "end_phase": self.end_phase,
                "clockwise": self.clockwise,
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLaneFixedWidth(AbstractLane):
    """
    A fixed-width lane defined by a set of points and approximated with a 2D Hermite polynomial.
    """

    def __init__(
        self,
        lane_points: list[tuple[float, float]],
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] | None = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        self.curve: LinearSpline2D = LinearSpline2D(lane_points)
        self.length: float = self.curve.length
        self.width: float = width
        self.line_types: tuple[LineType, LineType] = line_types if line_types else (LineType.STRIPED, LineType.STRIPED)
        self.forbidden: bool = forbidden
        self.speed_limit: float = speed_limit
        self.priority: int = priority

    def position(self, longitudinal: float, lateral: float) -> Position:
        x, y = self.curve(longitudinal)
        yaw = self.heading_at(longitudinal)
        return np.array([x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral])

    def local_coordinates(self, position: Position) -> tuple[float, float]:
        lon, lat = self.curve.cartesian_to_frenet(position)
        return lon, lat

    def heading_at(self, longitudinal: float) -> float:
        dx, dy = self.curve.get_dx_dy(longitudinal)
        return np.arctan2(dy, dx)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "lane_points": _to_serializable(
                    [_to_serializable(p.position) for p in self.curve.poses]
                ),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLane(PolyLaneFixedWidth):
    """
    A lane defined by a set of points and approximated with a 2D Hermite polynomial.
    """

    def __init__(
        self,
        lane_points: list[tuple[float, float]],
        left_boundary_points: list[tuple[float, float]],
        right_boundary_points: list[tuple[float, float]],
        line_types: tuple[LineType, LineType] | None = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ):
        super().__init__(
            lane_points=lane_points,
            line_types=line_types,
            forbidden=forbidden,
            speed_limit=speed_limit,
            priority=priority,
        )
        self.right_boundary: LinearSpline2D = LinearSpline2D(right_boundary_points)
        self.left_boundary: LinearSpline2D = LinearSpline2D(left_boundary_points)
        self._init_width()

    def width_at(self, longitudinal: float) -> float:
        if longitudinal < 0:
            return self.width_samples[0]
        elif longitudinal > len(self.width_samples) - 1:
            return self.width_samples[-1]
        else:
            return self.width_samples[int(longitudinal)]

    def _width_at_s(self, longitudinal: float) -> float:
        """
        Calculate width by taking the minimum distance between centerline and each boundary at a given s-value. This compensates indentations in boundary lines.
        """
        center = self.position(longitudinal, 0)
        right_x, right_y = self.right_boundary(
            self.right_boundary.cartesian_to_frenet(center)[0]
        )
        left_x, left_y = self.left_boundary(
            self.left_boundary.cartesian_to_frenet(center)[0]
        )

        dist_to_center_right = np.linalg.norm(
            np.array([right_x, right_y]) - center
        )
        dist_to_center_left = np.linalg.norm(
            np.array([left_x, left_y]) - center
        )

        return float(max(
            min(dist_to_center_right, dist_to_center_left) * 2,
            AbstractLane.DEFAULT_WIDTH,
        ))

    def _init_width(self):
        """
        Pre-calculate sampled width values in about 1m distance to reduce computation during runtime. It is assumed that the width does not change significantly within 1-2m.
        Using numpys linspace ensures that min and max s-values are contained in the samples.
        """
        s_samples = np.linspace(
            0,
            self.curve.length,
            num=int(np.ceil(self.curve.length)) + 1,
        )
        self.width_samples = [self._width_at_s(s) for s in s_samples]

    def to_config(self) -> dict:
        config = super().to_config()

        ordered_boundary_points = _to_serializable(
            [_to_serializable(p.position) for p in reversed(self.left_boundary.poses)]
        )
        ordered_boundary_points += _to_serializable(
            [_to_serializable(p.position) for p in self.right_boundary.poses]
        )

        config["class_name"] = self.__class__.__name__
        config["config"]["ordered_boundary_points"] = ordered_boundary_points
        del config["config"]["width"]

        return config


def _to_serializable(arg: np.ndarray | list) -> list:
    if isinstance(arg, np.ndarray):
        return arg.tolist()
    return arg


def lane_from_config(cfg: dict) -> AbstractLane:
    return class_from_path(cfg["class_path"])(**cfg["config"])
