from __future__ import annotations

import importlib
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from highway_env import utils
from highway_env.typing import Grid
from highway_env.vehicle.controller import MDPVehicle


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv
    from highway_env.envs.highway_env import HighwayEnv
    from highway_env.envs.merge_env import MergeEnv
    from highway_env.envs.roundabout_env import RoundaboutEnv


def finite_mdp(
    env: "HighwayEnv|MergeEnv|RoundaboutEnv", time_quantization: float = 1.0, horizon: float = 10.0
) -> object:
    """
    Time-To-Collision (TTC) representation of the state.

    The state reward is defined from a occupancy grid over different TTCs and lanes. The grid cells encode the
    probability that the ego-vehicle will collide with another vehicle if it is located on a given lane in a given
    duration, under the hypothesis that every vehicles observed will maintain a constant speed (including the
    ego-vehicle) and not change lane (excluding the ego-vehicle).

    For instance, in a three-lane road with a vehicle on the left lane with collision predicted in 5s the grid will
    be:
    [0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0]
    The TTC-state is a coordinate (lane, time) within this grid.

    If the ego-vehicle has the ability to change its speed, an additional layer is added to the occupancy grid
    to iterate over the different speed choices available.

    Finally, this state is flattened for compatibility with the FiniteMDPEnv environment.

    :param AbstractEnv env: an environment
    :param time_quantization: the time quantization used in the state representation [s]
    :param horizon: the horizon on which the collisions are predicted [s]
    """
    # Compute TTC grid
    grid = compute_ttc_grid(env, time_quantization, horizon)

    # Compute current state
    grid_state = (env.vehicle.speed_index, env.vehicle.lane_index[2], 0)
    state = np.ravel_multi_index(grid_state, grid.shape)

    # Compute transition function

    # @minokori only Discrete action space has field `n`
    # so we assume that the action space is Discrete

    transition_model_with_grid = partial(transition_model, grid=grid)
    transition = np.fromfunction(
        transition_model_with_grid, grid.shape + (env.action_space.n,), dtype=int
    )
    transition = np.reshape(transition, (np.size(grid), env.action_space.n))

    # Compute reward function
    v, l, t = grid.shape
    lanes = np.arange(l) / max(l - 1, 1)
    speeds = np.arange(v) / max(v - 1, 1)

    state_reward = (
        + env.config["collision_reward"] * grid
        + env.config["right_lane_reward"]
        * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t))
        + env.config["high_speed_reward"]
        * np.tile(speeds[:, np.newaxis, np.newaxis], (1, l, t))
    )

    state_reward = np.ravel(state_reward)
    action_reward = [
        env.config["lane_change_reward"],
        0,
        env.config["lane_change_reward"],
        0,
        0,
    ]
    reward = np.fromfunction(
        np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
        (np.size(state_reward), np.size(action_reward)),
        dtype=int,
    )

    # Compute terminal states
    collision = grid == 1
    end_of_horizon = np.fromfunction(
        lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int
    )
    terminal = np.ravel(collision | end_of_horizon)

    # Creation of a new finite MDP
    try:
        module = importlib.import_module("finite_mdp.mdp")
        mdp = module.DeterministicMDP(transition, reward, terminal, state=state)
        mdp.original_shape = grid.shape
        return mdp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"The finite_mdp module is required for conversion. {e}"
        )


def compute_ttc_grid(
    env: AbstractEnv,
    time_quantization: float,
    horizon: float,
    vehicle: MDPVehicle | None = None,
) -> Grid:
    """
    Compute the grid of predicted time-to-collision to each vehicle within the lane

    For each ego-speed and lane.
    :param env: environment
    :param time_quantization: time step of a grid cell
    :param horizon: time horizon of the grid
    :param vehicle: the observer vehicle
    :return: the time-co-collision grid, with axes SPEED x LANES x TIME
    """
    vehicle = vehicle or env.vehicle
    road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
    grid: Grid = np.zeros(
        (vehicle.target_speeds.size, len(road_lanes), int(horizon / time_quantization))
    )
    for speed_index in range(grid.shape[0]):
        ego_speed = vehicle.index_to_speed(speed_index)
        for other in env.road.vehicles:
            if (other is vehicle) or (ego_speed == other.speed):
                continue
            margin = other.LENGTH / 2 + vehicle.LENGTH / 2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
            for m, cost in collision_points:
                distance = vehicle.lane_distance_to(other) + m
                other_projected_speed = other.speed * np.dot(
                    other.direction, vehicle.direction
                )
                time_to_collision = distance / utils.not_zero(
                    ego_speed - other_projected_speed
                )
                if time_to_collision < 0:
                    continue
                if env.road.network.is_connected_road(
                    vehicle.lane_index, other.lane_index, route=vehicle.route, depth=3
                ):
                    # Same road, or connected road with same number of lanes
                    if len(env.road.network.all_side_lanes(other.lane_index)) == len(
                        env.road.network.all_side_lanes(vehicle.lane_index)
                    ):
                        lane: list[int] = [other.lane_index[2]]  # type: ignore
                    # Different road of different number of lanes: uncertainty on future lane, use all
                    else:
                        lane: list[int] | range = range(grid.shape[1])
                    # Quantize time-to-collision to both upper and lower values
                    for time in [
                        int(time_to_collision / time_quantization),
                        int(np.ceil(time_to_collision / time_quantization)),
                    ]:
                        if 0 <= time < grid.shape[2]:
                            # TODO: check lane overflow (e.g. vehicle with higher lane id than current road capacity)
                            grid[speed_index, lane, time] = np.maximum(
                                grid[speed_index, lane, time], cost
                            )
    return grid


def transition_model(h: NDArray[np.intp], i: NDArray[np.intp], j: NDArray[np.intp], a: NDArray[np.int_], grid: Grid) -> NDArray[np.intp]:
    """
    Deterministic transition from a position in the grid to the next.

    :param h: speed index
    :param i: lane index
    :param j: time index
    :param a: action index
    :param grid: ttc grid specifying the limits of speeds, lanes, time and actions
    """
    # Idle action (1) as default transition
    next_state: NDArray[np.intp] = clip_position(h, i, j + 1, grid)
    left: NDArray[np.bool_] = a == 0
    right: NDArray[np.bool_] = a == 2
    faster: NDArray[np.bool_] = (a == 3) & (j == 0)
    slower: NDArray[np.bool_] = (a == 4) & (j == 0)
    next_state[left] = clip_position(h[left], i[left] - 1, j[left] + 1, grid)  # pylint: disable=E1137
    next_state[right] = clip_position(h[right], i[right] + 1, j[right] + 1, grid)  # pylint: disable=E1137
    next_state[faster] = clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)  # pylint: disable=E1137
    next_state[slower] = clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)  # pylint: disable=E1137
    return next_state


def clip_position(h: NDArray[np.intp], i: NDArray[np.intp], j: NDArray[np.intp], grid: Grid) -> NDArray[np.intp]:
    """
    Clip a position in the TTC grid, so that it stays within bounds.

    :param h: speed index
    :param i: lane index
    :param j: time index
    :param grid: the ttc grid
    :return: The raveled index of the clipped position
    """
    h = np.clip(h, 0, grid.shape[0] - 1)
    i = np.clip(i, 0, grid.shape[1] - 1)
    j = np.clip(j, 0, grid.shape[2] - 1)
    indexes = np.ravel_multi_index((h, i, j), grid.shape)
    return indexes
