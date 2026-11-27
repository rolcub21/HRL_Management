"""E549-style chained SAC environment using OGBench AntMaze Large dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ogbench


COMMAND_HORIZON = 100
PHYSICAL_HORIZON = 1000
GOAL_THRESHOLD = 0.2
FORWARD_REWARD_SCALE = 10.0
CONTROL_COST_WEIGHT = 0.5
SURVIVAL_REWARD = 1.0


def dense_directional_reward(
    previous_xy: np.ndarray,
    achieved_xy: np.ndarray,
    desired_xy: np.ndarray,
    action: np.ndarray,
    dt: float,
) -> float:
    goal_vector = np.asarray(desired_xy) - np.asarray(achieved_xy)
    velocity = (np.asarray(achieved_xy) - np.asarray(previous_xy)) / float(dt)
    reward_forward = np.dot(velocity, goal_vector) / np.linalg.norm(goal_vector)
    reward_ctrl = -CONTROL_COST_WEIGHT * float(np.sum(np.square(action)))
    return float(FORWARD_REWARD_SCALE * reward_forward + reward_ctrl + SURVIVAL_REWARD)


class OGBenchLargeChainedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        command_bank: str,
        switch_on_contact: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        bank_path = Path(command_bank).expanduser().resolve()
        with np.load(bank_path, allow_pickle=False) as bank:
            self.command_distances = np.asarray(
                bank["requested_distance_m"], dtype=np.float64
            )
            self.reset_xy = np.asarray(bank["reset_xy"], dtype=np.float64)
        if (
            self.command_distances.ndim != 1
            or len(self.command_distances) == 0
            or np.any(self.command_distances <= 0.0)
            or not np.isfinite(self.command_distances).all()
        ):
            raise ValueError("R22 command-distance bank is invalid")
        if self.reset_xy.ndim != 2 or self.reset_xy.shape[1] != 2:
            raise ValueError("R22 reset bank is invalid")
        self.switch_on_contact = bool(switch_on_contact)
        self.render_mode = render_mode
        self.maze_env = ogbench.make_env_and_datasets(
            "antmaze-large-navigate-v0",
            env_only=True,
            terminate_at_goal=False,
            success_timing="post",
            max_episode_steps=PHYSICAL_HORIZON,
        )
        self.native = self.maze_env.unwrapped
        self.action_space = self.maze_env.action_space
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(31,), dtype=np.float64
        )
        self.goal = np.zeros(2, dtype=np.float64)
        self.command_start = np.zeros(2, dtype=np.float64)
        self.command_steps = 0
        self.command_index = 0
        self.last_command_event: Optional[dict] = None
        self._native_observation = np.zeros(29, dtype=np.float64)

    def _is_free(self, xy: np.ndarray) -> bool:
        ij = self.native.xy_to_ij(np.asarray(xy, dtype=np.float64))
        row, col = int(ij[0]), int(ij[1])
        maze = self.native.maze_map
        if row < 0 or col < 0 or row >= maze.shape[0] or col >= maze.shape[1]:
            return False
        if int(maze[row, col]) != 0:
            return False
        center = np.asarray(self.native.ij_to_xy((row, col)), dtype=np.float64)
        # This is OGBench's own add_noise support: maze_unit / 4.
        support = float(self.native._maze_unit * self.native._noise / 4.0)
        return bool(np.max(np.abs(np.asarray(xy) - center)) <= support)

    def _is_free_cell(self, xy: np.ndarray) -> bool:
        row, col = (
            int(value) for value in self.native.xy_to_ij(np.asarray(xy, dtype=np.float64))
        )
        maze = self.native.maze_map
        return bool(
            0 <= row < maze.shape[0]
            and 0 <= col < maze.shape[1]
            and int(maze[row, col]) == 0
        )

    def _sample_goal(self, achieved_xy: np.ndarray) -> np.ndarray:
        # Identical E549 law: uniform direction and empirical distance replay.
        for _ in range(10_000):
            angle = 2.0 * np.pi * self.np_random.uniform()
            distance = float(self.np_random.choice(self.command_distances))
            goal = np.asarray(achieved_xy, dtype=np.float64) + distance * np.asarray(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            )
            if self._is_free(goal):
                return goal
        raise RuntimeError("R22 could not sample a free command in 10,000 draws")

    def _set_goal(self, goal: np.ndarray, achieved_xy: np.ndarray) -> None:
        goal = np.asarray(goal, dtype=np.float64).reshape(2)
        if not self._is_free(goal):
            raise ValueError(f"R22 goal is outside OGBench free support: {goal}")
        self.goal = goal.copy()
        self.command_start = np.asarray(achieved_xy, dtype=np.float64).copy()
        self.command_steps = 0
        self.native.set_goal(goal_xy=self.goal)

    def set_external_goal(self, goal: np.ndarray) -> np.ndarray:
        achieved = self._native_observation[:2].copy()
        goal = np.asarray(goal, dtype=np.float64).reshape(2)
        # R21 targets are exact achieved states. They need only occupy a free
        # cell; the narrower reset-goal noise support remains training-only.
        if not self._is_free_cell(goal):
            raise ValueError(f"R21 target is not in an OGBench free cell: {goal}")
        self.goal = goal.copy()
        self.command_start = achieved
        self.command_steps = 0
        self.native.set_goal(goal_xy=self.goal)
        return self._observation(self._native_observation)

    def _observation(self, native_observation: np.ndarray) -> np.ndarray:
        native_observation = np.asarray(native_observation, dtype=np.float64).reshape(29)
        result = np.concatenate(
            [native_observation, self.goal - native_observation[:2]]
        )
        if result.shape != (31,) or result.dtype != np.float64:
            raise RuntimeError("R22 observation contract changed")
        if not np.array_equal(result[:29], native_observation):
            raise RuntimeError("R22 altered OGBench's native observation")
        return result

    @property
    def native_observation(self) -> np.ndarray:
        return self._native_observation.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        exact_reset = options.get("reset_xy")
        if exact_reset is None:
            reset_xy = self.reset_xy[int(self.np_random.integers(len(self.reset_xy)))]
        else:
            reset_xy = np.asarray(exact_reset, dtype=np.float64).reshape(2)
        reset_ij = tuple(int(value) for value in self.native.xy_to_ij(reset_xy))
        task_info = {
            "task_name": "reward_free_executor_reset",
            "init_ij": reset_ij,
            "goal_ij": reset_ij,
        }
        native_observation, info = self.maze_env.reset(
            seed=seed, options={"task_info": task_info}
        )
        if exact_reset is not None:
            self.native.set_xy(reset_xy)
            native_observation = self.native.get_ob()
        self._native_observation = np.asarray(native_observation, dtype=np.float64)
        achieved = self._native_observation[:2]
        self.command_index = 0
        self.last_command_event = None
        self._set_goal(self._sample_goal(achieved), achieved)
        info = dict(info)
        info.update(
            command_index=0,
            command_steps=0,
            command_goal=self.goal.copy(),
            command_start=self.command_start.copy(),
            native_task_row_accessed=False,
        )
        return self._observation(self._native_observation), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        previous_xy = self._native_observation[:2].copy()
        previous_goal = self.goal.copy()
        previous_start = self.command_start.copy()
        native_observation, _task_reward, terminated, truncated, info = (
            self.maze_env.step(action)
        )
        self._native_observation = np.asarray(native_observation, dtype=np.float64)
        achieved = self._native_observation[:2]
        reward = dense_directional_reward(
            previous_xy, achieved, previous_goal, action, self.native.dt
        )
        self.command_steps += 1
        distance = float(np.linalg.norm(achieved - previous_goal))
        reached = distance <= GOAL_THRESHOLD
        timed_out = self.command_steps >= COMMAND_HORIZON
        switched = bool(timed_out or (self.switch_on_contact and reached))
        event = None
        if switched:
            event = {
                "command_index": int(self.command_index),
                "reason": "reached" if reached else "horizon",
                "duration": int(self.command_steps),
                "start": previous_start.copy(),
                "goal": previous_goal.copy(),
                "endpoint": achieved.copy(),
                "endpoint_error_m": distance,
                "displacement_m": float(np.linalg.norm(achieved - previous_start)),
            }
            self.command_index += 1
            self._set_goal(self._sample_goal(achieved), achieved)
        self.last_command_event = event
        info = dict(info)
        info.update(
            command_switched=switched,
            command_event=event,
            command_index=int(self.command_index),
            command_steps=int(self.command_steps),
            command_goal=self.goal.copy(),
            command_start=self.command_start.copy(),
            native_task_reward_consumed=False,
        )
        return self._observation(self._native_observation), reward, terminated, truncated, info

    @property
    def torso_height(self) -> float:
        body = self.native.model.body("torso").id
        return float(self.native.data.xpos[body, 2])

    @property
    def torso_uprightness(self) -> float:
        body = self.native.model.body("torso").id
        return float(self.native.data.xmat[body].reshape(3, 3)[2, 2])

    def close(self) -> None:
        self.maze_env.close()
        super().close()
