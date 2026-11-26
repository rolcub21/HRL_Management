"""Farama GoalReachAnt reward/observation with maze walls and 100-step handoffs.

Method changes relative to the pinned Farama source are deliberately limited to:

1. the AntMaze Medium Diverse geometry and benchmark reset distribution; and
2. a continuing command contract that switches goals on contact or after 100 steps.

The 29-dimensional observation, dense directional reward, action space, and
underlying Ant dynamics are kept identical to Farama's GoalReachAnt setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import gymnasium_robotics  # noqa: F401 - registers AntMaze environments
import numpy as np


ENVIRONMENT = "AntMaze_Medium_Diverse_GR-v4"
COMMAND_HORIZON = 100


class GoalReachAnt(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuing_task: bool = True,
        goal_threshold: float = 0.20,
        forward_reward_scale: float = 10.0,
        command_horizon: int = COMMAND_HORIZON,
        command_distance_bank: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not continuing_task:
            raise ValueError("The chained executor protocol requires continuing_task=True")
        if int(command_horizon) != COMMAND_HORIZON:
            raise ValueError("The deployed TF-SN contract is exactly 100 primitive steps")
        if command_distance_bank is None:
            raise ValueError("A frozen empirical TF-SN command-distance bank is required")

        bank_path = Path(command_distance_bank).expanduser().resolve()
        with np.load(bank_path, allow_pickle=False) as stored:
            distances = np.asarray(stored["requested_distance_m"], dtype=np.float64)
        if distances.ndim != 1 or distances.size == 0:
            raise ValueError("The command-distance bank must be a nonempty vector")
        if not np.isfinite(distances).all() or np.any(distances <= 0.0):
            raise ValueError("The command-distance bank contains invalid distances")

        self.goal_threshold = float(goal_threshold)
        self.forward_reward_scale = float(forward_reward_scale)
        self.continuing_task = True
        self.command_horizon = int(command_horizon)
        self.command_distances = distances
        self.render_mode = render_mode

        # This is the only dynamics/geometry substitution: Farama's Ant is
        # instantiated inside the benchmark's Medium Diverse maze XML.
        self.maze_env = gym.make(
            ENVIRONMENT,
            render_mode=render_mode,
            continuing_task=True,
            reset_target=False,
            max_episode_steps=1000,
            **kwargs,
        )
        base = self.maze_env.unwrapped
        self.ant_env = base.ant_env
        self.maze = base.maze
        self.action_space = self.maze_env.action_space
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(29,), dtype=np.float64
        )
        self.goal = np.zeros(2, dtype=np.float64)
        self.command_start = np.zeros(2, dtype=np.float64)
        self.command_steps = 0
        self.command_index = 0
        self.last_command_event: Optional[dict] = None
        self._forced_sequence: Optional[np.ndarray] = None
        self._forced_sequence_index = 0

    @property
    def _base(self):
        return self.maze_env.unwrapped

    def _is_free(self, point: np.ndarray) -> bool:
        row, col = self.maze.cell_xy_to_rowcol(np.asarray(point, dtype=np.float64))
        if row < 0 or col < 0 or row >= self.maze.map_length or col >= self.maze.map_width:
            return False
        if self.maze.maze_map[int(row)][int(col)] == 1:
            return False
        # Reuse the benchmark's own goal-support footprint inside every free
        # cell. This keeps targets clear of wall geometry without introducing
        # an executor-specific clearance radius.
        center = self.maze.cell_rowcol_to_xy(np.asarray([row, col]))
        half_width = self._base.position_noise_range * self.maze.maze_size_scaling
        return bool(np.max(np.abs(np.asarray(point) - center)) <= half_width)

    def _sample_goal(self, achieved_goal: np.ndarray) -> np.ndarray:
        # Farama's direction remains uniform. Only the radial law changes: it
        # is replayed from TF-SN's frozen, reward-free requested distances.
        for _ in range(10_000):
            theta = 2.0 * np.pi * self.np_random.uniform()
            distance = float(self.np_random.choice(self.command_distances))
            goal = achieved_goal + distance * np.asarray(
                [np.cos(theta), np.sin(theta)], dtype=np.float64
            )
            if self._is_free(goal):
                return goal
        raise RuntimeError("Could not sample a free-space command after 10,000 draws")

    def _set_goal(self, goal: np.ndarray, achieved_goal: np.ndarray) -> None:
        goal = np.asarray(goal, dtype=np.float64).reshape(2)
        if not self._is_free(goal):
            raise ValueError(f"Command endpoint is not in free space: {goal.tolist()}")
        self.goal = goal.copy()
        self.command_start = np.asarray(achieved_goal, dtype=np.float64).copy()
        self.command_steps = 0
        self._base.goal = goal.copy()
        self._base.update_target_site_pos()

    def _next_goal(self, achieved_goal: np.ndarray) -> np.ndarray:
        if self._forced_sequence is not None:
            if self._forced_sequence_index >= len(self._forced_sequence):
                # Evaluation holds the last target after its frozen sequence.
                return self._forced_sequence[-1].copy()
            goal = self._forced_sequence[self._forced_sequence_index].copy()
            self._forced_sequence_index += 1
            return goal
        return self._sample_goal(achieved_goal)

    def install_goal_sequence(self, goals: np.ndarray) -> np.ndarray:
        goals = np.asarray(goals, dtype=np.float64)
        if goals.ndim != 2 or goals.shape[1] != 2 or len(goals) == 0:
            raise ValueError("A goal sequence must have shape (commands, 2)")
        if not all(self._is_free(goal) for goal in goals):
            raise ValueError("The frozen goal sequence contains a wall endpoint")
        achieved = self.command_start.copy()
        self._forced_sequence = goals.copy()
        self._forced_sequence_index = 1
        self.command_index = 0
        self.last_command_event = None
        self._set_goal(goals[0], achieved)
        return self._observation(self._last_maze_observation)

    def _observation(self, maze_observation: dict) -> np.ndarray:
        raw = np.asarray(maze_observation["observation"], dtype=np.float64).reshape(-1)
        achieved = np.asarray(maze_observation["achieved_goal"], dtype=np.float64).reshape(2)
        observation = np.concatenate([raw, self.goal - achieved])
        if observation.shape != (29,) or observation.dtype != np.float64:
            raise RuntimeError("The Farama 29D observation contract changed")

        # Semantic, not merely dimensional, assertion: AntMaze's 27D payload
        # must be the original Ant observation with absolute x/y removed.
        ant_raw = np.asarray(self.ant_env._get_obs(), dtype=np.float64)
        if ant_raw.shape != (29,) or not np.allclose(raw, ant_raw[2:], atol=0.0, rtol=0.0):
            raise RuntimeError("AntMaze proprioception differs from Farama's adapter")
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        maze_options = None if options is None else options.get("maze_options")
        maze_observation, info = self.maze_env.reset(seed=seed, options=maze_options)
        self._last_maze_observation = maze_observation
        achieved = np.asarray(maze_observation["achieved_goal"], dtype=np.float64)
        self._forced_sequence = None
        self._forced_sequence_index = 0
        self.command_index = 0
        self.last_command_event = None
        self._set_goal(self._sample_goal(achieved), achieved)
        info = dict(info)
        info.update(
            command_index=0,
            command_steps=0,
            command_goal=self.goal.copy(),
            command_start=self.command_start.copy(),
        )
        return self._observation(maze_observation), info

    def step(self, action):
        previous_goal = self.goal.copy()
        previous_start = self.command_start.copy()
        maze_observation, _maze_reward, terminated, truncated, info = self.maze_env.step(action)
        self._last_maze_observation = maze_observation
        achieved = np.asarray(maze_observation["achieved_goal"], dtype=np.float64)
        self.command_steps += 1

        # Reward is copied exactly from Farama and is computed against the
        # command that generated this primitive transition.
        reward = self.compute_reward(achieved, previous_goal, info)
        distance = float(np.linalg.norm(achieved - previous_goal))
        reached = distance <= self.goal_threshold
        timed_out = self.command_steps >= self.command_horizon
        switched = bool(reached or timed_out)
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
            self._set_goal(self._next_goal(achieved), achieved)
        self.last_command_event = event

        info = dict(info)
        info.update(
            command_switched=switched,
            command_event=event,
            command_index=int(self.command_index),
            command_steps=int(self.command_steps),
            command_goal=self.goal.copy(),
            command_start=self.command_start.copy(),
        )
        return self._observation(maze_observation), reward, terminated, truncated, info

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Farama's dense directional velocity reward, unchanged."""
        goal_vector = desired_goal - achieved_goal
        velocity_vector = [info["x_velocity"], info["y_velocity"]]
        reward_forward = np.dot(velocity_vector, goal_vector) / np.linalg.norm(goal_vector)
        return (
            self.forward_reward_scale * reward_forward
            + info["reward_ctrl"]
            + info["reward_survive"]
        )

    def render(self):
        return self.maze_env.render()

    @property
    def torso_height(self) -> float:
        return float(self.ant_env.data.body("torso").xpos[2])

    @property
    def torso_uprightness(self) -> float:
        return float(self.ant_env.data.body("torso").xmat.reshape(3, 3)[2, 2])

    def close(self) -> None:
        self.maze_env.close()
        super().close()

