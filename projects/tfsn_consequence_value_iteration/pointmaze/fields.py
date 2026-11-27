"""Exact task-memory fields used by both continuous domains."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.shared.settings import (
    FIELD_EVALUATION_BATCH,
)


class TaskFields:
    def __init__(self, sigma, goal_dim=2):
        self.sigma = float(sigma)
        self.goal_dim = int(goal_dim)
        if self.sigma <= 0.0:
            raise ValueError("landing precision must be positive")
        if self.goal_dim < 1:
            raise ValueError("goal dimension must be positive")
        self.coords = []
        self.rewards = []
        self._tree = None
        self._coord_array = None
        self._reward_array = None

    @property
    def contacted(self):
        return bool(self.rewards and np.max(self.rewards) > 0.0)

    def extend(self, coords, rewards):
        coords = np.asarray(coords, np.float32).reshape(-1, self.goal_dim)
        rewards = np.asarray(rewards, np.float32).reshape(-1)
        if len(coords) != len(rewards):
            raise ValueError("task coordinates and rewards must align")
        self.coords.extend(coords.tolist())
        self.rewards.extend(rewards.tolist())
        self._coord_array = np.asarray(self.coords, np.float32)
        self._reward_array = np.asarray(self.rewards, np.float32)
        self._tree = cKDTree(self._coord_array)

    def search(self, points):
        points = np.asarray(points, np.float32)
        shape = points.shape[:-1]
        flat = points.reshape(-1, self.goal_dim)
        if self._tree is None:
            return np.ones(shape, np.float32)
        distance = self._tree.query(flat, k=1, workers=-1)[0]
        searched = np.exp(-np.square(distance) / (2.0 * self.sigma**2))
        return (1.0 - searched).astype(np.float32).reshape(shape)

    def reward_rate(self, points, batch_size=FIELD_EVALUATION_BATCH):
        points = np.asarray(points, np.float32)
        shape = points.shape[:-1]
        flat = points.reshape(-1, self.goal_dim)
        if self._coord_array is None:
            return np.zeros(shape, np.float32)
        result = []
        denominator_scale = 2.0 * self.sigma**2
        for begin in range(0, len(flat), int(batch_size)):
            distance = flat[begin:begin + batch_size, None] - self._coord_array[None]
            kernel = np.exp(
                -np.square(distance).sum(axis=2) / denominator_scale
            )
            denominator = kernel.sum(axis=1)
            numerator = kernel @ self._reward_array
            result.append(np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0.0,
            ))
        return np.concatenate(result).astype(np.float32).reshape(shape)


class SuppliedGoalField:
    """Fixed Gaussian goal field; no acquisition or carried task memory."""

    def __init__(self, sigma, goal_xy):
        self.sigma = float(sigma)
        self.goal_xy = np.asarray(goal_xy, np.float32).reshape(-1)
        if not len(self.goal_xy):
            raise ValueError("a supplied goal must have at least one coordinate")
        self.goal_dim = int(len(self.goal_xy))
        self.coords = [self.goal_xy.tolist()]
        self.rewards = [1.0]
        self.planner_cache_token = (
            "supplied_goal", *map(float, self.goal_xy), self.sigma,
        )

    @property
    def contacted(self):
        return True

    def reward_rate(self, points):
        points = np.asarray(points, np.float32)
        distance2 = np.square(points - self.goal_xy).sum(axis=-1)
        return np.exp(-distance2 / (2.0 * self.sigma**2)).astype(np.float32)

    def search(self, points):
        raise RuntimeError("supplied-goal control has no acquisition field")

