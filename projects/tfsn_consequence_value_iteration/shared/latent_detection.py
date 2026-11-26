"""Exact hidden-goal inference with an unknown distance-threshold reward scale.

The support is deliberately finite and data-derived: callers must pass the
recorded roadmap states themselves. This module never constructs a spatial
grid and never substitutes a posterior when the declared model is falsified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, distance


class PosteriorFalsifiedError(RuntimeError):
    """Raised when no hypothesis remains under the declared observation model."""


def exact_support_diameter(points_xy: np.ndarray) -> float:
    """Return the exact Euclidean diameter of a finite two-dimensional set."""
    points = np.asarray(points_xy, np.float64).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("reachable support must be finite and nonempty")
    points = np.unique(points, axis=0)
    if len(points) < 2:
        return 0.0
    try:
        boundary = points[ConvexHull(points).vertices]
    except QhullError:
        boundary = points
    return float(distance.pdist(boundary, metric="euclidean").max())


@dataclass(frozen=True)
class PosteriorSummary:
    candidate_count: int
    feasible_candidate_count: int
    rho_min: float
    rho_max: float
    marginal_evidence: float
    positive_observations: int
    negative_observations: int


def _unique_recorded_support(recorded_goal_xy: np.ndarray) -> np.ndarray:
    goals = np.asarray(recorded_goal_xy, np.float64)
    if goals.ndim != 2 or goals.shape[1] != 2 or not len(goals):
        raise ValueError("recorded goal candidates must have shape [N,2]")
    if not np.isfinite(goals).all():
        raise ValueError("recorded goal candidates must be finite")
    # A support is a set, not a collection-frequency distribution.
    return np.unique(goals, axis=0)


class LatentRadiusPosterior:
    """Analytic posterior for a hidden goal and threshold radius.

    The prior is uniform over unique recorded states and log-uniform over rho
    in ``[sigma, exact support diameter]``. Each goal retains an exact feasible
    interval ``[lower, upper)`` for rho; no radius discretization is used.
    """

    def __init__(self, recorded_goal_xy: np.ndarray, sigma: float):
        if not np.isfinite(sigma) or float(sigma) <= 0.0:
            raise ValueError("landing sigma must be finite and positive")
        self.goal_xy = _unique_recorded_support(recorded_goal_xy)
        self.rho_min = float(sigma)
        self.rho_max = exact_support_diameter(self.goal_xy)
        if not self.rho_max > self.rho_min:
            raise ValueError("reachable-support diameter must exceed landing sigma")
        self._prior_per_goal = 1.0 / float(len(self.goal_xy))
        self._prior_log_width = float(np.log(self.rho_max / self.rho_min))
        self.lower = np.full(len(self.goal_xy), self.rho_min, np.float64)
        self.upper = np.full(len(self.goal_xy), self.rho_max, np.float64)
        self.positive_observations = 0
        self.negative_observations = 0
        self._refresh()

    def copy(self) -> "LatentRadiusPosterior":
        other = object.__new__(type(self))
        other.goal_xy = self.goal_xy.copy()
        other.rho_min = self.rho_min
        other.rho_max = self.rho_max
        other._prior_per_goal = self._prior_per_goal
        other._prior_log_width = self._prior_log_width
        other.lower = self.lower.copy()
        other.upper = self.upper.copy()
        other.positive_observations = self.positive_observations
        other.negative_observations = self.negative_observations
        other._refresh()
        return other

    def _refresh(self) -> None:
        self.feasible = self.upper > self.lower
        interval_mass = np.zeros(len(self.goal_xy), np.float64)
        interval_mass[self.feasible] = (
            np.log(self.upper[self.feasible] / self.lower[self.feasible])
            / self._prior_log_width
        )
        self.unnormalized_goal_mass = self._prior_per_goal * interval_mass
        self.marginal_evidence = float(self.unnormalized_goal_mass.sum())
        self.goal_probability = (
            self.unnormalized_goal_mass / self.marginal_evidence
            if self.marginal_evidence > 0.0
            else np.zeros(len(self.goal_xy), np.float64)
        )

    @property
    def exhausted(self) -> bool:
        return not self.marginal_evidence > 0.0

    def summary(self) -> PosteriorSummary:
        return PosteriorSummary(
            candidate_count=len(self.goal_xy),
            feasible_candidate_count=int(self.feasible.sum()),
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            marginal_evidence=self.marginal_evidence,
            positive_observations=self.positive_observations,
            negative_observations=self.negative_observations,
        )

    def assert_not_exhausted(self, context: str = "") -> None:
        if self.exhausted:
            suffix = f" ({context})" if context else ""
            raise PosteriorFalsifiedError(
                "latent-radius task model falsified: every recorded roadmap "
                f"goal candidate has zero posterior mass{suffix}; no fallback "
                "posterior is permitted"
            )

    def update(self, observed_xy: np.ndarray, reward: np.ndarray) -> None:
        xy = np.asarray(observed_xy, np.float64).reshape(-1, 2)
        outcome = np.asarray(reward).reshape(-1)
        if len(xy) != len(outcome):
            raise ValueError("observations and rewards must be aligned")
        if not np.isfinite(xy).all():
            raise ValueError("observed positions must be finite")
        if not np.all(np.isin(outcome, (0, 1, False, True))):
            raise ValueError("threshold rewards must be binary")
        for point, positive in zip(xy, outcome.astype(bool)):
            d = np.linalg.norm(self.goal_xy - point[None], axis=1)
            if positive:
                self.lower = np.maximum(self.lower, d)
                self.positive_observations += 1
            else:
                self.upper = np.minimum(self.upper, d)
                self.negative_observations += 1
        self._refresh()
        self.assert_not_exhausted("after observed task consequence")

    def contact_probability(self, passage_xy: np.ndarray) -> np.ndarray:
        """Marginal contact probability for paths, integrating rho exactly."""
        self.assert_not_exhausted("before candidate scoring")
        passage = np.asarray(passage_xy, np.float64)
        single = passage.ndim == 2
        passage = passage.reshape(-1, passage.shape[-2], 2)
        if not np.isfinite(passage).all():
            raise ValueError("predicted passages must be finite")
        output = np.zeros(len(passage), np.float64)
        for index, path in enumerate(passage):
            d = np.linalg.norm(
                path[:, None, :] - self.goal_xy[None, :, :], axis=2
            ).min(axis=0)
            conditional = np.zeros(len(self.goal_xy), np.float64)
            certain = self.feasible & (d <= self.lower)
            possible = self.feasible & (d > self.lower) & (d < self.upper)
            conditional[certain] = 1.0
            conditional[possible] = (
                np.log(self.upper[possible] / d[possible])
                / np.log(self.upper[possible] / self.lower[possible])
            )
            output[index] = np.dot(self.goal_probability, conditional)
        output = np.clip(output, 0.0, 1.0).astype(np.float32)
        return output[0] if single else output

    def imagined_no_contact(self, passage_xy: np.ndarray) -> "LatentRadiusPosterior":
        path = np.asarray(passage_xy, np.float64).reshape(-1, 2)
        other = self.copy()
        d = np.linalg.norm(
            path[:, None, :] - other.goal_xy[None, :, :], axis=2
        ).min(axis=0)
        other.upper = np.minimum(other.upper, d)
        other.negative_observations += len(path)
        other._refresh()
        other.assert_not_exhausted("during consumptive lookahead")
        return other


class SigmaKernelPosterior:
    """Discrete-support control realization of the frozen R11 observation."""

    def __init__(self, recorded_goal_xy: np.ndarray, sigma: float):
        if not np.isfinite(sigma) or float(sigma) <= 0.0:
            raise ValueError("landing sigma must be finite and positive")
        self.goal_xy = _unique_recorded_support(recorded_goal_xy)
        self.sigma = float(sigma)
        self.goal_probability = np.full(
            len(self.goal_xy), 1.0 / len(self.goal_xy), np.float64
        )
        self.positive_observations = 0
        self.negative_observations = 0

    def copy(self) -> "SigmaKernelPosterior":
        other = object.__new__(type(self))
        other.goal_xy = self.goal_xy.copy()
        other.sigma = self.sigma
        other.goal_probability = self.goal_probability.copy()
        other.positive_observations = self.positive_observations
        other.negative_observations = self.negative_observations
        return other

    @property
    def exhausted(self) -> bool:
        return not float(self.goal_probability.sum()) > 0.0

    def assert_not_exhausted(self, context: str = "") -> None:
        if self.exhausted:
            suffix = f" ({context})" if context else ""
            raise PosteriorFalsifiedError(
                "sigma-kernel control posterior exhausted" + suffix
                + "; no fallback posterior is permitted"
            )

    def _kernel(self, distance_value: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * np.square(np.asarray(distance_value) / self.sigma))

    def update(self, observed_xy: np.ndarray, reward: np.ndarray) -> None:
        xy = np.asarray(observed_xy, np.float64).reshape(-1, 2)
        outcome = np.asarray(reward).reshape(-1)
        if len(xy) != len(outcome):
            raise ValueError("observations and rewards must be aligned")
        for point, positive in zip(xy, outcome.astype(bool)):
            likelihood = self._kernel(
                np.linalg.norm(self.goal_xy - point[None], axis=1)
            )
            self.goal_probability *= likelihood if positive else (1.0 - likelihood)
            total = float(self.goal_probability.sum())
            if total > 0.0:
                self.goal_probability /= total
            if positive:
                self.positive_observations += 1
            else:
                self.negative_observations += 1
        self.assert_not_exhausted("after observed task consequence")

    def contact_probability(self, passage_xy: np.ndarray) -> np.ndarray:
        self.assert_not_exhausted("before candidate scoring")
        passage = np.asarray(passage_xy, np.float64)
        single = passage.ndim == 2
        passage = passage.reshape(-1, passage.shape[-2], 2)
        output = np.zeros(len(passage), np.float64)
        for index, path in enumerate(passage):
            d = np.linalg.norm(
                path[:, None, :] - self.goal_xy[None, :, :], axis=2
            ).min(axis=0)
            output[index] = np.dot(self.goal_probability, self._kernel(d))
        output = np.clip(output, 0.0, 1.0).astype(np.float32)
        return output[0] if single else output

    def imagined_no_contact(self, passage_xy: np.ndarray) -> "SigmaKernelPosterior":
        path = np.asarray(passage_xy, np.float64).reshape(-1, 2)
        other = self.copy()
        d = np.linalg.norm(
            path[:, None, :] - other.goal_xy[None, :, :], axis=2
        ).min(axis=0)
        other.goal_probability *= 1.0 - other._kernel(d)
        total = float(other.goal_probability.sum())
        if total > 0.0:
            other.goal_probability /= total
        other.negative_observations += len(path)
        other.assert_not_exhausted("during consumptive lookahead")
        return other
