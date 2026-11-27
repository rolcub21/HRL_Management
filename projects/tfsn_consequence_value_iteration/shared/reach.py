"""Domain-independent empirical one-command reach candidates."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


GOLDEN_FRACTION = (np.sqrt(5.0) - 1.0) / 2.0


def representative_reach_offsets(displacements, count, sample_weight=None):
    """Select radial representatives under a continuous empirical measure."""
    values = np.asarray(displacements, np.float32)
    if values.ndim != 2 or not len(values) or values.shape[1] < 1:
        raise ValueError("reach displacements must have shape [N,D]")
    count = int(count)
    if count <= 0:
        raise ValueError("reach representative count must be positive")
    if len(values) < count:
        raise ValueError("fewer recorded displacements than representatives")
    if not np.isfinite(values).all():
        raise ValueError("recorded displacements must be finite")
    if sample_weight is None:
        sample_weight = np.ones(len(values), np.float64)
    sample_weight = np.asarray(sample_weight, np.float64).reshape(-1)
    if len(sample_weight) != len(values) or np.any(sample_weight < 0.0):
        raise ValueError("reach weights must be nonnegative and aligned")
    positive = sample_weight > 0.0
    if int(positive.sum()) < count:
        raise ValueError("fewer positive-weight displacements than representatives")
    values = values[positive]
    sample_weight = sample_weight[positive]
    radius = np.linalg.norm(values, axis=1)
    order = np.argsort(radius, kind="stable")
    ordered_weight = sample_weight[order]
    cumulative = np.cumsum(ordered_weight)
    lower_mass = np.concatenate(([0.0], cumulative[:-1]))
    mass_center = np.empty(len(values), np.float64)
    mass_center[order] = 0.5 * (lower_mass + cumulative)
    boundaries = np.linspace(0.0, float(cumulative[-1]), count + 1)
    selected = []
    for index, (begin, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        overlap = (cumulative > begin) & (lower_mass < stop)
        candidates = order[overlap]
        if selected:
            candidates = candidates[
                ~np.isin(candidates, np.asarray(selected, np.int64))
            ]
        if len(candidates) == 0:
            remaining = np.setdiff1d(
                np.arange(len(values), dtype=np.int64),
                np.asarray(selected, np.int64),
                assume_unique=False,
            )
            midpoint = 0.5 * (begin + stop)
            candidates = remaining[
                np.argsort(np.abs(mass_center[remaining] - midpoint))[:1]
            ]
        if values.shape[1] == 2:
            # Preserve the frozen maze result exactly.
            angle = np.arctan2(values[candidates, 1], values[candidates, 0])
            desired = 2.0 * np.pi * ((index * GOLDEN_FRACTION) % 1.0)
            difference = np.abs(np.angle(np.exp(1j * (angle - desired))))
            chosen = int(candidates[int(np.argmin(difference))])
        else:
            # In higher-dimensional goal spaces, choose directional coverage
            # directly from the observed offsets.  The first direction is the
            # weighted empirical mean; subsequent directions maximize their
            # minimum angular separation from those already represented.  No
            # coordinate scale or environment threshold is introduced.
            norm = np.linalg.norm(values, axis=1, keepdims=True)
            unit = values / np.maximum(norm, np.finfo(np.float32).tiny)
            if not selected:
                mean_direction = np.sum(
                    sample_weight[:, None] * unit, axis=0
                )
                mean_norm = float(np.linalg.norm(mean_direction))
                if mean_norm > 0.0:
                    mean_direction /= mean_norm
                    score = unit[candidates] @ mean_direction
                else:
                    score = -np.abs(
                        mass_center[candidates] - 0.5 * (begin + stop)
                    )
            else:
                similarity = unit[candidates] @ unit[np.asarray(selected)].T
                score = -similarity.max(axis=1)
            chosen = int(candidates[int(np.argmax(score))])
        selected.append(chosen)
    return values[np.asarray(selected, np.int64)].copy()


class EmpiricalReachQuery:
    """Translate reach representatives and snap them to recorded endpoints."""

    def __init__(self, endpoint_xy, reach_offsets):
        self.endpoint_xy = np.asarray(endpoint_xy, np.float32)
        self.reach_offsets = np.asarray(reach_offsets, np.float32)
        if (
            self.endpoint_xy.ndim != 2
            or self.reach_offsets.ndim != 2
            or self.endpoint_xy.shape[1] != self.reach_offsets.shape[1]
        ):
            raise ValueError("reach endpoints and offsets must be aligned [N,D]")
        if len(self.endpoint_xy) == 0 or len(self.reach_offsets) == 0:
            raise ValueError("reach query needs endpoints and offsets")
        self.goal_dim = int(self.endpoint_xy.shape[1])
        self.tree = cKDTree(self.endpoint_xy)

    def query(self, xy, k=None, workers=-1):
        points = np.asarray(xy, np.float32)
        scalar = points.ndim == 1
        points = points.reshape(-1, self.goal_dim)
        count = len(self.reach_offsets) if k is None else int(k)
        if count != len(self.reach_offsets):
            raise ValueError(
                "empirical reach query requires its complete fixed candidate set"
            )
        desired = points[:, None, :] + self.reach_offsets[None, :, :]
        _distance, local = self.tree.query(
            desired.reshape(-1, self.goal_dim), k=1, workers=workers
        )
        local = np.asarray(local, np.int64).reshape(len(points), count)
        selected = self.endpoint_xy[local]
        distance = np.linalg.norm(
            selected - points[:, None, :], axis=2
        ).astype(np.float32)
        if scalar:
            return distance[0], local[0]
        return distance, local
