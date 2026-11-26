"""Domain-independent empirical one-command reach candidates."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


GOLDEN_FRACTION = (np.sqrt(5.0) - 1.0) / 2.0


def representative_reach_offsets(displacements, count, sample_weight=None):
    """Select radial representatives under a continuous empirical measure."""
    values = np.asarray(displacements, np.float32).reshape(-1, 2)
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
        angle = np.arctan2(values[candidates, 1], values[candidates, 0])
        desired = 2.0 * np.pi * ((index * GOLDEN_FRACTION) % 1.0)
        difference = np.abs(np.angle(np.exp(1j * (angle - desired))))
        selected.append(int(candidates[int(np.argmin(difference))]))
    return values[np.asarray(selected, np.int64)].copy()


class EmpiricalReachQuery:
    """Translate reach representatives and snap them to recorded endpoints."""

    def __init__(self, endpoint_xy, reach_offsets):
        self.endpoint_xy = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        self.reach_offsets = np.asarray(reach_offsets, np.float32).reshape(-1, 2)
        if len(self.endpoint_xy) == 0 or len(self.reach_offsets) == 0:
            raise ValueError("reach query needs endpoints and offsets")
        self.tree = cKDTree(self.endpoint_xy)

    def query(self, xy, k=None, workers=-1):
        points = np.asarray(xy, np.float32)
        scalar = points.ndim == 1
        points = points.reshape(-1, 2)
        count = len(self.reach_offsets) if k is None else int(k)
        if count != len(self.reach_offsets):
            raise ValueError(
                "empirical reach query requires its complete fixed candidate set"
            )
        desired = points[:, None, :] + self.reach_offsets[None, :, :]
        _distance, local = self.tree.query(
            desired.reshape(-1, 2), k=1, workers=workers
        )
        local = np.asarray(local, np.int64).reshape(len(points), count)
        selected = self.endpoint_xy[local]
        distance = np.linalg.norm(
            selected - points[:, None, :], axis=2
        ).astype(np.float32)
        if scalar:
            return distance[0], local[0]
        return distance, local
