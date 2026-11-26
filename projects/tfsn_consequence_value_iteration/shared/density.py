"""Reward-free density correction for repeated state/command situations."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _gaussian_density(query, reference, radius):
    query_tree = cKDTree(np.asarray(query, np.float32))
    reference_tree = cKDTree(np.asarray(reference, np.float32))
    matrix = query_tree.sparse_distance_matrix(
        reference_tree,
        max_distance=float(radius),
        output_type="coo_matrix",
    )
    density = np.bincount(
        matrix.row,
        weights=np.exp(-0.5 * np.square(matrix.data)),
        minlength=len(query),
    ).astype(np.float64)
    # A query outside every retained Gaussian tail is one distinct situation.
    return np.maximum(density, 1.0)


def situation_density_weights(
    states,
    commands,
    training_index,
    position_scale,
    truncation_sigmas,
):
    """Give each local state/command neighborhood approximately unit mass.

    Position and command coordinates use the learned landing scale.  Remaining
    state coordinates use their training-set standard deviation.  Density is
    estimated without rewards or environment geometry.
    """
    states = np.asarray(states, np.float32)
    commands = np.asarray(commands, np.float32).reshape(-1, 2)
    training_index = np.asarray(training_index, np.int64)
    if states.ndim != 2 or len(states) != len(commands):
        raise ValueError("states and commands must be aligned")
    if not len(training_index):
        raise ValueError("density correction needs training rows")
    position_scale = float(position_scale)
    if position_scale <= 0.0:
        raise ValueError("position scale must be positive")

    extra_scale = np.maximum(
        states[training_index, 2:].std(axis=0), 1e-6
    ).astype(np.float32)
    situation = np.concatenate(
        (
            states[:, :2] / position_scale,
            states[:, 2:] / extra_scale[None],
            commands / position_scale,
        ),
        axis=1,
    ).astype(np.float32)
    training = situation[training_index]
    density = _gaussian_density(
        situation, training, float(truncation_sigmas)
    )
    weight = 1.0 / density
    weight /= weight[training_index].mean()
    effective = float(
        np.square(weight[training_index].sum())
        / np.square(weight[training_index]).sum()
    )
    return weight.astype(np.float32), {
        "estimator": (
            "inverse_gaussian_state_command_density_truncated_at_"
            "global_kernel_tail"
        ),
        "position_scale_m": position_scale,
        "extra_state_scale": extra_scale.tolist(),
        "truncation_sigmas_global": float(truncation_sigmas),
        "training_weight_minimum": float(weight[training_index].min()),
        "training_weight_median": float(np.median(weight[training_index])),
        "training_weight_maximum": float(weight[training_index].max()),
        "effective_training_situations": effective,
    }
