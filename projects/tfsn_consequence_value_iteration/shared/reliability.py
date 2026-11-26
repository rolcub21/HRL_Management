"""Domain-independent reliability estimates for recorded command transitions."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
    RECORDED_RELIABILITY_NEIGHBORS,
)


def recorded_transition_reliability(
    starts,
    commands,
    endpoints,
    state_scale,
    command_scale,
    neighbors=RECORDED_RELIABILITY_NEIGHBORS,
    prior_weight=RECORDED_EVIDENCE_PRIOR_WEIGHT,
):
    """Kernel posterior of repeatable realized deltas.

    ``prior_weight`` is a conservative unobserved pseudo-count. Recorded
    transitions contribute continuously by similarity in standardized
    state-command space. The learned model does not receive a second vote here;
    it enters separately through consequence bridge weights. No environment
    distance or success threshold enters.
    """

    starts = np.asarray(starts, np.float32)
    commands = np.asarray(commands, np.float32)
    endpoints = np.asarray(endpoints, np.float32)
    state_scale = np.asarray(state_scale, np.float32).reshape(starts.shape[1])
    command_scale = float(command_scale)
    prior_weight = float(prior_weight)
    if not (len(starts) == len(commands) == len(endpoints)) or not len(starts):
        raise ValueError("recorded transition rows must be nonempty and aligned")
    if command_scale <= 0.0 or prior_weight <= 0.0:
        raise ValueError("reliability scales and prior weight must be positive")
    if np.any(state_scale <= 0.0):
        raise ValueError("state scale must be positive")

    feature = np.concatenate(
        (starts / state_scale[None], commands / command_scale), axis=1
    )
    count = min(int(neighbors), len(feature))
    distance, index = cKDTree(feature).query(feature, k=count, workers=-1)
    if count == 1:
        distance = distance[:, None]
        index = index[:, None]
    similarity = np.exp(-0.5 * np.square(distance))
    realized_delta = endpoints - starts
    difference = (
        realized_delta[index] - realized_delta[:, None]
    ) / state_scale[None, None]
    agreement = np.exp(
        -0.5 * np.square(difference).sum(axis=2, dtype=np.float64)
    )
    support = similarity.sum(axis=1)
    agreement_mass = (similarity * agreement).sum(axis=1)
    reliability = agreement_mass / (support + prior_weight)
    return (
        reliability.astype(np.float32),
        (agreement_mass / support).astype(np.float32),
        support.astype(np.float32),
    )


def posterior_mean(prior, similarity, outcomes):
    """Apply the same declared evidence prior to reward-free online outcomes."""

    prior = np.asarray(prior, np.float32)
    similarity = np.asarray(similarity, np.float32)
    outcomes = np.asarray(outcomes, np.float32).reshape(-1)
    if similarity.ndim != 2 or similarity.shape[1] != len(outcomes):
        raise ValueError("online evidence matrix and outcomes must align")
    support = similarity.sum(axis=1)
    evidence = similarity @ outcomes
    posterior = (
        RECORDED_EVIDENCE_PRIOR_WEIGHT * prior + evidence
    ) / (RECORDED_EVIDENCE_PRIOR_WEIGHT + support)
    return np.clip(posterior, 0.0, 1.0).astype(np.float32), support.astype(
        np.float32
    )
