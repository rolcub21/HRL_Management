"""Domain-independent computations used by the frozen R74 hub method.

The module contains no maze geometry, reward radius, or domain coordinates.
Adapters supply a recorded graph, learned link probabilities, and the
executor-goal-space projection used by the common continuous method.
"""

from __future__ import annotations

import numpy as np
import torch


def _goal_matrix(values, label):
    """Validate a nonempty finite matrix in an executor goal space."""
    matrix = np.asarray(values, np.float64)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] < 1:
        raise ValueError(f"{label} must have shape [N,D] with D >= 1")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be finite")
    return matrix


def alias_hub_candidates(node_xy: np.ndarray, sigma: float):
    """Choose one recorded state per data-derived landing-scale cell."""
    xy = _goal_matrix(node_xy, "node goal coordinates")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("landing sigma must be positive and finite")
    origin = xy.min(axis=0)
    cell = np.floor((xy - origin) / sigma).astype(np.int32)
    _, first, inverse = np.unique(
        cell, axis=0, return_index=True, return_inverse=True
    )
    return first.astype(np.int32), inverse.astype(np.int32), origin


def area_weights(goal_xy: np.ndarray, sigma: float):
    """Give every occupied landing-scale cell equal prior area mass."""
    xy = _goal_matrix(goal_xy, "goal support")
    if not np.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError("landing sigma must be positive and finite")
    origin = xy.min(axis=0)
    cell = np.floor((xy - origin) / float(sigma)).astype(np.int32)
    _, inverse, count = np.unique(
        cell, axis=0, return_inverse=True, return_counts=True
    )
    occupied = int(len(count))
    weight = 1.0 / (float(occupied) * count[inverse].astype(np.float64))
    if not np.isclose(weight.sum(), 1.0, atol=1e-12):
        raise RuntimeError("area-corrected prior does not sum to one")
    return weight, occupied, origin


def weighted_central_covariance(displacement, sample_weight):
    """Empirical reach covariance, with no fitted scale or regularizer."""
    displacement = _goal_matrix(displacement, "reach displacements")
    sample_weight = np.asarray(sample_weight, np.float64).reshape(-1)
    if len(displacement) != len(sample_weight):
        raise ValueError("displacements and weights have different lengths")
    if not np.isfinite(displacement).all() or not np.isfinite(
        sample_weight
    ).all():
        raise ValueError("reach observations must be finite")
    if np.any(sample_weight < 0.0):
        raise ValueError("reach weights must be nonnegative")
    total = float(sample_weight.sum())
    if total <= 0.0:
        raise ValueError("reach observations have zero total weight")
    mean = (sample_weight[:, None] * displacement).sum(axis=0) / total
    centered = displacement - mean
    covariance = (
        np.einsum("n,ni,nj->ij", sample_weight, centered, centered) / total
    )
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise ValueError("empirical reach covariance is not positive definite")
    return mean, covariance


def empirical_reach_covariance(
    node_xy, source, endpoint, commands, sigma,
):
    """R74 reach estimator applied to an adapter's recorded transitions."""
    node_xy = _goal_matrix(node_xy, "node goal coordinates")
    source = np.asarray(source, np.int64).reshape(-1)
    endpoint = np.asarray(endpoint, np.int64).reshape(-1)
    commands = _goal_matrix(commands, "commands")
    if commands.shape[1] != node_xy.shape[1]:
        raise ValueError("commands and node goals must have the same dimension")
    if len(source) != len(endpoint) or len(source) != len(commands):
        raise ValueError("recorded transition arrays must be aligned")
    start = node_xy[source]
    finish = node_xy[endpoint]
    denominator = 2.0 * float(sigma) ** 2
    requested = 1.0 - np.exp(
        -np.square(commands - start).sum(axis=1) / denominator
    )
    realized = 1.0 - np.exp(
        -np.square(finish - start).sum(axis=1) / denominator
    )
    weight = requested * realized
    mean, covariance = weighted_central_covariance(finish - start, weight)
    return {
        "mean_displacement_m": mean,
        "covariance_m2": covariance,
        "inverse_covariance_per_m2": np.linalg.inv(covariance),
        "principal_standard_deviation_m": np.sqrt(
            np.linalg.eigvalsh(covariance)
        ),
        "radial_rms_m": float(np.sqrt(np.trace(covariance))),
        "sample_weight_sum": float(weight.sum()),
        "positive_weight_rows": int(np.count_nonzero(weight > 0.0)),
        "recorded_rows": int(len(weight)),
    }


def route_similarity(
    source_xy, hub_xy, observed_source_xy, observed_hub_xy,
    inverse_covariance,
):
    """R74's two-sided continuous reach-scale route kernel."""
    source_delta = np.asarray(source_xy, np.float64) - np.asarray(
        observed_source_xy, np.float64
    )
    hub_delta = np.asarray(hub_xy, np.float64) - np.asarray(
        observed_hub_xy, np.float64
    )
    if source_delta.shape[-1] != hub_delta.shape[-1]:
        raise ValueError("route source and hub coordinates must have one dimension")
    dimension = int(source_delta.shape[-1])
    inverse = np.asarray(inverse_covariance, np.float64)
    if inverse.shape != (dimension, dimension):
        raise ValueError("inverse reach covariance has the wrong shape")
    source_distance = np.einsum(
        "...i,ij,...j->...", source_delta, inverse, source_delta
    )
    hub_distance = np.einsum(
        "...i,ij,...j->...", hub_delta, inverse, hub_delta
    )
    return np.exp(-0.5 * (source_distance + hub_distance))


def route_mass(
    records, source_xy, hub_xy, remaining, inverse_covariance,
):
    """Transfer uncensored route outcomes at the exact decision horizon."""
    hubs = _goal_matrix(hub_xy, "hub coordinates")
    support = np.zeros(len(hubs), np.float64)
    arrivals = np.zeros(len(hubs), np.float64)
    for record in records:
        if record.get("censored"):
            continue
        if int(record["remaining_commands"]) != int(remaining):
            continue
        similarity = route_similarity(
            np.asarray(source_xy, np.float64)[None],
            hubs,
            np.asarray(record["source_xy"], np.float64)[None],
            np.asarray(record["hub_xy"], np.float64)[None],
            inverse_covariance,
        )
        support += similarity
        arrivals += similarity * float(record["arrived"])
    return support, arrivals


def posterior_probability(prior_probability, support, outcome_mass, prior=1.0):
    """One-model-observation Bernoulli posterior used at both levels."""
    return np.clip(
        (
            float(prior) * np.asarray(prior_probability, np.float64)
            + np.asarray(outcome_mass, np.float64)
        ) / (float(prior) + np.asarray(support, np.float64)),
        0.0,
        1.0,
    )


def latent_hub_coverage(posterior, hub_xy, chunk_size=64):
    """Exact latent-radius hub coverage in bounded numpy chunks."""
    posterior.assert_not_exhausted("before frozen hub scoring")
    hubs = _goal_matrix(hub_xy, "hub coordinates")
    if hubs.shape[1] != posterior.goal_xy.shape[1]:
        raise ValueError("hubs and posterior support must share a goal space")
    output = np.empty(len(hubs), np.float64)
    log_width = np.zeros(len(posterior.goal_xy), np.float64)
    log_width[posterior.feasible] = np.log(
        posterior.upper[posterior.feasible]
        / posterior.lower[posterior.feasible]
    )
    for begin in range(0, len(hubs), int(chunk_size)):
        stop = min(begin + int(chunk_size), len(hubs))
        distance = np.linalg.norm(
            posterior.goal_xy[:, None] - hubs[None, begin:stop], axis=2
        )
        conditional = np.zeros(distance.shape, np.float64)
        certain = posterior.feasible[:, None] & (
            distance <= posterior.lower[:, None]
        )
        possible = (
            posterior.feasible[:, None]
            & (distance > posterior.lower[:, None])
            & (distance < posterior.upper[:, None])
        )
        conditional[certain] = 1.0
        row, column = np.nonzero(possible)
        conditional[row, column] = (
            np.log(posterior.upper[row] / distance[row, column])
            / log_width[row]
        )
        output[begin:stop] = posterior.goal_probability @ conditional
    return np.clip(output, 0.0, 1.0)


def latent_hub_coverage_torch(posterior, hub_xy, device, chunk_size=64):
    """The same exact coverage computation in bounded GPU chunks."""
    posterior.assert_not_exhausted("before frozen batched hub scoring")
    hubs = _goal_matrix(hub_xy, "hub coordinates")
    if hubs.shape[1] != posterior.goal_xy.shape[1]:
        raise ValueError("hubs and posterior support must share a goal space")
    device = torch.device(device)
    goal = torch.as_tensor(
        posterior.goal_xy, dtype=torch.float64, device=device
    )
    probability = torch.as_tensor(
        posterior.goal_probability, dtype=torch.float64, device=device
    )
    lower = torch.as_tensor(
        posterior.lower, dtype=torch.float64, device=device
    )
    upper = torch.as_tensor(
        posterior.upper, dtype=torch.float64, device=device
    )
    feasible = torch.as_tensor(
        posterior.feasible, dtype=torch.bool, device=device
    )
    log_width = torch.zeros_like(lower)
    log_width[feasible] = torch.log(upper[feasible] / lower[feasible])
    output = np.empty(len(hubs), np.float64)
    with torch.inference_mode():
        for begin in range(0, len(hubs), int(chunk_size)):
            stop = min(begin + int(chunk_size), len(hubs))
            target = torch.as_tensor(
                hubs[begin:stop], dtype=torch.float64, device=device
            )
            distance = torch.linalg.vector_norm(
                goal[:, None] - target[None], dim=2
            )
            certain = feasible[:, None] & (distance <= lower[:, None])
            possible = (
                feasible[:, None]
                & (distance > lower[:, None])
                & (distance < upper[:, None])
            )
            conditional = torch.zeros_like(distance)
            conditional[certain] = 1.0
            row, column = torch.nonzero(possible, as_tuple=True)
            conditional[row, column] = (
                torch.log(upper[row] / distance[row, column])
                / log_width[row]
            )
            output[begin:stop] = (
                probability @ conditional
            ).detach().cpu().numpy()
    return np.clip(output, 0.0, 1.0)


class ExactBatchedReachability:
    """Exact multi-target Bellman recursion with two rolling GPU layers."""

    def __init__(
        self, source, endpoint, edge_weight, bridge_target, device,
        target_batch=64,
    ):
        self.device = torch.device(device)
        self.target_batch = int(target_batch)
        self.source = torch.as_tensor(
            source, dtype=torch.long, device=self.device
        )
        self.endpoint = torch.as_tensor(
            endpoint, dtype=torch.long, device=self.device
        )
        self.edge_weight = torch.as_tensor(
            edge_weight, dtype=torch.float32, device=self.device
        )
        target = np.asarray(bridge_target, np.int64)
        self.node_count, self.bridge_count = target.shape
        self.bridge_target = torch.as_tensor(
            target.reshape(-1), dtype=torch.long, device=self.device
        )

    def solve(self, targets, horizon, current_node, bridge_weight):
        targets = np.asarray(targets, np.int64).reshape(-1)
        if len(targets) > self.target_batch:
            raise ValueError("target batch exceeds declared computational batch")
        if not len(targets):
            return np.empty(0, np.float32)
        weight = torch.as_tensor(
            np.asarray(bridge_weight, np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        target = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        batch = len(targets)
        row = torch.arange(batch, dtype=torch.long, device=self.device)
        previous = torch.zeros(
            (batch, self.node_count), dtype=torch.float32, device=self.device
        )
        previous[row, target] = 1.0
        source_index = self.source.unsqueeze(0).expand(batch, -1)
        with torch.inference_mode():
            for _ in range(1, int(horizon) + 1):
                destination = previous.index_select(
                    1, self.bridge_target
                ).reshape(batch, self.node_count, self.bridge_count)
                destination = (
                    destination * weight.unsqueeze(0)
                    + previous.unsqueeze(2) * (1.0 - weight).unsqueeze(0)
                )
                best = destination.amax(dim=2)
                recorded = (
                    self.edge_weight.unsqueeze(0)
                    * previous.index_select(1, self.endpoint)
                    + (1.0 - self.edge_weight).unsqueeze(0)
                    * previous.index_select(1, self.source)
                )
                best.scatter_reduce_(
                    1, source_index, recorded, reduce="amax", include_self=True
                )
                best[row, target] = 1.0
                previous = best
        return previous[:, int(current_node)].detach().cpu().numpy()


def exact_hub_argmax(
    coverage, hub_nodes, remaining, current_node, solver, bridge_weight,
    route_support, route_arrivals, prior=1.0,
):
    """Exact global argmax using coverage as a valid reach upper bound."""
    coverage = np.asarray(coverage, np.float64)
    order = np.argsort(-coverage, kind="stable")
    selected = int(order[0])
    best_score = -1.0
    evaluated, reaches = [], []
    stop = False
    for begin in range(0, len(order), solver.target_batch):
        if best_score >= float(coverage[order[begin]]):
            break
        indices = order[begin:begin + solver.target_batch]
        bases = solver.solve(
            np.asarray(hub_nodes)[indices], remaining, current_node,
            bridge_weight,
        )
        for index, base in zip(indices, bases):
            if best_score >= float(coverage[index]):
                stop = True
                break
            corrected = float(posterior_probability(
                [base], [route_support[index]], [route_arrivals[index]], prior
            )[0])
            score = float(coverage[index]) * corrected
            evaluated.append(int(index))
            reaches.append(float(base))
            if score > best_score:
                best_score = score
                selected = int(index)
        if stop:
            break
    return {
        "selected": selected,
        "score": best_score,
        "evaluated_indices": np.asarray(evaluated, np.int64),
        "reaches": np.asarray(reaches, np.float32),
    }
