"""Reward-free directed policies and finite-horizon GPI readouts.

The policies are fixed before a task is known.  Each policy is directed toward
one spatially spread reward-free experience anchor.  At every recorded state it
selects the recorded or model edge with the greatest expected reduction in
squared distance to its anchor::

    w_e (||x_i - a||^2 - ||x_{d_e} - a||^2).

This is the reduction in expected squared distance when edge failure leaves the
agent at its source.  It uses only the frozen edge probability and coordinates;
there is no distance cutoff, temperature, or environment-specific parameter.
"""

from __future__ import annotations

import numpy as np


def spread_anchor_indices(xy, count):
    """Deterministic farthest-first anchors over exact experience positions."""
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    count = int(count)
    if count <= 0:
        raise ValueError("anchor count must be positive")
    unique, first = np.unique(xy, axis=0, return_index=True)
    if count > len(unique):
        raise ValueError("anchor count exceeds unique experience positions")

    centroid = unique.mean(axis=0)
    squared = np.square(unique - centroid[None]).sum(axis=1)
    selected = [int(np.argmin(squared))]
    nearest = np.square(unique - unique[selected[0]][None]).sum(axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(nearest))
        selected.append(next_index)
        distance = np.square(unique - unique[next_index][None]).sum(axis=1)
        nearest = np.minimum(nearest, distance)
    return first[np.asarray(selected, np.int32)].astype(np.int32)


def directed_policy_edges(
    graph,
    bridge_targets,
    bridge_weight,
    anchor_index,
    node_chunk=256,
):
    """Choose one reward-free directed edge per policy and graph node.

    Returns arrays shaped ``[policies, nodes]``.  Kind is zero for a model
    bridge and one for an observed transition.
    """
    targets = np.asarray(bridge_targets, np.int32)
    weight = np.asarray(bridge_weight, np.float32)
    anchor_index = np.asarray(anchor_index, np.int32).reshape(-1)
    if targets.shape != weight.shape:
        raise ValueError("bridge target and weight matrices must align")
    if targets.shape[0] != graph.node_count:
        raise ValueError("one bridge row is required per graph node")
    if np.any(anchor_index < 0) or np.any(anchor_index >= graph.node_count):
        raise ValueError("anchor index falls outside graph")

    node_xy = np.asarray(graph.nodes[:, :2], np.float32)
    anchor_xy = node_xy[anchor_index]
    policy_count = len(anchor_index)
    policy_target = np.empty(
        (policy_count, graph.node_count), dtype=np.int32
    )
    policy_weight = np.empty(
        (policy_count, graph.node_count), dtype=np.float32
    )
    policy_kind = np.zeros(
        (policy_count, graph.node_count), dtype=np.uint8
    )
    policy_score = np.empty(
        (policy_count, graph.node_count), dtype=np.float32
    )

    # Select the best directed model edge for every policy and state.
    for begin in range(0, graph.node_count, int(node_chunk)):
        stop = min(begin + int(node_chunk), graph.node_count)
        source = node_xy[begin:stop]
        destination = node_xy[targets[begin:stop]]
        source_distance = np.square(
            source[:, None, :] - anchor_xy[None, :, :]
        ).sum(axis=2)
        destination_distance = np.square(
            destination[:, None, :, :] - anchor_xy[None, :, None, :]
        ).sum(axis=3)
        progress = weight[begin:stop, None, :] * (
            source_distance[:, :, None] - destination_distance
        )
        choice = np.argmax(progress, axis=2)
        row = np.arange(stop - begin)[:, None]
        policy = np.arange(policy_count)[None, :]
        policy_target[:, begin:stop] = targets[begin:stop][
            row, choice
        ].T
        policy_weight[:, begin:stop] = weight[begin:stop][
            row, choice
        ].T
        policy_score[:, begin:stop] = progress[row, policy, choice].T

    # Recorded transitions compete under exactly the same directed criterion.
    source_index = np.asarray(graph.source_index, np.int32)
    endpoint_index = np.asarray(graph.endpoint_index, np.int32)
    reliability = np.asarray(graph.edge_reliability, np.float32)
    edge_index = np.arange(len(source_index), dtype=np.int32)
    source_xy = node_xy[source_index]
    endpoint_xy = node_xy[endpoint_index]
    for policy, anchor in enumerate(anchor_xy):
        recorded_score = reliability * (
            np.square(source_xy - anchor[None]).sum(axis=1)
            - np.square(endpoint_xy - anchor[None]).sum(axis=1)
        )
        best_score = np.full(graph.node_count, -np.inf, np.float32)
        np.maximum.at(best_score, source_index, recorded_score)
        is_best = recorded_score == best_score[source_index]
        best_edge = np.full(graph.node_count, len(edge_index), np.int32)
        np.minimum.at(
            best_edge, source_index[is_best], edge_index[is_best]
        )
        use_recorded = best_score > policy_score[policy]
        nodes = np.flatnonzero(use_recorded)
        edges = best_edge[nodes]
        policy_target[policy, nodes] = endpoint_index[edges]
        policy_weight[policy, nodes] = reliability[edges]
        policy_kind[policy, nodes] = 1
        policy_score[policy, nodes] = best_score[nodes]

    return policy_target, policy_weight, policy_kind, policy_score


def gpi_successor_values(policy_target, policy_weight, node_field, horizon):
    """Read fixed directed successors and return their GPI envelope."""
    target = np.asarray(policy_target, np.int32)
    weight = np.asarray(policy_weight, np.float32)
    field = np.asarray(node_field, np.float32).reshape(-1)
    horizon = int(horizon)
    if target.shape != weight.shape or target.ndim != 2:
        raise ValueError("policy targets and weights must align in [L, N]")
    if target.shape[1] != len(field):
        raise ValueError("policy nodes and field must align")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if np.any(target < 0) or np.any(target >= len(field)):
        raise ValueError("policy target falls outside field")

    policy_value = np.zeros(target.shape, np.float32)
    envelope = np.zeros((horizon + 1, len(field)), np.float32)
    winning_policy = np.zeros(
        (horizon + 1, len(field)), dtype=np.uint16
    )
    coverage = []
    for remaining in range(1, horizon + 1):
        continuation = np.take_along_axis(policy_value, target, axis=1)
        policy_value = weight * (field[target] + continuation)
        winner = np.argmax(policy_value, axis=0)
        row = policy_value[winner, np.arange(len(field))]
        envelope[remaining] = row
        winning_policy[remaining] = winner.astype(np.uint16)
        coverage.append({
            "remaining": int(remaining),
            "positive_nodes": int(np.sum(row > 0)),
            "positive_fraction": float(np.mean(row > 0)),
            "meaningful_nodes": int(np.sum(row > 1e-6)),
            "meaningful_fraction": float(np.mean(row > 1e-6)),
            "maximum": float(row.max()),
            "policies_selected": int(len(np.unique(winner))),
        })
    return envelope, winning_policy, coverage
