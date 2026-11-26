"""Finite-horizon reductions over one frozen graph edge set."""

from __future__ import annotations

import numpy as np


def finite_horizon_edge_values(
    graph,
    targets,
    weight,
    node_field,
    horizon,
    reduction,
):
    """Back up identical recorded and model edges by max or uniform mean.

    The edge contribution exactly matches the frozen R22 implementation:
    ``w * (F(destination) + V(destination))``.  In particular, this helper
    deliberately does not add a failure self-loop that R22 did not contain.
    """
    targets = np.asarray(targets, np.int32)
    weight = np.asarray(weight, np.float32)
    node_field = np.asarray(node_field, np.float32).reshape(-1)
    horizon = int(horizon)
    if reduction not in {"max", "mean"}:
        raise ValueError("reduction must be 'max' or 'mean'")
    if targets.shape != weight.shape:
        raise ValueError("target and weight matrices must align")
    if targets.shape[0] != graph.node_count:
        raise ValueError("one target row is required per graph node")
    if len(node_field) != graph.node_count:
        raise ValueError("node field has the wrong length")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    values = np.zeros((horizon + 1, graph.node_count), np.float32)
    recorded_count = np.bincount(
        graph.source_index, minlength=graph.node_count
    ).astype(np.int32)
    out_count = recorded_count + int(targets.shape[1])
    if np.any(out_count <= 0):
        raise ValueError("every node must have an outgoing option")

    for remaining in range(1, horizon + 1):
        recorded_q = graph.edge_reliability * (
            node_field[graph.endpoint_index]
            + values[remaining - 1, graph.endpoint_index]
        )
        bridge_q = weight * (
            node_field[targets] + values[remaining - 1, targets]
        )
        if reduction == "max":
            row = np.zeros(graph.node_count, np.float32)
            np.maximum.at(row, graph.source_index, recorded_q)
            row = np.maximum(row, bridge_q.max(axis=1))
        else:
            row = np.zeros(graph.node_count, np.float32)
            np.add.at(row, graph.source_index, recorded_q)
            row += bridge_q.sum(axis=1, dtype=np.float32)
            row /= out_count
        values[remaining] = row
    return values


def value_coverage(values):
    rows = []
    for remaining, row in enumerate(np.asarray(values)[1:], start=1):
        rows.append({
            "remaining": int(remaining),
            "positive_nodes": int(np.sum(row > 0)),
            "positive_fraction": float(np.mean(row > 0)),
            "meaningful_nodes": int(np.sum(row > 1e-6)),
            "meaningful_fraction": float(np.mean(row > 1e-6)),
            "maximum": float(row.max()),
        })
    return rows
