"""Recorded-transition road map and finite-horizon TF-SN backups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.shared.reach import (
    EmpiricalReachQuery,
    representative_reach_offsets,
)
from projects.tfsn_consequence_value_iteration.shared.reliability import (
    recorded_transition_reliability,
)


@dataclass
class RecordedRoadmap:
    nodes: np.ndarray
    source_index: np.ndarray
    endpoint_index: np.ndarray
    commands: np.ndarray
    primitive_xy: np.ndarray
    state_scale: np.ndarray
    edge_reliability: np.ndarray | None = None
    goal_dim: int = 2

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, np.float32)
        self.source_index = np.asarray(self.source_index, np.int32)
        self.endpoint_index = np.asarray(self.endpoint_index, np.int32)
        self.commands = np.asarray(self.commands, np.float32)
        self.primitive_xy = np.asarray(self.primitive_xy, np.float32)
        self.state_scale = np.asarray(self.state_scale, np.float32)
        self.edge_reliability = (
            np.ones(len(self.source_index), np.float32)
            if self.edge_reliability is None
            else np.asarray(self.edge_reliability, np.float32)
        )
        self.goal_dim = int(self.goal_dim)
        if self.nodes.ndim != 2 or self.nodes.shape[1] < self.goal_dim:
            raise ValueError("roadmap nodes must be [N,state_dim >= goal_dim]")
        if not 1 <= self.goal_dim <= self.nodes.shape[1]:
            raise ValueError("roadmap goal dimension is invalid")
        edge_count = len(self.source_index)
        if not (
            len(self.endpoint_index) == edge_count
            and len(self.commands) == edge_count
            and len(self.primitive_xy) == edge_count
            and len(self.edge_reliability) == edge_count
        ):
            raise ValueError("roadmap edge arrays are not aligned")
        if (
            self.commands.ndim != 2
            or self.commands.shape[1] != self.goal_dim
        ):
            raise ValueError("commands must be [edge,goal_dim]")
        if (
            self.primitive_xy.ndim != 3
            or self.primitive_xy.shape[2] != self.goal_dim
        ):
            raise ValueError("primitive paths must be [edge,step,goal_dim]")
        if self.state_scale.shape != (self.nodes.shape[1],):
            raise ValueError("state scale must match roadmap state dimension")
        if np.any(self.source_index < 0) or np.any(self.endpoint_index < 0):
            raise ValueError("roadmap indices must be nonnegative")
        if np.any(self.source_index >= len(self.nodes)) or np.any(
            self.endpoint_index >= len(self.nodes)
        ):
            raise ValueError("roadmap edge index exceeds node table")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.nodes,
                self.commands,
                self.primitive_xy,
                self.edge_reliability,
            )
        ):
            raise ValueError("roadmap contains non-finite data")
        if np.any(self.edge_reliability < 0.0) or np.any(
            self.edge_reliability > 1.0
        ):
            raise ValueError("edge reliability must lie in [0,1]")
        self._tree = cKDTree(self.nodes / self.state_scale[None])

    @property
    def node_count(self):
        return int(len(self.nodes))

    @property
    def edge_count(self):
        return int(len(self.source_index))

    def project(self, states):
        states = np.asarray(states, np.float32).reshape(-1, self.nodes.shape[1])
        distance, index = self._tree.query(
            states / self.state_scale[None], k=1, workers=-1
        )
        return np.asarray(index, np.int32), np.asarray(distance, np.float32)

    def save(self, path):
        from .core import atomic_npz

        atomic_npz(
            path,
            nodes=self.nodes,
            source_index=self.source_index,
            endpoint_index=self.endpoint_index,
            commands=self.commands,
            primitive_xy=self.primitive_xy,
            state_scale=self.state_scale,
            edge_reliability=self.edge_reliability,
            goal_dim=np.asarray(self.goal_dim, np.int16),
        )

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as stored:
            values = {name: stored[name] for name in stored.files}
        return cls(**values)


def graph_from_dataset(dataset, sigma):
    """Preserve every command strand and every recorded transition."""
    node_rows, source_rows, endpoint_rows = [], [], []
    command_rows, primitive_rows = [], []
    node_offset = 0
    goal_dim = int(np.asarray(dataset.commands).shape[1])
    for trajectory in np.unique(dataset.trajectory_id):
        index = np.flatnonzero(dataset.trajectory_id == trajectory)
        starts = dataset.starts[index]
        endpoints = dataset.endpoints[index]
        # A shard is a contiguous command strand by construction.  Validate
        # this instead of silently snapping its boundaries together.
        if len(index) > 1 and not np.allclose(
            starts[1:], endpoints[:-1], rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"trajectory {trajectory} has broken boundaries")
        nodes = np.concatenate((starts[:1], endpoints), axis=0)
        node_rows.append(nodes)
        source_rows.append(node_offset + np.arange(len(index), dtype=np.int32))
        endpoint_rows.append(
            node_offset + 1 + np.arange(len(index), dtype=np.int32)
        )
        command_rows.append(dataset.commands[index])
        primitive_rows.append(dataset.primitives[index, :, :goal_dim])
        node_offset += len(nodes)
    nodes = np.concatenate(node_rows).astype(np.float32)
    scale = np.maximum(dataset.starts.std(axis=0), 1e-6).astype(np.float32)
    scale[:goal_dim] = float(sigma)
    reliability, _agreement, _support = recorded_transition_reliability(
        np.concatenate(
            [nodes[source] for nodes, source in zip(
                node_rows, [np.arange(len(row) - 1) for row in node_rows]
            )]
        ),
        np.concatenate(command_rows),
        np.concatenate(
            [nodes[1:] for nodes in node_rows]
        ),
        scale,
        sigma,
    )
    return RecordedRoadmap(
        nodes=nodes,
        source_index=np.concatenate(source_rows),
        endpoint_index=np.concatenate(endpoint_rows),
        commands=np.concatenate(command_rows),
        primitive_xy=np.concatenate(primitive_rows),
        state_scale=scale,
        edge_reliability=reliability,
        goal_dim=goal_dim,
    )


def reach_targets(
    graph,
    count,
    sigma=None,
    continuous_weighting=False,
):
    displacement = (
        graph.nodes[graph.endpoint_index, : graph.goal_dim]
        - graph.nodes[graph.source_index, : graph.goal_dim]
    ).astype(np.float32)
    sample_weight = None
    if continuous_weighting:
        if sigma is None or float(sigma) <= 0.0:
            raise ValueError("movement reach requires a positive sigma")
        requested = np.linalg.norm(
            graph.commands
            - graph.nodes[graph.source_index, : graph.goal_dim], axis=1
        )
        achieved = np.linalg.norm(displacement, axis=1)
        denominator = 2.0 * float(sigma) ** 2
        requested_advance = 1.0 - np.exp(-np.square(requested) / denominator)
        achieved_advance = 1.0 - np.exp(-np.square(achieved) / denominator)
        sample_weight = requested_advance * achieved_advance
    offsets = representative_reach_offsets(
        displacement, int(count), sample_weight=sample_weight
    )
    landmark_index = np.unique(graph.endpoint_index).astype(np.int32)
    query = EmpiricalReachQuery(
        graph.nodes[landmark_index, : graph.goal_dim], offsets
    )
    distance, local = query.query(
        graph.nodes[:, : graph.goal_dim], workers=-1
    )
    targets = landmark_index[local].astype(np.int32)
    return landmark_index, targets, distance, offsets


def persistent_values(graph, targets, weight, node_field, horizon):
    """Additive finite-horizon backup for return and supplied goals."""
    targets = np.asarray(targets, np.int32)
    weight = np.asarray(weight, np.float32)
    field = np.asarray(node_field, np.float32).reshape(graph.node_count)
    values = np.zeros((int(horizon) + 1, graph.node_count), np.float32)
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        row = np.full(graph.node_count, -np.inf, np.float32)
        recorded_success = (
            field[graph.endpoint_index] + previous[graph.endpoint_index]
        )
        recorded_q = (
            graph.edge_reliability * recorded_success
            + (1.0 - graph.edge_reliability) * previous[graph.source_index]
        )
        np.maximum.at(row, graph.source_index, recorded_q)
        bridge_success = field[targets] + previous[targets]
        bridge_q = (
            weight * bridge_success
            + (1.0 - weight) * previous[:, None]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = np.where(np.isfinite(row), row, previous)
    return values


def bounded_search_values(
    graph, targets, weight, recorded_gain, bridge_gain, horizon
):
    """Bounded consumption-aware first-contact backup for acquisition."""
    targets = np.asarray(targets, np.int32)
    weight = np.asarray(weight, np.float32)
    recorded_gain = np.clip(
        np.asarray(recorded_gain, np.float32), 0.0, 1.0
    )
    bridge_gain = np.clip(
        np.asarray(bridge_gain, np.float32), 0.0, 1.0
    )
    values = np.zeros((int(horizon) + 1, graph.node_count), np.float32)
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        recorded_success = recorded_gain + (
            1.0 - recorded_gain
        ) * previous[graph.endpoint_index]
        row = np.full(graph.node_count, -np.inf, np.float32)
        recorded_q = (
            graph.edge_reliability * recorded_success
            + (1.0 - graph.edge_reliability) * previous[graph.source_index]
        )
        np.maximum.at(row, graph.source_index, recorded_q)
        bridge_success = bridge_gain + (
            1.0 - bridge_gain
        ) * previous[targets]
        bridge_q = (
            weight * bridge_success
            + (1.0 - weight) * previous[:, None]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = np.clip(
            np.where(np.isfinite(row), row, previous), 0.0, 1.0
        )
    return values


def alias_kernel(distance, sigma):
    distance = np.asarray(distance, np.float32)
    return np.exp(-np.square(distance) / (2.0 * float(sigma) ** 2))
