"""Empirical primitive passages for consumption-aware PointMaze search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.shared.settings import PASSAGE_BINS

from .core import atomic_json, atomic_npz, sha256


def _linear_cell_index(cells, minimum, shape):
    local = np.asarray(cells, np.int64) - np.asarray(minimum, np.int64)
    if np.any(local < 0) or np.any(local >= np.asarray(shape)[None]):
        raise ValueError("passage lies outside its integration lattice")
    return (local[..., 0] * int(shape[1]) + local[..., 1]).astype(np.uint32)


def recorded_geometry(graph, sigma):
    start = graph.nodes[graph.source_index, :2]
    trajectory = np.concatenate((start[:, None], graph.primitive_xy), axis=1)
    delta = trajectory[:, 1:] - trajectory[:, :-1]
    midpoint = (trajectory[:, 1:] + trajectory[:, :-1]) * 0.5
    length = np.linalg.norm(delta, axis=2).astype(np.float32)
    cells = np.floor(midpoint / float(sigma)).astype(np.int32)
    groups = np.array_split(np.arange(length.shape[1]), PASSAGE_BINS)
    relative = np.zeros((graph.edge_count, PASSAGE_BINS, 2), np.float32)
    bin_length = np.zeros((graph.edge_count, PASSAGE_BINS), np.float32)
    for output, primitive in enumerate(groups):
        local_length = length[:, primitive]
        total = local_length.sum(axis=1)
        weighted = (
            midpoint[:, primitive] * local_length[:, :, None]
        ).sum(axis=1)
        fallback = midpoint[:, primitive].mean(axis=1)
        center = np.divide(
            weighted,
            total[:, None],
            out=fallback.copy(),
            where=total[:, None] > 0.0,
        )
        relative[:, output] = center - start
        bin_length[:, output] = total
    return start, midpoint, length, cells, relative, bin_length


def prepare_passages(graph, targets, sigma, output, shard_nodes=256):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bridge_directory = output / "bridge_shards"
    bridge_directory.mkdir(parents=True, exist_ok=True)
    (
        recorded_start,
        _midpoint,
        step_length,
        recorded_cells,
        relative,
        bin_length,
    ) = recorded_geometry(graph, sigma)
    displacement = (
        graph.nodes[graph.endpoint_index, :2] - recorded_start
    ).astype(np.float32)
    feature = np.concatenate((recorded_start, displacement), axis=1) / float(sigma)
    tree = cKDTree(feature)

    minimum = recorded_cells.reshape(-1, 2).min(axis=0)
    maximum = recorded_cells.reshape(-1, 2).max(axis=0)
    relative_cell = np.floor(relative / float(sigma)).astype(np.int32)
    node_cell = np.floor(graph.nodes[:, :2] / float(sigma)).astype(np.int32)
    minimum = np.minimum(
        minimum,
        node_cell.min(axis=0) + relative_cell.reshape(-1, 2).min(axis=0) - 1,
    )
    maximum = np.maximum(
        maximum,
        node_cell.max(axis=0) + relative_cell.reshape(-1, 2).max(axis=0) + 1,
    )
    shape = (maximum - minimum + 1).astype(np.int32)
    recorded_index = _linear_cell_index(recorded_cells, minimum, shape)
    recorded_path = output / "recorded.npz"
    atomic_npz(
        recorded_path,
        cell_index=recorded_index,
        step_length=step_length,
        relative_bin_xy=relative,
        bin_length=bin_length,
        grid_minimum_cell=minimum,
        grid_shape=shape,
    )

    rows = []
    width = targets.shape[1]
    match_distance = []
    for begin in range(0, graph.node_count, int(shard_nodes)):
        stop = min(begin + int(shard_nodes), graph.node_count)
        source = graph.nodes[begin:stop, :2]
        endpoint = graph.nodes[targets[begin:stop], :2]
        desired = endpoint - source[:, None]
        query = np.concatenate(
            (np.repeat(source[:, None], width, axis=1), desired), axis=2
        ).reshape(-1, 4) / float(sigma)
        distance, edge = tree.query(query, k=1, workers=-1)
        edge = np.asarray(edge, np.int32).reshape(stop - begin, width)
        translated = source[:, None, None] + relative[edge]
        cells = np.floor(translated / float(sigma)).astype(np.int32)
        cell_index = _linear_cell_index(cells, minimum, shape)
        path = bridge_directory / f"nodes_{begin:06d}_{stop:06d}.npz"
        atomic_npz(path, passage_edge_index=edge, cell_index=cell_index)
        rows.append({
            "file": path.name,
            "begin": int(begin),
            "stop": int(stop),
            "sha256": sha256(path),
        })
        match_distance.append(np.asarray(distance, np.float32))
        print(f"point passage nodes={stop}/{graph.node_count}", flush=True)
    match_distance = np.concatenate(match_distance)
    metadata = {
        "schema_version": 1,
        "integration_sigma_m": float(sigma),
        "passage_bins_computational_only_global": int(PASSAGE_BINS),
        "recorded_passage": "exact_reward_free_primitive_trajectory",
        "bridge_passage": "locally_matched_reward_free_trajectory_translated_to_source",
        "recorded_sha256": sha256(recorded_path),
        "grid_minimum_cell": minimum.tolist(),
        "grid_shape": shape.tolist(),
        "grid_cells": int(np.prod(shape)),
        "bridge_shards": rows,
        "match_standardized_distance_median": float(np.median(match_distance)),
        "match_standardized_distance_p90": float(np.quantile(match_distance, 0.9)),
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    atomic_json(output / "metadata.json", metadata)
    return metadata


class PassageGeometry:
    def __init__(self, directory, graph, targets, sigma):
        directory = Path(directory)
        self.graph = graph
        self.targets = np.asarray(targets, np.int32)
        self.sigma = float(sigma)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["integration_sigma_m"] != self.sigma:
            raise ValueError("passage geometry uses another landing precision")
        with np.load(directory / "recorded.npz") as stored:
            self.recorded_cell_index = stored["cell_index"]
            self.recorded_step_length = stored["step_length"]
            self.relative = stored["relative_bin_xy"].astype(np.float32)
            self.bin_length = stored["bin_length"].astype(np.float32)
            self.minimum = stored["grid_minimum_cell"].astype(np.int32)
            self.shape = stored["grid_shape"].astype(np.int32)
        x = (self.minimum[0] + np.arange(self.shape[0]) + 0.5) * self.sigma
        y = (self.minimum[1] + np.arange(self.shape[1]) + 0.5) * self.sigma
        xx, yy = np.meshgrid(x, y, indexing="ij")
        self.grid_xy = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
        self.bridge_shards = []
        for row in self.metadata["bridge_shards"]:
            path = directory / "bridge_shards" / row["file"]
            if sha256(path) != row["sha256"]:
                raise ValueError(f"passage shard changed: {path}")
            with np.load(path) as stored:
                self.bridge_shards.append({
                    "begin": row["begin"],
                    "stop": row["stop"],
                    "edge": stored["passage_edge_index"].astype(np.int32),
                    "cell_index": stored["cell_index"],
                })
        start = graph.nodes[graph.source_index, :2]
        displacement = graph.nodes[graph.endpoint_index, :2] - start
        self.local_tree = cKDTree(
            np.concatenate((start, displacement), axis=1) / self.sigma
        )

    def graph_gains(self, fields):
        grid_field = fields.search(self.grid_xy).astype(np.float32)
        recorded_line = (
            grid_field[self.recorded_cell_index] * self.recorded_step_length
        ).sum(axis=1, dtype=np.float64).astype(np.float32)
        bridge_line = np.zeros(self.targets.shape, np.float32)
        for shard in self.bridge_shards:
            value = (
                grid_field[shard["cell_index"]]
                * self.bin_length[shard["edge"]]
            ).sum(axis=2, dtype=np.float64).astype(np.float32)
            bridge_line[shard["begin"]:shard["stop"]] = value
        reachable = np.unique(self.recorded_cell_index)
        residual_mass = float(
            grid_field[reachable].sum(dtype=np.float64) * self.sigma**2
        )
        if residual_mass <= 0.0:
            return np.zeros_like(recorded_line), np.zeros_like(bridge_line)
        transverse = float(np.sqrt(2.0 * np.pi) * self.sigma)
        return (
            np.clip(transverse * recorded_line / residual_mass, 0.0, 1.0),
            np.clip(transverse * bridge_line / residual_mass, 0.0, 1.0),
        )

    def live_gains(self, start_xy, endpoint_xy, fields):
        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        start = np.broadcast_to(
            np.asarray(start_xy, np.float32).reshape(1, 2), endpoint.shape
        )
        query = np.concatenate((start, endpoint - start), axis=1) / self.sigma
        _distance, edge = self.local_tree.query(query, k=1, workers=-1)
        edge = np.asarray(edge, np.int32)
        sample = start[:, None] + self.relative[edge]
        line = (
            fields.search(sample) * self.bin_length[edge]
        ).sum(axis=1)
        grid_field = fields.search(self.grid_xy)
        residual_mass = float(
            grid_field[np.unique(self.recorded_cell_index)].sum(dtype=np.float64)
            * self.sigma**2
        )
        if residual_mass <= 0.0:
            return np.zeros(len(endpoint), np.float32)
        return np.clip(
            np.sqrt(2.0 * np.pi) * self.sigma * line / residual_mass,
            0.0,
            1.0,
        ).astype(np.float32)

    def live_passage_samples(self, start_xy, endpoint_xy):
        """Return the matched reward-free passage bins for live commands."""

        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        start = np.broadcast_to(
            np.asarray(start_xy, np.float32).reshape(1, 2), endpoint.shape
        )
        query = np.concatenate((start, endpoint - start), axis=1) / self.sigma
        _distance, edge = self.local_tree.query(query, k=1, workers=-1)
        edge = np.asarray(edge, np.int32).reshape(-1)
        return (start[:, None] + self.relative[edge]).astype(np.float32)
