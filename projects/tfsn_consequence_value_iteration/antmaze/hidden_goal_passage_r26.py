"""R26 hidden-goal acquisition with passage-integrated search gain.

R22 is unchanged for return.  During acquisition only, an edge's immediate
field readout is the line integral of the exact residual-search field along
the passage swept by that edge.  Recorded edges use their reward-free
primitive trajectories.  Learned bridges use the straight segment from the
source state to the recorded destination and retain R22's connection weight.

The numerical integration lattice and bridge sample spacing are both the
deployed R22 landing precision.  They are numerical realizations of the
integral, not additional planning scales or candidate filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_r22 as baseline,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    _save_npz_atomic,
)
from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    value_coverage,
)
from scripts.antmaze.current_tfsn.data import part_paths
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


METHOD_REVISION = "antmaze_hidden_goal_passage_gain_r26"
BASE_METHOD_REVISION = r22.METHOD_REVISION
CLAIM_LEVEL = "consumed_development_hidden_goal_passage_gain_test"
ALL_TASK_IDS = tuple(range(15))
GATE_TASK_IDS = baseline.GATE_TASK_IDS
DEFAULT_EPISODES = 5
NODE_SHARD_SIZE = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(root):
    directory = Path(root) / "model_planning" / "passage_gain_r26"
    return {
        "directory": directory,
        "metadata": directory / "metadata.json",
        "recorded": directory / "recorded_passages.npz",
        "bridge_shards": directory / "bridge_shards",
    }


def _round_contract(graph):
    rounds = [Path(value) for value in graph.metadata["reward_free_rounds"]]
    manifests = [directory / "manifest.json" for directory in rounds]
    return rounds, [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in manifests
    ]


def _recorded_geometry(graph, rounds, sigma):
    cell_rows, length_rows = [], []
    edge_offset = 0
    minimum = np.floor(graph.nodes[:, :2] / sigma).astype(np.int32).min(0)
    maximum = np.floor(graph.nodes[:, :2] / sigma).astype(np.int32).max(0)
    for directory in rounds:
        for path in part_paths(directory):
            with np.load(path) as data:
                xy = np.asarray(data["states"][:, :, :2], np.float32)
            count = len(xy)
            edge_slice = slice(edge_offset, edge_offset + count)
            expected_start = graph.nodes[graph.source_index[edge_slice], :2]
            expected_end = graph.nodes[graph.endpoint_index[edge_slice], :2]
            if not np.array_equal(xy[:, 0], expected_start):
                raise ValueError(f"recorded passage start mismatch: {path}")
            if not np.array_equal(xy[:, -1], expected_end):
                raise ValueError(f"recorded passage endpoint mismatch: {path}")
            delta = xy[:, 1:] - xy[:, :-1]
            midpoint = (xy[:, 1:] + xy[:, :-1]) * 0.5
            cells = np.floor(midpoint / sigma).astype(np.int32)
            cell_rows.append(cells)
            length_rows.append(
                np.linalg.norm(delta, axis=2).astype(np.float32)
            )
            minimum = np.minimum(minimum, cells.reshape(-1, 2).min(0))
            maximum = np.maximum(maximum, cells.reshape(-1, 2).max(0))
            edge_offset += count
    if edge_offset != graph.edge_count:
        raise ValueError("reward-free passages do not cover every graph edge")
    cells = np.concatenate(cell_rows, axis=0)
    lengths = np.concatenate(length_rows, axis=0)
    return cells, lengths, minimum, maximum


def _linear_cell_index(cells, minimum, shape):
    local = np.asarray(cells, np.int64) - np.asarray(minimum, np.int64)
    if np.any(local < 0) or np.any(local >= np.asarray(shape)[None]):
        raise ValueError("passage sample lies outside integration lattice")
    return (local[..., 0] * int(shape[1]) + local[..., 1]).astype(
        np.uint32
    )


def _bridge_shard(
    graph, targets, weight, sigma, minimum, shape, begin, stop
):
    width = targets.shape[1]
    source = np.repeat(
        graph.nodes[begin:stop, None, :2], width, axis=1
    ).reshape(-1, 2)
    target_index = targets[begin:stop].reshape(-1)
    endpoint = graph.nodes[target_index, :2]
    displacement = endpoint - source
    distance = np.linalg.norm(displacement, axis=1).astype(np.float32)
    local_weight = weight[begin:stop].reshape(-1)
    active = (local_weight > 0.0) & (distance > 0.0)
    active_index = np.flatnonzero(active).astype(np.int32)
    active_distance = distance[active]
    sample_count = np.ceil(active_distance / float(sigma)).astype(np.int32)
    sample_count = np.maximum(sample_count, 1)
    total = int(sample_count.sum())
    repeated = np.repeat(np.arange(len(active_index)), sample_count)
    offsets = np.repeat(
        np.cumsum(sample_count, dtype=np.int64) - sample_count,
        sample_count,
    )
    within = np.arange(total, dtype=np.int64) - offsets
    fraction = (
        (within.astype(np.float32) + 0.5)
        / np.repeat(sample_count, sample_count).astype(np.float32)
    )
    active_source = source[active]
    active_displacement = displacement[active]
    sample = (
        active_source[repeated]
        + fraction[:, None] * active_displacement[repeated]
    )
    cells = np.floor(sample / float(sigma)).astype(np.int32)
    cell_index = _linear_cell_index(cells, minimum, shape)
    return {
        "active_index": active_index,
        "sample_count": sample_count.astype(np.uint16),
        "cell_index": cell_index,
        "distance": active_distance,
    }


def prepare_geometry(root, graph, targets, weight):
    sigma = float(r22.SIGMA)
    artifact = artifact_paths(root)
    rounds, manifests = _round_contract(graph)
    contract = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "graph_sha256": sha256_file(r22.graph_path(root)),
        "reach_landmarks_sha256": sha256_file(
            r22.artifact_paths(root)["landmarks"]
        ),
        "r22_bridges_sha256": sha256_file(
            r22.artifact_paths(root)["bridges"]
        ),
        "reward_free_manifests": manifests,
        "integration_sigma_m": sigma,
        "recorded_passage": "exact_reward_free_primitive_trajectory",
        "bridge_passage": "straight_source_to_recorded_destination_segment",
        "bridge_sample_spacing_maximum_m": sigma,
        "new_planning_scale_introduced": False,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    if artifact["metadata"].exists():
        metadata = json.loads(artifact["metadata"].read_text())
        for key, expected in contract.items():
            if metadata.get(key) != expected:
                raise ValueError(f"saved passage geometry changed {key}")
        return metadata

    artifact["directory"].mkdir(parents=True, exist_ok=True)
    artifact["bridge_shards"].mkdir(parents=True, exist_ok=True)
    cells, lengths, minimum, maximum = _recorded_geometry(
        graph, rounds, sigma
    )
    shape = (maximum - minimum + 1).astype(np.int32)
    recorded_cell_index = _linear_cell_index(cells, minimum, shape)
    _save_npz_atomic(
        artifact["recorded"],
        cell_index=recorded_cell_index,
        step_length=lengths,
        grid_minimum_cell=minimum.astype(np.int32),
        grid_shape=shape,
    )
    shard_rows = []
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        path = artifact["bridge_shards"] / (
            f"nodes_{begin:06d}_{stop:06d}.npz"
        )
        payload = _bridge_shard(
            graph, targets, weight, sigma, minimum, shape, begin, stop
        )
        _save_npz_atomic(path, **payload)
        shard_rows.append({
            "file": path.name,
            "begin": int(begin),
            "stop": int(stop),
            "active_bridges": int(len(payload["active_index"])),
            "passage_samples": int(len(payload["cell_index"])),
            "sha256": sha256_file(path),
        })
        print(
            f"passage geometry nodes={stop}/{graph.node_count} "
            f"samples={len(payload['cell_index'])}",
            flush=True,
        )
    metadata = {
        **contract,
        "grid_minimum_cell": minimum.tolist(),
        "grid_shape": shape.tolist(),
        "grid_cells": int(np.prod(shape)),
        "recorded_edges": int(graph.edge_count),
        "recorded_primitive_segments": int(lengths.size),
        "recorded_passages_sha256": sha256_file(artifact["recorded"]),
        "bridge_shards": shard_rows,
        "bridge_passage_samples": int(sum(
            row["passage_samples"] for row in shard_rows
        )),
    }
    dump(artifact["metadata"], metadata)
    return metadata


class PassageGeometry:
    def __init__(self, root, graph, targets, weight):
        self.root = Path(root)
        self.graph = graph
        self.targets = np.asarray(targets, np.int32)
        self.weight = np.asarray(weight, np.float32)
        self.metadata = prepare_geometry(root, graph, targets, weight)
        artifact = artifact_paths(root)
        with np.load(artifact["recorded"]) as data:
            self.recorded_cell_index = data["cell_index"]
            self.recorded_step_length = data["step_length"]
            self.minimum = data["grid_minimum_cell"].astype(np.int32)
            self.shape = data["grid_shape"].astype(np.int32)
        x = (
            self.minimum[0] + np.arange(self.shape[0], dtype=np.float32) + 0.5
        ) * float(r22.SIGMA)
        y = (
            self.minimum[1] + np.arange(self.shape[1], dtype=np.float32) + 0.5
        ) * float(r22.SIGMA)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        self.grid_xy = np.stack((xx.ravel(), yy.ravel()), axis=1)
        self.bridge_shards = []
        for row in self.metadata["bridge_shards"]:
            path = artifact["bridge_shards"] / row["file"]
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"passage shard hash changed: {path}")
            with np.load(path) as data:
                self.bridge_shards.append({
                    "begin": row["begin"],
                    "stop": row["stop"],
                    "active_index": data["active_index"].astype(np.int32),
                    "sample_count": data["sample_count"].astype(np.int32),
                    "cell_index": data["cell_index"],
                    "distance": data["distance"].astype(np.float32),
                })

    def graph_gains(self, fields):
        grid_field = fields.search(self.grid_xy).astype(np.float32)
        recorded = (
            grid_field[self.recorded_cell_index]
            * self.recorded_step_length
        ).sum(axis=1, dtype=np.float64).astype(np.float32)
        bridge = np.zeros(self.targets.size, np.float32)
        width = self.targets.shape[1]
        for shard in self.bridge_shards:
            count = shard["sample_count"]
            offsets = np.cumsum(count, dtype=np.int64) - count
            sample_field = grid_field[shard["cell_index"]]
            total = np.add.reduceat(sample_field, offsets)
            value = shard["distance"] * total / count
            base = int(shard["begin"]) * width
            bridge[base + shard["active_index"]] = value
        return recorded, bridge.reshape(self.targets.shape)

    def live_segment_gain(self, start_xy, endpoint_xy, fields):
        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        start = np.broadcast_to(
            np.asarray(start_xy, np.float32).reshape(1, 2), endpoint.shape
        )
        displacement = endpoint - start
        distance = np.linalg.norm(displacement, axis=1).astype(np.float32)
        count = np.maximum(
            1, np.ceil(distance / float(r22.SIGMA)).astype(np.int32)
        )
        offsets = np.cumsum(count, dtype=np.int64) - count
        repeated = np.repeat(np.arange(len(endpoint)), count)
        within = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(
            offsets, count
        )
        fraction = (
            (within.astype(np.float32) + 0.5)
            / np.repeat(count, count).astype(np.float32)
        )
        sample = start[repeated] + fraction[:, None] * displacement[repeated]
        total = np.add.reduceat(fields.search(sample), offsets)
        return (distance * total / count).astype(np.float32)


def passage_stitched_values(
    graph, targets, weight, recorded_gain, bridge_gain, horizon=10
):
    values = np.zeros((int(horizon) + 1, graph.node_count), np.float32)
    for remaining in range(1, int(horizon) + 1):
        recorded_q = graph.edge_reliability * (
            recorded_gain + values[remaining - 1, graph.endpoint_index]
        )
        row = np.zeros(graph.node_count, np.float32)
        np.maximum.at(row, graph.source_index, recorded_q)
        bridge_q = weight * (
            bridge_gain + values[remaining - 1, targets]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = row
    return values, value_coverage(values)


class PassageHiddenGoalR22Planner(baseline.HiddenGoalR22Planner):
    method_revision = METHOD_REVISION
    planner_name = "hidden_goal_passage_gain_r26"

    def configure_passage_geometry(self, geometry):
        self.passage_geometry = geometry
        self.acquisition_active = False

    def _set_goal(self, fields):
        if fields.contacted:
            self.acquisition_active = False
            return super()._set_goal(fields)
        self.acquisition_active = True
        self.node_field = fields.search(self.graph.nodes[:, :2]).astype(
            np.float32
        )
        self.recorded_gain, self.bridge_gain = (
            self.passage_geometry.graph_gains(fields)
        )
        self.values, self.coverage = passage_stitched_values(
            self.graph,
            self.targets,
            self.weight,
            self.recorded_gain,
            self.bridge_gain,
            horizon=self.horizon,
        )
        self.goal_key = ("dynamic_acquisition_passage", len(fields.coords))

    def _candidate_value(
        self,
        state,
        commands,
        target_index,
        remaining,
        reliability,
        seed,
        return_members=False,
    ):
        if not self.acquisition_active:
            return super()._candidate_value(
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=return_members,
            )
        (_old_q, connection, _old_member_q, member_connection) = (
            super()._candidate_value(
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=True,
            )
        )
        gain = self.passage_geometry.live_segment_gain(
            state[:2], self.graph.nodes[target_index, :2], self.current_fields
        )
        continuation = gain + self.values[remaining - 1, target_index]
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        q = connection * reliability * continuation
        if return_members:
            member_q = (
                member_connection
                * reliability[:, None]
                * continuation[:, None]
            )
            return q, connection, member_q, member_connection
        return q, connection

    def _greedy_outgoing(self, node_index, remaining):
        if not self.acquisition_active:
            return super()._greedy_outgoing(node_index, remaining)
        node_index = np.asarray(node_index, np.int32).reshape(-1)
        model_target = self.targets[node_index]
        model_q = self.weight[node_index] * (
            self.bridge_gain[node_index]
            + self.values[remaining - 1, model_target]
        )
        model_choice = np.argmax(model_q, axis=1)
        row = np.arange(len(node_index))
        target = model_target[row, model_choice].astype(np.int32)
        command = self.graph.nodes[target, :2].copy()
        reliability = np.ones(len(node_index), np.float32)
        offline_score = model_q[row, model_choice].astype(np.float32)
        kind = np.full(
            len(node_index), "alias_next_model_bridge", dtype=object
        )
        for output_index, node in enumerate(node_index):
            begin = self.outgoing_offsets[node]
            stop = self.outgoing_offsets[node + 1]
            if begin == stop:
                continue
            edges = self.outgoing_order[begin:stop]
            endpoints = self.graph.endpoint_index[edges]
            recorded_q = self.graph.edge_reliability[edges] * (
                self.recorded_gain[edges]
                + self.values[remaining - 1, endpoints]
            )
            local = int(np.argmax(recorded_q))
            if recorded_q[local] <= offline_score[output_index]:
                continue
            edge = int(edges[local])
            command[output_index] = self.graph.commands[edge]
            target[output_index] = self.graph.endpoint_index[edge]
            reliability[output_index] = self.graph.edge_reliability[edge]
            offline_score[output_index] = recorded_q[local]
            kind[output_index] = "alias_next_recorded_edge"
        return command, target, reliability, offline_score, kind

    def choose(self, state, command_index, fields, *args, **kwargs):
        self.current_fields = fields
        command, diagnostic = super().choose(
            state, command_index, fields, *args, **kwargs
        )
        diagnostic.update({
            "acquisition_immediate_readout": (
                "passage_integral" if not fields.contacted else "unchanged_return"
            ),
            "recorded_passage_source": (
                "exact_reward_free_primitive_trajectory"
                if not fields.contacted else None
            ),
            "bridge_passage_source": (
                "straight_source_to_destination_segment"
                if not fields.contacted else None
            ),
        })
        return command, diagnostic


def _load_graph_artifacts(args):
    graph = r22._verify_graph(args)
    artifact = r22.artifact_paths(args.root)
    with np.load(artifact["landmarks"]) as data:
        landmarks = data["landmark_index"]
        targets = data["target_index"]
    with np.load(artifact["bridges"]) as data:
        if not np.array_equal(landmarks, data["landmark_index"]):
            raise ValueError("R22 passage landmarks changed")
        if not np.array_equal(targets, data["target_index"]):
            raise ValueError("R22 passage targets changed")
        weight = data["weight"]
    return graph, targets, weight


def command_prepare(args):
    graph, targets, weight = _load_graph_artifacts(args)
    metadata = prepare_geometry(args.root, graph, targets, weight)
    print(json.dumps({
        "method_revision": metadata["method_revision"],
        "recorded_primitive_segments": metadata["recorded_primitive_segments"],
        "bridge_passage_samples": metadata["bridge_passage_samples"],
        "grid_cells": metadata["grid_cells"],
    }, indent=2))


def _load_runtime(args):
    graph = r22._verify_graph(args)
    planner, hashes = r22.lazy._load_planner(
        args, PassageHiddenGoalR22Planner
    )
    geometry = PassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(float(r22.SIGMA))
    planner.configure_passage_geometry(geometry)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": sha256_file(
            r22.artifact_paths(args.root)["landmarks"]
        ),
        "passage_geometry_sha256": sha256_file(
            artifact_paths(args.root)["metadata"]
        ),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def _task_ids(name):
    return GATE_TASK_IDS if name == "gate2" else ALL_TASK_IDS


def output_directory(root, name):
    return paths(root)["eval"] / name


def _evaluate_task(args, planner, executor, env, task_id):
    task = r22._task(args.root, task_id)
    output = output_directory(args.root, args.output_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
        )
        if actual != expected:
            raise ValueError(f"existing task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved
    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = baseline.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        r22.SIGMA,
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "hidden_goal_r22_passage_integral_acquisition",
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "acquisition_search_sigma_m": float(r22.SIGMA),
        "acquisition_immediate_readout": "passage_integral",
        "return_planner": "unchanged_R22_endpoint_field_readout",
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate_set(args):
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in _task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    task_ids = _task_ids(args.set_name)
    output = output_directory(args.root, args.output_name)
    rows = [
        json.loads((output / f"task_{task_id:03d}.json").read_text())
        for task_id in task_ids
    ]
    contact = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        first = row["first_contact_episode"]
        if first is not None:
            returns.extend(row["successes"][int(first) + 1:])
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "acquisition_only_passage_integral_immediate_readout",
            "return_is_unchanged_r22",
            "one_meter_coverage_cells_are_diagnostic_only",
        ],
        "oracle_inputs": [],
        "set_name": args.set_name,
        "task_ids": list(task_ids),
        "tasks": len(rows),
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contact)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contact])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contact
        ])),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean(successes)),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "acquisition_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
        ])),
        "acquisition_two_cycle_count": int(sum(
            row["acquisition_two_cycle_count"] for row in rows
        )),
        "acquisition_search_sigma_m": float(r22.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count",
        )
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    commands = value.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.set_defaults(func=command_prepare)
    evaluate = commands.add_parser("evaluate-set")
    evaluate.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    evaluate.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    evaluate.add_argument("--output-name", required=True)
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    summarize.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    summarize.add_argument("--output-name", required=True)
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r22.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
