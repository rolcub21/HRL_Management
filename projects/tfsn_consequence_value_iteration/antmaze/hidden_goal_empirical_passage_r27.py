"""R27: empirical bridge passages and exact online evidence candidates.

R26 showed that passage gain can find a hidden goal quickly, but straight
bridge segments hallucinate travel through walls.  R27 replaces each bridge
segment with a locally matched, actually observed reward-free primitive
trajectory.  Matching uses only start xy and realized displacement, both in
the already-defined landing-precision units.  The observed path is translated
to the bridge source; no route, wall map, or task input is used.

Return remains R22, with one consistency repair: positive online evidence is
part of the online experience set and is therefore exposed as an exact live
command candidate.  Its score is computed by the same consequence ensemble,
landing kernel, mobility model, and R22 continuation values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_passage_r26 as r26,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_r22 as baseline,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    ENDPOINT_SAMPLES,
    LANDING_SIGMA,
    STATE_DIM,
    _save_npz_atomic,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)
from scripts.antmaze.current_tfsn.data import part_paths
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


METHOD_REVISION = "antmaze_hidden_goal_empirical_passage_r27"
BASE_METHOD_REVISION = r22.METHOD_REVISION
CLAIM_LEVEL = "consumed_development_hidden_goal_empirical_passage_test"
ALL_TASK_IDS = tuple(range(15))
GATE_TASK_IDS = baseline.GATE_TASK_IDS
DEFAULT_EPISODES = 5
PASSAGE_BINS = 16
NODE_SHARD_SIZE = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(root):
    directory = Path(root) / "model_planning" / "empirical_passage_r27"
    return {
        "directory": directory,
        "metadata": directory / "metadata.json",
        "recorded": directory / "recorded_passages.npz",
        "bridge_shards": directory / "bridge_shards",
    }


def _passage_bins_and_recorded_geometry(graph, rounds, sigma):
    relative_rows, bin_length_rows = [], []
    cell_rows, step_length_rows = [], []
    edge_offset = 0
    minimum = np.floor(graph.nodes[:, :2] / sigma).astype(np.int32).min(0)
    maximum = np.floor(graph.nodes[:, :2] / sigma).astype(np.int32).max(0)
    primitive_groups = np.array_split(np.arange(100), PASSAGE_BINS)
    for directory in rounds:
        for path in part_paths(directory):
            with np.load(path) as data:
                xy = np.asarray(data["states"][:, :, :2], np.float32)
            count = len(xy)
            edge_slice = slice(edge_offset, edge_offset + count)
            if not np.array_equal(
                xy[:, 0], graph.nodes[graph.source_index[edge_slice], :2]
            ):
                raise ValueError(f"empirical passage start mismatch: {path}")
            if not np.array_equal(
                xy[:, -1], graph.nodes[graph.endpoint_index[edge_slice], :2]
            ):
                raise ValueError(f"empirical passage endpoint mismatch: {path}")
            delta = xy[:, 1:] - xy[:, :-1]
            midpoint = (xy[:, 1:] + xy[:, :-1]) * 0.5
            step_length = np.linalg.norm(delta, axis=2).astype(np.float32)
            cells = np.floor(midpoint / sigma).astype(np.int32)
            relative = np.zeros((count, PASSAGE_BINS, 2), np.float32)
            bin_length = np.zeros((count, PASSAGE_BINS), np.float32)
            for output, primitive in enumerate(primitive_groups):
                length = step_length[:, primitive]
                total = length.sum(axis=1)
                weighted = (
                    midpoint[:, primitive]
                    * length[:, :, None]
                ).sum(axis=1)
                fallback = midpoint[:, primitive].mean(axis=1).copy()
                center = np.divide(
                    weighted,
                    total[:, None],
                    out=fallback,
                    where=total[:, None] > 0,
                )
                relative[:, output] = center - xy[:, 0]
                bin_length[:, output] = total
            relative_rows.append(relative)
            bin_length_rows.append(bin_length)
            cell_rows.append(cells)
            step_length_rows.append(step_length)
            minimum = np.minimum(minimum, cells.reshape(-1, 2).min(0))
            maximum = np.maximum(maximum, cells.reshape(-1, 2).max(0))
            edge_offset += count
    if edge_offset != graph.edge_count:
        raise ValueError("empirical passages do not cover every graph edge")
    return (
        np.concatenate(relative_rows),
        np.concatenate(bin_length_rows),
        np.concatenate(cell_rows),
        np.concatenate(step_length_rows),
        minimum,
        maximum,
    )


def _local_passage_tree(graph, sigma):
    start = graph.nodes[graph.source_index, :2]
    displacement = (
        graph.nodes[graph.endpoint_index, :2] - start
    ).astype(np.float32)
    feature = np.concatenate((start, displacement), axis=1) / float(sigma)
    return cKDTree(feature), start, displacement


def _bridge_mapping(tree, graph, targets, begin, stop, sigma):
    source = graph.nodes[begin:stop, :2]
    endpoint = graph.nodes[targets[begin:stop], :2]
    displacement = endpoint - source[:, None]
    query = np.concatenate(
        (
            np.repeat(source[:, None], targets.shape[1], axis=1),
            displacement,
        ),
        axis=2,
    ).reshape(-1, 4) / float(sigma)
    distance, edge = tree.query(query, k=1, workers=-1)
    return edge.astype(np.int32), distance.astype(np.float32)


def _linear_cell_index(cells, minimum, shape):
    local = np.asarray(cells, np.int64) - np.asarray(minimum, np.int64)
    if np.any(local < 0) or np.any(local >= np.asarray(shape)[None]):
        raise ValueError("empirical passage lies outside integration lattice")
    return (local[..., 0] * int(shape[1]) + local[..., 1]).astype(
        np.uint32
    )


def prepare_geometry(root, graph, targets, weight):
    sigma = float(r22.SIGMA)
    artifact = artifact_paths(root)
    rounds, manifests = r26._round_contract(graph)
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
        "bridge_passage": (
            "locally_matched_reward_free_realized_trajectory_translated_to_source"
        ),
        "bridge_match_features": (
            "start_xy_and_realized_displacement_in_landing_precision_units"
        ),
        "passage_bins_computational_only_global": PASSAGE_BINS,
        "new_planning_scale_introduced": False,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    if artifact["metadata"].exists():
        metadata = json.loads(artifact["metadata"].read_text())
        for key, expected in contract.items():
            if metadata.get(key) != expected:
                raise ValueError(f"saved empirical passage changed {key}")
        return metadata

    artifact["directory"].mkdir(parents=True, exist_ok=True)
    artifact["bridge_shards"].mkdir(parents=True, exist_ok=True)
    (
        relative,
        bin_length,
        recorded_cells,
        step_length,
        minimum,
        maximum,
    ) = _passage_bins_and_recorded_geometry(graph, rounds, sigma)

    # Every translated prototype lies within these data-derived bounds.
    relative_cell = np.floor(relative / sigma).astype(np.int32)
    node_cell = np.floor(graph.nodes[:, :2] / sigma).astype(np.int32)
    minimum = np.minimum(
        minimum,
        node_cell.min(0) + relative_cell.reshape(-1, 2).min(0) - 1,
    )
    maximum = np.maximum(
        maximum,
        node_cell.max(0) + relative_cell.reshape(-1, 2).max(0) + 1,
    )
    shape = (maximum - minimum + 1).astype(np.int32)
    recorded_cell_index = _linear_cell_index(
        recorded_cells, minimum, shape
    )
    _save_npz_atomic(
        artifact["recorded"],
        cell_index=recorded_cell_index,
        step_length=step_length,
        relative_bin_xy=relative,
        bin_length=bin_length,
        grid_minimum_cell=minimum,
        grid_shape=shape,
    )

    tree, recorded_start, recorded_displacement = _local_passage_tree(
        graph, sigma
    )
    shard_rows = []
    match_distance_rows = []
    source_error_rows = []
    displacement_error_rows = []
    width = targets.shape[1]
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        edge, match_distance = _bridge_mapping(
            tree, graph, targets, begin, stop, sigma
        )
        active = weight[begin:stop].reshape(-1) > 0
        active_index = np.flatnonzero(active).astype(np.int32)
        active_edge = edge[active]
        source = np.repeat(
            graph.nodes[begin:stop, None, :2], width, axis=1
        ).reshape(-1, 2)[active]
        translated = source[:, None] + relative[active_edge]
        cells = np.floor(translated / sigma).astype(np.int32)
        cell_index = _linear_cell_index(cells, minimum, shape)
        path = artifact["bridge_shards"] / (
            f"nodes_{begin:06d}_{stop:06d}.npz"
        )
        _save_npz_atomic(
            path,
            active_index=active_index,
            passage_edge_index=active_edge,
            cell_index=cell_index,
        )
        desired_displacement = (
            graph.nodes[targets[begin:stop], :2]
            - graph.nodes[begin:stop, None, :2]
        ).reshape(-1, 2)[active]
        source_error_rows.append(np.linalg.norm(
            recorded_start[active_edge] - source, axis=1
        ))
        displacement_error_rows.append(np.linalg.norm(
            recorded_displacement[active_edge] - desired_displacement,
            axis=1,
        ))
        match_distance_rows.append(match_distance[active])
        shard_rows.append({
            "file": path.name,
            "begin": int(begin),
            "stop": int(stop),
            "active_bridges": int(len(active_index)),
            "sha256": sha256_file(path),
        })
        print(
            f"empirical passage nodes={stop}/{graph.node_count} "
            f"active={len(active_index)}",
            flush=True,
        )
    match_distance = np.concatenate(match_distance_rows)
    source_error = np.concatenate(source_error_rows)
    displacement_error = np.concatenate(displacement_error_rows)
    metadata = {
        **contract,
        "grid_minimum_cell": minimum.tolist(),
        "grid_shape": shape.tolist(),
        "grid_cells": int(np.prod(shape)),
        "recorded_edges": int(graph.edge_count),
        "recorded_primitive_segments": int(step_length.size),
        "recorded_passages_sha256": sha256_file(artifact["recorded"]),
        "bridge_shards": shard_rows,
        "active_bridges": int(sum(x["active_bridges"] for x in shard_rows)),
        "match_standardized_distance_median": float(np.median(match_distance)),
        "match_standardized_distance_p90": float(np.quantile(match_distance, .9)),
        "matched_source_error_median_m": float(np.median(source_error)),
        "matched_source_error_p90_m": float(np.quantile(source_error, .9)),
        "matched_displacement_error_median_m": float(
            np.median(displacement_error)
        ),
        "matched_displacement_error_p90_m": float(
            np.quantile(displacement_error, .9)
        ),
    }
    dump(artifact["metadata"], metadata)
    return metadata


class EmpiricalPassageGeometry:
    def __init__(self, root, graph, targets, weight):
        self.graph = graph
        self.targets = np.asarray(targets, np.int32)
        self.weight = np.asarray(weight, np.float32)
        self.metadata = prepare_geometry(root, graph, targets, weight)
        artifact = artifact_paths(root)
        with np.load(artifact["recorded"]) as data:
            self.recorded_cell_index = data["cell_index"]
            self.recorded_step_length = data["step_length"]
            self.relative = data["relative_bin_xy"].astype(np.float32)
            self.bin_length = data["bin_length"].astype(np.float32)
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
        self.tree, _start, _displacement = _local_passage_tree(
            graph, r22.SIGMA
        )
        self.bridge_shards = []
        for row in self.metadata["bridge_shards"]:
            path = artifact["bridge_shards"] / row["file"]
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"empirical passage shard changed: {path}")
            with np.load(path) as data:
                self.bridge_shards.append({
                    "begin": row["begin"],
                    "stop": row["stop"],
                    "active_index": data["active_index"].astype(np.int32),
                    "passage_edge_index": data[
                        "passage_edge_index"
                    ].astype(np.int32),
                    "cell_index": data["cell_index"],
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
            edge = shard["passage_edge_index"]
            value = (
                grid_field[shard["cell_index"]]
                * self.bin_length[edge]
            ).sum(axis=1, dtype=np.float64).astype(np.float32)
            base = int(shard["begin"]) * width
            bridge[base + shard["active_index"]] = value
        return recorded, bridge.reshape(self.targets.shape)

    def live_segment_gain(self, start_xy, endpoint_xy, fields):
        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        start = np.broadcast_to(
            np.asarray(start_xy, np.float32).reshape(1, 2), endpoint.shape
        )
        displacement = endpoint - start
        query = np.concatenate((start, displacement), axis=1) / float(
            r22.SIGMA
        )
        _distance, edge = self.tree.query(query, k=1, workers=-1)
        edge = np.asarray(edge, np.int32)
        sample = start[:, None] + self.relative[edge]
        field = fields.search(sample.reshape(-1, 2)).reshape(
            len(endpoint), PASSAGE_BINS
        )
        return (field * self.bin_length[edge]).sum(axis=1).astype(np.float32)


class EmpiricalPassagePlanner(r26.PassageHiddenGoalR22Planner):
    method_revision = METHOD_REVISION
    planner_name = "hidden_goal_empirical_passage_r27"

    def configure_empirical_passage(self, geometry):
        self.configure_passage_geometry(geometry)
        self.node_xy_tree = cKDTree(self.graph.nodes[:, :2])

    def _positive_evidence_value(self, state, commands, fields, remaining, seed):
        source = np.repeat(state[None], len(commands), axis=0)
        inputs = consequence_inputs(source, commands)
        samples = ENDPOINT_SAMPLES // len(self.consequence.members)
        nearest = self.node_xy_tree.query(commands, k=1, workers=-1)[1]
        continuation = (
            fields.reward_rate(commands)
            + self.values[remaining - 1, nearest]
        )
        member_q, member_connection = [], []
        for member in range(len(self.consequence.members)):
            rng = np.random.default_rng(seed + 1_000_003 * member)
            delta = self.consequence.sample_member(
                inputs, member, samples, rng
            )
            endpoint = source[:, None] + delta
            spatial = np.exp(
                -np.square(endpoint[..., :2] - commands[:, None]).sum(axis=2)
                / (2.0 * LANDING_SIGMA**2)
            )
            mobility = self.mobility.predict(
                endpoint.reshape(-1, STATE_DIM)
            ).reshape(len(commands), samples)
            connection = (spatial * mobility).mean(axis=1)
            member_connection.append(connection)
            member_q.append(connection * continuation)
        return (
            np.stack(member_q, axis=1).min(axis=1),
            np.stack(member_connection, axis=1).min(axis=1),
        )

    def choose(self, state, command_index, fields, *args, **kwargs):
        command, diagnostic = super().choose(
            state, command_index, fields, *args, **kwargs
        )
        diagnostic.update({
            "acquisition_immediate_readout": (
                "empirical_reward_free_passage_integral"
                if not fields.contacted else "unchanged_return_field"
            ),
            "bridge_passage_source": (
                "locally_matched_reward_free_realized_trajectory"
                if not fields.contacted else None
            ),
            "bridge_passage_match_features": (
                "start_xy_and_realized_displacement"
                if not fields.contacted else None
            ),
            "bridge_passage_bins_computational_only_global": (
                PASSAGE_BINS if not fields.contacted else None
            ),
        })
        diagnostic["return_online_positive_evidence_candidates"] = 0
        if not fields.contacted:
            return command, diagnostic
        reward = np.asarray(fields.rewards, np.float32) > 0
        evidence = np.asarray(fields.coords, np.float32)[reward]
        evidence = np.unique(evidence, axis=0)
        diagnostic["return_online_positive_evidence_candidates"] = int(
            len(evidence)
        )
        if not len(evidence):
            return command, diagnostic
        remaining = self.horizon - int(command_index)
        q, connection = self._positive_evidence_value(
            np.asarray(state, np.float32).reshape(STATE_DIM),
            evidence,
            fields,
            remaining,
            self.seed + command_index * 10_007 + 389,
        )
        winner = int(np.argmax(q))
        diagnostic["best_positive_evidence_score"] = float(q[winner])
        diagnostic["best_positive_evidence_connection_weight"] = float(
            connection[winner]
        )
        if q[winner] <= float(diagnostic["selected_score"]):
            return command, diagnostic
        selected = evidence[winner].copy()
        diagnostic.update({
            "selected_score": float(q[winner]),
            "selected_kind": "online_positive_evidence",
            "selected_connection_weight": float(connection[winner]),
            "selected_command_distance": float(np.linalg.norm(
                selected - np.asarray(state, np.float32)[:2]
            )),
            "selected_alias_association": 0.0,
            "selected_execution_partition_weight": 1.0,
            "selected_alias_offline_score": 0.0,
        })
        return selected, diagnostic


def _load_graph_artifacts(args):
    return r26._load_graph_artifacts(args)


def command_prepare(args):
    graph, targets, weight = _load_graph_artifacts(args)
    metadata = prepare_geometry(args.root, graph, targets, weight)
    print(json.dumps({
        key: metadata[key]
        for key in (
            "method_revision", "recorded_primitive_segments",
            "active_bridges", "grid_cells",
            "matched_source_error_median_m",
            "matched_displacement_error_median_m",
        )
    }, indent=2))


def _load_runtime(args):
    graph = r22._verify_graph(args)
    planner, hashes = r22.lazy._load_planner(args, EmpiricalPassagePlanner)
    geometry = EmpiricalPassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(float(r22.SIGMA))
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": sha256_file(
            r22.artifact_paths(args.root)["landmarks"]
        ),
        "empirical_passage_geometry_sha256": sha256_file(
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
            int(args.episodes), float(r22.SIGMA), float(r22.SIGMA),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("acquisition_search_sigma_m"),
            saved.get("return_field_sigma_m"),
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
        "evaluation_arm": "hidden_goal_empirical_passage_and_evidence_candidates",
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "acquisition_search_sigma_m": float(r22.SIGMA),
        "return_field_sigma_m": float(r22.SIGMA),
        "return_field_sigma_estimator": r22.ESTIMATOR,
        "acquisition_immediate_readout": "empirical_passage_integral",
        "return_planner": "R22_plus_exact_online_positive_evidence_candidates",
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
            "acquisition_uses_empirical_passage_integral",
            "return_adds_exact_online_positive_evidence_candidates",
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
        "return_field_sigma_m": float(r22.SIGMA),
        "return_field_sigma_estimator": r22.ESTIMATOR,
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
