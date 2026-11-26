"""R22: R14 with bridge reach derived from reward-free executor outcomes.

R21 correctly used the deployed factorized model's landing precision for
spatial identity, but also used that precision to construct a sigma-net and
then selected the 128 nearest landmarks.  This made the physical reach of the
graph depend on landmark density.  R22 keeps sigma for the goal field,
landing kernels, and alias association only.  Bridge destinations instead
come from a fixed-size, global computational budget of representatives of the
empirical one-command displacement distribution.

Every representative is an observed reward-free displacement.  The observed
displacements are divided into equal-probability radial strata; within each
stratum the observation closest to a deterministic low-discrepancy direction
is retained.  From any state, those displacement representatives are
translated to that state and snapped to the nearest *recorded endpoint*.
The consequence model, not a distance mask, assigns feasibility and mobility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze import (
    factorized_sigma_r14 as r21,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    lazy_alias_advance as lazy,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    mobility_stitched_planning as mobility,
)
from projects.tfsn_consequence_value_iteration.antmaze.alias_partition_advance import (
    PartitionedAliasAdvancePlanner,
    _task,
)
from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    output_paths as expanded_output_paths,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    factorized_checkpoint,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
    sha256_file,
)
from projects.tfsn_consequence_value_iteration.antmaze.reach_bridge_candidates import (
    EmpiricalReachQuery,
    representative_reach_offsets,
)
from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import CANDIDATES, EPISODE_STEPS
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


SIGMA = r21.SIGMA
ESTIMATOR = r21.ESTIMATOR
BANDWIDTH_REPORT = r21.BANDWIDTH_REPORT
METHOD_REVISION = "antmaze_r14_empirical_reach_bridges_r22"
CLAIM_LEVEL = "expanded_consumed_development_reach_bridge_test"
ALL_TASK_IDS = tuple(range(15))
BRIDGE_CANDIDATES = int(CANDIDATES)


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def empirical_displacements(graph):
    """Exact realized xy displacements of reward-free recorded commands."""
    return (
        graph.nodes[graph.endpoint_index, :2]
        - graph.nodes[graph.source_index, :2]
    ).astype(np.float32)


def graph_path(root):
    return r21.graph_path(root)


def artifact_paths(root):
    directory = (
        Path(root)
        / "model_planning"
        / "mobility_stitch_r22_empirical_reach"
    )
    return {
        "directory": directory,
        "landmarks": directory / "landmarks.npz",
        "landmark_meta": directory / "landmarks.json",
        "mobility_shards": directory / "mobility_shards",
        "mobility_labels": directory / "mobility_labels.npz",
        "mobility": directory / "mobility.pt",
        "mobility_meta": directory / "mobility.json",
        "bridge_shards": directory / "bridge_shards",
        "bridges": directory / "bridges.npz",
        "bridge_meta": directory / "bridges.json",
    }


def _reach_contract(graph, graph_hash, consequence_hash, offsets):
    displacement = empirical_displacements(graph)
    radius = np.linalg.norm(displacement, axis=1)
    return {
        "method_revision": METHOD_REVISION,
        "base_graph_sha256": graph_hash,
        "factorized_consequence_sha256": consequence_hash,
        "candidate_rule": (
            "equal_mass_empirical_realized_displacement_strata_with_"
            "deterministic_low_discrepancy_direction"
        ),
        "destination_support": "all_exact_recorded_command_endpoints",
        "bridge_candidate_budget_global": int(BRIDGE_CANDIDATES),
        "reach_offsets_sha256": _hash_arrays(offsets),
        "empirical_commands": int(len(displacement)),
        "empirical_displacement_median_m": float(np.median(radius)),
        "empirical_displacement_p90_m": float(np.quantile(radius, 0.9)),
        "empirical_displacement_maximum_m": float(radius.max()),
        "landing_sigma_m": float(SIGMA),
        "landing_sigma_used_to_select_bridge_candidates": False,
    }


def _prepare_reach_landmarks(args, graph, consequence_path):
    artifact = artifact_paths(args.root)
    artifact["directory"].mkdir(parents=True, exist_ok=True)
    graph_hash = sha256_file(args.graph)
    consequence_hash = sha256_file(consequence_path)
    offsets = representative_reach_offsets(
        empirical_displacements(graph), count=BRIDGE_CANDIDATES
    )
    contract = _reach_contract(graph, graph_hash, consequence_hash, offsets)

    if artifact["landmarks"].exists() and artifact["landmark_meta"].exists():
        metadata = json.loads(artifact["landmark_meta"].read_text())
        for key, expected in contract.items():
            if metadata.get(key) != expected:
                raise ValueError(f"saved reach landmarks have another {key}")
        with np.load(artifact["landmarks"]) as data:
            landmark_index = data["landmark_index"]
            targets = data["target_index"]
            distance = data["target_distance"]
            saved_offsets = data["reach_offsets"]
        if not np.array_equal(offsets, saved_offsets):
            raise ValueError("saved empirical reach representatives changed")
        return artifact, landmark_index, targets, distance, metadata

    # Only realized command endpoints are legal destination landmarks.
    landmark_index = np.unique(graph.endpoint_index).astype(np.int32)
    query = EmpiricalReachQuery(graph.nodes[landmark_index, :2], offsets)
    distance, local = query.query(graph.nodes[:, :2], k=BRIDGE_CANDIDATES)
    targets = landmark_index[local].astype(np.int32)
    endpoint_tree = cKDTree(graph.nodes[landmark_index, :2])
    cover_distance = endpoint_tree.query(
        graph.nodes[:, :2], k=1, workers=-1
    )[0]
    unique_counts = np.asarray(
        [len(np.unique(row)) for row in targets], np.int32
    )
    mobility._save_npz_atomic(
        artifact["landmarks"],
        landmark_index=landmark_index,
        target_index=targets,
        target_distance=distance,
        reach_offsets=offsets,
    )
    metadata = {
        **contract,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "oracle_inputs": [],
        "landmark_rule": "translated_empirical_reach_to_recorded_endpoint",
        "landmarks_are_exact_recorded_endpoints": True,
        "recorded_nodes_deleted": False,
        "recorded_edges_deleted": False,
        "global_landmark_neighbors": int(targets.shape[1]),
        "candidate_landmarks_per_state": int(targets.shape[1]),
        "nodes": int(graph.node_count),
        "landmarks": int(len(landmark_index)),
        "unique_candidates_per_state_median": float(np.median(unique_counts)),
        "unique_candidates_per_state_minimum": int(unique_counts.min()),
        "maximum_support_to_recorded_endpoint_distance_m": float(
            cover_distance.max()
        ),
        "median_neighbor_radius_m": float(np.median(distance)),
        "p90_neighbor_radius_m": float(np.quantile(distance, 0.9)),
        "maximum_neighbor_radius_m": float(distance.max()),
        "base_graph": str(Path(args.graph).resolve()),
        "factorized_consequence": str(Path(consequence_path).resolve()),
    }
    dump(artifact["landmark_meta"], metadata)
    return artifact, landmark_index, targets, distance, metadata


# Patch the generic artifact builder and the already-imported R14 loader only
# inside this process.  Frozen R21 source and artifacts are not modified.
mobility._artifact_paths = artifact_paths
mobility._default_graph = graph_path
mobility._prepare_landmarks = _prepare_reach_landmarks
mobility.METHOD_REVISION = METHOD_REVISION
mobility.CLAIM_LEVEL = CLAIM_LEVEL
lazy._prepare_landmarks = _prepare_reach_landmarks


class ReachBasedR14Planner(PartitionedAliasAdvancePlanner):
    method_revision = METHOD_REVISION
    planner_name = "r14_empirical_reach_bridges"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        offsets = representative_reach_offsets(
            empirical_displacements(self.graph), count=BRIDGE_CANDIDATES
        )
        self.landmark_tree = EmpiricalReachQuery(
            self.graph.nodes[self.landmark_index, :2], offsets
        )


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "r14_empirical_reach_bridges_r22"
    )


def _verify_graph(args):
    graph = RecordedTransitionGraph.load(args.graph)
    expected = {
        "field_sigma_m": float(SIGMA),
        "deployed_consequence_sha256": sha256_file(
            factorized_checkpoint(args.root)
        ),
        "bandwidth_report_sha256": r21._sha256(BANDWIDTH_REPORT),
    }
    for key, value in expected.items():
        if graph.metadata.get(key) != value:
            raise ValueError(f"R22 base graph has another {key}")
    return graph


def command_prepare(args):
    _verify_graph(args)
    mobility.command_prepare_mobility(args)
    mobility.command_prepare_bridges(args)


def _load_runtime(args):
    _verify_graph(args)
    planner, hashes = lazy._load_planner(args, ReachBasedR14Planner)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": sha256_file(
            artifact_paths(args.root)["landmarks"]
        ),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    return planner, executor, env


def _evaluate_task(args, planner, executor, env, task_id):
    task = _task(args.root, task_id)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (task["goal"], task["reset_seed"], METHOD_REVISION, SIGMA)
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("field_sigma_m"),
        )
        if actual != expected:
            raise ValueError(f"existing task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = evaluate_supplied_goal_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        SIGMA,
        max_steps=EPISODE_STEPS,
        log_path=output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "r14_empirical_reach_bridges",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "field_sigma_m": SIGMA,
        "field_sigma_estimator": ESTIMATOR,
        "bridge_reach_source": "empirical_reward_free_realized_displacements",
        "bridge_candidate_budget_global": BRIDGE_CANDIDATES,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_task(args):
    planner, executor, env = _load_runtime(args)
    try:
        _evaluate_task(args, planner, executor, env, args.task_id)
    finally:
        env.close()


def command_evaluate_all(args):
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in ALL_TASK_IDS:
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    output = output_directory(args.root)
    rows = []
    for task_id in ALL_TASK_IDS:
        path = output / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    old_summary = json.loads((
        expanded_output_paths(args.root)["directory"]
        / "r10_partitioned_alias_advance_r14_long3"
        / "summary_all15.json"
    ).read_text())
    r21_summary = json.loads((
        expanded_output_paths(args.root)["directory"]
        / "r14_factorized_bandwidth_r21"
        / "summary.json"
    ).read_text())
    old = {int(row["task_id"]): row for row in old_summary["rows"]}
    previous = {int(row["task_id"]): row for row in r21_summary["rows"]}
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "landing_sigma_from_deployed_factorized_consequence",
            "bridge_reach_from_reward_free_realized_displacements",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "field_sigma_m": SIGMA,
        "field_sigma_estimator": ESTIMATOR,
        "bridge_candidate_budget_global": BRIDGE_CANDIDATES,
        "old_r14_successes": int(old_summary["successes"]),
        "r21_sigma_consistent_successes": int(r21_summary["successes"]),
        "paired_changes_vs_old_r14": {
            "gained_task_ids": [
                int(row["task_id"]) for row in rows
                if row["success"] and not old[int(row["task_id"])]["success"]
            ],
            "lost_task_ids": [
                int(row["task_id"]) for row in rows
                if not row["success"] and old[int(row["task_id"])]["success"]
            ],
        },
        "paired_changes_vs_r21": {
            "gained_task_ids": [
                int(row["task_id"]) for row in rows
                if row["success"] and not previous[int(row["task_id"])]["success"]
            ],
            "lost_task_ids": [
                int(row["task_id"]) for row in rows
                if not row["success"] and previous[int(row["task_id"])]["success"]
            ],
        },
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "successes", "tasks", "success_rate", "field_sigma_m",
            "old_r14_successes", "r21_sigma_consistent_successes",
            "paired_changes_vs_old_r14", "paired_changes_vs_r21",
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
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    task = subparsers.add_parser("task")
    task.add_argument("--task-id", type=int, choices=ALL_TASK_IDS, required=True)
    task.set_defaults(func=command_task)
    subparsers.add_parser("evaluate-all15").set_defaults(func=command_evaluate_all)
    subparsers.add_parser("summarize").set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
