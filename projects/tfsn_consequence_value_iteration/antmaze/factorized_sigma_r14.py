"""R21: R14 rebuilt at the deployed factorized model's bandwidth.

This is not a planner change.  It reruns the frozen R14 decision principle
with every sigma-dependent artifact derived from the deployed factorized R8
consequence model's held-out landing precision.  The old 0.2504 m graph,
landmarks, mobility labels, bridges, fields, and outputs remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BANDWIDTH_REPORT = (
    PROJECT / "reports" / "antmaze_factorized_r8_bandwidth.json"
)
_BANDWIDTH = json.loads(BANDWIDTH_REPORT.read_text())
SIGMA = float(_BANDWIDTH["endpoint_dispersion_m"])
ESTIMATOR = _BANDWIDTH["endpoint_dispersion_estimator"]
if not np.isfinite(SIGMA) or SIGMA <= 0:
    raise ValueError("factorized R8 bandwidth must be finite and positive")

# All downstream modules bind LANDING_SIGMA at import time.  Set the measured
# deployment value before importing them, without modifying the frozen SG
# implementation or the prior R14 artifacts.
from scripts.antmaze.current_tfsn import constants as _constants

_constants.LANDING_SIGMA = SIGMA
_constants.BOOTSTRAP_SIGMA = SIGMA

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import EPISODE_STEPS, STATE_DIM
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths

from projects.tfsn_consequence_value_iteration.antmaze import (
    mobility_stitched_planning as mobility,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    factorized_checkpoint,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    build_recorded_graph,
    sha256_file,
)


METHOD_REVISION = "antmaze_r14_factorized_bandwidth_r21"
CLAIM_LEVEL = "expanded_consumed_development_sigma_consistency_rerun"
ALL_TASK_IDS = tuple(range(15))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if _BANDWIDTH["deployed_checkpoint_sha256"] != _sha256(
    Path(_BANDWIDTH["deployed_checkpoint"])
):
    raise ValueError("bandwidth report no longer matches its R8 checkpoint")


def graph_path(root):
    return (
        Path(root)
        / "model_planning"
        / "round2_recorded_transition_graph_r21_factorized_sigma.npz"
    )


def artifact_paths(root):
    directory = (
        Path(root)
        / "model_planning"
        / "mobility_stitch_r21_factorized_sigma"
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


# Mobility preparation and LazyAliasAdvancePlanner resolve these functions at
# runtime.  Redirect them before importing R14 so no R10 artifact is reused.
mobility._artifact_paths = artifact_paths
mobility._default_graph = graph_path
mobility.METHOD_REVISION = METHOD_REVISION
mobility.CLAIM_LEVEL = CLAIM_LEVEL

from projects.tfsn_consequence_value_iteration.antmaze.alias_partition_advance import (
    PartitionedAliasAdvancePlanner,
    _task,
)
from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    output_paths as expanded_output_paths,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    _load_planner,
)


class FactorizedSigmaR14Planner(PartitionedAliasAdvancePlanner):
    method_revision = METHOD_REVISION
    planner_name = "r14_factorized_bandwidth"


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "r14_factorized_bandwidth_r21"
    )


def command_prepare_graph(args):
    checkpoint = factorized_checkpoint(args.root)
    if Path(args.graph).exists() and Path(args.graph).with_suffix(".json").exists():
        metadata = json.loads(Path(args.graph).with_suffix(".json").read_text())
        expected = (
            METHOD_REVISION,
            SIGMA,
            sha256_file(checkpoint),
            _sha256(BANDWIDTH_REPORT),
        )
        actual = (
            metadata.get("method_revision"),
            metadata.get("field_sigma_m"),
            metadata.get("deployed_consequence_sha256"),
            metadata.get("bandwidth_report_sha256"),
        )
        if actual != expected:
            raise ValueError("existing R21 graph has another contract")
        print(json.dumps({
            "artifact": str(Path(args.graph).resolve()),
            "artifact_sha256": sha256_file(args.graph),
            "field_sigma_m": SIGMA,
            "resumed": True,
        }, indent=2))
        return
    model = FactorizedConsequenceEnsemble(checkpoint, args.device)
    round_directories = [paths(args.root, index)["round"] for index in range(3)]
    starts, commands, endpoints = command_records(round_directories)
    state_scale = model.input_std[:STATE_DIM].copy()
    state_scale[:2] = SIGMA
    graph = build_recorded_graph(starts, commands, endpoints, state_scale)
    graph.metadata.update({
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "deployed_consequence": str(checkpoint.resolve()),
        "deployed_consequence_sha256": sha256_file(checkpoint),
        "bandwidth_report": str(BANDWIDTH_REPORT),
        "bandwidth_report_sha256": _sha256(BANDWIDTH_REPORT),
        "field_sigma_m": SIGMA,
        "field_sigma_estimator": ESTIMATOR,
        "reward_free_rounds": [str(path.resolve()) for path in round_directories],
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    })
    graph.save(args.graph)
    print(json.dumps({
        "artifact": str(args.graph.resolve()),
        "artifact_sha256": sha256_file(args.graph),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "field_sigma_m": SIGMA,
        "field_sigma_estimator": ESTIMATOR,
    }, indent=2))


def command_prepare_mobility(args):
    mobility.command_prepare_mobility(args)


def command_prepare_bridges(args):
    mobility.command_prepare_bridges(args)


def command_prepare(args):
    command_prepare_graph(args)
    command_prepare_mobility(args)
    command_prepare_bridges(args)


def _evaluate_task(args, planner, executor, env, task_id):
    task = _task(args.root, task_id)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION, SIGMA
        )
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
        "evaluation_arm": "r14_factorized_bandwidth",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "field_sigma_m": SIGMA,
        "field_sigma_estimator": ESTIMATOR,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def _load_runtime(args):
    planner, hashes = _load_planner(args, FactorizedSigmaR14Planner)
    planner.artifact_hashes = hashes
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    return planner, executor, env


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
    old = {int(row["task_id"]): row for row in old_summary["rows"]}
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "sigma_recomputed_from_deployed_factorized_consequence",
            "all_sigma_dependent_artifacts_rebuilt",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "field_sigma_m": SIGMA,
        "field_sigma_estimator": ESTIMATOR,
        "bandwidth_report": str(BANDWIDTH_REPORT),
        "bandwidth_report_sha256": _sha256(BANDWIDTH_REPORT),
        "old_r14_sigma_m": float(_BANDWIDTH["current_runtime_sigma_m"]),
        "old_r14_successes": int(old_summary["successes"]),
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
            "old_r14_successes", "paired_changes_vs_old_r14",
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
    subparsers.add_parser("prepare-graph").set_defaults(func=command_prepare_graph)
    subparsers.add_parser("prepare-mobility").set_defaults(func=command_prepare_mobility)
    subparsers.add_parser("prepare-bridges").set_defaults(func=command_prepare_bridges)
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
