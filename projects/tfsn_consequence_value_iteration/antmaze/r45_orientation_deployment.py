"""Deploy R43 posture uncertainty in the frozen R39 two-goal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    lazy_alias_advance as lazy,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    mobility_stitched_planning as mobility,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r39_live_posterior_passage as r39,
)
from projects.tfsn_consequence_value_iteration.antmaze.orientation_factorized_consequence_r43 import (
    OrientationFactorizedConsequenceEnsemble,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
METHOD_REFERENCE = HERE / "R45_ORIENTATION_DEPLOYMENT_REFERENCE.md"
PROTOCOL = HERE / "r45_orientation_deployment_protocol.json"
METHOD_REVISION = "antmaze_r43_orientation_deployment_r45"
RESULT_DIRECTORY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "orientation_factorized_deployment_r45"
)
R43_DIRECTORY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "orientation_factorized_consequence_r43"
)
R43_CHECKPOINT = R43_DIRECTORY / "consequence.pt"
R43_SUMMARY = R43_DIRECTORY / "summary.json"
R31_PLANNER = r31.RESULT_DIRECTORY / "planner"


def load_contract():
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": (
            "consumed_development_orientation_consequence_component_gate_only"
        ),
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_preparation": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "gate_task_ids": [3, 5],
        "episodes_per_task": 5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R45 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R45 method reference changed")
    return protocol


PROTOCOL_VALUE = load_contract()


def r43_checkpoint(_root=None):
    return R43_CHECKPOINT


def artifact_paths(_root=None):
    """R45 owns bridges; all explicitly frozen artifacts remain in R31."""
    directory = RESULT_DIRECTORY / "planner"
    return {
        "directory": directory,
        "landmarks": R31_PLANNER / "landmarks.npz",
        "landmark_meta": R31_PLANNER / "landmarks.json",
        "mobility_shards": R31_PLANNER / "mobility_shards",
        "mobility_labels": R31_PLANNER / "mobility_labels.npz",
        "mobility": R31_PLANNER / "mobility.pt",
        "mobility_meta": R31_PLANNER / "mobility.json",
        "bridge_shards": directory / "bridge_shards",
        "bridges": directory / "bridges.npz",
        "bridge_meta": directory / "bridges.json",
    }


def verify_r43():
    if not R43_CHECKPOINT.exists() or not R43_SUMMARY.exists():
        raise FileNotFoundError("full R43 checkpoint and summary are required")
    summary = json.loads(R43_SUMMARY.read_text())
    expected = {
        "method_revision": "antmaze_orientation_factorized_consequence_r43",
        "profile": "full",
        "hidden_goals_accessed": False,
        "task_reward_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"R43 summary has another {key}")
    if summary.get("checkpoint_sha256") != r31.sha256(R43_CHECKPOINT):
        raise ValueError("R43 checkpoint no longer matches its summary")
    return summary


def _frozen_landmarks(args, graph, consequence_path):
    """Load, rather than rebuild, the exact R31 landmark destinations."""
    del consequence_path
    artifact = artifact_paths(args.root)
    original = r31.artifact_paths(args.root)
    for key in ("landmarks", "landmark_meta", "mobility", "mobility_meta"):
        if artifact[key].resolve() != original[key].resolve():
            raise ValueError(f"R45 no longer reuses frozen R31 {key}")
        if not artifact[key].exists():
            raise FileNotFoundError(artifact[key])
    metadata = json.loads(artifact["landmark_meta"].read_text())
    if metadata.get("base_graph_sha256") != r31.recorded.sha256_file(args.graph):
        raise ValueError("frozen landmarks refer to another graph")
    with np.load(artifact["landmarks"]) as data:
        landmark_index = data["landmark_index"]
        targets = data["target_index"]
        distance = data["target_distance"]
    if targets.shape[0] != graph.node_count:
        raise ValueError("frozen landmark candidates no longer match graph")
    return artifact, landmark_index, targets, distance, metadata


def bind_runtime():
    """Redirect only full-endpoint consequence calls and bridge outputs."""
    verify_r43()
    mobility.FactorizedConsequenceEnsemble = (
        OrientationFactorizedConsequenceEnsemble
    )
    mobility.factorized_checkpoint = r43_checkpoint
    mobility._artifact_paths = artifact_paths
    mobility._default_graph = r31.graph_path
    mobility._prepare_landmarks = _frozen_landmarks
    mobility.METHOD_REVISION = METHOD_REVISION
    mobility.CLAIM_LEVEL = PROTOCOL_VALUE["claim_level"]

    # lazy_alias_advance imported this function by value.
    lazy._prepare_landmarks = _frozen_landmarks


class OrientationAwareLivePosteriorPlanner(r39.LivePosteriorPassagePlanner):
    method_revision = METHOD_REVISION
    planner_name = "live_posterior_passage_with_R43_orientation_risk"


def _frozen_passage_geometry(root, graph, targets, weight):
    """Load R31 geometry while the R45 bridge artifact binding is active."""
    saved = r31.r22.artifact_paths
    try:
        r31.r22.artifact_paths = r31.artifact_paths
        return r31.r28.BoundedPassageGeometry(
            root, graph, targets, weight
        )
    finally:
        r31.r22.artifact_paths = saved


def _load_runtime(args):
    bind_runtime()
    graph = r31.verify_graph(args.graph)
    planner, hashes = lazy._load_planner(
        args, OrientationAwareLivePosteriorPlanner
    )
    geometry = _frozen_passage_geometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(r31.SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.configure_posterior_support(
        geometry.grid_xy[geometry.reachable_cell_index]
    )
    planner.artifact_hashes = {
        **hashes,
        "r43_summary_sha256": r31.sha256(R43_SUMMARY),
        "frozen_r31_landmarks_sha256": r31.recorded.sha256_file(
            r31.artifact_paths()["landmarks"]
        ),
        "frozen_r31_mobility_sha256": r31.recorded.sha256_file(
            r31.artifact_paths()["mobility"]
        ),
        "frozen_r31_empirical_passage_sha256": r31.recorded.sha256_file(
            r31.passage_paths()["metadata"]
        ),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def evaluation_directory():
    return RESULT_DIRECTORY / "evaluation_gate2"


def task_result_path(task_id):
    return evaluation_directory() / f"task_{int(task_id):03d}.json"


def _evaluate_task(args, planner, executor, env, task_id):
    task = r31.r22._task(args.root, task_id)
    output = evaluation_directory()
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), r31.sha256(R43_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(f"existing R45 task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = r39.r37.evaluate_with_exact_fields(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        output / f"task_{int(task_id):03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "R39_with_R43_orientation_factorized_consequence",
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "posterior_likelihood": r39.PROTOCOL_VALUE["posterior_likelihood"],
        "acquisition_value": r39.PROTOCOL_VALUE["acquisition_value"],
        "posterior_support_cells": int(len(planner.posterior_support_xy)),
        "landing_sigma_m": float(r31.SIGMA),
        "landing_sigma_estimator": r31.SIGMA_ESTIMATOR,
        "acquisition_detection_radius_m": r39.r35.DETECTION_FOOTPRINT_M,
        "return_field_sigma_m": float(r31.SIGMA),
        "return_planner": "R39_return_with_R43_consequence_and_R45_bridges",
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_prepare_bridges(args):
    bind_runtime()
    r31.verify_graph(args.graph)
    mobility.command_prepare_bridges(args)


def command_evaluate(args):
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes_per_task"]):
        raise ValueError("R45 gate requires five episodes")
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in PROTOCOL_VALUE["gate_task_ids"]:
            _evaluate_task(args, planner, executor, env, int(task_id))
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    rows = [
        json.loads(task_result_path(task_id).read_text())
        for task_id in PROTOCOL_VALUE["gate_task_ids"]
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
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["gate_task_ids"],
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
        "factorized_consequence_sha256": r31.sha256(R43_CHECKPOINT),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    output = evaluation_directory()
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


def command_audit(_args):
    summary = verify_r43()
    original = r31.artifact_paths()
    report = {
        "method_revision": METHOD_REVISION,
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "r43_checkpoint_sha256": r31.sha256(R43_CHECKPOINT),
        "r43_held_flip_observed_rate": summary[
            "held_out_flip_calibration"
        ]["observed_rate"],
        "r43_held_flip_predicted_rate": summary[
            "held_out_flip_calibration"
        ]["predicted_rate"],
        "frozen_graph_sha256": r31.recorded.sha256_file(r31.graph_path()),
        "frozen_landmarks_sha256": r31.recorded.sha256_file(
            original["landmarks"]
        ),
        "frozen_mobility_sha256": r31.recorded.sha256_file(
            original["mobility"]
        ),
        "r45_bridge_output": str(artifact_paths()["bridges"]),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "confirmation_panel_accessed": False,
    }
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("prepare-bridges").set_defaults(
        func=command_prepare_bridges
    )
    evaluate = commands.add_parser("evaluate-gate2")
    evaluate.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    evaluate.set_defaults(func=command_evaluate)
    summarize = commands.add_parser("summarize")
    summarize.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r31.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
