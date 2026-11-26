"""Deploy R30 in the unchanged R28 hidden-goal development gate.

The frozen R28 modules bind several model-derived quantities at import time.
This module redirects those bindings into an isolated R31 artifact tree; it
does not overwrite R28/R22 source or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn import constants as ant_constants
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths

from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_bounded_consumption_r28 as r28,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_empirical_passage_r27 as r27,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    lazy_alias_advance as lazy,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    mobility_stitched_planning as mobility,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    recorded_transition_planning as recorded,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
METHOD_REFERENCE = HERE / "R31_ALIGNED_HIDDEN_GOAL_REFERENCE.md"
PROTOCOL = HERE / "r31_aligned_hidden_goal_protocol.json"
METHOD_REVISION = "antmaze_r30_aligned_hidden_goal_r31"
PREPARATION_CLAIM_LEVEL = "reward_free_R30_deployment_artifacts_only"
R30_DIRECTORY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r29_aligned_consequence_r30"
)
R30_SUMMARY = R30_DIRECTORY / "summary.json"
R30_CHECKPOINT = R30_DIRECTORY / "consequence.pt"
RESULT_DIRECTORY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r30_deployment_r31"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> tuple[dict, dict, float]:
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "oracle_inputs": [],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_preparation": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "gate_task_ids": [3, 5],
        "development_task_ids": list(range(15)),
        "episodes_per_task": 5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R31 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R31 method reference changed")

    report = json.loads(R30_SUMMARY.read_text())
    if report.get("method_revision") != (
        "antmaze_r29_aligned_factorized_consequence_r30"
    ):
        raise ValueError("R31 requires the full aligned R30 consequence model")
    if report.get("profile") != "full":
        raise ValueError("R31 cannot deploy an R30 smoke checkpoint")
    if report.get("checkpoint_sha256") != sha256(R30_CHECKPOINT):
        raise ValueError("R30 checkpoint no longer matches its report")
    if report.get("hidden_goals_accessed") is not False:
        raise ValueError("R30 accessed hidden goals")
    sigma = float(
        report["aligned_model_on_same_holdout"][
            "landing_precision_m_current_specification"
        ]
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("R30 landing precision must be finite and positive")
    return protocol, report, sigma


PROTOCOL_VALUE, R30_REPORT, SIGMA = load_contract()
SIGMA_ESTIMATOR = (
    "held_out_R29_mean_ensemble_and_mixture_weighted_"
    "within_mode_radial_scale"
)


def graph_path(_root=None) -> Path:
    return RESULT_DIRECTORY / "recorded_transition_graph.npz"


def artifact_paths(_root=None) -> dict[str, Path]:
    directory = RESULT_DIRECTORY / "planner"
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


def passage_paths(_root=None) -> dict[str, Path]:
    directory = RESULT_DIRECTORY / "empirical_passage"
    return {
        "directory": directory,
        "metadata": directory / "metadata.json",
        "recorded": directory / "recorded_passages.npz",
        "bridge_shards": directory / "bridge_shards",
    }


def aligned_checkpoint(_root=None) -> Path:
    return R30_CHECKPOINT


def aligned_alias_kernel(distance, sigma=SIGMA):
    distance = np.asarray(distance, np.float32)
    return np.exp(
        -np.square(distance) / (2.0 * float(sigma) ** 2)
    ).astype(np.float32)


def bind_runtime() -> None:
    """Point every model-derived R28 binding at isolated R31 artifacts."""
    ant_constants.LANDING_SIGMA = SIGMA
    ant_constants.BOOTSTRAP_SIGMA = SIGMA
    recorded.LANDING_SIGMA = SIGMA

    mobility.LANDING_SIGMA = SIGMA
    mobility.factorized_checkpoint = aligned_checkpoint
    mobility._artifact_paths = artifact_paths
    mobility._default_graph = graph_path
    mobility.METHOD_REVISION = METHOD_REVISION
    mobility.CLAIM_LEVEL = PREPARATION_CLAIM_LEVEL

    r22.SIGMA = SIGMA
    r22.ESTIMATOR = SIGMA_ESTIMATOR
    r22.BANDWIDTH_REPORT = R30_SUMMARY
    r22.METHOD_REVISION = METHOD_REVISION
    r22.CLAIM_LEVEL = PREPARATION_CLAIM_LEVEL
    r22.graph_path = graph_path
    r22.artifact_paths = artifact_paths
    r22.factorized_checkpoint = aligned_checkpoint
    mobility._prepare_landmarks = r22._prepare_reach_landmarks
    lazy._prepare_landmarks = r22._prepare_reach_landmarks
    lazy.alias_kernel = aligned_alias_kernel

    # R27 reads these globals when constructing evidence and passage values.
    r27.LANDING_SIGMA = SIGMA
    r27.artifact_paths = passage_paths
    r27.METHOD_REVISION = METHOD_REVISION
    r27.BASE_METHOD_REVISION = METHOD_REVISION


bind_runtime()


class AlignedBoundedConsumptionPlanner(r28.BoundedConsumptionPlanner):
    method_revision = METHOD_REVISION
    planner_name = "hidden_goal_R28_with_R30_aligned_consequence"


def verify_graph(path: Path) -> recorded.RecordedTransitionGraph:
    graph = recorded.RecordedTransitionGraph.load(path)
    expected = {
        "method_revision": METHOD_REVISION,
        "field_sigma_m": float(SIGMA),
        "deployed_consequence_sha256": sha256(R30_CHECKPOINT),
        "r30_summary_sha256": sha256(R30_SUMMARY),
    }
    for key, value in expected.items():
        if graph.metadata.get(key) != value:
            raise ValueError(f"R31 graph has another {key}")
    return graph


def command_prepare_graph(args) -> None:
    path = Path(args.graph)
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        graph = verify_graph(path)
        print(json.dumps({
            "artifact": str(path.resolve()),
            "artifact_sha256": recorded.sha256_file(path),
            "nodes": graph.node_count,
            "edges": graph.edge_count,
            "landing_sigma_m": SIGMA,
            "resumed": True,
        }, indent=2))
        return

    model = FactorizedConsequenceEnsemble(R30_CHECKPOINT, args.device)
    round_directories = [paths(args.root, index)["round"] for index in range(3)]
    starts, commands, endpoints = command_records(round_directories)
    state_scale = model.input_std[: ant_constants.STATE_DIM].copy()
    state_scale[:2] = SIGMA
    graph = recorded.build_recorded_graph(
        starts, commands, endpoints, state_scale
    )
    graph.metadata.update({
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PREPARATION_CLAIM_LEVEL,
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "deployed_consequence": str(R30_CHECKPOINT.resolve()),
        "deployed_consequence_sha256": sha256(R30_CHECKPOINT),
        "r30_summary": str(R30_SUMMARY.resolve()),
        "r30_summary_sha256": sha256(R30_SUMMARY),
        "field_sigma_m": float(SIGMA),
        "field_sigma_estimator": SIGMA_ESTIMATOR,
        "reward_free_rounds": [
            str(directory.resolve()) for directory in round_directories
        ],
        "r29_alignment_used_by_consequence_only": True,
        "r29_held_out_commands_inserted_into_graph": False,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "confirmation_panel_accessed": False,
    })
    graph.save(path)
    print(json.dumps({
        "artifact": str(path.resolve()),
        "artifact_sha256": recorded.sha256_file(path),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "landing_sigma_m": SIGMA,
        "landing_sigma_estimator": SIGMA_ESTIMATOR,
    }, indent=2))


def command_prepare_mobility(args) -> None:
    verify_graph(args.graph)
    mobility.command_prepare_mobility(args)


def command_prepare_bridges(args) -> None:
    verify_graph(args.graph)
    mobility.command_prepare_bridges(args)


def _load_graph_bridges(args):
    graph, _consequence, consequence_path = mobility._load_base(args)
    artifact, landmark_index, targets, _distance, _metadata = (
        r22._prepare_reach_landmarks(args, graph, consequence_path)
    )
    with np.load(artifact["bridges"]) as stored:
        saved_landmarks = stored["landmark_index"]
        saved_targets = stored["target_index"]
        weight = stored["weight"]
    if not np.array_equal(landmark_index, saved_landmarks):
        raise ValueError("R31 bridge landmarks changed")
    if not np.array_equal(targets, saved_targets):
        raise ValueError("R31 bridge targets changed")
    mobility._verify_bridges(
        args, graph, consequence_path, artifact,
        landmark_index, targets, weight,
    )
    return graph, targets, weight


def command_prepare_passage(args) -> None:
    verify_graph(args.graph)
    graph, targets, weight = _load_graph_bridges(args)
    metadata = r27.prepare_geometry(args.root, graph, targets, weight)
    print(json.dumps({
        key: metadata[key]
        for key in (
            "method_revision",
            "recorded_primitive_segments",
            "active_bridges",
            "grid_cells",
            "matched_source_error_median_m",
            "matched_displacement_error_median_m",
        )
    }, indent=2))


def command_prepare_all(args) -> None:
    command_prepare_graph(args)
    command_prepare_mobility(args)
    command_prepare_bridges(args)
    command_prepare_passage(args)


def _load_runtime(args):
    graph = verify_graph(args.graph)
    planner, hashes = lazy._load_planner(
        args, AlignedBoundedConsumptionPlanner
    )
    geometry = r28.BoundedPassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": recorded.sha256_file(
            artifact_paths()["landmarks"]
        ),
        "empirical_passage_geometry_sha256": recorded.sha256_file(
            passage_paths()["metadata"]
        ),
        "r30_summary_sha256": sha256(R30_SUMMARY),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def task_ids(set_name: str) -> tuple[int, ...]:
    if set_name == "gate2":
        return tuple(int(value) for value in PROTOCOL_VALUE["gate_task_ids"])
    if set_name == "all15":
        return tuple(
            int(value) for value in PROTOCOL_VALUE["development_task_ids"]
        )
    raise ValueError(set_name)


def evaluation_directory(set_name: str) -> Path:
    return RESULT_DIRECTORY / f"evaluation_{set_name}"


def task_result_path(set_name: str, task_id: int) -> Path:
    gate_path = evaluation_directory("gate2") / f"task_{task_id:03d}.json"
    if (
        set_name == "all15"
        and task_id in task_ids("gate2")
        and gate_path.exists()
    ):
        return gate_path
    return evaluation_directory(set_name) / f"task_{task_id:03d}.json"


def _evaluate_task(args, planner, executor, env, task_id: int) -> dict:
    task = r22._task(args.root, task_id)
    output = evaluation_directory(args.set_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(args.set_name, task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), float(SIGMA),
            sha256(R30_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("landing_sigma_m"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(
                f"existing R31 task has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = r28.baseline.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        SIGMA,
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "R28_with_R30_aligned_consequence_and_rebuilt_artifacts",
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_task_time_method_revision": r28.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "landing_sigma_m": float(SIGMA),
        "landing_sigma_estimator": SIGMA_ESTIMATOR,
        "acquisition_search_sigma_m": float(SIGMA),
        "return_field_sigma_m": float(SIGMA),
        "acquisition_immediate_readout": "empirical_passage_integral",
        "acquisition_backup": "bounded_consumption_aware_first_contact",
        "return_planner": "R22_plus_exact_online_positive_evidence_candidates",
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate_set(args) -> None:
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, int(task_id))
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args) -> None:
    selected_task_ids = task_ids(args.set_name)
    output = evaluation_directory(args.set_name)
    rows = [
        json.loads(task_result_path(args.set_name, int(task_id)).read_text())
        for task_id in selected_task_ids
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
        "method_reference_sha256": sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_task_time_method_revision": r28.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "set_name": args.set_name,
        "task_ids": list(selected_task_ids),
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
        "landing_sigma_m": float(SIGMA),
        "landing_sigma_estimator": SIGMA_ESTIMATOR,
        "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
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
            "acquisition_two_cycle_count", "landing_sigma_m",
        )
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(
        func=lambda _args: print(json.dumps({
            "method_revision": METHOD_REVISION,
            "method_reference": str(METHOD_REFERENCE),
            "r30_checkpoint": str(R30_CHECKPOINT),
            "r30_checkpoint_sha256": sha256(R30_CHECKPOINT),
            "landing_sigma_m": SIGMA,
            "landing_sigma_estimator": SIGMA_ESTIMATOR,
            "artifact_root": str(RESULT_DIRECTORY),
            "confirmation_panel_accessed": False,
        }, indent=2))
    )
    commands.add_parser("prepare-graph").set_defaults(func=command_prepare_graph)
    commands.add_parser("prepare-mobility").set_defaults(func=command_prepare_mobility)
    commands.add_parser("prepare-bridges").set_defaults(func=command_prepare_bridges)
    commands.add_parser("prepare-passage").set_defaults(func=command_prepare_passage)
    commands.add_parser("prepare-all").set_defaults(func=command_prepare_all)
    evaluate = commands.add_parser("evaluate-set")
    evaluate.add_argument(
        "--set-name", choices=("gate2", "all15"), required=True
    )
    evaluate.add_argument(
        "--episodes", type=int, default=int(PROTOCOL_VALUE["episodes_per_task"])
    )
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument(
        "--set-name", choices=("gate2", "all15"), required=True
    )
    summarize.add_argument(
        "--episodes", type=int, default=int(PROTOCOL_VALUE["episodes_per_task"])
    )
    summarize.set_defaults(func=command_summarize)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.graph is None:
        args.graph = graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()

