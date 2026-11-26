"""Repeat the R51 stalled pair with E549 and E549-matched R52 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT,
    E549_CHECKPOINT_SHA256,
    E549SACExecutor,
    inspect_e549_checkpoint,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r53_e549_r51_protocol.json"
REFERENCE = HERE / "R53_E549_R51_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_5m_destination_arrival_r53"
EVALUATION_ARM = "R51_destination_arrival_with_E549_R52_preparation"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_destination_arrival_r53/stalled_pair"
)
FARAMA_R51_SUMMARY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "destination_arrival_correction_r51/audit_stalled_pair_farama/summary.json"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "oracle_inputs": [],
        "task_ids": [2, 13],
        "episodes": 5,
        "evaluation_seed": 607,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R53 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R53 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


class E549DestinationArrivalPlanner(r51.DestinationArrivalOnlinePlanner):
    method_revision = METHOD_REVISION
    planner_name = "R51_destination_arrival_with_E549_R52_preparation"


def command_audit(args) -> dict:
    executor = inspect_e549_checkpoint()
    paths = r52.output_paths(args.preparation)
    required = (
        paths["consequence"],
        paths["consequence_summary"],
        paths["graph"],
        paths["runtime"] / "planner/mobility.pt",
        paths["runtime"] / "planner/bridges.npz",
        paths["runtime"] / "empirical_passage/metadata.json",
    )
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": PROTOCOL_VALUE["episodes"],
        "executor": executor,
        "preparation": str(Path(args.preparation).resolve()),
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()} for path in required
        ],
        "ready_for_evaluation": bool(all(path.is_file() for path in required)),
    }
    print(json.dumps(report, indent=2))
    return report


def _load_runtime(args):
    r52.configure_r52_runtime(args.preparation)
    paths = r52.output_paths(args.preparation)
    args.graph = paths["graph"]
    graph, artifact, landmark, targets, weight = r48._load_bridges(args)
    consequence = r48.AntConsequenceAdapter(r48.R30_CHECKPOINT, args.device)
    endpoint_mobility = r48.mobility.MobilityModel(
        artifact["mobility"], args.device
    )
    planner = E549DestinationArrivalPlanner(
        consequence,
        endpoint_mobility,
        graph,
        landmark,
        targets,
        weight,
        np.random.default_rng(args.seed),
        args.seed,
    )
    geometry = r48.r31.r28.BoundedPassageGeometry(
        args.root, graph, targets, weight
    )
    r48.verify_passage_metadata(args, geometry.metadata)
    planner.configure_hidden_fields(r48.SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        "graph_sha256": r48.sha256(args.graph),
        "factorized_consequence_sha256": r48.sha256(r48.R30_CHECKPOINT),
        "mobility_sha256": r48.sha256(artifact["mobility"]),
        "bridges_sha256": r48.sha256(artifact["bridges"]),
        "reach_landmarks_sha256": r48.sha256(artifact["landmarks"]),
        "empirical_passage_geometry_sha256": r48.sha256(
            r48.passage_paths()["metadata"]
        ),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
    }
    executor = E549SACExecutor(E549_CHECKPOINT, args.device)
    env = r48.make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def _evaluate_task(args, planner, executor, env, task_id: int, output: Path):
    task_path = output / f"task_{task_id:03d}.json"
    log_path = output / f"task_{task_id:03d}.jsonl"
    if task_path.exists():
        saved = json.loads(task_path.read_text())
        expected = {
            "task_id": int(task_id),
            "method_revision": METHOD_REVISION,
            "episodes": int(args.episodes),
            "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        }
        for key, item in expected.items():
            if saved.get(key) != item:
                raise ValueError(f"saved R53 task has another {key}")
        return saved
    task = r48.r31.r22._task(args.root, task_id)
    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    planner.reset_online_correction_task()
    row = r48.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        log_path,
        r48.PROTOCOL_VALUE["evaluator"]["success_radius_m_evaluator_only"],
    )
    outcomes = np.asarray(planner.online_outcomes, np.float32)
    row.update(task)
    row.update({
        "evaluation_arm": EVALUATION_ARM,
        "executor": PROTOCOL_VALUE["executor"],
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "selection_rule": PROTOCOL_VALUE["selection_rule"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "landing_sigma_m": float(r48.SIGMA),
        "online_outcome": PROTOCOL_VALUE["online_outcome"],
        "online_consequence_observations": int(len(outcomes)),
        "online_destination_arrival_rate": (
            float(outcomes.mean()) if len(outcomes) else None
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    r48.dump(task_path, row)
    return row


def _prior_comparison(rows: list[dict]) -> dict | None:
    if not FARAMA_R51_SUMMARY.is_file():
        return None
    prior = json.loads(FARAMA_R51_SUMMARY.read_text())
    prior_rows = {int(row["task_id"]): row for row in prior.get("rows", [])}
    comparisons = []
    for row in rows:
        old = prior_rows.get(int(row["task_id"]))
        if old is None:
            continue
        comparisons.append({
            "task_id": int(row["task_id"]),
            "farama_acquired": old.get("first_contact_episode") is not None,
            "e549_acquired": row.get("first_contact_episode") is not None,
            "farama_new_1m_cells_per_command": old.get(
                "acquisition_new_1m_cells_per_command"
            ),
            "e549_new_1m_cells_per_command": row.get(
                "acquisition_new_1m_cells_per_command"
            ),
            "farama_zero_new_1m_cell_fraction": old.get(
                "acquisition_zero_new_1m_cell_fraction"
            ),
            "e549_zero_new_1m_cell_fraction": row.get(
                "acquisition_zero_new_1m_cell_fraction"
            ),
        })
    return {
        "reference": str(FARAMA_R51_SUMMARY),
        "reference_sha256": r52.sha256(FARAMA_R51_SUMMARY),
        "rows": comparisons,
    }


def command_evaluate(args) -> dict:
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes"]):
        raise ValueError("R53 requires the frozen five-episode contract")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R53 requires the same R51 evaluation seed")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError("R52 preparation is incomplete")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if (
            saved.get("method_revision") != METHOD_REVISION
            or saved.get("task_ids") != PROTOCOL_VALUE["task_ids"]
            or saved.get("executor_checkpoint_sha256")
            != E549_CHECKPOINT_SHA256
        ):
            raise ValueError("saved R53 summary has another contract")
        print(json.dumps(saved, indent=2))
        return saved
    planner, executor, env = _load_runtime(args)
    try:
        rows = [
            _evaluate_task(args, planner, executor, env, task_id, output)
            for task_id in PROTOCOL_VALUE["task_ids"]
        ]
    finally:
        env.close()
    contacts = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        contact = row["first_contact_episode"]
        if contact is not None:
            returns.extend(row["successes"][int(contact) + 1:])
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "selection_rule": PROTOCOL_VALUE["selection_rule"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "audit_scope": (
            "two_outcome_selected_consumed_development_tasks_component_"
            "audit_not_a_performance_estimate"
        ),
        "executor": PROTOCOL_VALUE["executor"],
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < int(args.episodes)
            for value in contacts
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
        "online_consequence_observations": int(sum(
            row["online_consequence_observations"] for row in rows
        )),
        "online_destination_arrival_rate": float(np.mean([
            row["online_destination_arrival_rate"] for row in rows
        ])),
        "landing_sigma_m": float(r48.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "farama_r51_comparison": _prior_comparison(rows),
        "rows": rows,
    }
    r48.dump(summary_path, result)
    print(json.dumps({
        key: result[key] for key in (
            "tasks",
            "task_ids",
            "acquired_tasks",
            "acquisition_at_1",
            "acquisition_at_5",
            "pooled_conditional_return",
            "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "online_destination_arrival_rate",
        )
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=5)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
