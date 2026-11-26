#!/usr/bin/env python3
"""Hidden-goal original-nine panel for the E549/R52/R51 stack."""

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
from projects.tfsn_consequence_value_iteration.antmaze import (
    r53_e549_r51_evaluation as r53,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT_SHA256,
    inspect_e549_checkpoint,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r59_e549_hidden_goal_all9_protocol.json"
REFERENCE = HERE / "R59_E549_HIDDEN_GOAL_ALL9_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_5m_hidden_goal_original9_r59"
EVALUATION_ARM = "R51_destination_arrival_with_E549_R52_preparation"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_hidden_goal_r59/original9"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "oracle_inputs": [],
        "task_ids": list(range(9)),
        "episodes": 5,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R59 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R59 method reference changed")
    # The expanded panel must preserve R53's task-level evaluation contract.
    for key in ("episodes", "evaluation_seed", "online_outcome"):
        if value[key] != r53.PROTOCOL_VALUE[key]:
            raise ValueError(f"R59 no longer matches R53 on {key}")
    return value


PROTOCOL_VALUE = load_protocol()


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
    panel_path = Path(args.root).resolve() / "panels/development.json"
    panel = json.loads(panel_path.read_text())
    panel_ids = [int(row["task_id"]) for row in panel.get("tasks", [])]
    panel_ok = bool(
        panel.get("name") == "development"
        and int(panel.get("goal_offset", -1)) == 2
        and panel_ids == list(range(9))
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
        "commands_per_episode": PROTOCOL_VALUE["commands_per_episode"],
        "executor": executor,
        "preparation": str(Path(args.preparation).resolve()),
        "development_panel": str(panel_path),
        "development_panel_is_frozen_original_nine": panel_ok,
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()} for path in required
        ],
        "ready_for_evaluation": bool(
            panel_ok and all(path.is_file() for path in required)
        ),
    }
    print(json.dumps(report, indent=2))
    return report


def evaluate_task(args, planner, executor, env, task_id: int, output: Path):
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
                raise ValueError(f"saved R59 task has another {key}")
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
    row.update(
        {
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
            "task_reward_used": False,
            "task_gradient_updates": 0,
            "confirmation_panel_accessed": False,
            **planner.artifact_hashes,
        }
    )
    r48.dump(task_path, row)
    print(
        json.dumps(
            {
                "task_id": int(task_id),
                "first_contact_episode": row["first_contact_episode"],
                "successes": row["successes"],
                "conditional_return": row["conditional_return"],
                "mean_e2e": row["mean_e2e"],
                "online_destination_arrival_rate": row[
                    "online_destination_arrival_rate"
                ],
            }
        ),
        flush=True,
    )
    return row


def command_evaluate(args) -> dict:
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes"]):
        raise ValueError("R59 requires the frozen five-episode contract")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R59 requires the frozen R53 evaluation seed")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError("R52 preparation or original-nine panel is incomplete")
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
            raise ValueError("saved R59 summary has another contract")
        print(json.dumps(saved, indent=2))
        return saved

    planner, executor, env = r53._load_runtime(args)
    try:
        rows = [
            evaluate_task(args, planner, executor, env, task_id, output)
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
            returns.extend(row["successes"][int(contact) + 1 :])
    observations = int(sum(row["online_consequence_observations"] for row in rows))
    arrivals = float(
        sum(
            row["online_consequence_observations"]
            * row["online_destination_arrival_rate"]
            for row in rows
            if row["online_destination_arrival_rate"] is not None
        )
    )
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
            "original_nine_consumed_development_tasks_hidden_goal_component_"
            "panel_not_confirmation_evidence"
        ),
        "executor": PROTOCOL_VALUE["executor"],
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(
            np.mean(
                [
                    value is not None and value < int(args.episodes)
                    for value in contacts
                ]
            )
        ),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean(successes)),
        "acquisition_new_1m_cells_per_command": float(
            np.mean([row["acquisition_new_1m_cells_per_command"] for row in rows])
        ),
        "acquisition_zero_new_1m_cell_fraction": float(
            np.mean([row["acquisition_zero_new_1m_cell_fraction"] for row in rows])
        ),
        "acquisition_revisited_endpoint_fraction": float(
            np.mean([row["acquisition_revisited_endpoint_fraction"] for row in rows])
        ),
        "online_consequence_observations": observations,
        "online_destination_arrival_rate": (
            arrivals / observations if observations else None
        ),
        "landing_sigma_m": float(r48.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    r48.dump(summary_path, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
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
            },
            indent=2,
        )
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
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
