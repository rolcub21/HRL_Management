#!/usr/bin/env python3
"""Supplied-goal original-nine audit for the E549/R52/R51 stack."""

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
from projects.tfsn_consequence_value_iteration.antmaze import (
    r56_e549_supplied_goal as r56,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT_SHA256,
    inspect_e549_checkpoint,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r58_e549_supplied_goal_all9_protocol.json"
REFERENCE = HERE / "R58_E549_SUPPLIED_GOAL_ALL9_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_5m_supplied_goal_original9_r58"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_supplied_goal_r58/original9"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "oracle_inputs": [],
        "task_ids": list(range(9)),
        "episodes_per_task": 1,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "goal_visible_to_planner_at_time_zero": True,
        "goal_visible_to_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R58 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R58 method reference changed")
    # R58 reuses R56's evaluator only where their execution contract is exact.
    for key in ("commands_per_episode", "fixed_command_duration"):
        if value[key] != r56.PROTOCOL_VALUE[key]:
            raise ValueError(f"R58 no longer matches R56 on {key}")
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
        "episodes_per_task": PROTOCOL_VALUE["episodes_per_task"],
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


def evaluate_one(args, planner, executor, env, task_id, output):
    task_path = output / f"task_{task_id:03d}.json"
    log_path = output / f"task_{task_id:03d}.jsonl"
    if task_path.exists():
        saved = json.loads(task_path.read_text())
        if (
            saved.get("method_revision") != METHOD_REVISION
            or saved.get("task_id") != int(task_id)
            or saved.get("executor_checkpoint_sha256") != E549_CHECKPOINT_SHA256
        ):
            raise ValueError("saved R58 task has another contract")
        return saved

    task = r48.r31.r22._task(args.root, int(task_id))
    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    planner.reset_online_correction_task()
    row = r56.evaluate_supplied_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        int(task["reset_seed"]),
        log_path,
    )
    row.update(task)
    row.update(
        {
            "method_reference": str(REFERENCE),
            "method_reference_sha256": r52.sha256(REFERENCE),
            "method_revision": METHOD_REVISION,
            "base_preparation_revision": r52.METHOD_REVISION,
            "online_correction_revision": r51.METHOD_REVISION,
            "claim_level": PROTOCOL_VALUE["claim_level"],
            "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
            "oracle_inputs": [],
            "executor": PROTOCOL_VALUE["executor"],
            "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
            "landing_sigma_m": float(r48.SIGMA),
            "goal_visible_to_planner_at_time_zero": True,
            "goal_visible_to_executor_or_models": False,
            "success_radius_read_by_planner": False,
            "task_reward_used": False,
            "task_gradient_updates": 0,
            "confirmation_panel_accessed": False,
            **planner.artifact_hashes,
        }
    )
    r48.dump(task_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate(args) -> dict:
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R58 requires the frozen R53/R56 evaluation seed")
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
        ):
            raise ValueError("saved R58 summary has another contract")
        print(json.dumps(saved, indent=2))
        return saved

    planner, executor, env = r53._load_runtime(args)
    try:
        rows = [
            evaluate_one(args, planner, executor, env, task_id, output)
            for task_id in PROTOCOL_VALUE["task_ids"]
        ]
    finally:
        env.close()

    observations = int(sum(row["online_consequence_observations"] for row in rows))
    arrivals = float(
        sum(
            row["online_consequence_observations"]
            * row["online_destination_arrival_rate"]
            for row in rows
            if row["online_destination_arrival_rate"] is not None
        )
    )
    successful_steps = [row["steps_to_goal"] for row in rows if row["success"]]
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
        "executor": PROTOCOL_VALUE["executor"],
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": PROTOCOL_VALUE["episodes_per_task"],
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_minimum_distance_m": float(
            np.mean([row["minimum_distance_m"] for row in rows])
        ),
        "mean_steps_to_goal_success": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "online_consequence_observations": observations,
        "online_destination_arrival_rate": (
            arrivals / observations if observations else None
        ),
        "landing_sigma_m": float(r48.SIGMA),
        "goal_visible_to_planner_at_time_zero": True,
        "goal_visible_to_executor_or_models": False,
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
                    "successes",
                    "success_rate",
                    "mean_minimum_distance_m",
                    "mean_steps_to_goal_success",
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
