"""Equal-budget one-long-episode diagnostic for frozen AntMaze R39."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r39_live_posterior_passage as r39,
)
from scripts.antmaze.current_tfsn.main import dump


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "r40_continuous_episode_protocol.json"
REFERENCE = HERE / "R40_CONTINUOUS_EPISODE_DIAGNOSTIC_REFERENCE.md"
METHOD_REVISION = "antmaze_r39_continuous_50_command_diagnostic_r40"
OUTPUT = r39.RESULT_DIRECTORY / "diagnostic_gate2_continuous50"


def contract():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": "consumed_development_episode_structure_diagnostic_only",
        "oracle_inputs": [],
        "gate_task_ids": [3, 5],
        "physical_episodes_per_task": 1,
        "commands_per_physical_episode": 50,
        "planner_horizon_commands": 10,
        "primitive_steps_per_command": 100,
        "total_commands_per_task": 50,
        "total_primitive_steps_per_task": 5000,
        "baseline_physical_episodes_per_task": 5,
        "baseline_commands_per_physical_episode": 10,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R40 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = r39.r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE:
        raise ValueError("R40 method reference changed")
    return value


CONTRACT = contract()


def task_path(task_id):
    return OUTPUT / f"task_{int(task_id):03d}.json"


def evaluate_task(args, planner, executor, env, task_id):
    task = r39.r31.r22._task(args.root, task_id)
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_path(task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            CONTRACT["commands_per_physical_episode"],
            CONTRACT["total_primitive_steps_per_task"],
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"),
            saved.get("commands_per_physical_episode"),
            saved.get("total_primitive_steps_per_task"),
        )
        if actual != expected:
            raise ValueError(f"existing R40 task has another contract: {result_path}")
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
        CONTRACT["physical_episodes_per_task"],
        output / f"task_{task_id:03d}.jsonl",
        commands_per_episode=CONTRACT["commands_per_physical_episode"],
        planner_horizon_commands=CONTRACT["planner_horizon_commands"],
    )
    row.update(task)
    row.update({
        "evaluation_arm": "frozen_R39_one_continuous_50_command_episode",
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r39.r31.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": CONTRACT["claim_level"],
        "declared_deviations": CONTRACT["declared_deviations"],
        "oracle_inputs": [],
        "physical_episodes": CONTRACT["physical_episodes_per_task"],
        "commands_per_physical_episode": CONTRACT[
            "commands_per_physical_episode"
        ],
        "planner_horizon_commands": CONTRACT["planner_horizon_commands"],
        "total_commands_per_task": CONTRACT["total_commands_per_task"],
        "total_primitive_steps_per_task": CONTRACT[
            "total_primitive_steps_per_task"
        ],
        "baseline_total_commands_per_task": 50,
        "equal_total_interaction_budget": True,
        "task_memory_persisted": True,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "return_evaluated": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def summarize():
    rows = [json.loads(task_path(task_id).read_text()) for task_id in CONTRACT["gate_task_ids"]]
    baseline_path = r39.evaluation_directory("gate2") / "summary.json"
    baseline = json.loads(baseline_path.read_text())
    contact = [row["first_contact_episode"] for row in rows]
    result = {
        "schema_version": 1,
        "experiment_id": CONTRACT["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r39.r31.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": CONTRACT["claim_level"],
        "declared_deviations": CONTRACT["declared_deviations"],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": CONTRACT["gate_task_ids"],
        "equal_total_commands_per_task": 50,
        "baseline_structure": "5_physical_episodes_times_10_commands",
        "diagnostic_structure": "1_physical_episode_times_50_commands",
        "baseline_acquired_tasks": int(baseline["acquired_tasks"]),
        "continuous_acquired_tasks": int(sum(x is not None for x in contact)),
        "continuous_acquisition_rate": float(np.mean([x is not None for x in contact])),
        "continuous_contact_latency_mean_steps": float(np.mean([
            row["contact_latency"] for row in rows
        ])),
        "continuous_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "continuous_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "continuous_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
        ])),
        "continuous_two_cycle_count": int(sum(
            row["acquisition_two_cycle_count"] for row in rows
        )),
        "return_evaluated": False,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(OUTPUT / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "equal_total_commands_per_task",
            "baseline_acquired_tasks", "continuous_acquired_tasks",
            "continuous_acquisition_rate",
            "continuous_contact_latency_mean_steps",
            "continuous_new_1m_cells_per_command",
            "continuous_zero_new_1m_cell_fraction",
            "continuous_revisited_endpoint_fraction",
            "continuous_two_cycle_count", "return_evaluated",
        )
    }, indent=2))


def command_evaluate(args):
    planner, executor, env = r39._load_runtime(args)
    try:
        for task_id in CONTRACT["gate_task_ids"]:
            evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    summarize()


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=lambda _args: print(json.dumps(CONTRACT, indent=2)))
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    commands.add_parser("summarize").set_defaults(func=lambda _args: summarize())
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r39.r31.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
