"""Run frozen R14 with a deliberately extended 20-command diagnostic budget.

The canonical AntMaze result remains the 1,000-step, ten-command evaluation.
This module changes only the diagnostic horizon to 2,000 primitive steps and
twenty fixed 100-step commands. It exists to answer whether R14's two remaining
long-route failures are time-limited or trapped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.collect import (
    SuppliedGoalField,
    execute_command,
)
from scripts.antmaze.current_tfsn.constants import (
    COMMAND_STEPS,
    CONTACT_RADIUS,
    LANDING_SIGMA,
)
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import dump, paths

from projects.tfsn_consequence_value_iteration.antmaze.alias_partition_advance import (
    PartitionedAliasAdvancePlanner,
)
from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    PANEL_REVISION,
    output_paths as expanded_output_paths,
    prepare_panel,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    LONG_ROUTE_TASK_IDS,
    _load_planner,
)


METHOD_REVISION = "antmaze_r14_extended_20_command_diagnostic_r15"
CLAIM_LEVEL = "consumed_development_nonbenchmark_budget_diagnostic"
COMMAND_BUDGET = 20
STEP_BUDGET = COMMAND_BUDGET * COMMAND_STEPS


class ExtendedBudgetPlanner(PartitionedAliasAdvancePlanner):
    method_revision = METHOD_REVISION
    planner_name = "r14_partitioned_alias_extended_budget"
    horizon = COMMAND_BUDGET


def evaluate_extended_task(
    env, executor, planner, goal, reset_seed, sigma, log_path
):
    """Exact supplied-goal evaluator with the declared nonbenchmark horizon."""
    goal = np.asarray(goal, np.float32)
    fields = SuppliedGoalField(sigma, goal)
    observation, _ = env.reset(seed=int(reset_seed))
    state = full_state(observation)
    initial_distance = float(np.linalg.norm(state[:2] - goal))
    minimum_distance = initial_distance
    success = initial_distance <= CONTACT_RADIUS
    steps_to_goal = 0 if success else None
    steps_executed = 0
    decisions = []

    for command_index in range(COMMAND_BUDGET):
        if success:
            break
        command, diagnostic = planner.choose(state, command_index, fields)
        trajectory, rewards = execute_command(
            env, executor, state, command, hidden_goal=goal
        )
        distances = np.linalg.norm(trajectory[1:, :2] - goal[None], axis=1)
        minimum_distance = min(minimum_distance, float(distances.min()))
        if rewards.any():
            steps_to_goal = (
                steps_executed + int(np.flatnonzero(rewards)[0]) + 1
            )
            success = True
        steps_executed += COMMAND_STEPS
        diagnostic.update({
            "command_index": int(command_index),
            "command": command.tolist(),
            "actual_endpoint": trajectory[-1, :2].tolist(),
            "moved_distance": float(
                np.linalg.norm(trajectory[-1, :2] - state[:2])
            ),
            "minimum_goal_distance_in_command": float(distances.min()),
            "reward": bool(rewards.any()),
        })
        decisions.append(diagnostic)
        state = trajectory[-1]

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(json.dumps(row) for row in decisions) + "\n"
    )
    return {
        "initial_distance": initial_distance,
        "success": bool(success),
        "steps_to_goal": steps_to_goal,
        "steps_executed": steps_executed,
        "minimum_distance": minimum_distance,
        "commands_executed": len(decisions),
        "contact_radius": CONTACT_RADIUS,
    }


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "r14_extended_20_command_diagnostic_r15_long3"
    )


def _task(root, task_id):
    selected = prepare_panel(root)
    task_id = int(task_id)
    if task_id not in LONG_ROUTE_TASK_IDS:
        raise ValueError(f"task id must be one of {LONG_ROUTE_TASK_IDS}")
    return next(
        row for row in selected["tasks"] if row["task_id"] == task_id
    )


def command_task(args):
    task = _task(args.root, args.task_id)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task['task_id']:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (task["goal"], task["reset_seed"], METHOD_REVISION)
        actual = (
            saved.get("goal"),
            saved.get("reset_seed"),
            saved.get("method_revision"),
        )
        if actual != expected:
            raise ValueError(
                f"existing task result has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return

    planner, hashes = _load_planner(args, ExtendedBudgetPlanner)
    task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=20_000)
    row = evaluate_extended_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        LANDING_SIGMA,
        output / f"task_{task['task_id']:03d}.jsonl",
    )
    env.close()
    row.update(task)
    row.update({
        "evaluation_arm": "r14_extended_20_command_diagnostic",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "primitive_step_budget": STEP_BUDGET,
        "command_budget": COMMAND_BUDGET,
        "command_duration_steps": COMMAND_STEPS,
        "canonical_benchmark_budget": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        **hashes,
    })
    dump(result_path, row)
    print(json.dumps(row))


def command_summarize(args):
    output = output_directory(args.root)
    rows = []
    for task_id in LONG_ROUTE_TASK_IDS:
        task_path = output / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows.append(json.loads(task_path.read_text()))
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "three_consumed_long_route_failures_only",
            "primitive_step_budget_increased_from_1000_to_2000",
            "command_horizon_increased_from_10_to_20",
        ],
        "oracle_inputs": [],
        "primitive_step_budget": STEP_BUDGET,
        "command_budget": COMMAND_BUDGET,
        "canonical_benchmark_budget": False,
        "tasks": len(rows),
        "task_ids": list(LONG_ROUTE_TASK_IDS),
        "successes": int(sum(row["success"] for row in rows)),
        "minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in rows
        },
        "steps_to_goal": {
            str(row["task_id"]): row["steps_to_goal"] for row in rows
        },
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "successes": result["successes"],
        "tasks": result["tasks"],
        "minimum_distance": result["minimum_distance"],
        "steps_to_goal": result["steps_to_goal"],
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
    task = subparsers.add_parser("task")
    task.add_argument("--task-id", type=int, required=True)
    task.set_defaults(func=command_task)
    subparsers.add_parser("summarize").set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = (
            Path(args.root)
            / "model_planning"
            / "round2_recorded_transition_graph_r6.npz"
        )
    args.func(args)


if __name__ == "__main__":
    main()
