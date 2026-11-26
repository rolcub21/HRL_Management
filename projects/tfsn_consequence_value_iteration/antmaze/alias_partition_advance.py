"""R14 diagnostic: partition same-place bridges from their next actions.

R13 added a direct next action for every R10 landmark candidate, but retained
the original bridge at full value. Consequently the planner could still spend
a physical command on a spatial alias. R14 makes the alternatives a
continuous partition using the existing landing kernel:

    bridge weight = 1 - kappa_sigma(current_xy, landmark_xy)
    advance weight = kappa_sigma(current_xy, landmark_xy)

No distance threshold is introduced. Far landmarks retain their R10 bridge;
at the same place, the physical bridge vanishes and its one-step continuation
is evaluated directly from the live full state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import EPISODE_STEPS, LANDING_SIGMA
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths

from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    PANEL_REVISION,
    output_paths as expanded_output_paths,
    prepare_panel,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    LONG_ROUTE_TASK_IDS,
    LazyAliasAdvancePlanner,
    _load_planner,
)


METHOD_REVISION = "antmaze_r10_partitioned_alias_advance_r14"
CLAIM_LEVEL = "consumed_development_long_route_diagnostic"
ALL_DEVELOPMENT_TASK_IDS = tuple(range(15))


class PartitionedAliasAdvancePlanner(LazyAliasAdvancePlanner):
    method_revision = METHOD_REVISION
    planner_name = "r10_partitioned_alias_advance"

    def _association_weights(self, association):
        association = np.asarray(association, np.float32)
        return 1.0 - association, association


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "r10_partitioned_alias_advance_r14_long3"
    )


def _task(root, task_id):
    task_id = int(task_id)
    if task_id not in ALL_DEVELOPMENT_TASK_IDS:
        raise ValueError(
            f"task id must be one of {ALL_DEVELOPMENT_TASK_IDS}"
        )
    if task_id < 9:
        selected = json.loads(
            (paths(root)["panels"] / "development.json").read_text()
        )
    else:
        selected = prepare_panel(root)
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

    planner, hashes = _load_planner(args, PartitionedAliasAdvancePlanner)
    task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    row = evaluate_supplied_goal_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        LANDING_SIGMA,
        max_steps=EPISODE_STEPS,
        log_path=output / f"task_{task['task_id']:03d}.jsonl",
    )
    env.close()
    row.update(task)
    row.update({
        "evaluation_arm": "r10_partitioned_alias_advance",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
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
    r10_root = expanded_output_paths(args.root)["r10"]
    references = [
        json.loads((r10_root / f"task_{task_id:03d}.json").read_text())
        for task_id in LONG_ROUTE_TASK_IDS
    ]
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "three_consumed_long_route_failures_only",
            "continuous_partition_of_bridge_and_direct_continuation",
            "partition_uses_existing_sigma_kernel_without_new_threshold",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": list(LONG_ROUTE_TASK_IDS),
        "successes": int(sum(row["success"] for row in rows)),
        "r10_reference_successes": int(
            sum(row["success"] for row in references)
        ),
        "minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in rows
        },
        "r10_reference_minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in references
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
        "r10_reference_minimum_distance": result[
            "r10_reference_minimum_distance"
        ],
    }, indent=2))


def _r10_reference_rows(root):
    original = json.loads((
        paths(root)["eval"]
        / "development"
        / "mobility_stitched_value_iteration_r10"
        / "summary.json"
    ).read_text())["rows"]
    extension = json.loads((
        expanded_output_paths(root)["r10"] / "summary.json"
    ).read_text())["rows"]
    rows = original + extension
    if [int(row["task_id"]) for row in rows] != list(
        ALL_DEVELOPMENT_TASK_IDS
    ):
        raise ValueError("the frozen R10 references are not the 15 tasks")
    return rows


def command_summarize_all(args):
    output = output_directory(args.root)
    rows = []
    for task_id in ALL_DEVELOPMENT_TASK_IDS:
        task_path = output / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows.append(json.loads(task_path.read_text()))
    references = _r10_reference_rows(args.root)
    successes = int(sum(row["success"] for row in rows))
    original_successes = int(sum(row["success"] for row in rows[:9]))
    extension_successes = int(sum(row["success"] for row in rows[9:]))
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": "expanded_consumed_development_regression",
        "declared_deviations": [
            "fifteen_consumed_development_goals",
            "continuous_partition_of_bridge_and_direct_continuation",
            "partition_uses_existing_sigma_kernel_without_new_threshold",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "successes": successes,
        "success_rate": float(successes / len(rows)),
        "original_9_successes": original_successes,
        "new_6_successes": extension_successes,
        "r10_reference_successes": int(
            sum(row["success"] for row in references)
        ),
        "r10_original_9_successes": int(
            sum(row["success"] for row in references[:9])
        ),
        "r10_new_6_successes": int(
            sum(row["success"] for row in references[9:])
        ),
        "paired_changes": {
            "gained_task_ids": [
                int(row["task_id"])
                for row, reference in zip(rows, references)
                if row["success"] and not reference["success"]
            ],
            "lost_task_ids": [
                int(row["task_id"])
                for row, reference in zip(rows, references)
                if not row["success"] and reference["success"]
            ],
        },
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary_all15.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "successes",
            "tasks",
            "success_rate",
            "original_9_successes",
            "new_6_successes",
            "r10_reference_successes",
            "paired_changes",
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
    task = subparsers.add_parser("task")
    task.add_argument("--task-id", type=int, required=True)
    task.set_defaults(func=command_task)
    summarize = subparsers.add_parser("summarize")
    summarize.set_defaults(func=command_summarize)
    summarize_all = subparsers.add_parser("summarize-all")
    summarize_all.set_defaults(func=command_summarize_all)
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
