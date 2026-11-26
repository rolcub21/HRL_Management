"""R18: bounded first-occupancy Bellman planning with exact mixtures.

R17 removed endpoint Monte Carlo noise but retained accumulated occupancy:
``F + V``.  R18 instead backs up the probability of first contact,

    F + (1 - F) V,

with ``V_0 = 0``.  The same bounded objective is used in the recorded/model
bridge graph and in both levels of the live Bellman backup.  R17's exact
spatial-mixture integration is unchanged.  The initial gate contains only the
three consumed goals (0, 4, 8) that R17 lost relative to R14.
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

from projects.tfsn_consequence_value_iteration.antmaze.alias_partition_advance import (
    ALL_DEVELOPMENT_TASK_IDS,
    _r10_reference_rows,
    _task,
)
from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    PANEL_REVISION,
    output_paths as expanded_output_paths,
)
from projects.tfsn_consequence_value_iteration.antmaze.integrated_bellman_planning import (
    IntegratedBellmanPlanner,
    _calibration,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    _load_planner,
)


METHOD_REVISION = "antmaze_first_occupancy_integrated_bellman_r18"
CLAIM_LEVEL = "expanded_consumed_development_planner_diagnostic"
REGRESSION_GATE_TASK_IDS = (0, 4, 8)


def first_occupancy_backup(contact, continuation):
    """Probability of contact now or, if not, during the continuation."""
    contact = np.asarray(contact, np.float32)
    continuation = np.asarray(continuation, np.float32)
    if contact.shape != continuation.shape:
        raise ValueError("first-occupancy backup terms must align")
    if np.any(contact < 0) or np.any(contact > 1):
        raise ValueError("contact values must lie in [0, 1]")
    if np.any(continuation < 0) or np.any(continuation > 1 + 1e-6):
        raise ValueError("continuation values must lie in [0, 1]")
    return np.clip(
        contact + (1.0 - contact) * continuation, 0.0, 1.0
    ).astype(np.float32)


def first_occupancy_stitched_values(
    graph, targets, weight, goal, horizon=10
):
    """Finite-horizon first-contact values on the frozen R10 graph."""
    goal = np.asarray(goal, np.float32)
    node_field = np.exp(
        -np.square(graph.nodes[:, :2] - goal[None]).sum(axis=1)
        / (2.0 * LANDING_SIGMA**2)
    ).astype(np.float32)
    values = np.zeros((int(horizon) + 1, graph.node_count), np.float32)
    coverage = []
    for remaining in range(1, int(horizon) + 1):
        endpoint = graph.endpoint_index
        recorded_arrival = first_occupancy_backup(
            node_field[endpoint], values[remaining - 1, endpoint]
        )
        recorded_q = graph.edge_reliability * recorded_arrival
        row = np.zeros(graph.node_count, np.float32)
        np.maximum.at(row, graph.source_index, recorded_q)

        bridge_arrival = first_occupancy_backup(
            node_field[targets], values[remaining - 1, targets]
        )
        bridge_q = weight * bridge_arrival
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = np.clip(row, 0.0, 1.0)
        coverage.append({
            "remaining": remaining,
            "positive_nodes": int(np.sum(row > 0)),
            "positive_fraction": float(np.mean(row > 0)),
            "meaningful_nodes": int(np.sum(row > 1e-6)),
            "meaningful_fraction": float(np.mean(row > 1e-6)),
            "maximum": float(row.max()),
        })
    return values, coverage, node_field


class FirstOccupancyBellmanPlanner(IntegratedBellmanPlanner):
    method_revision = METHOD_REVISION
    planner_name = "first_occupancy_integrated_bellman"

    def _backup(self, immediate, continuation):
        return first_occupancy_backup(immediate, continuation)

    def _set_goal(self, fields):
        key = tuple(np.asarray(fields.goal, np.float32))
        if key != self.goal_key:
            self.values, self.coverage, self.node_field = (
                first_occupancy_stitched_values(
                    self.graph,
                    self.targets,
                    self.weight,
                    fields.goal,
                    horizon=self.horizon,
                )
            )
            self.goal_key = key


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "first_occupancy_integrated_bellman_r18"
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

    planner, hashes = _load_planner(args, FirstOccupancyBellmanPlanner)
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
        "evaluation_arm": "first_occupancy_integrated_bellman",
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


def _r14_rows(args, task_ids):
    rows = json.loads((
        expanded_output_paths(args.root)["directory"]
        / "r10_partitioned_alias_advance_r14_long3"
        / "summary_all15.json"
    ).read_text())["rows"]
    by_id = {int(row["task_id"]): row for row in rows}
    return [by_id[int(task_id)] for task_id in task_ids]


def _summarize(args, task_ids, filename, claim_level):
    output = output_directory(args.root)
    rows = []
    for task_id in task_ids:
        task_path = output / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows.append(json.loads(task_path.read_text()))
    r14 = _r14_rows(args, task_ids)
    successes = int(sum(row["success"] for row in rows))
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": claim_level,
        "declared_deviations": [
            "consumed_development_goals",
            "bounded_first_occupancy_objective",
            "deterministic_spatial_mixture_component_mean_quadrature",
            "conditional_mean_posture_per_spatial_component",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": [int(task_id) for task_id in task_ids],
        "successes": successes,
        "success_rate": float(successes / len(rows)),
        "r14_target_successes": int(sum(row["success"] for row in r14)),
        "paired_vs_r14": {
            "gained_task_ids": [
                int(row["task_id"])
                for row, reference in zip(rows, r14)
                if row["success"] and not reference["success"]
            ],
            "lost_task_ids": [
                int(row["task_id"])
                for row, reference in zip(rows, r14)
                if not row["success"] and reference["success"]
            ],
        },
        "selected_command_calibration": _calibration(output, rows),
        "value_range_contract": [0.0, 1.0],
        "endpoint_random_draws": 0,
        "alias_suppression_used": False,
        "alias_distance_test_used": False,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    if tuple(task_ids) == REGRESSION_GATE_TASK_IDS:
        result["gate"] = {
            "target_successes": len(REGRESSION_GATE_TASK_IDS),
            "passed": successes == len(REGRESSION_GATE_TASK_IDS),
            "continue_all15": successes == len(REGRESSION_GATE_TASK_IDS),
        }
    if len(task_ids) == len(ALL_DEVELOPMENT_TASK_IDS):
        references = _r10_reference_rows(args.root)
        result["r10_reference_successes"] = int(
            sum(row["success"] for row in references)
        )
        result["full_target"] = {
            "successes": 11,
            "met": successes >= 11,
        }
    dump(output / filename, result)
    printed = {
        "successes": result["successes"],
        "tasks": result["tasks"],
        "success_rate": result["success_rate"],
        "r14_target_successes": result["r14_target_successes"],
        "paired_vs_r14": result["paired_vs_r14"],
        "selected_command_calibration": result[
            "selected_command_calibration"
        ],
    }
    if "gate" in result:
        printed["gate"] = result["gate"]
    if "full_target" in result:
        printed["full_target"] = result["full_target"]
    print(json.dumps(printed, indent=2))


def command_summarize_gate(args):
    _summarize(
        args,
        REGRESSION_GATE_TASK_IDS,
        "summary_gate.json",
        "consumed_development_regression_gate",
    )


def command_summarize(args):
    _summarize(
        args,
        ALL_DEVELOPMENT_TASK_IDS,
        "summary.json",
        CLAIM_LEVEL,
    )


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
    gate = subparsers.add_parser("summarize-gate")
    gate.set_defaults(func=command_summarize_gate)
    summarize = subparsers.add_parser("summarize")
    summarize.set_defaults(func=command_summarize)
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
