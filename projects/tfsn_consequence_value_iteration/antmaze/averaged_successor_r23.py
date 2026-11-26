"""R23: uniform-behavior successor ablation of frozen R22.

All graph, model, field, live-decision, and alias-partition components are the
frozen R22 components.  The only behavioral change is the reduction over the
offline future edge set: uniform mean replaces max.  The live action is still
selected by argmax, and the alias layer still exposes its best advancing
action exactly as in R22.

The requested equations included a failure self-loop.  Frozen R22 does not:
its implemented edge contribution is w * (F(d) + V(d)).  R23 averages that
exact contribution so the paired comparison changes only max to mean.  The
equation mismatch is explicit in every summary rather than silently adding a
second change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    finite_horizon_edge_values,
    value_coverage,
)
from scripts.antmaze.current_tfsn.collect import (
    SuppliedGoalField,
    evaluate_supplied_goal_task,
)
from scripts.antmaze.current_tfsn.constants import EPISODE_STEPS
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import dump, paths


METHOD_REVISION = "antmaze_r22_uniform_successor_ablation_r23"
CLAIM_LEVEL = "expanded_consumed_development_max_vs_mean_ablation"
ALL_TASK_IDS = tuple(range(15))
HORIZON = 10
IMPLEMENTED_EDGE_TERM = "w_e * (F(d_e) + V_{r-1}(d_e))"
REQUESTED_EDGE_TERM = (
    "w_e * (F(d_e) + V_{r-1}(d_e)) + "
    "(1 - w_e) * V_{r-1}(i)"
)


def averaged_stitched_values(graph, targets, weight, goal, horizon=HORIZON):
    goal = np.asarray(goal, np.float32)
    node_field = np.exp(
        -np.square(graph.nodes[:, :2] - goal[None]).sum(axis=1)
        / (2.0 * r22.SIGMA**2)
    ).astype(np.float32)
    values = finite_horizon_edge_values(
        graph,
        targets,
        weight,
        node_field,
        horizon,
        reduction="mean",
    )
    return values, value_coverage(values), node_field


class AveragedSuccessorPlanner(r22.ReachBasedR14Planner):
    method_revision = METHOD_REVISION
    planner_name = "r23_uniform_behavior_successor"

    def _set_goal(self, fields):
        key = tuple(np.asarray(fields.goal, np.float32))
        if key != self.goal_key:
            self.values, self.coverage, self.node_field = (
                averaged_stitched_values(
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
        r22.expanded_output_paths(root)["directory"]
        / "averaged_successor_r23"
    )


def contrast_path(root):
    return output_directory(root) / "reset_value_contrast.json"


def _first_trace_decision(path):
    """Return the first logged command, or None for a zero-command task."""
    for line in path.read_text().splitlines():
        if line.strip():
            return json.loads(line)
    return None


def _load_planner(args, planner_class):
    r22._verify_graph(args)
    planner, hashes = r22.lazy._load_planner(args, planner_class)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": r22.sha256_file(
            r22.artifact_paths(args.root)["landmarks"]
        ),
    }
    return planner


def _reset_contrast_rows(args, planner_class, arm, validate_r22=False):
    planner = _load_planner(args, planner_class)
    env = make_env(max_episode_steps=10_000)
    rows = []
    try:
        for task_id in ALL_TASK_IDS:
            task = r22._task(args.root, task_id)
            task_seed = int(args.seed) + 1_000_003 * int(task_id)
            planner.rng = np.random.default_rng(task_seed)
            planner.seed = task_seed
            planner.goal_key = None
            observation, _ = env.reset(seed=int(task["reset_seed"]))
            state = full_state(observation)
            fields = SuppliedGoalField(r22.SIGMA, task["goal"])
            command, diagnostic = planner.choose(state, 0, fields)
            row = {
                "task_id": int(task_id),
                "arm": arm,
                "candidate_score_best": diagnostic[
                    "candidate_score_best"
                ],
                "candidate_score_median": diagnostic[
                    "candidate_score_median"
                ],
                "best_to_median_ratio": diagnostic[
                    "candidate_score_best_to_median_ratio"
                ],
                "median_is_zero": diagnostic[
                    "candidate_score_median_is_zero"
                ],
                "selected_command": np.asarray(command).tolist(),
            }
            if validate_r22:
                log = (
                    r22.output_directory(args.root)
                    / f"task_{task_id:03d}.jsonl"
                )
                first = _first_trace_decision(log)
                row["frozen_r22_had_first_command"] = first is not None
                if first is None:
                    # The evaluator issues no command when the reset already
                    # satisfies the supplied goal (task 12 in this panel).
                    row["matches_frozen_r22_first_command"] = None
                else:
                    matches = bool(np.allclose(
                        command,
                        np.asarray(first["command"], np.float32),
                        rtol=0.0,
                        atol=1e-6,
                    ))
                    row["matches_frozen_r22_first_command"] = matches
                    if not matches:
                        raise ValueError(
                            "R22 reset contrast changed candidates for "
                            f"task {task_id}"
                        )
            rows.append(row)
    finally:
        env.close()
    return rows


def command_reset_contrast(args):
    output_directory(args.root).mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "contrast_point": "physical_reset_before_first_command",
        "candidate_set_identical": True,
        "live_argmax_identical": True,
        "r22": _reset_contrast_rows(
            args, r22.ReachBasedR14Planner, "r22_max", validate_r22=True
        ),
        "r23": _reset_contrast_rows(
            args, AveragedSuccessorPlanner, "r23_uniform_mean"
        ),
    }
    dump(contrast_path(args.root), result)
    validated = [
        row for row in result["r22"]
        if row["frozen_r22_had_first_command"]
    ]
    print(json.dumps({
        "artifact": str(contrast_path(args.root).resolve()),
        "r22_tasks": len(result["r22"]),
        "r23_tasks": len(result["r23"]),
        "r22_first_commands_validated": len(validated),
        "r22_zero_command_tasks": [
            row["task_id"] for row in result["r22"]
            if not row["frozen_r22_had_first_command"]
        ],
        "r22_first_commands_reproduced": all(
            row["matches_frozen_r22_first_command"] for row in validated
        ),
    }, indent=2))


def _load_runtime(args):
    planner = _load_planner(args, AveragedSuccessorPlanner)
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    return planner, executor, env


def _evaluate_task(args, planner, executor, env, task_id):
    task = r22._task(args.root, task_id)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION, r22.SIGMA
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
        r22.SIGMA,
        max_steps=EPISODE_STEPS,
        log_path=output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "r23_uniform_behavior_successor",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "field_sigma_m": r22.SIGMA,
        "future_edge_reduction": "uniform_mean",
        "live_decision_reduction": "argmax_unchanged",
        "alias_advance_reduction": "argmax_unchanged",
        "implemented_edge_term": IMPLEMENTED_EDGE_TERM,
        "requested_edge_term": REQUESTED_EDGE_TERM,
        "failure_self_loop_added": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_task(args):
    planner, executor, env = _load_runtime(args)
    try:
        _evaluate_task(args, planner, executor, env, args.task_id)
    finally:
        env.close()


def _reference_rows(root):
    direct_original = json.loads((
        paths(root)["eval"] / "development" / "goal_given_reference.json"
    ).read_text())["rows"]
    direct_extension = json.loads((
        r22.expanded_output_paths(root)["direct"] / "summary.json"
    ).read_text())["rows"]
    oracle_original = json.loads((
        paths(root)["eval"]
        / "development"
        / "oracle_fixed_command_contract_r7"
        / "summary.json"
    ).read_text())["rows"]
    oracle_extension = json.loads((
        r22.expanded_output_paths(root)["oracle"] / "summary.json"
    ).read_text())["rows"]
    direct = direct_original + direct_extension
    oracle = oracle_original + oracle_extension
    expected = list(ALL_TASK_IDS)
    if [int(row["task_id"]) for row in direct] != expected:
        raise ValueError("direct reference is not the frozen 15-goal panel")
    if [int(row["task_id"]) for row in oracle] != expected:
        raise ValueError("oracle reference is not the frozen 15-goal panel")
    return direct, oracle


def _contrast_summary(rows, task_ids):
    selected = [rows[index] for index in task_ids]
    ratios = [
        float(row["best_to_median_ratio"])
        for row in selected
        if row["best_to_median_ratio"] is not None
    ]
    return {
        "tasks": len(selected),
        "finite_ratio_tasks": len(ratios),
        "median_best_to_median_ratio": (
            float(np.median(ratios)) if ratios else None
        ),
        "mean_best_to_median_ratio": (
            float(np.mean(ratios)) if ratios else None
        ),
        "zero_median_task_ids": [
            int(row["task_id"]) for row in selected if row["median_is_zero"]
        ],
    }


def command_summarize(args):
    output = output_directory(args.root)
    rows = []
    for task_id in ALL_TASK_IDS:
        path = output / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    if not contrast_path(args.root).exists():
        command_reset_contrast(args)
    contrast = json.loads(contrast_path(args.root).read_text())
    direct, oracle = _reference_rows(args.root)
    r22_summary = json.loads((
        r22.output_directory(args.root) / "summary.json"
    ).read_text())
    r22_rows = r22_summary["rows"]
    detour_ids = [
        task_id for task_id in ALL_TASK_IDS
        if not direct[task_id]["success"] and oracle[task_id]["success"]
    ]
    per_goal = []
    for task_id in ALL_TASK_IDS:
        per_goal.append({
            "task_id": int(task_id),
            "detour_goal": bool(task_id in detour_ids),
            "r23_success": bool(rows[task_id]["success"]),
            "r22_success": bool(r22_rows[task_id]["success"]),
            "direct_success": bool(direct[task_id]["success"]),
            "oracle_success": bool(oracle[task_id]["success"]),
            "r23_reset_best_to_median_ratio": contrast["r23"][task_id][
                "best_to_median_ratio"
            ],
            "r22_reset_best_to_median_ratio": contrast["r22"][task_id][
                "best_to_median_ratio"
            ],
        })
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "requested_equation_has_failure_self_loop_absent_from_frozen_r22",
            "failure_self_loop_omitted_in_r23_to_keep_max_vs_mean_isolated",
        ],
        "oracle_inputs": [],
        "analysis_reference_inputs": [
            "frozen_direct_executor_success",
            "frozen_oracle_fixed_contract_success",
        ],
        "controlled_change": "offline_future_edge_reduction_max_to_uniform_mean",
        "candidate_set_identical": True,
        "live_argmax_identical": True,
        "alias_advance_argmax_identical": True,
        "implemented_edge_term": IMPLEMENTED_EDGE_TERM,
        "requested_edge_term": REQUESTED_EDGE_TERM,
        "failure_self_loop_added": False,
        "all_15": {
            "tasks": 15,
            "r23_successes": int(sum(row["success"] for row in rows)),
            "r22_successes": int(sum(row["success"] for row in r22_rows)),
            "direct_successes": int(sum(row["success"] for row in direct)),
            "oracle_successes": int(sum(row["success"] for row in oracle)),
        },
        "detour": {
            "definition": "direct_executor_failure_and_oracle_success",
            "task_ids": list(detour_ids),
            "tasks": len(detour_ids),
            "r23_successes": int(sum(rows[i]["success"] for i in detour_ids)),
            "r22_successes": int(sum(r22_rows[i]["success"] for i in detour_ids)),
            "direct_successes": int(sum(direct[i]["success"] for i in detour_ids)),
            "oracle_successes": int(sum(oracle[i]["success"] for i in detour_ids)),
        },
        "reset_value_contrast": {
            "r23_all15": _contrast_summary(contrast["r23"], ALL_TASK_IDS),
            "r22_all15": _contrast_summary(contrast["r22"], ALL_TASK_IDS),
            "r23_detour": _contrast_summary(contrast["r23"], detour_ids),
            "r22_detour": _contrast_summary(contrast["r22"], detour_ids),
        },
        "per_goal": per_goal,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "all_15": result["all_15"],
        "detour": result["detour"],
        "reset_value_contrast": result["reset_value_contrast"],
    }, indent=2))


def command_evaluate_all(args):
    if not contrast_path(args.root).exists():
        command_reset_contrast(args)
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in ALL_TASK_IDS:
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


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
    task.add_argument("--task-id", type=int, choices=ALL_TASK_IDS, required=True)
    task.set_defaults(func=command_task)
    subparsers.add_parser("reset-contrast").set_defaults(
        func=command_reset_contrast
    )
    subparsers.add_parser("evaluate-all15").set_defaults(
        func=command_evaluate_all
    )
    subparsers.add_parser("summarize").set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r22.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
