"""Evaluate the exact R6 graph with the factorized R8 consequence model.

The graph, recorded edges, reliability estimates, command proposal, backup,
executor, tasks, resets, and random-seed schedule are unchanged.  Only the
one-step consequence distribution connecting the live state to the recorded
graph is replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import EPISODE_STEPS, LANDING_SIGMA
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, panel, paths

from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    factorized_checkpoint,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
    RecordedTransitionPlanner,
    sha256_file,
)


METHOD_REVISION = (
    "antmaze_factorized_recorded_transition_value_iteration_r9"
)


def verify_graph_compatibility(graph, consequence):
    """Check that R9 changes the live model, not the empirical graph."""
    if graph.metadata.get("transition_model_inside_graph") is not False:
        raise ValueError("graph contains learned transitions")
    if graph.metadata.get("predicted_endpoint_projection_inside_graph") is not False:
        raise ValueError("graph contains predicted endpoint projection")
    expected_scale = consequence.input_std[:len(graph.state_scale)].copy()
    expected_scale[:2] = LANDING_SIGMA
    if not np.array_equal(graph.state_scale, expected_scale):
        maximum = float(np.max(np.abs(graph.state_scale - expected_scale)))
        raise ValueError(f"graph state scale mismatch: max difference {maximum}")
    if graph.edge_count != 30_000:
        raise ValueError(f"expected 30000 recorded edges, got {graph.edge_count}")


class FactorizedRecordedTransitionPlanner(RecordedTransitionPlanner):
    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        command, diagnostic = super().choose(
            state,
            command_index,
            fields,
            online_states,
            online_trajectories,
        )
        diagnostic["planner"] = (
            "factorized_recorded_transition_value_iteration"
        )
        diagnostic["method_revision"] = METHOD_REVISION
        diagnostic["live_consequence_model"] = (
            "factorized_spatial_then_posture_ensemble"
        )
        return command, diagnostic


def command_evaluate(args):
    graph = RecordedTransitionGraph.load(args.graph)
    checkpoint = factorized_checkpoint(args.root)
    consequence = FactorizedConsequenceEnsemble(checkpoint, args.device)
    verify_graph_compatibility(graph, consequence)

    planner = FactorizedRecordedTransitionPlanner(
        consequence, graph, np.random.default_rng(args.seed)
    )
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    selected = panel(args.root, "development", 9, 37701)
    output = (
        paths(args.root)["eval"]
        / "development"
        / "factorized_recorded_transition_value_iteration_r9"
    )
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=10_000)
    rows = []
    for task in selected["tasks"]:
        planner.rng = np.random.default_rng(
            int(args.seed) + 1_000_003 * int(task["task_id"])
        )
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
        row.update(task)
        rows.append(row)
        print(json.dumps(row), flush=True)
    env.close()

    result = {
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_supplied_goal_diagnostic",
        "changed_component_from_r6": (
            "one_step_live_consequence_model_only"
        ),
        "unchanged_components": [
            "exact_recorded_transition_graph",
            "recorded_edge_reliability",
            "command_proposal",
            "finite_horizon_backup",
            "executor",
            "tasks_and_resets",
            "planner_seed_schedule",
        ],
        "panel": "development",
        "tasks": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "successes": int(sum(row["success"] for row in rows)),
        "mean_steps_to_goal_success": (
            float(np.mean([
                row["steps_to_goal"] for row in rows if row["success"]
            ]))
            if any(row["success"] for row in rows) else None
        ),
        "search_enabled": False,
        "task_gradient_updates": 0,
        "graph": str(Path(args.graph).resolve()),
        "graph_sha256": sha256_file(args.graph),
        "factorized_consequence": str(checkpoint.resolve()),
        "factorized_consequence_sha256": sha256_file(checkpoint),
        "fixed_contract_oracle_reference": {
            "successes": 8,
            "tasks": 9,
        },
        "comparison_references": {
            "direct_executor_successes": 2,
            "behavior_successor_tfsn_successes": 3,
            "joint_model_recorded_graph_r6_successes": 1,
        },
        "fresh_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "successes": result["successes"],
        "tasks": result["tasks"],
        "success_rate": result["success_rate"],
        "mean_steps_to_goal_success": result[
            "mean_steps_to_goal_success"
        ],
    }, indent=2))


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=605)
    parser.set_defaults(func=command_evaluate)
    return parser


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = (
            args.root
            / "model_planning"
            / "round2_recorded_transition_graph_r6.npz"
        )
    args.func(args)


if __name__ == "__main__":
    main()
