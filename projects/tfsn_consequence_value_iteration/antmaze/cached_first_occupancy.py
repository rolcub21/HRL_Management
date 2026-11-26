"""R19: cached first-occupancy planning with one live model lookahead.

The frozen R10 bridge artifact is the goal-independent consequence cache for
recorded graph nodes.  For a supplied goal, bounded first-occupancy values are
computed once for all ten horizons and reused at every physical decision.

At a live state, the factorized consequence model is evaluated exactly once
for the 256 candidate commands.  Every spatial mixture-component mean is
projected with the graph's full 29-dimensional standardized state metric, and
its continuation is read from the cached value table.  A continuous support
factor from that same full-state distance prevents an unsupported endpoint
from inheriting a recorded continuation.  There is no nested consequence
call, coordinate-only snapping, alias rule, or endpoint sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import (
    CANDIDATES,
    EPISODE_STEPS,
    LANDING_SIGMA,
    STATE_DIM,
)
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
from projects.tfsn_consequence_value_iteration.antmaze.first_occupancy_bellman import (
    FirstOccupancyBellmanPlanner,
    _r14_rows,
    first_occupancy_backup,
)
from projects.tfsn_consequence_value_iteration.antmaze.integrated_bellman_planning import (
    _calibration,
    mixture_expectation,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    _load_planner,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


METHOD_REVISION = "antmaze_cached_first_occupancy_r19"
CLAIM_LEVEL = "expanded_consumed_development_planner_diagnostic"
REGRESSION_GATE_TASK_IDS = (0, 4, 8)


def projected_first_occupancy(
    mixture_weight, contact, continuation, projection_support
):
    """Bounded component expectation after full-state value projection."""
    mixture_weight = np.asarray(mixture_weight, np.float32)
    contact = np.asarray(contact, np.float32)
    continuation = np.asarray(continuation, np.float32)
    projection_support = np.asarray(projection_support, np.float32)
    if not (
        mixture_weight.shape
        == contact.shape
        == continuation.shape
        == projection_support.shape
    ):
        raise ValueError("projected first-occupancy arrays must align")
    supported_continuation = projection_support * continuation
    component_value = first_occupancy_backup(
        contact, supported_continuation
    )
    return mixture_expectation(mixture_weight, component_value)


class CachedFirstOccupancyPlanner(FirstOccupancyBellmanPlanner):
    method_revision = METHOD_REVISION
    planner_name = "cached_first_occupancy"
    output_name = "cached_first_occupancy_r19"
    evaluation_arm = "cached_first_occupancy"
    claim_level = CLAIM_LEVEL
    ensemble_aggregation = "uniform_mean"

    def _aggregate_member_values(self, member_q):
        """Combine independently integrated ensemble-member values."""
        return np.asarray(member_q, np.float32).mean(axis=1)

    def _live_candidate_values(
        self, state, commands, remaining, fields
    ):
        state = np.asarray(state, np.float32).reshape(STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        source = np.repeat(state[None], len(commands), axis=0)
        inputs = consequence_inputs(source, commands)
        member_q = []
        member_projection_support = []
        member_projection_distance = []
        member_immediate = []
        component_count = None
        for member in range(len(self.consequence.members)):
            weight, endpoint = (
                self.consequence.spatial_component_mean_endpoints(
                    inputs, member
                )
            )
            component_count = int(weight.shape[1])
            flat_endpoint = endpoint.reshape(-1, STATE_DIM)
            projected, distance = self.graph.project(flat_endpoint)
            projected = projected.reshape(weight.shape)
            distance = distance.reshape(weight.shape)
            support = np.exp(-0.5 * np.square(distance)).astype(np.float32)
            contact = fields.field(
                endpoint[..., :2].reshape(-1, 2)
            ).reshape(weight.shape)
            continuation = self.values[remaining - 1, projected]
            member_q.append(projected_first_occupancy(
                weight, contact, continuation, support
            ))
            member_projection_support.append(
                mixture_expectation(weight, support)
            )
            member_projection_distance.append(
                mixture_expectation(weight, distance)
            )
            member_immediate.append(mixture_expectation(weight, contact))
        member_q = np.stack(member_q, axis=1)
        return (
            self._aggregate_member_values(member_q),
            member_q,
            np.stack(member_immediate, axis=1),
            np.stack(member_projection_support, axis=1),
            np.stack(member_projection_distance, axis=1),
            int(component_count),
        )

    def _selected_endpoint_mobility(self, state, command):
        """Expected learned mobility for one selected command per member."""
        state = np.asarray(state, np.float32).reshape(1, STATE_DIM)
        command = np.asarray(command, np.float32).reshape(1, 2)
        inputs = consequence_inputs(state, command)
        values = []
        for member in range(len(self.consequence.members)):
            weight, endpoint = (
                self.consequence.spatial_component_mean_endpoints(
                    inputs, member
                )
            )
            mobility = self.mobility.predict(
                endpoint.reshape(-1, STATE_DIM)
            ).reshape(weight.shape)
            values.append(float(mixture_expectation(weight, mobility)[0]))
        return values

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        del online_states, online_trajectories
        goal_key = tuple(np.asarray(fields.goal, np.float32))
        graph_value_recomputed = goal_key != self.goal_key
        self._set_goal(fields)
        remaining = self.horizon - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds planning horizon")
        state = np.asarray(state, np.float32).reshape(STATE_DIM)

        _distance, local = self.landmark_tree.query(
            state[:2], k=min(CANDIDATES, len(self.landmark_index))
        )
        landmark_target = self.landmark_index[np.atleast_1d(local)]
        landmark_command = self.graph.nodes[landmark_target, :2]
        recorded_command, recorded_edge, proposal = self.graph.sample_commands(
            self.rng, state, CANDIDATES
        )
        recorded_target = self.graph.endpoint_index[recorded_edge]
        command = np.concatenate((landmark_command, recorded_command), axis=0)
        target = np.concatenate((landmark_target, recorded_target), axis=0)
        kind = np.asarray(
            ["landmark_command"] * len(landmark_command)
            + ["recorded_command"] * len(recorded_command)
        )
        (
            score,
            member_q,
            member_immediate,
            member_support,
            member_distance,
            components,
        ) = self._live_candidate_values(state, command, remaining, fields)
        winner = int(np.argmax(score))
        top = np.argsort(-score, kind="stable")[:5]
        target_xy = self.graph.nodes[target[winner], :2]
        connection, connection_members = self._connection_probability(
            state, command[winner], target_xy
        )
        member_mobility = self._selected_endpoint_mobility(
            state, command[winner]
        )
        diagnostic = {
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "supplied_goal",
            "remaining": int(remaining),
            "selected_score": float(score[winner]),
            "selected_kind": str(kind[winner]),
            "selected_command_distance": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "selected_intended_target_index": int(target[winner]),
            "selected_intended_target_xy": target_xy.tolist(),
            "selected_predicted_connection_probability": connection,
            "selected_member_connection_probabilities": [
                float(value) for value in connection_members
            ],
            "selected_member_scores": member_q[winner].tolist(),
            "selected_member_immediate": member_immediate[winner].tolist(),
            "selected_member_projection_support": member_support[winner].tolist(),
            "selected_member_projection_distance": member_distance[winner].tolist(),
            "selected_member_predicted_endpoint_mobility": (
                member_mobility
            ),
            "top_scores": score[top].tolist(),
            "top_commands": command[top].tolist(),
            "spatial_components_per_member": int(components),
            "ensemble_members": int(len(self.consequence.members)),
            "endpoint_random_draws": 0,
            "task_time_consequence_depth": 1,
            "graph_value_recomputed_this_decision": bool(
                graph_value_recomputed
            ),
            "graph_consequence_cache": "frozen_r10_bridge_weights",
            "continuation_lookup": "nearest_full_state_with_continuous_support",
            "ensemble_aggregation": self.ensemble_aggregation,
            "posture_at_spatial_mode": "conditional_mixture_mean",
            "alias_suppression_used": False,
            "alias_distance_test_used": False,
            "coordinate_only_projection_used": False,
            "search_enabled": False,
            "acquisition_search_mode": "supplied_goal",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.sigma),
            "coarse_search_weight": 0.0,
            "fine_unsearched_mass": None,
            "coarse_unsearched_mass": None,
        }
        diagnostic.update(proposal)
        return command[winner].copy(), diagnostic


def output_directory(root, planner_class=CachedFirstOccupancyPlanner):
    return (
        expanded_output_paths(root)["directory"]
        / planner_class.output_name
    )


def _evaluate_task(args, planner, executor, env, task):
    output = output_directory(args.root, type(planner))
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task['task_id']:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], planner.method_revision
        )
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
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
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
    row.update({
        "evaluation_arm": planner.evaluation_arm,
        "method_revision": planner.method_revision,
        "claim_level": planner.claim_level,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row))
    return row


def _load_runtime(args, planner_class=CachedFirstOccupancyPlanner):
    planner, hashes = _load_planner(args, planner_class)
    planner.artifact_hashes = hashes
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    return planner, executor, env


def command_task_for(args, planner_class):
    planner, executor, env = _load_runtime(args, planner_class)
    try:
        _evaluate_task(args, planner, executor, env, _task(args.root, args.task_id))
    finally:
        env.close()


def command_evaluate_set_for(args, planner_class):
    task_ids = (
        REGRESSION_GATE_TASK_IDS
        if args.set_name == "gate"
        else ALL_DEVELOPMENT_TASK_IDS
    )
    planner, executor, env = _load_runtime(args, planner_class)
    try:
        for task_id in task_ids:
            _evaluate_task(
                args, planner, executor, env, _task(args.root, task_id)
            )
    finally:
        env.close()
    _summarize(args, task_ids, args.set_name, planner_class)


def command_task(args):
    command_task_for(args, CachedFirstOccupancyPlanner)


def command_evaluate_set(args):
    command_evaluate_set_for(args, CachedFirstOccupancyPlanner)


def _summarize(
    args, task_ids, set_name, planner_class=CachedFirstOccupancyPlanner
):
    output = output_directory(args.root, planner_class)
    rows = []
    for task_id in task_ids:
        path = output / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    r14 = _r14_rows(args, task_ids)
    successes = int(sum(row["success"] for row in rows))
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": planner_class.method_revision,
        "claim_level": (
            "consumed_development_regression_gate"
            if set_name == "gate"
            else planner_class.claim_level
        ),
        "declared_deviations": [
            "consumed_development_goals",
            "bounded_first_occupancy_objective",
            "frozen_goal_independent_r10_bridge_cache",
            "deterministic_spatial_mixture_component_mean_quadrature",
            "full_state_projected_live_continuation",
            f"ensemble_aggregation_{planner_class.ensemble_aggregation}",
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
        "task_time_consequence_depth": 1,
        "graph_value_computations_per_goal": 1,
        "endpoint_random_draws": 0,
        "alias_suppression_used": False,
        "alias_distance_test_used": False,
        "coordinate_only_projection_used": False,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    if set_name == "gate":
        result["gate"] = {
            "target_successes": len(REGRESSION_GATE_TASK_IDS),
            "passed": successes == len(REGRESSION_GATE_TASK_IDS),
            "continue_all15": successes == len(REGRESSION_GATE_TASK_IDS),
        }
        filename = "summary_gate.json"
    else:
        references = _r10_reference_rows(args.root)
        result["r10_reference_successes"] = int(
            sum(row["success"] for row in references)
        )
        result["full_target"] = {
            "successes": 11,
            "met": successes >= 11,
        }
        filename = "summary.json"
    dump(output / filename, result)
    keys = {
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
        keys["gate"] = result["gate"]
    if "full_target" in result:
        keys["full_target"] = result["full_target"]
    print(json.dumps(keys, indent=2))


def command_summarize_for(args, planner_class):
    task_ids = (
        REGRESSION_GATE_TASK_IDS
        if args.set_name == "gate"
        else ALL_DEVELOPMENT_TASK_IDS
    )
    _summarize(args, task_ids, args.set_name, planner_class)


def command_summarize(args):
    command_summarize_for(args, CachedFirstOccupancyPlanner)


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
    evaluate = subparsers.add_parser("evaluate-set")
    evaluate.add_argument("--set-name", choices=("gate", "all15"), required=True)
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--set-name", choices=("gate", "all15"), required=True)
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
