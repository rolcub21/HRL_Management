"""R36: choose a reachable member of a posterior hypothesis batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r35_detection_footprint_acquisition as r35,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


r34 = r35.r34
r31 = r35.r31
HERE = Path(__file__).resolve().parent
METHOD_REFERENCE = HERE / "R36_REACHABLE_POSTERIOR_REFERENCE.md"
PROTOCOL = HERE / "r36_reachable_posterior_protocol.json"
EXTENDED_PROTOCOL = HERE / "r36_extended_10ep_protocol.json"
METHOD_REVISION = "antmaze_reachable_posterior_acquisition_r36"
POSTERIOR_HYPOTHESES = 16
RESULT_DIRECTORY = (
    r31.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "reachable_posterior_acquisition_r36"
)


def load_contract():
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r35.METHOD_REVISION,
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_preparation": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "posterior_hypotheses_per_selection_global": POSTERIOR_HYPOTHESES,
        "hypothesis_selection_value": (
            "frozen_R22_supplied_goal_graph_value_at_current_full_state_"
            "projection_and_remaining_horizon"
        ),
        "gate_task_ids": [3, 5],
        "development_task_ids": list(range(15)),
        "episodes_per_task": 5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R36 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R36 method reference changed")
    return protocol


PROTOCOL_VALUE = load_contract()


def load_extended_contract():
    protocol = json.loads(EXTENDED_PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_evaluation_protocol": PROTOCOL_VALUE["experiment_id"],
        "claim_level": "consumed_development_extended_budget_diagnostic",
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_preparation": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "development_task_ids": list(range(15)),
        "episodes_per_task": 10,
        "commands_per_episode": 10,
        "primitive_steps_per_episode": 1000,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R36 extended protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R36 extended method reference changed")
    return protocol


EXTENDED_PROTOCOL_VALUE = load_extended_contract()


def evaluation_protocol(set_name):
    return (
        EXTENDED_PROTOCOL_VALUE
        if set_name == "all15_ep10"
        else PROTOCOL_VALUE
    )


def posterior_batch_indices(mass, count, rng):
    mass = np.asarray(mass, np.float64).reshape(-1)
    if not len(mass) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("posterior mass must be finite and nonnegative")
    positive = np.flatnonzero(mass > 0)
    if not len(positive):
        raise ValueError("posterior mass must have positive support")
    size = min(int(count), len(positive))
    probability = mass[positive] / mass[positive].sum()
    return np.asarray(
        rng.choice(positive, size=size, replace=False, p=probability),
        np.int32,
    )


class ReachablePosteriorPlanner(r35.DetectionFootprintPlanner):
    method_revision = METHOD_REVISION
    planner_name = "reachable_posterior_hidden_goal_r36"

    def reset_posterior_task(self):
        super().reset_posterior_task()
        self.posterior_batch_goals = None
        self.posterior_batch_values = None
        self.posterior_batch_selected_index = None
        self.posterior_batch_projection_distance = None
        self.posterior_batch_remaining = None

    def _sample_posterior_goal(self, fields):
        if not hasattr(self, "_posterior_selection_state"):
            raise RuntimeError("R36 selection requires the current state")
        mass, total, exhausted = r34.r33.posterior_mass(
            self.posterior_support_xy, fields
        )
        selected = posterior_batch_indices(
            mass, POSTERIOR_HYPOTHESES, self.rng
        )
        goals = self.posterior_support_xy[selected]
        state = np.asarray(self._posterior_selection_state, np.float32)
        source, projection_distance = self.graph.project(state[None])
        source = int(source[0])
        remaining = int(self._posterior_selection_remaining)
        scores = []
        results = []
        for goal in goals:
            values, coverage, node_field = r31.lazy.stitched_values(
                self.graph,
                self.targets,
                self.weight,
                goal,
                horizon=self.horizon,
            )
            scores.append(float(values[remaining, source]))
            results.append((values, coverage, node_field))
        winner = int(np.argmax(scores))
        goal_index = int(selected[winner])
        self.posterior_goal = goals[winner].copy()
        self.posterior_probability = float(mass[goal_index] / total)
        self.posterior_residual_mass = float(total)
        self.posterior_positive_cells = int(np.sum(mass > 0.0))
        self.posterior_exhausted = bool(exhausted)
        self.posterior_episode += 1
        self.values, self.coverage, self.node_field = results[winner]
        self.goal_key = tuple(np.asarray(self.posterior_goal, np.float32))
        self.posterior_batch_goals = goals.copy()
        self.posterior_batch_values = np.asarray(scores, np.float32)
        self.posterior_batch_selected_index = winner
        self.posterior_batch_projection_distance = float(
            projection_distance[0]
        )
        self.posterior_batch_remaining = remaining

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        self._posterior_selection_state = np.asarray(state, np.float32).copy()
        self._posterior_selection_remaining = self.horizon - int(command_index)
        command, diagnostic = super().choose(
            state,
            command_index,
            fields,
            online_states,
            online_trajectories,
        )
        if not fields.contacted and self.posterior_batch_values is not None:
            values = self.posterior_batch_values
            diagnostic.update({
                "planner": self.planner_name,
                "method_revision": self.method_revision,
                "posterior_hypothesis_candidates": int(len(values)),
                "posterior_hypothesis_reach_value_selected": float(
                    values[self.posterior_batch_selected_index]
                ),
                "posterior_hypothesis_reach_value_median": float(
                    np.median(values)
                ),
                "posterior_hypothesis_reach_value_maximum": float(
                    values.max()
                ),
                "posterior_hypothesis_projection_distance": (
                    self.posterior_batch_projection_distance
                ),
                "posterior_hypothesis_selection_remaining": (
                    self.posterior_batch_remaining
                ),
                "posterior_hypothesis_selection": (
                    "maximum_frozen_R22_graph_value_in_posterior_batch"
                ),
            })
        return command, diagnostic


def _load_runtime(args):
    graph = r31.verify_graph(args.graph)
    planner, hashes = r31.lazy._load_planner(
        args, ReachablePosteriorPlanner
    )
    geometry = r31.r28.BoundedPassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(r31.SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.configure_posterior_support(
        geometry.grid_xy[geometry.reachable_cell_index]
    )
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": r31.recorded.sha256_file(
            r31.artifact_paths()["landmarks"]
        ),
        "empirical_passage_geometry_sha256": r31.recorded.sha256_file(
            r31.passage_paths()["metadata"]
        ),
        "r30_summary_sha256": r31.sha256(r31.R30_SUMMARY),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def task_ids(set_name):
    if set_name == "gate2":
        return tuple(int(x) for x in PROTOCOL_VALUE["gate_task_ids"])
    if set_name in ("all15", "all15_ep10"):
        return tuple(int(x) for x in PROTOCOL_VALUE["development_task_ids"])
    raise ValueError(set_name)


def evaluation_directory(set_name):
    return RESULT_DIRECTORY / f"evaluation_{set_name}"


def task_result_path(set_name, task_id):
    gate = evaluation_directory("gate2") / f"task_{task_id:03d}.json"
    if set_name == "all15" and task_id in task_ids("gate2") and gate.exists():
        return gate
    return evaluation_directory(set_name) / f"task_{task_id:03d}.json"


def _evaluate_task(args, planner, executor, env, task_id):
    evaluation = evaluation_protocol(args.set_name)
    task = r31.r22._task(args.root, task_id)
    output = evaluation_directory(args.set_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(args.set_name, task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), float(r31.SIGMA),
            r35.DETECTION_FOOTPRINT_M, POSTERIOR_HYPOTHESES,
            r31.sha256(r31.R30_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("landing_sigma_m"),
            saved.get("acquisition_search_sigma_m"),
            saved.get("posterior_hypotheses_per_selection_global"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(
                f"existing R36 task has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.reset_posterior_task()
    row = r31.r28.baseline.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        r31.SIGMA,
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": (
            "reachable_posterior_acquisition_with_frozen_R35"
        ),
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r35.METHOD_REVISION,
        "evaluation_protocol": str(
            EXTENDED_PROTOCOL if args.set_name == "all15_ep10" else PROTOCOL
        ),
        "evaluation_protocol_sha256": r31.sha256(
            EXTENDED_PROTOCOL if args.set_name == "all15_ep10" else PROTOCOL
        ),
        "claim_level": evaluation["claim_level"],
        "declared_deviations": evaluation["declared_deviations"],
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "posterior_prior": r34.PROTOCOL_VALUE["posterior_prior"],
        "posterior_likelihood": (
            "residual_detection_footprint_field_from_actual_visited_states"
        ),
        "within_episode_retention": (
            r34.PROTOCOL_VALUE["within_episode_retention"]
        ),
        "posterior_hypotheses_per_selection_global": POSTERIOR_HYPOTHESES,
        "hypothesis_selection_value": (
            PROTOCOL_VALUE["hypothesis_selection_value"]
        ),
        "posterior_support_cells": int(len(planner.posterior_support_xy)),
        "landing_sigma_m": float(r31.SIGMA),
        "landing_sigma_estimator": r31.SIGMA_ESTIMATOR,
        "acquisition_search_sigma_m": r35.DETECTION_FOOTPRINT_M,
        "acquisition_search_sigma_source": (
            "benchmark_reward_contact_radius"
        ),
        "return_field_sigma_m": float(r31.SIGMA),
        "return_planner": "unchanged_R31_return",
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate_set(args):
    expected_episodes = int(evaluation_protocol(args.set_name)["episodes_per_task"])
    if int(args.episodes) != expected_episodes:
        raise ValueError(
            f"{args.set_name} requires exactly {expected_episodes} episodes"
        )
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    evaluation = evaluation_protocol(args.set_name)
    selected = task_ids(args.set_name)
    rows = [
        json.loads(task_result_path(args.set_name, task_id).read_text())
        for task_id in selected
    ]
    contact = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        first = row["first_contact_episode"]
        if first is not None:
            returns.extend(row["successes"][int(first) + 1:])
    result = {
        "schema_version": 1,
        "experiment_id": evaluation["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r35.METHOD_REVISION,
        "evaluation_protocol": str(
            EXTENDED_PROTOCOL if args.set_name == "all15_ep10" else PROTOCOL
        ),
        "evaluation_protocol_sha256": r31.sha256(
            EXTENDED_PROTOCOL if args.set_name == "all15_ep10" else PROTOCOL
        ),
        "claim_level": evaluation["claim_level"],
        "declared_deviations": evaluation["declared_deviations"],
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "set_name": args.set_name,
        "task_ids": list(selected),
        "tasks": len(rows),
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(x is not None for x in contact)),
        "acquisition_at_1": float(np.mean([x == 0 for x in contact])),
        "acquisition_at_5": float(np.mean([
            x is not None and x < 5 for x in contact
        ])),
        "acquisition_at_10": float(np.mean([
            x is not None and x < 10 for x in contact
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
        "acquisition_two_cycle_count": int(sum(
            row["acquisition_two_cycle_count"] for row in rows
        )),
        "posterior_hypotheses_per_selection_global": POSTERIOR_HYPOTHESES,
        "landing_sigma_m": float(r31.SIGMA),
        "acquisition_search_sigma_m": r35.DETECTION_FOOTPRINT_M,
        "return_field_sigma_m": float(r31.SIGMA),
        "posterior_support_cells": int(rows[0]["posterior_support_cells"]),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    output = evaluation_directory(args.set_name)
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "acquisition_at_10",
            "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count",
            "posterior_hypotheses_per_selection_global",
        )
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(
        func=lambda _args: print(json.dumps({
            "method_revision": METHOD_REVISION,
            "method_reference": str(METHOD_REFERENCE),
            "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
            "base_method_revision": r35.METHOD_REVISION,
            "posterior_hypotheses_per_selection_global": (
                POSTERIOR_HYPOTHESES
            ),
            "landing_sigma_m": r31.SIGMA,
            "acquisition_search_sigma_m": r35.DETECTION_FOOTPRINT_M,
            "return_field_sigma_m": r31.SIGMA,
            "hidden_goal_visible_to_planner_executor_or_models": False,
            "confirmation_panel_accessed": False,
        }, indent=2))
    )
    evaluate = commands.add_parser("evaluate-set")
    evaluate.add_argument(
        "--set-name", choices=("gate2", "all15", "all15_ep10"), required=True
    )
    evaluate.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument(
        "--set-name", choices=("gate2", "all15", "all15_ep10"), required=True
    )
    summarize.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r31.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
