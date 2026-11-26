"""R39: adaptive-greedy exact posterior mass on actual live candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r37_exact_detection_posterior as r37,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r38_route_posterior_mass as r38,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    MobilityStitchedPlanner,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


r36 = r37.r36
r35 = r37.r35
r34 = r37.r34
r31 = r37.r31
HERE = Path(__file__).resolve().parent
METHOD_REFERENCE = HERE / "R39_LIVE_POSTERIOR_PASSAGE_REFERENCE.md"
PROTOCOL = HERE / "r39_live_posterior_passage_protocol.json"
METHOD_REVISION = "antmaze_live_posterior_passage_r39"
RESULT_DIRECTORY = (
    r31.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "live_posterior_passage_r39"
)


def load_contract():
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r37.METHOD_REVISION,
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_preparation": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "posterior_likelihood": r37.PROTOCOL_VALUE["posterior_likelihood"],
        "acquisition_value": (
            "pessimistic_live_connection_probability_times_exact_residual_"
            "posterior_mass_in_empirical_detection_tube"
        ),
        "gate_task_ids": [3, 5],
        "development_task_ids": list(range(15)),
        "episodes_per_task": 5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R39 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R39 method reference changed")
    return protocol


PROTOCOL_VALUE = load_contract()


def normalized_detection_mass(mass, masks):
    mass = np.asarray(mass, np.float64).reshape(-1)
    masks = np.asarray(masks, bool)
    if masks.ndim != 2 or masks.shape[1] != len(mass):
        raise ValueError("candidate masks must index posterior support")
    total = float(mass.sum())
    if total <= 0.0:
        return np.zeros(len(masks), np.float32)
    return (masks @ mass / total).astype(np.float32)


class LivePosteriorPassagePlanner(r31.AlignedBoundedConsumptionPlanner):
    method_revision = METHOD_REVISION
    planner_name = "live_posterior_passage_hidden_goal_r39"

    def configure_hidden_fields(self, landing_sigma):
        if float(landing_sigma) != float(r31.SIGMA):
            raise ValueError("R39 must retain deployed landing precision")
        self.acquisition_search_mode = "exact_detection_posterior"
        self.coarse_search_sigma = r35.DETECTION_FOOTPRINT_M
        self.coverage_reference_xy = None

    def configure_posterior_support(self, support_xy):
        support = np.asarray(support_xy, np.float32).reshape(-1, 2)
        if not len(support) or not np.isfinite(support).all():
            raise ValueError("posterior support must be finite and nonempty")
        if len(np.unique(support, axis=0)) != len(support):
            raise ValueError("posterior support must contain unique cells")
        self.posterior_support_xy = support

    def _live_detection_gain(self, start_xy, endpoint_xy, fields):
        endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
        start = np.broadcast_to(
            np.asarray(start_xy, np.float32).reshape(1, 2), endpoint.shape
        )
        displacement = endpoint - start
        query = np.concatenate((start, displacement), axis=1) / float(
            r31.SIGMA
        )
        _distance, edge = self.passage_geometry.tree.query(
            query, k=1, workers=-1
        )
        edge = np.asarray(edge, np.int32)
        mass, _total, _exhausted = r34.r33.posterior_mass(
            self.posterior_support_xy, fields
        )
        masks = []
        for begin, end, passage_edge in zip(start, endpoint, edge):
            middle = begin[None] + self.passage_geometry.relative[passage_edge]
            polyline = np.concatenate(
                (begin[None], middle, end[None]), axis=0
            )
            masks.append(r38.points_within_polyline(
                self.posterior_support_xy,
                polyline,
                r35.DETECTION_FOOTPRINT_M,
            ))
        return normalized_detection_mass(mass, np.asarray(masks))

    def _candidate_value(
        self,
        state,
        commands,
        target_index,
        remaining,
        reliability,
        seed,
        return_members=False,
    ):
        if not self.acquisition_active:
            return super()._candidate_value(
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=return_members,
            )
        (_unused_q, connection, _unused_members, member_connection) = (
            MobilityStitchedPlanner._candidate_value(
                self,
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=True,
            )
        )
        gain = self._live_detection_gain(
            state[:2], self.graph.nodes[target_index, :2], self.current_fields
        )
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        member_q = (
            member_connection
            * reliability[:, None]
            * gain[:, None]
        )
        q = member_q.min(axis=1)
        self._last_live_gain = gain
        if return_members:
            return q, connection, member_q, member_connection
        return q, connection

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
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
        })
        if not fields.contacted:
            score = float(diagnostic["selected_score"])
            diagnostic.update({
                "phase": "acquisition",
                "acquisition_search_mode": "exact_detection_posterior",
                "acquisition_backup": "adaptive_greedy_live_passage_mass",
                "acquisition_immediate_readout": (
                    "exact_residual_posterior_mass_in_empirical_detection_tube"
                ),
                "selected_expected_posterior_mass_fraction": score,
                "posterior_support_cells": int(len(self.posterior_support_xy)),
            })
        return command, diagnostic


def _load_runtime(args):
    graph = r31.verify_graph(args.graph)
    planner, hashes = r31.lazy._load_planner(
        args, LivePosteriorPassagePlanner
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
    if set_name == "all15":
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
    task = r31.r22._task(args.root, task_id)
    output = evaluation_directory(args.set_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(args.set_name, task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), float(r31.SIGMA),
            r35.DETECTION_FOOTPRINT_M, r31.sha256(r31.R30_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("landing_sigma_m"),
            saved.get("acquisition_detection_radius_m"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(
                f"existing R39 task has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = r37.evaluate_with_exact_fields(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "live_posterior_passage_with_frozen_R37",
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r37.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "posterior_likelihood": PROTOCOL_VALUE["posterior_likelihood"],
        "acquisition_value": PROTOCOL_VALUE["acquisition_value"],
        "posterior_support_cells": int(len(planner.posterior_support_xy)),
        "landing_sigma_m": float(r31.SIGMA),
        "landing_sigma_estimator": r31.SIGMA_ESTIMATOR,
        "acquisition_detection_radius_m": r35.DETECTION_FOOTPRINT_M,
        "acquisition_detection_radius_source": "benchmark_reward_contact_radius",
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
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes_per_task"]):
        raise ValueError("R39 development evaluation requires five episodes")
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
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
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r37.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
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
        "acquisition_value": PROTOCOL_VALUE["acquisition_value"],
        "landing_sigma_m": float(r31.SIGMA),
        "acquisition_detection_radius_m": r35.DETECTION_FOOTPRINT_M,
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
            "acquisition_at_5", "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count",
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
            "base_method_revision": r37.METHOD_REVISION,
            "posterior_likelihood": PROTOCOL_VALUE["posterior_likelihood"],
            "acquisition_value": PROTOCOL_VALUE["acquisition_value"],
            "landing_sigma_m": r31.SIGMA,
            "acquisition_detection_radius_m": r35.DETECTION_FOOTPRINT_M,
            "return_field_sigma_m": r31.SIGMA,
            "hidden_goal_visible_to_planner_executor_or_models": False,
            "confirmation_panel_accessed": False,
        }, indent=2))
    )
    evaluate = commands.add_parser("evaluate-set")
    evaluate.add_argument(
        "--set-name", choices=("gate2", "all15"), required=True
    )
    evaluate.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument(
        "--set-name", choices=("gate2", "all15"), required=True
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
