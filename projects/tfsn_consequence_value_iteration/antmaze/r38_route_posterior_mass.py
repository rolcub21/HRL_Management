"""R38: rank exact-posterior hypotheses by route-level detection mass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r37_exact_detection_posterior as r37,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


r36 = r37.r36
r35 = r37.r35
r34 = r37.r34
r31 = r37.r31
HERE = Path(__file__).resolve().parent
METHOD_REFERENCE = HERE / "R38_ROUTE_POSTERIOR_MASS_REFERENCE.md"
PROTOCOL = HERE / "r38_route_posterior_mass_protocol.json"
METHOD_REVISION = "antmaze_route_posterior_mass_r38"
RESULT_DIRECTORY = (
    r31.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "route_posterior_mass_r38"
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
        "posterior_hypotheses_per_selection_global": r36.POSTERIOR_HYPOTHESES,
        "hypothesis_selection_value": (
            "expected_residual_posterior_mass_first_covered_by_frozen_R22_"
            "greedy_route_empirical_detection_tubes"
        ),
        "gate_task_ids": [3, 5],
        "development_task_ids": list(range(15)),
        "episodes_per_task": 5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R38 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R38 method reference changed")
    return protocol


PROTOCOL_VALUE = load_contract()


def points_within_polyline(points, polyline, radius):
    """Exact point-to-segment tube membership for a spatial polyline."""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    polyline = np.asarray(polyline, np.float32).reshape(-1, 2)
    if not len(polyline):
        return np.zeros(len(points), bool)
    if len(polyline) == 1:
        distance2 = np.square(points - polyline[0]).sum(axis=1)
        return distance2 <= float(radius) ** 2
    best = np.full(len(points), np.inf, np.float32)
    for begin, end in zip(polyline[:-1], polyline[1:]):
        delta = end - begin
        denominator = float(np.dot(delta, delta))
        if denominator == 0.0:
            distance2 = np.square(points - begin).sum(axis=1)
        else:
            position = np.clip(
                ((points - begin) @ delta) / denominator, 0.0, 1.0
            )
            projection = begin[None] + position[:, None] * delta[None]
            distance2 = np.square(points - projection).sum(axis=1)
        best = np.minimum(best, distance2)
    return best <= float(radius) ** 2


def expected_first_coverage(mass, passage_masks, reliabilities):
    """Expected mass first covered along a route whose failures terminate it."""
    mass = np.asarray(mass, np.float64).reshape(-1)
    if len(passage_masks) != len(reliabilities):
        raise ValueError("every route passage requires one reliability")
    uncovered = mass > 0.0
    reached = 1.0
    expected = 0.0
    for mask, reliability in zip(passage_masks, reliabilities):
        reliability = float(reliability)
        if not np.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("route reliability must lie in [0, 1]")
        reached *= reliability
        newly_covered = uncovered & np.asarray(mask, bool).reshape(-1)
        expected += reached * float(mass[newly_covered].sum())
        uncovered[newly_covered] = False
    return expected


class RoutePosteriorMassPlanner(r37.ExactDetectionPosteriorPlanner):
    method_revision = METHOD_REVISION
    planner_name = "route_posterior_mass_hidden_goal_r38"

    def configure_empirical_passage(self, geometry):
        super().configure_empirical_passage(geometry)
        width = self.targets.shape[1]
        mapping = np.full(self.targets.size, -1, np.int32)
        for shard in geometry.bridge_shards:
            base = int(shard["begin"]) * width
            mapping[base + shard["active_index"]] = shard[
                "passage_edge_index"
            ]
        self.bridge_passage_edge = mapping.reshape(self.targets.shape)

    def reset_posterior_task(self):
        super().reset_posterior_task()
        self.posterior_batch_route_mass = None
        self.posterior_batch_route_steps = None
        self.posterior_batch_selected_reach_value = None

    def _recorded_polyline(self, edge):
        geometry = self.passage_geometry
        cell = np.asarray(geometry.recorded_cell_index[int(edge)], np.int64)
        local = np.stack(
            (cell // int(geometry.shape[1]), cell % int(geometry.shape[1])),
            axis=1,
        )
        xy = (
            geometry.minimum[None] + local.astype(np.float32) + 0.5
        ) * float(r31.SIGMA)
        source = self.graph.nodes[self.graph.source_index[int(edge)], :2]
        endpoint = self.graph.nodes[self.graph.endpoint_index[int(edge)], :2]
        return np.concatenate((source[None], xy, endpoint[None]), axis=0)

    def _bridge_polyline(self, source, column):
        edge = int(self.bridge_passage_edge[int(source), int(column)])
        begin = self.graph.nodes[int(source), :2]
        endpoint = self.graph.nodes[self.targets[int(source), int(column)], :2]
        if edge < 0:
            return np.stack((begin, endpoint), axis=0)
        middle = begin[None] + self.passage_geometry.relative[edge]
        return np.concatenate((begin[None], middle, endpoint[None]), axis=0)

    def _greedy_route(self, values, node_field, source, remaining):
        node = int(source)
        route = []
        for horizon in range(int(remaining), 0, -1):
            best_q = -np.inf
            best = None
            begin = self.outgoing_offsets[node]
            stop = self.outgoing_offsets[node + 1]
            if begin != stop:
                edge = self.outgoing_order[begin:stop]
                endpoint = self.graph.endpoint_index[edge]
                q = self.graph.edge_reliability[edge] * (
                    node_field[endpoint] + values[horizon - 1, endpoint]
                )
                local = int(np.argmax(q))
                best_q = float(q[local])
                selected = int(edge[local])
                best = (
                    "recorded", selected,
                    int(self.graph.endpoint_index[selected]),
                    float(self.graph.edge_reliability[selected]),
                )
            bridge_q = self.weight[node] * (
                node_field[self.targets[node]]
                + values[horizon - 1, self.targets[node]]
            )
            column = int(np.argmax(bridge_q))
            if float(bridge_q[column]) > best_q:
                best = (
                    "bridge", column, int(self.targets[node, column]),
                    float(self.weight[node, column]),
                )
                best_q = float(bridge_q[column])
            if best is None or best_q <= 0.0:
                break
            kind, index, destination, reliability = best
            polyline = (
                self._recorded_polyline(index)
                if kind == "recorded"
                else self._bridge_polyline(node, index)
            )
            route.append((polyline, reliability))
            node = destination
        return route

    def _route_posterior_fraction(self, route, mass, total):
        support = self.posterior_support_xy
        masks = [
            points_within_polyline(
                support, polyline, r35.DETECTION_FOOTPRINT_M
            )
            for polyline, _reliability in route
        ]
        expected = expected_first_coverage(
            mass, masks, [reliability for _polyline, reliability in route]
        )
        return float(expected / float(total)) if total > 0.0 else 0.0

    def _sample_posterior_goal(self, fields):
        if not hasattr(self, "_posterior_selection_state"):
            raise RuntimeError("R38 selection requires the current state")
        mass, total, exhausted = r34.r33.posterior_mass(
            self.posterior_support_xy, fields
        )
        selected = r36.posterior_batch_indices(
            mass, r36.POSTERIOR_HYPOTHESES, self.rng
        )
        goals = self.posterior_support_xy[selected]
        state = np.asarray(self._posterior_selection_state, np.float32)
        source, projection_distance = self.graph.project(state[None])
        source = int(source[0])
        remaining = int(self._posterior_selection_remaining)
        route_mass, reach_scores, route_steps, results = [], [], [], []
        for goal in goals:
            values, coverage, node_field = r31.lazy.stitched_values(
                self.graph,
                self.targets,
                self.weight,
                goal,
                horizon=self.horizon,
            )
            route = self._greedy_route(
                values, node_field, source, remaining
            )
            route_mass.append(
                self._route_posterior_fraction(route, mass, total)
            )
            reach_scores.append(float(values[remaining, source]))
            route_steps.append(len(route))
            results.append((values, coverage, node_field))
        # Route posterior mass is the method score.  Reach value only provides
        # a deterministic tie break between exactly equal route scores.
        order = np.lexsort((np.asarray(reach_scores), np.asarray(route_mass)))
        winner = int(order[-1])
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
        self.posterior_batch_values = np.asarray(reach_scores, np.float32)
        self.posterior_batch_selected_index = winner
        self.posterior_batch_projection_distance = float(projection_distance[0])
        self.posterior_batch_remaining = remaining
        self.posterior_batch_route_mass = np.asarray(route_mass, np.float32)
        self.posterior_batch_route_steps = np.asarray(route_steps, np.int32)
        self.posterior_batch_selected_reach_value = float(reach_scores[winner])

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
        if (
            not fields.contacted
            and self.posterior_batch_route_mass is not None
        ):
            score = self.posterior_batch_route_mass
            winner = self.posterior_batch_selected_index
            diagnostic.update({
                "planner": self.planner_name,
                "method_revision": self.method_revision,
                "posterior_hypothesis_selection": (
                    "maximum_expected_exact_detection_mass_of_frozen_R22_route"
                ),
                "posterior_route_mass_fraction_selected": float(score[winner]),
                "posterior_route_mass_fraction_median": float(np.median(score)),
                "posterior_route_mass_fraction_maximum": float(score.max()),
                "posterior_route_steps_selected": int(
                    self.posterior_batch_route_steps[winner]
                ),
                "posterior_route_selected_reach_value": (
                    self.posterior_batch_selected_reach_value
                ),
            })
        return command, diagnostic


def _load_runtime(args):
    graph = r31.verify_graph(args.graph)
    planner, hashes = r31.lazy._load_planner(
        args, RoutePosteriorMassPlanner
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
            r35.DETECTION_FOOTPRINT_M, r36.POSTERIOR_HYPOTHESES,
            r31.sha256(r31.R30_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("landing_sigma_m"),
            saved.get("acquisition_detection_radius_m"),
            saved.get("posterior_hypotheses_per_selection_global"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(
                f"existing R38 task has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.reset_posterior_task()
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
        "evaluation_arm": "route_posterior_mass_with_frozen_R37",
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
        "posterior_prior": r34.PROTOCOL_VALUE["posterior_prior"],
        "posterior_likelihood": PROTOCOL_VALUE["posterior_likelihood"],
        "hypothesis_selection_value": PROTOCOL_VALUE[
            "hypothesis_selection_value"
        ],
        "posterior_hypotheses_per_selection_global": r36.POSTERIOR_HYPOTHESES,
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
        raise ValueError("R38 development evaluation requires five episodes")
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
        "posterior_hypotheses_per_selection_global": r36.POSTERIOR_HYPOTHESES,
        "hypothesis_selection_value": PROTOCOL_VALUE[
            "hypothesis_selection_value"
        ],
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
            "base_method_revision": r37.METHOD_REVISION,
            "posterior_likelihood": PROTOCOL_VALUE["posterior_likelihood"],
            "hypothesis_selection_value": PROTOCOL_VALUE[
                "hypothesis_selection_value"
            ],
            "posterior_hypotheses_per_selection_global": (
                r36.POSTERIOR_HYPOTHESES
            ),
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
