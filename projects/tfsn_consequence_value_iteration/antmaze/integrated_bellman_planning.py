"""Deterministic mixture integration for Bellman-consistent Ant planning.

R16 used one random endpoint draw per consequence-ensemble member inside an
argmax over hundreds of commands.  R17 keeps the Bellman-consistent backup but
integrates every spatial mixture component at its mean, weighted by its exact
mixture probability.  Posture at a spatial component is its conditional
mixture mean, so each continuation is still evaluated from a complete Ant
state.  Ensemble members are averaged uniformly, as a mixture approximation
to the predictive distribution.  There are no random endpoint draws, alias
tests, or suppression rules.
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
from projects.tfsn_consequence_value_iteration.antmaze.bellman_consistent_planning import (
    BellmanConsistentPlanner,
    bellman_endpoint_backup,
)
from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    PANEL_REVISION,
    output_paths as expanded_output_paths,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    LONG_ROUTE_TASK_IDS,
    _load_planner,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    LANDMARK_NEIGHBORS,
    _diagonal_gaussian_circle_probability,
)
from projects.tfsn_consequence_value_iteration.antmaze.oracle_contract_diagnostic import (
    calibration_bins,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


METHOD_REVISION = "antmaze_integrated_bellman_live_endpoints_r17"
CLAIM_LEVEL = "expanded_consumed_development_planner_diagnostic"
PAIR_BATCH_SIZE = 8192


def mixture_expectation(weights, values):
    """Expectation over the last, aligned mixture-component dimension."""
    weights = np.asarray(weights, np.float32)
    values = np.asarray(values, np.float32)
    if weights.shape != values.shape:
        raise ValueError("mixture weights and values must align")
    return (weights * values).sum(axis=-1)


class IntegratedBellmanPlanner(BellmanConsistentPlanner):
    method_revision = METHOD_REVISION
    planner_name = "integrated_bellman_live_endpoints"

    def _backup(self, immediate, continuation):
        """R17's accumulated-occupancy objective."""
        return bellman_endpoint_backup(immediate, continuation)

    def _terminal_graph_values(
        self,
        states,
        commands,
        target_index,
        reliability,
        graph_remaining,
        fields,
        seed,
    ):
        """Integrate second-action outcomes into graph continuation values."""
        del seed
        states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        target_index = np.asarray(target_index, np.int32).reshape(-1)
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        if not (
            len(states) == len(commands) == len(target_index) == len(reliability)
        ):
            raise ValueError("second-action arrays do not align")
        values = np.empty(len(states), np.float32)
        for begin in range(0, len(states), PAIR_BATCH_SIZE):
            stop = min(begin + PAIR_BATCH_SIZE, len(states))
            source = states[begin:stop]
            command = commands[begin:stop]
            target = target_index[begin:stop]
            inputs = consequence_inputs(source, command)
            target_state = self.graph.nodes[target]
            member_q = []
            for member in range(len(self.consequence.members)):
                weight, endpoint = (
                    self.consequence.spatial_component_mean_endpoints(
                        inputs, member
                    )
                )
                immediate = fields.field(
                    endpoint[..., :2].reshape(-1, 2)
                ).reshape(weight.shape)
                if graph_remaining > 0:
                    spatial = np.exp(
                        -np.square(
                            endpoint[..., :2] - target_state[:, None, :2]
                        ).sum(axis=2)
                        / (2.0 * LANDING_SIGMA**2)
                    )
                    endpoint_mobility = self.mobility.predict(
                        endpoint.reshape(-1, STATE_DIM)
                    ).reshape(weight.shape)
                    continuation = (
                        spatial
                        * endpoint_mobility
                        * self.values[graph_remaining, target, None]
                    )
                else:
                    continuation = np.zeros_like(immediate)
                member_q.append(mixture_expectation(
                    weight, self._backup(immediate, continuation)
                ))
            # The ensemble is part of the predictive mixture, not a source of
            # Monte Carlo draws.  Uniform integration follows the declared
            # expectation in the R17 protocol.
            values[begin:stop] = (
                np.stack(member_q, axis=1).mean(axis=1)
                * reliability[begin:stop]
            )
        return values

    def _own_option_values(self, endpoints, remaining, fields, seed):
        """Max over commands evaluated from each component-mean endpoint."""
        endpoints = np.asarray(endpoints, np.float32).reshape(-1, STATE_DIM)
        if remaining <= 0:
            return np.zeros(len(endpoints), np.float32)

        _distance, local = self.landmark_tree.query(
            endpoints[:, :2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        if np.asarray(local).ndim == 1:
            local = np.asarray(local)[:, None]
        landmark_target = self.landmark_index[np.asarray(local, np.int32)]
        landmark_command = self.graph.nodes[landmark_target, :2]

        recorded_commands = []
        recorded_edges = []
        proposal_rng = np.random.default_rng(seed + 37)
        for endpoint in endpoints:
            command, edge, _diagnostic = self.graph.sample_commands(
                proposal_rng, endpoint, CANDIDATES
            )
            recorded_commands.append(command)
            recorded_edges.append(edge)
        recorded_command = np.stack(recorded_commands)
        recorded_edge = np.stack(recorded_edges)
        recorded_target = self.graph.endpoint_index[recorded_edge]

        command = np.concatenate((landmark_command, recorded_command), axis=1)
        target = np.concatenate((landmark_target, recorded_target), axis=1)
        reliability = np.concatenate((
            np.ones(landmark_target.shape, np.float32),
            self.graph.edge_reliability[recorded_edge],
        ), axis=1)
        sources = np.repeat(endpoints[:, None], command.shape[1], axis=1)
        candidate_value = self._terminal_graph_values(
            sources.reshape(-1, STATE_DIM),
            command.reshape(-1, 2),
            target.reshape(-1),
            reliability.reshape(-1),
            graph_remaining=remaining - 1,
            fields=fields,
            seed=seed + 101,
        ).reshape(len(endpoints), command.shape[1])
        return candidate_value.max(axis=1)

    def _bellman_candidate_values(
        self, state, commands, remaining, fields, seed
    ):
        state = np.asarray(state, np.float32).reshape(STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        source = np.repeat(state[None], len(commands), axis=0)
        inputs = consequence_inputs(source, commands)
        member_q = []
        member_immediate = []
        member_own = []
        component_count = None
        for member in range(len(self.consequence.members)):
            weight, endpoint = (
                self.consequence.spatial_component_mean_endpoints(
                    inputs, member
                )
            )
            component_count = int(weight.shape[1])
            immediate = fields.field(
                endpoint[..., :2].reshape(-1, 2)
            ).reshape(weight.shape)
            own = self._own_option_values(
                endpoint.reshape(-1, STATE_DIM),
                remaining=remaining - 1,
                fields=fields,
                seed=seed + member * 2_000_003 + 503,
            ).reshape(weight.shape)
            q = self._backup(immediate, own)
            member_q.append(mixture_expectation(weight, q))
            member_immediate.append(mixture_expectation(weight, immediate))
            member_own.append(mixture_expectation(weight, own))
        member_q = np.stack(member_q, axis=1)
        return (
            member_q.mean(axis=1),
            member_q,
            np.stack(member_immediate, axis=1),
            np.stack(member_own, axis=1),
            int(component_count),
        )

    def _connection_probability(self, state, command, target_xy):
        """Predict probability of landing within sigma of the intended node."""
        state = np.asarray(state, np.float32).reshape(STATE_DIM)
        command = np.asarray(command, np.float32).reshape(1, 2)
        target_xy = np.asarray(target_xy, np.float32).reshape(1, 2)
        inputs = consequence_inputs(state[None], command)
        member_probability = []
        for member in range(len(self.consequence.members)):
            weight, residual, scale = self.consequence.spatial_parameters(
                inputs, member
            )
            mean = command[:, None] + residual
            component_probability = _diagonal_gaussian_circle_probability(
                mean, scale, target_xy, LANDING_SIGMA
            )
            member_probability.append(
                mixture_expectation(weight, component_probability)[0]
            )
        return float(np.mean(member_probability)), member_probability

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        del online_states, online_trajectories
        self._set_goal(fields)
        remaining = self.horizon - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds planning horizon")
        state = np.asarray(state, np.float32).reshape(STATE_DIM)

        _distance, local = self.landmark_tree.query(
            state[:2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
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
        score, member_q, member_immediate, member_own, components = (
            self._bellman_candidate_values(
                state,
                command,
                remaining,
                fields,
                self.seed + command_index * 10_007,
            )
        )
        winner = int(np.argmax(score))
        top = np.argsort(-score, kind="stable")[:5]
        target_xy = self.graph.nodes[target[winner], :2]
        connection, connection_members = self._connection_probability(
            state, command[winner], target_xy
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
            "selected_member_own_option": member_own[winner].tolist(),
            "top_scores": score[top].tolist(),
            "top_commands": command[top].tolist(),
            "spatial_components_per_member": int(components),
            "ensemble_members": int(len(self.consequence.members)),
            "endpoint_random_draws": 0,
            "ensemble_aggregation": "uniform_mean",
            "posture_at_spatial_mode": "conditional_mixture_mean",
            "alias_suppression_used": False,
            "alias_distance_test_used": False,
            "predicted_endpoint_inherits_intended_node_value": False,
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


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "integrated_bellman_live_endpoints_r17"
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

    planner, hashes = _load_planner(args, IntegratedBellmanPlanner)
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
        "evaluation_arm": "integrated_bellman_live_endpoints",
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


def _calibration(output, rows):
    predicted = []
    observed = []
    distance = []
    for row in rows:
        log_path = output / f"task_{int(row['task_id']):03d}.jsonl"
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            probability = event.get(
                "selected_predicted_connection_probability"
            )
            target = event.get("selected_intended_target_xy")
            endpoint = event.get("actual_endpoint")
            if probability is None or target is None or endpoint is None:
                continue
            separation = float(np.linalg.norm(
                np.asarray(endpoint, np.float32)
                - np.asarray(target, np.float32)
            ))
            predicted.append(float(probability))
            distance.append(separation)
            observed.append(separation <= LANDING_SIGMA)
    probability = np.asarray(predicted, np.float64)
    outcome = np.asarray(observed, bool)
    return {
        "commands": int(len(probability)),
        "connection_radius_m": float(LANDING_SIGMA),
        "predicted_mean": float(probability.mean()),
        "observed_rate": float(outcome.mean()),
        "prediction_minus_observed": float(
            probability.mean() - outcome.mean()
        ),
        "brier": float(np.square(probability - outcome).mean()),
        "endpoint_to_intended_target_mean_m": float(np.mean(distance)),
        "endpoint_to_intended_target_median_m": float(np.median(distance)),
        "bins": calibration_bins(probability, outcome),
    }


def _summarize(args, task_ids, name, claim_level):
    output = output_directory(args.root)
    rows = []
    for task_id in task_ids:
        task_path = output / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows.append(json.loads(task_path.read_text()))
    r14_all = json.loads((
        expanded_output_paths(args.root)["directory"]
        / "r10_partitioned_alias_advance_r14_long3"
        / "summary_all15.json"
    ).read_text())["rows"]
    r14_by_id = {int(row["task_id"]): row for row in r14_all}
    r14 = [r14_by_id[int(task_id)] for task_id in task_ids]
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": claim_level,
        "declared_deviations": [
            "consumed_development_goals",
            "deterministic_spatial_mixture_component_mean_quadrature",
            "conditional_mean_posture_per_spatial_component",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": [int(task_id) for task_id in task_ids],
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
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
        "endpoint_random_draws": 0,
        "alias_suppression_used": False,
        "alias_distance_test_used": False,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    if len(task_ids) == len(ALL_DEVELOPMENT_TASK_IDS):
        references = _r10_reference_rows(args.root)
        result["r10_reference_successes"] = int(
            sum(row["success"] for row in references)
        )
    dump(output / name, result)
    print(json.dumps({
        "successes": result["successes"],
        "tasks": result["tasks"],
        "success_rate": result["success_rate"],
        "r14_target_successes": result["r14_target_successes"],
        "paired_vs_r14": result["paired_vs_r14"],
        "selected_command_calibration": result[
            "selected_command_calibration"
        ],
    }, indent=2))


def command_summarize(args):
    _summarize(
        args,
        ALL_DEVELOPMENT_TASK_IDS,
        "summary.json",
        CLAIM_LEVEL,
    )


def command_summarize_long3(args):
    _summarize(
        args,
        LONG_ROUTE_TASK_IDS,
        "summary_long3.json",
        "consumed_development_long_route_diagnostic",
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
    summarize = subparsers.add_parser("summarize")
    summarize.set_defaults(func=command_summarize)
    summarize_long3 = subparsers.add_parser("summarize-long3")
    summarize_long3.set_defaults(func=command_summarize_long3)
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
