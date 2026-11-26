"""R51 destination-arrival correction for the current AntMaze planner.

R46/R48 used a Gaussian landing score as the Bernoulli observation for an
edge's connection probability.  That gave partial positive evidence to a
stuck Ant whenever its immobile endpoint was merely near the requested
command.  R51 observes the event the edge weight actually denotes: whether
the executed command ended within the deployed landing scale of the graph
edge's destination.

The module deliberately subclasses the frozen R48 implementation.  Historical
R46/R48 evidence and artifacts remain immutable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "r51_destination_arrival_protocol.json"
REFERENCE = HERE / "R51_DESTINATION_ARRIVAL_REFERENCE.md"
METHOD_REVISION = "antmaze_destination_arrival_online_correction_r51"
RESULT_DIRECTORY = (
    r48.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "destination_arrival_correction_r51/audit_task003_farama"
)
STALLED_PAIR_RESULT_DIRECTORY = (
    r48.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "destination_arrival_correction_r51/audit_stalled_pair_farama"
)
R48_BASE_OUTPUT = (
    r48.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r39_r46_rule_removal_r48_full15"
)


def destination_arrival(endpoint_xy, destination_xy, sigma):
    """Return the exact Bernoulli event represented by a link weight."""
    endpoint = np.asarray(endpoint_xy, np.float32).reshape(2)
    destination = np.asarray(destination_xy, np.float32).reshape(2)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    return float(np.linalg.norm(endpoint - destination) <= sigma)


class DestinationArrivalOnlinePlanner(r48.RuleFreeOnlinePlanner):
    """R48 planner with correctly specified online edge observations."""

    method_revision = METHOD_REVISION
    planner_name = "r48_with_destination_arrival_online_correction"

    def reset_online_correction_task(self):
        super().reset_online_correction_task()
        self.online_destinations = []
        self.pending_destination = None

    def _ingest_pending(self, online_states):
        new_outcome = None
        if (
            self.pending_start is not None
            and len(online_states) > int(self.pending_online_length)
        ):
            if self.pending_destination is None:
                raise RuntimeError("selected graph destination was not recorded")
            endpoint = np.asarray(online_states[-1], np.float32)
            outcome = destination_arrival(
                endpoint[:2], self.pending_destination, r48.SIGMA
            )
            self.online_starts.append(self.pending_start.copy())
            self.online_commands.append(self.pending_command.copy())
            self.online_destinations.append(self.pending_destination.copy())
            self.online_outcomes.append(outcome)
            new_outcome = outcome
        self.pending_start = None
        self.pending_command = None
        self.pending_destination = None
        return new_outcome

    def _online_link_mass(self, state, commands, destinations):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        destinations = np.asarray(destinations, np.float32).reshape(-1, 2)
        if not self.online_starts:
            zeros = np.zeros(len(commands), np.float32)
            return zeros, zeros
        starts = np.asarray(self.online_starts, np.float32)
        observed_commands = np.asarray(self.online_commands, np.float32)
        observed_destinations = np.asarray(
            self.online_destinations, np.float32
        )
        outcomes = np.asarray(self.online_outcomes, np.float32)
        state_distance = np.square(
            (starts - np.asarray(state, np.float32)[None])
            / self.graph.state_scale[None]
        ).sum(axis=1, dtype=np.float64)
        command_distance = np.square(
            (observed_commands[:, None] - commands[None]) / r48.SIGMA
        ).sum(axis=2, dtype=np.float64)
        destination_distance = np.square(
            (
                observed_destinations[:, None] - destinations[None]
            ) / r48.SIGMA
        ).sum(axis=2, dtype=np.float64)
        similarity = np.exp(-0.5 * (
            state_distance[:, None] + command_distance + destination_distance
        ))
        return (
            similarity.sum(axis=0).astype(np.float32),
            (similarity * outcomes[:, None]).sum(axis=0).astype(np.float32),
        )

    def _candidate_value(
        self, state, commands, target_index, remaining, reliability, seed,
        return_members=False,
    ):
        (_q, connection, member_q, member_connection) = (
            r48.RuleFreeR39Planner._candidate_value(
                self, state, commands, target_index, remaining, reliability,
                seed, return_members=True,
            )
        )
        target = np.asarray(target_index, np.int32).reshape(-1)
        destinations = self.graph.nodes[target, :2].astype(
            np.float32, copy=True
        )
        support, outcome = self._online_link_mass(
            state, commands, destinations
        )
        corrected_connection = r48.bayesian_local_connection(
            member_connection, support, outcome
        )
        continuation = np.divide(
            member_q, member_connection, out=np.zeros_like(member_q),
            where=member_connection > 0.0,
        )
        corrected_q = continuation * corrected_connection
        empirical = np.divide(
            outcome, support, out=np.zeros_like(outcome), where=support > 0.0
        )
        self._online_candidate_batches.append({
            "commands": np.asarray(commands, np.float32).copy(),
            "destinations": destinations,
            "offline_connection": np.asarray(connection, np.float32).copy(),
            "corrected_connection": corrected_connection.min(axis=1),
            "support": support,
            "empirical": empirical,
        })
        if return_members:
            return (
                corrected_q.min(axis=1),
                corrected_connection.min(axis=1),
                corrected_q,
                corrected_connection,
            )
        return corrected_q.min(axis=1), corrected_connection.min(axis=1)

    def _selected_destination(self, command, selected_connection):
        if not self._online_candidate_batches:
            raise RuntimeError("no candidate batches were evaluated")
        commands = np.concatenate([
            row["commands"] for row in self._online_candidate_batches
        ])
        corrected = np.concatenate([
            row["corrected_connection"] for row in self._online_candidate_batches
        ])
        destinations = np.concatenate([
            row["destinations"] for row in self._online_candidate_batches
        ])
        exact = np.flatnonzero(np.all(commands == command[None], axis=1))
        if not len(exact):
            raise RuntimeError("selected command is absent from candidate batches")
        index = exact[np.argmin(np.abs(
            corrected[exact] - float(selected_connection)
        ))]
        return destinations[index].copy()

    def choose(
        self, state, command_index, fields, online_states=(),
        online_trajectories=(),
    ):
        command, diagnostic = super().choose(
            state, command_index, fields, online_states, online_trajectories
        )
        command = np.asarray(command, np.float32)
        if diagnostic.get("selected_kind") == "online_positive_evidence":
            # This candidate is appended by the return planner after the graph
            # batches are scored.  Its command coordinate is also its intended
            # destination, so no graph lookup is required.
            self.pending_destination = command.copy()
        else:
            self.pending_destination = self._selected_destination(
                command,
                float(diagnostic["selected_connection_weight"]),
            )
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "online_outcome_definition": (
                "binary_arrival_within_landing_sigma_of_graph_destination"
            ),
            "selected_link_destination_xy": self.pending_destination.tolist(),
        })
        return command, diagnostic


def audit():
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r48.METHOD_REVISION,
        "method_reference": str(REFERENCE.relative_to(HERE.parents[2])),
        "online_outcome": (
            "binary_arrival_within_landing_sigma_of_graph_destination"
        ),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R51 protocol changed: {key}")
    report = {
        **expected,
        "landing_sigma_m": float(r48.SIGMA),
        "pointmaze_shared_implementation": False,
        "pointmaze_r11_changed": False,
        "passed": True,
    }
    print(json.dumps(report, indent=2))
    return report


def _evaluation_contract():
    protocol = json.loads(PROTOCOL.read_text())
    evaluation = protocol.get("evaluation", {})
    expected = {
        "claim_level": "single_consumed_development_task_component_audit",
        "task_id": 3,
        "episodes": 5,
        "executor": "frozen_Farama_18m_open_plane",
        "base_artifacts": "frozen_R48_full15_current_manifest_artifacts",
        "confirmation_panel_accessed": False,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise ValueError(f"R51 evaluation contract changed: {key}")
    return evaluation


def _stalled_pair_contract():
    protocol = json.loads(PROTOCOL.read_text())
    evaluation = protocol.get("stalled_pair_evaluation", {})
    expected = {
        "claim_level": "two_consumed_development_tasks_component_audit",
        "task_ids": [2, 13],
        "selection_rule": (
            "frozen_R48_failed_tasks_tied_for_most_sub_0p03m_commands_"
            "22_of_50"
        ),
        "episodes": 5,
        "executor": "frozen_Farama_18m_open_plane",
        "base_artifacts": "frozen_R48_full15_current_manifest_artifacts",
        "confirmation_panel_accessed": False,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise ValueError(f"R51 stalled-pair contract changed: {key}")
    return evaluation


def _load_runtime(args):
    """Load the frozen R48 artifacts with only the R51 planner substituted."""
    r48.configure_result_directory(args.base_output)
    args.graph = r48.graph_path(args.root)
    graph, artifact, landmark, targets, weight = r48._load_bridges(args)
    consequence = r48.AntConsequenceAdapter(r48.R30_CHECKPOINT, args.device)
    endpoint_mobility = r48.mobility.MobilityModel(
        artifact["mobility"], args.device
    )
    planner = DestinationArrivalOnlinePlanner(
        consequence,
        endpoint_mobility,
        graph,
        landmark,
        targets,
        weight,
        np.random.default_rng(args.seed),
        args.seed,
    )
    geometry = r48.r31.r28.BoundedPassageGeometry(
        args.root, graph, targets, weight
    )
    r48.verify_passage_metadata(args, geometry.metadata)
    planner.configure_hidden_fields(r48.SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        "graph_sha256": r48.sha256(args.graph),
        "factorized_consequence_sha256": r48.sha256(r48.R30_CHECKPOINT),
        "mobility_sha256": r48.sha256(artifact["mobility"]),
        "bridges_sha256": r48.sha256(artifact["bridges"]),
        "reach_landmarks_sha256": r48.sha256(artifact["landmarks"]),
        "empirical_passage_geometry_sha256": r48.sha256(
            r48.passage_paths()["metadata"]
        ),
        "shared_method_manifest_sha256": r48.sha256(r48.MANIFEST_PATH),
    }
    executor = r48.FaramaSACExecutor(
        r48.paths(args.root)["executor"], args.device
    )
    env = r48.make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def _task3_avoidance(log_path):
    rows = [
        json.loads(line)
        for line in Path(log_path).read_text().splitlines()
        if line.strip()
    ]
    first = next(
        (
            row for row in rows
            if row["episode"] == 0 and row["command_index"] == 1
        ),
        None,
    )
    second = next(
        (
            row for row in rows
            if row["episode"] == 1 and row["command_index"] == 1
        ),
        None,
    )
    if first is None or second is None:
        return {}
    first_command = np.asarray(first["command"], np.float32)
    second_command = np.asarray(second["command"], np.float32)
    return {
        "task3_episode2_command2_avoids_episode1_command2": bool(
            not np.array_equal(first_command, second_command)
        ),
        "task3_episode2_command2_distance_from_episode1_command2_m": float(
            np.linalg.norm(second_command - first_command)
        ),
        "task3_episode1_command2": first["command"],
        "task3_episode2_command2": second["command"],
    }


def command_evaluate_task3(args):
    contract = _evaluation_contract()
    if int(args.episodes) != int(contract["episodes"]):
        raise ValueError("R51 task-3 audit requires five carried episodes")
    task_id = int(contract["task_id"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "summary.json"
    task_path = output / "task_003.json"
    log_path = output / "task_003.jsonl"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if (
            saved.get("method_revision") != METHOD_REVISION
            or saved.get("episodes_per_task") != int(args.episodes)
            or saved.get("task_ids") != [task_id]
        ):
            raise ValueError("existing R51 task-3 result has another contract")
        print(json.dumps(saved, indent=2))
        return saved

    planner, executor, env = _load_runtime(args)
    try:
        task = r48.r31.r22._task(args.root, task_id)
        task_seed = int(args.seed) + 1_000_003 * task_id
        planner.rng = np.random.default_rng(task_seed)
        planner.seed = task_seed
        planner.goal_key = None
        planner.reset_online_correction_task()
        row = r48.evaluate_hidden_task(
            env,
            executor,
            planner,
            np.asarray(task["goal"], np.float32),
            task["reset_seed"],
            int(args.episodes),
            log_path,
            r48.PROTOCOL_VALUE["evaluator"][
                "success_radius_m_evaluator_only"
            ],
        )
    finally:
        env.close()

    row.update(task)
    row.update({
        "evaluation_arm": "R48_with_R51_destination_arrival_correction",
        "executor": contract["executor"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r48.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r48.METHOD_REVISION,
        "claim_level": contract["claim_level"],
        "declared_deviations": json.loads(PROTOCOL.read_text())[
            "declared_deviations"
        ],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "landing_sigma_m": float(r48.SIGMA),
        "online_outcome": (
            "binary_arrival_within_landing_sigma_of_graph_destination"
        ),
        "online_similarity": json.loads(PROTOCOL.read_text())[
            "online_similarity"
        ],
        "online_prior_weight": float(r48.RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "online_consequence_observations": int(len(planner.online_starts)),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    r48.dump(task_path, row)

    contact = row["first_contact_episode"]
    successes = list(row["successes"])
    returns = successes[int(contact) + 1:] if contact is not None else []
    outcome = np.asarray(planner.online_outcomes, np.float32)
    result = {
        "schema_version": 1,
        "experiment_id": (
            "antmaze_destination_arrival_online_correction_r51_"
            "task003_farama_audit"
        ),
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r48.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r48.METHOD_REVISION,
        "claim_level": contract["claim_level"],
        "declared_deviations": row["declared_deviations"],
        "oracle_inputs": [],
        "audit_scope": (
            "single_consumed_development_task_component_audit_not_a_"
            "performance_estimate"
        ),
        "executor": contract["executor"],
        "tasks": 1,
        "task_ids": [task_id],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(contact is not None),
        "acquisition_at_1": float(contact == 0),
        "acquisition_at_5": float(contact is not None and contact < 5),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean(successes)),
        "acquisition_new_1m_cells_per_command": row[
            "acquisition_new_1m_cells_per_command"
        ],
        "acquisition_zero_new_1m_cell_fraction": row[
            "acquisition_zero_new_1m_cell_fraction"
        ],
        "acquisition_revisited_endpoint_fraction": row[
            "acquisition_revisited_endpoint_fraction"
        ],
        "acquisition_two_cycle_count": row[
            "acquisition_two_cycle_count"
        ],
        "online_consequence_observations": int(len(outcome)),
        "online_destination_arrival_rate": (
            float(outcome.mean()) if len(outcome) else None
        ),
        "online_outcome": row["online_outcome"],
        "online_similarity": row["online_similarity"],
        "online_prior_weight": row["online_prior_weight"],
        "landing_sigma_m": float(r48.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **_task3_avoidance(log_path),
        "rows": [row],
    }
    r48.dump(result_path, result)
    print(json.dumps({
        key: result.get(key) for key in (
            "tasks",
            "acquired_tasks",
            "acquisition_at_1",
            "acquisition_at_5",
            "pooled_conditional_return",
            "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "online_consequence_observations",
            "online_destination_arrival_rate",
            "task3_episode2_command2_avoids_episode1_command2",
            "task3_episode2_command2_distance_from_episode1_command2_m",
        )
    }, indent=2))
    return result


def _evaluate_followup_task(
    args, planner, executor, env, task_id, output, contract
):
    task_path = output / f"task_{int(task_id):03d}.json"
    log_path = output / f"task_{int(task_id):03d}.jsonl"
    if task_path.exists():
        saved = json.loads(task_path.read_text())
        expected = {
            "task_id": int(task_id),
            "method_revision": METHOD_REVISION,
            "episodes": int(args.episodes),
            "claim_level": contract["claim_level"],
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(
                    f"existing R51 follow-up task has another {key}"
                )
        return saved

    task = r48.r31.r22._task(args.root, task_id)
    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    planner.reset_online_correction_task()
    row = r48.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        log_path,
        r48.PROTOCOL_VALUE["evaluator"]["success_radius_m_evaluator_only"],
    )
    outcome = np.asarray(planner.online_outcomes, np.float32)
    row.update(task)
    row.update({
        "evaluation_arm": "R48_with_R51_destination_arrival_correction",
        "executor": contract["executor"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r48.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r48.METHOD_REVISION,
        "claim_level": contract["claim_level"],
        "selection_rule": contract["selection_rule"],
        "declared_deviations": json.loads(PROTOCOL.read_text())[
            "declared_deviations"
        ],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "landing_sigma_m": float(r48.SIGMA),
        "online_outcome": (
            "binary_arrival_within_landing_sigma_of_graph_destination"
        ),
        "online_similarity": json.loads(PROTOCOL.read_text())[
            "online_similarity"
        ],
        "online_prior_weight": float(r48.RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "online_consequence_observations": int(len(outcome)),
        "online_destination_arrival_rate": (
            float(outcome.mean()) if len(outcome) else None
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    r48.dump(task_path, row)
    return row


def command_evaluate_stalled_pair(args):
    contract = _stalled_pair_contract()
    if int(args.episodes) != int(contract["episodes"]):
        raise ValueError("R51 stalled-pair audit requires five episodes")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if (
            saved.get("method_revision") != METHOD_REVISION
            or saved.get("task_ids") != contract["task_ids"]
            or saved.get("episodes_per_task") != int(args.episodes)
        ):
            raise ValueError("existing R51 stalled-pair summary changed")
        print(json.dumps(saved, indent=2))
        return saved

    planner, executor, env = _load_runtime(args)
    try:
        rows = [
            _evaluate_followup_task(
                args, planner, executor, env, int(task_id), output, contract
            )
            for task_id in contract["task_ids"]
        ]
    finally:
        env.close()

    contacts = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        contact = row["first_contact_episode"]
        if contact is not None:
            returns.extend(row["successes"][int(contact) + 1:])
    result = {
        "schema_version": 1,
        "experiment_id": (
            "antmaze_destination_arrival_online_correction_r51_"
            "stalled_pair_farama_audit"
        ),
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r48.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r48.METHOD_REVISION,
        "claim_level": contract["claim_level"],
        "selection_rule": contract["selection_rule"],
        "declared_deviations": json.loads(PROTOCOL.read_text())[
            "declared_deviations"
        ],
        "oracle_inputs": [],
        "audit_scope": (
            "two_outcome_selected_consumed_development_tasks_component_"
            "audit_not_a_performance_estimate"
        ),
        "executor": contract["executor"],
        "tasks": len(rows),
        "task_ids": contract["task_ids"],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(x is not None for x in contacts)),
        "acquisition_at_1": float(np.mean([x == 0 for x in contacts])),
        "acquisition_at_5": float(np.mean([
            x is not None and x < int(args.episodes) for x in contacts
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
        "online_consequence_observations": int(sum(
            row["online_consequence_observations"] for row in rows
        )),
        "online_destination_arrival_rate": float(np.mean([
            row["online_destination_arrival_rate"] for row in rows
        ])),
        "landing_sigma_m": float(r48.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    r48.dump(summary_path, result)
    print(json.dumps({
        key: result[key] for key in (
            "tasks",
            "task_ids",
            "acquired_tasks",
            "acquisition_at_1",
            "acquisition_at_5",
            "pooled_conditional_return",
            "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "online_destination_arrival_rate",
        )
    }, indent=2))
    return result


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument(
        "--base-output", type=Path, default=R48_BASE_OUTPUT
    )
    value.add_argument("--output", type=Path, default=RESULT_DIRECTORY)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=lambda _args: audit())
    evaluate = commands.add_parser("evaluate-task3")
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.set_defaults(func=command_evaluate_task3)
    stalled_pair = commands.add_parser("evaluate-stalled-pair")
    stalled_pair.add_argument("--episodes", type=int, default=5)
    stalled_pair.set_defaults(func=command_evaluate_stalled_pair)
    return value


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
