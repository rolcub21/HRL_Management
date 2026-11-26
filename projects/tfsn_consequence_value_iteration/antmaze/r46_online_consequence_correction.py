"""Reward-free online consequence correction on the frozen R39 planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r39_live_posterior_passage as r39,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


r31 = r39.r31
HERE = Path(__file__).resolve().parent
METHOD_REFERENCE = HERE / "R46_ONLINE_CONSEQUENCE_CORRECTION_REFERENCE.md"
PROTOCOL = HERE / "r46_online_consequence_protocol.json"
METHOD_REVISION = "antmaze_r39_online_consequence_correction_r46"
RESULT_DIRECTORY = (
    r31.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "online_consequence_correction_r46"
)


def load_contract():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": (
            "consumed_development_online_consequence_component_gate_only"
        ),
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_online_correction": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "online_outcome": (
            "continuous_landing_kernel_at_observed_endpoint"
        ),
        "online_similarity": (
            "gaussian_in_frozen_standardized_full_state_and_absolute_"
            "command_space"
        ),
        "offline_prior_effective_count_global": 1.0,
        "memory_scope": "one_task_across_physical_episode_resets",
        "early_episode_termination": False,
        "gate_task_ids": [3, 5],
        "episodes_per_task": 5,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R46 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R46 method reference changed")
    return value


PROTOCOL_VALUE = load_contract()
PRIOR_COUNT = float(PROTOCOL_VALUE["offline_prior_effective_count_global"])


def bayesian_local_connection(prior, support, outcome_mass):
    """Update member-wise connection probabilities with local observations."""
    prior = np.asarray(prior, np.float32)
    support = np.asarray(support, np.float32).reshape(-1)
    outcome_mass = np.asarray(outcome_mass, np.float32).reshape(-1)
    if prior.ndim != 2 or prior.shape[0] != len(support):
        raise ValueError("prior must have one row per candidate")
    if len(outcome_mass) != len(support):
        raise ValueError("online masses must align")
    posterior = (
        PRIOR_COUNT * prior + outcome_mass[:, None]
    ) / (PRIOR_COUNT + support[:, None])
    return np.clip(posterior, 0.0, 1.0).astype(np.float32)


class OnlineConsequencePlanner(r39.LivePosteriorPassagePlanner):
    method_revision = METHOD_REVISION
    planner_name = "r39_with_reward_free_online_consequence_correction"

    def reset_online_correction_task(self):
        self.online_starts = []
        self.online_commands = []
        self.online_outcomes = []
        self.pending_start = None
        self.pending_command = None
        self.pending_online_length = 0
        self._online_candidate_batches = []

    def _ingest_pending(self, online_states):
        new_outcome = None
        if (
            self.pending_start is not None
            and len(online_states) > int(self.pending_online_length)
        ):
            endpoint = np.asarray(online_states[-1], np.float32)
            squared = float(np.square(
                endpoint[:2] - self.pending_command
            ).sum())
            outcome = float(np.exp(
                -squared / (2.0 * float(r31.SIGMA) ** 2)
            ))
            self.online_starts.append(self.pending_start.copy())
            self.online_commands.append(self.pending_command.copy())
            self.online_outcomes.append(outcome)
            new_outcome = outcome
        self.pending_start = None
        self.pending_command = None
        return new_outcome

    def _online_mass(self, state, commands):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        if not self.online_starts:
            zeros = np.zeros(len(commands), np.float32)
            return zeros, zeros
        starts = np.asarray(self.online_starts, np.float32)
        observed_commands = np.asarray(self.online_commands, np.float32)
        outcomes = np.asarray(self.online_outcomes, np.float32)
        state = np.asarray(state, np.float32)
        state_distance = np.square(
            (starts - state[None]) / self.graph.state_scale[None]
        ).sum(axis=1, dtype=np.float64)
        command_distance = np.square(
            (
                observed_commands[:, None]
                - commands[None]
            ) / float(r31.SIGMA)
        ).sum(axis=2, dtype=np.float64)
        weight = np.exp(-0.5 * (state_distance[:, None] + command_distance))
        support = weight.sum(axis=0)
        outcome_mass = (weight * outcomes[:, None]).sum(axis=0)
        return support.astype(np.float32), outcome_mass.astype(np.float32)

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
        (_q, connection, member_q, member_connection) = super()._candidate_value(
            state,
            commands,
            target_index,
            remaining,
            reliability,
            seed,
            return_members=True,
        )
        support, outcome_mass = self._online_mass(state, commands)
        corrected_member_connection = bayesian_local_connection(
            member_connection, support, outcome_mass
        )
        continuation = np.divide(
            member_q,
            member_connection,
            out=np.zeros_like(member_q),
            where=member_connection > 0.0,
        )
        corrected_member_q = continuation * corrected_member_connection
        corrected_q = corrected_member_q.min(axis=1)
        corrected_connection = corrected_member_connection.min(axis=1)
        empirical = np.divide(
            outcome_mass,
            support,
            out=np.zeros_like(outcome_mass),
            where=support > 0.0,
        )
        self._online_candidate_batches.append({
            "commands": np.asarray(commands, np.float32).copy(),
            "offline_connection": np.asarray(connection, np.float32).copy(),
            "corrected_connection": corrected_connection.copy(),
            "support": support.copy(),
            "empirical": empirical.copy(),
        })
        if return_members:
            return (
                corrected_q,
                corrected_connection,
                corrected_member_q,
                corrected_member_connection,
            )
        return corrected_q, corrected_connection

    def _selected_online_diagnostic(self, command, selected_connection):
        if not self._online_candidate_batches:
            return {}
        commands = np.concatenate([
            item["commands"] for item in self._online_candidate_batches
        ])
        offline = np.concatenate([
            item["offline_connection"] for item in self._online_candidate_batches
        ])
        corrected = np.concatenate([
            item["corrected_connection"]
            for item in self._online_candidate_batches
        ])
        support = np.concatenate([
            item["support"] for item in self._online_candidate_batches
        ])
        empirical = np.concatenate([
            item["empirical"] for item in self._online_candidate_batches
        ])
        exact = np.flatnonzero(np.all(commands == command[None], axis=1))
        if not len(exact):
            return {}
        local = exact[np.argmin(np.abs(
            corrected[exact] - float(selected_connection)
        ))]
        return {
            "selected_offline_connection_weight": float(offline[local]),
            "selected_online_corrected_connection_weight": float(
                corrected[local]
            ),
            "selected_online_support_mass": float(support[local]),
            "selected_online_empirical_connection": (
                float(empirical[local]) if support[local] > 0.0 else None
            ),
        }

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        if not hasattr(self, "online_starts"):
            self.reset_online_correction_task()
        new_outcome = self._ingest_pending(online_states)
        self._online_candidate_batches = []
        command, diagnostic = super().choose(
            state,
            command_index,
            fields,
            online_states,
            online_trajectories,
        )
        command = np.asarray(command, np.float32)
        diagnostic.update(self._selected_online_diagnostic(
            command, diagnostic["selected_connection_weight"]
        ))
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "online_consequence_observations": int(len(self.online_starts)),
            "new_online_outcome_from_previous_command": (
                float(new_outcome) if new_outcome is not None else None
            ),
            "online_outcome_definition": PROTOCOL_VALUE["online_outcome"],
            "online_offline_prior_effective_count": PRIOR_COUNT,
            "early_episode_termination": False,
        })
        self.pending_start = np.asarray(state, np.float32).copy()
        self.pending_command = command.copy()
        self.pending_online_length = len(online_states)
        return command, diagnostic


def _load_runtime(args):
    graph = r31.verify_graph(args.graph)
    planner, hashes = r31.lazy._load_planner(
        args, OnlineConsequencePlanner
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


def evaluation_directory(output_name="evaluation_gate2"):
    """Return an isolated evaluation directory under the frozen R46 results."""
    output_name = str(output_name)
    if Path(output_name).name != output_name:
        raise ValueError("R46 output name must be one path component")
    return RESULT_DIRECTORY / output_name


def task_result_path(task_id, output_name="evaluation_gate2"):
    return evaluation_directory(output_name) / f"task_{int(task_id):03d}.json"


def _evaluate_task(
    args, planner, executor, env, task_id, output_name="evaluation_gate2"
):
    task = r31.r22._task(args.root, task_id)
    output = evaluation_directory(output_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(task_id, output_name)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), r31.sha256(r31.R30_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(f"existing R46 task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    planner.reset_online_correction_task()
    row = r39.r37.evaluate_with_exact_fields(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        output / f"task_{int(task_id):03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "R39_with_reward_free_online_consequence_correction",
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_observation_model_inputs": ["reward_contact_radius_m"],
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "online_consequence_memory_persisted_across_physical_resets": True,
        "online_consequence_observations": int(len(planner.online_starts)),
        "online_outcome": PROTOCOL_VALUE["online_outcome"],
        "online_similarity": PROTOCOL_VALUE["online_similarity"],
        "offline_prior_effective_count_global": PRIOR_COUNT,
        "early_episode_termination": False,
        "landing_sigma_m": float(r31.SIGMA),
        "acquisition_detection_radius_m": r39.r35.DETECTION_FOOTPRINT_M,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def _task3_avoidance(log_path):
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    first = next((row for row in rows if (
        row["episode"] == 0 and row["command_index"] == 1
    )), None)
    second = next((row for row in rows if (
        row["episode"] == 1 and row["command_index"] == 1
    )), None)
    if first is None or second is None:
        return {
            "task3_episode2_command2_avoids_episode1_command2": None,
            "task3_episode2_command2_distance_from_episode1_command2_m": None,
        }
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


def command_evaluate(args):
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes_per_task"]):
        raise ValueError("R46 gate requires five episodes")
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in PROTOCOL_VALUE["gate_task_ids"]:
            _evaluate_task(args, planner, executor, env, int(task_id))
    finally:
        env.close()
    command_summarize(args)


def command_evaluate_single(args):
    """Rerun one consumed development task for command-level behavior audit."""
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes_per_task"]):
        raise ValueError("R46 single-task audit requires five carried episodes")
    task_id = int(args.task_id)
    if task_id not in PROTOCOL_VALUE["gate_task_ids"]:
        raise ValueError("R46 single-task audit is limited to consumed gate tasks")
    planner, executor, env = _load_runtime(args)
    try:
        row = _evaluate_task(
            args, planner, executor, env, task_id, args.output_name
        )
    finally:
        env.close()

    output = evaluation_directory(args.output_name)
    contact = row["first_contact_episode"]
    successes = list(row["successes"])
    returns = (
        successes[int(contact) + 1:]
        if contact is not None else []
    )
    task3 = (
        _task3_avoidance(output / "task_003.jsonl")
        if task_id == 3 else {}
    )
    result = {
        "schema_version": 1,
        "experiment_id": (
            f"{PROTOCOL_VALUE['experiment_id']}_single_task_{task_id:03d}_audit"
        ),
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "audit_scope": (
            "single_consumed_development_task_behavior_audit_not_a_"
            "performance_estimate"
        ),
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
        "online_outcome": PROTOCOL_VALUE["online_outcome"],
        "online_similarity": PROTOCOL_VALUE["online_similarity"],
        "offline_prior_effective_count_global": PRIOR_COUNT,
        "early_episode_termination": False,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "acquisition_detection_radius_m": row[
            "acquisition_detection_radius_m"
        ],
        **task3,
        "rows": [row],
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "task_id": task_id,
        "goal": row["goal"],
        "successes": successes,
        "first_contact_episode": contact,
        "mean_e2e": result["mean_e2e"],
        "task3_episode2_command2_avoids_episode1_command2": result.get(
            "task3_episode2_command2_avoids_episode1_command2"
        ),
        "task3_episode2_command2_distance_from_episode1_command2_m": (
            result.get(
                "task3_episode2_command2_distance_from_episode1_command2_m"
            )
        ),
        "output": str(output),
    }, indent=2))


def command_summarize(args):
    rows = [
        json.loads(task_result_path(task_id).read_text())
        for task_id in PROTOCOL_VALUE["gate_task_ids"]
    ]
    contacts = [row["first_contact_episode"] for row in rows]
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
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["gate_task_ids"],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contacts
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
        "online_outcome": PROTOCOL_VALUE["online_outcome"],
        "online_similarity": PROTOCOL_VALUE["online_similarity"],
        "offline_prior_effective_count_global": PRIOR_COUNT,
        "early_episode_termination": False,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **_task3_avoidance(evaluation_directory() / "task_003.jsonl"),
        "rows": rows,
    }
    dump(evaluation_directory() / "summary.json", result)
    keys = (
        "tasks", "acquired_tasks", "acquisition_at_1", "acquisition_at_5",
        "pooled_conditional_return", "mean_e2e",
        "acquisition_new_1m_cells_per_command",
        "acquisition_zero_new_1m_cell_fraction",
        "acquisition_revisited_endpoint_fraction",
        "task3_episode2_command2_avoids_episode1_command2",
        "task3_episode2_command2_distance_from_episode1_command2_m",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2))


def command_audit(_args):
    print(json.dumps({
        "method_revision": METHOD_REVISION,
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "base_method_revision": r39.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "online_outcome": PROTOCOL_VALUE["online_outcome"],
        "online_similarity": PROTOCOL_VALUE["online_similarity"],
        "offline_prior_effective_count_global": PRIOR_COUNT,
        "memory_scope": PROTOCOL_VALUE["memory_scope"],
        "early_episode_termination": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
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
    commands.add_parser("audit").set_defaults(func=command_audit)
    evaluate = commands.add_parser("evaluate-gate2")
    evaluate.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    evaluate.set_defaults(func=command_evaluate)
    single = commands.add_parser("evaluate-single")
    single.add_argument(
        "--task-id", type=int,
        choices=tuple(int(x) for x in PROTOCOL_VALUE["gate_task_ids"]),
        default=3,
    )
    single.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    single.add_argument("--output-name", default="audit_task003")
    single.set_defaults(func=command_evaluate_single)
    summarize = commands.add_parser("summarize")
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
