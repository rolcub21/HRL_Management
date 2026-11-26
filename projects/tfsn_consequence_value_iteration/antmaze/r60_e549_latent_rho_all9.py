#!/usr/bin/env python3
"""R60 latent-radius memory on the E549/R52/R51 original-nine panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r49_latent_passage_factorial as r49,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT,
    E549_CHECKPOINT_SHA256,
    E549SACExecutor,
    inspect_e549_checkpoint,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    PosteriorFalsifiedError,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r60_e549_latent_rho_all9_protocol.json"
REFERENCE = HERE / "R60_E549_LATENT_RHO_ALL9_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_5m_latent_rho_original9_r60"
EVALUATION_ARM = "latent_rho_with_E549_R52_R51"
RETIRE_SEARCH_POSTERIOR_AT_FIRST_CONTACT = False
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_latent_rho_r60/original9"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "oracle_inputs": [],
        "task_ids": list(range(9)),
        "episodes": 5,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "acquisition_observation_memory": "analytic_latent_radius_posterior",
        "acquisition_passage_source": (
            "locally_matched_reward_free_empirical_passage"
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R60 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R60 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


class E549LatentRhoPlanner(
    r49.FactorialAntPlanner,
    r51.DestinationArrivalOnlinePlanner,
):
    """R49 latent memory with R51 destination-arrival online evidence."""

    method_revision = METHOD_REVISION
    planner_name = EVALUATION_ARM

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
            return r51.DestinationArrivalOnlinePlanner._candidate_value(
                self,
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=return_members,
            )
        del remaining, seed
        target = np.asarray(target_index, np.int32).reshape(-1)
        destinations = self.graph.nodes[target, :2].astype(
            np.float32, copy=True
        )
        member_connection = self._exact_connections(
            state, commands, target_index
        )
        offline_connection = member_connection.min(axis=1)
        support, outcome = self._online_link_mass(
            state, commands, destinations
        )
        corrected_connection = r48.bayesian_local_connection(
            member_connection, support, outcome
        )
        connection = corrected_connection.min(axis=1)
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        gain, passage_member_gain = self._factorial_gain(
            state, commands, target_index
        )
        member_q = corrected_connection * reliability[:, None] * gain[:, None]
        q = member_q.min(axis=1)
        empirical = np.divide(
            outcome,
            support,
            out=np.zeros_like(outcome),
            where=support > 0.0,
        )
        self._online_candidate_batches.append(
            {
                "commands": np.asarray(commands, np.float32).copy(),
                "destinations": destinations,
                "offline_connection": offline_connection,
                "corrected_connection": connection,
                "support": support,
                "empirical": empirical,
            }
        )
        self._last_live_gain = gain
        self._last_passage_member_gain = passage_member_gain
        if return_members:
            return q, connection, member_q, corrected_connection
        return q, connection

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        if fields.contacted:
            return super().choose(
                state,
                command_index,
                fields,
                online_states,
                online_trajectories,
            )
        if not hasattr(self, "online_starts"):
            self.reset_online_correction_task()
        new_outcome = self._ingest_pending(online_states)
        self._online_candidate_batches = []
        command, diagnostic = r48.lazy.LazyAliasAdvancePlanner.choose(
            self,
            state,
            command_index,
            fields,
            online_states,
            online_trajectories,
        )
        command = np.asarray(command, np.float32)
        selected_connection = float(diagnostic["selected_connection_weight"])
        diagnostic.update(
            self._selected_online_diagnostic(command, selected_connection)
        )
        self.pending_destination = self._selected_destination(
            command, selected_connection
        )
        diagnostic.update(self.factorial.diagnostic())
        diagnostic.update(
            {
                "planner": self.planner_name,
                "method_revision": self.method_revision,
                "phase": "acquisition",
                "acquisition_backup": "R39_immediate_adaptive_greedy",
                "advance_prefilter": (
                    "observation_neutral_maximum_reliability"
                ),
                "selected_expected_contact_score": float(
                    diagnostic["selected_score"]
                ),
                "online_consequence_observations": len(self.online_starts),
                "new_online_outcome_from_previous_command": new_outcome,
                "online_offline_prior_effective_count_global": (
                    r48.RECORDED_EVIDENCE_PRIOR_WEIGHT
                ),
                "online_outcome_definition": (
                    "binary_arrival_within_landing_sigma_of_graph_destination"
                ),
                "selected_link_destination_xy": (
                    self.pending_destination.tolist()
                ),
            }
        )
        self.pending_start = np.asarray(state, np.float32).copy()
        self.pending_command = command.copy()
        self.pending_online_length = len(online_states)
        return command, diagnostic


def command_audit(args) -> dict:
    executor = inspect_e549_checkpoint()
    paths = r52.output_paths(args.preparation)
    required = (
        paths["consequence"],
        paths["consequence_summary"],
        paths["graph"],
        paths["runtime"] / "planner/mobility.pt",
        paths["runtime"] / "planner/bridges.npz",
        paths["runtime"] / "empirical_passage/metadata.json",
    )
    panel_path = Path(args.root).resolve() / "panels/development.json"
    panel = json.loads(panel_path.read_text())
    panel_ids = [int(row["task_id"]) for row in panel.get("tasks", [])]
    panel_ok = bool(
        panel.get("name") == "development"
        and int(panel.get("goal_offset", -1)) == 2
        and panel_ids == list(range(9))
    )
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": PROTOCOL_VALUE["episodes"],
        "executor": executor,
        "preparation": str(Path(args.preparation).resolve()),
        "development_panel": str(panel_path),
        "development_panel_is_frozen_original_nine": panel_ok,
        "acquisition_observation_memory": PROTOCOL_VALUE[
            "acquisition_observation_memory"
        ],
        "search_posterior_update_scope": PROTOCOL_VALUE.get(
            "search_posterior_update_scope",
            "all_observed_primitive_consequences",
        ),
        "rho_min": PROTOCOL_VALUE["rho_min"],
        "rho_max": PROTOCOL_VALUE["rho_max"],
        "rho_prior": PROTOCOL_VALUE["rho_prior"],
        "success_radius_read_by_planner": False,
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()}
            for path in required
        ],
        "ready_for_evaluation": bool(
            panel_ok and all(path.is_file() for path in required)
        ),
    }
    print(json.dumps(report, indent=2))
    return report


def load_runtime(args):
    r52.configure_r52_runtime(args.preparation)
    paths = r52.output_paths(args.preparation)
    args.graph = paths["graph"]
    graph, artifact, landmark, targets, weight = r48._load_bridges(args)
    consequence = r48.AntConsequenceAdapter(r48.R30_CHECKPOINT, args.device)
    endpoint_mobility = r48.mobility.MobilityModel(
        artifact["mobility"], args.device
    )
    planner = E549LatentRhoPlanner(
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
    planner.configure_factorial(
        observation_level="latent_radius",
        passage_level="nearest_recorded",
        learned=None,
    )
    planner.artifact_hashes = {
        "graph_sha256": r48.sha256(args.graph),
        "factorized_consequence_sha256": r48.sha256(r48.R30_CHECKPOINT),
        "mobility_sha256": r48.sha256(artifact["mobility"]),
        "bridges_sha256": r48.sha256(artifact["bridges"]),
        "reach_landmarks_sha256": r48.sha256(artifact["landmarks"]),
        "empirical_passage_geometry_sha256": r48.sha256(
            r48.passage_paths()["metadata"]
        ),
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
    }
    executor = E549SACExecutor(E549_CHECKPOINT, args.device)
    env = r48.make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def coverage_cell(xy) -> tuple[int, int]:
    return tuple(int(value) for value in np.floor(np.asarray(xy, np.float32)))


def observe_detection_consequences(
    planner,
    xy,
    rewards,
    *,
    search_active: bool,
) -> None:
    """Observe all outcomes in R60, or only the search prefix in R61.

    R60 retains its historical behavior by default.  Its R61 successor sets
    the module contract flag and retires this search posterior at first
    contact; contact and later outcomes belong to the return field.
    """
    positions = np.asarray(xy, np.float32).reshape(-1, 2)
    outcomes = np.asarray(rewards, np.uint8).reshape(-1)
    if len(positions) != len(outcomes):
        raise ValueError("detection positions and outcomes have different sizes")
    if not RETIRE_SEARCH_POSTERIOR_AT_FIRST_CONTACT:
        planner.observe_detection_consequence(positions, outcomes)
        return
    if not search_active or not len(outcomes):
        return
    contacts = np.flatnonzero(outcomes)
    stop = int(contacts[0]) if len(contacts) else len(outcomes)
    if stop:
        planner.observe_detection_consequence(
            positions[:stop], outcomes[:stop]
        )


def evaluate_hidden_task(
    env,
    executor,
    planner,
    goal,
    reset_seed,
    episodes,
    log_path,
    success_radius,
):
    fields = r48.TaskFields(
        r48.SIGMA, search_mode=planner.acquisition_search_mode
    )
    online, completed = [], []
    contact_episode, successes, logs = None, [], []
    first_step, elapsed = None, 0
    visited, endpoints = set(), []
    acquisition_commands = acquisition_new = acquisition_zero = 0
    acquisition_revisited = acquisition_cycles = 0
    episode_coverage = []
    for episode in range(int(episodes)):
        observation, _ = env.reset(seed=int(reset_seed))
        state = r48.full_state(observation)
        trajectory_memory = [state.copy()]
        reset_reward = np.uint8(
            np.linalg.norm(state[:2] - goal) <= float(success_radius)
        )
        search_active_at_reset = not fields.contacted
        fields.add(state[:2], reset_reward)
        observe_detection_consequences(
            planner,
            state[None, :2],
            [reset_reward],
            search_active=search_active_at_reset,
        )
        visited.add(coverage_cell(state[:2]))
        episode_success = bool(reset_reward)
        episode_commands = episode_new = episode_zero = 0
        for command_index in range(r48.HORIZON):
            if episode_success:
                break
            fields.rebuild()
            phase = "return" if fields.contacted else "acquisition"
            command, diagnostic = planner.choose(
                state,
                command_index,
                fields,
                online,
                completed + [trajectory_memory],
            )
            trajectory, rewards = r48._execute_evaluator_command(
                env, executor, state, command, goal, success_radius
            )
            observe_detection_consequences(
                planner,
                trajectory[1:, :2],
                rewards,
                search_active=phase == "acquisition",
            )
            cells = {coverage_cell(xy) for xy in trajectory[1:, :2]}
            new = cells - visited
            endpoint = coverage_cell(trajectory[-1, :2])
            revisited = endpoint in visited
            cycle = bool(
                len(endpoints) >= 2
                and endpoint == endpoints[-2]
                and endpoint != endpoints[-1]
            )
            if phase == "acquisition":
                acquisition_commands += 1
                acquisition_new += len(new)
                acquisition_zero += int(not new)
                acquisition_revisited += int(revisited)
                acquisition_cycles += int(cycle)
                episode_commands += 1
                episode_new += len(new)
                episode_zero += int(not new)
                endpoints.append(endpoint)
            visited.update(cells)
            for next_state, reward in zip(trajectory[1:], rewards):
                fields.add(next_state[:2], reward)
                online.append(next_state.copy())
                trajectory_memory.append(next_state.copy())
            if first_step is None and rewards.any():
                first_step = elapsed + int(np.flatnonzero(rewards)[0]) + 1
            elapsed += r48.COMMAND_STEPS
            episode_success = bool(rewards.any())
            diagnostic.update(
                {
                    "episode": episode,
                    "command_index": command_index,
                    "command": np.asarray(command).tolist(),
                    "actual_endpoint": trajectory[-1, :2].tolist(),
                    "moved_distance": float(
                        np.linalg.norm(trajectory[-1, :2] - state[:2])
                    ),
                    "reward": episode_success,
                    "new_1m_cells": len(new),
                    "revisited_endpoint_1m_cell": revisited,
                    "acquisition_two_cycle_1m_cell": (
                        cycle if phase == "acquisition" else None
                    ),
                }
            )
            logs.append(diagnostic)
            # Persist diagnostic evidence command by command so an intentional
            # posterior-falsification gate does not discard the path that led
            # to it.  This is reporting-only and does not affect decisions.
            Path(log_path).write_text(
                "\n".join(json.dumps(row) for row in logs) + "\n"
            )
            state = trajectory[-1]
            if episode_success:
                if contact_episode is None:
                    contact_episode = episode
                break
        completed.append(np.asarray(trajectory_memory))
        successes.append(episode_success)
        episode_coverage.append(
            {
                "episode": episode,
                "acquisition_commands": episode_commands,
                "new_1m_cells": episode_new,
                "new_1m_cells_per_command": (
                    float(episode_new / episode_commands)
                    if episode_commands else None
                ),
                "zero_new_1m_cell_fraction": (
                    float(episode_zero / episode_commands)
                    if episode_commands else None
                ),
            }
        )
    Path(log_path).write_text(
        "\n".join(json.dumps(row) for row in logs) + "\n"
    )
    return {
        "successes": successes,
        "first_contact_episode": contact_episode,
        "contact_latency": (
            first_step
            if first_step is not None
            else int(episodes) * r48.HORIZON * r48.COMMAND_STEPS
        ),
        "mean_e2e": float(np.mean(successes)),
        "conditional_return": (
            float(np.mean(successes[contact_episode + 1 :]))
            if contact_episode is not None
            and contact_episode + 1 < int(episodes)
            else None
        ),
        "acquisition_commands": acquisition_commands,
        "acquisition_new_1m_cells": acquisition_new,
        "acquisition_new_1m_cells_per_command": (
            float(acquisition_new / acquisition_commands)
            if acquisition_commands else 0.0
        ),
        "acquisition_zero_new_1m_cell_fraction": (
            float(acquisition_zero / acquisition_commands)
            if acquisition_commands else 0.0
        ),
        "acquisition_revisited_endpoint_fraction": (
            float(acquisition_revisited / acquisition_commands)
            if acquisition_commands else 0.0
        ),
        "acquisition_two_cycle_count": acquisition_cycles,
        "episode_acquisition_coverage": episode_coverage,
    }


def evaluate_task(args, planner, executor, env, task_id: int, output: Path):
    task_path = output / f"task_{task_id:03d}.json"
    log_path = output / f"task_{task_id:03d}.jsonl"
    if task_path.exists():
        saved = json.loads(task_path.read_text())
        expected = {
            "task_id": int(task_id),
            "method_revision": METHOD_REVISION,
            "episodes": int(args.episodes),
            "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        }
        for key, item in expected.items():
            if saved.get(key) != item:
                raise ValueError(f"saved R60 task has another {key}")
        return saved

    task = r48.r31.r22._task(args.root, task_id)
    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    planner.reset_online_correction_task()
    try:
        row = evaluate_hidden_task(
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
    except PosteriorFalsifiedError as error:
        posterior = planner.factorial.posterior.summary()
        failure = {
            "schema_version": 1,
            "method_revision": METHOD_REVISION,
            "task_id": int(task_id),
            "status": "model_falsified",
            "error": str(error),
            "positive_observations": int(posterior.positive_observations),
            "negative_observations": int(posterior.negative_observations),
            "search_posterior_update_scope": PROTOCOL_VALUE.get(
                "search_posterior_update_scope",
                "all_observed_primitive_consequences",
            ),
            **planner.factorial.diagnostic(),
        }
        r48.dump(
            output / f"task_{task_id:03d}_MODEL_FALSIFIED.json", failure
        )
        raise
    outcomes = np.asarray(planner.online_outcomes, np.float32)
    row.update(task)
    row.update(
        {
            "evaluation_arm": EVALUATION_ARM,
            "executor": PROTOCOL_VALUE["executor"],
            "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
            "method_reference": str(REFERENCE),
            "method_reference_sha256": r52.sha256(REFERENCE),
            "method_revision": METHOD_REVISION,
            "base_preparation_revision": r52.METHOD_REVISION,
            "online_correction_revision": r51.METHOD_REVISION,
            "claim_level": PROTOCOL_VALUE["claim_level"],
            "selection_rule": PROTOCOL_VALUE["selection_rule"],
            "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
            "oracle_inputs": [],
            "episodes": int(args.episodes),
            "landing_sigma_m": float(r48.SIGMA),
            "online_outcome": PROTOCOL_VALUE["online_outcome"],
            "search_posterior_update_scope": PROTOCOL_VALUE.get(
                "search_posterior_update_scope",
                "all_observed_primitive_consequences",
            ),
            "online_consequence_observations": int(len(outcomes)),
            "online_destination_arrival_rate": (
                float(outcomes.mean()) if len(outcomes) else None
            ),
            "hidden_goal_visible_to_planner_executor_or_models": False,
            "success_radius_read_by_planner": False,
            "task_reward_used": False,
            "task_gradient_updates": 0,
            "confirmation_panel_accessed": False,
            **planner.factorial.diagnostic(),
            **planner.artifact_hashes,
        }
    )
    r48.dump(task_path, row)
    print(
        json.dumps(
            {
                "task_id": int(task_id),
                "first_contact_episode": row["first_contact_episode"],
                "successes": row["successes"],
                "conditional_return": row["conditional_return"],
                "mean_e2e": row["mean_e2e"],
                "episode_acquisition_coverage": row[
                    "episode_acquisition_coverage"
                ],
                "online_destination_arrival_rate": row[
                    "online_destination_arrival_rate"
                ],
            }
        ),
        flush=True,
    )
    return row


def command_evaluate(args) -> dict:
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes"]):
        raise ValueError("R60 requires the frozen five-episode contract")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R60 requires the frozen R59 evaluation seed")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError(
            "R52 preparation or original-nine panel is incomplete"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if (
            saved.get("method_revision") != METHOD_REVISION
            or saved.get("task_ids") != PROTOCOL_VALUE["task_ids"]
            or saved.get("executor_checkpoint_sha256")
            != E549_CHECKPOINT_SHA256
        ):
            raise ValueError("saved R60 summary has another contract")
        print(json.dumps(saved, indent=2))
        return saved

    planner, executor, env = load_runtime(args)
    try:
        rows = [
            evaluate_task(args, planner, executor, env, task_id, output)
            for task_id in PROTOCOL_VALUE["task_ids"]
        ]
    finally:
        env.close()

    contacts = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        contact = row["first_contact_episode"]
        if contact is not None:
            returns.extend(row["successes"][int(contact) + 1 :])
    observations = int(
        sum(row["online_consequence_observations"] for row in rows)
    )
    arrivals = float(
        sum(
            row["online_consequence_observations"]
            * row["online_destination_arrival_rate"]
            for row in rows
            if row["online_destination_arrival_rate"] is not None
        )
    )
    episode_coverage = []
    for episode in range(int(args.episodes)):
        values = [
            row["episode_acquisition_coverage"][episode]
            for row in rows
        ]
        commands = int(sum(value["acquisition_commands"] for value in values))
        cells = int(sum(value["new_1m_cells"] for value in values))
        episode_coverage.append(
            {
                "episode": episode,
                "acquisition_commands": commands,
                "new_1m_cells": cells,
                "new_1m_cells_per_command": (
                    float(cells / commands) if commands else None
                ),
            }
        )
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "selection_rule": PROTOCOL_VALUE["selection_rule"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "audit_scope": (
            "same_original_nine_consumed_development_tasks_as_R59_not_"
            "confirmation_evidence"
        ),
        "executor": PROTOCOL_VALUE["executor"],
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(
            np.mean(
                [
                    value is not None and value < int(args.episodes)
                    for value in contacts
                ]
            )
        ),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean(successes)),
        "acquisition_new_1m_cells_per_command": float(
            np.mean(
                [row["acquisition_new_1m_cells_per_command"] for row in rows]
            )
        ),
        "acquisition_zero_new_1m_cell_fraction": float(
            np.mean(
                [
                    row["acquisition_zero_new_1m_cell_fraction"]
                    for row in rows
                ]
            )
        ),
        "acquisition_revisited_endpoint_fraction": float(
            np.mean(
                [
                    row["acquisition_revisited_endpoint_fraction"]
                    for row in rows
                ]
            )
        ),
        "episode_acquisition_coverage": episode_coverage,
        "online_consequence_observations": observations,
        "online_destination_arrival_rate": (
            arrivals / observations if observations else None
        ),
        "landing_sigma_m": float(r48.SIGMA),
        "search_posterior_update_scope": PROTOCOL_VALUE.get(
            "search_posterior_update_scope",
            "all_observed_primitive_consequences",
        ),
        "rho_min_data_derived_m": float(
            planner.factorial.posterior.rho_min
        ),
        "rho_max_data_derived_m": float(
            planner.factorial.posterior.rho_max
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    r48.dump(summary_path, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "tasks",
                    "task_ids",
                    "acquired_tasks",
                    "acquisition_at_1",
                    "acquisition_at_5",
                    "pooled_conditional_return",
                    "mean_e2e",
                    "acquisition_new_1m_cells_per_command",
                    "episode_acquisition_coverage",
                    "online_destination_arrival_rate",
                    "rho_min_data_derived_m",
                    "rho_max_data_derived_m",
                )
            },
            indent=2,
        )
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=5)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
