#!/usr/bin/env python3
"""Replay task 8 with exact R51-corrected hub reachability.

R70 retains R69 except that online destination-arrival evidence is generalized
to modeled graph bridges before each hub selection. Full-bank selection is an
exact branch-and-bound argmax, not a fixed-size shortlist.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    plot_r59_hidden_goal_original9 as maze_plot,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r69_e549_dynamic_hub_long_budget as r69,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R70_CORRECTED_HUB_REACHABILITY_REFERENCE.md"
PROTOCOL = HERE / "r70_corrected_hub_reachability_protocol.json"
METHOD_REVISION = "antmaze_r69_task8_corrected_hub_reachability_replay_r70"
TASK_ID = 8
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r69_task8_corrected_hub_reachability_replay_r70"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": (
            "antmaze_destination_arrival_online_correction_r51"
        ),
        "base_evaluation_revision": r69.METHOD_REVISION,
        "task_ids": [TASK_ID],
        "episodes": r69.EPISODES,
        "commands_per_episode": r69.HORIZON,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "corrected_hub_selection": (
            "exact_global_argmax_by_coverage_upper_bound_branch_and_bound"
        ),
        "hub_evaluation_cutoff": None,
        "graph_links_corrected": "model_bridges_only",
        "recorded_edges": "unchanged_exact_preparation_transitions",
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R70 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R70 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def _r69_table_paths() -> dict[str, Path]:
    return r68._artifact_paths(r69.DEFAULT_OUTPUT)


def command_audit(args) -> dict:
    preparation = r52.output_paths(args.preparation)
    table = _r69_table_paths()
    required = [
        preparation["consequence"],
        preparation["graph"],
        preparation["runtime"] / "planner/bridges.npz",
        table["table"],
        table["metadata"],
        table["candidates"],
    ]
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": [TASK_ID],
        "episodes": r69.EPISODES,
        "commands_per_episode": r69.HORIZON,
        "full_state_dimension": 29,
        "corrected_hub_selection": PROTOCOL_VALUE[
            "corrected_hub_selection"
        ],
        "hub_evaluation_cutoff": None,
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()}
            for path in required
        ],
        "ready_for_replay": all(path.is_file() for path in required),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    print(json.dumps(report, indent=2))
    return report


def _bind_r70_evaluation_contract() -> None:
    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE
    r67._score_hubs = r68._score_all_hubs


def _load_r69_runtime(args):
    table_args = copy.copy(args)
    table_args.output = r69.DEFAULT_OUTPUT
    loaded = r69._load_runtime(table_args)
    _bind_r70_evaluation_contract()
    planner = loaded[0]
    planner.log_full_command_states = True
    arrays = r67._graph_arrays(args.preparation)
    planner.hub_score_override = CorrectedHubScorer(
        planner, arrays, loaded[3]
    )
    return loaded


def command_replay(args) -> dict:
    audit = command_audit(args)
    if not audit["ready_for_replay"]:
        raise FileNotFoundError("R52 preparation or R69 reachability is incomplete")
    if int(args.episodes) != r69.EPISODES or int(args.seed) != 607:
        raise ValueError("R70 replays the exact R69 task-8 budget and seed")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / f"task_{TASK_ID:03d}.json"
    if task_path.is_file():
        saved = json.loads(task_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R70 replay has another method revision")
        print(json.dumps(saved, indent=2))
        return saved

    (
        planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_r69_runtime(args)
    try:
        row = r67._evaluate_task(
            args,
            planner,
            executor,
            env,
            TASK_ID,
            output,
            hub_nodes,
            hub_xy,
            reachability,
            nodes,
            float(r48.SIGMA),
        )
    finally:
        env.close()
    log_path = output / f"task_{TASK_ID:03d}.jsonl"
    logs = _load_logs(log_path)
    if not logs or any(
        len(item.get("start_state_full", [])) != 29
        or len(item.get("actual_endpoint_state_full", [])) != 29
        for item in logs
    ):
        raise RuntimeError("R70 replay did not retain every full 29D state")
    row["full_state_logging"] = True
    row["full_state_dimension"] = 29
    row["corrected_hub_selection"] = PROTOCOL_VALUE[
        "corrected_hub_selection"
    ]
    r48.dump(task_path, row)
    print(json.dumps({
        "task_id": TASK_ID,
        "first_contact_episode": row["first_contact_episode"],
        "successes": row["successes"],
        "commands_logged_with_full_state": len(logs),
        "online_destination_arrival_rate": row[
            "online_destination_arrival_rate"
        ],
    }, indent=2))
    return row


def _load_logs(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def corrected_bridge_weights(
    prior_weight: np.ndarray,
    support: np.ndarray,
    outcome_mass: np.ndarray,
) -> np.ndarray:
    """Apply the R51 Bernoulli posterior to stored pessimistic weights."""
    prior = float(RECORDED_EVIDENCE_PRIOR_WEIGHT)
    return np.clip(
        (prior * np.asarray(prior_weight) + np.asarray(outcome_mass))
        / (prior + np.asarray(support)),
        0.0,
        1.0,
    ).astype(np.float32)


def _add_observation_mass(
    support: np.ndarray,
    outcome_mass: np.ndarray,
    graph_nodes: np.ndarray,
    state_scale: np.ndarray,
    bridge_target_xy: np.ndarray,
    start_state: np.ndarray,
    command: np.ndarray,
    destination: np.ndarray,
    outcome: float,
    sigma: float,
) -> None:
    state_distance = np.square(
        (graph_nodes - start_state[None]) / state_scale[None]
    ).sum(axis=1, dtype=np.float64)
    command_distance = np.square(
        (bridge_target_xy - command[None, None]) / sigma
    ).sum(axis=2, dtype=np.float64)
    destination_distance = np.square(
        (bridge_target_xy - destination[None, None]) / sigma
    ).sum(axis=2, dtype=np.float64)
    similarity = (
        np.exp(-0.5 * state_distance)[:, None]
        * np.exp(-0.5 * (command_distance + destination_distance))
    ).astype(np.float32)
    support += similarity
    if float(outcome) != 0.0:
        outcome_mass += similarity * float(outcome)


class CorrectedHubScorer:
    """Exact full-bank hub scorer using generalized R51 bridge updates."""

    def __init__(self, planner, arrays, hub_nodes):
        (
            _paths, _consequence, self.nodes, self.source, self.endpoint,
            self.edge_weight, self.bridge_target, self.bridge_weight,
        ) = arrays
        self.planner = planner
        self.bridge_target_xy = self.nodes[self.bridge_target, :2].astype(
            np.float32
        )
        self.support = np.zeros(self.bridge_weight.shape, np.float32)
        self.outcome_mass = np.zeros(self.bridge_weight.shape, np.float32)
        self.processed_observations = 0
        self.hub_nodes = np.asarray(hub_nodes, np.int32).copy()

    def _synchronize(self, online_states) -> float:
        started = time.perf_counter()
        # Hub scoring precedes planner.choose in the evaluator. Ingest the
        # completed previous command here so hub and route decisions use the
        # same evidence. planner.choose subsequently sees no pending command.
        self.planner._ingest_pending(online_states)
        while self.processed_observations < len(self.planner.online_starts):
            index = self.processed_observations
            _add_observation_mass(
                self.support,
                self.outcome_mass,
                self.nodes,
                self.planner.graph.state_scale,
                self.bridge_target_xy,
                np.asarray(self.planner.online_starts[index], np.float32),
                np.asarray(self.planner.online_commands[index], np.float32),
                np.asarray(
                    self.planner.online_destinations[index], np.float32
                ),
                float(self.planner.online_outcomes[index]),
                float(r48.SIGMA),
            )
            self.processed_observations += 1
        return time.perf_counter() - started

    def __call__(
        self, posterior, hub_xy, offline_reachability, graph, state,
        remaining, online_states,
    ):
        started = time.perf_counter()
        update_seconds = self._synchronize(online_states)
        coverage = r68._all_hub_coverage(posterior, hub_xy)
        current_node = int(graph.project(np.asarray(state)[None])[0][0])
        if len(self.hub_nodes) != len(hub_xy):
            raise ValueError("R70 hub-node contract changed")

        if self.processed_observations == 0:
            reach = np.asarray(
                offline_reachability[:, int(remaining), current_node],
                np.float64,
            )
            score = coverage * reach
            selected = int(np.argmax(score))
            evaluated = 0
            solver_seconds = 0.0
            score_semantics = "complete_exact_cached_offline_equal_to_R51"
        else:
            corrected_weight = corrected_bridge_weights(
                self.bridge_weight, self.support, self.outcome_mass
            )
            reach = np.zeros(len(hub_xy), np.float64)
            score = np.zeros(len(hub_xy), np.float64)
            order = np.argsort(-coverage, kind="stable")
            selected = int(order[0])
            best_score = -1.0
            evaluated = 0
            solve_started = time.perf_counter()
            for index in order:
                # coverage is an exact upper bound because reach <= 1.
                if best_score >= float(coverage[index]):
                    break
                target = int(self.hub_nodes[index])
                value = r67._one_target_layers(
                    target,
                    int(remaining),
                    len(self.nodes),
                    self.source,
                    self.endpoint,
                    self.edge_weight,
                    self.bridge_target,
                    corrected_weight,
                )[int(remaining), current_node]
                reach[index] = float(value)
                score[index] = float(coverage[index]) * float(value)
                evaluated += 1
                if float(score[index]) > best_score:
                    best_score = float(score[index])
                    selected = int(index)
            solver_seconds = time.perf_counter() - solve_started
            score_semantics = (
                "evaluated_exact_corrected_scores_else_zero_lower_bound;"
                "global_argmax_exact_by_coverage_upper_bound"
            )

        offline_selected = float(
            offline_reachability[
                selected, int(remaining), current_node
            ]
        )
        self.planner.hub_selection_extra = {
            "offline_cached_reachability_same_hub": offline_selected,
            "R51_corrected_reachability_same_hub": float(reach[selected]),
            "corrected_hubs_evaluated": int(evaluated),
            "corrected_hub_candidates_total": int(len(hub_xy)),
            "branch_and_bound_rule": (
                "stop_only_when_next_coverage_upper_bound_le_best_exact_score"
            ),
            "hub_evaluation_cutoff": None,
            "score_vector_semantics": score_semantics,
            "online_observations_used_by_hub_scorer": int(
                self.processed_observations
            ),
            "online_bridge_update_seconds": float(update_seconds),
            "corrected_reachability_solve_seconds": float(solver_seconds),
        }
        return (
            selected,
            coverage,
            reach,
            score,
            current_node,
            time.perf_counter() - started,
        )


def _wall_boxes() -> list[tuple[np.ndarray, np.ndarray]]:
    boxes = []
    half = maze_plot.MAZE_SCALE_M / 2.0
    # Shrink by numerical epsilon: touching a boundary is not labelled as
    # crossing the wall interior.
    epsilon = 1e-6
    for row, column in np.argwhere(maze_plot.MAZE_MAP == 1):
        center = maze_plot.maze_cell_center(int(row), int(column))
        boxes.append((center - half + epsilon, center + half - epsilon))
    return boxes


def segment_crosses_wall(start_xy, destination_xy) -> bool:
    start = np.asarray(start_xy, np.float64).reshape(2)
    stop = np.asarray(destination_xy, np.float64).reshape(2)
    delta = stop - start
    for lower, upper in _wall_boxes():
        enter, leave = 0.0, 1.0
        intersects = True
        for dimension in range(2):
            if abs(delta[dimension]) < 1e-12:
                if not (
                    lower[dimension] <= start[dimension] <= upper[dimension]
                ):
                    intersects = False
                    break
                continue
            first = (lower[dimension] - start[dimension]) / delta[dimension]
            second = (upper[dimension] - start[dimension]) / delta[dimension]
            enter = max(enter, min(first, second))
            leave = min(leave, max(first, second))
            if enter > leave:
                intersects = False
                break
        if intersects and leave >= 0.0 and enter <= 1.0:
            return True
    return False


def _all_bridge_wall_mask(
    nodes: np.ndarray, bridge_target: np.ndarray, chunk: int = 512,
) -> np.ndarray:
    output = np.zeros(bridge_target.shape, bool)
    boxes = _wall_boxes()
    for begin in range(0, len(nodes), int(chunk)):
        stop = min(begin + int(chunk), len(nodes))
        start = np.repeat(
            nodes[begin:stop, None, :2], bridge_target.shape[1], axis=1
        ).reshape(-1, 2).astype(np.float64)
        target = nodes[bridge_target[begin:stop], :2].reshape(-1, 2).astype(
            np.float64
        )
        delta = target - start
        crossing = np.zeros(len(start), bool)
        for lower, upper in boxes:
            enter = np.zeros(len(start), np.float64)
            leave = np.ones(len(start), np.float64)
            valid = np.ones(len(start), bool)
            for dimension in range(2):
                moving = np.abs(delta[:, dimension]) >= 1e-12
                stationary_inside = (
                    (start[:, dimension] >= lower[dimension])
                    & (start[:, dimension] <= upper[dimension])
                )
                valid &= moving | stationary_inside
                safe_delta = np.where(moving, delta[:, dimension], 1.0)
                first = (lower[dimension] - start[:, dimension]) / safe_delta
                second = (upper[dimension] - start[:, dimension]) / safe_delta
                enter = np.maximum(enter, np.minimum(first, second))
                leave = np.minimum(leave, np.maximum(first, second))
            crossing |= valid & (enter <= leave) & (leave >= 0.0) & (enter <= 1.0)
        output[begin:stop] = crossing.reshape(stop - begin, -1)
    return output


def _selection_record(
    row: dict,
    corrected_weight: np.ndarray,
    nodes: np.ndarray,
    source: np.ndarray,
    endpoint: np.ndarray,
    edge_weight: np.ndarray,
    bridge_target: np.ndarray,
) -> dict:
    selection = row["hub_selection"]
    target = int(selection["hub_node_index"])
    remaining = int(row["remaining"])
    current = int(selection["current_graph_node"])
    started = time.perf_counter()
    corrected = r67._one_target_layers(
        target,
        remaining,
        len(nodes),
        source,
        endpoint,
        edge_weight,
        bridge_target,
        corrected_weight,
    )[remaining, current]
    offline = float(selection.get(
        "offline_cached_reachability_same_hub",
        selection["remaining_horizon_reachability"],
    ))
    behavioral_corrected = float(selection.get(
        "R51_corrected_reachability_same_hub",
        selection["remaining_horizon_reachability"],
    ))
    return {
        "episode": int(row["episode"]),
        "command_index": int(row["command_index"]),
        "hub_bank_index": int(selection["hub_bank_index"]),
        "hub_node_index": target,
        "hub_xy": selection["hub_xy"],
        "current_graph_node": current,
        "remaining_commands": remaining,
        "offline_cached_reachability": offline,
        "behavioral_R51_corrected_reachability": behavioral_corrected,
        "posthoc_recomputed_R51_corrected_reachability": float(corrected),
        "behavioral_to_posthoc_absolute_error": abs(
            behavioral_corrected - float(corrected)
        ),
        "corrected_to_offline_ratio": float(
            behavioral_corrected / max(offline, 1e-12)
        ),
        "online_observations_before_selection": int(
            int(row["episode"]) * r69.HORIZON + int(row["command_index"])
        ),
        "posthoc_recompute_seconds": time.perf_counter() - started,
    }


def command_analyze(args) -> dict:
    output = Path(args.output).resolve()
    log_path = output / f"task_{TASK_ID:03d}.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError("run the R70 task-8 replay first")
    logs = _load_logs(log_path)
    if any(
        len(row.get("start_state_full", [])) != 29
        or len(row.get("actual_endpoint_state_full", [])) != 29
        for row in logs
    ):
        raise ValueError("R70 analysis requires full 29D replay states")

    r52.configure_r52_runtime(args.preparation)
    (
        paths, consequence, nodes, source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    ) = r67._graph_arrays(args.preparation)
    sigma = float(consequence["landing_sigma_m"])
    graph = r48.ManifestRecordedTransitionGraph.load(paths["graph"])
    bridge_target_xy = nodes[bridge_target, :2].astype(np.float32)
    support = np.zeros(bridge_weight.shape, np.float32)
    outcome_mass = np.zeros(bridge_weight.shape, np.float32)
    selections = []
    failed_links = []
    previous_failed_crossing = None

    for row_index, row in enumerate(logs):
        if row.get("hub_selected_this_command"):
            corrected = corrected_bridge_weights(
                bridge_weight, support, outcome_mass
            )
            record = _selection_record(
                row,
                corrected,
                nodes,
                source,
                endpoint,
                edge_weight,
                bridge_target,
            )
            record["support_mass_mean"] = float(support.mean())
            record["support_mass_maximum"] = float(support.max())
            selections.append(record)

        destination = np.asarray(row["selected_link_destination_xy"], np.float32)
        endpoint_state = np.asarray(
            row["actual_endpoint_state_full"], np.float32
        )
        outcome = float(
            np.linalg.norm(endpoint_state[:2] - destination) <= sigma
        )
        start_state = np.asarray(row["start_state_full"], np.float32)
        command = np.asarray(row["command"], np.float32)
        is_bridge = "bridge" in str(row.get("selected_kind", ""))
        crosses = bool(
            is_bridge and segment_crosses_wall(start_state[:2], destination)
        )
        if is_bridge and outcome == 0.0:
            failed = {
                "row_index": int(row_index),
                "episode": int(row["episode"]),
                "command_index": int(row["command_index"]),
                "selected_kind": row["selected_kind"],
                "start_xy": start_state[:2].astype(float).tolist(),
                "command_xy": command.astype(float).tolist(),
                "destination_xy": destination.astype(float).tolist(),
                "actual_endpoint_xy": endpoint_state[:2].astype(float).tolist(),
                "crosses_wall_diagnostic": crosses,
                "offline_connection_weight": row.get(
                    "selected_offline_connection_weight"
                ),
                "online_corrected_connection_weight": row.get(
                    "selected_online_corrected_connection_weight"
                ),
                "live_R51_support_mass": row.get("selected_online_support_mass"),
            }
            if crosses and previous_failed_crossing is not None:
                failed.update({
                    "source_full_state_standardized_distance_from_previous_failed_crossing": float(
                        np.linalg.norm(
                            (start_state - previous_failed_crossing["state"])
                            / graph.state_scale
                        )
                    ),
                    "command_distance_from_previous_failed_crossing_m": float(
                        np.linalg.norm(command - previous_failed_crossing["command"])
                    ),
                    "destination_distance_from_previous_failed_crossing_m": float(
                        np.linalg.norm(
                            destination - previous_failed_crossing["destination"]
                        )
                    ),
                })
            if crosses:
                previous_failed_crossing = {
                    "state": start_state.copy(),
                    "command": command.copy(),
                    "destination": destination.copy(),
                }
            failed_links.append(failed)

        _add_observation_mass(
            support,
            outcome_mass,
            nodes,
            graph.state_scale,
            bridge_target_xy,
            start_state,
            command,
            destination,
            outcome,
            sigma,
        )

    wall_mask = _all_bridge_wall_mask(nodes, bridge_target)
    crossing_weight = bridge_weight[wall_mask]
    noncrossing_weight = bridge_weight[~wall_mask]
    hub_2415 = [
        row for row in selections if int(row["hub_bank_index"]) == 2415
    ]
    crossing_failures = [
        row for row in failed_links if row["crosses_wall_diagnostic"]
    ]
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_id": TASK_ID,
        "commands": len(logs),
        "full_state_dimension": 29,
        "landing_sigma_m": sigma,
        "selected_hub_only_audit": False,
        "corrected_reachability_changes_hub_behavior": True,
        "corrected_hub_selection": PROTOCOL_VALUE[
            "corrected_hub_selection"
        ],
        "selected_hub_reachability": selections,
        "hub_2415_reachability": hub_2415,
        "failed_model_links": failed_links,
        "failed_model_links_count": len(failed_links),
        "failed_wall_crossing_model_links_count": len(crossing_failures),
        "failed_wall_crossing_fraction": (
            len(crossing_failures) / len(failed_links) if failed_links else None
        ),
        "offline_bridge_wall_audit": {
            "bridges": int(bridge_weight.size),
            "wall_crossing_bridges": int(wall_mask.sum()),
            "wall_crossing_fraction": float(wall_mask.mean()),
            "wall_crossing_weight_mean": float(crossing_weight.mean()),
            "wall_crossing_weight_median": float(np.median(crossing_weight)),
            "noncrossing_weight_mean": float(noncrossing_weight.mean()),
            "noncrossing_weight_median": float(np.median(noncrossing_weight)),
            "wall_crossing_weight_at_least_0p5": int(
                np.sum(crossing_weight >= 0.5)
            ),
            "wall_crossing_weight_at_least_0p7": int(
                np.sum(crossing_weight >= 0.7)
            ),
            "wall_crossing_weight_at_least_0p9": int(
                np.sum(crossing_weight >= 0.9)
            ),
            "maze_geometry_role": "diagnostic_only_not_planner_input",
        },
        "interpretation_gate": {
            "hub_2415_corrected_values_available": bool(hub_2415),
            "nearby_failed_crossing_variant_distances_reported_without_cutoff": bool(
                any(
                    "destination_distance_from_previous_failed_crossing_m" in row
                    for row in crossing_failures
                )
            ),
            "behavioral_adoption_decision": (
                "compare_corrected_task8_path_and_neighbor_switching"
            ),
        },
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "artifact_hashes": {
            "graph_sha256": r52.sha256(paths["graph"]),
            "bridges_sha256": r52.sha256(paths["bridges"]),
            "replay_log_sha256": r52.sha256(log_path),
        },
    }
    report_path = output / "corrected_hub_reachability_audit.json"
    r48.dump(report_path, report)
    print(json.dumps({
        "task_id": TASK_ID,
        "commands": len(logs),
        "hub_selections": len(selections),
        "hub_2415_reachability": hub_2415,
        "failed_model_links_count": len(failed_links),
        "failed_wall_crossing_model_links_count": len(crossing_failures),
        "offline_bridge_wall_audit": report["offline_bridge_wall_audit"],
        "report": str(report_path),
    }, indent=2))
    return report


def command_run(args) -> dict:
    command_replay(args)
    return command_analyze(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=r69.EPISODES)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("replay").set_defaults(func=command_replay)
    commands.add_parser("analyze").set_defaults(func=command_analyze)
    commands.add_parser("run").set_defaults(func=command_run)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
