#!/usr/bin/env python3
"""Area-weighted hub routing gate on consumed AntMaze tasks 1, 2, and 8."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    r53_e549_r51_evaluation as r53,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r56_e549_supplied_goal as r56,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r64_r52_offline_hub_ranking as r64,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r65_r52_area_weighted_hubs as r65,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    LatentRadiusPosterior,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R67_E549_HUB_ROUTED_EXPLORATION_REFERENCE.md"
PROTOCOL = HERE / "r67_e549_hub_routed_exploration_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_area_hub_routed_development_r67"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_area_hub_routed_development_r67/tasks_1_2_8"
)
R61_SUMMARY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_phase_separated_latent_rho_r61/original9/summary.json"
)
EPISODE_COLORS = ["#2f6fbb", "#7655a5", "#17866b", "#dd7b22", "#c13b67"]


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "area_hub_revision": r65.METHOD_REVISION,
        "task_ids": [1, 2, 8],
        "episodes": 5,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "hub_candidate_count_computational": 20,
        "reachability_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R67 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R67 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def _required(args) -> list[Path]:
    paths = r52.output_paths(args.preparation)
    return [
        paths["consequence"],
        paths["consequence_summary"],
        paths["graph"],
        paths["runtime"] / "planner/mobility.pt",
        paths["runtime"] / "planner/bridges.npz",
        paths["runtime"] / "empirical_passage/metadata.json",
        r65.DEFAULT_OUTPUT / "summary.json",
        R61_SUMMARY,
    ]


def command_audit(args) -> dict:
    required = _required(args)
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": PROTOCOL_VALUE["episodes"],
        "hub_candidates": PROTOCOL_VALUE[
            "hub_candidate_count_computational"
        ],
        "reachability_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()} for path in required
        ],
        "ready_for_evaluation": all(path.is_file() for path in required),
    }
    print(json.dumps(report, indent=2))
    return report


def _graph_arrays(preparation: Path):
    paths = r64.preparation_paths(Path(preparation).resolve())
    consequence = json.loads(paths["consequence_summary"].read_text())
    with np.load(paths["graph"], allow_pickle=False) as stored:
        nodes = stored["nodes"].astype(np.float32)
        source = stored["source_index"].astype(np.int32)
        endpoint = stored["endpoint_index"].astype(np.int32)
        edge_weight = stored["edge_reliability"].astype(np.float32)
    with np.load(paths["bridges"], allow_pickle=False) as stored:
        bridge_target = stored["target_index"].astype(np.int32)
        bridge_weight = stored["weight"].astype(np.float32)
    return (
        paths,
        consequence,
        nodes,
        source,
        endpoint,
        edge_weight,
        bridge_target,
        bridge_weight,
    )


def _one_target_layers(
    target: int,
    horizon: int,
    node_count: int,
    source: np.ndarray,
    endpoint: np.ndarray,
    edge_weight: np.ndarray,
    bridge_target: np.ndarray,
    bridge_weight: np.ndarray,
) -> np.ndarray:
    layers = np.zeros((int(horizon) + 1, node_count), np.float32)
    layers[0, int(target)] = 1.0
    for remaining in range(1, int(horizon) + 1):
        previous = layers[remaining - 1]
        bridge_q = (
            bridge_weight * previous[bridge_target]
            + (1.0 - bridge_weight) * previous[:, None]
        )
        best = bridge_q.max(axis=1)
        recorded_q = (
            edge_weight * previous[endpoint]
            + (1.0 - edge_weight) * previous[source]
        )
        np.maximum.at(best, source, recorded_q)
        best[int(target)] = 1.0
        layers[remaining] = best
    return layers


def _reachability_cache(
    args, nodes, source, endpoint, edge_weight, bridge_target, bridge_weight,
    hub_nodes,
):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / "hub_reachability_layers.npy"
    metadata_path = output / "hub_reachability_layers.json"
    graph_path = r52.output_paths(args.preparation)["graph"]
    bridge_path = r52.output_paths(args.preparation)["runtime"] / "planner/bridges.npz"
    contract = {
        "method_revision": METHOD_REVISION,
        "graph_sha256": r52.sha256(graph_path),
        "bridges_sha256": r52.sha256(bridge_path),
        "hub_nodes": np.asarray(hub_nodes, np.int32).tolist(),
        "horizon": int(r48.HORIZON),
        "shape": [len(hub_nodes), int(r48.HORIZON) + 1, len(nodes)],
    }
    if array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata != contract:
            raise ValueError("saved R67 reachability cache has another contract")
        return np.load(array_path, mmap_mode="r"), 0.0, True
    start = time.perf_counter()
    layers = np.stack([
        _one_target_layers(
            int(target), r48.HORIZON, len(nodes), source, endpoint,
            edge_weight, bridge_target, bridge_weight,
        )
        for target in hub_nodes
    ])
    with array_path.open("wb") as stream:
        np.save(stream, layers, allow_pickle=False)
    metadata_path.write_text(json.dumps(contract, indent=2) + "\n")
    return np.load(array_path, mmap_mode="r"), time.perf_counter() - start, False


def _hub_contract(args):
    base = json.loads((r65.DEFAULT_OUTPUT / "summary.json").read_text())
    if base.get("method_revision") != r65.METHOD_REVISION:
        raise ValueError("R65 hub artifact revision changed")
    count = int(PROTOCOL_VALUE["hub_candidate_count_computational"])
    hubs = base["hubs"][:count]
    if len(hubs) != count:
        raise ValueError("R65 does not contain the declared R67 hub bank")
    arrays = _graph_arrays(args.preparation)
    consequence, nodes = arrays[1], arrays[2]
    sigma = float(consequence["landing_sigma_m"])
    hub_nodes = np.asarray([row["node_index"] for row in hubs], np.int32)
    hub_xy = nodes[hub_nodes, :2].astype(np.float64)
    declared_xy = np.asarray([row["xy"] for row in hubs], np.float64)
    if not np.allclose(hub_xy, declared_xy, rtol=0.0, atol=1e-7):
        raise ValueError("R65 hub nodes do not match the R52 graph")
    return hubs, hub_nodes, hub_xy, sigma, arrays


def _area_posterior(nodes: np.ndarray, sigma: float) -> LatentRadiusPosterior:
    support = np.unique(nodes[:, :2].astype(np.float64), axis=0)
    weights, _occupied, _origin = r65.area_weights(support, sigma)
    posterior = LatentRadiusPosterior(support, sigma)
    if not np.array_equal(posterior.goal_xy, support):
        raise ValueError("latent posterior changed R65 support order")
    # LatentRadiusPosterior's exact interval algebra accepts a vector prior;
    # replacing the scalar uniform prior changes no likelihood computation.
    posterior._prior_per_goal = weights.copy()
    posterior._refresh()
    return posterior


def _negative_path_update(
    posterior: LatentRadiusPosterior, passage_xy: np.ndarray
) -> float:
    passage = np.asarray(passage_xy, np.float64).reshape(-1, 2)
    if not len(passage):
        return 0.0
    start = time.perf_counter()
    distance = np.linalg.norm(
        passage[:, None, :] - posterior.goal_xy[None, :, :], axis=2
    ).min(axis=0)
    posterior.upper = np.minimum(posterior.upper, distance)
    posterior.negative_observations += len(passage)
    posterior._refresh()
    posterior.assert_not_exhausted("after R67 traversed path")
    return time.perf_counter() - start


def _score_hubs(
    posterior: LatentRadiusPosterior,
    hub_xy: np.ndarray,
    reachability: np.ndarray,
    graph,
    state: np.ndarray,
    remaining: int,
):
    start = time.perf_counter()
    coverage = np.asarray([
        posterior.contact_probability(xy.reshape(1, 2)) for xy in hub_xy
    ], np.float64)
    current_node = int(graph.project(np.asarray(state)[None])[0][0])
    reach = np.asarray(reachability[:, int(remaining), current_node], np.float64)
    score = coverage * reach
    selected = int(np.argmax(score))
    return selected, coverage, reach, score, current_node, time.perf_counter() - start


def _coverage_cell(xy):
    return tuple(int(value) for value in np.floor(np.asarray(xy, np.float32)))


def _task_seed(seed: int, task_id: int) -> int:
    return int(seed) + 1_000_003 * int(task_id)


def _evaluate_task(
    args, planner, executor, env, task_id, output, hub_nodes, hub_xy,
    reachability, nodes, sigma,
):
    task_path = output / f"task_{task_id:03d}.json"
    log_path = output / f"task_{task_id:03d}.jsonl"
    if task_path.is_file():
        saved = json.loads(task_path.read_text())
        expected = {
            "method_revision": METHOD_REVISION,
            "task_id": int(task_id),
            "episodes": int(args.episodes),
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"saved R67 task has another {key}")
        print(json.dumps(saved), flush=True)
        return saved

    task = r48.r31.r22._task(args.root, int(task_id))
    goal = np.asarray(task["goal"], np.float32)
    success_radius = float(
        r48.PROTOCOL_VALUE["evaluator"]["success_radius_m_evaluator_only"]
    )
    planner.rng = np.random.default_rng(_task_seed(args.seed, task_id))
    planner.seed = _task_seed(args.seed, task_id)
    planner.goal_key = None
    planner.reset_online_correction_task()
    hub_attempt_observer = getattr(planner, "hub_attempt_observer", None)
    if hub_attempt_observer is not None:
        hub_attempt_observer.reset_task()
    posterior = _area_posterior(nodes, sigma)
    fields = r48.TaskFields(
        sigma,
        search_mode=planner.acquisition_search_mode,
        coarse_sigma=planner.coarse_search_sigma,
        coverage_reference_xy=planner.coverage_reference_xy,
    )
    online_states, completed, logs = [], [], []
    first_contact_episode = None
    first_contact_step = None
    successes = []
    elapsed_steps = 0
    visited, endpoints = set(), []
    acquisition_commands = acquisition_new = acquisition_zero = 0
    acquisition_revisited = acquisition_cycles = 0
    hub_selections = hub_arrivals = 0
    selected_hub_indices = []
    timing_memory, timing_hubs, timing_route = [], [], []

    for episode in range(int(args.episodes)):
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = r48.full_state(observation)
        trajectory_memory = [state.copy()]
        reset_reward = bool(np.linalg.norm(state[:2] - goal) <= success_radius)
        fields.add(state[:2], reset_reward)
        memory_seconds = 0.0
        if not fields.contacted:
            memory_seconds += _negative_path_update(
                posterior, state[:2].reshape(1, 2)
            )
        visited.add(_coverage_cell(state[:2]))
        active_hub = None
        episode_success = reset_reward
        if reset_reward and first_contact_episode is None:
            first_contact_episode = episode
            first_contact_step = elapsed_steps

        for command_index in range(r48.HORIZON):
            if episode_success:
                break
            fields.rebuild()
            phase = "return" if fields.contacted else "acquisition"
            hub_seconds = 0.0
            selected_now = False
            selection = None
            if phase == "acquisition":
                if active_hub is None:
                    remaining = r48.HORIZON - int(command_index)
                    hub_score_override = getattr(
                        planner, "hub_score_override", None
                    )
                    if hub_score_override is None:
                        (
                            active_hub, coverage, reach, score, current_node,
                            hub_seconds,
                        ) = _score_hubs(
                            posterior, hub_xy, reachability, planner.graph,
                            state, remaining,
                        )
                    else:
                        (
                            active_hub, coverage, reach, score, current_node,
                            hub_seconds,
                        ) = hub_score_override(
                            posterior,
                            hub_xy,
                            reachability,
                            planner.graph,
                            state,
                            remaining,
                            online_states,
                        )
                    hub_selections += 1
                    selected_hub_indices.append(int(active_hub))
                    selected_now = True
                    selection = {
                        "hub_bank_index": int(active_hub),
                        "r65_rank": int(active_hub) + 1,
                        "hub_node_index": int(hub_nodes[active_hub]),
                        "hub_xy": hub_xy[active_hub].astype(float).tolist(),
                        "current_graph_node": current_node,
                        "posterior_contact_probability": float(
                            coverage[active_hub]
                        ),
                        "remaining_horizon_reachability": float(
                            reach[active_hub]
                        ),
                        "combined_score": float(score[active_hub]),
                        "best_score": float(score.max()),
                        "median_score": float(np.median(score)),
                    }
                    selection.update(
                        getattr(planner, "hub_selection_extra", {})
                    )
                    if hub_attempt_observer is not None:
                        selection["hub_route_attempt"] = (
                            hub_attempt_observer.start_attempt(
                                source_xy=state[:2],
                                hub_bank_index=int(active_hub),
                                hub_xy=hub_xy[active_hub],
                                remaining_commands=int(remaining),
                                predicted_reachability=float(
                                    reach[active_hub]
                                ),
                            )
                        )
                route_fields = r56.R56SuppliedGoalField(
                    sigma, hub_xy[active_hub]
                )
            else:
                route_fields = fields

            route_start = time.perf_counter()
            command, diagnostic = planner.choose(
                state,
                command_index,
                route_fields,
                online_states,
                completed + [trajectory_memory],
            )
            route_seconds = time.perf_counter() - route_start
            trajectory, rewards = r48._execute_evaluator_command(
                env, executor, state, command, goal, success_radius
            )
            cells = {_coverage_cell(xy) for xy in trajectory[1:, :2]}
            new_cells = cells - visited
            endpoint_cell = _coverage_cell(trajectory[-1, :2])
            revisited = endpoint_cell in visited
            cycle = bool(
                len(endpoints) >= 2
                and endpoint_cell == endpoints[-2]
                and endpoint_cell != endpoints[-1]
            )
            if phase == "acquisition":
                acquisition_commands += 1
                acquisition_new += len(new_cells)
                acquisition_zero += int(not new_cells)
                acquisition_revisited += int(revisited)
                acquisition_cycles += int(cycle)
                endpoints.append(endpoint_cell)
            visited.update(cells)
            for next_state, reward in zip(trajectory[1:], rewards):
                fields.add(next_state[:2], reward)
                online_states.append(next_state.copy())
                trajectory_memory.append(next_state.copy())

            memory_seconds = 0.0
            if phase == "acquisition":
                positive = np.flatnonzero(rewards)
                negative_stop = int(positive[0]) if len(positive) else len(rewards)
                memory_seconds = _negative_path_update(
                    posterior, trajectory[1 : 1 + negative_stop, :2]
                )
                hub_distance = np.linalg.norm(
                    trajectory[:, :2] - hub_xy[active_hub][None], axis=1
                )
                arrived = bool(hub_distance.min() <= sigma)
                hub_arrivals += int(arrived)
            else:
                arrived = None

            reward = bool(rewards.any())
            hub_route_attempt_outcome = None
            if phase == "acquisition" and hub_attempt_observer is not None:
                if arrived:
                    hub_route_attempt_outcome = (
                        hub_attempt_observer.finish_attempt(
                            arrived=True,
                            endpoint_xy=trajectory[-1, :2],
                        )
                    )
                elif reward:
                    hub_route_attempt_outcome = (
                        hub_attempt_observer.cancel_attempt(
                            reason="task_contact_censored"
                        )
                    )
                elif command_index + 1 == r48.HORIZON:
                    hub_route_attempt_outcome = (
                        hub_attempt_observer.finish_attempt(
                            arrived=False,
                            endpoint_xy=trajectory[-1, :2],
                        )
                    )
            if reward and first_contact_episode is None:
                first_contact_episode = episode
                first_contact_step = (
                    elapsed_steps + int(np.flatnonzero(rewards)[0]) + 1
                )
            elapsed_steps += r48.COMMAND_STEPS
            timing_memory.append(memory_seconds)
            timing_hubs.append(hub_seconds)
            timing_route.append(route_seconds)
            diagnostic.update({
                "phase": phase,
                "episode": int(episode),
                "command_index": int(command_index),
                "start_xy": state[:2].astype(float).tolist(),
                "command": np.asarray(command, np.float32).tolist(),
                "actual_endpoint": trajectory[-1, :2].astype(float).tolist(),
                "moved_distance": float(
                    np.linalg.norm(trajectory[-1, :2] - state[:2])
                ),
                "reward": reward,
                "new_1m_cells": len(new_cells),
                "revisited_endpoint_1m_cell": bool(revisited),
                "acquisition_two_cycle_1m_cell": (
                    bool(cycle) if phase == "acquisition" else None
                ),
                "hub_selected_this_command": bool(selected_now),
                "hub_selection": selection,
                "active_hub_bank_index": (
                    int(active_hub) if phase == "acquisition" else None
                ),
                "active_hub_xy": (
                    hub_xy[active_hub].astype(float).tolist()
                    if phase == "acquisition" else None
                ),
                "hub_arrived_within_landing_sigma": arrived,
                "hub_route_attempt_outcome": hub_route_attempt_outcome,
                "timing_seconds": {
                    "memory_update": float(memory_seconds),
                    "hub_scoring": float(hub_seconds),
                    "routing": float(route_seconds),
                },
            })
            if bool(getattr(planner, "log_full_command_states", False)):
                # Diagnostic-only instrumentation.  The base R67--R69
                # protocols retain their compact xy logs; focused replays can
                # opt in when the 29-dimensional state is required to
                # reconstruct R51's state-conditioned online kernel exactly.
                diagnostic.update({
                    "start_state_full": state.astype(float).tolist(),
                    "actual_endpoint_state_full": (
                        trajectory[-1].astype(float).tolist()
                    ),
                })
            logs.append(diagnostic)
            state = trajectory[-1]
            if phase == "acquisition" and arrived:
                active_hub = None
            if reward:
                episode_success = True
                break

        completed.append(np.asarray(trajectory_memory))
        successes.append(bool(episode_success))

    planner._ingest_pending(online_states)
    log_path.write_text("\n".join(json.dumps(row) for row in logs) + "\n")
    outcomes = np.asarray(planner.online_outcomes, np.float32)
    hub_attempt_summary = (
        hub_attempt_observer.summary()
        if hub_attempt_observer is not None
        else {}
    )
    contact = first_contact_episode
    row = {
        **task,
        "task_id": int(task_id),
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "successes": successes,
        "first_contact_episode": contact,
        "contact_latency": (
            int(first_contact_step)
            if first_contact_step is not None
            else int(args.episodes) * r48.HORIZON * r48.COMMAND_STEPS
        ),
        "mean_e2e": float(np.mean(successes)),
        "conditional_return": (
            float(np.mean(successes[contact + 1 :]))
            if contact is not None and contact + 1 < int(args.episodes)
            else None
        ),
        "acquisition_commands": int(acquisition_commands),
        "acquisition_new_1m_cells": int(acquisition_new),
        "acquisition_new_1m_cells_per_command": (
            acquisition_new / acquisition_commands
            if acquisition_commands else None
        ),
        "acquisition_zero_new_1m_cell_fraction": (
            acquisition_zero / acquisition_commands
            if acquisition_commands else None
        ),
        "acquisition_revisited_endpoint_fraction": (
            acquisition_revisited / acquisition_commands
            if acquisition_commands else None
        ),
        "acquisition_two_cycle_count": int(acquisition_cycles),
        "hub_selections": int(hub_selections),
        "hub_arrivals": int(hub_arrivals),
        "unique_hubs_selected": int(len(set(selected_hub_indices))),
        "selected_r65_ranks": [value + 1 for value in selected_hub_indices],
        "online_destination_arrival_rate": (
            float(outcomes.mean()) if len(outcomes) else None
        ),
        **hub_attempt_summary,
        "posterior_feasible_candidates": int(posterior.feasible.sum()),
        "posterior_marginal_evidence": float(posterior.marginal_evidence),
        "landing_sigma_m": float(sigma),
        "timing_seconds": {
            "memory_update_total": float(np.sum(timing_memory)),
            "memory_update_mean_per_command": float(np.mean(timing_memory)),
            "hub_scoring_total": float(np.sum(timing_hubs)),
            "hub_scoring_mean_per_selection": (
                float(np.sum(timing_hubs) / hub_selections)
                if hub_selections else 0.0
            ),
            "routing_total": float(np.sum(timing_route)),
            "routing_mean_per_command": float(np.mean(timing_route)),
        },
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used_for_posterior_and_phase_switch_only": True,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    }
    r48.dump(task_path, row)
    print(json.dumps({
        key: row[key] for key in (
            "task_id", "first_contact_episode", "successes", "mean_e2e",
            "acquisition_new_1m_cells_per_command", "hub_selections",
            "hub_arrivals", "unique_hubs_selected", "selected_r65_ranks",
            "online_destination_arrival_rate", "timing_seconds",
        )
    }), flush=True)
    return row


def _baseline_rows() -> dict[int, dict]:
    summary = json.loads(R61_SUMMARY.read_text())
    return {
        int(row["task_id"]): row
        for row in summary.get("rows", [])
        if int(row["task_id"]) in PROTOCOL_VALUE["task_ids"]
    }


def _plot(output: Path, rows: list[dict]) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), constrained_layout=True)
    for ax, task in zip(axes, rows):
        maze_plot.draw_maze(ax)
        logs = [
            json.loads(line)
            for line in (output / f"task_{task['task_id']:03d}.jsonl")
            .read_text().splitlines()
            if line.strip()
        ]
        goal = np.asarray(task["goal"], np.float64)
        ax.scatter(*goal, marker="X", s=85, color="#b74343", zorder=9)
        for episode in range(int(PROTOCOL_VALUE["episodes"])):
            episode_rows = [
                row for row in logs
                if row["phase"] == "acquisition"
                and int(row["episode"]) == episode
            ]
            if not episode_rows:
                continue
            start = np.asarray(episode_rows[0]["start_xy"], np.float64)
            endpoints = np.asarray(
                [row["actual_endpoint"] for row in episode_rows], np.float64
            )
            path = np.vstack((start, endpoints))
            color = EPISODE_COLORS[episode]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, zorder=3)
            ax.scatter(endpoints[:, 0], endpoints[:, 1], s=15, color=color, zorder=5)
            selected = [
                row for row in episode_rows if row["hub_selected_this_command"]
            ]
            if selected:
                target = np.asarray(
                    [row["active_hub_xy"] for row in selected], np.float64
                )
                ax.scatter(
                    target[:, 0], target[:, 1], marker="D", s=52,
                    facecolor="none", edgecolor=color, linewidth=1.1, zorder=7,
                )
        status = (
            f"contact E{task['first_contact_episode'] + 1}"
            if task["first_contact_episode"] is not None else "no contact"
        )
        ax.set_title(
            f"Task {task['task_id']}: {status}\n"
            f"{task['unique_hubs_selected']} unique hubs, "
            f"{task['acquisition_new_1m_cells_per_command']:.2f} new cells/cmd",
            fontsize=10,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle(
        "R67 E549-5M hub-routed acquisition — diamonds are selected hubs",
        fontsize=14, fontweight="bold",
    )
    stem = output / "r67_hub_routed_tasks_1_2_8"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf"))}


def _load(args):
    command_audit(args)
    hubs, hub_nodes, hub_xy, sigma, arrays = _hub_contract(args)
    (
        _paths, _consequence, nodes, source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    ) = arrays
    reachability, precompute_seconds, reused = _reachability_cache(
        args, nodes, source, endpoint, edge_weight, bridge_target,
        bridge_weight, hub_nodes,
    )
    planner, executor, env = r53._load_runtime(args)
    return (
        planner, executor, env, hubs, hub_nodes, hub_xy, sigma, nodes,
        reachability, precompute_seconds, reused,
    )


def command_timing(args) -> dict:
    (
        planner, _executor, env, _hubs, hub_nodes, hub_xy, sigma, nodes,
        reachability, precompute_seconds, reused,
    ) = _load(args)
    try:
        task_id = int(PROTOCOL_VALUE["task_ids"][0])
        task = r48.r31.r22._task(args.root, task_id)
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = r48.full_state(observation)
        posterior = _area_posterior(nodes, sigma)
        memory_seconds = _negative_path_update(
            posterior, state[:2].reshape(1, 2)
        )
        planner.rng = np.random.default_rng(_task_seed(args.seed, task_id))
        planner.seed = _task_seed(args.seed, task_id)
        planner.goal_key = None
        planner.reset_online_correction_task()
        (
            selected, coverage, reach, score, current_node, hub_seconds,
        ) = _score_hubs(
            posterior, hub_xy, reachability, planner.graph, state,
            r48.HORIZON,
        )
        route_fields = r56.R56SuppliedGoalField(sigma, hub_xy[selected])
        route_start = time.perf_counter()
        command, diagnostic = planner.choose(
            state, 0, route_fields, [], [[state.copy()]]
        )
        route_seconds = time.perf_counter() - route_start
        result = {
            "schema_version": 1,
            "method_revision": METHOD_REVISION,
            "environment_steps": 0,
            "task_id_reset_only": task_id,
            "precompute_seconds_this_invocation": precompute_seconds,
            "precompute_cache_reused": reused,
            "decision_timing_seconds": {
                "memory_update": memory_seconds,
                "hub_scoring": hub_seconds,
                "routing": route_seconds,
                "total": memory_seconds + hub_seconds + route_seconds,
            },
            "selected_hub": {
                "r65_rank": selected + 1,
                "node_index": int(hub_nodes[selected]),
                "xy": hub_xy[selected].astype(float).tolist(),
                "current_graph_node": current_node,
                "posterior_contact_probability": float(coverage[selected]),
                "remaining_horizon_reachability": float(reach[selected]),
                "combined_score": float(score[selected]),
            },
            "issued_command_without_execution": np.asarray(command).tolist(),
            "router_selected_kind": diagnostic.get("selected_kind"),
        }
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        r48.dump(output / "one_decision_timing.json", result)
        print(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def command_evaluate(args) -> dict:
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes"]):
        raise ValueError("R67 requires the frozen five-episode development gate")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R67 requires evaluation seed 607")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R67 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved
    (
        planner, executor, env, _hubs, hub_nodes, hub_xy, sigma, nodes,
        reachability, precompute_seconds, reused,
    ) = _load(args)
    try:
        rows = [
            _evaluate_task(
                args, planner, executor, env, task_id, output, hub_nodes,
                hub_xy, reachability, nodes, sigma,
            )
            for task_id in PROTOCOL_VALUE["task_ids"]
        ]
    finally:
        env.close()
    contacts = [row["first_contact_episode"] for row in rows]
    returns = [
        success
        for row in rows
        if row["first_contact_episode"] is not None
        for success in row["successes"][row["first_contact_episode"] + 1 :]
    ]
    baseline = _baseline_rows()
    comparison = [{
        "task_id": int(row["task_id"]),
        "r61_first_contact_episode": baseline[int(row["task_id"])][
            "first_contact_episode"
        ],
        "r67_first_contact_episode": row["first_contact_episode"],
        "r61_new_1m_cells_per_command": baseline[int(row["task_id"])][
            "acquisition_new_1m_cells_per_command"
        ],
        "r67_new_1m_cells_per_command": row[
            "acquisition_new_1m_cells_per_command"
        ],
    } for row in rows]
    figures = _plot(output, rows)
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([value is not None for value in contacts])),
        "pooled_conditional_return": float(np.mean(returns)) if returns else None,
        "mean_e2e": float(np.mean([
            success for row in rows for success in row["successes"]
        ])),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "hub_selections": int(sum(row["hub_selections"] for row in rows)),
        "hub_arrivals": int(sum(row["hub_arrivals"] for row in rows)),
        "timing_seconds": {
            "reachability_precompute_this_invocation": precompute_seconds,
            "reachability_cache_reused": reused,
            "memory_update_total": float(sum(
                row["timing_seconds"]["memory_update_total"] for row in rows
            )),
            "hub_scoring_total": float(sum(
                row["timing_seconds"]["hub_scoring_total"] for row in rows
            )),
            "routing_total": float(sum(
                row["timing_seconds"]["routing_total"] for row in rows
            )),
        },
        "r61_same_task_comparison": comparison,
        "figures": figures,
        "rows": rows,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(summary_path, result)
    print(json.dumps({
        key: result[key] for key in (
            "tasks", "task_ids", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command", "hub_selections",
            "hub_arrivals", "timing_seconds", "r61_same_task_comparison",
            "figures",
        )
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=5)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("timing").set_defaults(func=command_timing)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
