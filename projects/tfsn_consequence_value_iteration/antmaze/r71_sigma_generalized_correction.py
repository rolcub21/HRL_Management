#!/usr/bin/env python3
"""Nine-goal R69 panel with sigma-generalized online link correction.

R71 changes only the generalization geometry of R51's destination-arrival
evidence.  Source position, command target, and graph destination are compared
at the deployed landing scale.  The same posterior link weights drive live
routing and exact dynamic-hub reachability.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

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
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r69_e549_dynamic_hub_long_budget as r69,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r70_corrected_hub_reachability as r70,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R71_SIGMA_GENERALIZED_CORRECTION_REFERENCE.md"
PROTOCOL = HERE / "r71_sigma_generalized_correction_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_sigma_generalized_online_correction_r71"
HORIZON = 20
EPISODES = 10
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_sigma_generalized_correction_r71/original9_long_budget"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "task_ids": list(range(9)),
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "online_link_similarity": (
            "continuous_gaussian_over_source_xy_command_xy_and_"
            "destination_xy_each_scaled_by_deployed_landing_sigma"
        ),
        "similarity_cutoff": None,
        "corrected_hub_selection": (
            "exact_global_argmax_by_coverage_upper_bound_branch_and_bound"
        ),
        "hub_evaluation_cutoff": None,
        "graph_links_corrected": "model_bridges_only",
        "recorded_edges": "unchanged_exact_preparation_transitions",
        "online_observation": (
            "binary_arrival_within_landing_sigma_of_graph_destination"
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R71 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R71 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def sigma_link_similarity(
    source_xy,
    command_xy,
    destination_xy,
    observed_source_xy,
    observed_command_xy,
    observed_destination_xy,
    sigma,
) -> np.ndarray:
    """Continuous similarity between links, with no cutoff or posture term."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("landing sigma must be positive and finite")
    source = np.asarray(source_xy, np.float64)
    command = np.asarray(command_xy, np.float64)
    destination = np.asarray(destination_xy, np.float64)
    source_distance = np.square(
        (source - np.asarray(observed_source_xy, np.float64)) / sigma
    ).sum(axis=-1)
    command_distance = np.square(
        (command - np.asarray(observed_command_xy, np.float64)) / sigma
    ).sum(axis=-1)
    destination_distance = np.square(
        (
            destination
            - np.asarray(observed_destination_xy, np.float64)
        ) / sigma
    ).sum(axis=-1)
    return np.exp(
        -0.5 * (source_distance + command_distance + destination_distance)
    )


class SigmaGeneralizedDestinationArrivalPlanner(
    r53.E549DestinationArrivalPlanner
):
    method_revision = METHOD_REVISION
    planner_name = "E549_R51_destination_arrival_sigma_link_generalization"

    def _online_link_mass(self, state, commands, destinations):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        destinations = np.asarray(destinations, np.float32).reshape(-1, 2)
        if not self.online_starts:
            zeros = np.zeros(len(commands), np.float32)
            return zeros, zeros
        observed_sources = np.asarray(self.online_starts, np.float32)[:, :2]
        observed_commands = np.asarray(self.online_commands, np.float32)
        observed_destinations = np.asarray(
            self.online_destinations, np.float32
        )
        outcomes = np.asarray(self.online_outcomes, np.float32)
        source_distance = np.square(
            (
                observed_sources
                - np.asarray(state, np.float32)[None, :2]
            ) / float(r48.SIGMA)
        ).sum(axis=1, dtype=np.float64)
        command_distance = np.square(
            (
                observed_commands[:, None] - commands[None]
            ) / float(r48.SIGMA)
        ).sum(axis=2, dtype=np.float64)
        destination_distance = np.square(
            (
                observed_destinations[:, None] - destinations[None]
            ) / float(r48.SIGMA)
        ).sum(axis=2, dtype=np.float64)
        similarity = np.exp(-0.5 * (
            source_distance[:, None]
            + command_distance
            + destination_distance
        ))
        return (
            similarity.sum(axis=0).astype(np.float32),
            (similarity * outcomes[:, None]).sum(axis=0).astype(np.float32),
        )


def add_sigma_observation_mass(
    support: np.ndarray,
    outcome_mass: np.ndarray,
    graph_nodes: np.ndarray,
    bridge_target_xy: np.ndarray,
    start_state: np.ndarray,
    command: np.ndarray,
    destination: np.ndarray,
    outcome: float,
    sigma: float,
) -> None:
    """Generalize one observation continuously to every modeled bridge."""
    similarity = sigma_link_similarity(
        graph_nodes[:, None, :2],
        bridge_target_xy,
        bridge_target_xy,
        np.asarray(start_state)[:2],
        command,
        destination,
        sigma,
    ).astype(np.float32)
    support += similarity
    if float(outcome) != 0.0:
        outcome_mass += similarity * float(outcome)


class SigmaGeneralizedHubScorer(r70.CorrectedHubScorer):
    """R70 exact hub scorer using R71's spatial link kernel."""

    def _reset_if_new_task(self) -> None:
        if len(self.planner.online_starts) < self.processed_observations:
            self.support.fill(0.0)
            self.outcome_mass.fill(0.0)
            self.processed_observations = 0

    def _synchronize(self, online_states) -> float:
        started = time.perf_counter()
        self.planner._ingest_pending(online_states)
        self._reset_if_new_task()
        while self.processed_observations < len(self.planner.online_starts):
            index = self.processed_observations
            add_sigma_observation_mass(
                self.support,
                self.outcome_mass,
                self.nodes,
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

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        extra = self.planner.hub_selection_extra
        corrected = extra.pop("R51_corrected_reachability_same_hub")
        extra["sigma_generalized_corrected_reachability_same_hub"] = corrected
        extra["online_link_similarity"] = PROTOCOL_VALUE[
            "online_link_similarity"
        ]
        extra["similarity_cutoff"] = None
        return result


def _bind_r71_contract() -> None:
    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE
    r67._score_hubs = r68._score_all_hubs


def _r69_args(args):
    value = copy.copy(args)
    value.output = Path(
        getattr(args, "hub_table", None) or r69.DEFAULT_OUTPUT
    ).resolve()
    return value


def command_audit(args) -> dict:
    table_args = _r69_args(args)
    table_report = r69._audit(table_args)
    consequence = r67._graph_arrays(args.preparation)[1]
    landing_sigma = float(consequence["landing_sigma_m"])
    paths = r52.output_paths(args.preparation)
    required = [
        paths["consequence"],
        paths["graph"],
        paths["runtime"] / "planner/bridges.npz",
        r68._artifact_paths(table_args.output)["table"],
        r68._artifact_paths(table_args.output)["metadata"],
    ]
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "landing_sigma_m": landing_sigma,
        "online_link_similarity": PROTOCOL_VALUE[
            "online_link_similarity"
        ],
        "similarity_cutoff": None,
        "hub_evaluation_cutoff": None,
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()}
            for path in required
        ],
        "ready_for_evaluation": bool(
            table_report["ready_for_evaluation"]
            and all(path.is_file() for path in required)
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    print(json.dumps(report, indent=2))
    return report


def _load_runtime(args):
    loaded = list(r69._load_runtime(_r69_args(args)))
    _bind_r71_contract()
    planner = loaded[0]
    # The subclass adds methods but no storage; retain the exactly loaded R52
    # model, graph, fields, and RNG while replacing only link generalization.
    planner.__class__ = SigmaGeneralizedDestinationArrivalPlanner
    planner.method_revision = METHOD_REVISION
    planner.planner_name = (
        "E549_R51_destination_arrival_sigma_link_generalization"
    )
    arrays = r67._graph_arrays(args.preparation)
    planner.hub_score_override = SigmaGeneralizedHubScorer(
        planner, arrays, loaded[3]
    )
    loaded[0] = planner
    return tuple(loaded)


def _plot(output: Path, rows: list[dict]) -> dict:
    colors = r67.plt.get_cmap("tab10")(np.linspace(0.0, 0.9, EPISODES))
    figure, axes = r67.plt.subplots(
        3, 3, figsize=(17.0, 16.0), constrained_layout=True
    )
    for axis, task in zip(axes.ravel(), rows):
        r67.maze_plot.draw_maze(axis)
        logs = [
            json.loads(line)
            for line in (
                output / f"task_{int(task['task_id']):03d}.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        goal = np.asarray(task["goal"], np.float64)
        axis.scatter(
            *goal, marker="X", s=90, color="#b74343", zorder=12
        )
        for episode in range(EPISODES):
            episode_rows = [
                row for row in logs
                if int(row["episode"]) == episode
            ]
            if not episode_rows:
                continue
            color = colors[episode]
            previous = np.asarray(episode_rows[0]["start_xy"], np.float64)
            for row in episode_rows:
                endpoint = np.asarray(row["actual_endpoint"], np.float64)
                axis.plot(
                    [previous[0], endpoint[0]],
                    [previous[1], endpoint[1]],
                    color=color,
                    linestyle="-" if row["phase"] == "acquisition" else "--",
                    linewidth=1.15,
                    alpha=0.88,
                    zorder=3,
                )
                axis.scatter(*endpoint, s=8, color=color, zorder=5)
                if row.get("hub_selected_this_command"):
                    hub = np.asarray(row["active_hub_xy"], np.float64)
                    axis.scatter(
                        *hub, marker="D", s=38, facecolor="none",
                        edgecolor=color, linewidth=1.0, zorder=8,
                    )
                previous = endpoint
        contact = task["first_contact_episode"]
        status = f"contact E{contact + 1}" if contact is not None else "no contact"
        closest = task.get("closest_goal_distance_m")
        closest_text = "n/a" if closest is None else f"{closest:.2f} m"
        axis.set_title(
            f"T{int(task['task_id'])}: {status}; closest {closest_text}\n"
            f"{task['acquisition_new_1m_cells_per_command']:.2f} cells/cmd; "
            f"{int(task['unique_hubs_selected'])} hubs",
            fontsize=9,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    handles = [
        Line2D([0], [0], color=colors[index], linewidth=2, label=f"E{index + 1}")
        for index in range(EPISODES)
    ]
    handles.extend([
        Line2D([0], [0], color="black", linestyle="-", label="acquisition"),
        Line2D([0], [0], color="black", linestyle="--", label="return"),
        Line2D(
            [0], [0], marker="D", markerfacecolor="none",
            markeredgecolor="black", linestyle="none", label="selected hub",
        ),
    ])
    figure.legend(
        handles=handles, loc="lower center", ncol=13,
        bbox_to_anchor=(0.5, -0.01), fontsize=8,
    )
    figure.suptitle(
        "R71 E549-5M σ-generalized correction — 20 commands × 10 episodes",
        fontsize=14, fontweight="bold",
    )
    stem = output / "r71_sigma_generalized_correction_original9"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    r67.plt.close(figure)
    return {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }


def command_evaluate(args) -> dict:
    if int(args.episodes) != EPISODES:
        raise ValueError("R71 requires ten episodes per development task")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R71 requires evaluation seed 607")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError("R52 preparation or R69 horizon table is incomplete")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R71 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved

    (
        planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_runtime(args)
    rows = []
    try:
        for task_id in PROTOCOL_VALUE["task_ids"]:
            row = r67._evaluate_task(
                args,
                planner,
                executor,
                env,
                int(task_id),
                output,
                hub_nodes,
                hub_xy,
                reachability,
                nodes,
                float(r48.SIGMA),
            )
            log_path = output / f"task_{int(task_id):03d}.jsonl"
            row = r68._normalize_task_record(row, log_path, hub_nodes)
            row["closest_goal_distance_m"] = r69._closest_goal_distance(
                log_path, row["goal"]
            )
            r48.dump(log_path.with_suffix(".json"), row)
            rows.append(row)
    finally:
        env.close()

    contacts = [row["first_contact_episode"] for row in rows]
    returns = [
        success
        for row in rows
        if row["first_contact_episode"] is not None
        for success in row["successes"][row["first_contact_episode"] + 1 :]
    ]
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
        "episodes_per_task": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contacts
        ])),
        "acquisition_at_10": float(np.mean([
            value is not None for value in contacts
        ])),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean([
            success for row in rows for success in row["successes"]
        ])),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "acquisition_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
        ])),
        "hub_selections": int(sum(row["hub_selections"] for row in rows)),
        "hub_arrivals": int(sum(row["hub_arrivals"] for row in rows)),
        "online_destination_arrival_rate": float(np.mean([
            row["online_destination_arrival_rate"] for row in rows
            if row["online_destination_arrival_rate"] is not None
        ])),
        "online_link_similarity": PROTOCOL_VALUE["online_link_similarity"],
        "similarity_cutoff": None,
        "figures": figures,
        "rows": rows,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(summary_path, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "task_ids", "episodes_per_task",
            "commands_per_episode", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "acquisition_at_10",
            "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "online_destination_arrival_rate", "figures",
        )
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=EPISODES)
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
