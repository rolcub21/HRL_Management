#!/usr/bin/env python3
"""R68 dynamic hubs with 20 commands and 10 episodes on tasks 1, 2, 8."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.lines import Line2D

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
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


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R69_E549_DYNAMIC_HUB_LONG_BUDGET_REFERENCE.md"
PROTOCOL = HERE / "r69_e549_dynamic_hub_long_budget_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_dynamic_hub_long_budget_development_r69"
HORIZON = 20
EPISODES = 10
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_dynamic_hub_long_budget_development_r69/tasks_1_2_8"
)
R68_SUMMARY = r68.DEFAULT_OUTPUT / "summary.json"


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "base_dynamic_hub_revision": r68.METHOD_REVISION,
        "task_ids": [1, 2, 8],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "hub_candidate_count": "data_derived",
        "reachability_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R69 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R69 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def _install_contract() -> None:
    """Bind the reused R68/R67 implementation to R69's explicit budget."""
    r48.HORIZON = HORIZON
    r48.RuleFreeR39Planner.horizon = HORIZON
    r51.DestinationArrivalOnlinePlanner.horizon = HORIZON
    r53.E549DestinationArrivalPlanner.horizon = HORIZON

    r68.REFERENCE = REFERENCE
    r68.PROTOCOL = PROTOCOL
    r68.METHOD_REVISION = METHOD_REVISION
    r68.PROTOCOL_VALUE = PROTOCOL_VALUE
    r68.DEFAULT_OUTPUT = DEFAULT_OUTPUT

    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE
    r67._score_hubs = r68._score_all_hubs


def _audit(args) -> dict:
    _install_contract()
    required = r67._required(args)
    contract, hub_nodes, _hub_xy, _arrays = r68._candidate_contract(args)
    ready = r68._ready(args)
    artifact = r68._artifact_paths(args.output)
    table_bytes = int(np.prod(contract["shape"], dtype=np.int64) * 4)
    free_bytes = shutil.disk_usage(MANAGEMENT_ROOT).free
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
        "hub_candidates_data_derived": int(len(hub_nodes)),
        "reachability_horizon_commands": int(contract["horizon"]),
        "reachability_cutoff": None,
        "reachability_table_bytes": table_bytes,
        "reachability_table_gib": table_bytes / 1024**3,
        "free_space_gib": free_bytes / 1024**3,
        "reachability_table_ready": ready,
        "reachability_table": str(artifact["table"]),
        "preparation_artifacts_ready": all(path.is_file() for path in required),
        "ready_for_evaluation": bool(
            ready and all(path.is_file() for path in required)
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    print(json.dumps(report, indent=2))
    return report


def command_audit(args) -> dict:
    return _audit(args)


def command_prepare(args) -> dict:
    _install_contract()
    return r68.command_prepare(args)


def _load_runtime(args):
    report = _audit(args)
    if not report["ready_for_evaluation"]:
        raise FileNotFoundError("prepare the R69 horizon-20 table first")
    contract, hub_nodes, hub_xy, arrays = r68._candidate_contract(args)
    artifact = r68._artifact_paths(args.output)
    r68._validate_metadata(artifact["metadata"], contract)
    reachability = np.load(artifact["table"], mmap_mode="r")
    planner, executor, env = r53._load_runtime(args)
    planner.horizon = HORIZON
    return planner, executor, env, hub_nodes, hub_xy, arrays[2], reachability


def _closest_goal_distance(log_path: Path, goal) -> Optional[float]:
    goal = np.asarray(goal, np.float64)
    distances = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        endpoint = np.asarray(json.loads(line)["actual_endpoint"], np.float64)
        distances.append(float(np.linalg.norm(endpoint - goal)))
    return min(distances) if distances else None


def _plot(output: Path, rows: list[dict]) -> dict:
    colors = r67.plt.get_cmap("tab10")(np.linspace(0.0, 0.9, EPISODES))
    figure, axes = r67.plt.subplots(
        1, 3, figsize=(17.5, 6.1), constrained_layout=True
    )
    for axis, task in zip(axes, rows):
        r67.maze_plot.draw_maze(axis)
        log_path = output / f"task_{int(task['task_id']):03d}.jsonl"
        logs = [
            json.loads(line) for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        goal = np.asarray(task["goal"], np.float64)
        axis.scatter(
            *goal, marker="X", s=100, color="#b74343", zorder=12,
            label="hidden goal",
        )
        for episode in range(EPISODES):
            episode_rows = [
                row for row in logs if int(row["episode"]) == episode
            ]
            if not episode_rows:
                continue
            color = colors[episode]
            start = np.asarray(episode_rows[0]["start_xy"], np.float64)
            previous = start
            for row in episode_rows:
                endpoint = np.asarray(row["actual_endpoint"], np.float64)
                linestyle = "-" if row["phase"] == "acquisition" else "--"
                axis.plot(
                    [previous[0], endpoint[0]],
                    [previous[1], endpoint[1]],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.45,
                    alpha=0.9,
                    zorder=3,
                )
                axis.scatter(
                    endpoint[0], endpoint[1], s=11, color=color,
                    alpha=0.9, zorder=5,
                )
                if row.get("hub_selected_this_command"):
                    hub = np.asarray(row["active_hub_xy"], np.float64)
                    axis.scatter(
                        hub[0], hub[1], marker="D", s=48,
                        facecolor="none", edgecolor=color,
                        linewidth=1.1, zorder=8,
                    )
                if row.get("reward"):
                    axis.scatter(
                        endpoint[0], endpoint[1], marker="*", s=100,
                        color="#ffd24a", edgecolor="#5d4a00",
                        linewidth=0.7, zorder=11,
                    )
                previous = endpoint

        contact = task["first_contact_episode"]
        status = f"contact E{contact + 1}" if contact is not None else "no contact"
        closest = task.get("closest_goal_distance_m")
        closest_text = "n/a" if closest is None else f"{closest:.2f} m"
        axis.set_title(
            f"Task {int(task['task_id'])}: {status}\n"
            f"closest {closest_text}, {int(task['unique_hubs_selected'])} hubs, "
            f"{task['acquisition_new_1m_cells_per_command']:.2f} cells/cmd",
            fontsize=10,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    handles = [
        Line2D([0], [0], color=colors[index], linewidth=2,
               label=f"E{index + 1}")
        for index in range(EPISODES)
    ]
    handles.extend([
        Line2D([0], [0], color="black", linestyle="-",
               label="acquisition"),
        Line2D([0], [0], color="black", linestyle="--",
               label="return"),
        Line2D([0], [0], marker="D", markerfacecolor="none",
               markeredgecolor="black", linestyle="none",
               label="selected hub"),
    ])
    figure.legend(
        handles=handles, loc="lower center", ncol=13,
        bbox_to_anchor=(0.5, -0.02), fontsize=8,
    )
    figure.suptitle(
        "R69 E549-5M dynamic hubs — 20 commands × 10 episodes",
        fontsize=14,
        fontweight="bold",
    )
    stem = output / "r69_long_budget_dynamic_hubs_tasks_1_2_8"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    r67.plt.close(figure)
    return {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }


def command_plot(args) -> dict:
    output = Path(args.output).resolve()
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("run the R69 evaluation before plotting")
    summary = json.loads(summary_path.read_text())
    if summary.get("method_revision") != METHOD_REVISION:
        raise ValueError("saved R69 summary has another method")
    figures = _plot(output, summary["rows"])
    summary["figures"] = figures
    r48.dump(summary_path, summary)
    print(json.dumps(figures, indent=2))
    return figures


def command_evaluate(args) -> dict:
    if int(args.episodes) != EPISODES:
        raise ValueError("R69 requires ten episodes per development task")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R69 requires evaluation seed 607")
    _install_contract()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R69 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved

    planner, executor, env, hub_nodes, hub_xy, nodes, reachability = (
        _load_runtime(args)
    )
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
            row["closest_goal_distance_m"] = _closest_goal_distance(
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
    baseline_rows = {}
    if R68_SUMMARY.is_file():
        baseline_rows = {
            int(row["task_id"]): row
            for row in json.loads(R68_SUMMARY.read_text()).get("rows", [])
        }
    comparison = []
    for row in rows:
        task_id = int(row["task_id"])
        base = baseline_rows.get(task_id, {})
        comparison.append({
            "task_id": task_id,
            "r68_first_contact_episode": base.get("first_contact_episode"),
            "r69_first_contact_episode": row["first_contact_episode"],
            "r68_new_1m_cells_per_command": base.get(
                "acquisition_new_1m_cells_per_command"
            ),
            "r69_new_1m_cells_per_command": row[
                "acquisition_new_1m_cells_per_command"
            ],
            "r69_closest_goal_distance_m": row["closest_goal_distance_m"],
        })

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
        "dynamic_hub_candidates": int(len(hub_nodes)),
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
        "hub_selections": int(sum(row["hub_selections"] for row in rows)),
        "hub_arrivals": int(sum(row["hub_arrivals"] for row in rows)),
        "r68_same_task_comparison": comparison,
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
            "r68_same_task_comparison",
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
    commands.add_parser("prepare").set_defaults(func=command_prepare)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    commands.add_parser("plot").set_defaults(func=command_plot)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

