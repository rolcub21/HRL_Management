#!/usr/bin/env python3
"""Twenty-episode budget diagnostic for the unchanged R74 planner."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r72_hub_route_correction as base,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R75_R74_TWENTY_EPISODE_REFERENCE.md"
PROTOCOL = HERE / "r75_r74_twenty_episode_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_r74_twenty_episode_r75"
HORIZON = r74.HORIZON
EPISODES = 20
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_r74_twenty_episode_r75/original9"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": r74.METHOD_REVISION,
        "task_ids": list(range(9)),
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "maximum_primitive_steps_per_task": (
            EPISODES * HORIZON * r48.COMMAND_STEPS
        ),
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "hub_route_similarity": r74.PROTOCOL_VALUE[
            "hub_route_similarity"
        ],
        "route_source_scale": "R52_empirical_reach_covariance",
        "route_hub_scale": "R52_empirical_reach_covariance",
        "hub_route_similarity_cutoff": None,
        "hub_evaluation_cutoff": None,
        "reachability_computational_backend": "exact_batched_rolling_torch",
        "coverage_computational_backend": "exact_chunked_torch",
        "return_route_correction": (
            "none_R51_link_level_correction_only_unchanged_from_R74"
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R75 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R75 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def configure_implementation() -> None:
    base.REFERENCE = REFERENCE
    base.PROTOCOL = PROTOCOL
    base.METHOD_REVISION = METHOD_REVISION
    base.PROTOCOL_VALUE = PROTOCOL_VALUE
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EPISODES = EPISODES
    base.PLOT_TITLE = (
        "R75 unchanged R74 planner — 20 commands × 20 episodes"
    )
    base.PLOT_STEM = "r75_r74_twenty_episode_original9"
    base._load_runtime = _load_runtime
    base.command_audit = command_audit


def _load_runtime(args):
    loaded = list(r74._load_runtime(args))
    planner = loaded[0]
    planner.method_revision = METHOD_REVISION
    planner.planner_name = (
        "E549_R74_empirical_reach_route_correction_twenty_episode_budget"
    )
    loaded[0] = planner
    return tuple(loaded)


def command_audit(args) -> dict:
    # R74 owns the method audit and exact-certificate validation.  Restore the
    # R75 reporting contract afterwards because R74 deliberately rebinds its
    # base module while running that audit.
    with contextlib.redirect_stdout(io.StringIO()):
        report = r74.command_audit(args)
    configure_implementation()
    report.update({
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": (
            HORIZON * r48.COMMAND_STEPS
        ),
        "maximum_primitive_steps_per_task": (
            EPISODES * HORIZON * r48.COMMAND_STEPS
        ),
        "base_evaluation_revision": r74.METHOD_REVISION,
        "planner_change_from_R74": "none",
        "return_route_correction": PROTOCOL_VALUE[
            "return_route_correction"
        ],
        "return_route_correction_audit": {
            "route_level_hub_scorer_called_only_during_acquisition": True,
            "return_uses_R51_link_level_online_correction": True,
            "return_uses_route_level_outcome_correction": False,
        },
    })
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    r48.dump(output / "audit.json", report)
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(args) -> dict:
    configure_implementation()
    if int(args.episodes) != EPISODES:
        raise ValueError("R75 requires twenty episodes per development task")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R75 requires evaluation seed 607")
    audit = command_audit(args)
    configure_implementation()
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError("R74 artifacts or audit gates are incomplete")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R75 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved

    (
        planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_runtime(args)
    rows = []
    try:
        for task_id in PROTOCOL_VALUE["task_ids"]:
            row = base.r67._evaluate_task(
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
            row = base.r68._normalize_task_record(
                row, log_path, hub_nodes
            )
            row["closest_goal_distance_m"] = (
                base.r69._closest_goal_distance(log_path, row["goal"])
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
        for success in row["successes"][
            row["first_contact_episode"] + 1 :
        ]
    ]
    figures = base._plot(output, rows)
    total_route_attempts = sum(row["hub_route_attempts"] for row in rows)
    total_route_arrivals = sum(row["hub_route_arrivals"] for row in rows)
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "base_evaluation_revision": r74.METHOD_REVISION,
        "planner_change_from_R74": "none",
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": (
            HORIZON * r48.COMMAND_STEPS
        ),
        "maximum_primitive_steps_per_task": (
            EPISODES * HORIZON * r48.COMMAND_STEPS
        ),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contacts
        ])),
        "acquisition_at_10": float(np.mean([
            value is not None and value < 10 for value in contacts
        ])),
        "acquisition_at_20": float(np.mean([
            value is not None and value < 20 for value in contacts
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
        "hub_route_attempts": int(total_route_attempts),
        "hub_route_arrivals": int(total_route_arrivals),
        "hub_route_failures": int(
            total_route_attempts - total_route_arrivals
        ),
        "hub_route_arrival_rate": (
            float(total_route_arrivals / total_route_attempts)
            if total_route_attempts else None
        ),
        "online_destination_arrival_rate": float(np.mean([
            row["online_destination_arrival_rate"] for row in rows
            if row["online_destination_arrival_rate"] is not None
        ])),
        "hub_route_similarity": PROTOCOL_VALUE["hub_route_similarity"],
        "hub_route_similarity_cutoff": None,
        "hub_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "route_source_scale": PROTOCOL_VALUE["route_source_scale"],
        "route_hub_scale": PROTOCOL_VALUE["route_hub_scale"],
        "reachability_computational_backend": PROTOCOL_VALUE[
            "reachability_computational_backend"
        ],
        "coverage_computational_backend": PROTOCOL_VALUE[
            "coverage_computational_backend"
        ],
        "return_route_correction": PROTOCOL_VALUE[
            "return_route_correction"
        ],
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
            "tasks", "episodes_per_task", "commands_per_episode",
            "acquired_tasks", "acquisition_at_1", "acquisition_at_5",
            "acquisition_at_10", "acquisition_at_20",
            "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "online_destination_arrival_rate", "hub_route_attempts",
            "hub_route_arrivals", "hub_route_failures",
            "hub_route_arrival_rate", "figures",
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
    configure_implementation()
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
