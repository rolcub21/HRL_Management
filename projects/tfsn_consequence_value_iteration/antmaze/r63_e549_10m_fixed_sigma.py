#!/usr/bin/env python3
"""R59 fixed-sigma original-nine panel with E549-10M/R54."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r59_e549_hidden_goal_all9 as implementation,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r54_e549_10m_preparation as r54,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_10M_CHECKPOINT,
    E549_10M_CHECKPOINT_SHA256,
    E54910MSACExecutor,
    inspect_e549_10m_checkpoint,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r63_e549_10m_fixed_sigma_protocol.json"
REFERENCE = HERE / "R63_E549_10M_FIXED_SIGMA_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_10m_fixed_sigma_hidden_goal_original9_r63"
EVALUATION_ARM = "R51_fixed_R54_sigma_with_E549_10M_R54_preparation"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_10m_fixed_sigma_r63/original9"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r54.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "oracle_inputs": [],
        "task_ids": list(range(9)),
        "fresh_execution_task_ids": list(range(9)),
        "episodes": 5,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "executor_checkpoint_sha256": E549_10M_CHECKPOINT_SHA256,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R63 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R63 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def configure_implementation() -> None:
    """Install the matched 10M fixed-sigma contract in this process."""
    r54.configure_implementation()
    implementation.r52 = r54
    implementation.PROTOCOL = PROTOCOL
    implementation.REFERENCE = REFERENCE
    implementation.METHOD_REVISION = METHOD_REVISION
    implementation.EVALUATION_ARM = EVALUATION_ARM
    implementation.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    implementation.E549_CHECKPOINT_SHA256 = E549_10M_CHECKPOINT_SHA256
    implementation.inspect_e549_checkpoint = inspect_e549_10m_checkpoint
    implementation.PROTOCOL_VALUE = PROTOCOL_VALUE

    runtime = implementation.r53
    runtime.r52 = r54
    runtime.E549_CHECKPOINT = E549_10M_CHECKPOINT
    runtime.E549_CHECKPOINT_SHA256 = E549_10M_CHECKPOINT_SHA256
    runtime.E549SACExecutor = E54910MSACExecutor
    runtime.inspect_e549_checkpoint = inspect_e549_10m_checkpoint
    runtime.EVALUATION_ARM = EVALUATION_ARM
    runtime.E549DestinationArrivalPlanner.method_revision = METHOD_REVISION
    runtime.E549DestinationArrivalPlanner.planner_name = EVALUATION_ARM


def command_audit(args) -> dict:
    configure_implementation()
    report = implementation.command_audit(args)
    report.update(
        {
            "acquisition_observation_memory": PROTOCOL_VALUE[
                "acquisition_observation_memory"
            ],
            "acquisition_search_sigma": PROTOCOL_VALUE[
                "acquisition_search_sigma"
            ],
            "fresh_execution_task_ids": PROTOCOL_VALUE[
                "fresh_execution_task_ids"
            ],
            "latent_radius_posterior_used": False,
            "five_million_artifacts_admissible": False,
        }
    )
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(args) -> dict:
    configure_implementation()
    result = implementation.command_evaluate(args)
    result.update(
        {
            "acquisition_observation_memory": PROTOCOL_VALUE[
                "acquisition_observation_memory"
            ],
            "acquisition_search_sigma": PROTOCOL_VALUE[
                "acquisition_search_sigma"
            ],
            "fresh_execution_task_ids": PROTOCOL_VALUE[
                "fresh_execution_task_ids"
            ],
            "latent_radius_posterior_used": False,
            "five_million_trajectories_reused": False,
        }
    )
    implementation.r48.dump(Path(args.output).resolve() / "summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=5)
    value.add_argument("--preparation", type=Path, default=r54.DEFAULT_OUTPUT)
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
