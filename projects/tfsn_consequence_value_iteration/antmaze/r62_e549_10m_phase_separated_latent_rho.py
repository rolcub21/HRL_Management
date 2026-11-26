#!/usr/bin/env python3
"""Fresh original-nine R61-method evaluation with E549-10M/R54."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r60_e549_latent_rho_all9 as implementation,
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
PROTOCOL = HERE / "r62_e549_10m_phase_separated_latent_rho_protocol.json"
REFERENCE = HERE / "R62_E549_10M_PHASE_SEPARATED_LATENT_RHO_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_10m_phase_separated_latent_rho_original9_r62"
EVALUATION_ARM = "phase_separated_latent_rho_with_E549_10M_R54_R51"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_10m_phase_separated_latent_rho_r62/original9"
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
            raise ValueError(f"R62 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R62 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def configure_implementation() -> None:
    """Install the matched 10M contract in this process only."""
    r54.configure_implementation()
    implementation.r52 = r54
    implementation.PROTOCOL = PROTOCOL
    implementation.REFERENCE = REFERENCE
    implementation.METHOD_REVISION = METHOD_REVISION
    implementation.EVALUATION_ARM = EVALUATION_ARM
    implementation.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    implementation.E549_CHECKPOINT = E549_10M_CHECKPOINT
    implementation.E549_CHECKPOINT_SHA256 = E549_10M_CHECKPOINT_SHA256
    implementation.E549SACExecutor = E54910MSACExecutor
    implementation.inspect_e549_checkpoint = inspect_e549_10m_checkpoint
    implementation.PROTOCOL_VALUE = PROTOCOL_VALUE
    implementation.RETIRE_SEARCH_POSTERIOR_AT_FIRST_CONTACT = True
    implementation.E549LatentRhoPlanner.method_revision = METHOD_REVISION
    implementation.E549LatentRhoPlanner.planner_name = EVALUATION_ARM


def command_audit(args) -> dict:
    configure_implementation()
    report = implementation.command_audit(args)
    report.update(
        {
            "search_posterior_update_scope": PROTOCOL_VALUE[
                "search_posterior_update_scope"
            ],
            "fresh_execution_task_ids": PROTOCOL_VALUE[
                "fresh_execution_task_ids"
            ],
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
            "search_posterior_update_scope": PROTOCOL_VALUE[
                "search_posterior_update_scope"
            ],
            "fresh_execution_task_ids": PROTOCOL_VALUE[
                "fresh_execution_task_ids"
            ],
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
