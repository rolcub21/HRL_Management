#!/usr/bin/env python3
"""R61 phase-correct latent-radius search, reusing proven-equivalent R60 rows."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    r60_e549_latent_rho_all9 as r60,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT_SHA256,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r61_e549_phase_separated_latent_rho_protocol.json"
REFERENCE = HERE / "R61_E549_PHASE_SEPARATED_LATENT_RHO_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_5m_phase_separated_latent_rho_original9_r61"
EVALUATION_ARM = "phase_separated_latent_rho_with_E549_R52_R51"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_phase_separated_latent_rho_r61/original9"
)
R60_OUTPUT = r60.DEFAULT_OUTPUT


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "online_correction_revision": r51.METHOD_REVISION,
        "oracle_inputs": [],
        "task_ids": list(range(9)),
        "execution_reuse_task_ids": list(range(7)),
        "fresh_execution_task_ids": [7, 8],
        "episodes": 5,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R61 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R61 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def install_r61_contract() -> None:
    """Install R61 metadata and phase behavior in the shared R60 runtime."""
    r60.PROTOCOL = PROTOCOL
    r60.REFERENCE = REFERENCE
    r60.METHOD_REVISION = METHOD_REVISION
    r60.EVALUATION_ARM = EVALUATION_ARM
    r60.PROTOCOL_VALUE = PROTOCOL_VALUE
    r60.RETIRE_SEARCH_POSTERIOR_AT_FIRST_CONTACT = True
    r60.E549LatentRhoPlanner.method_revision = METHOD_REVISION
    r60.E549LatentRhoPlanner.planner_name = EVALUATION_ARM


def _read_log(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _validate_execution_equivalence(task_id: int) -> tuple[dict, Path, Path]:
    task_path = R60_OUTPUT / f"task_{task_id:03d}.json"
    log_path = R60_OUTPUT / f"task_{task_id:03d}.jsonl"
    if not task_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(
            f"R61 execution reuse requires completed R60 task {task_id}"
        )
    row = json.loads(task_path.read_text())
    expected = {
        "task_id": task_id,
        "method_revision": "antmaze_e549_5m_latent_rho_original9_r60",
        "episodes": PROTOCOL_VALUE["episodes"],
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
    }
    for key, item in expected.items():
        if row.get(key) != item:
            raise ValueError(f"R60 reuse task {task_id} has another {key}")
    logs = _read_log(log_path)
    positives = [index for index, item in enumerate(logs) if item["reward"]]
    contact = row["first_contact_episode"]
    if contact is None:
        if positives or any(item["phase"] != "acquisition" for item in logs):
            raise ValueError(
                f"R60 no-contact task {task_id} is not R61-equivalent"
            )
    else:
        if not positives:
            raise ValueError(f"R60 contact task {task_id} has no positive log")
        first = positives[0]
        if logs[first]["phase"] != "acquisition":
            raise ValueError(
                f"R60 task {task_id} first contact was not acquisition"
            )
        if any(item["reward"] for item in logs[:first]):
            raise ValueError(f"R60 task {task_id} has an earlier contact")
        if any(
            item["phase"] == "acquisition" for item in logs[first + 1 :]
        ):
            raise ValueError(
                f"R60 task {task_id} made acquisition decisions after contact"
            )
    return row, task_path, log_path


def import_equivalent_r60_tasks(output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in PROTOCOL_VALUE["execution_reuse_task_ids"]:
        destination = output / f"task_{task_id:03d}.json"
        destination_log = output / f"task_{task_id:03d}.jsonl"
        row, source, source_log = _validate_execution_equivalence(task_id)
        record = {
            "task_id": task_id,
            "source_task": str(source),
            "source_task_sha256": r52.sha256(source),
            "source_log": str(source_log),
            "source_log_sha256": r52.sha256(source_log),
            "equivalence_reason": (
                "no_contact_negative_updates_are_identical"
                if row["first_contact_episode"] is None
                else "identical_through_first_contact_and_return_does_not_read_search_posterior"
            ),
        }
        records.append(record)
        if destination.is_file():
            saved = json.loads(destination.read_text())
            if saved.get("execution_reuse") != record:
                raise ValueError(f"saved R61 reuse record changed for task {task_id}")
            if not destination_log.is_file():
                raise FileNotFoundError(destination_log)
            continue
        migrated = dict(row)
        for key in (
            "feasible_goal_candidates",
            "posterior_exhausted",
            "posterior_exhaustion_behavior",
            "posterior_marginal_evidence",
        ):
            migrated.pop(key, None)
        migrated.update(
            {
                "evaluation_arm": EVALUATION_ARM,
                "method_reference": str(REFERENCE),
                "method_reference_sha256": r52.sha256(REFERENCE),
                "method_revision": METHOD_REVISION,
                "claim_level": PROTOCOL_VALUE["claim_level"],
                "selection_rule": PROTOCOL_VALUE["selection_rule"],
                "declared_deviations": PROTOCOL_VALUE[
                    "declared_deviations"
                ],
                "search_posterior_update_scope": PROTOCOL_VALUE[
                    "search_posterior_update_scope"
                ],
                "posterior_diagnostic_status": (
                    "not_reconstructed_for_execution_equivalent_reuse"
                ),
                "execution_reuse": record,
            }
        )
        r48.dump(destination, migrated)
        shutil.copy2(source_log, destination_log)
    r48.dump(output / "EXECUTION_REUSE_AUDIT.json", {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "passed": True,
        "records": records,
    })
    return records


def command_audit(args) -> dict:
    install_r61_contract()
    report = r60.command_audit(args)
    report.update(
        {
            "search_posterior_update_scope": PROTOCOL_VALUE[
                "search_posterior_update_scope"
            ],
            "execution_reuse_task_ids": PROTOCOL_VALUE[
                "execution_reuse_task_ids"
            ],
            "fresh_execution_task_ids": PROTOCOL_VALUE[
                "fresh_execution_task_ids"
            ],
        }
    )
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(args) -> dict:
    install_r61_contract()
    output = Path(args.output).resolve()
    reuse = import_equivalent_r60_tasks(output)
    result = r60.command_evaluate(args)
    result.update(
        {
            "search_posterior_update_scope": PROTOCOL_VALUE[
                "search_posterior_update_scope"
            ],
            "execution_reuse_audit_passed": True,
            "execution_reused_task_ids": [
                row["task_id"] for row in reuse
            ],
            "fresh_execution_task_ids": PROTOCOL_VALUE[
                "fresh_execution_task_ids"
            ],
        }
    )
    r48.dump(output / "summary.json", result)
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
