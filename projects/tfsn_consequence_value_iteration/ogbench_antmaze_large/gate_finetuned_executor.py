#!/usr/bin/env python3
"""Apply the predeclared chained-CRL executor gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_executor import chain_inversion_summary, sha256


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def inversion(audit: dict) -> dict:
    return audit.get("chain_inversion") or chain_inversion_summary(audit["rows"])


def gate(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    executor = json.loads(args.executor.read_text())
    if not baseline.get("passed") or not candidate.get("passed"):
        raise RuntimeError("both chained audits must pass conformance before gating")
    if candidate["executor_checkpoint_sha256"] != executor["checkpoint_sha256"]:
        raise RuntimeError("candidate audit did not evaluate the fine-tuned checkpoint")
    baseline_inversion = inversion(baseline)
    candidate_inversion = inversion(candidate)
    maximum = float(protocol["gate"]["maximum_chain_inversion_rate"])
    minimum_contact = float(protocol["gate"]["minimum_per_command_contact_rate"])
    checks = {
        "chain_inversion_rate_at_most_predeclared_maximum": float(
            candidate_inversion["chain_inversion_rate"]
        ) <= maximum,
        "per_command_contact_rate_no_worse_than_frozen_baseline": float(
            candidate["overall"]["contact_rate_at_evaluator_radius"]
        ) >= minimum_contact,
        "same_frozen_bank": candidate["bank_sha256"] == baseline["bank_sha256"],
        "no_native_panel_access": not bool(
            candidate["native_evaluation_panel_accessed"]
        ),
        "no_task_reward": not bool(candidate["task_reward_used"]),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "finetuning_environment_steps": executor["finetuning_environment_steps"],
        "reported_total_executor_steps": executor["reported_total_executor_steps"],
        "executor_checkpoint": executor["checkpoint"],
        "executor_checkpoint_sha256": executor["checkpoint_sha256"],
        "baseline_audit": str(args.baseline.resolve()),
        "baseline_audit_sha256": sha256(args.baseline),
        "candidate_audit": str(args.candidate.resolve()),
        "candidate_audit_sha256": sha256(args.candidate),
        "thresholds_frozen_before_finetuning": {
            "maximum_chain_inversion_rate": maximum,
            "minimum_per_command_contact_rate": minimum_contact,
        },
        "baseline": {
            "chain_inversion_rate": baseline_inversion["chain_inversion_rate"],
            "per_command_contact_rate": baseline["overall"][
                "contact_rate_at_evaluator_radius"
            ],
        },
        "candidate": {
            "chain_inversion_rate": candidate_inversion["chain_inversion_rate"],
            "chains_with_inversion": candidate_inversion["chains_with_inversion"],
            "inverted_chains_recovered_by_chain_end": candidate_inversion[
                "inverted_chains_recovered_by_chain_end"
            ],
            "per_command_contact_rate": candidate["overall"][
                "contact_rate_at_evaluator_radius"
            ],
            "positive_progress_rate": candidate["overall"]["positive_progress_rate"],
            "endpoint_error_mean_m": candidate["overall"]["endpoint_error_mean_m"],
        },
        "checks": checks,
        "passed": all(checks.values()),
        "next_action": (
            "freeze_executor_and_start_reward_free_TF_SN_preparation"
            if all(checks.values())
            else (
                protocol["gate"].get(
                    "next_action_after_500000_failure",
                    "continue_to_the_predeclared_1000000_step_checkpoint",
                )
                if int(executor["finetuning_environment_steps"]) == 500000
                else "audit_the_chained_data_contract_before_any_more_training"
            )
        ),
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    gate(parser.parse_args())


if __name__ == "__main__":
    main()
