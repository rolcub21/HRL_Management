#!/usr/bin/env python3
"""Confirm the model-faithful E12 macro-execution correction.

The original E12 and execution-audit artifacts are immutable parents.  This
runner deliberately authenticates their saved hashes without rebuilding their
historical source contracts, because the executor sources are the factor being
changed.  It reuses the same frozen instances, checkpoints, selector, and four
predeclared audit cases; there is no training or checkpoint selection.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    audit_execution_consistency as original_audit,
    evaluate as core,
    program,
)
from methods.conditioned_vcg.representation_ablation import (
    FULL_RELATIONAL_SUCCESSOR,
)
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.relocation_family_certification import physical_recovery_state
from PSLAP.viability import RecoveryState
from PSLAP.viability_dataset import recovery_state_to_dict
from PSLAP.viability_filter import online_fixed_obstacles
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e12_execution_fix_confirmation_94k_v1"
SCHEMA_VERSION = 1
DEFAULT_E12_OUTPUT = program.DEFAULT_OUTPUT
DEFAULT_AUDIT_OUTPUT = original_audit.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e12-execution-fix-confirmation-94k"
)
EXPECTED_DELIVERIES = e11.EXPECTED_BLOCKS

SOURCE_PATHS = (
    "benchmark_viability_critic_priority.py",
    "train_viability_graph_smdp.py",
    "example/Options/certified_path.py",
    "example/Options/DirectDeliverOption.py",
    "example/Options/ReconfigureOption.py",
    "example/small_rooms_env.py",
    "PSLAP/viability.py",
    "PSLAP/viability_candidates.py",
    "PSLAP/viability_filter.py",
    "experiments/conditioned_vcg/E12_representation_ablation_94k/"
    "audit_execution_consistency.py",
    "experiments/conditioned_vcg/E12_representation_ablation_94k/"
    "confirm_execution_fix.py",
)


class ExecutionFixConfirmationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ExecutionFixConfirmationError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_self_hashed(path: Path, hash_field: str, label: str) -> dict:
    value = program.load_json(path, label=label)
    if value.get(hash_field) != program.digest(value, hash_field=hash_field):
        raise ExecutionFixConfirmationError(f"{label} self-hash mismatch")
    return value


def _frozen_parents(e12_output: Path, audit_output: Path) -> dict:
    e12_contract = _load_self_hashed(
        e12_output / program.CONTRACT_NAME,
        "contract_sha256",
        "frozen E12 contract",
    )
    if e12_contract.get("protocol") != program.PROTOCOL:
        raise ExecutionFixConfirmationError("unexpected frozen E12 protocol")

    e11_manifest_path = e11.DEFAULT_OUTPUT / e11.MANIFEST_NAME
    e11_manifest = _load_self_hashed(
        e11_manifest_path, "manifest_sha256", "frozen E11 manifest"
    )
    if (
        _sha256(e11_manifest_path) != e12_contract["e11_manifest_sha256"]
        or len(e11_manifest.get("records", ()))
        != len(e11.REGIMES) * len(e11.INSTANCE_SEEDS)
    ):
        raise ExecutionFixConfirmationError("frozen E11 parent binding changed")

    audit_contract_path = audit_output / "audit-contract.json"
    audit_report_path = audit_output / "execution-consistency-report.json"
    audit_contract = _load_self_hashed(
        audit_contract_path, "contract_sha256", "original execution audit contract"
    )
    audit_report = _load_self_hashed(
        audit_report_path, "report_sha256", "original execution audit report"
    )
    if (
        audit_contract.get("protocol") != original_audit.PROTOCOL
        or audit_report.get("contract_sha256")
        != audit_contract["contract_sha256"]
        or audit_report.get("status") != "complete"
        or audit_report.get("observed_failures_reproduced") != 3
        or not audit_report.get("instrumentation_preserved_parent_behavior")
    ):
        raise ExecutionFixConfirmationError("original execution audit is incomplete")

    audit_runs = {}
    for case in audit_contract["cases"]:
        case_id = str(case["case_id"])
        run_path = audit_output / "runs" / f"{case_id}.json"
        run = _load_self_hashed(run_path, "run_sha256", "original audit run")
        if (
            run.get("contract_sha256") != audit_contract["contract_sha256"]
            or run.get("case", {}).get("case_id") != case_id
        ):
            raise ExecutionFixConfirmationError("original audit run binding changed")
        audit_runs[case_id] = run

    return {
        "e12_contract": e12_contract,
        "e11_manifest": e11_manifest,
        "e11_manifest_path": e11_manifest_path,
        "audit_contract": audit_contract,
        "audit_contract_path": audit_contract_path,
        "audit_report": audit_report,
        "audit_report_path": audit_report_path,
        "audit_runs": audit_runs,
    }


def _contract(e12_output: Path, audit_output: Path) -> dict:
    parents = _frozen_parents(e12_output, audit_output)
    cases = []
    for case in parents["audit_contract"]["cases"]:
        original_run = parents["audit_runs"][case["case_id"]]
        cases.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "regime_id": case["regime_id"],
                "instance_seed": case["instance_seed"],
                "model_seed": case["model_seed"],
                "preference_lambda": case["preference_lambda"],
                "episode_instance_id": case["episode_instance_id"],
                "original_audit_run_sha256": original_run["run_sha256"],
                "original_selected_macro_keys": [
                    item["candidate_key"] for item in original_run["macros"]
                ],
            }
        )
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": (
            "does_binding_the_certified_paths_restore_model_faithful_"
            "execution_on_the_predeclared_E12_failure_cases"
        ),
        "factor_changed": (
            "delivery_and_relocation_execute_certified_initial_paths_and_"
            "repairs_respect_the_same_fixed_obstacles"
        ),
        "frozen_factors": [
            "E11_episode_instances",
            "E12_full_relational_successor_checkpoints",
            "selector",
            "preference_values",
            "candidate_generator",
            "exact_verifier",
            "liveness_guard",
        ],
        "training_or_checkpoint_selection": False,
        "cases": cases,
        "acceptance": {
            "strict_safe_complete": "4/4",
            "macro_failures": 0,
            "illegal_drops": 0,
            "deliveries_per_case": EXPECTED_DELIVERIES,
            "successful_recovery_macros_match_certified_physical_successor": True,
        },
        "parents": {
            "frozen_e12_contract_file_sha256": _sha256(
                e12_output / program.CONTRACT_NAME
            ),
            "frozen_e11_manifest_file_sha256": _sha256(
                parents["e11_manifest_path"]
            ),
            "original_audit_contract_file_sha256": _sha256(
                parents["audit_contract_path"]
            ),
            "original_audit_report_file_sha256": _sha256(
                parents["audit_report_path"]
            ),
        },
        "source_sha256": {
            path: _sha256(PROJECT_ROOT / path) for path in SOURCE_PATHS
        },
    }
    return program.with_hash(semantic, "contract_sha256")


def prepare(output: Path, e12_output: Path, audit_output: Path) -> dict:
    output = output.resolve()
    expected = _contract(e12_output.resolve(), audit_output.resolve())
    path = output / "confirmation-contract.json"
    if path.is_file():
        observed = program.load_json(path, label="execution-fix contract")
        if observed != expected:
            raise ExecutionFixConfirmationError(
                "execution-fix contract, parents, or sources changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ExecutionFixConfirmationError(
                "nonempty execution-fix output has no contract"
            )
        program.atomic_json(path, expected)
    return {
        "status": "prepared",
        "inference_runs": len(expected["cases"]),
        "training_runs": 0,
        "output": str(output),
    }


def _authenticate(output: Path, e12_output: Path, audit_output: Path) -> dict:
    observed = _load_self_hashed(
        output / "confirmation-contract.json",
        "contract_sha256",
        "execution-fix contract",
    )
    if observed != _contract(e12_output.resolve(), audit_output.resolve()):
        raise ExecutionFixConfirmationError(
            "execution-fix contract, parents, or sources changed"
        )
    return observed


def _record_for_case(manifest: Mapping, case: Mapping) -> Mapping:
    matches = [
        item
        for item in manifest["records"]
        if item["regime_id"] == case["regime_id"]
        and int(item["seed"]) == int(case["instance_seed"])
    ]
    if len(matches) != 1:
        raise ExecutionFixConfirmationError("frozen instance coordinate missing")
    return matches[0]


def _live_recovery_state(env) -> RecoveryState:
    yard = YardSnapshot.from_env(env)
    agent = tuple(env.current_state)
    fixed = online_fixed_obstacles(env, reserve_queue_cells=True) - {agent}
    return RecoveryState.from_yard_snapshot(
        yard,
        agent,
        fixed_obstacles=fixed,
        reserved_cells=(),
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    )


class SuccessorTracker:
    def __init__(self) -> None:
        self.records = []

    def execute(self, original, env, candidate, **kwargs):
        expected = candidate.successor_state
        execution = original(env, candidate, **kwargs)
        actual = _live_recovery_state(env)
        self.records.append(
            {
                "candidate_key": candidate.key,
                "action_type": candidate.action_type.value,
                "recovery_action": original_audit._recovery_action_signature(
                    candidate.recovery_action
                ),
                "execution": original_audit._jsonable(asdict(execution)),
                "option_outcome": original_audit._jsonable(
                    getattr(candidate.option, "last_outcome", None)
                ),
                "certified_successor": recovery_state_to_dict(expected),
                "actual_successor": recovery_state_to_dict(actual),
                "exact_successor_match": actual == expected,
                "physical_successor_match": (
                    physical_recovery_state(actual)
                    == physical_recovery_state(expected)
                ),
            }
        )
        return execution


@contextmanager
def _track_successors(tracker: SuccessorTracker):
    original = benchmark.execute_certified_macro

    def traced(env, candidate, **kwargs):
        return tracker.execute(original, env, candidate, **kwargs)

    with patch.object(benchmark, "execute_certified_macro", traced):
        yield


def _run_path(output: Path, case_id: str) -> Path:
    return output / "runs" / f"{case_id}.json"


def _run_one(
    output: Path,
    e12_output: Path,
    contract: Mapping,
    parents: Mapping,
    case: Mapping,
    *,
    device: torch.device,
) -> dict:
    path = _run_path(output, case["case_id"])
    if path.is_file():
        value = _load_self_hashed(path, "run_sha256", "execution-fix run")
        if value.get("contract_sha256") != contract["contract_sha256"]:
            raise ExecutionFixConfirmationError("execution-fix run binding changed")
        return value

    record = _record_for_case(parents["e11_manifest"], case)
    instance = e11._load_instance(e11.DEFAULT_OUTPUT, record)
    op_payload, factory, checkpoint_hashes = core._checkpoint_factory(
        e12_output,
        variant=FULL_RELATIONAL_SUCCESSOR,
        model_seed=int(case["model_seed"]),
        value=float(case["preference_lambda"]),
        device=device,
    )
    tracker = SuccessorTracker()
    regime = e11.REGIME_BY_ID[case["regime_id"]]
    with core._runtime(factory, regime), _track_successors(tracker):
        raw = benchmark.run_arm(
            arm=benchmark.EXACT_FULL,
            controller_payload=op_payload,
            instance=instance,
            instance_seed=int(instance.seed),
            search_config=benchmark._search_config(op_payload),
            liveness_rule=benchmark._liveness_rule(op_payload),
            prioritizer=None,
            max_steps=program.MAX_STEPS,
            device=device,
        )
    selected = [item["selected_key"] for item in raw["decisions"]]
    original_selected = list(case["original_selected_macro_keys"])
    common = min(len(selected), len(original_selected))
    first_divergence = next(
        (
            index
            for index in range(common)
            if selected[index] != original_selected[index]
        ),
        None,
    )
    if first_divergence is None and len(selected) != len(original_selected):
        first_divergence = common
    strict = bool(
        raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and int(raw["macro_failures"]) == 0
        and int(raw["illegal_drops"]) == 0
        and len(raw["delivery_deviations"]) == EXPECTED_DELIVERIES
    )
    value = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "case": dict(case),
            "episode_instance_id": instance.instance_id,
            "checkpoints": checkpoint_hashes,
            "result": {
                "strict_safe_complete": strict,
                "method_failure_reason": raw["method_failure_reason"],
                "behavior_digest": raw["behavior_digest"],
                "macro_decisions": int(raw["macro_decisions"]),
                "steps": int(raw["steps"]),
                "deliveries": len(raw["delivery_deviations"]),
                "macro_failures": int(raw["macro_failures"]),
                "illegal_drops": int(raw["illegal_drops"]),
                "all_frontiers_exact": bool(
                    raw["complete_frontier_exactly_verified"]
                ),
                "successful_macro_physical_successor_matches": sum(
                    item["execution"]["option_success"]
                    and item["physical_successor_match"]
                    for item in tracker.records
                ),
                "successful_macros": sum(
                    item["execution"]["option_success"]
                    for item in tracker.records
                ),
                "first_selected_macro_divergence_from_original": first_divergence,
                "original_selected_macro_count": len(original_selected),
            },
            "macros": tracker.records,
        },
        "run_sha256",
    )
    program.atomic_json(path, value)
    return value


def run(
    output: Path,
    e12_output: Path,
    audit_output: Path,
    *,
    device_name: str,
    case_ids: Sequence[str],
) -> dict:
    prepare(output, e12_output, audit_output)
    contract = _authenticate(output, e12_output, audit_output)
    parents = _frozen_parents(e12_output, audit_output)
    case_map = {item["case_id"]: item for item in contract["cases"]}
    selected = tuple(case_ids) if case_ids else tuple(case_map)
    unknown = set(selected) - set(case_map)
    if unknown:
        raise ExecutionFixConfirmationError(f"unknown cases: {sorted(unknown)}")
    device = pilot._device(device_name)
    for index, case_id in enumerate(selected, start=1):
        result = _run_one(
            output.resolve(),
            e12_output.resolve(),
            contract,
            parents,
            case_map[case_id],
            device=device,
        )
        print(
            json.dumps(
                {
                    "case": f"{index}/{len(selected)}",
                    "case_id": case_id,
                    "strict": result["result"]["strict_safe_complete"],
                    "deliveries": result["result"]["deliveries"],
                    "macros": result["result"]["macro_decisions"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return analyze(output, e12_output, audit_output, allow_partial=bool(case_ids))


def analyze(
    output: Path,
    e12_output: Path,
    audit_output: Path,
    *,
    allow_partial: bool,
) -> dict:
    contract = _authenticate(output, e12_output, audit_output)
    rows = []
    missing = []
    for case in contract["cases"]:
        path = _run_path(output, case["case_id"])
        if not path.is_file():
            missing.append(case["case_id"])
            continue
        row = _load_self_hashed(path, "run_sha256", "execution-fix run")
        if row.get("contract_sha256") != contract["contract_sha256"]:
            raise ExecutionFixConfirmationError("execution-fix run binding changed")
        rows.append(row)
    if missing and not allow_partial:
        raise ExecutionFixConfirmationError(f"missing cases: {missing}")
    total_successful = sum(
        row["result"]["successful_macros"] for row in rows
    )
    total_matching = sum(
        row["result"]["successful_macro_physical_successor_matches"]
        for row in rows
    )
    report = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if missing else "complete",
            "completed_cases": len(rows),
            "expected_cases": len(contract["cases"]),
            "missing_cases": missing,
            "strict_safe_complete": sum(
                row["result"]["strict_safe_complete"] for row in rows
            ),
            "successful_macros": total_successful,
            "successful_macro_physical_successor_matches": total_matching,
            "all_successful_macros_match_certified_physical_successor": bool(
                total_successful == total_matching
            ),
            "acceptance_passed": bool(
                not missing
                and len(rows) == len(contract["cases"])
                and all(row["result"]["strict_safe_complete"] for row in rows)
                and total_successful == total_matching
            ),
            "cases": [
                {
                    "case_id": row["case"]["case_id"],
                    **row["result"],
                }
                for row in rows
            ],
        },
        "report_sha256",
    )
    program.atomic_json(output / "execution-fix-report.json", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e12-output", type=Path, default=DEFAULT_E12_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--case", action="append", choices=tuple(item["case_id"] for item in original_audit.CASES)
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output, args.e12_output, args.audit_output)
    elif args.command == "run":
        result = run(
            args.output,
            args.e12_output,
            args.audit_output,
            device_name=args.device,
            case_ids=tuple(args.case or ()),
        )
    else:
        result = analyze(
            args.output,
            args.e12_output,
            args.audit_output,
            allow_partial=args.allow_partial,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
