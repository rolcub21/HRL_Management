#!/usr/bin/env python3
"""Lifecycle shim for the 30-instance frozen-lambda confirmation.

The frozen evaluator inherited V2.3's validation-coordinate helper, whose
domain is the old 12-instance development grid.  The confirmation grid has
30 instances.  This shim changes only that coordinate-domain constant while
the evaluator runs; policy RNG values, checkpoints, instances, metrics, and
the predeclared decision rule remain unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import evaluate_vcg_unified_frozen_lambda_confirmation as evaluation
import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23


REPAIR_PROTOCOL = "vcg_unified_frozen_lambda_005_confirmation_87k_grid_bound_repair_v1"
REPAIR_NAME = "normalization-grid-repair.json"


class ConfirmationRepairError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmationRepairError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise ConfirmationRepairError(f"{path} must contain an object")
    return value


def _repair_payload(output: Path) -> dict:
    contract_path = output / "confirmation-contract.json"
    activation_path = output / "panel-activation.json"
    manifest_path = output / "episode-instance-manifest.json"
    contract = _read(contract_path)
    activation = _read(activation_path)
    manifest = _read(manifest_path)
    if (
        contract.get("protocol") != evaluation.PROTOCOL
        or activation.get("contract_sha256") != contract.get("contract_sha256")
        or manifest.get("contract_sha256") != contract.get("contract_sha256")
        or manifest.get("instance_count") != 30
        or activation.get("panel_opened") is not True
    ):
        raise ConfirmationRepairError("confirmation parent artifacts mismatch")
    result = {
        "schema_version": 1,
        "repair_protocol": REPAIR_PROTOCOL,
        "parent_protocol": evaluation.PROTOCOL,
        "parent_contract_sha256": contract["contract_sha256"],
        "parent_contract_raw_sha256": _sha(contract_path),
        "panel_activation_raw_sha256": _sha(activation_path),
        "instance_manifest_sha256": manifest["manifest_sha256"],
        "instance_manifest_raw_sha256": _sha(manifest_path),
        "frozen_evaluator_source_sha256": _sha(Path(evaluation.__file__)),
        "repair_shim_source_sha256": _sha(Path(__file__)),
        "defect": "inherited validation RNG coordinate helper allowed 12 instance indices",
        "mechanical_correction": "set validation coordinate domain to the predeclared 30 instance seeds",
        "policy_rng_formula_unchanged": "624000000 + 4*instance_index + rng_index",
        "checkpoints_instances_metrics_and_success_rule_unchanged": True,
        "persisted_evaluation_rows_before_repair": 0,
        "training_or_learning": False,
        "panel_was_already_materialized": True,
        "panel_metrics_observed_before_repair": False,
    }
    result["repair_sha256"] = v23.contract_hash(result)
    return result


def _install_or_validate_repair(output: Path) -> dict:
    output = Path(output).resolve()
    path = output / REPAIR_NAME
    expected = _repair_payload(output)
    if path.exists():
        observed = _read(path)
        if v23._json_safe(observed) != v23._json_safe(expected):
            raise ConfirmationRepairError("normalization repair record drifted")
    else:
        ledger_root = output / "validation-ledger"
        if ledger_root.exists() and any(ledger_root.rglob("*.json")):
            raise ConfirmationRepairError("cannot install repair after persisted rows")
        if (output / "confirmation-report.json").exists():
            raise ConfirmationRepairError("cannot install repair after analysis")
        atomic_io._atomic_json(expected, path)
    return expected


@contextmanager
def _thirty_instance_coordinate_domain():
    previous = v23.DEFAULT_VALIDATION_SEEDS
    try:
        v23.DEFAULT_VALIDATION_SEEDS = evaluation.INSTANCE_SEEDS
        yield
    finally:
        v23.DEFAULT_VALIDATION_SEEDS = previous


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = evaluation.build_parser().parse_args(argv)
    if args.action == "prepare":
        result: Mapping = evaluation.prepare(
            args.output_dir, args.parent_root, device=args.device
        )
    else:
        repair = _install_or_validate_repair(Path(args.output_dir))
        with _thirty_instance_coordinate_domain():
            if args.action == "run":
                result = {
                    "normalization_repair": repair,
                    "evaluation": evaluation.evaluate(
                        args.output_dir, args.parent_root, device=args.device
                    ),
                    "analysis": evaluation.analyze(
                        args.output_dir, args.parent_root, device=args.device
                    ),
                }
            else:
                result = {
                    "normalization_repair": repair,
                    "analysis": evaluation.analyze(
                        args.output_dir, args.parent_root, device=args.device
                    ),
                }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
