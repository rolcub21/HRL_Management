#!/usr/bin/env python3
"""Post-freeze schema repair for the VCG V2.3 seed-stability analysis.

The frozen rollout schema stores signed ``delivery_deviations`` and their MAE,
whereas the frozen analyzer also expects three deterministic timing summaries.
This module never changes that frozen source or any rollout artifact.  It
authenticates them, derives the missing summaries in memory, executes the
otherwise unchanged frozen analysis, and writes a separate versioned result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import vcg_v2_3_seed_stability as frozen
from vcg_v2_3_seed_stability_protocol import (
    MODEL_SEEDS,
    StabilityProtocolError,
    atomic_json,
    digest_json,
    load_json,
    sha256_file,
    verify_self_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL = "vcg_v2_3_seed_stability_postfreeze_analysis_schema_repair_v2"
SCHEMA_VERSION = 2
TARGET_WINDOW = 20.0
DEFAULT_ORIGINAL_OUTPUT_DIR = (
    PROJECT_ROOT / "results/vcg-v2-3-seed-stability-85k"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/vcg-v2-3-seed-stability-analysis-repair-v2-85k"
)
DEFAULT_FREEZE_SPEC = frozen.DEFAULT_FREEZE_SPEC
DEFAULT_REPAIR_DIR = frozen.DEFAULT_REPAIR_DIR
CONTRACT_NAME = "analysis-repair-contract.json"
REPORT_NAME = "analysis-v2-report.json"
AUDIT_NAME = "analysis-v2-audit.json"
_DERIVED_FIELDS = (
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
)


def _json_equivalent(value):
    if isinstance(value, Mapping):
        return {str(key): _json_equivalent(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_equivalent(item) for item in value]
    return value


def _require(observed, expected, *, name: str) -> None:
    if _json_equivalent(observed) != _json_equivalent(expected):
        raise StabilityProtocolError(
            f"{name} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _canonical_root(path: Path, *, name: str, require_exists: bool) -> Path:
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise StabilityProtocolError(f"{name} must not be a symlink")
    resolved = requested.resolve()
    if requested != resolved:
        raise StabilityProtocolError(f"{name} must be canonical")
    if require_exists and not resolved.is_dir():
        raise StabilityProtocolError(f"{name} is missing")
    if resolved.exists() and not resolved.is_dir():
        raise StabilityProtocolError(f"{name} is not a directory")
    return resolved


def _require_regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise StabilityProtocolError(f"{name} must be a regular non-symlink file")


def _validate_repair_tree(output_dir: Path, *, phase: str) -> None:
    root = _canonical_root(
        output_dir, name="analysis-repair output root", require_exists=True,
    )
    entries = {path.name: path for path in root.iterdir()}
    expected_by_phase = {
        "prepared": {CONTRACT_NAME},
        "report_written": {CONTRACT_NAME, REPORT_NAME},
        "complete": {CONTRACT_NAME, REPORT_NAME, AUDIT_NAME},
    }
    if phase not in expected_by_phase:
        raise StabilityProtocolError(f"unknown analysis-repair phase: {phase}")
    _require(set(entries), expected_by_phase[phase], name="analysis-repair file set")
    for name, path in entries.items():
        _require_regular_file(path, name=f"analysis-repair artifact {name}")


def _runtime_source_paths() -> dict[str, Path]:
    return {
        "vcg_v2_3_seed_stability_analysis_repair_v2.py": Path(__file__).resolve(),
        "experiments/vcg_v2_3_seed_stability_analysis_repair_v2/run.sh": (
            PROJECT_ROOT
            / "experiments/vcg_v2_3_seed_stability_analysis_repair_v2/run.sh"
        ),
    }


def _runtime_source_hashes() -> dict[str, str]:
    result = {}
    for name, path in _runtime_source_paths().items():
        _require_regular_file(path, name=f"analysis-repair source {name}")
        result[name] = sha256_file(path)
    return result


def _require_output_disjoint_from_inputs(
    *, output_dir: Path, original_output_dir: Path, freeze_spec_path: Path,
    repair_dir: Path,
) -> None:
    """Reject any output capable of adding to or containing a trust tree."""

    output = Path(output_dir).resolve()
    protected_directories = (
        Path(original_output_dir).resolve(),
        Path(repair_dir).resolve(),
        Path(freeze_spec_path).resolve().parent,
        _runtime_source_paths()[
            "experiments/vcg_v2_3_seed_stability_analysis_repair_v2/run.sh"
        ].resolve().parent,
    )
    for protected in protected_directories:
        if output == protected or output in protected.parents or protected in output.parents:
            raise StabilityProtocolError(
                f"analysis-repair output root overlaps a protected input tree: {protected}"
            )
    # The analyzer is at the project root, whose sibling `results/` is the
    # intended output location.  Protect the file itself and reject an output
    # ancestor that could contain/replace it without forbidding that sibling.
    analyzer = Path(__file__).resolve()
    if output == analyzer or output in analyzer.parents:
        raise StabilityProtocolError(
            "analysis-repair output root contains a runtime source"
        )


def _original_inventory(original_output_dir: Path) -> dict[str, str]:
    root = _canonical_root(
        original_output_dir, name="original stability output root", require_exists=True,
    )
    expected_top = {
        frozen.RUN_MANIFEST_NAME,
        *(f"seed-{seed}" for seed in MODEL_SEEDS),
    }
    entries = {path.name: path for path in root.iterdir()}
    _require(set(entries), expected_top, name="original stability top-level tree")
    manifest_path = entries[frozen.RUN_MANIFEST_NAME]
    _require_regular_file(manifest_path, name="original stability run manifest")
    inventory = {frozen.RUN_MANIFEST_NAME: sha256_file(manifest_path)}
    for seed in MODEL_SEEDS:
        seed_name = f"seed-{seed}"
        seed_dir = entries[seed_name]
        if seed_dir.is_symlink() or not seed_dir.is_dir():
            raise StabilityProtocolError(f"{seed_name} must be a regular directory")
        tree = frozen._seed_tree_raw_sha256(seed_dir, allow_completion=True)
        completion_path = seed_dir / frozen.COMPLETION_NAME
        _require_regular_file(completion_path, name=f"seed {seed} completion")
        for relative, raw_sha in tree.items():
            inventory[f"{seed_name}/{relative}"] = raw_sha
        inventory[f"{seed_name}/{frozen.COMPLETION_NAME}"] = sha256_file(
            completion_path
        )
    return dict(sorted(inventory.items()))


def _authenticate_inputs(
    *, original_output_dir: Path, freeze_spec_path: Path, repair_dir: Path,
) -> dict:
    original_output_dir = _canonical_root(
        original_output_dir, name="original stability output root", require_exists=True,
    )
    inventory = _original_inventory(original_output_dir)
    manifest, binding, repair = frozen.authenticate_run_manifest(
        output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    completions = {}
    selections = {}
    for seed in MODEL_SEEDS:
        completion, selected = frozen.authenticate_seed_completion(
            seed,
            output_dir=original_output_dir,
            run_manifest=manifest,
            binding=binding,
        )
        completions[str(seed)] = completion
        selections[str(seed)] = selected
        training_contract = load_json(
            original_output_dir / f"seed-{seed}" / "training-contract.json",
            name=f"seed {seed} training contract",
        )
        objective = training_contract.get("dense_objective_spec")
        if (
            not isinstance(objective, Mapping)
            or isinstance(objective.get("window"), bool)
            or not isinstance(objective.get("window"), (int, float))
            or float(objective["window"]) != TARGET_WINDOW
        ):
            raise StabilityProtocolError(
                f"seed {seed} target window does not authenticate the repair formula"
            )
    frozen._reauthenticate_run_trust(
        output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        expected_manifest=manifest,
        expected_binding=binding,
        expected_repair=repair,
    )
    _require(
        _original_inventory(original_output_dir), inventory,
        name="original stability inventory during authentication",
    )
    return {
        "root": original_output_dir,
        "manifest": manifest,
        "binding": binding,
        "repair": repair,
        "completions": completions,
        "selections": selections,
        "inventory": inventory,
    }


def _selected_input_records(authenticated: Mapping) -> tuple[dict, ...]:
    result = []
    root = Path(authenticated["root"])
    for seed in MODEL_SEEDS:
        completion = authenticated["completions"][str(seed)]
        selected = authenticated["selections"][str(seed)]
        record = {
            "model_seed": seed,
            "completion_sha256": completion["completion_sha256"],
            "completion_raw_sha256": authenticated["inventory"][
                f"seed-{seed}/{frozen.COMPLETION_NAME}"
            ],
            "development_candidate_eligible": selected is not None,
            "selected_episode": None,
            "selected_validation_ledger_relative_path": None,
            "selected_validation_ledger_raw_sha256": None,
            "selected_validation_ledger_sha256": None,
        }
        if selected is not None:
            episode = int(selected["selected_episode"])
            ledger_relative = f"seed-{seed}/validation-ledger/episode-{episode:04d}.json"
            ledger = selected["selected_diagnostic"]["validation_ledger"]
            record.update(
                {
                    "selected_episode": episode,
                    "selected_validation_ledger_relative_path": ledger_relative,
                    "selected_validation_ledger_raw_sha256": sha256_file(
                        root / ledger_relative
                    ),
                    "selected_validation_ledger_sha256": ledger["ledger_sha256"],
                }
            )
        result.append(record)
    return tuple(result)


def build_contract(
    *, output_dir: Path, original_output_dir: Path, freeze_spec_path: Path,
    repair_dir: Path,
) -> dict:
    output_dir = _canonical_root(
        output_dir, name="analysis-repair output root", require_exists=False,
    )
    original_output_dir = _canonical_root(
        original_output_dir, name="original stability output root", require_exists=True,
    )
    _require_output_disjoint_from_inputs(
        output_dir=output_dir,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    authenticated = _authenticate_inputs(
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    manifest = authenticated["manifest"]
    repair = authenticated["repair"]
    selected = _selected_input_records(authenticated)
    selected_row_count = sum(
        len(item["selected_diagnostic"]["validation_ledger"]["rows"])
        for item in authenticated["selections"].values()
        if item is not None
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "frozen_postoutcome_deterministic_analysis_schema_repair",
        "development_only": True,
        "post_outcome_repair": True,
        "training_or_evaluation_rollouts_authorized": False,
        "selection_change_authorized": False,
        "baseline_registry_change_authorized": False,
        "complete_case_filtering_authorized": False,
        "final_86xxx_panel_opened": False,
        "protected_622xxx_policy_rng_opened": False,
        "output_root": str(output_dir),
        "output_file_set": (CONTRACT_NAME, REPORT_NAME, AUDIT_NAME),
        "original_output_root": str(authenticated["root"]),
        "original_top_level_report_or_audit_present": False,
        "original_run_manifest": {
            "raw_sha256": authenticated["inventory"][frozen.RUN_MANIFEST_NAME],
            "manifest_sha256": manifest["manifest_sha256"],
            "implementation_source_sha256": manifest[
                "implementation_source_sha256"
            ],
        },
        "original_tree_raw_sha256": authenticated["inventory"],
        "selected_inputs": selected,
        "selected_row_count": selected_row_count,
        "expected_selected_row_count_if_all_seeds_eligible": 144,
        "repair_v2_trust": frozen._repair_roots(repair),
        "frozen_baseline_method_ids": frozen.BASELINE_METHODS,
        "development_seed10_is_not_a_gate_comparator": True,
        "schema_defect": {
            "classification": "producer_analyzer_derived_field_schema_mismatch",
            "producer_fields_used": (
                "delivery_deviations", "mean_absolute_error",
            ),
            "missing_analyzer_fields": _DERIVED_FIELDS,
            "target_window": TARGET_WINDOW,
            "signed_deviation_convention": "delivery_time_minus_target_time",
            "mean_tardiness_formula": "mean(max(deviation,0))",
            "mean_earliness_formula": "mean(max(-deviation,0))",
            "within_target_window_rate_formula": "mean(abs(deviation)<=20)",
            "derivation_scope": "in_memory_selected_row_copies_only",
            "stored_mae_must_equal_mean_absolute_deviation": True,
            "preexisting_derived_fields_authorized": False,
        },
        "runtime_source_sha256": _runtime_source_hashes(),
    }
    payload["contract_sha256"] = digest_json(payload)
    return payload


def prepare(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    original_output_dir: Path = DEFAULT_ORIGINAL_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> dict:
    output_dir = _canonical_root(
        output_dir, name="analysis-repair output root", require_exists=False,
    )
    expected = build_contract(
        output_dir=output_dir,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    contract_path = output_dir / CONTRACT_NAME
    if output_dir.exists():
        entries = tuple(output_dir.iterdir())
        if entries:
            _validate_repair_tree(output_dir, phase="prepared")
            observed = load_json(contract_path, name="analysis-repair contract")
            verify_self_hash(
                observed, "contract_sha256", name="analysis-repair contract",
            )
            _require(observed, expected, name="existing analysis-repair contract")
            return observed
    atomic_json(contract_path, expected)
    _validate_repair_tree(output_dir, phase="prepared")
    observed = load_json(contract_path, name="analysis-repair contract")
    verify_self_hash(observed, "contract_sha256", name="analysis-repair contract")
    _require(observed, expected, name="serialized analysis-repair contract")
    return observed


def authenticate_contract(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    original_output_dir: Path = DEFAULT_ORIGINAL_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
    phase: str,
) -> dict:
    output_dir = _canonical_root(
        output_dir, name="analysis-repair output root", require_exists=True,
    )
    _validate_repair_tree(output_dir, phase=phase)
    contract = load_json(output_dir / CONTRACT_NAME, name="analysis-repair contract")
    verify_self_hash(contract, "contract_sha256", name="analysis-repair contract")
    expected = build_contract(
        output_dir=output_dir,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    _require(contract, expected, name="analysis-repair contract")
    return contract


def derive_timing_fields(row: Mapping, *, target_window: float = TARGET_WINDOW) -> dict:
    """Return an enriched copy after fail-closed deterministic derivation."""

    if not isinstance(row, Mapping):
        raise StabilityProtocolError("selected row must be a mapping")
    if float(target_window) != TARGET_WINDOW:
        raise StabilityProtocolError("analysis-repair target window must remain 20")
    present = [field for field in _DERIVED_FIELDS if field in row]
    if present:
        raise StabilityProtocolError(
            f"selected row unexpectedly contains repaired fields: {present!r}"
        )
    deviations = row.get("delivery_deviations")
    if not isinstance(deviations, (list, tuple)) or not deviations:
        raise StabilityProtocolError("delivery_deviations must be a nonempty sequence")
    values = []
    for value in deviations:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise StabilityProtocolError("delivery_deviations contain a nonfinite value")
        values.append(float(value))
    required = row.get("required_deliveries")
    delivered = row.get("delivery_count")
    if type(required) is not int or type(delivered) is not int or required <= 0:
        raise StabilityProtocolError("delivery counts are invalid for timing derivation")
    if delivered != required or len(values) != required:
        raise StabilityProtocolError("delivery deviations/counts do not match")
    stored_mae = row.get("mean_absolute_error")
    if (
        isinstance(stored_mae, bool)
        or not isinstance(stored_mae, (int, float))
        or not math.isfinite(float(stored_mae))
    ):
        raise StabilityProtocolError("stored MAE is invalid")
    derived_mae = float(fmean(abs(value) for value in values))
    if abs(derived_mae - float(stored_mae)) > 1e-12:
        raise StabilityProtocolError("stored MAE does not match delivery deviations")
    enriched = deepcopy(dict(row))
    enriched["mean_tardiness"] = float(
        fmean(max(value, 0.0) for value in values)
    )
    enriched["mean_earliness"] = float(
        fmean(max(-value, 0.0) for value in values)
    )
    enriched["within_target_window_rate"] = float(
        fmean(abs(value) <= TARGET_WINDOW for value in values)
    )
    return enriched


def _serialized_json_bytes(payload) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _capture_frozen_analysis(
    *, original_output_dir: Path, freeze_spec_path: Path, repair_dir: Path,
) -> tuple[dict, dict, dict]:
    """Execute frozen aggregation with a temporary in-memory schema adapter."""

    original_output_dir = Path(original_output_dir).resolve()
    original_summary = frozen.summarize_selected_rows
    original_atomic_json = frozen.atomic_json
    original_sha256_file = frozen.sha256_file
    captured = {}
    derivation = {
        "selected_rows_derived": 0,
        "stored_mae_crosschecks_passed": 0,
        "derived_field_values_created": 0,
    }

    def repaired_summary(model_seed: int, rows: Sequence[Mapping]) -> dict:
        enriched = [derive_timing_fields(row) for row in rows]
        derivation["selected_rows_derived"] += len(enriched)
        derivation["stored_mae_crosschecks_passed"] += len(enriched)
        derivation["derived_field_values_created"] += len(enriched) * 3
        return original_summary(model_seed, enriched)

    def capture_atomic_json(path: Path, payload) -> None:
        resolved = Path(path).resolve()
        if resolved.parent != original_output_dir or resolved.name not in {
            frozen.REPORT_NAME, frozen.AUDIT_NAME,
        }:
            raise StabilityProtocolError(
                f"frozen analyzer attempted an unexpected write: {resolved}"
            )
        if resolved.name in captured:
            raise StabilityProtocolError("frozen analyzer attempted a duplicate write")
        captured[resolved.name] = deepcopy(payload)

    def capture_sha256_file(path: Path) -> str:
        resolved = Path(path).resolve()
        report_path = original_output_dir / frozen.REPORT_NAME
        if resolved == report_path:
            if frozen.REPORT_NAME not in captured:
                raise StabilityProtocolError("frozen report hash requested before capture")
            return hashlib.sha256(
                _serialized_json_bytes(captured[frozen.REPORT_NAME])
            ).hexdigest()
        return original_sha256_file(path)

    try:
        frozen.summarize_selected_rows = repaired_summary
        frozen.atomic_json = capture_atomic_json
        frozen.sha256_file = capture_sha256_file
        returned = frozen.analyze(
            output_dir=original_output_dir,
            freeze_spec_path=freeze_spec_path,
            repair_dir=repair_dir,
        )
    finally:
        frozen.summarize_selected_rows = original_summary
        frozen.atomic_json = original_atomic_json
        frozen.sha256_file = original_sha256_file

    _require(
        set(captured), {frozen.REPORT_NAME, frozen.AUDIT_NAME},
        name="captured frozen analyzer artifacts",
    )
    report = captured[frozen.REPORT_NAME]
    audit = captured[frozen.AUDIT_NAME]
    _require(returned, report, name="frozen analyzer returned report")
    verify_self_hash(report, "report_sha256", name="reconstructed frozen report")
    verify_self_hash(audit, "audit_sha256", name="reconstructed frozen audit")
    expected_report_raw = hashlib.sha256(_serialized_json_bytes(report)).hexdigest()
    _require(
        audit["stability_report_raw_sha256"], expected_report_raw,
        name="reconstructed frozen report raw hash",
    )
    return report, audit, derivation


def _versioned_report(
    *, frozen_report: Mapping, frozen_audit: Mapping, contract: Mapping,
    derivation: Mapping,
) -> dict:
    upstream_self = frozen_report["report_sha256"]
    upstream_raw = hashlib.sha256(_serialized_json_bytes(frozen_report)).hexdigest()
    payload = deepcopy(dict(frozen_report))
    payload.pop("report_sha256", None)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "upstream_frozen_protocol": frozen_report["protocol"],
            "upstream_frozen_schema_version": frozen_report["schema_version"],
            "upstream_reconstructed_report_sha256": upstream_self,
            "upstream_reconstructed_report_raw_sha256": upstream_raw,
            "upstream_reconstructed_audit_sha256": frozen_audit["audit_sha256"],
            "analysis_repair_contract_sha256": contract["contract_sha256"],
            "post_outcome_analysis_schema_repair": True,
            "schema_repair_changed_training_or_selection": False,
            "schema_repair_changed_baseline_registry_or_gate": False,
            "schema_repair_changed_any_persisted_input": False,
            "schema_repair_derivation": {
                **dict(contract["schema_defect"]),
                **dict(derivation),
            },
            "development_seed10_used_as_gate_comparator": False,
            "environment_rollouts_executed_during_repair": 0,
        }
    )
    payload["report_sha256"] = digest_json(payload)
    return payload


def _expected_audit(
    *, report: Mapping, contract: Mapping, before_inventory: Mapping,
    after_inventory: Mapping, frozen_report: Mapping, frozen_audit: Mapping,
    derivation: Mapping,
) -> dict:
    _require(after_inventory, before_inventory, name="original input before/after inventory")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "development_only": True,
        "post_outcome_analysis_schema_repair": True,
        "analysis_repair_contract_sha256": contract["contract_sha256"],
        "analysis_repair_contract_raw_sha256": sha256_file(
            Path(contract["output_root"]) / CONTRACT_NAME
        ),
        "analysis_v2_report_sha256": report["report_sha256"],
        "analysis_v2_report_raw_sha256": hashlib.sha256(
            _serialized_json_bytes(report)
        ).hexdigest(),
        "upstream_reconstructed_report_sha256": frozen_report["report_sha256"],
        "upstream_reconstructed_report_raw_sha256": hashlib.sha256(
            _serialized_json_bytes(frozen_report)
        ).hexdigest(),
        "upstream_reconstructed_audit_sha256": frozen_audit["audit_sha256"],
        "original_tree_raw_sha256_before": digest_json(before_inventory),
        "original_tree_raw_sha256_after": digest_json(after_inventory),
        "original_tree_file_count": len(before_inventory),
        "original_inputs_byte_identical_before_after": True,
        "all_original_sources_and_repair_roots_reauthenticated": True,
        "selected_rows_authenticated": derivation["selected_rows_derived"],
        "selected_rows_with_derived_timing": derivation["selected_rows_derived"],
        "stored_mae_crosschecks_passed": derivation[
            "stored_mae_crosschecks_passed"
        ],
        "complete_case_filtering_used": False,
        "training_or_evaluation_rollouts_executed": 0,
        "persisted_input_mutations": 0,
        "final_86xxx_panel_opened": False,
        "protected_622xxx_policy_rng_opened": False,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
    }
    payload["audit_sha256"] = digest_json(payload)
    return payload


def analyze(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    original_output_dir: Path = DEFAULT_ORIGINAL_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> dict:
    contract = authenticate_contract(
        output_dir=output_dir,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        phase="prepared",
    )
    before = _original_inventory(original_output_dir)
    _require(
        before, contract["original_tree_raw_sha256"],
        name="analysis input inventory before reconstruction",
    )
    frozen_report, frozen_audit, derivation = _capture_frozen_analysis(
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    _require(
        derivation["selected_rows_derived"], contract["selected_row_count"],
        name="derived selected row count",
    )
    report = _versioned_report(
        frozen_report=frozen_report,
        frozen_audit=frozen_audit,
        contract=contract,
        derivation=derivation,
    )
    after = _original_inventory(original_output_dir)
    _require(after, before, name="original input inventory after reconstruction")
    authenticate_contract(
        output_dir=output_dir,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        phase="prepared",
    )
    atomic_json(Path(output_dir) / REPORT_NAME, report)
    _validate_repair_tree(output_dir, phase="report_written")
    observed_report = load_json(
        Path(output_dir) / REPORT_NAME, name="analysis-v2 report",
    )
    verify_self_hash(observed_report, "report_sha256", name="analysis-v2 report")
    _require(observed_report, report, name="serialized analysis-v2 report")

    final_inputs = _authenticate_inputs(
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    _require(
        final_inputs["inventory"], before,
        name="original input inventory after report write",
    )
    authenticate_contract(
        output_dir=output_dir,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        phase="report_written",
    )
    audit = _expected_audit(
        report=report,
        contract=contract,
        before_inventory=before,
        after_inventory=final_inputs["inventory"],
        frozen_report=frozen_report,
        frozen_audit=frozen_audit,
        derivation=derivation,
    )
    atomic_json(Path(output_dir) / AUDIT_NAME, audit)
    _validate_repair_tree(output_dir, phase="complete")
    observed_audit = load_json(
        Path(output_dir) / AUDIT_NAME, name="analysis-v2 audit",
    )
    verify_self_hash(observed_audit, "audit_sha256", name="analysis-v2 audit")
    _require(observed_audit, audit, name="serialized analysis-v2 audit")
    return observed_report


def validate(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    original_output_dir: Path = DEFAULT_ORIGINAL_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> dict:
    root = _canonical_root(
        output_dir, name="analysis-repair output root", require_exists=True,
    )
    entries = {path.name for path in root.iterdir()}
    if entries == {CONTRACT_NAME}:
        contract = authenticate_contract(
            output_dir=root,
            original_output_dir=original_output_dir,
            freeze_spec_path=freeze_spec_path,
            repair_dir=repair_dir,
            phase="prepared",
        )
        return {"status": "prepared", "contract_sha256": contract["contract_sha256"]}
    _validate_repair_tree(root, phase="complete")
    contract = authenticate_contract(
        output_dir=root,
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        phase="complete",
    )
    report = load_json(root / REPORT_NAME, name="analysis-v2 report")
    audit = load_json(root / AUDIT_NAME, name="analysis-v2 audit")
    verify_self_hash(report, "report_sha256", name="analysis-v2 report")
    verify_self_hash(audit, "audit_sha256", name="analysis-v2 audit")
    _require(
        audit["analysis_repair_contract_sha256"], contract["contract_sha256"],
        name="audit contract binding",
    )
    _require(
        audit["analysis_v2_report_sha256"], report["report_sha256"],
        name="audit report self-hash binding",
    )
    _require(
        audit["analysis_v2_report_raw_sha256"], sha256_file(root / REPORT_NAME),
        name="audit report raw-hash binding",
    )
    _require(
        _original_inventory(original_output_dir),
        contract["original_tree_raw_sha256"],
        name="validated original input inventory",
    )
    before = _original_inventory(original_output_dir)
    frozen_report, frozen_audit, derivation = _capture_frozen_analysis(
        original_output_dir=original_output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    expected_report = _versioned_report(
        frozen_report=frozen_report,
        frozen_audit=frozen_audit,
        contract=contract,
        derivation=derivation,
    )
    _require(report, expected_report, name="recomputed analysis-v2 report")
    after = _original_inventory(original_output_dir)
    expected_audit = _expected_audit(
        report=expected_report,
        contract=contract,
        before_inventory=before,
        after_inventory=after,
        frozen_report=frozen_report,
        frozen_audit=frozen_audit,
        derivation=derivation,
    )
    _require(audit, expected_audit, name="recomputed analysis-v2 audit")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Versioned post-freeze VCG V2.3 stability analysis repair",
    )
    parser.add_argument("command", choices=("prepare", "analyze", "validate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--original-output-dir", type=Path, default=DEFAULT_ORIGINAL_OUTPUT_DIR,
    )
    parser.add_argument("--freeze-spec", type=Path, default=DEFAULT_FREEZE_SPEC)
    parser.add_argument("--repair-dir", type=Path, default=DEFAULT_REPAIR_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    kwargs = {
        "output_dir": args.output_dir,
        "original_output_dir": args.original_output_dir,
        "freeze_spec_path": args.freeze_spec,
        "repair_dir": args.repair_dir,
    }
    if args.command == "prepare":
        result = prepare(**kwargs)
    elif args.command == "analyze":
        result = analyze(**kwargs)
    else:
        result = validate(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
