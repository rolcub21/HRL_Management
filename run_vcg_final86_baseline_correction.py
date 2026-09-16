#!/usr/bin/env python3
"""Rerun only the PSLAP/GA rows affected by the final86 index adapter bug."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_final86_four_method as final86


PROTOCOL = final86.PROTOCOL + "_baseline_normalization_correction_v1"
SOURCE_ROOT = final86.DEFAULT_OUTPUT_DIR
OUTPUT_ROOT = final86.PROJECT_ROOT / "results" / "vcg-final86-baseline-correction"
CONTRACT_NAME = "correction-contract.json"
REPORT_NAME = "corrected-final-report.json"


class CorrectionError(RuntimeError):
    pass


def _source_report() -> dict:
    path = SOURCE_ROOT / final86.REPORT_NAME
    report = final86._load_json(path, name="source final report")
    final86._self_hash(report, "report_sha256", name="source final report")
    methods = {record["method"]: record for record in report.get("methods", ())}
    for method in (final86.V11_METHOD, final86.V23_METHOD):
        if methods.get(method, {}).get("whole_method_eligible") is not True:
            raise CorrectionError(f"source learned method is not eligible: {method}")
    expected = {final86.DYNAMIC_METHOD: 30, final86.GA_METHOD: 120}
    for method, count in expected.items():
        record = methods.get(method, {})
        failures = record.get("failures")
        if (
            record.get("whole_method_eligible") is not False
            or not isinstance(failures, list)
            or len(failures) != count
            or any(
                failure.get("reason")
                != "ValueError: tuple.index(x): x not in tuple"
                for failure in failures
            )
        ):
            raise CorrectionError(f"source failure is not the frozen adapter defect: {method}")
    return report


def _contract(source_contract: Mapping, manifest: Mapping, report: Mapping) -> dict:
    payload = {
        "protocol": PROTOCOL,
        "status": "baseline_only_correction",
        "defect": "matched_baseline_identity_used_old_85000_85011_panel_tuple",
        "correction": "bind_identity_index_to_exact_final_86000_86029_panel",
        "scientific_policy_or_rng_change": False,
        "source_output_root": str(SOURCE_ROOT),
        "source_contract_sha256": source_contract["contract_sha256"],
        "source_instance_manifest_sha256": manifest["manifest_sha256"],
        "source_report_sha256": report["report_sha256"],
        "source_contract_raw_sha256": final86._sha256_file(
            SOURCE_ROOT / final86.CONTRACT_NAME, name="source final contract"
        ),
        "source_manifest_raw_sha256": final86._sha256_file(
            SOURCE_ROOT / final86.INSTANCE_MANIFEST_NAME,
            name="source final instance manifest",
        ),
        "source_report_raw_sha256": final86._sha256_file(
            SOURCE_ROOT / final86.REPORT_NAME, name="source final report"
        ),
        "correction_source_sha256": final86._sha256_file(
            Path(__file__).resolve(), name="baseline correction source"
        ),
        "methods_rerun": [final86.DYNAMIC_METHOD, final86.GA_METHOD],
        "expected_rows": 150,
        "learned_rows_reused_without_execution": 450,
        "instance_seeds": list(final86.INSTANCE_SEEDS),
        "dynamic_rows": 30,
        "ga_rows": 120,
        "ga_rng_formula_unchanged": True,
        "complete_case_filtering": False,
    }
    payload["contract_sha256"] = final86._digest(payload)
    return payload


def prepare(output_root: Path = OUTPUT_ROOT) -> tuple[dict, dict, dict, dict]:
    source_contract = final86.authenticate_contract(output_dir=SOURCE_ROOT)
    manifest = final86.authenticate_instance_manifest(output_dir=SOURCE_ROOT)
    source_report = _source_report()
    output_root = Path(output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    contract = _contract(source_contract, manifest, source_report)
    path = output_root / CONTRACT_NAME
    if path.exists():
        observed = final86._load_json(path, name="correction contract")
        final86._require_equal("correction contract", observed, contract)
    else:
        final86._atomic_json(path, contract)
    return source_contract, manifest, source_report, contract


@contextmanager
def _final_panel_identity_context():
    original = final86.matched.PANEL_SEEDS
    final86.matched.PANEL_SEEDS = final86.INSTANCE_SEEDS
    try:
        yield
    finally:
        final86.matched.PANEL_SEEDS = original


def _dynamic_rows(
    *, output_root: Path, source_contract: Mapping, manifest: Mapping,
    auth: Mapping, execute: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    identities = final86._identity_map(manifest)
    records = final86._record_by_seed(manifest)
    args = final86._baseline_runtime_args(auth)
    args.device = "cpu"
    for seed in final86.INSTANCE_SEEDS:
        identity = identities[seed]
        input_contract = final86._input_contract(
            contract=source_contract,
            manifest=manifest,
            method=final86.DYNAMIC_METHOD,
            identity=identity,
            model_seed=None,
            rng_index=None,
            rng_seed=None,
            checkpoint_sha256=None,
        )

        def run_one(seed=seed, identity=identity):
            try:
                instance = final86._load_instance(SOURCE_ROOT, records[seed])
                raw = final86.evaluate_assignment_ablation_one(
                    args,
                    seed,
                    None,
                    assignment_source=final86.matched.DETERMINISTIC_METHOD_TO_SOURCE[
                        final86.matched.DYNAMIC_METHOD
                    ],
                    episode_instance=instance,
                )
                with _final_panel_identity_context():
                    normalized = final86.matched._normalize_deterministic_baseline(
                        final86.matched.DYNAMIC_METHOD, raw, instance, identities,
                    )
                return final86._common_row(
                    method=final86.DYNAMIC_METHOD,
                    raw=normalized,
                    identity=identity,
                    model_seed=None,
                    rng_index=None,
                    rng_seed=None,
                    checkpoint_sha256=None,
                    evidence={
                        "assignment_source": normalized["assignment_source"],
                        "method_audit": normalized.get("method_audit"),
                        "device": "cpu",
                        "normalization_correction_protocol": PROTOCOL,
                    },
                )
            except Exception as error:
                return final86._failed_row(
                    method=final86.DYNAMIC_METHOD,
                    identity=identity,
                    model_seed=None,
                    rng_index=None,
                    rng_seed=None,
                    checkpoint_sha256=None,
                    error=error,
                )

        row = final86._load_or_execute(
            root=output_root,
            input_contract=input_contract,
            execute=execute,
            executor=run_one,
        )
        (missing if row is None else rows).append(
            dict(input_contract) if row is None else row
        )
    return rows, missing


def _ga_rows(
    *, output_root: Path, source_contract: Mapping, manifest: Mapping,
    auth: Mapping, execute: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    identities = final86._identity_map(manifest)
    records = final86._record_by_seed(manifest)
    base_args = final86._baseline_runtime_args(auth)
    base_args.device = "cpu"
    implementation = final86.repair_v2._impl
    method = implementation.REPAIRED_2009_METHOD
    final86._require_equal("correction GA method", method, final86.GA_METHOD)
    for seed in final86.INSTANCE_SEEDS:
        instance_index = seed - final86.INSTANCE_SEEDS[0]
        identity = identities[seed]
        for rng_index in final86.RNG_INDICES:
            rng_seed = final86.GA_RNG_BASE + final86.GA_RNG_STRIDE * (
                4 * instance_index + rng_index
            )
            input_contract = final86._input_contract(
                contract=source_contract,
                manifest=manifest,
                method=final86.GA_METHOD,
                identity=identity,
                model_seed=None,
                rng_index=rng_index,
                rng_seed=rng_seed,
                checkpoint_sha256=None,
            )

            def run_one(
                seed=seed, identity=identity, rng_index=rng_index,
                rng_seed=rng_seed,
            ):
                try:
                    instance = final86._load_instance(SOURCE_ROOT, records[seed])
                    args = deepcopy(base_args)
                    args.ga_seed_base = rng_seed - seed
                    raw = final86.evaluate_assignment_ablation_one(
                        args,
                        seed,
                        None,
                        assignment_source=implementation.REPAIRED_METHOD_TO_SOURCE[
                            method
                        ],
                        episode_instance=instance,
                    )
                    with _final_panel_identity_context(), final86._ga_validation_seed_base(
                        rng_seed - seed
                    ):
                        normalized = implementation._normalize_repaired_run(
                            method, raw, instance=instance, identities=identities,
                        )
                    return final86._common_row(
                        method=final86.GA_METHOD,
                        raw=normalized,
                        identity=identity,
                        model_seed=None,
                        rng_index=rng_index,
                        rng_seed=rng_seed,
                        checkpoint_sha256=None,
                        evidence={
                            "assignment_source": normalized["assignment_source"],
                            "assignment_source_version": normalized[
                                "assignment_source_version"
                            ],
                            "capacity_aware_source_audit": normalized[
                                "capacity_aware_source_audit"
                            ],
                            "capacity_aware_selector_audit": normalized[
                                "capacity_aware_selector_audit"
                            ],
                            "capacity_aware_reservation_audit": normalized[
                                "capacity_aware_reservation_audit"
                            ],
                            "device": "cpu",
                            "ga_rollout_base_seed": rng_seed,
                            "normalization_correction_protocol": PROTOCOL,
                        },
                    )
                except Exception as error:
                    return final86._failed_row(
                        method=final86.GA_METHOD,
                        identity=identity,
                        model_seed=None,
                        rng_index=rng_index,
                        rng_seed=rng_seed,
                        checkpoint_sha256=None,
                        error=error,
                    )

            row = final86._load_or_execute(
                root=output_root,
                input_contract=input_contract,
                execute=execute,
                executor=run_one,
            )
            (missing if row is None else rows).append(
                dict(input_contract) if row is None else row
            )
    return rows, missing


def run(*, output_root: Path = OUTPUT_ROOT, execute: bool = True) -> dict:
    source_contract, manifest, _, contract = prepare(output_root)
    auth = final86._authenticate_all()
    dynamic, missing_dynamic = _dynamic_rows(
        output_root=output_root,
        source_contract=source_contract,
        manifest=manifest,
        auth=auth,
        execute=execute,
    )
    ga, missing_ga = _ga_rows(
        output_root=output_root,
        source_contract=source_contract,
        manifest=manifest,
        auth=auth,
        execute=execute,
    )
    final86.authenticate_contract(output_dir=SOURCE_ROOT)
    final86.authenticate_instance_manifest(output_dir=SOURCE_ROOT)
    return {
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "executed_or_loaded": len(dynamic) + len(ga),
        "strict_safe_complete": sum(
            row["strict_safe_complete"] is True for row in (*dynamic, *ga)
        ),
        "missing": len(missing_dynamic) + len(missing_ga),
    }


def analyze(*, output_root: Path = OUTPUT_ROOT) -> dict:
    source_contract, manifest, source_report, contract = prepare(output_root)
    auth = final86._authenticate_all()
    dynamic, missing_dynamic = _dynamic_rows(
        output_root=output_root, source_contract=source_contract,
        manifest=manifest, auth=auth, execute=False,
    )
    ga, missing_ga = _ga_rows(
        output_root=output_root, source_contract=source_contract,
        manifest=manifest, auth=auth, execute=False,
    )
    if missing_dynamic or missing_ga or len(dynamic) != 30 or len(ga) != 120:
        raise CorrectionError("corrected baseline grid is incomplete")
    v11, missing_v11 = final86._v11_rows(
        root=SOURCE_ROOT, contract=source_contract, manifest=manifest,
        auth=auth, execute=False,
    )
    v23, missing_v23 = final86._v23_rows(
        root=SOURCE_ROOT, contract=source_contract, manifest=manifest,
        auth=auth, execute=False,
    )
    if missing_v11 or missing_v23:
        raise CorrectionError("source learned grid is incomplete")
    corrected = final86._build_report(
        contract=source_contract,
        manifest=manifest,
        rows_by_method={
            final86.V11_METHOD: v11,
            final86.V23_METHOD: v23,
            final86.DYNAMIC_METHOD: dynamic,
            final86.GA_METHOD: ga,
        },
    )
    report = {
        "protocol": PROTOCOL,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "source_report_sha256": source_report["report_sha256"],
        "baseline_rows_rerun": 150,
        "learned_rows_reused": 450,
        "corrected_final_report": corrected,
    }
    report["report_sha256"] = final86._digest(report)
    path = Path(output_root).absolute() / REPORT_NAME
    if path.exists():
        final86._require_equal(
            "corrected report",
            final86._load_json(path, name="corrected report"),
            report,
        )
    else:
        final86._atomic_json(path, report)
    final86.authenticate_contract(output_dir=SOURCE_ROOT)
    final86.authenticate_instance_manifest(output_dir=SOURCE_ROOT)
    return report


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "inspect", "analyze"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        _, _, _, result = prepare()
    elif args.command == "run":
        result = run(execute=True)
    elif args.command == "inspect":
        result = run(execute=False)
    else:
        result = analyze()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


if __name__ == "__main__":
    main()
