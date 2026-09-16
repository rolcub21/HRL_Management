#!/usr/bin/env python3
"""Full, separately versioned E12 confirmation with corrected macro execution."""

from __future__ import annotations

import argparse
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
    evaluate as core,
    program,
)
from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    screen_corrected_executor_parity as parity,
)
from methods.conditioned_vcg.representation_ablation import (
    REPRESENTATION_VARIANTS,
)
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e12_corrected_executor_full_confirmation_94k_v1"
SCHEMA_VERSION = 1
PARENT_E12 = program.DEFAULT_OUTPUT
PARITY_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e12-executor-parity-screen-94k"
)
FIX_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e12-execution-fix-confirmation-94k"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e12-corrected-executor-full-94k"
)
CONTRACT_NAME = "corrected-full-contract.json"
REPORT_NAME = "corrected-full-report.json"
TABLE_NAME = "corrected-full-summary.md"
EXPECTED_ROWS = (
    len(e11.REGIMES)
    * len(e11.INSTANCE_SEEDS)
    * len(REPRESENTATION_VARIANTS)
    * len(program.MODEL_SEEDS)
    * len(program.DEPLOYMENT_LAMBDAS)
)

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
    "experiments/conditioned_vcg/E12_representation_ablation_94k/evaluate.py",
    "experiments/conditioned_vcg/E12_representation_ablation_94k/"
    "screen_corrected_executor_parity.py",
    "experiments/conditioned_vcg/E12_representation_ablation_94k/"
    "confirm_corrected_executor_full.py",
)


class CorrectedFullError(RuntimeError):
    pass


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = program.load_json(path, label=label)
    if value.get(field) != program.digest(value, hash_field=field):
        raise CorrectedFullError(f"{label} self-hash mismatch")
    return value


def _proof_parents() -> dict:
    fix_path = FIX_OUTPUT / "execution-fix-report.json"
    fix = _self_hashed(fix_path, "report_sha256", "execution-fix report")
    parity_path = PARITY_OUTPUT / "parity-screen-report.json"
    screen = _self_hashed(parity_path, "report_sha256", "parity-screen report")
    if not (
        fix.get("status") == "complete"
        and fix.get("acceptance_passed") is True
        and int(fix.get("strict_safe_complete", -1)) == 4
        and screen.get("status") == "complete"
        and screen.get("acceptance_passed") is True
        and int(screen.get("corrected_strict_safe_complete", -1)) == 108
        and int(screen.get("historical_failures_repaired", -1)) == 3
        and int(screen.get("historical_successes_retained", -1)) == 105
    ):
        raise CorrectedFullError("corrected-executor prerequisites are incomplete")
    return {
        "execution_fix_report_file_sha256": program.sha256(fix_path),
        "execution_fix_report_sha256": fix["report_sha256"],
        "parity_screen_report_file_sha256": program.sha256(parity_path),
        "parity_screen_report_sha256": screen["report_sha256"],
    }


def _records(parent: Mapping) -> list[dict]:
    return [
        parent["records"][(regime.regime_id, int(seed))]
        for regime in e11.REGIMES
        for seed in e11.INSTANCE_SEEDS
    ]


def _specs(model_seeds: Sequence[int] = program.MODEL_SEEDS) -> tuple[dict, ...]:
    return tuple(
        {
            "representation_variant": variant,
            "model_seed": int(seed),
            "preference_lambda": float(value),
        }
        for variant in REPRESENTATION_VARIANTS
        for seed in model_seeds
        for value in program.DEPLOYMENT_LAMBDAS
    )


def _historical_index(parent: Mapping) -> dict:
    entries = []
    strict = 0
    for record in _records(parent):
        for spec in _specs():
            path, ledger = parity._historical_ledger(
                PARENT_E12, parent["contract"], record, spec
            )
            entries.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": program.sha256(path),
                }
            )
            strict += int(ledger["row"]["strict_safe_complete"])
    if len(entries) != EXPECTED_ROWS or strict != 7_557:
        raise CorrectedFullError(
            f"historical E12 panel changed: {len(entries)} rows, {strict} strict"
        )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"count": len(entries), "strict_safe_complete": strict, "sha256": digest}


def _checkpoint_index() -> dict:
    entries = []
    for variant in REPRESENTATION_VARIANTS:
        for seed in program.MODEL_SEEDS:
            operational = (
                PARENT_E12
                / "training"
                / "operational"
                / variant
                / f"seed-{seed}"
                / "best.pth"
            )
            handling = (
                PARENT_E12
                / "training"
                / "handling"
                / variant
                / f"seed-{seed}"
                / "terminal.pth"
            )
            entries.append(
                {
                    "representation_variant": variant,
                    "model_seed": int(seed),
                    "operational_sha256": program.sha256(operational),
                    "handling_sha256": program.sha256(handling),
                }
            )
    return {"count": len(entries), "entries": entries}


def expected_contract() -> dict:
    parent = parity._frozen_parent(PARENT_E12)
    historical = _historical_index(parent)
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "question": (
            "does_the_corrected_executor_retain_or_improve_the_complete_"
            "original_E12_panel_without_retraining"
        ),
        "separately_versioned_from_original_E12": True,
        "original_E12_results_remain_immutable": True,
        "training_runs": 0,
        "expected_evaluation_rows": EXPECTED_ROWS,
        "representations": list(REPRESENTATION_VARIANTS),
        "model_seeds": list(program.MODEL_SEEDS),
        "deployment_lambdas": list(program.DEPLOYMENT_LAMBDAS),
        "regimes": [regime.regime_id for regime in e11.REGIMES],
        "instance_seeds": list(e11.INSTANCE_SEEDS),
        "parent_row_schema_protocol": program.PROTOCOL,
        "acceptance": {
            "all_rows_executed": EXPECTED_ROWS,
            "no_regression_of_historical_successes": True,
            "all_three_historical_failures_repaired": True,
            "all_corrected_selected_frontiers_exact": True,
        },
        "parents": {
            "frozen_e12_contract_file_sha256": program.sha256(
                parent["contract_path"]
            ),
            "frozen_e11_manifest_file_sha256": program.sha256(
                parent["manifest_path"]
            ),
            "historical_e12_ledger_set": historical,
            "checkpoints": _checkpoint_index(),
            **_proof_parents(),
        },
        "source_sha256": {
            path: program.sha256(PROJECT_ROOT / path) for path in SOURCE_PATHS
        },
    }
    return program.with_hash(value, "contract_sha256")


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = expected_contract()
    path = output / CONTRACT_NAME
    if path.is_file():
        observed = _self_hashed(path, "contract_sha256", "corrected-full contract")
        if observed != expected:
            raise CorrectedFullError(
                "corrected-full contract, parents, or sources changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise CorrectedFullError("nonempty corrected-full output has no contract")
        program.atomic_json(path, expected)
    return {
        "status": "prepared",
        "evaluation_runs": EXPECTED_ROWS,
        "training_runs": 0,
        "output": str(output),
    }


def authenticate(output: Path) -> dict:
    observed = _self_hashed(
        output.resolve() / CONTRACT_NAME,
        "contract_sha256",
        "corrected-full contract",
    )
    if observed != expected_contract():
        raise CorrectedFullError(
            "corrected-full contract, parents, or sources changed"
        )
    return observed


def _run_path(output: Path, record: Mapping, spec: Mapping) -> Path:
    return (
        output
        / "runs"
        / str(record["regime_id"])
        / str(spec["representation_variant"])
        / f"model-{spec['model_seed']}"
        / f"lambda-{parity._lambda_label(spec['preference_lambda'])}"
        / f"instance-{record['seed']}.json"
    )


def _run_one(
    output: Path,
    contract: Mapping,
    parent: Mapping,
    record: Mapping,
    spec: Mapping,
    checkpoint_cache: Mapping,
    *,
    device: torch.device,
) -> dict:
    path = _run_path(output, record, spec)
    case_id = parity._case_id(
        record["regime_id"],
        int(record["seed"]),
        spec["representation_variant"],
        int(spec["model_seed"]),
        float(spec["preference_lambda"]),
    )
    if path.is_file():
        value = _self_hashed(path, "run_sha256", "corrected-full run")
        if (
            value.get("contract_sha256") != contract["contract_sha256"]
            or value.get("case_id") != case_id
        ):
            raise CorrectedFullError("corrected-full run binding changed")
        return value

    historical_path, historical = parity._historical_ledger(
        PARENT_E12, parent["contract"], record, spec
    )
    instance = e11._load_instance(e11.DEFAULT_OUTPUT, record)
    key = (
        spec["representation_variant"],
        int(spec["model_seed"]),
        float(spec["preference_lambda"]),
    )
    op_payload, factory, checkpoint_hashes = checkpoint_cache[key]
    regime = e11.REGIME_BY_ID[record["regime_id"]]
    try:
        with core._runtime(factory, regime):
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
        corrected = core._row(
            raw, instance, record, spec, core.CONFIRMATION_PANEL
        )
        execution = {
            "macro_failures": int(raw.get("macro_failures", 0)),
            "illegal_drops": int(raw.get("illegal_drops", 0)),
            "delivery_count": len(raw.get("delivery_deviations", ())),
            "all_frontiers_exact": bool(
                raw.get("complete_frontier_exactly_verified", False)
            ),
        }
    except Exception as error:
        corrected = core._failed_row(
            error, record, spec, core.CONFIRMATION_PANEL
        )
        execution = {
            "macro_failures": None,
            "illegal_drops": None,
            "delivery_count": None,
            "all_frontiers_exact": None,
        }
    value = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "case_id": case_id,
            "record": {
                "regime_id": record["regime_id"],
                "instance_seed": int(record["seed"]),
                "episode_instance_id": record["episode_instance_id"],
            },
            "spec": dict(spec),
            "checkpoints": checkpoint_hashes,
            "historical_ledger_file_sha256": program.sha256(historical_path),
            "historical_ledger_sha256": historical["ledger_sha256"],
            "corrected_row": corrected,
            "execution_summary": execution,
            "comparison": parity._comparison(historical["row"], corrected),
        },
        "run_sha256",
    )
    program.atomic_json(path, value)
    return value


def _load_runs(output: Path, contract: Mapping) -> tuple[list[dict], list[str]]:
    parent = parity._frozen_parent(PARENT_E12)
    rows = []
    missing = []
    for record in _records(parent):
        for spec in _specs():
            path = _run_path(output, record, spec)
            if not path.is_file():
                missing.append(str(path.relative_to(output)))
                continue
            value = _self_hashed(path, "run_sha256", "corrected-full run")
            expected_id = parity._case_id(
                record["regime_id"],
                int(record["seed"]),
                spec["representation_variant"],
                int(spec["model_seed"]),
                float(spec["preference_lambda"]),
            )
            if (
                value.get("contract_sha256") != contract["contract_sha256"]
                or value.get("case_id") != expected_id
            ):
                raise CorrectedFullError("corrected-full run binding changed")
            rows.append(value)
    return rows, missing


def _summary_table(report: Mapping) -> str:
    return "\n".join(
        [
            "# E12 corrected-executor full confirmation",
            "",
            "| Version | Strict completion | Scope |",
            "|---|---:|---|",
            f"| Original E12 | {report['historical_strict_safe_complete']}/"
            f"{report['completed_runs']} | immutable original evaluation |",
            f"| Corrected executor | {report['corrected_strict_safe_complete']}/"
            f"{report['completed_runs']} | separately versioned confirmation |",
            "",
            f"Historical successes retained: {report['historical_successes_retained']}/"
            f"{report['historical_successes_observed']}.  ",
            f"Historical failures repaired: {report['historical_failures_repaired']}/"
            f"{report['historical_failures_observed']}.  ",
            f"New regressions: {report['historical_success_regressions']}.",
            "",
            "The original 7,557/7,560 result remains reportable as the original "
            "implementation result; this table reports the corrected version separately.",
            "",
        ]
    )


def _derived_e12_analysis(
    output: Path,
    contract: Mapping,
    corrected_rows: Sequence[Mapping],
    *,
    allow_partial: bool,
) -> dict:
    def corrected_authenticate(_output: Path):
        return contract, {"authentication_scope": "corrected_full_confirmation"}

    def corrected_rows_source(_output: Path, _panel: str):
        return list(corrected_rows)

    with patch.object(
        core.program, "authenticate", corrected_authenticate
    ), patch.object(core, "_all_rows", corrected_rows_source):
        return core.analyze(
            output,
            panel=core.CONFIRMATION_PANEL,
            allow_partial=allow_partial,
        )


def analyze(output: Path, *, allow_partial: bool = False) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    runs, missing = _load_runs(output, contract)
    if missing and not allow_partial:
        raise CorrectedFullError(f"missing {len(missing)} corrected-full runs")
    comparisons = [item["comparison"] for item in runs]
    historical_successes = [
        item for item in comparisons if item["historical_strict_safe_complete"]
    ]
    historical_failures = [
        item for item in comparisons if not item["historical_strict_safe_complete"]
    ]
    corrected_rows = [item["corrected_row"] for item in runs]
    derived = _derived_e12_analysis(
        output,
        contract,
        corrected_rows,
        allow_partial=bool(missing),
    )
    report = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if missing else "complete",
            "completed_runs": len(runs),
            "expected_runs": EXPECTED_ROWS,
            "missing_runs": len(missing),
            "training_runs": 0,
            "historical_strict_safe_complete": sum(
                item["historical_strict_safe_complete"] for item in comparisons
            ),
            "corrected_strict_safe_complete": sum(
                item["corrected_strict_safe_complete"] for item in comparisons
            ),
            "historical_successes_observed": len(historical_successes),
            "historical_successes_retained": sum(
                item["historical_success_retained"] for item in historical_successes
            ),
            "historical_success_regressions": sum(
                not item["corrected_strict_safe_complete"]
                for item in historical_successes
            ),
            "historical_failures_observed": len(historical_failures),
            "historical_failures_repaired": sum(
                item["historical_failure_repaired"] for item in historical_failures
            ),
            "corrected_frontiers_all_exact": sum(
                item["execution_summary"]["all_frontiers_exact"] is True
                for item in runs
            ),
            "historical_success_behavior_digest_equal": sum(
                item["behavior_digest_equal"] for item in historical_successes
            ),
            "historical_success_metric_equal": sum(
                item["all_comparison_fields_equal"] for item in historical_successes
            ),
            "derived_e12_analysis": {
                "path": "analysis/e11_93k-report.json",
                "report_sha256": derived["report_sha256"],
                "strict_safe_complete_rows": derived[
                    "strict_safe_complete_rows"
                ],
            },
            "acceptance_passed": bool(
                not missing
                and all(item["corrected_strict_safe_complete"] for item in comparisons)
                and all(item["historical_success_retained"] for item in historical_successes)
                and all(item["historical_failure_repaired"] for item in historical_failures)
                and all(
                    item["execution_summary"]["all_frontiers_exact"] is True
                    for item in runs
                )
            ),
            "reporting_rule": (
                "retain_original_7557_of_7560_and_report_this_corrected_full_"
                "confirmation_as_a_separate_version"
            ),
        },
        "report_sha256",
    )
    program.atomic_json(output / REPORT_NAME, report)
    (output / TABLE_NAME).write_text(_summary_table(report), encoding="utf-8")
    return report


def run(
    output: Path,
    *,
    device_name: str,
    model_seeds: Sequence[int],
) -> dict:
    prepare(output)
    output = output.resolve()
    contract = authenticate(output)
    parent = parity._frozen_parent(PARENT_E12)
    selected = tuple(model_seeds or program.MODEL_SEEDS)
    if any(seed not in program.MODEL_SEEDS for seed in selected):
        raise CorrectedFullError("model seeds must lie in {0,1,2}")
    device = pilot._device(device_name)
    records = _records(parent)
    total = len(records) * len(REPRESENTATION_VARIANTS) * len(
        program.DEPLOYMENT_LAMBDAS
    ) * len(selected)
    completed = 0
    for seed in selected:
        cache = {}
        for spec in _specs((seed,)):
            key = (
                spec["representation_variant"],
                int(spec["model_seed"]),
                float(spec["preference_lambda"]),
            )
            cache[key] = core._checkpoint_factory(
                PARENT_E12,
                variant=key[0],
                model_seed=key[1],
                value=key[2],
                device=device,
            )
        for record in records:
            for spec in _specs((seed,)):
                value = _run_one(
                    output,
                    contract,
                    parent,
                    record,
                    spec,
                    cache,
                    device=device,
                )
                completed += 1
                print(
                    f"E12-corrected {completed}/{total} | "
                    f"{record['regime_id']} {record['seed']} | "
                    f"{spec['representation_variant']} seed={seed} "
                    f"lambda={spec['preference_lambda']:.2f} | "
                    f"strict={int(value['corrected_row']['strict_safe_complete'])}",
                    flush=True,
                )
        del cache
    return analyze(output, allow_partial=bool(model_seeds))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = run(
            args.output,
            device_name=args.device,
            model_seeds=tuple(args.model_seed or ()),
        )
    else:
        result = analyze(args.output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
