#!/usr/bin/env python3
"""Bounded frozen-policy parity screen for the corrected macro executors.

This diagnostic never edits the completed E12 ledgers. It authenticates 108
historical E12 rows and replays their exact frozen instances/checkpoints with
the corrected delivery/relocation executors. The panel crosses every E12
representation, model seed, and deployment preference on one reference and
two combined-shift episodes. It includes all three historical E12 failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

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
from methods.conditioned_vcg.representation_ablation import (
    REPRESENTATION_VARIANTS,
)
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e12_corrected_executor_parity_screen_94k_v1"
SCHEMA_VERSION = 1
DEFAULT_E12_OUTPUT = program.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e12-executor-parity-screen-94k"
)

# Reference control plus both combined-shift episodes implicated by the three
# historical failures. Crossed with 3 variants x 3 seeds x 4 lambdas = 108.
SCREEN_EPISODES = (
    ("reference", 93_000),
    ("combined_shift", 93_008),
    ("combined_shift", 93_028),
)

EXPECTED_HISTORICAL_FAILURES = {
    ("combined_shift", 93_008, "full_relational_successor", 2, 0.05),
    ("combined_shift", 93_028, "full_relational_successor", 2, 0.10),
    ("combined_shift", 93_028, "full_relational_successor", 2, 0.20),
}

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
)

COMPARISON_FIELDS = (
    "dense_return",
    "mean_absolute_error",
    "mean_signed_deviation",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "steps",
    "physical_storage_relocations",
    "physical_rehandles_per_100_required_deliveries",
    "observed_steps_to_stop",
    "observed_rehandles_to_stop",
    "macro_decisions",
)


class ParityScreenError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ParityScreenError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_self_hashed(path: Path, hash_field: str, label: str) -> dict:
    value = program.load_json(path, label=label)
    if value.get(hash_field) != program.digest(value, hash_field=hash_field):
        raise ParityScreenError(f"{label} self-hash mismatch")
    return value


def _lambda_label(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _case_id(
    regime_id: str,
    instance_seed: int,
    variant: str,
    model_seed: int,
    value: float,
) -> str:
    return (
        f"{regime_id}-instance-{instance_seed}--{variant}--"
        f"model-{model_seed}--lambda-{_lambda_label(value)}"
    )


def _frozen_parent(e12_output: Path) -> dict:
    e12_output = e12_output.resolve()
    contract_path = e12_output / program.CONTRACT_NAME
    contract = _load_self_hashed(
        contract_path, "contract_sha256", "frozen E12 contract"
    )
    if contract.get("protocol") != program.PROTOCOL:
        raise ParityScreenError("unexpected frozen E12 protocol")

    manifest_path = e11.DEFAULT_OUTPUT / e11.MANIFEST_NAME
    manifest = _load_self_hashed(
        manifest_path, "manifest_sha256", "frozen E11 manifest"
    )
    if (
        manifest.get("protocol") != e11.PROTOCOL
        or _sha256(manifest_path) != contract.get("e11_manifest_sha256")
    ):
        raise ParityScreenError("frozen E11 parent binding changed")
    records = {
        (str(record["regime_id"]), int(record["seed"])): record
        for record in manifest.get("records", ())
    }
    if len(records) != len(e11.REGIMES) * len(e11.INSTANCE_SEEDS):
        raise ParityScreenError("frozen E11 coordinate panel changed")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "records": records,
    }


def _historical_ledger(
    e12_output: Path,
    e12_contract: Mapping,
    record: Mapping,
    spec: Mapping,
) -> tuple[Path, dict]:
    path = core._ledger_path(
        e12_output, core.CONFIRMATION_PANEL, spec, record
    )
    ledger = _load_self_hashed(path, "ledger_sha256", "historical E12 ledger")
    if (
        ledger.get("contract_sha256") != e12_contract["contract_sha256"]
        or ledger.get("panel") != core.CONFIRMATION_PANEL
        or ledger.get("spec") != dict(spec)
        or ledger.get("episode_instance_id") != record["episode_instance_id"]
    ):
        raise ParityScreenError("historical E12 ledger binding changed")
    return path, ledger


def _case_key(case: Mapping) -> tuple:
    return (
        case["regime_id"],
        int(case["instance_seed"]),
        case["representation_variant"],
        int(case["model_seed"]),
        float(case["preference_lambda"]),
    )


def _contract(e12_output: Path) -> dict:
    e12_output = e12_output.resolve()
    parent = _frozen_parent(e12_output)
    cases = []
    observed_failures = set()
    for regime_id, instance_seed in SCREEN_EPISODES:
        try:
            record = parent["records"][(regime_id, instance_seed)]
        except KeyError as error:
            raise ParityScreenError(
                f"screen episode is absent: {regime_id}/{instance_seed}"
            ) from error
        for variant in REPRESENTATION_VARIANTS:
            for model_seed in program.MODEL_SEEDS:
                for value in program.DEPLOYMENT_LAMBDAS:
                    spec = {
                        "representation_variant": variant,
                        "model_seed": int(model_seed),
                        "preference_lambda": float(value),
                    }
                    path, ledger = _historical_ledger(
                        e12_output, parent["contract"], record, spec
                    )
                    row = ledger["row"]
                    case = {
                        "case_id": _case_id(
                            regime_id,
                            instance_seed,
                            variant,
                            model_seed,
                            value,
                        ),
                        "regime_id": regime_id,
                        "instance_seed": int(instance_seed),
                        "instance_index": int(record["instance_index"]),
                        "episode_instance_id": record["episode_instance_id"],
                        **spec,
                        "historical_ledger_file_sha256": _sha256(path),
                        "historical_ledger_sha256": ledger["ledger_sha256"],
                        "historical_strict_safe_complete": bool(
                            row["strict_safe_complete"]
                        ),
                    }
                    cases.append(case)
                    if not row["strict_safe_complete"]:
                        observed_failures.add(_case_key(case))

    if len(cases) != 108:
        raise ParityScreenError(f"expected 108 cases, found {len(cases)}")
    if observed_failures != EXPECTED_HISTORICAL_FAILURES:
        raise ParityScreenError(
            "the selected panel no longer contains exactly the three known failures"
        )

    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": (
            "does_the_corrected_executor_preserve_historical_E12_successes_"
            "while_repairing_the_three_known_execution_faithfulness_failures"
        ),
        "diagnostic_not_replacement_for_full_E12_confirmation": True,
        "training_or_checkpoint_selection": False,
        "screen_episodes": [
            {"regime_id": regime, "instance_seed": seed}
            for regime, seed in SCREEN_EPISODES
        ],
        "crossing": {
            "representations": list(REPRESENTATION_VARIANTS),
            "model_seeds": list(program.MODEL_SEEDS),
            "deployment_lambdas": list(program.DEPLOYMENT_LAMBDAS),
        },
        "cases": cases,
        "expected_runs": len(cases),
        "expected_historical_successes": 105,
        "expected_historical_failures": 3,
        "acceptance": {
            "corrected_strict_safe_complete": "108/108",
            "historical_successes_retained": "105/105",
            "historical_failures_repaired": "3/3",
            "corrected_frontiers_all_exact": True,
        },
        "parents": {
            "frozen_e12_contract_file_sha256": _sha256(
                parent["contract_path"]
            ),
            "frozen_e11_manifest_file_sha256": _sha256(
                parent["manifest_path"]
            ),
        },
        "source_sha256": {
            path: _sha256(PROJECT_ROOT / path) for path in SOURCE_PATHS
        },
    }
    return program.with_hash(semantic, "contract_sha256")


def prepare(output: Path, e12_output: Path) -> dict:
    output = output.resolve()
    expected = _contract(e12_output)
    path = output / "parity-screen-contract.json"
    if path.is_file():
        observed = program.load_json(path, label="parity-screen contract")
        if observed != expected:
            raise ParityScreenError(
                "parity-screen contract, parents, or sources changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ParityScreenError("nonempty parity-screen output has no contract")
        program.atomic_json(path, expected)
    return {
        "status": "prepared",
        "inference_runs": len(expected["cases"]),
        "historical_successes": expected["expected_historical_successes"],
        "historical_failures": expected["expected_historical_failures"],
        "training_runs": 0,
        "output": str(output),
    }


def _authenticate(output: Path, e12_output: Path) -> dict:
    observed = _load_self_hashed(
        output / "parity-screen-contract.json",
        "contract_sha256",
        "parity-screen contract",
    )
    if observed != _contract(e12_output.resolve()):
        raise ParityScreenError(
            "parity-screen contract, parents, or sources changed"
        )
    return observed


def _run_path(output: Path, case: Mapping) -> Path:
    return (
        output
        / "runs"
        / case["regime_id"]
        / case["representation_variant"]
        / f"model-{case['model_seed']}"
        / f"lambda-{_lambda_label(case['preference_lambda'])}"
        / f"instance-{case['instance_seed']}.json"
    )


def _comparison(historical: Mapping, corrected: Mapping) -> dict:
    field_equality = {
        field: historical.get(field) == corrected.get(field)
        for field in COMPARISON_FIELDS
    }
    historical_strict = bool(historical["strict_safe_complete"])
    corrected_strict = bool(corrected["strict_safe_complete"])
    return {
        "historical_strict_safe_complete": historical_strict,
        "corrected_strict_safe_complete": corrected_strict,
        "historical_success_retained": bool(historical_strict and corrected_strict),
        "historical_failure_repaired": bool(
            not historical_strict and corrected_strict
        ),
        "behavior_digest_equal": (
            historical.get("behavior_digest") == corrected.get("behavior_digest")
        ),
        "comparison_field_equality": field_equality,
        "all_comparison_fields_equal": all(field_equality.values()),
        "changed_comparison_fields": [
            field for field, equal in field_equality.items() if not equal
        ],
    }


def _run_one(
    output: Path,
    e12_output: Path,
    contract: Mapping,
    parent: Mapping,
    case: Mapping,
    checkpoint_cache: Mapping,
    *,
    device: torch.device,
) -> dict:
    path = _run_path(output, case)
    if path.is_file():
        value = _load_self_hashed(path, "run_sha256", "parity-screen run")
        if (
            value.get("contract_sha256") != contract["contract_sha256"]
            or value.get("case", {}).get("case_id") != case["case_id"]
        ):
            raise ParityScreenError("parity-screen run binding changed")
        return value

    record = parent["records"][(case["regime_id"], case["instance_seed"])]
    spec = {
        "representation_variant": case["representation_variant"],
        "model_seed": int(case["model_seed"]),
        "preference_lambda": float(case["preference_lambda"]),
    }
    historical_path, historical_ledger = _historical_ledger(
        e12_output, parent["contract"], record, spec
    )
    if (
        _sha256(historical_path) != case["historical_ledger_file_sha256"]
        or historical_ledger["ledger_sha256"]
        != case["historical_ledger_sha256"]
    ):
        raise ParityScreenError("historical ledger changed after preparation")

    instance = e11._load_instance(e11.DEFAULT_OUTPUT, record)
    key = (
        case["representation_variant"],
        int(case["model_seed"]),
        float(case["preference_lambda"]),
    )
    op_payload, factory, checkpoint_hashes = checkpoint_cache[key]
    regime = e11.REGIME_BY_ID[case["regime_id"]]
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
        corrected_row = core._row(
            raw, instance, record, spec, core.CONFIRMATION_PANEL
        )
        execution_summary = {
            "macro_failures": int(raw.get("macro_failures", 0)),
            "illegal_drops": int(raw.get("illegal_drops", 0)),
            "delivery_count": len(raw.get("delivery_deviations", ())),
            "all_frontiers_exact": bool(
                raw.get("complete_frontier_exactly_verified", False)
            ),
        }
    except Exception as error:  # preserve a diagnostic row instead of aborting the panel
        corrected_row = core._failed_row(
            error, record, spec, core.CONFIRMATION_PANEL
        )
        execution_summary = {
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
            "case": dict(case),
            "checkpoints": checkpoint_hashes,
            "historical_row": historical_ledger["row"],
            "corrected_row": corrected_row,
            "execution_summary": execution_summary,
            "comparison": _comparison(
                historical_ledger["row"], corrected_row
            ),
        },
        "run_sha256",
    )
    program.atomic_json(path, value)
    return value


def run(
    output: Path,
    e12_output: Path,
    *,
    device_name: str,
    model_seeds: Sequence[int],
) -> dict:
    prepare(output, e12_output)
    output = output.resolve()
    e12_output = e12_output.resolve()
    contract = _authenticate(output, e12_output)
    parent = _frozen_parent(e12_output)
    selected_seeds = tuple(model_seeds or program.MODEL_SEEDS)
    if any(seed not in program.MODEL_SEEDS for seed in selected_seeds):
        raise ParityScreenError("model seeds must lie in {0,1,2}")
    selected_cases = [
        case for case in contract["cases"] if case["model_seed"] in selected_seeds
    ]
    device = pilot._device(device_name)
    checkpoint_cache = {}
    for case in selected_cases:
        key = (
            case["representation_variant"],
            int(case["model_seed"]),
            float(case["preference_lambda"]),
        )
        if key not in checkpoint_cache:
            checkpoint_cache[key] = core._checkpoint_factory(
                e12_output,
                variant=key[0],
                model_seed=key[1],
                value=key[2],
                device=device,
            )

    for index, case in enumerate(selected_cases, start=1):
        row = _run_one(
            output,
            e12_output,
            contract,
            parent,
            case,
            checkpoint_cache,
            device=device,
        )
        print(
            json.dumps(
                {
                    "case": f"{index}/{len(selected_cases)}",
                    "case_id": case["case_id"],
                    "historical_strict": row["comparison"][
                        "historical_strict_safe_complete"
                    ],
                    "corrected_strict": row["comparison"][
                        "corrected_strict_safe_complete"
                    ],
                    "behavior_equal": row["comparison"][
                        "behavior_digest_equal"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return analyze(output, e12_output, allow_partial=bool(model_seeds))


def analyze(output: Path, e12_output: Path, *, allow_partial: bool) -> dict:
    output = output.resolve()
    contract = _authenticate(output, e12_output.resolve())
    rows = []
    missing = []
    for case in contract["cases"]:
        path = _run_path(output, case)
        if not path.is_file():
            missing.append(case["case_id"])
            continue
        row = _load_self_hashed(path, "run_sha256", "parity-screen run")
        if (
            row.get("contract_sha256") != contract["contract_sha256"]
            or row.get("case", {}).get("case_id") != case["case_id"]
        ):
            raise ParityScreenError("parity-screen run binding changed")
        rows.append(row)
    if missing and not allow_partial:
        raise ParityScreenError(f"missing {len(missing)} cases")

    historical_successes = [
        row
        for row in rows
        if row["comparison"]["historical_strict_safe_complete"]
    ]
    historical_failures = [
        row
        for row in rows
        if not row["comparison"]["historical_strict_safe_complete"]
    ]
    successes_retained = sum(
        row["comparison"]["historical_success_retained"]
        for row in historical_successes
    )
    failures_repaired = sum(
        row["comparison"]["historical_failure_repaired"]
        for row in historical_failures
    )
    corrected_strict = sum(
        row["comparison"]["corrected_strict_safe_complete"] for row in rows
    )
    exact_frontiers = sum(
        row["execution_summary"]["all_frontiers_exact"] is True for row in rows
    )
    success_behavior_equal = sum(
        row["comparison"]["behavior_digest_equal"]
        for row in historical_successes
    )
    success_metrics_equal = sum(
        row["comparison"]["all_comparison_fields_equal"]
        for row in historical_successes
    )
    complete = not missing
    acceptance_passed = bool(
        complete
        and corrected_strict == len(contract["cases"])
        and successes_retained == contract["expected_historical_successes"]
        and failures_repaired == contract["expected_historical_failures"]
        and exact_frontiers == len(contract["cases"])
    )
    report = program.with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if missing else "complete",
            "completed_runs": len(rows),
            "expected_runs": len(contract["cases"]),
            "missing_runs": len(missing),
            "training_runs": 0,
            "historical_successes_observed": len(historical_successes),
            "historical_successes_retained": successes_retained,
            "historical_failures_observed": len(historical_failures),
            "historical_failures_repaired": failures_repaired,
            "corrected_strict_safe_complete": corrected_strict,
            "corrected_frontiers_all_exact": exact_frontiers,
            "historical_success_behavior_digest_equal": success_behavior_equal,
            "historical_success_all_comparison_fields_equal": success_metrics_equal,
            "historical_success_behavior_changes": (
                len(historical_successes) - success_behavior_equal
            ),
            "historical_success_metric_changes": (
                len(historical_successes) - success_metrics_equal
            ),
            "acceptance_passed": acceptance_passed,
            "interpretation": {
                "acceptance": (
                    "bounded evidence that corrected execution retains sampled "
                    "successes and repairs all known failures"
                ),
                "scope": (
                    "diagnostic screen only; it is not the corrected 7560-row "
                    "E12 aggregate confirmation"
                ),
            },
            "changed_historical_successes": [
                {
                    "case_id": row["case"]["case_id"],
                    "behavior_digest_equal": row["comparison"][
                        "behavior_digest_equal"
                    ],
                    "changed_comparison_fields": row["comparison"][
                        "changed_comparison_fields"
                    ],
                }
                for row in historical_successes
                if (
                    not row["comparison"]["behavior_digest_equal"]
                    or not row["comparison"]["all_comparison_fields_equal"]
                )
            ],
            "corrected_failures": [
                {
                    "case_id": row["case"]["case_id"],
                    "repaired": row["comparison"]["historical_failure_repaired"],
                    "method_failure_reason": row["corrected_row"][
                        "method_failure_reason"
                    ],
                }
                for row in historical_failures
            ],
        },
        "report_sha256",
    )
    program.atomic_json(output / "parity-screen-report.json", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e12-output", type=Path, default=DEFAULT_E12_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output, args.e12_output)
    elif args.command == "run":
        result = run(
            args.output,
            args.e12_output,
            device_name=args.device,
            model_seeds=tuple(args.model_seed or ()),
        )
    else:
        result = analyze(
            args.output,
            args.e12_output,
            allow_partial=args.allow_partial,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
