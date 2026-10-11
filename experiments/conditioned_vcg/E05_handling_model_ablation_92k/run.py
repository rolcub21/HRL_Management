#!/usr/bin/env python3
"""E5: final handling-model validation and conditioning-input ablation.

E5(a) assembles already-completed Monte-Carlo calibration and fixed-bank
diagnostics.  E5(b) reuses E4's conditioned controls and changes only the
preference value supplied to the frozen handling network.  The deployment
multiplier remains the requested lambda.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e05_handling_model_ablation_92k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = e04.MODEL_SEEDS
INSTANCE_SEEDS = e04.INSTANCE_SEEDS
DEPLOYMENT_LAMBDAS = e04.CONDITIONED_LAMBDAS
CLAMPED_INPUT_LAMBDA = 0.10
NEW_CLAMPED_LAMBDAS = tuple(
    value for value in DEPLOYMENT_LAMBDAS if value != CLAMPED_INPUT_LAMBDA
)
CLAMPED = "conditioned_vcg_clamped_qn_input"
FULL = e04.CONDITIONED_SAFE

CONTRACT_NAME = "e05-contract.json"
VALIDATION_NAME = "e05a-model-validation.json"
VALIDATION_TABLE_NAME = "e05a-model-validation-table.md"
REPORT_NAME = "e05b-report.json"
TABLE_NAME = "e05b-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e05-handling-ablation-92k"
E04_OUTPUT = e04.DEFAULT_OUTPUT
D9_REPORT = (
    PROJECT_ROOT
    / "results/vcg-v1-1-conditioned-handling-fixed-merit-bank/fixed-merit-bank-report.json"
)
SEED1_FINAL = (
    PROJECT_ROOT
    / "results/vcg-v1-1-conditioned-handling-seed1-convergence-continuation/seed1-continuation-summary.json"
)


class E5Error(RuntimeError):
    pass


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha256(path: Path) -> str:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E5Error(f"missing canonical file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E5Error(f"missing canonical {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E5Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E5Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, hash_field=field):
        raise E5Error(f"{label} self hash mismatch")


class ClampedPreferenceNetwork(nn.Module):
    """Use one Q_N preference input while leaving merit scalarization untouched."""

    def __init__(self, network: nn.Module, input_lambda: float) -> None:
        super().__init__()
        self.network = network
        self.input_lambda = float(input_lambda)

    def forward(self, features, preference_lambda):
        return self.network(features, self.input_lambda)


def _assemble_validation() -> dict:
    d9 = _load_json(D9_REPORT, label="D9 fixed-bank report")
    seed1 = _load_json(SEED1_FINAL, label="final seed-1 continuation report")
    try:
        final1 = seed1["round_records"][-1]
        sources = {
            0: {
                "calibration": d9["seeds"]["0"]["terminal_calibration"],
                "fixed_bank": d9["seeds"]["0"]["fixed_bank"],
                "source": "D9_final_seed_0",
            },
            1: {
                "calibration": final1["deployed_validation"],
                "fixed_bank": final1["fixed_bank_diagnostic"]["summary"],
                "source": "seed_1_final_continuation_round_10",
            },
            2: {
                "calibration": d9["seeds"]["2"]["terminal_calibration"],
                "fixed_bank": d9["seeds"]["2"]["fixed_bank"],
                "source": "D9_final_seed_2",
            },
        }
    except (KeyError, IndexError, TypeError) as error:
        raise E5Error("handling validation artifacts have an unexpected schema") from error

    records = {}
    for seed, source in sources.items():
        calibration = source["calibration"]
        fixed = source["fixed_bank"]
        records[str(seed)] = {
            "source": source["source"],
            "mc_validation_sample_count": int(calibration["sample_count"]),
            "mc_prediction_mae": float(calibration["mae"]),
            "mc_prediction_bias": float(calibration["bias"]),
            "fixed_state_count": int(fixed["state_count"]),
            "fixed_candidate_count": int(fixed["candidate_count"]),
            "qn_direction_reversal_fraction": float(
                fixed["qn_direction_reversal_fraction"]
            ),
            "qn_nonincreasing_fraction": float(fixed["qn_nonincreasing_fraction"]),
            "lambda_qn_nondecreasing_fraction": float(
                fixed["lambda_qn_nondecreasing_fraction"]
            ),
            "maximum_adjacent_qn_change": float(
                fixed["maximum_adjacent_qn_change"]
            ),
            "states_with_policy_switch_fraction": float(
                fixed["states_with_policy_switch_fraction"]
            ),
        }
    if any(
        item["fixed_state_count"] != 34
        or item["fixed_candidate_count"] != 686
        or item["qn_direction_reversal_fraction"] != 0.0
        or item["lambda_qn_nondecreasing_fraction"] != 1.0
        for item in records.values()
    ):
        raise E5Error("final fixed-bank handling diagnostics do not match the declared result")
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "component": "E5a_handling_model_validation",
            "new_rollouts": 0,
            "checkpoint_selection_or_retraining": False,
            "d9_report_sha256": _sha256(D9_REPORT),
            "seed1_final_report_sha256": _sha256(SEED1_FINAL),
            "fixed_bank_lambda_count": len(d9["lambda_grid"]),
            "seeds": records,
            "supported_scope": [
                "MC_prediction_error_and_bias_against_realized_remaining_handling",
                "smooth_QN_dependence_on_lambda_on_the_fixed_bank",
                "monotone_effective_lambda_QN_pressure_on_the_fixed_bank",
            ],
            "not_an_ablation": True,
        },
        "validation_sha256",
    )


def _validation_table(validation: Mapping) -> str:
    lines = [
        "# E5(a) handling-model validation",
        "",
        "| Seed | MC n | MAE | Bias | Q_N reversals | lambda Q_N monotone | Fixed states/candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, item in validation["seeds"].items():
        lines.append(
            f"| {seed} | {item['mc_validation_sample_count']} | "
            f"{item['mc_prediction_mae']:.3f} | {item['mc_prediction_bias']:+.3f} | "
            f"{item['qn_direction_reversal_fraction']:.0%} | "
            f"{item['lambda_qn_nondecreasing_fraction']:.0%} | "
            f"{item['fixed_state_count']}/{item['fixed_candidate_count']} |"
        )
    lines.extend(
        [
            "",
            "This is validation of the frozen internal consequence model; it is not a causal ablation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _contract(project_root: Path, e04_output: Path) -> dict:
    e04_contract = e04.authenticate_contract(project_root, e04_output)
    manifest = e04.authenticate_manifest(project_root, e04_output)
    validation = _assemble_validation()
    source_paths = (
        Path(__file__).resolve(),
        Path(e04.__file__).resolve(),
        project_root / "vcg_v11_conditioned_handling.py",
    )
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "does_inference_time_preference_dependent_handling_prediction_add_value_"
                "beyond_scalar_weighting_of_one_fixed_prediction"
            ),
            "e04_contract_sha256": e04_contract["contract_sha256"],
            "e04_manifest_sha256": manifest["manifest_sha256"],
            "e05a_validation_sha256": validation["validation_sha256"],
            "deployment_lambdas": list(DEPLOYMENT_LAMBDAS),
            "clamped_handling_input_lambda": CLAMPED_INPUT_LAMBDA,
            "new_clamped_lambdas": list(NEW_CLAMPED_LAMBDAS),
            "full_conditioned_controls_reused_from_E4": True,
            "midpoint_is_exact_construction_identity_and_reuses_E4_control": True,
            "same_frozen_network_weights": True,
            "same_Qop_verifier_liveness_selector_candidates_and_executor": True,
            "deployment_multiplier_remains_requested_lambda": True,
            "training_or_checkpoint_selection": False,
            "claim_boundary": (
                "tests_deployment_use_of_learned_conditioning_not_necessity_of_conditioned_training"
            ),
            "separately_trained_unconditioned_model": False,
            "pilot_scope": {
                "model_seed": 0,
                "instance_count": e04.PILOT_INSTANCE_COUNT,
                "interpretation": "debugging_and_mechanism_check_only",
                "scientific_results_do_not_authorize_retuning": True,
            },
            "source_sha256": {
                str(path.relative_to(project_root)): _sha256(path)
                for path in source_paths
            },
        },
        "contract_sha256",
    )


def prepare(project_root: Path, output_dir: Path, e04_output: Path) -> dict:
    e04.prepare(project_root, e04_output)
    output_dir = output_dir.absolute()
    if output_dir.is_symlink():
        raise E5Error("output directory must not be a symlink")
    expected = _contract(project_root, e04_output)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, label="E5 contract")
        _verify_hash(observed, "contract_sha256", label="E5 contract")
        if observed != expected:
            raise E5Error("E5 contract, parent E4 panel, frozen inputs, or sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise E5Error("nonempty E5 output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        final90._atomic_json(path, expected)
    validation = _assemble_validation()
    final90._atomic_json(output_dir / VALIDATION_NAME, validation)
    final90._atomic_text(
        output_dir / VALIDATION_TABLE_NAME, _validation_table(validation)
    )
    return {
        "status": "prepared",
        "contract": str(path),
        "e05a_validation": str(output_dir / VALIDATION_NAME),
        "new_e05b_full_rows": (
            len(MODEL_SEEDS) * len(NEW_CLAMPED_LAMBDAS) * len(INSTANCE_SEEDS)
        ),
    }


def authenticate_contract(project_root: Path, output_dir: Path, e04_output: Path) -> dict:
    observed = _load_json(output_dir / CONTRACT_NAME, label="E5 contract")
    _verify_hash(observed, "contract_sha256", label="E5 contract")
    if observed != _contract(project_root, e04_output):
        raise E5Error("E5 contract, parent E4 panel, frozen inputs, or sources changed")
    return observed


def _identity(record: Mapping) -> dict:
    return e04._identity(record)


def _spec(model_seed: int, value: float) -> dict:
    return {
        "ablation_condition": CLAMPED,
        "model_seed": int(model_seed),
        "deployment_lambda": float(value),
        "handling_input_lambda": CLAMPED_INPUT_LAMBDA,
    }


def _ledger_path(output_dir: Path, spec: Mapping, instance_seed: int) -> Path:
    return (
        output_dir
        / "run-ledger"
        / f"model-{int(spec['model_seed'])}"
        / f"lambda-{float(spec['deployment_lambda']):.3f}"
        / f"instance-{int(instance_seed)}.json"
    )


def _factory(project_root: Path, auth: Mapping, model_seed: int, value: float):
    def factory(base):
        agent = final90._load_conditioned_agent(
            project_root,
            auth,
            model_seed=model_seed,
            base=base,
            device=base.device,
        )
        agent.set_epsilon(0.0)
        agent.handling_network = ClampedPreferenceNetwork(
            agent.handling_network, CLAMPED_INPUT_LAMBDA
        ).to(base.device).eval()
        return conditioned_seed0._FixedLambdaAgent(agent, value)

    return factory


def _row(raw: Mapping, compact: Mapping, *, identity: Mapping, spec: Mapping) -> dict:
    proxy_spec = {
        "ranking_signal": CLAMPED,
        "model_seed": spec["model_seed"],
        "preference_lambda": spec["deployment_lambda"],
        "ranking_seed": None,
    }
    row = e04._row(raw, compact, identity=identity, spec=proxy_spec)
    row.update(
        {
            "protocol": PROTOCOL,
            "ablation_condition": CLAMPED,
            "deployment_lambda": spec["deployment_lambda"],
            "handling_input_lambda": spec["handling_input_lambda"],
        }
    )
    return row


def _failed_row(error: Exception, *, identity: Mapping, spec: Mapping) -> dict:
    proxy_spec = {
        "ranking_signal": CLAMPED,
        "model_seed": spec["model_seed"],
        "preference_lambda": spec["deployment_lambda"],
        "ranking_seed": None,
    }
    row = e04._failed_row(error, identity=identity, spec=proxy_spec)
    row.update(
        {
            "protocol": PROTOCOL,
            "ablation_condition": CLAMPED,
            "deployment_lambda": spec["deployment_lambda"],
            "handling_input_lambda": spec["handling_input_lambda"],
        }
    )
    return row


def run(
    project_root: Path,
    output_dir: Path,
    e04_output: Path,
    *,
    selected_seed: Optional[int],
    instance_limit: Optional[int],
    device_name: str,
) -> dict:
    contract = authenticate_contract(project_root, output_dir, e04_output)
    manifest = e04.authenticate_manifest(project_root, e04_output)
    seeds = MODEL_SEEDS if selected_seed is None else (selected_seed,)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise E5Error("model seed must be 0, 1, or 2")
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E5Error("E5 is frozen to CPU execution")
    records = manifest["instances"]
    if instance_limit is not None:
        if instance_limit < 1 or instance_limit > len(records):
            raise E5Error("invalid E5 instance limit")
        records = records[:instance_limit]
    auth = final90._authenticate_inputs(project_root)
    conditioned = auth["conditioned"]
    complete = safe = 0
    for model_seed in seeds:
        arm = conditioned["inputs"]["arms"][model_seed]
        for value in NEW_CLAMPED_LAMBDAS:
            spec = _spec(model_seed, value)
            for record in records:
                identity = _identity(record)
                path = _ledger_path(output_dir, spec, identity["instance_seed"])
                if path.is_file():
                    ledger = _load_json(path, label="E5 rollout ledger")
                    _verify_hash(ledger, "ledger_sha256", label="E5 rollout ledger")
                    if (
                        ledger.get("contract_sha256") != contract["contract_sha256"]
                        or ledger.get("e04_manifest_sha256") != manifest["manifest_sha256"]
                        or ledger.get("spec") != spec
                    ):
                        raise E5Error("E5 rollout ledger binding changed")
                    row = ledger["run"]
                else:
                    instance = e04._load_instance(e04_output, record)
                    try:
                        raw = pilot._run_raw(
                            arm,
                            instance,
                            device=device,
                            wrapper_factory=_factory(
                                project_root, auth, model_seed, value
                            ),
                        )
                        compact = pilot._compact_row(raw, instance)
                        row = _row(raw, compact, identity=identity, spec=spec)
                    except Exception as error:
                        row = _failed_row(error, identity=identity, spec=spec)
                    ledger = _with_hash(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "protocol": PROTOCOL,
                            "contract_sha256": contract["contract_sha256"],
                            "e04_manifest_sha256": manifest["manifest_sha256"],
                            "spec": spec,
                            "run": row,
                        },
                        "ledger_sha256",
                    )
                    final90._atomic_json(path, ledger)
                complete += 1
                safe += int(row["strict_safe_complete"])
                if complete % 10 == 0:
                    print(
                        f"E5(b) {complete}/{len(seeds) * len(NEW_CLAMPED_LAMBDAS) * len(records)} "
                        f"| safe={safe}",
                        flush=True,
                    )
    return {
        "status": "complete",
        "new_clamped_rows": complete,
        "strict_safe_complete_rows": safe,
        "model_seeds": list(seeds),
        "instances": len(records),
    }


def _all_clamped_rows(output_dir: Path) -> list[dict]:
    rows = []
    root = output_dir / "run-ledger"
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("**/instance-*.json")):
        ledger = _load_json(path, label="E5 rollout ledger")
        _verify_hash(ledger, "ledger_sha256", label="E5 rollout ledger")
        rows.append(ledger["run"])
    return rows


def _control_row(e04_output: Path, model_seed: int, value: float, instance_seed: int):
    spec = {
        "ranking_signal": FULL,
        "model_seed": model_seed,
        "preference_lambda": value,
        "ranking_seed": None,
    }
    path = e04._ledger_path(e04_output, spec, instance_seed)
    if not path.is_file():
        return None
    ledger = e04._load_json(path, label="E4 conditioned control ledger")
    e04._verify_hash(ledger, "ledger_sha256", label="E4 conditioned control ledger")
    return ledger["run"]


METRICS = (
    "dense_return",
    "mean_absolute_error",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


def _aggregate(rows: Sequence[Mapping]) -> dict:
    safe = [row for row in rows if row["strict_safe_complete"]]
    result = {
        "rows": len(rows),
        "strict_safe_complete": len(safe),
        "strict_completion_rate": len(safe) / len(rows) if rows else None,
        "complete_case_metrics_suppressed": len(safe) != len(rows),
    }
    for metric in METRICS:
        result[metric] = (
            fmean(float(row[metric]) for row in safe)
            if rows and len(safe) == len(rows)
            else None
        )
    return result


def _table(report: Mapping) -> str:
    lines = [
        "# E5(b) conditioning-input ablation",
        "",
        "| Deployment lambda | Condition | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for value in DEPLOYMENT_LAMBDAS:
        for condition in (FULL, CLAMPED):
            item = report["aggregate"][f"{condition}_lambda_{value:.2f}"]
            def metric(name: str, digits: int = 2) -> str:
                observed = item[name]
                return "—" if observed is None else f"{observed:.{digits}f}"
            lines.append(
                f"| {value:.2f} | {condition} | "
                f"{item['strict_safe_complete']}/{item['rows']} | "
                f"{metric('dense_return')} | {metric('mean_absolute_error')} | "
                f"{metric('within_target_window_rate', 3)} | {metric('steps')} | "
                f"{metric('physical_rehandles_per_100_required_deliveries')} |"
            )
    lines.extend(
        [
            "",
            "At lambda=.10 the two conditions are the same computation by construction; the E4 row is reused for both.",
            "This tests inference-time use of conditioning, not whether conditioned training is necessary.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    project_root: Path,
    output_dir: Path,
    e04_output: Path,
    *,
    allow_partial: bool,
) -> dict:
    contract = authenticate_contract(project_root, output_dir, e04_output)
    manifest = e04.authenticate_manifest(project_root, e04_output)
    clamped_rows = _all_clamped_rows(output_dir)
    expected_new = len(MODEL_SEEDS) * len(NEW_CLAMPED_LAMBDAS) * len(INSTANCE_SEEDS)
    if not allow_partial and len(clamped_rows) != expected_new:
        raise E5Error(f"E5(b) clamped grid is incomplete: {len(clamped_rows)}/{expected_new}")

    clamped_by_key = {
        (
            int(row["model_seed"]),
            float(row["deployment_lambda"]),
            int(row["instance_seed"]),
        ): row
        for row in clamped_rows
    }
    control_rows = []
    for model_seed, value, instance_seed in sorted(clamped_by_key):
        control = _control_row(e04_output, model_seed, value, instance_seed)
        if control is None:
            raise E5Error("a matched E4 conditioned control is missing")
        control_rows.append(control)

    # The midpoint is identical by construction. Reuse every available E4
    # midpoint row, rather than spending rollouts to reproduce an identity.
    midpoint_controls = []
    for model_seed in MODEL_SEEDS:
        for instance_seed in INSTANCE_SEEDS:
            row = _control_row(
                e04_output, model_seed, CLAMPED_INPUT_LAMBDA, instance_seed
            )
            if row is not None:
                midpoint_controls.append(row)

    aggregate = {}
    for value in NEW_CLAMPED_LAMBDAS:
        full_rows = [
            row for row in control_rows if float(row["preference_lambda"]) == value
        ]
        clamp = [
            row for row in clamped_rows if float(row["deployment_lambda"]) == value
        ]
        aggregate[f"{FULL}_lambda_{value:.2f}"] = _aggregate(full_rows)
        aggregate[f"{CLAMPED}_lambda_{value:.2f}"] = _aggregate(clamp)
    midpoint_aggregate = _aggregate(midpoint_controls)
    aggregate[f"{FULL}_lambda_{CLAMPED_INPUT_LAMBDA:.2f}"] = midpoint_aggregate
    aggregate[f"{CLAMPED}_lambda_{CLAMPED_INPUT_LAMBDA:.2f}"] = dict(
        midpoint_aggregate
    )

    paired_differences = {}
    for value in NEW_CLAMPED_LAMBDAS:
        pairs = []
        for key, clamp in clamped_by_key.items():
            model_seed, observed_lambda, instance_seed = key
            if observed_lambda != value:
                continue
            control = _control_row(e04_output, model_seed, value, instance_seed)
            if control is None:
                raise E5Error("a matched E4 conditioned control is missing")
            pairs.append((control, clamp))
        paired_differences[f"lambda_{value:.2f}"] = {
            "pairs": len(pairs),
            "strict_completion_clamped_minus_full": fmean(
                int(clamp["strict_safe_complete"])
                - int(control["strict_safe_complete"])
                for control, clamp in pairs
            ) if pairs else None,
            **{
                f"{metric}_clamped_minus_full": (
                    fmean(
                        float(clamp[metric]) - float(control[metric])
                        for control, clamp in pairs
                    )
                    if pairs and all(
                        control["strict_safe_complete"] and clamp["strict_safe_complete"]
                        for control, clamp in pairs
                    )
                    else None
                )
                for metric in METRICS
            },
        }

    complete = len(clamped_rows) == expected_new and len(midpoint_controls) == (
        len(MODEL_SEEDS) * len(INSTANCE_SEEDS)
    )
    validation = _assemble_validation()
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete" if complete else "partial",
            "paper_evidence": complete,
            "contract_sha256": contract["contract_sha256"],
            "e04_manifest_sha256": manifest["manifest_sha256"],
            "e05a_validation_sha256": validation["validation_sha256"],
            "new_clamped_rows": len(clamped_rows),
            "expected_new_clamped_rows": expected_new,
            "midpoint_identity_rows_reused": len(midpoint_controls),
            "midpoint_exact_identity_by_construction": True,
            "aggregate": aggregate,
            "paired_clamped_minus_full": paired_differences,
            "claim_boundary": (
                "inference_time_conditioning_input_ablation_not_conditioned_training_ablation"
            ),
        },
        "report_sha256",
    )
    final90._atomic_json(output_dir / REPORT_NAME, report)
    final90._atomic_text(output_dir / TABLE_NAME, _table(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "run-all", "analyze"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e04-output", type=Path, default=E04_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    e04_output = args.e04_output.resolve()
    if args.command == "prepare":
        result = prepare(project_root, output_dir, e04_output)
    elif args.command == "run":
        result = run(
            project_root,
            output_dir,
            e04_output,
            selected_seed=args.seed,
            instance_limit=args.instance_limit,
            device_name=args.device,
        )
    elif args.command == "run-all":
        result = run(
            project_root,
            output_dir,
            e04_output,
            selected_seed=None,
            instance_limit=None,
            device_name=args.device,
        )
    else:
        result = analyze(
            project_root,
            output_dir,
            e04_output,
            allow_partial=args.allow_partial,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
