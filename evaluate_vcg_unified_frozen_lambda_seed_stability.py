#!/usr/bin/env python3
"""Test frozen deployment lambda 0 versus .05 across model seeds 15--17."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import torch

import compare_unified_vcg as pair
import evaluate_vcg_unified_frozen_lambda_sweep as base
import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import train_vcg_unified_seed_stability as training_seeds


PROTOCOL = "vcg_unified_frozen_lambda_005_seed_stability_85k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (15, 16, 17)
LAMBDAS = (0.0, 0.05)
EPISODE = 200
EXPECTED_METHOD = "vcg_unified_v1"
EXPECTED_FAMILY = "vcg_unified_v1_terminal_development"
MAE_POINT_NONINFERIORITY_MARGIN = 2.0
FIELDS = base.FIELDS


class FrozenLambdaSeedError(ValueError):
    """Raised when the frozen-lambda seed protocol is violated."""


def _read_mapping(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenLambdaSeedError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise FrozenLambdaSeedError(f"{path} must contain an object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(value: float) -> str:
    return f"lambda-{float(value):.2f}"


def _parent(root: Path, model_seed: int) -> dict:
    path = Path(root).resolve() / f"seed-{model_seed}" / "vcg"
    summary_path = path / "training-summary.json"
    contract_path = path / "training-contract.json"
    checkpoint_path = path / "final-model.pth"
    summary = _read_mapping(summary_path)
    contract = _read_mapping(contract_path)
    if (
        summary.get("status") != "complete"
        or summary.get("variant") != unified.VCG
        or summary.get("method_version") != EXPECTED_METHOD
        or summary.get("completed_training_episodes") != EPISODE
        or float(summary.get("lambda_state", {}).get("lambda_value", -1.0)) != 0.0
        or summary.get("final_validation", {}).get("strict_integrity_gate") is not True
    ):
        raise FrozenLambdaSeedError(f"seed {model_seed} lambda-zero parent mismatch")
    base_contract = contract.get("base_v2_3_runtime_contract")
    if (
        contract.get("method_version") != EXPECTED_METHOD
        or not isinstance(base_contract, dict)
        or base_contract.get("model_seed") != model_seed
        or base_contract.get("episodes") != EPISODE
    ):
        raise FrozenLambdaSeedError(f"seed {model_seed} training contract mismatch")
    if not checkpoint_path.is_file():
        raise FrozenLambdaSeedError(f"seed {model_seed} checkpoint is missing")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise FrozenLambdaSeedError(f"seed {model_seed} checkpoint cannot load safely") from error
    expected = {
        "checkpoint_family": EXPECTED_FAMILY,
        "method_version": EXPECTED_METHOD,
        "unified_variant": unified.VCG,
        "completed_episodes": EPISODE,
        "terminal_checkpoint": True,
        "runtime_lambda": 0.0,
        "training_contract_sha256": contract.get("contract_sha256"),
        "development_only": True,
        "deployment_checkpoint_eligible": False,
        "final_86xxx_panel_opened": False,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise FrozenLambdaSeedError(
            f"seed {model_seed} checkpoint mismatch: {mismatches!r}"
        )
    return {
        "root": path,
        "summary": summary,
        "contract": contract,
        "base_contract": base_contract,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _sha(checkpoint_path),
        "summary_sha256": _sha(summary_path),
        "contract_sha256_raw": _sha(contract_path),
    }


def _contract(parents: Mapping[int, Mapping], *, device: str) -> dict:
    if any(parent["base_contract"].get("device") != device for parent in parents.values()):
        raise FrozenLambdaSeedError("device must match every frozen parent")
    source = Path(__file__).resolve()
    base_source = Path(base.__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_role": "developmental_frozen_lambda_candidate_seed_stability",
        "model_seeds": MODEL_SEEDS,
        "lambda_values": LAMBDAS,
        "checkpoint_episode": EPISODE,
        "parents": tuple(
            {
                "model_seed": seed,
                "root": str(parents[seed]["root"]),
                "checkpoint": str(parents[seed]["checkpoint_path"]),
                "checkpoint_raw_sha256": parents[seed]["checkpoint_sha256"],
                "summary_raw_sha256": parents[seed]["summary_sha256"],
                "contract_raw_sha256": parents[seed]["contract_sha256_raw"],
                "contract_canonical_sha256": parents[seed]["contract"]["contract_sha256"],
            }
            for seed in MODEL_SEEDS
        ),
        "validation_instance_seeds": tuple(v23.DEFAULT_VALIDATION_SEEDS),
        "action_rng_range": (
            unified.VALIDATION_POLICY_RNG_BASE,
            unified.VALIDATION_POLICY_RNG_BASE + 47,
        ),
        "rows_per_seed_lambda": 48,
        "expected_total_rows": len(MODEL_SEEDS) * len(LAMBDAS) * 48,
        "aggregation": (
            "four action RNGs within EpisodeInstance, equal model seeds within "
            "EpisodeInstance, then 12 EpisodeInstances"
        ),
        "candidate_lambda": 0.05,
        "continuation_criterion": {
            "all_rows_strict_safe_complete": True,
            "aggregate_rehandle_difference_below_zero": True,
            "at_least_two_of_three_seed_rehandle_differences_below_zero": True,
            "aggregate_mae_point_difference_maximum": MAE_POINT_NONINFERIORITY_MARGIN,
        },
        "device": device,
        "source_sha256": _sha(source),
        "base_evaluator_source_sha256": _sha(base_source),
        "training_or_learning": False,
        "development_only": True,
        "final_86xxx_panel_opened": False,
    }
    result["contract_sha256"] = v23.contract_hash(result)
    return result


def prepare(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    parents = {seed: _parent(parent_root, seed) for seed in MODEL_SEEDS}
    contract = _contract(parents, device=device)
    output = Path(output_dir).resolve()
    path = output / "evaluation-contract.json"
    if path.exists():
        if v23._json_safe(_read_mapping(path)) != v23._json_safe(contract):
            raise FrozenLambdaSeedError("existing evaluation contract drifted")
    else:
        if output.exists() and any(output.iterdir()):
            raise FrozenLambdaSeedError("new evaluation root is not empty")
        atomic_io._atomic_json(contract, path)
    return contract


def _validate_ledger(ledger: Mapping, *, seed: int, fixed_lambda: float) -> list[dict]:
    canonical = dict(ledger)
    received_sha = canonical.pop("ledger_sha256", None)
    if (
        ledger.get("protocol") != PROTOCOL
        or ledger.get("model_seed") != seed
        or not math.isclose(float(ledger.get("fixed_lambda")), fixed_lambda, abs_tol=1e-12)
        or ledger.get("row_count") != 48
        or ledger.get("validation_summary", {}).get("strict_integrity_gate") is not True
        or not isinstance(received_sha, str)
        or v23.contract_hash(canonical) != received_sha
    ):
        raise FrozenLambdaSeedError(f"seed {seed} {_key(fixed_lambda)} ledger mismatch")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        raise FrozenLambdaSeedError(f"seed {seed} {_key(fixed_lambda)} rows missing")
    expected = {
        (
            int(instance_seed),
            rng_index,
            unified.VALIDATION_POLICY_RNG_BASE + 4 * instance_index + rng_index,
        )
        for instance_index, instance_seed in enumerate(v23.DEFAULT_VALIDATION_SEEDS)
        for rng_index in range(4)
    }
    observed = {
        (
            int(row["instance_seed"]),
            int(row["policy_rng_index"]),
            int(row["policy_rng_seed"]),
        )
        for row in rows
    }
    if observed != expected or any(
        not math.isclose(float(row["dual_lambda"]), fixed_lambda, abs_tol=1e-12)
        or row.get("strict_method_success") is not True
        or row.get("delivery_count") != row.get("required_deliveries")
        or row.get("method_failure_reason") is not None
        or row.get("all_selected_candidates_exact_safe") is not True
        or any(int(row.get(field, -1)) for field in ("illegal_drops", "fallbacks", "witness_mismatches"))
        for row in rows
    ):
        raise FrozenLambdaSeedError(f"seed {seed} {_key(fixed_lambda)} unsafe/grid drift")
    pair._instance_points(rows)
    return rows


def evaluate(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_root, device=device)
    parents = {seed: _parent(parent_root, seed) for seed in MODEL_SEEDS}
    output = Path(output_dir).resolve()
    completed = {}
    for seed in MODEL_SEEDS:
        profile = training_seeds.profile_for_seed(seed)
        with ExitStack() as stack:
            stack.enter_context(training_seeds.activated_paired_profile(profile))
            stack.enter_context(unified.activated_unified_seed_profile())
            runtime = base._runtime(parents[seed], device=device)
            q_local = runtime._module_signature(runtime.agent.Q_local)
            q_target = runtime._module_signature(runtime.agent.Q_target)
            schedule = v23._install_schedule(
                runtime, v23.schedule_for_episode(EPISODE, validation=True)
            )
            for fixed_lambda in LAMBDAS:
                path = (
                    output
                    / "validation-ledger"
                    / f"seed-{seed}"
                    / f"{_key(fixed_lambda)}.json"
                )
                if path.exists():
                    rows = _validate_ledger(
                        _read_mapping(path), seed=seed, fixed_lambda=fixed_lambda
                    )
                else:
                    runtime.set_dual_lambda(fixed_lambda)
                    rows, batch = unified._validation_rows(
                        runtime,
                        SimpleNamespace(
                            validation_seeds=v23.DEFAULT_VALIDATION_SEEDS,
                            max_steps=2_000,
                        ),
                        schedule,
                        fixed_lambda,
                    )
                    summary = v23.summarize_validation(
                        rows,
                        v23.V23DualConfig(),
                        checkpoint_episode=EPISODE,
                        validation_lambda=fixed_lambda,
                        schedule_state=schedule,
                    )
                    if summary.get("strict_integrity_gate") is not True:
                        raise FrozenLambdaSeedError(
                            f"seed {seed} {_key(fixed_lambda)} unsafe/incomplete"
                        )
                    ledger = {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "model_seed": seed,
                        "parent_checkpoint_raw_sha256": parents[seed]["checkpoint_sha256"],
                        "fixed_lambda": fixed_lambda,
                        "training_or_learning": False,
                        "batch_audit": batch,
                        "validation_summary": summary,
                        "rows": tuple(rows),
                        "row_count": len(rows),
                        "final_86xxx_panel_opened": False,
                    }
                    ledger["ledger_sha256"] = v23.contract_hash(ledger)
                    atomic_io._atomic_json(ledger, path)
                    rows = _validate_ledger(
                        _read_mapping(path), seed=seed, fixed_lambda=fixed_lambda
                    )
                completed[f"seed-{seed}/{_key(fixed_lambda)}"] = len(rows)
            if (
                runtime._module_signature(runtime.agent.Q_local) != q_local
                or runtime._module_signature(runtime.agent.Q_target) != q_target
            ):
                raise FrozenLambdaSeedError(f"seed {seed} weights changed")
    return {
        "status": "evaluation_complete",
        "completed_rows": completed,
        "weights_unchanged": True,
        "final_86xxx_panel_opened": False,
    }


def _means(points: Mapping[int, Mapping]) -> dict:
    return {
        field: float(fmean(float(point[field]) for point in points.values()))
        for field in FIELDS
    }


def analyze(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_root, device=device)
    output = Path(output_dir).resolve()
    points = {seed: {} for seed in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        for fixed_lambda in LAMBDAS:
            ledger = _read_mapping(
                output
                / "validation-ledger"
                / f"seed-{seed}"
                / f"{_key(fixed_lambda)}.json"
            )
            points[seed][fixed_lambda] = pair._instance_points(
                _validate_ledger(ledger, seed=seed, fixed_lambda=fixed_lambda)
            )
    instance_seeds = sorted(points[MODEL_SEEDS[0]][0.0])
    aggregate = {
        fixed_lambda: {
            instance_seed: {
                field: fmean(
                    points[seed][fixed_lambda][instance_seed][field]
                    for seed in MODEL_SEEDS
                )
                for field in FIELDS
            }
            for instance_seed in instance_seeds
        }
        for fixed_lambda in LAMBDAS
    }
    aggregate_difference = {
        field: pair._paired_difference(aggregate[0.05], aggregate[0.0], field)
        for field in FIELDS
    }
    per_seed = {}
    improving = 0
    for seed in MODEL_SEEDS:
        difference = {
            field: pair._paired_difference(
                points[seed][0.05], points[seed][0.0], field
            )
            for field in FIELDS
        }
        improving += int(
            difference["physical_rehandles_per_100"]["mean_difference"] < 0.0
        )
        per_seed[str(seed)] = {
            "lambda0": _means(points[seed][0.0]),
            "lambda005": _means(points[seed][0.05]),
            "paired_difference_lambda005_minus_lambda0": difference,
        }
    all_safe = True
    aggregate_rehandles_lower = (
        aggregate_difference["physical_rehandles_per_100"]["mean_difference"] < 0.0
    )
    timing_ok = (
        aggregate_difference["mean_absolute_error"]["mean_difference"]
        <= MAE_POINT_NONINFERIORITY_MARGIN
    )
    passed = bool(all_safe and aggregate_rehandles_lower and improving >= 2 and timing_ok)
    report = {
        "status": "passed" if passed else "did_not_pass",
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seeds": MODEL_SEEDS,
        "candidate_lambda": 0.05,
        "aggregate": {
            "lambda0": _means(aggregate[0.0]),
            "lambda005": _means(aggregate[0.05]),
            "paired_difference_lambda005_minus_lambda0": aggregate_difference,
        },
        "per_seed": per_seed,
        "continuation_criterion": {
            "all_288_rows_strict_safe_complete": all_safe,
            "aggregate_rehandles_lower": aggregate_rehandles_lower,
            "seeds_with_lower_rehandles": improving,
            "at_least_two_of_three": improving >= 2,
            "strong_three_of_three": improving == 3,
            "aggregate_mae_point_difference_maximum": MAE_POINT_NONINFERIORITY_MARGIN,
            "aggregate_mae_point_noninferior": timing_ok,
            "passed": passed,
        },
        "same_frozen_weights_within_each_model_seed": True,
        "training_or_learning": False,
        "development_only": True,
        "final_86xxx_panel_opened": False,
    }
    atomic_io._atomic_json(report, output / "frozen-lambda-seed-stability.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.output_dir, args.parent_root, device=args.device)
    elif args.action == "run":
        result = {
            "evaluation": evaluate(args.output_dir, args.parent_root, device=args.device),
            "analysis": analyze(args.output_dir, args.parent_root, device=args.device),
        }
    else:
        result = analyze(args.output_dir, args.parent_root, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

