#!/usr/bin/env python3
"""Analyze the unified-VCG fixed-handling-weight development sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import compare_unified_vcg as pair
import compare_vcg_unified_budget_sweep as budget_analysis
import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import train_vcg_unified_budget_sweep as budget_sweep
import train_vcg_unified_fixed_lambda_sweep as sweep


FIELDS = budget_analysis.FIELDS


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error


def _load_fixed_arm(root: Path, arm: sweep.FixedLambdaArm) -> dict:
    root = Path(root).resolve()
    summary = _read_json(root / "training-summary.json")
    contract = _read_json(root / "training-contract.json")
    ledger = _read_json(
        root / "validation-ledger" / f"episode-{sweep.EPISODES:04d}.json"
    )
    training = _read_json(root / "training-history.json")
    lambda_history = _read_json(root / "lambda-history.json")
    instance_manifest = _read_json(root / "validation-instance-manifest.json")

    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise ValueError(f"{arm.name} training is not complete")
    if summary.get("method_version") != sweep.METHOD_VERSION:
        raise ValueError(f"{arm.name} method version mismatch")
    if summary.get("variant") != unified.VCG_FIXED_HANDLING_WEIGHT:
        raise ValueError(f"{arm.name} variant mismatch")
    if summary.get("fixed_lambda_sweep_arm") != arm.to_dict():
        raise ValueError(f"{arm.name} arm binding mismatch")
    if summary.get("completed_training_episodes") != sweep.EPISODES:
        raise ValueError(f"{arm.name} fixed horizon is incomplete")
    if contract.get("method_version") != sweep.METHOD_VERSION:
        raise ValueError(f"{arm.name} contract method mismatch")
    variant = contract.get("variant", {})
    if variant.get("name") != unified.VCG_FIXED_HANDLING_WEIGHT or not math.isclose(
        float(variant.get("fixed_lambda_after_warmup")),
        arm.fixed_lambda,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{arm.name} fixed-lambda contract mismatch")

    validation = summary.get("final_validation")
    state = summary.get("lambda_state")
    if not isinstance(validation, dict) or not isinstance(state, dict):
        raise ValueError(f"{arm.name} terminal state is missing")
    if validation.get("checkpoint_episode") != sweep.EPISODES:
        raise ValueError(f"{arm.name} terminal checkpoint mismatch")
    if validation.get("strict_integrity_gate") is not True:
        raise ValueError(f"{arm.name} terminal validation is unsafe or incomplete")
    if not math.isclose(
        float(validation.get("validation_lambda")),
        arm.fixed_lambda,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{arm.name} terminal lambda mismatch")
    if state.get("observed_block_count") != sweep.TOTAL_BLOCKS - v23.WARMUP_BLOCKS:
        raise ValueError(f"{arm.name} block count mismatch")
    if state.get("applied_update_count") != 0:
        raise ValueError(f"{arm.name} unexpectedly applied a dual update")
    if not math.isclose(
        float(state.get("lambda_value")), arm.fixed_lambda, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{arm.name} lambda state drifted")

    rows = ledger.get("rows") if isinstance(ledger, dict) else None
    if (
        ledger.get("checkpoint_episode") != sweep.EPISODES
        or ledger.get("method_version") != sweep.METHOD_VERSION
        or not isinstance(rows, list)
        or len(rows) != 48
        or ledger.get("row_count") != 48
    ):
        raise ValueError(f"{arm.name} terminal ledger is incomplete")
    if not isinstance(training, list) or len(training) != sweep.EPISODES:
        raise ValueError(f"{arm.name} training history is incomplete")
    warmup = training[: v23.WARMUP_EPISODES]
    active = training[v23.WARMUP_EPISODES :]
    if any(float(row["dual_lambda"]) != 0.0 for row in warmup):
        raise ValueError(f"{arm.name} did not use the common lambda-zero warm-up")
    if any(
        not math.isclose(
            float(row["dual_lambda"]), arm.fixed_lambda, rel_tol=0.0, abs_tol=1e-12
        )
        for row in active
    ):
        raise ValueError(f"{arm.name} fixed lambda drifted during training")
    if not isinstance(lambda_history, list) or len(lambda_history) != (
        sweep.TOTAL_BLOCKS - v23.WARMUP_BLOCKS
    ):
        raise ValueError(f"{arm.name} lambda history is incomplete")
    if any(
        item.get("applied") is not False
        or item.get("reason_not_applied") != "fixed_handling_weight"
        for item in lambda_history
    ):
        raise ValueError(f"{arm.name} contains an adaptive dual update")

    return {
        "arm": arm,
        "summary": summary,
        "contract": contract,
        "points": pair._instance_points(rows),
        "instance_manifest": instance_manifest,
    }


def _means(points: Mapping[int, Mapping]) -> dict:
    return {
        field: float(fmean(float(point[field]) for point in points.values()))
        for field in FIELDS
    }


def _dominates(left: Mapping, right: Mapping) -> bool:
    a = (float(left["mean_absolute_error"]), float(left["physical_rehandles_per_100"]))
    b = (float(right["mean_absolute_error"]), float(right["physical_rehandles_per_100"]))
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def analyze(root: Path, lambda_zero_dir: Path) -> dict:
    root = Path(root).resolve()
    parent = budget_analysis._load_arm(
        Path(lambda_zero_dir), budget_sweep.ARMS[budget_sweep.LAMBDA_ZERO]
    )
    fixed = {
        arm.name: _load_fixed_arm(root / arm.name, arm)
        for arm in sweep.ARMS
    }
    parent_common = budget_analysis._common_base_configuration(parent["contract"])
    for name, loaded in fixed.items():
        if budget_analysis._common_base_configuration(loaded["contract"]) != parent_common:
            raise ValueError(f"{name} differs from lambda0 beyond lambda control")
        if loaded["instance_manifest"] != parent["instance_manifest"]:
            raise ValueError(f"{name} validation EpisodeInstances drifted")

    points = {"lambda-0.00": parent["points"]}
    points.update({name: loaded["points"] for name, loaded in fixed.items()})
    metrics = {name: _means(value) for name, value in points.items()}
    order = ("lambda-0.00",) + tuple(arm.name for arm in sweep.ARMS)
    frontier = tuple(
        name
        for name in order
        if not any(
            _dominates(metrics[other], metrics[name])
            for other in order
            if other != name
        )
    )
    paired = {
        name: {
            field: pair._paired_difference(points[name], points["lambda-0.00"], field)
            for field in FIELDS
        }
        for name in order[1:]
    }
    lambda_values = (0.0,) + sweep.FIXED_LAMBDAS
    rehandle_values = tuple(metrics[name]["physical_rehandles_per_100"] for name in order)
    table = [
        {
            "arm": name,
            "fixed_lambda": fixed_lambda,
            **metrics[name],
            "developmental_point_estimate_pareto": name in frontier,
        }
        for name, fixed_lambda in zip(order, lambda_values)
    ]
    return {
        "status": "complete",
        "protocol": sweep.TRAINING_PROTOCOL,
        "scientific_role": (
            "post_hoc_single_seed_developmental_fixed_handling_weight_sweep"
        ),
        "model_seed": sweep.MODEL_SEED,
        "fixed_training_horizon": sweep.EPISODES,
        "lambda_zero_parent_reused": str(Path(lambda_zero_dir).resolve()),
        "lambda_values": lambda_values,
        "table": table,
        "paired_differences_vs_lambda0": paired,
        "developmental_mae_rehandle_point_estimate_frontier": frontier,
        "rehandles_nonincreasing_as_lambda_increases": all(
            later <= earlier + 1e-12
            for earlier, later in zip(rehandle_values, rehandle_values[1:])
        ),
        "aggregation": (
            "four action RNGs averaged within each of 12 EpisodeInstances; "
            "EpisodeInstance is the paired unit"
        ),
        "claim_scope": (
            "fixed-lambda operating points for one developmental model seed; "
            "no budget satisfaction, KKT, seed-stability, or final-panel claim"
        ),
        "all_terminal_rows_strict_safe_complete": True,
        "new_final_panel_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lambda-zero-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = analyze(args.root, args.lambda_zero_dir)
    if args.output is not None:
        atomic_io._atomic_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
