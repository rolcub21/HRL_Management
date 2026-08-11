#!/usr/bin/env python3
"""Analyze the one-seed unified-VCG developmental handling-budget sweep."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import compare_unified_vcg as pair
import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified_budget_sweep as sweep


LAMBDA_ZERO_TOLERANCE = 0.01
ACTIVE_RESIDUAL_TOLERANCE_PER_100 = 2.0
COMPLEMENTARITY_TOLERANCE = 0.20

FIELDS = (
    "dense_return",
    "mean_absolute_error",
    "physical_rehandles_per_100",
    "steps",
    "within_window_percentage",
    "mean_earliness",
    "mean_tardiness",
)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error


def _common_base_configuration(contract: Mapping) -> dict:
    base = deepcopy(contract.get("base_v2_3_runtime_contract"))
    if not isinstance(base, dict):
        raise ValueError("training contract lacks the V2.3 runtime contract")
    base.pop("contract_sha256", None)
    base.pop("dual", None)
    return base


def _load_arm(root: Path, arm: sweep.SweepArm) -> dict:
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
    if summary.get("variant") != arm.variant:
        raise ValueError(f"{arm.name} variant mismatch")
    if summary.get("sweep_arm") != arm.to_dict():
        raise ValueError(f"{arm.name} sweep-arm binding mismatch")
    if summary.get("completed_training_episodes") != sweep.EPISODES:
        raise ValueError(f"{arm.name} did not complete the fixed horizon")
    if contract.get("method_version") != sweep.METHOD_VERSION:
        raise ValueError(f"{arm.name} contract method mismatch")
    if contract.get("base_v2_3_runtime_contract", {}).get("episodes") != sweep.EPISODES:
        raise ValueError(f"{arm.name} contract horizon mismatch")
    if contract.get("variant", {}).get("name") != arm.variant:
        raise ValueError(f"{arm.name} contract variant mismatch")

    final_validation = summary.get("final_validation")
    lambda_state = summary.get("lambda_state")
    if not isinstance(final_validation, dict) or not isinstance(lambda_state, dict):
        raise ValueError(f"{arm.name} terminal state is missing")
    if final_validation.get("checkpoint_episode") != sweep.EPISODES:
        raise ValueError(f"{arm.name} was not evaluated at episode {sweep.EPISODES}")
    if final_validation.get("strict_integrity_gate") is not True:
        raise ValueError(f"{arm.name} terminal validation is unsafe or incomplete")
    if not math.isclose(
        float(final_validation.get("budget_per_100_required_deliveries")),
        arm.budget_per_100,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{arm.name} handling budget mismatch")
    if lambda_state.get("observed_block_count") != sweep.TOTAL_BLOCKS - v23.WARMUP_BLOCKS:
        raise ValueError(f"{arm.name} dual block count mismatch")
    if lambda_state.get("last_completed_block") != sweep.TOTAL_BLOCKS:
        raise ValueError(f"{arm.name} terminal dual block mismatch")
    expected_updates = 0 if not arm.constrained else sweep.TOTAL_BLOCKS - v23.WARMUP_BLOCKS - 1
    if lambda_state.get("applied_update_count") != expected_updates:
        raise ValueError(f"{arm.name} applied dual-update count mismatch")
    if not arm.constrained and float(lambda_state.get("lambda_value")) != 0.0:
        raise ValueError("lambda-zero arm changed lambda")

    if not isinstance(ledger, dict) or ledger.get("checkpoint_episode") != sweep.EPISODES:
        raise ValueError(f"{arm.name} terminal ledger mismatch")
    if ledger.get("method_version") != sweep.METHOD_VERSION:
        raise ValueError(f"{arm.name} ledger method mismatch")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != 48 or ledger.get("row_count") != 48:
        raise ValueError(f"{arm.name} requires all 48 terminal rows")
    if not isinstance(training, list) or len(training) != sweep.EPISODES:
        raise ValueError(f"{arm.name} training history is incomplete")
    if not isinstance(lambda_history, list) or len(lambda_history) != (
        sweep.TOTAL_BLOCKS - v23.WARMUP_BLOCKS
    ):
        raise ValueError(f"{arm.name} lambda history is incomplete")
    if lambda_history[-1].get("block_number") != sweep.TOTAL_BLOCKS:
        raise ValueError(f"{arm.name} terminal dual proposal is missing")
    if lambda_history[-1].get("applied") is not False:
        raise ValueError(f"{arm.name} terminal dual proposal was applied")

    points = pair._instance_points(rows)
    final_block = training[-v23.BLOCK_EPISODES :]
    if [int(row.get("training_episode_number", -1)) for row in final_block] != list(
        range(sweep.EPISODES - v23.BLOCK_EPISODES + 1, sweep.EPISODES + 1)
    ):
        raise ValueError(f"{arm.name} final training block is not contiguous")
    required = sum(int(row["required_deliveries"]) for row in final_block)
    terminal_training_rate = (
        100.0 * sum(int(row["physical_rehandles"]) for row in final_block) / required
    )
    return {
        "arm": arm,
        "summary": summary,
        "contract": contract,
        "rows": rows,
        "points": points,
        "instance_manifest": instance_manifest,
        "terminal_training_rate": terminal_training_rate,
        "terminal_dual_proposal": lambda_history[-1],
    }


def _means(points: Mapping[int, Mapping]) -> dict:
    return {
        field: float(fmean(float(point[field]) for point in points.values()))
        for field in FIELDS
    }


def _kkt_classification(lambda_value: float, residual: float) -> str:
    product = lambda_value * residual
    if lambda_value <= LAMBDA_ZERO_TOLERANCE and residual < 0.0:
        return "inactive"
    if (
        lambda_value > LAMBDA_ZERO_TOLERANCE
        and abs(residual) <= ACTIVE_RESIDUAL_TOLERANCE_PER_100
        and abs(product) <= COMPLEMENTARITY_TOLERANCE
    ):
        return "active"
    return "unconverged"


def _diagnostics(loaded: Mapping) -> dict:
    arm = loaded["arm"]
    summary = loaded["summary"]
    validation = summary["final_validation"]
    lambda_value = float(summary["lambda_state"]["lambda_value"])
    if not arm.constrained:
        return {
            "classification": "unconstrained_lambda_fixed_zero",
            "lambda": lambda_value,
            "declared_budget_per_100": None,
            "validation_J_N_per_100": float(
                validation["expected_physical_rehandles_per_100_required_deliveries"]
            ),
            "validation_primal_residual_g": None,
            "validation_complementarity_lambda_g": None,
            "terminal_training_block_J_N_per_100": loaded["terminal_training_rate"],
            "terminal_training_block_primal_residual_g": None,
            "terminal_training_block_complementarity_lambda_g": None,
        }

    budget = arm.budget_per_100
    validation_cost = float(
        validation["expected_physical_rehandles_per_100_required_deliveries"]
    )
    validation_residual = validation_cost - budget
    training_cost = float(loaded["terminal_training_rate"])
    training_residual = training_cost - budget
    return {
        "classification": _kkt_classification(lambda_value, validation_residual),
        "training_distribution_classification": _kkt_classification(
            lambda_value, training_residual
        ),
        "lambda": lambda_value,
        "declared_budget_per_100": budget,
        "validation_J_N_per_100": validation_cost,
        "validation_primal_residual_g": validation_residual,
        "validation_complementarity_lambda_g": lambda_value * validation_residual,
        "validation_one_sided_95_ucb": float(
            validation["physical_rehandle_rate_one_sided_95_ucb"]
        ),
        "terminal_training_block_J_N_per_100": training_cost,
        "terminal_training_block_primal_residual_g": training_residual,
        "terminal_training_block_complementarity_lambda_g": (
            lambda_value * training_residual
        ),
        "terminal_dual_proposal": loaded["terminal_dual_proposal"],
    }


def _dominates(left: Mapping, right: Mapping) -> bool:
    left_pair = (
        float(left["mean_absolute_error"]),
        float(left["physical_rehandles_per_100"]),
    )
    right_pair = (
        float(right["mean_absolute_error"]),
        float(right["physical_rehandles_per_100"]),
    )
    return all(a <= b for a, b in zip(left_pair, right_pair)) and any(
        a < b for a, b in zip(left_pair, right_pair)
    )


def analyze(root: Path) -> dict:
    root = Path(root).resolve()
    loaded = {
        name: _load_arm(root / sweep.ARMS[name].output_name, sweep.ARMS[name])
        for name in sweep.ARM_ORDER
    }
    common = _common_base_configuration(loaded[sweep.LAMBDA_ZERO]["contract"])
    common_hash = v23.contract_hash(common)
    for name in sweep.ARM_ORDER[1:]:
        observed = _common_base_configuration(loaded[name]["contract"])
        if observed != common:
            raise ValueError(f"{name} differs from lambda0 beyond budget/dual control")
        if loaded[name]["instance_manifest"] != loaded[sweep.LAMBDA_ZERO][
            "instance_manifest"
        ]:
            raise ValueError(f"{name} validation EpisodeInstances drifted")

    metrics = {name: _means(loaded[name]["points"]) for name in sweep.ARM_ORDER}
    diagnostics = {name: _diagnostics(loaded[name]) for name in sweep.ARM_ORDER}
    frontier = tuple(
        name
        for name in sweep.ARM_ORDER
        if not any(
            _dominates(metrics[other], metrics[name])
            for other in sweep.ARM_ORDER
            if other != name
        )
    )
    paired = {
        name: {
            field: pair._paired_difference(
                loaded[name]["points"],
                loaded[sweep.LAMBDA_ZERO]["points"],
                field,
            )
            for field in FIELDS
        }
        for name in sweep.ARM_ORDER[1:]
    }
    table = []
    labels = {
        sweep.LAMBDA_ZERO: "VCG (lambda=0)",
        sweep.BUDGET_10: "VCG + handling constraint (B=10)",
        sweep.BUDGET_8: "VCG + handling constraint (B=8)",
        sweep.BUDGET_5: "VCG + handling constraint (B=5)",
    }
    for name in sweep.ARM_ORDER:
        row = {
            "arm": name,
            "display_name": labels[name],
            **metrics[name],
            **diagnostics[name],
            "developmental_point_estimate_pareto": name in frontier,
        }
        table.append(row)

    return {
        "status": "complete",
        "protocol": sweep.TRAINING_PROTOCOL,
        "scientific_role": "single_seed_post_hoc_developmental_budget_sweep",
        "model_seed": sweep.MODEL_SEED,
        "fixed_training_horizon": sweep.EPISODES,
        "arms": sweep.ARM_ORDER,
        "common_configuration_excluding_budget_and_dual_sha256": common_hash,
        "aggregation": (
            "four action RNGs averaged within each of 12 EpisodeInstances; "
            "arm means and paired differences use EpisodeInstance as the unit"
        ),
        "kkt_tolerances": {
            "lambda_zero": LAMBDA_ZERO_TOLERANCE,
            "active_absolute_residual_per_100": ACTIVE_RESIDUAL_TOLERANCE_PER_100,
            "absolute_complementarity": COMPLEMENTARITY_TOLERANCE,
        },
        "table": table,
        "paired_differences_vs_lambda0": paired,
        "developmental_mae_rehandle_point_estimate_frontier": frontier,
        "frontier_claim_scope": (
            "four predeclared operating points for one developmental training seed; "
            "not a population-level or seed-stability frontier"
        ),
        "all_terminal_rows_strict_safe_complete": True,
        "new_final_panel_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = analyze(args.root)
    if args.output is not None:
        atomic_io._atomic_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

