#!/usr/bin/env python3
"""Three-seed fixed-lambda frontier sweep for nested VCG on the 85k panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_v11_nested_handling_pilot as pilot
import run_vcg_v11_nested_handling_seed_stability as stability
from vcg_v11_nested_handling import HandlingAugmentedV11Agent


PROTOCOL = "vcg_v1_1_nested_fixed_lambda_frontier_85k_dev_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = tuple(range(85_000, 85_012))
LAMBDA_GRID = (0.0, 0.025, 0.05, 0.1, 0.2)
NEW_LAMBDAS = (0.05, 0.1, 0.2)
NEW_MODEL_SEEDS = (1, 2)
EXPECTED_ROWS = len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(LAMBDA_GRID)
EXPECTED_NEW_ROWS = len(NEW_MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(NEW_LAMBDAS)
CONTRACT_NAME = "frontier-contract.json"
REPORT_NAME = "frontier-report.json"


class FrontierError(RuntimeError):
    pass


def _key(value: float) -> str:
    return str(float(value))


def _paths(project_root: Path) -> dict:
    root = project_root.resolve()
    return {
        "pilot": root / "results/vcg-v1-1-nested-handling-seed0-85k-development",
        "stability": root
        / "results/vcg-v1-1-nested-handling-seed-stability-85k-development",
    }


def _source_hashes(project_root: Path) -> dict:
    root = project_root.resolve()
    paths = {
        "runner": Path(__file__).resolve(),
        "nested_controller": root / "vcg_v11_nested_handling.py",
        "pilot_runner": root / "run_vcg_v11_nested_handling_pilot.py",
        "stability_runner": root / "run_vcg_v11_nested_handling_seed_stability.py",
    }
    return {name: pilot._sha256_file(path) for name, path in sorted(paths.items())}


def _authenticate(project_root: Path):
    arms, sources, latest, datasets, parent, records = stability._authenticate_inputs(
        project_root
    )
    paths = _paths(project_root)
    report = json.loads((paths["stability"] / stability.REPORT_NAME).read_text())
    if (
        report.get("status") != "passed"
        or report.get("selected_lambda") != 0.025
        or report.get("stability_gate", {}).get("passed") is not True
    ):
        raise FrontierError("the three-seed nested parent did not pass")
    pilot_sweep = parent["sweep"]
    for value in LAMBDA_GRID:
        if value in (0.0, 0.025) or value in NEW_LAMBDAS:
            rows = pilot_sweep.get("rows", {}).get(_key(value))
            if not isinstance(rows, list) or len(rows) != len(INSTANCE_SEEDS):
                raise FrontierError(f"seed-0 pilot rows are incomplete for lambda={value}")
    return arms, sources, parent, report, paths


def _contract(project_root: Path) -> dict:
    arms, sources, parent, report, paths = _authenticate(project_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_one_model_fixed_lambda_frontier_on_opened_85k",
        "lambda_grid": list(LAMBDA_GRID),
        "model_seeds": list(MODEL_SEEDS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "total_grid_rows": EXPECTED_ROWS,
        "new_rows": EXPECTED_NEW_ROWS,
        "reused_rows": EXPECTED_ROWS - EXPECTED_NEW_ROWS,
        "training_or_learning": False,
        "same_frozen_qop_and_cost_head_within_each_model_seed": True,
        "lambda_zero_direct_vcg_1_1_delegation": True,
        "primary_plane": ["mean_absolute_error", "physical_rehandles_per_100"],
        "frontier_claim_rule": {
            "minimum_distinct_aggregate_nondominated_lambdas": 3,
            "minimum_seed_support_for_each_claimed_lambda": 2,
            "rehandles_broadly_nonincreasing": True,
            "complete_case_filtering_allowed": False,
        },
        "advance_rule": (
            "advance all five frozen lambda levels to an unseen 89k panel only "
            "if at least three distinct aggregate nondominated points have "
            "support in at least two of three seeds and handling inversions are not material"
        ),
        "redesign_rule": (
            "if coverage is narrow, unstable, plateaued, or materially nonmonotone, "
            "replace the historical-policy MC head with Q_N(s,c,lambda) trained "
            "on lambda-stratified policy rollouts while keeping Qop frozen"
        ),
        "selected_checkpoints": {
            str(seed): {
                "raw_sha256": arms[seed].checkpoint_sha256,
                "policy_digest": arms[seed].deployment_policy_digest,
            }
            for seed in MODEL_SEEDS
        },
        "cost_head_sha256": {
            "0": pilot._sha256_file(paths["pilot"] / pilot.COST_NAME),
            "1": pilot._sha256_file(stability._cost_path(paths["stability"], 1)),
            "2": pilot._sha256_file(stability._cost_path(paths["stability"], 2)),
        },
        "source_sha256": _source_hashes(project_root),
    }


def prepare(project_root: Path, output_dir: Path) -> dict:
    contract = _contract(project_root)
    path = output_dir / CONTRACT_NAME
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise FrontierError("existing frontier contract changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FrontierError("nonempty output directory lacks frontier contract")
        pilot._atomic_json(path, contract)
    return {
        "status": "prepared",
        "new_rows": EXPECTED_NEW_ROWS,
        "training": False,
        "contract": str(path),
    }


def _require_contract(project_root: Path, output_dir: Path) -> dict:
    path = output_dir / CONTRACT_NAME
    if not path.is_file():
        raise FrontierError("run prepare first")
    observed = json.loads(path.read_text())
    if observed != _contract(project_root):
        raise FrontierError("frontier contract or bound inputs changed")
    return observed


def _load_cost(project_root: Path, output_dir: Path, arm, *, seed: int, device, config):
    paths = _paths(project_root)
    if seed == 0:
        return pilot._load_bound_cost(
            paths["pilot"] / pilot.COST_NAME,
            arm,
            device=device,
            config=config,
        )
    return stability._load_cost(
        paths["stability"], arm, seed=seed, device=device, config=config
    )


def _ledger_path(output_dir: Path, seed: int, value: float) -> Path:
    token = f"{value:.3f}".replace(".", "p")
    return output_dir / "evaluation-ledger" / f"seed-{seed}-lambda-{token}.json"


def evaluate_seed_lambda(
    project_root: Path,
    output_dir: Path,
    *,
    seed: int,
    value: float,
    device_name: str,
) -> dict:
    _require_contract(project_root, output_dir)
    if seed not in NEW_MODEL_SEEDS or value not in NEW_LAMBDAS:
        raise FrontierError("only the predeclared 72 missing cells may execute")
    path = _ledger_path(output_dir, seed, value)
    if path.is_file():
        ledger = json.loads(path.read_text())
        if (
            ledger.get("model_seed") != seed
            or ledger.get("fixed_lambda") != value
            or len(ledger.get("rows", ())) != len(INSTANCE_SEEDS)
            or not all(row.get("strict_safe_complete") is True for row in ledger["rows"])
        ):
            raise FrontierError("existing frontier ledger changed")
        return ledger
    arms, sources, parent, report, paths = _authenticate(project_root)
    arm = arms[seed]
    device = pilot._device(device_name)
    probe = pilot._fresh_base(arm, device)
    _load_cost(
        project_root,
        output_dir,
        arm,
        seed=seed,
        device=device,
        config=probe.config,
    )
    rows = []
    for instance_seed in INSTANCE_SEEDS:
        instance = sources.instances[instance_seed]

        def wrapper_factory(base):
            cost = _load_cost(
                project_root,
                output_dir,
                arm,
                seed=seed,
                device=device,
                config=base.config,
            )
            return HandlingAugmentedV11Agent(base, cost, handling_lambda=value)

        raw = pilot._run_raw(
            arm, instance, device=device, wrapper_factory=wrapper_factory
        )
        row = pilot._compact_row(raw, instance)
        row.update({"model_seed": seed, "fixed_lambda": value})
        rows.append(row)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "model_seed": seed,
        "fixed_lambda": value,
        "row_count": len(rows),
        "summary": pilot._summary(rows),
        "rows": rows,
    }
    pilot._atomic_json(path, ledger)
    return ledger


def run_missing(
    project_root: Path, output_dir: Path, *, device_name: str
) -> dict:
    completed = []
    for seed in NEW_MODEL_SEEDS:
        for value in NEW_LAMBDAS:
            ledger = evaluate_seed_lambda(
                project_root,
                output_dir,
                seed=seed,
                value=value,
                device_name=device_name,
            )
            completed.append(
                {
                    "model_seed": seed,
                    "lambda": value,
                    "rows": len(ledger["rows"]),
                }
            )
    return {"status": "complete", "new_rows": EXPECTED_NEW_ROWS, "jobs": completed}


def _existing_rows(project_root: Path, seed: int, value: float, sources, parent) -> list[dict]:
    paths = _paths(project_root)
    if seed == 0:
        return list(parent["sweep"]["rows"][_key(value)])
    if value == 0.0:
        return stability._lambda_zero_rows(sources, seed)
    if value == 0.025:
        path = stability._ledger_path(paths["stability"], seed)
        ledger = json.loads(path.read_text())
        if ledger.get("fixed_lambda") != value:
            raise FrontierError("stability lambda ledger changed")
        return list(ledger["rows"])
    path = _ledger_path(output_dir=paths["stability"], seed=seed, value=value)
    raise FrontierError(f"unexpected existing row request: {path}")


def _all_rows(project_root: Path, output_dir: Path) -> dict:
    arms, sources, parent, report, paths = _authenticate(project_root)
    result = {seed: {} for seed in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        for value in LAMBDA_GRID:
            if seed == 0 or value in (0.0, 0.025):
                rows = _existing_rows(project_root, seed, value, sources, parent)
            else:
                path = _ledger_path(output_dir, seed, value)
                if not path.is_file():
                    raise FrontierError(f"missing evaluation ledger: {path}")
                ledger = json.loads(path.read_text())
                if ledger.get("model_seed") != seed or ledger.get("fixed_lambda") != value:
                    raise FrontierError("frontier ledger identity changed")
                rows = list(ledger["rows"])
            if len(rows) != len(INSTANCE_SEEDS):
                raise FrontierError("frontier row grid is incomplete")
            result[seed][value] = rows
    return result


def _aggregate(seed_summaries: Mapping[int, Mapping]) -> dict | None:
    if any(summary.get("metrics") is None for summary in seed_summaries.values()):
        return None
    metrics = [seed_summaries[seed]["metrics"] for seed in MODEL_SEEDS]
    total = sum(int(metric["total_physical_rehandles"]) for metric in metrics)
    return {
        "mean_dense_return": fmean(metric["mean_dense_return"] for metric in metrics),
        "mean_absolute_error": fmean(
            metric["mean_absolute_error"] for metric in metrics
        ),
        "mean_steps": fmean(metric["mean_steps"] for metric in metrics),
        "total_physical_rehandles": total,
        "physical_rehandles_per_100": 100.0
        * total
        / (len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * 8),
    }


def _nondominated(metrics: Mapping[float, Mapping]) -> list[float]:
    result = []
    for value, point in metrics.items():
        dominated = False
        for other_value, other in metrics.items():
            if other_value == value:
                continue
            no_worse = (
                other["mean_absolute_error"] <= point["mean_absolute_error"]
                and other["physical_rehandles_per_100"]
                <= point["physical_rehandles_per_100"]
            )
            strict = (
                other["mean_absolute_error"] < point["mean_absolute_error"]
                or other["physical_rehandles_per_100"]
                < point["physical_rehandles_per_100"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            result.append(value)
    return sorted(result)


def _monotonicity(metrics: Mapping[float, Mapping]) -> dict:
    inversions = []
    for lower, upper in zip(LAMBDA_GRID, LAMBDA_GRID[1:]):
        delta = (
            metrics[upper]["physical_rehandles_per_100"]
            - metrics[lower]["physical_rehandles_per_100"]
        )
        if delta > 1e-12:
            inversions.append({"from": lower, "to": upper, "increase": delta})
    return {
        "inversion_count": len(inversions),
        "total_inversion_magnitude": sum(item["increase"] for item in inversions),
        "inversions": inversions,
    }


def analyze(project_root: Path, output_dir: Path) -> dict:
    _require_contract(project_root, output_dir)
    rows = _all_rows(project_root, output_dir)
    all_safe = all(
        row.get("strict_safe_complete") is True
        for seed in MODEL_SEEDS
        for value in LAMBDA_GRID
        for row in rows[seed][value]
    )
    summaries = {
        seed: {value: pilot._summary(rows[seed][value]) for value in LAMBDA_GRID}
        for seed in MODEL_SEEDS
    }
    if not all_safe or any(
        summaries[seed][value].get("metrics") is None
        for seed in MODEL_SEEDS
        for value in LAMBDA_GRID
    ):
        report = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "failed_incomplete_or_unsafe",
            "whole_method_metrics_suppressed": True,
        }
        pilot._atomic_json(output_dir / REPORT_NAME, report)
        return report
    seed_metrics = {
        seed: {value: summaries[seed][value]["metrics"] for value in LAMBDA_GRID}
        for seed in MODEL_SEEDS
    }
    aggregate = {
        value: _aggregate({seed: summaries[seed][value] for seed in MODEL_SEEDS})
        for value in LAMBDA_GRID
    }
    aggregate_nd = _nondominated(aggregate)
    seed_nd = {seed: _nondominated(seed_metrics[seed]) for seed in MODEL_SEEDS}
    coverage = {
        value: sum(value in seed_nd[seed] for seed in MODEL_SEEDS)
        for value in LAMBDA_GRID
    }
    supported_nd = [value for value in aggregate_nd if coverage[value] >= 2]
    monotonicity = {
        "aggregate": _monotonicity(aggregate),
        "per_seed": {seed: _monotonicity(seed_metrics[seed]) for seed in MODEL_SEEDS},
    }
    distinct_points = len(
        {
            (
                round(aggregate[value]["mean_absolute_error"], 12),
                round(aggregate[value]["physical_rehandles_per_100"], 12),
            )
            for value in LAMBDA_GRID
        }
    )
    material_seed_inversion_count = sum(
        monotonicity["per_seed"][seed]["total_inversion_magnitude"] > 1.0
        for seed in MODEL_SEEDS
    )
    criteria = {
        "all_180_rows_strict_safe_complete": all_safe,
        "at_least_three_distinct_aggregate_nondominated_points": (
            len(aggregate_nd) >= 3 and distinct_points >= 3
        ),
        "at_least_three_nondominated_points_supported_in_two_of_three_seeds": (
            len(supported_nd) >= 3
        ),
        "aggregate_rehandles_nonincreasing": (
            monotonicity["aggregate"]["inversion_count"] == 0
        ),
        "material_rehandle_reversal_in_at_most_one_seed": (
            material_seed_inversion_count <= 1
        ),
    }
    advance = all(criteria.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": "development_frontier_diagnostic",
        "lambda_grid": list(LAMBDA_GRID),
        "row_count": EXPECTED_ROWS,
        "new_row_count": EXPECTED_NEW_ROWS,
        "seed_metrics": {
            str(seed): {_key(value): seed_metrics[seed][value] for value in LAMBDA_GRID}
            for seed in MODEL_SEEDS
        },
        "aggregate_metrics": {
            _key(value): aggregate[value] for value in LAMBDA_GRID
        },
        "aggregate_nondominated_lambdas": aggregate_nd,
        "per_seed_nondominated_lambdas": {
            str(seed): seed_nd[seed] for seed in MODEL_SEEDS
        },
        "frontier_support_count": {_key(value): coverage[value] for value in LAMBDA_GRID},
        "supported_aggregate_nondominated_lambdas": supported_nd,
        "handling_monotonicity": monotonicity,
        "sampled_frontier_criteria": criteria,
        "advance_all_five_levels_to_unseen_89k": advance,
        "interpretation": (
            "one frozen VCG 1.1 Qop plus one detached handling head per model "
            "seed, with lambda changed only in action merit; development only"
        ),
    }
    path = output_dir / REPORT_NAME
    if path.exists() and json.loads(path.read_text()) != report:
        raise FrontierError("existing frontier report changed")
    if not path.exists():
        pilot._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-missing", "analyze", "run-all"))
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "results/vcg-v1-1-nested-lambda-frontier-85k-development"
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(root, output)
    elif args.command == "run-missing":
        result = run_missing(root, output, device_name=args.device)
    elif args.command == "analyze":
        result = analyze(root, output)
    else:
        prepare(root, output)
        run_missing(root, output, device_name=args.device)
        result = analyze(root, output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
