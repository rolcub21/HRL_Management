#!/usr/bin/env python3
"""Aggregate the three fresh paired unified-VCG seed replicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence

import compare_unified_vcg as pair
import train_vcg_unified as unified
from train_vcg_unified_seed_stability import MODEL_SEEDS, SEED_PROFILES


FIELDS = (
    "dense_return",
    "mean_absolute_error",
    "physical_rehandles_per_100",
    "steps",
    "within_window_percentage",
    "mean_earliness",
    "mean_tardiness",
)


def _read_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_seed(root: Path, model_seed: int) -> dict:
    profile = SEED_PROFILES[model_seed]
    seed_root = root / f"seed-{model_seed}"
    unconstrained_dir = seed_root / unified.VCG
    constrained_dir = seed_root / unified.VCG_HANDLING_CONSTRAINT
    u_summary, u_rows = pair._load_arm(unconstrained_dir, unified.VCG)
    c_summary, c_rows = pair._load_arm(
        constrained_dir, unified.VCG_HANDLING_CONSTRAINT
    )
    for arm_root in (unconstrained_dir, constrained_dir):
        contract = _read_mapping(arm_root / "training-contract.json")
        shared = contract.get("shared_configuration", {})
        expected = {
            "model_seed": profile.model_seed,
            "train_seed_base": profile.train_seed_base,
            "replay_rng_seed": profile.replay_rng_seed,
            "training_policy_rng_formula": (
                f"{profile.behavior_rng_base} + episode_number - 1"
            ),
        }
        for field, value in expected.items():
            if shared.get(field) != value:
                raise ValueError(
                    f"seed {model_seed} {arm_root.name} has the wrong {field}"
                )
    if u_summary["shared_configuration_sha256"] != c_summary[
        "shared_configuration_sha256"
    ]:
        raise ValueError(f"seed {model_seed} arms are not paired")
    u_points = pair._instance_points(u_rows)
    c_points = pair._instance_points(c_rows)
    comparison = pair.compare(unconstrained_dir, constrained_dir)
    return {
        "model_seed": model_seed,
        "comparison": comparison,
        "unconstrained_points": u_points,
        "constrained_points": c_points,
        "constrained_final_lambda": c_summary["lambda_state"]["lambda_value"],
        "constrained_rehandle_ucb": c_summary["final_validation"][
            "physical_rehandle_rate_one_sided_95_ucb"
        ],
    }


def _equal_seed_instance_points(
    loaded: Sequence[Mapping], key: str
) -> dict[int, dict]:
    instance_seeds = sorted(loaded[0][key])
    if any(sorted(item[key]) != instance_seeds for item in loaded):
        raise ValueError("the seed replicates do not share the same EpisodeInstances")
    return {
        instance_seed: {
            field: fmean(float(item[key][instance_seed][field]) for item in loaded)
            for field in FIELDS
        }
        for instance_seed in instance_seeds
    }


def _arm_means(points: Mapping[int, Mapping]) -> dict:
    return {
        field: fmean(float(point[field]) for point in points.values())
        for field in FIELDS
    }


def analyze(root: Path) -> dict:
    loaded = [_load_seed(Path(root), model_seed) for model_seed in MODEL_SEEDS]
    unconstrained = _equal_seed_instance_points(loaded, "unconstrained_points")
    constrained = _equal_seed_instance_points(loaded, "constrained_points")
    aggregate_differences = {
        field: pair._paired_difference(constrained, unconstrained, field)
        for field in FIELDS
    }
    per_seed_rehandle_differences = {
        str(item["model_seed"]): item["comparison"]["paired_differences"][
            "physical_rehandles_per_100"
        ]["mean_difference"]
        for item in loaded
    }
    improving = sum(value < 0.0 for value in per_seed_rehandle_differences.values())
    all_safe_complete = all(
        item["comparison"].get("status") == "complete" for item in loaded
    )
    aggregate_improved = (
        aggregate_differences["physical_rehandles_per_100"]["mean_difference"] < 0.0
    )
    stability_passed = bool(all_safe_complete and aggregate_improved and improving >= 2)
    per_seed_values = list(per_seed_rehandle_differences.values())
    return {
        "status": "passed" if stability_passed else "did_not_pass",
        "method_version": unified.METHOD_VERSION,
        "scientific_role": "prospective_three_seed_development_stability_replication",
        "development_seed_14_excluded_from_primary_aggregate": True,
        "model_seeds": MODEL_SEEDS,
        "aggregation_order": (
            "four action RNGs within EpisodeInstance, then equal model seeds "
            "within EpisodeInstance, then 12 EpisodeInstances"
        ),
        "arms": {
            unified.VCG: _arm_means(unconstrained),
            unified.VCG_HANDLING_CONSTRAINT: _arm_means(constrained),
        },
        "paired_differences": aggregate_differences,
        "per_seed": {
            str(item["model_seed"]): {
                "arms": item["comparison"]["arms"],
                "paired_differences": item["comparison"]["paired_differences"],
                "constrained_final_lambda": item["constrained_final_lambda"],
                "constrained_rehandle_ucb": item["constrained_rehandle_ucb"],
            }
            for item in loaded
        },
        "stability_criterion": {
            "all_six_models_strict_safe_complete": all_safe_complete,
            "aggregate_rehandle_difference_below_zero": aggregate_improved,
            "seeds_with_lower_rehandles_under_constraint": improving,
            "at_least_two_of_three_seeds": improving >= 2,
            "strong_three_of_three": improving == 3,
            "passed": stability_passed,
        },
        "training_seed_rehandle_difference": {
            "values": per_seed_rehandle_differences,
            "mean": fmean(per_seed_values),
            "standard_deviation": stdev(per_seed_values),
            "minimum": min(per_seed_values),
            "maximum": max(per_seed_values),
        },
        "handling_budget_per_100": unified.HANDLING_BUDGET_PER_100,
        "development_only": True,
        "new_final_panel_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(args.root)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()

