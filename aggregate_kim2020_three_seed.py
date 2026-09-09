#!/usr/bin/env python3
"""Aggregate the frozen three-training-seed Kim paper comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

from compare_kim2020_a3c_spatial import (
    COMPARISON_VERSION,
    KIM_MAP,
    KIM_STOCHASTIC,
    PRIMARY_METRICS,
    json_safe,
    paired_contrast,
    summarize_method,
)
from PSLAP.track_a import TRACK_A_DYNAMIC, TRACK_A_NEAREST_FREE


AGGREGATION_VERSION = "kim2020_spatial_three_training_seed_aggregate_v1"
EXPECTED_TRAINING_SEEDS = (0, 1, 2)
EXPECTED_SCHEDULE_SEEDS = tuple(range(97_000, 97_050))
EXPECTED_POLICY_SEEDS = tuple(range(20_000_000, 20_000_005))
EXPECTED_EXECUTION_CONTRACT = (
    "retrieve_deliver_option_v3",
    "named_atomic_retrieval_executor_v3",
    "canonical_plan_plus_complete_current_live_multileg_v3",
)
METHODS = (KIM_STOCHASTIC, KIM_MAP, TRACK_A_DYNAMIC, TRACK_A_NEAREST_FREE)
BASELINES = (TRACK_A_DYNAMIC, TRACK_A_NEAREST_FREE)


def _load(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"comparison is not a dictionary: {path}")
    return payload


def _training_seed(payload):
    audit = payload.get("config", {}).get(
        "kim_checkpoint_provenance_audit", {}
    )
    if audit.get("verified") is not True:
        raise AssertionError("checkpoint provenance is not verified")
    return int(audit["training_seed"])


def _validate_payloads(items):
    by_seed = {}
    reference_config = None
    fixed_config_fields = (
        "comparison_version",
        "scheduler_variant",
        "source_neutral_scheduler",
        "assignment_commitment",
        "stochastic_aggregation",
        "lambda",
        "mu",
        "geometry",
        "schedule_seeds",
        "policy_seeds",
        "stochastic_rollouts",
        "max_steps",
        "max_defer_steps",
        "lookahead_margin_steps",
        "target_window",
    )
    digests = set()
    for path, payload in items:
        config = payload.get("config", {})
        seed = _training_seed(payload)
        if seed in by_seed:
            raise AssertionError(f"duplicate training seed {seed}")
        if config.get("comparison_version") != COMPARISON_VERSION:
            raise AssertionError(f"wrong comparison version in {path}")
        if tuple(config.get("schedule_seeds", ())) != EXPECTED_SCHEDULE_SEEDS:
            raise AssertionError(f"wrong paper schedule set in {path}")
        if tuple(config.get("policy_seeds", ())) != EXPECTED_POLICY_SEEDS:
            raise AssertionError(f"wrong policy-seed set in {path}")
        if config.get("seed_overlap_override") is not False:
            raise AssertionError(f"seed-overlap override used in {path}")
        seed_audit = config.get("seed_disjointness_audit", {}).get("kim2020", {})
        if seed_audit.get("disjoint") is not True:
            raise AssertionError(f"training/evaluation seed overlap in {path}")
        digest = config.get("kim_deployment_digest")
        if not digest or digest in digests:
            raise AssertionError("deployment digests must be present and unique")
        digests.add(digest)

        fixed = {field: config.get(field) for field in fixed_config_fields}
        if reference_config is None:
            reference_config = fixed
        elif fixed != reference_config:
            raise AssertionError(f"comparison configuration drift in {path}")
        if len(payload.get("raw_runs", ())) != 400:
            raise AssertionError(f"unexpected raw-run count in {path}")
        if len(payload.get("per_instance", ())) != 200:
            raise AssertionError(f"unexpected schedule-row count in {path}")
        for run in payload["raw_runs"]:
            audit = run.get("urgency_scheduler_audit", {})
            observed = (
                audit.get("retrieve_deliver_option_version"),
                audit.get("retrieval_executor_version"),
                audit.get("retrieval_start_contract"),
            )
            if observed != EXPECTED_EXECUTION_CONTRACT:
                raise AssertionError(
                    f"execution contract mismatch in {path}: {observed!r}"
                )
        by_seed[seed] = (path, payload)
    if tuple(sorted(by_seed)) != EXPECTED_TRAINING_SEEDS:
        raise AssertionError(
            f"expected training seeds {EXPECTED_TRAINING_SEEDS}, got {tuple(sorted(by_seed))}"
        )
    return by_seed


def _rows_by_method_and_schedule(payload):
    result = {}
    for row in payload["per_instance"]:
        key = (row["comparison_method"], int(row["schedule_seed"]))
        if key in result:
            raise AssertionError(f"duplicate schedule row: {key!r}")
        result[key] = row
    expected = {
        (method, schedule_seed)
        for method in METHODS
        for schedule_seed in EXPECTED_SCHEDULE_SEEDS
    }
    if set(result) != expected:
        raise AssertionError("per-instance method/schedule grid is incomplete")
    return result


def _assert_baseline_repeats_match(rows_by_seed):
    fields = (
        "instance_id",
        "schedule_id",
        "protocol_valid",
        "success",
        "strict_method_success",
        "reservation_integrity_rate",
        "invalid_assignment_count",
        "fallback_count",
        "illegal_drops",
        *PRIMARY_METRICS,
    )
    for method in BASELINES:
        for schedule_seed in EXPECTED_SCHEDULE_SEEDS:
            rows = [
                rows_by_seed[seed][(method, schedule_seed)]
                for seed in EXPECTED_TRAINING_SEEDS
            ]
            reference = {field: rows[0].get(field) for field in fields}
            if any(
                {field: row.get(field) for field in fields} != reference
                for row in rows[1:]
            ):
                raise AssertionError(
                    f"deterministic baseline drift: {method}, seed {schedule_seed}"
                )


def _combined_rows(rows_by_seed):
    rows = []
    for method in METHODS:
        for schedule_seed in EXPECTED_SCHEDULE_SEEDS:
            source_rows = [
                rows_by_seed[seed][(method, schedule_seed)]
                for seed in EXPECTED_TRAINING_SEEDS
            ]
            if method in BASELINES:
                row = dict(source_rows[0])
                row["training_seed_count"] = 1
                row["repeated_baseline_execution_count"] = 3
            else:
                row = {
                    "comparison_method": method,
                    "instance_id": source_rows[0]["instance_id"],
                    "schedule_id": source_rows[0]["schedule_id"],
                    "schedule_seed": schedule_seed,
                    "training_seed_count": 3,
                    "training_seeds": list(EXPECTED_TRAINING_SEEDS),
                    "rollout_count": sum(
                        int(item["rollout_count"]) for item in source_rows
                    ),
                    "protocol_valid": all(
                        item["protocol_valid"] for item in source_rows
                    ),
                    "success": mean(item["success"] for item in source_rows),
                    "strict_method_success": mean(
                        item["strict_method_success"] for item in source_rows
                    ),
                    "reservation_integrity_rate": mean(
                        item["reservation_integrity_rate"] for item in source_rows
                    ),
                    "invalid_assignment_count": sum(
                        int(item["invalid_assignment_count"])
                        for item in source_rows
                    ),
                    "fallback_count": sum(
                        int(item["fallback_count"]) for item in source_rows
                    ),
                    "illegal_drops": sum(
                        int(item["illegal_drops"]) for item in source_rows
                    ),
                }
                for metric in PRIMARY_METRICS:
                    row[metric] = (
                        mean(float(item[metric]) for item in source_rows)
                        if row["protocol_valid"]
                        else None
                    )
            rows.append(row)
    return rows


def _sample_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _model_variability(by_seed):
    result = {}
    for method in (KIM_STOCHASTIC, KIM_MAP):
        method_result = {}
        for metric in PRIMARY_METRICS:
            summary_key = metric if metric.startswith("mean_") else f"mean_{metric}"
            method_result[metric] = _sample_summary(
                by_seed[seed][1]["summaries"][method][summary_key]
                for seed in EXPECTED_TRAINING_SEEDS
            )
        result[method] = method_result
    return result


def _paired_model_seed_uncertainty(by_seed):
    # With n=3, this deliberately exposes the very wide model-seed t interval.
    t_critical_95_df2 = 4.302652729911275
    result = {}
    for comparator in (KIM_MAP, *BASELINES):
        metric_result = {}
        for metric in PRIMARY_METRICS:
            key = f"mean_delta_{metric}"
            values = [
                float(by_seed[seed][1]["paired_contrasts"][comparator][key])
                for seed in EXPECTED_TRAINING_SEEDS
            ]
            center = mean(values)
            spread = stdev(values)
            half_width = t_critical_95_df2 * spread / math.sqrt(len(values))
            metric_result[metric] = {
                **_sample_summary(values),
                "t95_interval": [center - half_width, center + half_width],
                "interval_unit": "independent_training_seed_mean_delta_df2",
            }
        result[comparator] = metric_result
    return result


def aggregate(items):
    by_seed = _validate_payloads(items)
    rows_by_seed = {
        seed: _rows_by_method_and_schedule(payload)
        for seed, (_, payload) in by_seed.items()
    }
    _assert_baseline_repeats_match(rows_by_seed)
    combined = _combined_rows(rows_by_seed)
    summaries = {
        method: summarize_method(
            [row for row in combined if row["comparison_method"] == method]
        )
        for method in METHODS
    }
    contrasts = {
        comparator: paired_contrast(combined, KIM_STOCHASTIC, comparator)
        for comparator in (KIM_MAP, *BASELINES)
    }
    return json_safe(
        {
            "aggregation_version": AGGREGATION_VERSION,
            "estimand": (
                "equal_weight_training_seed_then_policy_rollout_within_schedule_v1"
            ),
            "training_seeds": list(EXPECTED_TRAINING_SEEDS),
            "schedule_seeds": list(EXPECTED_SCHEDULE_SEEDS),
            "policy_seeds": list(EXPECTED_POLICY_SEEDS),
            "input_artifacts": {
                str(seed): str(by_seed[seed][0].resolve())
                for seed in EXPECTED_TRAINING_SEEDS
            },
            "checkpoint_digests": {
                str(seed): by_seed[seed][1]["config"]["kim_deployment_digest"]
                for seed in EXPECTED_TRAINING_SEEDS
            },
            "model_specific_summaries": {
                str(seed): by_seed[seed][1]["summaries"]
                for seed in EXPECTED_TRAINING_SEEDS
            },
            "model_specific_paired_contrasts": {
                str(seed): by_seed[seed][1]["paired_contrasts"]
                for seed in EXPECTED_TRAINING_SEEDS
            },
            "across_training_seed_summary": _model_variability(by_seed),
            "training_seed_paired_uncertainty": (
                _paired_model_seed_uncertainty(by_seed)
            ),
            "schedule_averaged_across_models": {
                "summaries": summaries,
                "paired_contrasts": contrasts,
                "rows": combined,
            },
            "protocol_notes": {
                "stochastic_rolls_averaged_before_schedule_comparison": True,
                "training_models_equal_weight": True,
                "deterministic_baseline_repetitions_verified_identical": True,
                "deterministic_baseline_repetitions_count_as_independent": False,
                "failed_model_schedule_requires_cost_censoring": True,
            },
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    items = [(path, _load(path)) for path in args.inputs]
    result = aggregate(items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "training_seed_count": 3,
                "schedule_count": 50,
                "all_primary_protocol_valid": result[
                    "schedule_averaged_across_models"
                ]["summaries"][KIM_STOCHASTIC]["all_protocol_valid"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

