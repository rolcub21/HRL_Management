#!/usr/bin/env python3
"""Seed-aware timing--relocation analysis for the VCG-Dense diagnostic.

This module deliberately contains no checkpoint loading, file I/O, or command
line handling.  A protocol runner is responsible for authenticating artifacts
and normalizing its episode outcomes to the row contract below.  The analysis
then fails closed unless it receives a complete crossed panel:

* one selected and one final diagnostic policy for each of three independent
  training seeds;
* one or more deterministic baselines, represented once (not repeated for
  each learned policy); and
* the identical ``(instance_id, schedule_id)`` panel for every method.

Episode instances are the resampling unit within a fixed set of learned
policies.  Training seeds are the replication unit for learned-method
uncertainty.  The two are never flattened into ``3 * N`` independent rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from numbers import Integral, Real
from statistics import fmean, stdev
from typing import Callable, Mapping, Optional, Sequence

import numpy as np


PROTOCOL = "vcg_dense_timing_relocation_pareto_analysis_v1"
SELECTED_BEST = "selected_best"
FINAL_DIAGNOSTIC = "episode500_final_diagnostic"
BASELINE = "baseline"
POLICY_GROUPS = (SELECTED_BEST, FINAL_DIAGNOSTIC, BASELINE)
LEARNED_GROUPS = (SELECTED_BEST, FINAL_DIAGNOSTIC)

DEFAULT_RNG_SEED = 20_260_808
T95_DF2 = 4.302652729911275
_COMPARISON_TOLERANCE = 1e-12

REQUIRED_ROW_FIELDS = (
    "method_id",
    "policy_group",
    "model_seed",
    "instance_id",
    "schedule_id",
    "strict_method_success",
    "completion_rate",
    "delivery_count",
    "mean_absolute_error",
    "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "steps",
    "dense_rescored_return",
    "legacy_rescored_return",
    "relocations",
)

_FINITE_ROW_FIELDS = (
    "strict_method_success",
    "completion_rate",
    "mean_absolute_error",
    "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "dense_rescored_return",
    "legacy_rescored_return",
)

_NONNEGATIVE_ROW_FIELDS = (
    "mean_absolute_error",
    "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
)

# Point metric -> normalized per-episode row field.  Relocations are a ratio of
# sums and are handled separately.
_POINT_MEAN_FIELDS = {
    "mean_absolute_error": "mean_absolute_error",
    "first_two_mean_absolute_error": "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error": (
        "positions_three_plus_mean_absolute_error"
    ),
    "mean_tardiness": "mean_tardiness",
    "mean_earliness": "mean_earliness",
    "within_target_window_rate": "within_target_window_rate",
    "mean_steps": "steps",
    "mean_dense_rescored_return": "dense_rescored_return",
    "mean_legacy_rescored_return": "legacy_rescored_return",
}
RELOCATION_METRIC = "relocations_per_100_deliveries"
POINT_METRICS = (*_POINT_MEAN_FIELDS, RELOCATION_METRIC)

PRIMARY_METRICS = ("mean_absolute_error", RELOCATION_METRIC)
GUARDRAIL_METRICS = (
    "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
)
UNCERTAINTY_METRICS = (*PRIMARY_METRICS, *GUARDRAIL_METRICS)


class ParetoAnalysisError(ValueError):
    """Raised when normalized analysis rows violate the protocol contract."""


@dataclass(frozen=True)
class _Panel:
    rows: tuple[dict, ...]
    pairs: tuple[tuple[str, str], ...]
    by_method: Mapping[str, Mapping[tuple[str, str], dict]]
    arrays: Mapping[str, Mapping[str, np.ndarray]]
    method_meta: Mapping[str, tuple[str, Optional[int]]]
    learned_methods: Mapping[tuple[str, int], str]
    training_seeds: tuple[int, ...]
    all_baseline_methods: tuple[str, ...]
    baseline_methods: tuple[str, ...]
    method_safety: Mapping[str, Mapping]


def _require_nonempty_string(value, *, field: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be a non-empty string"
        )
    return value


def _finite_float(value, *, field: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be finite"
        )
    return result


def _optional_float(value, *, field: str, row_index: int) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        if field == "strict_method_success":
            return float(value)
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be numeric or None"
        )
    if not isinstance(value, Real):
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be numeric or None"
        )
    return float(value)


def _positive_integer(value, *, field: str, row_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be an integer"
        )
    result = int(value)
    if result <= 0:
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be positive"
        )
    return result


def _nonnegative_integer(value, *, field: str, row_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be an integer"
        )
    result = int(value)
    if result < 0:
        raise ParetoAnalysisError(
            f"row {row_index} field {field!r} must be non-negative"
        )
    return result


def _normalize_row(row: Mapping, row_index: int) -> dict:
    if not isinstance(row, Mapping):
        raise ParetoAnalysisError(f"row {row_index} must be a mapping")
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ParetoAnalysisError(
            f"row {row_index} is missing required fields: {missing}"
        )

    normalized = dict(row)
    normalized["method_id"] = _require_nonempty_string(
        row["method_id"], field="method_id", row_index=row_index
    )
    normalized["policy_group"] = _require_nonempty_string(
        row["policy_group"], field="policy_group", row_index=row_index
    )
    if normalized["policy_group"] not in POLICY_GROUPS:
        raise ParetoAnalysisError(
            f"row {row_index} has unknown policy_group "
            f"{normalized['policy_group']!r}"
        )
    normalized["instance_id"] = _require_nonempty_string(
        row["instance_id"], field="instance_id", row_index=row_index
    )
    normalized["schedule_id"] = _require_nonempty_string(
        row["schedule_id"], field="schedule_id", row_index=row_index
    )

    if normalized["policy_group"] in LEARNED_GROUPS:
        model_seed = row["model_seed"]
        if isinstance(model_seed, bool) or not isinstance(model_seed, Integral):
            raise ParetoAnalysisError(
                f"row {row_index} learned model_seed must be an integer"
            )
        normalized["model_seed"] = int(model_seed)
    elif row["model_seed"] is not None:
        raise ParetoAnalysisError(
            f"row {row_index} baseline model_seed must be None"
        )

    for field in _FINITE_ROW_FIELDS:
        normalized[field] = _optional_float(
            row[field], field=field, row_index=row_index
        )
    normalized["delivery_count"] = _nonnegative_integer(
        row["delivery_count"], field="delivery_count", row_index=row_index
    )
    normalized["steps"] = _nonnegative_integer(
        row["steps"], field="steps", row_index=row_index
    )
    normalized["relocations"] = _nonnegative_integer(
        row["relocations"], field="relocations", row_index=row_index
    )

    return normalized


def _row_safety_issues(row: Mapping) -> list[str]:
    issues = []
    for field in _FINITE_ROW_FIELDS:
        value = row[field]
        if value is None or not math.isfinite(float(value)):
            issues.append(f"missing_or_nonfinite:{field}")
    strict = row["strict_method_success"]
    if strict is None or not math.isfinite(float(strict)) or float(strict) != 1.0:
        issues.append("strict_method_failure")
    completion = row["completion_rate"]
    if (
        completion is None
        or not math.isfinite(float(completion))
        or float(completion) != 1.0
    ):
        issues.append("incomplete_episode")
    if int(row["delivery_count"]) <= 0:
        issues.append("no_completed_deliveries")
    if int(row["steps"]) <= 0:
        issues.append("nonpositive_steps")
    for field in _NONNEGATIVE_ROW_FIELDS:
        value = row[field]
        if value is not None and math.isfinite(float(value)) and float(value) < 0.0:
            issues.append(f"negative:{field}")
    window = row["within_target_window_rate"]
    if window is not None and math.isfinite(float(window)) and not 0.0 <= float(
        window
    ) <= 1.0:
        issues.append("outside_unit_interval:within_target_window_rate")
    if row.get("method_failure_reason") is not None:
        issues.append("method_failure_reason_present")
    return issues


def _method_safety_ledger(
    method_id: str,
    meta: tuple[str, Optional[int]],
    method_rows: Mapping[tuple[str, str], Mapping],
) -> dict:
    failures = []
    strict_rows = 0
    complete_rows = 0
    finite_metric_rows = 0
    for pair, row in sorted(method_rows.items()):
        issues = _row_safety_issues(row)
        strict = row["strict_method_success"]
        completion = row["completion_rate"]
        if strict is not None and math.isfinite(float(strict)) and float(strict) == 1.0:
            strict_rows += 1
        if (
            completion is not None
            and math.isfinite(float(completion))
            and float(completion) == 1.0
        ):
            complete_rows += 1
        if all(
            row[field] is not None and math.isfinite(float(row[field]))
            for field in _FINITE_ROW_FIELDS
        ):
            finite_metric_rows += 1
        if issues:
            failures.append(
                {
                    "instance_id": pair[0],
                    "schedule_id": pair[1],
                    "strict_method_success": row["strict_method_success"],
                    "completion_rate": row["completion_rate"],
                    "delivery_count": row["delivery_count"],
                    "method_failure_reason": row.get("method_failure_reason"),
                    "issues": issues,
                }
            )
    reason_counts: dict[str, int] = {}
    for failure in failures:
        reason = failure["method_failure_reason"] or "unspecified"
        reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "method_id": method_id,
        "policy_group": meta[0],
        "model_seed": meta[1],
        "row_count": len(method_rows),
        "strict_successful_row_count": strict_rows,
        "complete_row_count": complete_rows,
        "finite_metric_row_count": finite_metric_rows,
        "failed_or_invalid_row_count": len(failures),
        "eligible_for_numeric_pareto": not failures,
        "whole_method_excluded": bool(failures),
        "failure_reason_counts": reason_counts,
        "failures": failures,
    }


def _prepare_panel(rows: Sequence[Mapping]) -> _Panel:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ParetoAnalysisError("rows must be a non-empty sequence of mappings")
    normalized = tuple(_normalize_row(row, index) for index, row in enumerate(rows))

    by_method: dict[str, dict[tuple[str, str], dict]] = {}
    method_meta: dict[str, tuple[str, Optional[int]]] = {}
    instance_to_schedule: dict[str, str] = {}
    schedule_to_instance: dict[str, str] = {}
    for index, row in enumerate(normalized):
        method_id = row["method_id"]
        meta = (row["policy_group"], row["model_seed"])
        previous_meta = method_meta.setdefault(method_id, meta)
        if previous_meta != meta:
            raise ParetoAnalysisError(
                f"method_id {method_id!r} maps to inconsistent policy metadata"
            )
        pair = (row["instance_id"], row["schedule_id"])
        method_rows = by_method.setdefault(method_id, {})
        if pair in method_rows:
            raise ParetoAnalysisError(
                f"duplicate method/instance pair for {method_id!r}: {pair!r}"
            )
        method_rows[pair] = row

        prior_schedule = instance_to_schedule.setdefault(pair[0], pair[1])
        if prior_schedule != pair[1]:
            raise ParetoAnalysisError(
                f"instance_id {pair[0]!r} maps to multiple schedules"
            )
        prior_instance = schedule_to_instance.setdefault(pair[1], pair[0])
        if prior_instance != pair[0]:
            raise ParetoAnalysisError(
                f"schedule_id {pair[1]!r} maps to multiple instances"
            )

    learned_methods: dict[tuple[str, int], str] = {}
    baseline_methods = []
    for method_id, (group, model_seed) in method_meta.items():
        if group in LEARNED_GROUPS:
            key = (group, int(model_seed))
            if key in learned_methods:
                raise ParetoAnalysisError(
                    f"multiple method_ids represent learned arm {key!r}"
                )
            learned_methods[key] = method_id
        else:
            baseline_methods.append(method_id)

    selected_seeds = {
        seed for group, seed in learned_methods if group == SELECTED_BEST
    }
    final_seeds = {
        seed for group, seed in learned_methods if group == FINAL_DIAGNOSTIC
    }
    if len(selected_seeds) != 3 or selected_seeds != final_seeds:
        raise ParetoAnalysisError(
            "selected and final groups must contain the same exactly three "
            f"training seeds; found selected={sorted(selected_seeds)}, "
            f"final={sorted(final_seeds)}"
        )
    if not baseline_methods:
        raise ParetoAnalysisError("at least one deterministic baseline is required")

    reference_method = sorted(by_method)[0]
    reference_grid = set(by_method[reference_method])
    if not reference_grid:
        raise ParetoAnalysisError("the paired instance grid is empty")
    for method_id, method_rows in by_method.items():
        observed = set(method_rows)
        if observed != reference_grid:
            missing = sorted(reference_grid - observed)
            extra = sorted(observed - reference_grid)
            raise ParetoAnalysisError(
                f"method {method_id!r} has an incomplete paired grid; "
                f"missing={missing}, extra={extra}"
            )

    pairs = tuple(sorted(reference_grid))
    method_safety = {
        method_id: _method_safety_ledger(
            method_id, method_meta[method_id], method_rows
        )
        for method_id, method_rows in by_method.items()
    }
    eligible_methods = {
        method_id
        for method_id, ledger in method_safety.items()
        if ledger["eligible_for_numeric_pareto"]
    }
    eligible_baselines = sorted(set(baseline_methods) & eligible_methods)

    # Eligible methods must have the same delivered workload.  Failed methods
    # remain in the full grid and safety ledger but are deliberately absent
    # from this comparison rather than being filtered to successful rows.
    for pair in pairs:
        delivery_counts = {
            by_method[method_id][pair]["delivery_count"]
            for method_id in eligible_methods
        }
        if delivery_counts and len(delivery_counts) != 1:
            raise ParetoAnalysisError(
                f"eligible methods on paired instance {pair!r} have "
                "inconsistent delivery counts: "
                f"{sorted(delivery_counts)}"
            )

    array_fields = set(_POINT_MEAN_FIELDS.values()) | {
        "delivery_count",
        "relocations",
    }
    arrays = {
        method_id: {
            field: np.asarray(
                [method_rows[pair][field] for pair in pairs], dtype=float
            )
            for field in array_fields
        }
        for method_id, method_rows in by_method.items()
        if method_id in eligible_methods
    }
    return _Panel(
        rows=normalized,
        pairs=pairs,
        by_method=by_method,
        arrays=arrays,
        method_meta=method_meta,
        learned_methods=learned_methods,
        training_seeds=tuple(sorted(selected_seeds)),
        all_baseline_methods=tuple(sorted(baseline_methods)),
        baseline_methods=tuple(eligible_baselines),
        method_safety=method_safety,
    )


def _validation_audit(panel: _Panel) -> dict:
    excluded = [
        method_id
        for method_id, ledger in panel.method_safety.items()
        if not ledger["eligible_for_numeric_pareto"]
    ]
    return {
        "row_count": len(panel.rows),
        "method_count": len(panel.by_method),
        "instance_count": len(panel.pairs),
        "training_seed_count": len(panel.training_seeds),
        "training_seeds": list(panel.training_seeds),
        "learned_method_ids": {
            group: {
                str(seed): panel.learned_methods[(group, seed)]
                for seed in panel.training_seeds
            }
            for group in LEARNED_GROUPS
        },
        "baseline_method_ids": list(panel.all_baseline_methods),
        "eligible_baseline_method_ids": list(panel.baseline_methods),
        "excluded_method_ids": sorted(excluded),
        "pairing_grid": [
            {"instance_id": instance_id, "schedule_id": schedule_id}
            for instance_id, schedule_id in panel.pairs
        ],
        "complete_pairing_grid": True,
        "all_rows_finite": not excluded,
        "safety_gate_passed": not any(
            panel.method_safety[method_id]["policy_group"] in LEARNED_GROUPS
            for method_id in excluded
        )
        and bool(panel.baseline_methods),
        "all_rows_strictly_successful_and_complete": not excluded,
        "delivery_counts_match_within_instance": True,
        "deterministic_baselines_counted_once": True,
    }


def _safety_gate(panel: _Panel) -> dict:
    excluded = sorted(
        method_id
        for method_id, ledger in panel.method_safety.items()
        if not ledger["eligible_for_numeric_pareto"]
    )
    ineligible_learned = [
        method_id
        for method_id in excluded
        if panel.method_safety[method_id]["policy_group"] in LEARNED_GROUPS
    ]
    excluded_baselines = [
        method_id
        for method_id in excluded
        if panel.method_safety[method_id]["policy_group"] == BASELINE
    ]
    can_analyze = not ineligible_learned and bool(panel.baseline_methods)
    return {
        "passed_for_remaining_eligible_methods": can_analyze,
        "policy": (
            "retain every paired row; if any row fails safety or finiteness, "
            "exclude that whole method from every numeric point, contrast, "
            "bootstrap, and Pareto frontier"
        ),
        "complete_case_filtering_used": False,
        "whole_method_exclusion_used": bool(excluded),
        "eligible_method_ids": sorted(set(panel.by_method) - set(excluded)),
        "excluded_method_ids": excluded,
        "excluded_baseline_method_ids": excluded_baselines,
        "ineligible_learned_method_ids": ineligible_learned,
        "eligible_baseline_method_ids": list(panel.baseline_methods),
        "method_ledgers": {
            method_id: panel.method_safety[method_id]
            for method_id in sorted(panel.method_safety)
        },
        "numeric_pareto_authorized": can_analyze,
    }


def validate_normalized_rows(rows: Sequence[Mapping]) -> dict:
    """Validate normalized rows and return a JSON-safe panel audit."""

    return _validation_audit(_prepare_panel(rows))


def _full_instance_indices(panel: _Panel) -> np.ndarray:
    return np.arange(len(panel.pairs), dtype=int)


def _method_metric(
    panel: _Panel, method_id: str, metric: str, instance_indices: np.ndarray
) -> float:
    arrays = panel.arrays[method_id]
    if metric == RELOCATION_METRIC:
        deliveries = float(arrays["delivery_count"][instance_indices].sum())
        if deliveries <= 0.0:  # Defensive; validation already excludes this.
            raise ParetoAnalysisError("bootstrap draw has no deliveries")
        relocations = float(arrays["relocations"][instance_indices].sum())
        return 100.0 * relocations / deliveries
    try:
        field = _POINT_MEAN_FIELDS[metric]
    except KeyError as exc:
        raise ParetoAnalysisError(f"unknown point metric {metric!r}") from exc
    return float(arrays[field][instance_indices].mean())


def _group_metric(
    panel: _Panel,
    group: str,
    metric: str,
    seed_draw: Sequence[int],
    instance_indices: np.ndarray,
) -> float:
    values = [
        _method_metric(
            panel,
            panel.learned_methods[(group, int(seed))],
            metric,
            instance_indices,
        )
        for seed in seed_draw
    ]
    return float(fmean(values))


def _metrics_for_group(
    panel: _Panel,
    group: str,
    metrics: Sequence[str],
    seed_draw: Sequence[int],
    instance_indices: np.ndarray,
) -> dict:
    return {
        metric: _group_metric(panel, group, metric, seed_draw, instance_indices)
        for metric in metrics
    }


def _method_point(panel: _Panel, method_id: str) -> dict:
    indices = _full_instance_indices(panel)
    arrays = panel.arrays[method_id]
    return {
        "episodes": len(panel.pairs),
        "strict_method_success_rate": 1.0,
        "completion_rate": 1.0,
        "pareto_safety_eligible": True,
        "total_deliveries": int(arrays["delivery_count"].sum()),
        "total_relocations": int(arrays["relocations"].sum()),
        **{
            metric: _method_metric(panel, method_id, metric, indices)
            for metric in POINT_METRICS
        },
    }


def _group_point(panel: _Panel, group: str) -> dict:
    indices = _full_instance_indices(panel)
    point = _metrics_for_group(
        panel, group, POINT_METRICS, panel.training_seeds, indices
    )
    return {
        "training_seeds": list(panel.training_seeds),
        "n_training_seeds": len(panel.training_seeds),
        "instances_per_training_seed": len(panel.pairs),
        "equal_training_seed_weight": True,
        "strict_method_success_rate": 1.0,
        "completion_rate": 1.0,
        "pareto_safety_eligible": True,
        "raw_total_deliveries_across_seed_panels": sum(
            int(panel.arrays[panel.learned_methods[(group, seed)]]["delivery_count"].sum())
            for seed in panel.training_seeds
        ),
        "raw_total_relocations_across_seed_panels": sum(
            int(panel.arrays[panel.learned_methods[(group, seed)]]["relocations"].sum())
            for seed in panel.training_seeds
        ),
        **point,
    }


def summarize_seed_values(values_by_seed: Mapping[int, float]) -> dict:
    """Summarize exactly three independent seed-level values with a df=2 CI."""

    if not isinstance(values_by_seed, Mapping) or len(values_by_seed) != 3:
        raise ParetoAnalysisError("seed summary requires exactly three values")
    ordered = sorted((int(seed), float(value)) for seed, value in values_by_seed.items())
    if len({seed for seed, _ in ordered}) != 3 or any(
        not math.isfinite(value) for _, value in ordered
    ):
        raise ParetoAnalysisError("seed values must have three finite unique seeds")
    values = [value for _, value in ordered]
    center = float(fmean(values))
    sample_sd = float(stdev(values))
    standard_error = sample_sd / math.sqrt(3.0)
    half_width = T95_DF2 * standard_error
    return {
        "n_training_seeds": 3,
        "values_by_model_seed": {str(seed): value for seed, value in ordered},
        "mean": center,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "t95_interval": [center - half_width, center + half_width],
        "degrees_of_freedom": 2,
        "t_critical": T95_DF2,
        "interval_unit": "independent_training_seed_panel_mean_df2",
    }


def _derived_rng_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return [float(low), float(high)]


def _bootstrap(
    panel: _Panel,
    estimator: Callable[[Sequence[int], np.ndarray], Mapping[str, float]],
    original: Mapping[str, float],
    *,
    samples: int,
    rng_seed: int,
    label: str,
    resample_training_seeds: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    stream_seed = _derived_rng_seed(rng_seed, label)
    rng = np.random.default_rng(stream_seed)
    metrics = tuple(original)
    draws = {metric: np.empty(samples, dtype=float) for metric in metrics}
    seeds = panel.training_seeds
    n_instances = len(panel.pairs)
    for index in range(samples):
        instance_draw = rng.integers(0, n_instances, size=n_instances)
        if resample_training_seeds:
            seed_positions = rng.integers(0, len(seeds), size=len(seeds))
            seed_draw = tuple(seeds[position] for position in seed_positions)
        else:
            seed_draw = seeds
        estimate = estimator(seed_draw, instance_draw)
        if tuple(estimate) != metrics:
            raise RuntimeError("bootstrap estimator changed its metric schema")
        for metric in metrics:
            value = float(estimate[metric])
            if not math.isfinite(value):
                raise ParetoAnalysisError(
                    f"bootstrap produced non-finite {metric!r}"
                )
            draws[metric][index] = value

    resampling = (
        "independent_training_seed_and_paired_instance_crossed_bootstrap"
        if resample_training_seeds
        else "paired_instance_bootstrap_conditional_on_three_trained_policies"
    )
    report = {
        "samples": samples,
        "base_rng_seed": int(rng_seed),
        "stream_rng_seed": stream_seed,
        "resampling": resampling,
        "metrics": {
            metric: {
                "estimate": float(original[metric]),
                "bootstrap_mean": float(draws[metric].mean()),
                "percentile_95_interval": _percentile_interval(draws[metric]),
            }
            for metric in metrics
        },
    }
    return report, draws


def _comparison_sign(value: float) -> int:
    if value > _COMPARISON_TOLERANCE:
        return 1
    if value < -_COMPARISON_TOLERANCE:
        return -1
    return 0


def _relation_from_gains(
    first_gain: float,
    second_gain: float,
    *,
    left_label: str,
    right_label: str,
) -> str:
    first = _comparison_sign(first_gain)
    second = _comparison_sign(second_gain)
    if first >= 0 and second >= 0 and (first > 0 or second > 0):
        return f"{left_label}_dominates_{right_label}"
    if first <= 0 and second <= 0 and (first < 0 or second < 0):
        return f"{right_label}_dominates_{left_label}"
    if first == 0 and second == 0:
        return "tie"
    return "tradeoff"


def pareto_relation(left: Mapping[str, float], right: Mapping[str, float]) -> str:
    """Return the point relation for two lower-is-better MAE/relocation points."""

    for metric in PRIMARY_METRICS:
        if metric not in left or metric not in right:
            raise ParetoAnalysisError(f"Pareto point lacks {metric!r}")
        if not math.isfinite(float(left[metric])) or not math.isfinite(
            float(right[metric])
        ):
            raise ParetoAnalysisError("Pareto coordinates must be finite")
    return _relation_from_gains(
        float(right["mean_absolute_error"]) - float(left["mean_absolute_error"]),
        float(right[RELOCATION_METRIC]) - float(left[RELOCATION_METRIC]),
        left_label="left",
        right_label="right",
    )


def _capacity_estimate(
    panel: _Panel, seed_draw: Sequence[int], instance_indices: np.ndarray
) -> dict:
    selected = _metrics_for_group(
        panel, SELECTED_BEST, POINT_METRICS, seed_draw, instance_indices
    )
    final = _metrics_for_group(
        panel, FINAL_DIAGNOSTIC, POINT_METRICS, seed_draw, instance_indices
    )
    return {
        "mae_cost": final["mean_absolute_error"] - selected["mean_absolute_error"],
        "relocation_saving_per_100_deliveries": (
            selected[RELOCATION_METRIC] - final[RELOCATION_METRIC]
        ),
        "first_two_mae_cost": (
            final["first_two_mean_absolute_error"]
            - selected["first_two_mean_absolute_error"]
        ),
        "positions_three_plus_mae_cost": (
            final["positions_three_plus_mean_absolute_error"]
            - selected["positions_three_plus_mean_absolute_error"]
        ),
        "tardiness_cost": final["mean_tardiness"] - selected["mean_tardiness"],
        "earliness_cost": final["mean_earliness"] - selected["mean_earliness"],
        "within_window_change": (
            final["within_target_window_rate"]
            - selected["within_target_window_rate"]
        ),
        "step_cost": final["mean_steps"] - selected["mean_steps"],
        "dense_return_change": (
            final["mean_dense_rescored_return"]
            - selected["mean_dense_rescored_return"]
        ),
        "legacy_return_change": (
            final["mean_legacy_rescored_return"]
            - selected["mean_legacy_rescored_return"]
        ),
    }


def _capacity_relation(estimate: Mapping[str, float]) -> str:
    return _relation_from_gains(
        -float(estimate["mae_cost"]),
        float(estimate["relocation_saving_per_100_deliveries"]),
        left_label="final",
        right_label="selected",
    )


def _joint_relation_fractions(
    first: np.ndarray,
    second: np.ndarray,
    relation: Callable[[float, float], str],
) -> dict:
    counts: dict[str, int] = {}
    for left, right in zip(first, second):
        category = relation(float(left), float(right))
        counts[category] = counts.get(category, 0) + 1
    total = len(first)
    return {key: count / total for key, count in sorted(counts.items())}


_QUADRANTS = (
    "timing_better_and_relocation_better",
    "timing_better_and_relocation_worse",
    "timing_worse_and_relocation_better",
    "timing_worse_and_relocation_worse",
    "on_axis_or_tie",
)


def _quadrant_name(timing_gain: float, relocation_gain: float) -> str:
    timing = _comparison_sign(timing_gain)
    relocation = _comparison_sign(relocation_gain)
    if timing > 0 and relocation > 0:
        return "timing_better_and_relocation_better"
    if timing > 0 and relocation < 0:
        return "timing_better_and_relocation_worse"
    if timing < 0 and relocation > 0:
        return "timing_worse_and_relocation_better"
    if timing < 0 and relocation < 0:
        return "timing_worse_and_relocation_worse"
    return "on_axis_or_tie"


def _quadrant_summary(gains: Sequence[tuple[float, float]]) -> dict:
    counts = {name: 0 for name in _QUADRANTS}
    for timing_gain, relocation_gain in gains:
        counts[_quadrant_name(timing_gain, relocation_gain)] += 1
    total = len(gains)
    if total == 0:
        raise ParetoAnalysisError("cannot summarize an empty paired quadrant panel")
    return {
        "n_paired_instances": total,
        "counts": counts,
        "fractions": {name: counts[name] / total for name in _QUADRANTS},
    }


def _capacity_instance_quadrants(panel: _Panel) -> dict:
    by_seed = {}
    for seed in panel.training_seeds:
        selected_id = panel.learned_methods[(SELECTED_BEST, seed)]
        final_id = panel.learned_methods[(FINAL_DIAGNOSTIC, seed)]
        gains = []
        for instance_index in range(len(panel.pairs)):
            drawn = np.asarray([instance_index], dtype=int)
            timing_gain = _method_metric(
                panel, selected_id, "mean_absolute_error", drawn
            ) - _method_metric(panel, final_id, "mean_absolute_error", drawn)
            relocation_gain = _method_metric(
                panel, selected_id, RELOCATION_METRIC, drawn
            ) - _method_metric(panel, final_id, RELOCATION_METRIC, drawn)
            gains.append((timing_gain, relocation_gain))
        by_seed[str(seed)] = _quadrant_summary(gains)
    return {
        "gain_orientation": (
            "positive timing and relocation gains favor episode500 final"
        ),
        "per_model_seed": by_seed,
        "equal_seed_mean_fractions": {
            name: float(
                fmean(item["fractions"][name] for item in by_seed.values())
            )
            for name in _QUADRANTS
        },
    }


def _baseline_instance_quadrants(
    panel: _Panel, group: str, baseline_method: str
) -> dict:
    by_seed = {}
    for seed in panel.training_seeds:
        learned_id = panel.learned_methods[(group, seed)]
        gains = []
        for instance_index in range(len(panel.pairs)):
            drawn = np.asarray([instance_index], dtype=int)
            timing_gain = _method_metric(
                panel, baseline_method, "mean_absolute_error", drawn
            ) - _method_metric(panel, learned_id, "mean_absolute_error", drawn)
            relocation_gain = _method_metric(
                panel, baseline_method, RELOCATION_METRIC, drawn
            ) - _method_metric(panel, learned_id, RELOCATION_METRIC, drawn)
            gains.append((timing_gain, relocation_gain))
        by_seed[str(seed)] = _quadrant_summary(gains)
    return {
        "gain_orientation": "positive timing and relocation gains favor learned",
        "per_model_seed": by_seed,
        "equal_seed_mean_fractions": {
            name: float(
                fmean(item["fractions"][name] for item in by_seed.values())
            )
            for name in _QUADRANTS
        },
    }


def _capacity_analysis(
    panel: _Panel,
    *,
    mae_noninferiority_margin: float,
    bootstrap_samples: int,
    rng_seed: int,
) -> dict:
    indices = _full_instance_indices(panel)
    original = _capacity_estimate(panel, panel.training_seeds, indices)
    per_seed = {
        seed: _capacity_estimate(panel, (seed,), indices)
        for seed in panel.training_seeds
    }
    seed_t = {
        metric: summarize_seed_values(
            {seed: values[metric] for seed, values in per_seed.items()}
        )
        for metric in original
    }

    estimator = lambda seeds, drawn: _capacity_estimate(panel, seeds, drawn)
    conditional, conditional_draws = _bootstrap(
        panel,
        estimator,
        original,
        samples=bootstrap_samples,
        rng_seed=rng_seed,
        label="best_vs_final:conditional_instances",
        resample_training_seeds=False,
    )
    crossed, crossed_draws = _bootstrap(
        panel,
        estimator,
        original,
        samples=bootstrap_samples,
        rng_seed=rng_seed,
        label="best_vs_final:crossed",
        resample_training_seeds=True,
    )
    for report, draws in (
        (conditional, conditional_draws),
        (crossed, crossed_draws),
    ):
        report["joint_pareto_relation_fractions"] = _joint_relation_fractions(
            draws["mae_cost"],
            draws["relocation_saving_per_100_deliveries"],
            lambda mae_cost, saving: _capacity_relation(
                {
                    "mae_cost": mae_cost,
                    "relocation_saving_per_100_deliveries": saving,
                }
            ),
        )
        report["capacity_margin_fraction"] = float(
            np.mean(
                (draws["mae_cost"] <= mae_noninferiority_margin)
                & (draws["relocation_saving_per_100_deliveries"] > 0.0)
            )
        )

    per_seed_relations = {
        str(seed): _capacity_relation(values) for seed, values in per_seed.items()
    }
    per_seed_margin = {
        str(seed): bool(
            values["mae_cost"] <= mae_noninferiority_margin
            and values["relocation_saving_per_100_deliveries"] > 0.0
        )
        for seed, values in per_seed.items()
    }
    return {
        "comparison": "episode500_final_diagnostic_minus_selected_best",
        "diagnostic_only": True,
        "deployment_claim_eligible": False,
        "metric_orientation": {
            "mae_cost": "positive_means_final_has_higher_mae",
            "relocation_saving_per_100_deliveries": (
                "positive_means_final_uses_fewer_relocations"
            ),
            "other_costs": "positive_means_final_is_worse",
            "within_window_change": "positive_means_final_is_better",
            "return_change": "positive_means_final_is_better",
        },
        "point_estimate": original,
        "point_pareto_relation": _capacity_relation(original),
        "per_model_seed": {str(seed): values for seed, values in per_seed.items()},
        "per_model_seed_pareto_relation": per_seed_relations,
        "paired_instance_quadrants": _capacity_instance_quadrants(panel),
        "seed_level_t95": seed_t,
        "conditional_paired_instance_bootstrap": conditional,
        "crossed_bootstrap_sensitivity": crossed,
        "mae_noninferiority": {
            "margin": mae_noninferiority_margin,
            "point_estimate_pass": bool(
                original["mae_cost"] <= mae_noninferiority_margin
            ),
            "seed_t95_upper_bound_pass": bool(
                seed_t["mae_cost"]["t95_interval"][1]
                <= mae_noninferiority_margin
            ),
            "per_seed_pass": {
                str(seed): bool(
                    values["mae_cost"] <= mae_noninferiority_margin
                )
                for seed, values in per_seed.items()
            },
        },
        "capacity_signal": {
            "definition": (
                "positive relocation saving with MAE cost no larger than the "
                "predeclared noninferiority margin"
            ),
            "point_estimate_supports_signal": bool(
                original["relocation_saving_per_100_deliveries"] > 0.0
                and original["mae_cost"] <= mae_noninferiority_margin
            ),
            "per_seed_supports_signal": per_seed_margin,
            "supporting_seed_count": sum(per_seed_margin.values()),
        },
    }


def _baseline_contrast_estimate(
    panel: _Panel,
    group: str,
    baseline_method: str,
    seed_draw: Sequence[int],
    instance_indices: np.ndarray,
) -> dict:
    learned = _metrics_for_group(
        panel, group, POINT_METRICS, seed_draw, instance_indices
    )
    baseline = {
        metric: _method_metric(panel, baseline_method, metric, instance_indices)
        for metric in POINT_METRICS
    }
    return {
        "mae_reduction": baseline["mean_absolute_error"]
        - learned["mean_absolute_error"],
        "relocation_reduction_per_100_deliveries": (
            baseline[RELOCATION_METRIC] - learned[RELOCATION_METRIC]
        ),
        "first_two_mae_reduction": (
            baseline["first_two_mean_absolute_error"]
            - learned["first_two_mean_absolute_error"]
        ),
        "positions_three_plus_mae_reduction": (
            baseline["positions_three_plus_mean_absolute_error"]
            - learned["positions_three_plus_mean_absolute_error"]
        ),
        "tardiness_reduction": baseline["mean_tardiness"]
        - learned["mean_tardiness"],
        "earliness_reduction": baseline["mean_earliness"]
        - learned["mean_earliness"],
        "within_window_advantage": learned["within_target_window_rate"]
        - baseline["within_target_window_rate"],
        "step_reduction": baseline["mean_steps"] - learned["mean_steps"],
        "dense_return_advantage": learned["mean_dense_rescored_return"]
        - baseline["mean_dense_rescored_return"],
        "legacy_return_advantage": learned["mean_legacy_rescored_return"]
        - baseline["mean_legacy_rescored_return"],
    }


def _baseline_relation(estimate: Mapping[str, float]) -> str:
    return _relation_from_gains(
        float(estimate["mae_reduction"]),
        float(estimate["relocation_reduction_per_100_deliveries"]),
        left_label="learned",
        right_label="baseline",
    )


def _baseline_contrasts(
    panel: _Panel,
    *,
    bootstrap_samples: int,
    rng_seed: int,
) -> list[dict]:
    indices = _full_instance_indices(panel)
    results = []
    for group in LEARNED_GROUPS:
        for baseline_method in panel.baseline_methods:
            original = _baseline_contrast_estimate(
                panel,
                group,
                baseline_method,
                panel.training_seeds,
                indices,
            )
            per_seed = {
                seed: _baseline_contrast_estimate(
                    panel, group, baseline_method, (seed,), indices
                )
                for seed in panel.training_seeds
            }
            seed_t = {
                metric: summarize_seed_values(
                    {seed: values[metric] for seed, values in per_seed.items()}
                )
                for metric in original
            }

            def estimator(seed_draw, drawn):
                return _baseline_contrast_estimate(
                    panel, group, baseline_method, seed_draw, drawn
                )

            conditional, conditional_draws = _bootstrap(
                panel,
                estimator,
                original,
                samples=bootstrap_samples,
                rng_seed=rng_seed,
                label=f"{group}:{baseline_method}:conditional_instances",
                resample_training_seeds=False,
            )
            crossed, crossed_draws = _bootstrap(
                panel,
                estimator,
                original,
                samples=bootstrap_samples,
                rng_seed=rng_seed,
                label=f"{group}:{baseline_method}:crossed",
                resample_training_seeds=True,
            )
            for report, draws in (
                (conditional, conditional_draws),
                (crossed, crossed_draws),
            ):
                report["joint_pareto_relation_fractions"] = (
                    _joint_relation_fractions(
                        draws["mae_reduction"],
                        draws["relocation_reduction_per_100_deliveries"],
                        lambda mae_gain, relocation_gain: _relation_from_gains(
                            mae_gain,
                            relocation_gain,
                            left_label="learned",
                            right_label="baseline",
                        ),
                    )
                )

            seed_relations = {
                str(seed): _baseline_relation(values)
                for seed, values in per_seed.items()
            }
            results.append(
                {
                    "policy_group": group,
                    "baseline_method_id": baseline_method,
                    "diagnostic_only": group == FINAL_DIAGNOSTIC,
                    "deployment_claim_eligible": group == SELECTED_BEST,
                    "metric_orientation": "positive_favors_learned_policy_group",
                    "point_estimate": original,
                    "point_pareto_relation": _baseline_relation(original),
                    "per_model_seed": {
                        str(seed): values for seed, values in per_seed.items()
                    },
                    "per_model_seed_pareto_relation": seed_relations,
                    "paired_instance_quadrants": _baseline_instance_quadrants(
                        panel, group, baseline_method
                    ),
                    "seed_joint_dominance_count": sum(
                        relation == "learned_dominates_baseline"
                        for relation in seed_relations.values()
                    ),
                    "seed_level_t95": seed_t,
                    "conditional_paired_instance_bootstrap": conditional,
                    "crossed_bootstrap_sensitivity": crossed,
                }
            )
    return results


def _point_uncertainty(
    panel: _Panel,
    group: str,
    point: Mapping[str, float],
    *,
    bootstrap_samples: int,
    rng_seed: int,
) -> dict:
    indices = _full_instance_indices(panel)
    original = {metric: float(point[metric]) for metric in UNCERTAINTY_METRICS}
    per_seed_points = {
        seed: {
            metric: _method_metric(
                panel,
                panel.learned_methods[(group, seed)],
                metric,
                indices,
            )
            for metric in POINT_METRICS
        }
        for seed in panel.training_seeds
    }
    seed_t = {
        metric: summarize_seed_values(
            {seed: values[metric] for seed, values in per_seed_points.items()}
        )
        for metric in POINT_METRICS
    }

    def estimator(seed_draw, drawn):
        return _metrics_for_group(
            panel, group, UNCERTAINTY_METRICS, seed_draw, drawn
        )

    conditional, _ = _bootstrap(
        panel,
        estimator,
        original,
        samples=bootstrap_samples,
        rng_seed=rng_seed,
        label=f"point:{group}:conditional_instances",
        resample_training_seeds=False,
    )
    crossed, _ = _bootstrap(
        panel,
        estimator,
        original,
        samples=bootstrap_samples,
        rng_seed=rng_seed,
        label=f"point:{group}:crossed",
        resample_training_seeds=True,
    )
    return {
        "per_model_seed": {
            str(seed): values for seed, values in per_seed_points.items()
        },
        "seed_level_t95": seed_t,
        "conditional_paired_instance_bootstrap": conditional,
        "crossed_bootstrap_sensitivity": crossed,
    }


def _baseline_point_uncertainty(
    panel: _Panel,
    method_id: str,
    point: Mapping[str, float],
    *,
    bootstrap_samples: int,
    rng_seed: int,
) -> dict:
    original = {metric: float(point[metric]) for metric in UNCERTAINTY_METRICS}

    def estimator(_seed_draw, drawn):
        return {
            metric: _method_metric(panel, method_id, metric, drawn)
            for metric in UNCERTAINTY_METRICS
        }

    conditional, _ = _bootstrap(
        panel,
        estimator,
        original,
        samples=bootstrap_samples,
        rng_seed=rng_seed,
        label=f"point:{method_id}:conditional_instances",
        resample_training_seeds=False,
    )
    return {
        "conditional_paired_instance_bootstrap": conditional,
        "training_seed_interval": None,
        "training_seed_interval_reason": (
            "deterministic_baseline_has_no_training_seed_replication"
        ),
    }


def _pareto_report(
    group_points: Mapping[str, Mapping],
    baseline_points: Mapping[str, Mapping],
) -> dict:
    participants = {
        SELECTED_BEST: {
            **group_points[SELECTED_BEST]["point"],
            "policy_group": SELECTED_BEST,
            "diagnostic_only": False,
        },
        FINAL_DIAGNOSTIC: {
            **group_points[FINAL_DIAGNOSTIC]["point"],
            "policy_group": FINAL_DIAGNOSTIC,
            "diagnostic_only": True,
        },
        **{
            method_id: {
                **record["point"],
                "policy_group": BASELINE,
                "diagnostic_only": False,
            }
            for method_id, record in baseline_points.items()
        },
    }
    pairwise = []
    ids = tuple(participants)
    for left_id, right_id in itertools.combinations(ids, 2):
        relation = pareto_relation(participants[left_id], participants[right_id])
        pairwise.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "relation": relation,
                "left_minus_right": {
                    "mean_absolute_error": (
                        participants[left_id]["mean_absolute_error"]
                        - participants[right_id]["mean_absolute_error"]
                    ),
                    RELOCATION_METRIC: (
                        participants[left_id][RELOCATION_METRIC]
                        - participants[right_id][RELOCATION_METRIC]
                    ),
                },
            }
        )

    def frontier(candidate_ids: Sequence[str]) -> list[str]:
        output = []
        for candidate in candidate_ids:
            dominated = False
            for comparator in candidate_ids:
                if candidate == comparator:
                    continue
                relation = pareto_relation(
                    participants[comparator], participants[candidate]
                )
                if relation == "left_dominates_right":
                    dominated = True
                    break
            if not dominated:
                output.append(candidate)
        return output

    deployment_ids = tuple(item for item in ids if item != FINAL_DIAGNOSTIC)
    return {
        "axes": {
            "x": "mean_absolute_error_lower_is_better",
            "y": "relocations_per_100_deliveries_lower_is_better",
        },
        "safety_is_lexicographic_prerequisite": True,
        "participants": {
            point_id: {
                "policy_group": point["policy_group"],
                "diagnostic_only": point["diagnostic_only"],
                "mean_absolute_error": point["mean_absolute_error"],
                RELOCATION_METRIC: point[RELOCATION_METRIC],
            }
            for point_id, point in participants.items()
        },
        "pairwise_point_relations": pairwise,
        "all_points_frontier_including_diagnostic_final": frontier(ids),
        "deployment_frontier_excluding_diagnostic_final": frontier(deployment_ids),
    }


def build_pareto_report(
    rows: Sequence[Mapping],
    *,
    mae_noninferiority_margin: float,
    bootstrap_samples: int = 10_000,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> dict:
    """Build the complete seed-aware VCG timing--relocation report.

    ``mae_noninferiority_margin`` is expressed in the same units as episode
    mean absolute delivery error.  It must be chosen before inspecting the
    diagnostic outcome; the analysis never invents an acceptable tradeoff.
    """

    margin = _finite_float(
        mae_noninferiority_margin,
        field="mae_noninferiority_margin",
        row_index=-1,
    )
    if margin < 0.0:
        raise ParetoAnalysisError("mae_noninferiority_margin must be non-negative")
    if isinstance(bootstrap_samples, bool) or not isinstance(
        bootstrap_samples, Integral
    ):
        raise ParetoAnalysisError("bootstrap_samples must be a positive integer")
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples <= 0:
        raise ParetoAnalysisError("bootstrap_samples must be a positive integer")
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, Integral):
        raise ParetoAnalysisError("rng_seed must be an integer")
    rng_seed = int(rng_seed)

    panel = _prepare_panel(rows)
    safety_gate = _safety_gate(panel)
    if not safety_gate["numeric_pareto_authorized"]:
        return {
            "protocol": PROTOCOL,
            "configuration": {
                "mae_noninferiority_margin": margin,
                "bootstrap_samples": bootstrap_samples,
                "rng_seed": rng_seed,
                "training_seed_t_interval_degrees_of_freedom": 2,
                "training_seed_t_critical_95": T95_DF2,
            },
            "validation": _validation_audit(panel),
            "safety_gate": safety_gate,
            "method_points": {},
            "policy_group_points": {},
            "baseline_points": {},
            "best_vs_final": None,
            "learned_vs_baselines": [],
            "pareto": {
                "analysis_performed": False,
                "reason": (
                    "all six learned arms and at least one baseline must be "
                    "fully eligible"
                ),
                "participants": {},
                "pairwise_point_relations": [],
                "all_points_frontier_including_diagnostic_final": [],
                "deployment_frontier_excluding_diagnostic_final": [],
            },
            "guardrails": None,
            "interpretation_limits": {
                "development_diagnostic_only": True,
                "numeric_pareto_withheld_by_safety_gate": True,
            },
        }
    method_points = {
        method_id: {
            "method_id": method_id,
            "policy_group": panel.method_meta[method_id][0],
            "model_seed": panel.method_meta[method_id][1],
            "point": _method_point(panel, method_id),
        }
        for method_id in sorted(panel.arrays)
    }

    group_points = {}
    for group in LEARNED_GROUPS:
        point = _group_point(panel, group)
        group_points[group] = {
            "policy_group": group,
            "diagnostic_only": group == FINAL_DIAGNOSTIC,
            "deployment_claim_eligible": group == SELECTED_BEST,
            "point": point,
            **_point_uncertainty(
                panel,
                group,
                point,
                bootstrap_samples=bootstrap_samples,
                rng_seed=rng_seed,
            ),
        }

    baseline_points = {}
    for method_id in panel.baseline_methods:
        point = method_points[method_id]["point"]
        baseline_points[method_id] = {
            "method_id": method_id,
            "policy_group": BASELINE,
            "diagnostic_only": False,
            "point": point,
            **_baseline_point_uncertainty(
                panel,
                method_id,
                point,
                bootstrap_samples=bootstrap_samples,
                rng_seed=rng_seed,
            ),
        }

    best_vs_final = _capacity_analysis(
        panel,
        mae_noninferiority_margin=margin,
        bootstrap_samples=bootstrap_samples,
        rng_seed=rng_seed,
    )
    baseline_contrasts = _baseline_contrasts(
        panel,
        bootstrap_samples=bootstrap_samples,
        rng_seed=rng_seed,
    )
    return {
        "protocol": PROTOCOL,
        "configuration": {
            "mae_noninferiority_margin": margin,
            "bootstrap_samples": bootstrap_samples,
            "rng_seed": rng_seed,
            "training_seed_t_interval_degrees_of_freedom": 2,
            "training_seed_t_critical_95": T95_DF2,
        },
        "estimand_and_units": {
            "learned_policy_point": (
                "equal-weight mean over three training-seed panel estimates"
            ),
            "instance_weighting": "equal EpisodeInstance weight within seed",
            "relocation_estimand": (
                "equal-weight training-seed mean of 100 times the ratio of "
                "relocation sums to delivery sums"
            ),
            "training_replication_unit": "independent model_seed",
            "paired_instance_unit": "(instance_id, schedule_id)",
            "deliveries_or_macros_are_independent_replicates": False,
            "flattened_seed_by_instance_rows_are_independent_replicates": False,
        },
        "validation": _validation_audit(panel),
        "safety_gate": safety_gate,
        "method_points": method_points,
        "policy_group_points": group_points,
        "baseline_points": baseline_points,
        "best_vs_final": best_vs_final,
        "learned_vs_baselines": baseline_contrasts,
        "pareto": _pareto_report(group_points, baseline_points),
        "guardrails": {
            "metrics": {
                "first_two_mean_absolute_error": "lower_is_better",
                "positions_three_plus_mean_absolute_error": "lower_is_better",
                "mean_tardiness": "lower_is_better",
                "mean_earliness": "lower_is_better",
                "within_target_window_rate": "higher_is_better",
            },
            "mae_noninferiority_margin": margin,
            "best_vs_final_point_estimate": {
                key: best_vs_final["point_estimate"][key]
                for key in (
                    "first_two_mae_cost",
                    "positions_three_plus_mae_cost",
                    "tardiness_cost",
                    "earliness_cost",
                    "within_window_change",
                )
            },
        },
        "interpretation_limits": {
            "development_diagnostic_only": True,
            "final_weights_are_diagnostic_only": True,
            "final_weights_deployment_claim_eligible": False,
            "selected_best_is_the_only_learned_deployment_group": True,
            "conditional_instance_bootstrap_conditions_on_three_trained_policies": True,
            "crossed_bootstrap_is_sensitivity_not_sole_confirmatory_interval": True,
            "three_seed_t_intervals_are_descriptive_and_wide_df2": True,
            "absence_of_capacity_signal_does_not_prove_architectural_incapacity": True,
            "panel_becomes_development_data_if_used_for_method_choice": True,
            "untouched_test_panel_required_after_method_freeze": True,
        },
    }


__all__ = (
    "BASELINE",
    "DEFAULT_RNG_SEED",
    "FINAL_DIAGNOSTIC",
    "GUARDRAIL_METRICS",
    "PRIMARY_METRICS",
    "PROTOCOL",
    "ParetoAnalysisError",
    "RELOCATION_METRIC",
    "REQUIRED_ROW_FIELDS",
    "SELECTED_BEST",
    "T95_DF2",
    "build_pareto_report",
    "pareto_relation",
    "summarize_seed_values",
    "validate_normalized_rows",
)
