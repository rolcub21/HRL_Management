#!/usr/bin/env python3
"""Pure matched-branch analysis for proactive VCG relocations.

The simulator runner is responsible for authenticating checkpoints and episode
instances, replaying the exact prefix, and producing one normalized row for
each branch at an eligible decision boundary.  This module performs no file
I/O and never loads an environment or a checkpoint.

The primary contrast is the factual VCG relocation (``R``) versus the
highest-Q exactly-safe direct-delivery alternative (``D_Q``).  An exactly-safe
defer branch (``W``) is optional and is reported only as a secondary contrast.
Every delta is oriented so that a positive number favours ``R``.

Events are not independent replicates.  The primary estimand first averages
events within each ``model_seed x EpisodeInstance`` cluster, gives eligible
instances equal weight within a model seed, and finally gives the three fixed
model seeds equal weight.  The bootstrap resamples whole eligible-instance
clusters within each fixed model seed; it does not resample events or trained
models.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from numbers import Integral, Real
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

import numpy as np


PROTOCOL = "vcg_proactive_relocation_amortization_analysis_v1"
R_BRANCH = "R"
D_Q_BRANCH = "D_Q"
W_BRANCH = "W"
D_SELF_BRANCH = "D_SELF"
BRANCH_IDS = (R_BRANCH, D_Q_BRANCH, D_SELF_BRANCH, W_BRANCH)
DEFAULT_TARGET_WINDOW = 20.0
DEFAULT_RNG_SEED = 20_260_808
T95_DF2 = 4.302652729911275
_TOL = 1e-12


class RelocationAmortizationAnalysisError(ValueError):
    """Raised when branch rows violate the frozen analysis contract."""


# Canonical fields and the runner-facing aliases accepted during migration.
# Supplying two names with different values is always an error.
_ALIASES = {
    "branch_id": ("branch",),
    "initial_physical_rehandles": ("initial_macro_physical_relocations",),
    "dense_return": ("total_dense_return_raw",),
    "discounted_dense_return": ("total_dense_discounted_return",),
    "post_initial_physical_rehandles": (
        "future_only_physical_relocations",
        "future_physical_rehandles",
    ),
    "total_physical_rehandles": ("total_physical_relocations",),
    "steps_from_boundary": ("total_steps", "remaining_steps"),
    "complete": ("success",),
    "signed_delivery_errors": ("labeled_errors",),
    "guard_activations": ("guard_activations_delta",),
    "guard_forced_decisions": ("guard_forced_decisions_delta",),
    "exact_safe_execution": (
        "all_executed_candidates_exact_safe",
        "all_candidates_exact_safe",
    ),
}

REQUIRED_FIELDS = (
    "event_id",
    "method_id",
    "model_seed",
    "instance_seed",
    "instance_id",
    "schedule_id",
    "event_index",
    "decision_index",
    "branch_id",
    "eligible",
    "initial_action_type",
    "initial_physical_rehandles",
    "dense_return",
    "discounted_dense_return",
    "post_initial_physical_rehandles",
    "total_physical_rehandles",
    "steps_from_boundary",
    "strict_method_success",
    "complete",
    "terminal",
    "remaining_delivery_count",
    "completed_remaining_delivery_count",
    "expected_remaining_labels",
    "signed_delivery_errors",
    "guard_activations",
    "guard_forced_decisions",
    "exact_safe_execution",
    "macro_failures",
    "illegal_drops",
    "method_failure_reason",
)

IDENTITY_FIELDS = (
    "event_id",
    "method_id",
    "model_seed",
    "instance_seed",
    "instance_id",
    "schedule_id",
    "event_index",
    "decision_index",
)

DELTA_METRICS = (
    "dense_return_gain",
    "discounted_dense_return_gain",
    "total_physical_rehandles_avoided",
    "post_initial_physical_rehandles_avoided",
    "steps_avoided",
    "sum_absolute_timing_error_reduced",
    "mean_absolute_timing_error_reduced",
    "sum_tardiness_reduced",
    "mean_tardiness_reduced",
    "within_window_deliveries_gained",
    "within_window_rate_gain",
    "guard_activations_avoided",
    "guard_forced_decisions_avoided",
)

MECHANISM_FLAGS = (
    "avoids_any_future_rehandle",
    "immediate_rehandle_cost_amortized",
    "strict_net_physical_benefit",
    "timing_accuracy_benefit",
    "tardiness_benefit",
    "target_window_benefit",
    "execution_efficiency_benefit",
    "undiscounted_return_benefit",
    "discounted_return_benefit",
    "guard_intervention_benefit",
)

PARETO_VERDICTS = (
    "R_dominates_comparator",
    "comparator_dominates_R",
    "tie",
    "tradeoff",
)
VALUE_VERDICTS = (
    "safety_beneficial",
    "return_beneficial",
    "return_neutral",
    "return_harmful",
)

_PRODUCTIVITY_MECHANISMS = (
    "avoids_any_future_rehandle",
    "strict_net_physical_benefit",
    "timing_accuracy_benefit",
    "tardiness_benefit",
    "target_window_benefit",
    "execution_efficiency_benefit",
    "guard_intervention_benefit",
)


def _equivalent(left, right) -> bool:
    if isinstance(left, Real) and not isinstance(left, bool):
        if isinstance(right, Real) and not isinstance(right, bool):
            return float(left) == float(right)
    return left == right


def _resolve(row: Mapping, canonical: str, row_index: int):
    names = (canonical, *_ALIASES.get(canonical, ()))
    present = [(name, row[name]) for name in names if name in row]
    if not present:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} is missing required field {canonical!r}"
        )
    value = present[0][1]
    for name, candidate in present[1:]:
        if not _equivalent(value, candidate):
            raise RelocationAmortizationAnalysisError(
                f"row {row_index} has conflicting aliases for {canonical!r}: "
                f"{present[0][0]!r} versus {name!r}"
            )
    return value


def _string(value, *, field: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} field {field!r} must be a non-empty string"
        )
    return value


def _integer(value, *, field: str, row_index: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} field {field!r} must be an integer"
        )
    result = int(value)
    if result < minimum:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} field {field!r} must be >= {minimum}"
        )
    return result


def _finite(value, *, field: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} field {field!r} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} field {field!r} must be finite"
        )
    return result


def _boolean(value, *, field: str, row_index: int) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} field {field!r} must be boolean"
        )
    return bool(value)


def _labels(value, *, row_index: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} expected_remaining_labels must be a sequence"
        )
    labels = tuple(
        _string(label, field="expected_remaining_labels", row_index=row_index)
        for label in value
    )
    if not labels:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} expected_remaining_labels cannot be empty"
        )
    if len(set(labels)) != len(labels):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} expected_remaining_labels contains duplicates"
        )
    return labels


def _error_map(value, *, labels: tuple[str, ...], row_index: int) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} signed_delivery_errors must be a mapping"
        )
    keys = set(value)
    expected = set(labels)
    if not keys.issubset(expected):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} signed_delivery_errors contains labels outside "
            f"the expected workload: extra={sorted(keys - expected)}"
        )
    return {
        label: _finite(
            value[label], field=f"signed_delivery_errors[{label!r}]", row_index=row_index
        )
        for label in labels
        if label in value
    }


def _normalize_row(row: Mapping, row_index: int, target_window: float) -> dict:
    if not isinstance(row, Mapping):
        raise RelocationAmortizationAnalysisError(f"row {row_index} must be a mapping")
    normalized = dict(row)
    for field in REQUIRED_FIELDS:
        normalized[field] = _resolve(row, field, row_index)

    for field in ("event_id", "method_id", "instance_id", "schedule_id"):
        normalized[field] = _string(
            normalized[field], field=field, row_index=row_index
        )
    for field in (
        "model_seed",
        "instance_seed",
        "event_index",
        "decision_index",
        "initial_physical_rehandles",
        "post_initial_physical_rehandles",
        "total_physical_rehandles",
        "steps_from_boundary",
        "remaining_delivery_count",
        "completed_remaining_delivery_count",
        "guard_activations",
        "guard_forced_decisions",
        "macro_failures",
        "illegal_drops",
    ):
        normalized[field] = _integer(
            normalized[field], field=field, row_index=row_index
        )
    for field in ("dense_return", "discounted_dense_return"):
        normalized[field] = _finite(
            normalized[field], field=field, row_index=row_index
        )
    for field in (
        "eligible",
        "strict_method_success",
        "complete",
        "terminal",
        "exact_safe_execution",
    ):
        normalized[field] = _boolean(
            normalized[field], field=field, row_index=row_index
        )

    branch = _string(normalized["branch_id"], field="branch_id", row_index=row_index)
    if branch not in BRANCH_IDS:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} has unknown branch_id {branch!r}"
        )
    normalized["branch_id"] = branch
    action = _string(
        normalized["initial_action_type"],
        field="initial_action_type",
        row_index=row_index,
    ).upper()
    expected_action = {
        R_BRANCH: "RECONFIGURE",
        D_Q_BRANCH: "DELIVER",
        D_SELF_BRANCH: "DELIVER",
        W_BRANCH: "DEFER",
    }[branch]
    if action != expected_action:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} branch {branch!r} requires initial_action_type "
            f"{expected_action!r}, found {action!r}"
        )
    normalized["initial_action_type"] = action

    labels = _labels(normalized["expected_remaining_labels"], row_index=row_index)
    errors = _error_map(
        normalized["signed_delivery_errors"], labels=labels, row_index=row_index
    )
    normalized["expected_remaining_labels"] = labels
    normalized["signed_delivery_errors"] = errors

    if "target_window" in row:
        row_window = _finite(row["target_window"], field="target_window", row_index=row_index)
        if not math.isclose(row_window, target_window, rel_tol=0.0, abs_tol=_TOL):
            raise RelocationAmortizationAnalysisError(
                f"row {row_index} target_window {row_window} does not match "
                f"analysis target_window {target_window}"
            )

    reason = normalized["method_failure_reason"]
    if reason is not None:
        normalized["method_failure_reason"] = _string(
            reason, field="method_failure_reason", row_index=row_index
        )
    if not normalized["eligible"]:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} is not an eligible frozen-protocol event"
        )
    if normalized["remaining_delivery_count"] != len(labels):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} remaining_delivery_count does not match expected labels"
        )
    if normalized["completed_remaining_delivery_count"] != len(errors):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} completed_remaining_delivery_count does not match "
            "its labeled timing outcomes"
        )
    if normalized["steps_from_boundary"] <= 0:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} steps_from_boundary must be positive"
        )
    if (
        normalized["total_physical_rehandles"]
        != normalized["initial_physical_rehandles"]
        + normalized["post_initial_physical_rehandles"]
    ):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} physical-rehandle decomposition is inconsistent"
        )
    if branch == R_BRANCH and normalized["initial_physical_rehandles"] != 1:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} factual R branch must execute exactly one initial rehandle"
        )
    if branch in (D_Q_BRANCH, D_SELF_BRANCH, W_BRANCH) and normalized["initial_physical_rehandles"] != 0:
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} comparator branch cannot rehandle in its initial macro"
        )

    values = list(errors.values())
    absolute = [abs(value) for value in values]
    tardiness = [max(0.0, value) for value in values]
    complete_label_timing = tuple(errors) == tuple(labels)
    normalized["timing_complete"] = complete_label_timing
    normalized["safety_complete"] = bool(
        normalized["strict_method_success"]
        and normalized["complete"]
        and normalized["terminal"]
        and normalized["exact_safe_execution"]
        and normalized["macro_failures"] == 0
        and normalized["illegal_drops"] == 0
        and normalized["method_failure_reason"] is None
        and complete_label_timing
    )
    if normalized["strict_method_success"] and not bool(
        normalized["complete"]
        and normalized["terminal"]
        and normalized["macro_failures"] == 0
        and normalized["illegal_drops"] == 0
        and normalized["method_failure_reason"] is None
        and complete_label_timing
    ):
        raise RelocationAmortizationAnalysisError(
            f"row {row_index} claims strict success but has an inconsistent "
            "completion, failure, or timing record"
        )
    normalized["timing"] = {
        "delivery_count": len(values),
        "sum_absolute_error": float(sum(absolute)) if values else None,
        "mean_absolute_error": float(fmean(absolute)) if values else None,
        "sum_tardiness": float(sum(tardiness)) if values else None,
        "mean_tardiness": float(fmean(tardiness)) if values else None,
        "within_window_count": (
            sum(value <= target_window for value in absolute) if values else None
        ),
        "within_window_rate": (
            float(sum(value <= target_window for value in absolute) / len(values))
            if values
            else None
        ),
    }
    return normalized


def _event_key(row: Mapping) -> tuple:
    return tuple(row[field] for field in IDENTITY_FIELDS)


def _prepare_events(
    rows: Sequence[Mapping],
    *,
    target_window: float,
    expected_model_seeds: Optional[Sequence[int]],
    require_three_model_seeds: bool = True,
) -> tuple[list[dict], tuple[int, ...]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise RelocationAmortizationAnalysisError(
            "rows must be a non-empty sequence of branch mappings"
        )
    normalized = [
        _normalize_row(row, index, target_window) for index, row in enumerate(rows)
    ]
    grouped: dict[str, list[dict]] = {}
    event_identity: dict[str, tuple] = {}
    branch_seen: set[tuple[str, str]] = set()
    model_method: dict[int, tuple[str, Optional[str]]] = {}
    instance_map: dict[str, tuple[int, str]] = {}
    for row in normalized:
        event_id = row["event_id"]
        key = _event_key(row)
        prior = event_identity.setdefault(event_id, key)
        if prior != key:
            raise RelocationAmortizationAnalysisError(
                f"globally unique event_id {event_id!r} maps to inconsistent identity"
            )
        branch_key = (event_id, row["branch_id"])
        if branch_key in branch_seen:
            raise RelocationAmortizationAnalysisError(
                f"duplicate branch {row['branch_id']!r} for event {event_id!r}"
            )
        branch_seen.add(branch_key)
        grouped.setdefault(event_id, []).append(row)

        checkpoint = row.get("checkpoint_sha256")
        meta = (row["method_id"], None if checkpoint is None else str(checkpoint))
        previous_meta = model_method.setdefault(row["model_seed"], meta)
        if previous_meta != meta:
            raise RelocationAmortizationAnalysisError(
                f"model_seed {row['model_seed']} maps to inconsistent method/checkpoint"
            )
        instance_meta = (row["instance_seed"], row["schedule_id"])
        previous_instance = instance_map.setdefault(row["instance_id"], instance_meta)
        if previous_instance != instance_meta:
            raise RelocationAmortizationAnalysisError(
                f"instance_id {row['instance_id']!r} maps to inconsistent seed/schedule"
            )

    events = []
    for event_id in sorted(grouped):
        branches = {row["branch_id"]: row for row in grouped[event_id]}
        if R_BRANCH not in branches or D_Q_BRANCH not in branches:
            raise RelocationAmortizationAnalysisError(
                f"event {event_id!r} must contain exactly one R and one D_Q branch"
            )
        if set(branches) - set(BRANCH_IDS):  # Defensive; normalized already checks.
            raise RelocationAmortizationAnalysisError(
                f"event {event_id!r} contains an unknown branch"
            )
        reference = branches[R_BRANCH]
        if not reference["safety_complete"]:
            raise RelocationAmortizationAnalysisError(
                f"event {event_id!r} factual R branch is not strict, safe, and complete"
            )
        for branch_id, row in branches.items():
            for field in IDENTITY_FIELDS:
                if row[field] != reference[field]:
                    raise RelocationAmortizationAnalysisError(
                        f"event {event_id!r} branch {branch_id!r} mismatches {field!r}"
                    )
            if row["expected_remaining_labels"] != reference["expected_remaining_labels"]:
                raise RelocationAmortizationAnalysisError(
                    f"event {event_id!r} branches do not share identical expected labels"
                )
            if row["remaining_delivery_count"] != reference["remaining_delivery_count"]:
                raise RelocationAmortizationAnalysisError(
                    f"event {event_id!r} branches have different remaining workload"
                )
        events.append({"event_id": event_id, "branches": branches})

    seeds = tuple(sorted(model_method))
    if expected_model_seeds is not None:
        expected = tuple(sorted(_validate_seed_sequence(expected_model_seeds)))
        if seeds != expected:
            raise RelocationAmortizationAnalysisError(
                f"observed model seeds {list(seeds)} do not match expected {list(expected)}"
            )
    if require_three_model_seeds and len(seeds) != 3:
        raise RelocationAmortizationAnalysisError(
            f"analysis requires exactly three model seeds; found {list(seeds)}"
        )
    if not require_three_model_seeds and not 1 <= len(seeds) <= 3:
        raise RelocationAmortizationAnalysisError(
            f"partial smoke validation requires one to three model seeds; found {list(seeds)}"
        )
    return events, seeds


def _validate_seed_sequence(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RelocationAmortizationAnalysisError(
            "expected_model_seeds must be a sequence of three integers"
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RelocationAmortizationAnalysisError(
                "expected_model_seeds must contain integers"
            )
        result.append(int(value))
    if len(result) != 3 or len(set(result)) != 3:
        raise RelocationAmortizationAnalysisError(
            "expected_model_seeds must contain exactly three unique integers"
        )
    return tuple(result)


def _sign(value: float) -> int:
    if value > _TOL:
        return 1
    if value < -_TOL:
        return -1
    return 0


def _pareto_verdict(deltas: Mapping[str, float]) -> str:
    # Operational axes only.  Discounted return is a learner-objective
    # diagnostic and guard counters are safety diagnostics, so neither is
    # allowed to create an operational dominance result.
    axes = (
        "dense_return_gain",
        "total_physical_rehandles_avoided",
        "steps_avoided",
        "sum_absolute_timing_error_reduced",
        "sum_tardiness_reduced",
        "within_window_deliveries_gained",
    )
    signs = [_sign(float(deltas[field])) for field in axes]
    if all(sign >= 0 for sign in signs) and any(sign > 0 for sign in signs):
        return "R_dominates_comparator"
    if all(sign <= 0 for sign in signs) and any(sign < 0 for sign in signs):
        return "comparator_dominates_R"
    if all(sign == 0 for sign in signs):
        return "tie"
    return "tradeoff"


def _contrast(event: Mapping, comparator_id: str) -> dict:
    branches = event["branches"]
    r = branches[R_BRANCH]
    comparator = branches[comparator_id]
    numeric_eligible = bool(r["safety_complete"] and comparator["safety_complete"])
    if not numeric_eligible:
        return {
            **{field: r[field] for field in IDENTITY_FIELDS},
            "comparator_branch_id": comparator_id,
            "remaining_delivery_count": r["remaining_delivery_count"],
            "numeric_comparison_eligible": False,
            "numeric_ineligibility_reason": (
                "both R and comparator must be strict, exact-safe, terminal, "
                "failure-free, and complete the full matched label set"
            ),
            "safety_verdict": (
                "R_safety_beneficial"
                if r["safety_complete"] and not comparator["safety_complete"]
                else "safety_incomparable"
            ),
            "branch_safety": {
                R_BRANCH: _branch_safety(r),
                comparator_id: _branch_safety(comparator),
            },
            "branch_outcomes": {
                R_BRANCH: _branch_outcome(r),
                comparator_id: _branch_outcome(comparator),
            },
            "immediate_rehandle_cost_of_R": (
                r["initial_physical_rehandles"]
                - comparator["initial_physical_rehandles"]
            ),
            "deltas_positive_favors_R": {
                metric: None for metric in DELTA_METRICS
            },
            "mechanism_flags": {flag: None for flag in MECHANISM_FLAGS},
            "value_verdict": "safety_beneficial",
            "operationally_productive": False,
            "operational_productivity_reason": (
                "safety benefit observed, but incomplete comparator outcome "
                "does not establish a downstream mechanism benefit"
            ),
            "pareto_verdict": None,
            "per_label_timing": None,
        }
    rt = r["timing"]
    ct = comparator["timing"]
    deltas = {
        "dense_return_gain": r["dense_return"] - comparator["dense_return"],
        "discounted_dense_return_gain": (
            r["discounted_dense_return"] - comparator["discounted_dense_return"]
        ),
        "total_physical_rehandles_avoided": (
            comparator["total_physical_rehandles"] - r["total_physical_rehandles"]
        ),
        "post_initial_physical_rehandles_avoided": (
            comparator["post_initial_physical_rehandles"]
            - r["post_initial_physical_rehandles"]
        ),
        "steps_avoided": comparator["steps_from_boundary"] - r["steps_from_boundary"],
        "sum_absolute_timing_error_reduced": (
            ct["sum_absolute_error"] - rt["sum_absolute_error"]
        ),
        "mean_absolute_timing_error_reduced": (
            ct["mean_absolute_error"] - rt["mean_absolute_error"]
        ),
        "sum_tardiness_reduced": ct["sum_tardiness"] - rt["sum_tardiness"],
        "mean_tardiness_reduced": ct["mean_tardiness"] - rt["mean_tardiness"],
        "within_window_deliveries_gained": (
            rt["within_window_count"] - ct["within_window_count"]
        ),
        "within_window_rate_gain": (
            rt["within_window_rate"] - ct["within_window_rate"]
        ),
        "guard_activations_avoided": (
            comparator["guard_activations"] - r["guard_activations"]
        ),
        "guard_forced_decisions_avoided": (
            comparator["guard_forced_decisions"] - r["guard_forced_decisions"]
        ),
    }
    immediate_cost = (
        r["initial_physical_rehandles"] - comparator["initial_physical_rehandles"]
    )
    flags = {
        "avoids_any_future_rehandle": (
            deltas["post_initial_physical_rehandles_avoided"] >= 1.0 - _TOL
        ),
        "immediate_rehandle_cost_amortized": (
            immediate_cost > _TOL
            and deltas["post_initial_physical_rehandles_avoided"]
            >= immediate_cost - _TOL
        ),
        "strict_net_physical_benefit": (
            deltas["total_physical_rehandles_avoided"] > _TOL
        ),
        "timing_accuracy_benefit": (
            deltas["sum_absolute_timing_error_reduced"] > _TOL
        ),
        "tardiness_benefit": deltas["sum_tardiness_reduced"] > _TOL,
        "target_window_benefit": deltas["within_window_deliveries_gained"] > _TOL,
        "execution_efficiency_benefit": deltas["steps_avoided"] > _TOL,
        "undiscounted_return_benefit": deltas["dense_return_gain"] > _TOL,
        "discounted_return_benefit": deltas["discounted_dense_return_gain"] > _TOL,
        "guard_intervention_benefit": (
            deltas["guard_activations_avoided"] > _TOL
            or deltas["guard_forced_decisions_avoided"] > _TOL
        ),
    }
    verdict = _pareto_verdict(deltas)
    value_sign = _sign(deltas["dense_return_gain"])
    value_verdict = {
        1: "return_beneficial",
        0: "return_neutral",
        -1: "return_harmful",
    }[value_sign]
    operationally_productive = bool(
        value_verdict == "return_beneficial"
        and any(flags[name] for name in _PRODUCTIVITY_MECHANISMS)
    )
    labels = r["expected_remaining_labels"]
    per_label = {
        label: {
            "R_signed_error": r["signed_delivery_errors"][label],
            f"{comparator_id}_signed_error": comparator["signed_delivery_errors"][label],
            "absolute_error_reduced": (
                abs(comparator["signed_delivery_errors"][label])
                - abs(r["signed_delivery_errors"][label])
            ),
            "tardiness_reduced": (
                max(0.0, comparator["signed_delivery_errors"][label])
                - max(0.0, r["signed_delivery_errors"][label])
            ),
        }
        for label in labels
    }
    return {
        **{field: r[field] for field in IDENTITY_FIELDS},
        "comparator_branch_id": comparator_id,
        "remaining_delivery_count": r["remaining_delivery_count"],
        "numeric_comparison_eligible": True,
        "numeric_ineligibility_reason": None,
        "safety_verdict": "safety_tie",
        "branch_safety": {
            R_BRANCH: _branch_safety(r),
            comparator_id: _branch_safety(comparator),
        },
        "immediate_rehandle_cost_of_R": immediate_cost,
        "branch_outcomes": {
            R_BRANCH: _branch_outcome(r),
            comparator_id: _branch_outcome(comparator),
        },
        "deltas_positive_favors_R": {key: float(value) for key, value in deltas.items()},
        "mechanism_flags": flags,
        "value_verdict": value_verdict,
        "operationally_productive": operationally_productive,
        "operational_productivity_reason": (
            "positive dense return and at least one positive operational mechanism"
            if operationally_productive
            else "the return-benefit and mechanism-benefit conjunction was not met"
        ),
        "pareto_verdict": verdict,
        "per_label_timing": per_label,
    }


def _branch_safety(row: Mapping) -> dict:
    return {
        "safety_complete": bool(row["safety_complete"]),
        "strict_method_success": bool(row["strict_method_success"]),
        "complete": bool(row["complete"]),
        "terminal": bool(row["terminal"]),
        "exact_safe_execution": bool(row["exact_safe_execution"]),
        "macro_failures": int(row["macro_failures"]),
        "illegal_drops": int(row["illegal_drops"]),
        "method_failure_reason": row["method_failure_reason"],
        "expected_remaining_delivery_count": int(row["remaining_delivery_count"]),
        "completed_remaining_delivery_count": int(
            row["completed_remaining_delivery_count"]
        ),
    }


def _branch_outcome(row: Mapping) -> dict:
    return {
        "dense_return": row["dense_return"],
        "discounted_dense_return": row["discounted_dense_return"],
        "initial_physical_rehandles": row["initial_physical_rehandles"],
        "post_initial_physical_rehandles": row["post_initial_physical_rehandles"],
        "total_physical_rehandles": row["total_physical_rehandles"],
        "steps_from_boundary": row["steps_from_boundary"],
        "guard_activations": row["guard_activations"],
        "guard_forced_decisions": row["guard_forced_decisions"],
        **row["timing"],
    }


def _mean_dict(items: Sequence[Mapping[str, float]], fields: Sequence[str]) -> dict:
    if not items:
        raise RelocationAmortizationAnalysisError("cannot average an empty collection")
    return {field: float(fmean(float(item[field]) for item in items)) for field in fields}


def _event_numeric(event: Mapping) -> dict:
    values = dict(event["deltas_positive_favors_R"])
    values.update(
        {f"fraction_{flag}": float(event["mechanism_flags"][flag]) for flag in MECHANISM_FLAGS}
    )
    values.update(
        {
            f"fraction_value_{verdict}": float(event["value_verdict"] == verdict)
            for verdict in VALUE_VERDICTS
        }
    )
    values["fraction_operationally_productive"] = float(
        event["operationally_productive"]
    )
    values.update(
        {
            f"fraction_pareto_{verdict}": float(event["pareto_verdict"] == verdict)
            for verdict in PARETO_VERDICTS
        }
    )
    return values


def _summarize_numeric(values: Mapping[str, float]) -> dict:
    deltas = {field: float(values[field]) for field in DELTA_METRICS}
    mechanisms = {
        flag: float(values[f"fraction_{flag}"]) for flag in MECHANISM_FLAGS
    }
    verdicts = {
        verdict: float(values[f"fraction_pareto_{verdict}"])
        for verdict in PARETO_VERDICTS
    }
    value_verdicts = {
        verdict: float(values[f"fraction_value_{verdict}"])
        for verdict in VALUE_VERDICTS
    }
    return {
        "deltas_positive_favors_R": deltas,
        "mechanism_fractions": mechanisms,
        "pareto_verdict_fractions": verdicts,
        "value_verdict_fractions": value_verdicts,
        "operationally_productive_fraction": float(
            values["fraction_operationally_productive"]
        ),
    }


def _cluster_events(primary_events: Sequence[Mapping]) -> dict[int, dict[tuple, list[dict]]]:
    clusters: dict[int, dict[tuple, list[dict]]] = {}
    for event in primary_events:
        seed = int(event["model_seed"])
        instance = (
            int(event["instance_seed"]),
            event["instance_id"],
            event["schedule_id"],
        )
        clusters.setdefault(seed, {}).setdefault(instance, []).append(_event_numeric(event))
    return clusters


def _aggregate_primary(primary_events: Sequence[Mapping], seeds: Sequence[int]) -> tuple[dict, dict]:
    clusters = _cluster_events(primary_events)
    numeric_fields = (
        *DELTA_METRICS,
        *(f"fraction_{x}" for x in MECHANISM_FLAGS),
        *(f"fraction_pareto_{x}" for x in PARETO_VERDICTS),
        *(f"fraction_value_{x}" for x in VALUE_VERDICTS),
        "fraction_operationally_productive",
    )
    cluster_means: dict[int, dict[tuple, dict]] = {}
    per_seed = {}
    for seed in seeds:
        seed_clusters = clusters.get(seed, {})
        if not seed_clusters:
            raise RelocationAmortizationAnalysisError(
                f"model_seed {seed} has no eligible proactive-relocation events"
            )
        cluster_means[seed] = {
            key: _mean_dict(events, numeric_fields)
            for key, events in seed_clusters.items()
        }
        seed_mean = _mean_dict(list(cluster_means[seed].values()), numeric_fields)
        per_seed[str(seed)] = {
            "eligible_instance_count": len(seed_clusters),
            "event_count": sum(len(events) for events in seed_clusters.values()),
            "eligible_instances": [
                {
                    "instance_seed": key[0],
                    "instance_id": key[1],
                    "schedule_id": key[2],
                    "event_count": len(seed_clusters[key]),
                }
                for key in sorted(seed_clusters)
            ],
            **_summarize_numeric(seed_mean),
        }
    overall_values = {
        field: float(
            fmean(_summary_numeric_value(per_seed[str(seed)], field) for seed in seeds)
        )
        for field in numeric_fields
    }
    return {
        "weighting": (
            "mean events within model_seed x eligible EpisodeInstance, then "
            "equal eligible-instance weight within model seed, then equal weight "
            "over the three fixed model seeds"
        ),
        "per_model_seed": per_seed,
        "equal_model_seed_mean": _summarize_numeric(overall_values),
    }, cluster_means


def _numeric_location(field: str) -> tuple[str, str]:
    if field in DELTA_METRICS:
        return "deltas_positive_favors_R", field
    prefix = "fraction_pareto_"
    if field.startswith(prefix):
        return "pareto_verdict_fractions", field[len(prefix) :]
    prefix = "fraction_value_"
    if field.startswith(prefix):
        return "value_verdict_fractions", field[len(prefix) :]
    if field == "fraction_operationally_productive":
        return "operationally_productive_fraction", ""
    prefix = "fraction_"
    if field.startswith(prefix):
        return "mechanism_fractions", field[len(prefix) :]
    raise RuntimeError(f"unknown numeric aggregate field {field!r}")


def _summary_numeric_value(summary: Mapping, field: str) -> float:
    section, name = _numeric_location(field)
    value = summary[section] if not name else summary[section][name]
    return float(value)


def _aggregate_event_weighted(primary_events: Sequence[Mapping], seeds: Sequence[int]) -> dict:
    numeric_fields = (
        *DELTA_METRICS,
        *(f"fraction_{x}" for x in MECHANISM_FLAGS),
        *(f"fraction_pareto_{x}" for x in PARETO_VERDICTS),
        *(f"fraction_value_{x}" for x in VALUE_VERDICTS),
        "fraction_operationally_productive",
    )
    by_seed = {}
    raw_by_seed = {}
    for seed in seeds:
        values = [_event_numeric(event) for event in primary_events if event["model_seed"] == seed]
        raw = _mean_dict(values, numeric_fields)
        raw_by_seed[seed] = raw
        by_seed[str(seed)] = {"event_count": len(values), **_summarize_numeric(raw)}
    equal_seed = {
        field: float(fmean(raw_by_seed[seed][field] for seed in seeds))
        for field in numeric_fields
    }
    pooled = _mean_dict([_event_numeric(event) for event in primary_events], numeric_fields)
    return {
        "secondary_only": True,
        "warning": (
            "event-weighted summaries overweight instances where the frozen "
            "policy chose more proactive relocations; events are not independent"
        ),
        "per_model_seed": by_seed,
        "equal_model_seed_mean": _summarize_numeric(equal_seed),
        "pooled_event_mean": _summarize_numeric(pooled),
    }


def _safety_ledger(events: Sequence[Mapping], seeds: Sequence[int]) -> dict:
    per_seed = {}
    for seed in seeds:
        selected = [event for event in events if int(event["model_seed"]) == int(seed)]
        eligible = sum(bool(event["numeric_comparison_eligible"]) for event in selected)
        beneficial = sum(event["safety_verdict"] == "R_safety_beneficial" for event in selected)
        reasons = Counter(
            event["branch_safety"][event["comparator_branch_id"]][
                "method_failure_reason"
            ]
            or "unsafe_or_incomplete_without_reason"
            for event in selected
            if not event["numeric_comparison_eligible"]
        )
        value_counts = Counter(event["value_verdict"] for event in selected)
        productive_counts = Counter(
            bool(event["operationally_productive"]) for event in selected
        )
        per_seed[str(seed)] = {
            "total_event_count": len(selected),
            "numeric_complete_pair_count": eligible,
            "numeric_complete_pair_rate": eligible / len(selected),
            "R_safety_beneficial_count": beneficial,
            "value_verdict_counts": {
                verdict: value_counts.get(verdict, 0) for verdict in VALUE_VERDICTS
            },
            "operationally_productive_count": productive_counts.get(True, 0),
            "operationally_productive_fraction_full_denominator": (
                productive_counts.get(True, 0) / len(selected)
            ),
            "comparator_failure_reason_counts": dict(sorted(reasons.items())),
        }
    total = len(events)
    numeric = sum(bool(event["numeric_comparison_eligible"]) for event in events)
    value_counts = Counter(event["value_verdict"] for event in events)
    productive = sum(bool(event["operationally_productive"]) for event in events)
    return {
        "full_event_denominator": total,
        "numeric_complete_pair_numerator": numeric,
        "numeric_complete_pair_rate": numeric / total,
        "R_safety_beneficial_count": sum(
            event["safety_verdict"] == "R_safety_beneficial" for event in events
        ),
        "value_verdict_counts": {
            verdict: value_counts.get(verdict, 0) for verdict in VALUE_VERDICTS
        },
        "operationally_productive_count": productive,
        "operationally_productive_fraction_full_denominator": productive / total,
        "no_failed_comparator_event_removed_from_event_ledger": True,
        "per_model_seed": per_seed,
    }


def _seed_t_summaries(primary: Mapping, seeds: Sequence[int]) -> dict:
    output = {}
    for metric in DELTA_METRICS:
        values = {
            int(seed): float(
                primary["per_model_seed"][str(seed)]["deltas_positive_favors_R"][metric]
            )
            for seed in seeds
        }
        ordered = [values[int(seed)] for seed in seeds]
        center = float(fmean(ordered))
        sd = float(stdev(ordered))
        se = sd / math.sqrt(3.0)
        half = T95_DF2 * se
        output[metric] = {
            "values_by_model_seed": {str(seed): values[int(seed)] for seed in seeds},
            "mean": center,
            "sample_standard_deviation": sd,
            "standard_error": se,
            "t95_interval": [center - half, center + half],
            "degrees_of_freedom": 2,
            "t_critical": T95_DF2,
        }
    return output


def _derived_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _bootstrap_clusters(
    cluster_means: Mapping[int, Mapping[tuple, Mapping]],
    seeds: Sequence[int],
    *,
    samples: int,
    rng_seed: int,
) -> dict:
    rng_stream = _derived_seed(rng_seed, "fixed-model-instance-cluster-bootstrap")
    rng = np.random.default_rng(rng_stream)
    fields = DELTA_METRICS
    draws = {field: np.empty(samples, dtype=float) for field in fields}
    for draw_index in range(samples):
        seed_estimates = {field: [] for field in fields}
        for seed in seeds:
            clusters = list(cluster_means[int(seed)].values())
            indices = rng.integers(0, len(clusters), size=len(clusters))
            for field in fields:
                seed_estimates[field].append(
                    float(fmean(float(clusters[index][field]) for index in indices))
                )
        for field in fields:
            draws[field][draw_index] = float(fmean(seed_estimates[field]))
    return {
        "samples": samples,
        "base_rng_seed": int(rng_seed),
        "stream_rng_seed": rng_stream,
        "resampling": (
            "eligible_EpisodeInstance_clusters_within_each_fixed_model_seed; "
            "all events in a sampled cluster remain together; model seeds are "
            "conditioned on and are not resampled"
        ),
        "conditional_on_model_seed_count": 3,
        "metrics": {
            field: {
                "estimate": float(fmean(
                    fmean(float(cluster[field]) for cluster in cluster_means[int(seed)].values())
                    for seed in seeds
                )),
                "bootstrap_mean": float(draws[field].mean()),
                "percentile_95_interval": [
                    float(value)
                    for value in np.quantile(draws[field], (0.025, 0.975))
                ],
            }
            for field in fields
        },
    }


def validate_branch_rows(
    rows: Sequence[Mapping],
    *,
    target_window: float = DEFAULT_TARGET_WINDOW,
    expected_model_seeds: Optional[Sequence[int]] = None,
    allow_partial_smoke: bool = False,
) -> dict:
    """Validate all rows and return a compact, JSON-safe audit.

    Validation fails closed on malformed identity, factual-R safety, or
    inconsistent accounting. Comparator failures remain valid causal outcomes
    in the full ledger and make only that event's numeric comparison ineligible.
    """

    window = _finite(target_window, field="target_window", row_index=-1)
    if window < 0.0:
        raise RelocationAmortizationAnalysisError("target_window must be non-negative")
    if not isinstance(allow_partial_smoke, bool):
        raise RelocationAmortizationAnalysisError("allow_partial_smoke must be boolean")
    events, seeds = _prepare_events(
        rows,
        target_window=window,
        expected_model_seeds=expected_model_seeds,
        require_three_model_seeds=not allow_partial_smoke,
    )
    branch_counts = Counter(
        branch_id
        for event in events
        for branch_id in event["branches"]
    )
    primary_complete = sum(
        bool(event["branches"][D_Q_BRANCH]["safety_complete"])
        for event in events
    )
    all_rows = [row for event in events for row in event["branches"].values()]
    return {
        "row_count": int(sum(branch_counts.values())),
        "event_count": len(events),
        "model_seeds": list(seeds),
        "partial_smoke_validation_only": bool(allow_partial_smoke),
        "full_three_model_analysis_authorized": len(seeds) == 3,
        "branch_counts": {branch: branch_counts.get(branch, 0) for branch in BRANCH_IDS},
        "primary_R_D_Q_pair_complete_for_every_event": True,
        "optional_W_event_count": branch_counts.get(W_BRANCH, 0),
        "optional_D_SELF_event_count": branch_counts.get(D_SELF_BRANCH, 0),
        "factual_R_strict_safe_complete_terminal_for_every_event": True,
        "all_branches_strict_safe_complete_terminal": all(
            row["safety_complete"] for row in all_rows
        ),
        "primary_numeric_complete_pair_count": primary_complete,
        "primary_numeric_complete_pair_denominator": len(events),
        "all_timing_values_observed_are_finite": True,
        "all_complete_branch_timing_labels_full_and_matched": True,
        "all_rehandle_decompositions_valid": True,
    }


def build_relocation_amortization_report(
    rows: Sequence[Mapping],
    *,
    target_window: float = DEFAULT_TARGET_WINDOW,
    expected_model_seeds: Optional[Sequence[int]] = None,
    bootstrap_samples: int = 10_000,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> dict:
    """Build the complete, seed-aware relocation-amortization report."""

    window = _finite(target_window, field="target_window", row_index=-1)
    if window < 0.0:
        raise RelocationAmortizationAnalysisError("target_window must be non-negative")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, Integral):
        raise RelocationAmortizationAnalysisError(
            "bootstrap_samples must be a positive integer"
        )
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples <= 0:
        raise RelocationAmortizationAnalysisError(
            "bootstrap_samples must be a positive integer"
        )
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, Integral):
        raise RelocationAmortizationAnalysisError("rng_seed must be an integer")
    rng_seed = int(rng_seed)

    events, seeds = _prepare_events(
        rows, target_window=window, expected_model_seeds=expected_model_seeds
    )
    primary = [_contrast(event, D_Q_BRANCH) for event in events]
    self_delivery = [
        _contrast(event, D_SELF_BRANCH)
        for event in events
        if D_SELF_BRANCH in event["branches"]
    ]
    wait = [_contrast(event, W_BRANCH) for event in events if W_BRANCH in event["branches"]]
    numeric_primary = [
        event for event in primary if event["numeric_comparison_eligible"]
    ]
    numeric_seed_coverage = {
        seed for event in numeric_primary for seed in (int(event["model_seed"]),)
    }
    numeric_analysis_performed = numeric_seed_coverage == set(seeds)
    primary_aggregate = None
    event_weighted = None
    seed_t = None
    bootstrap = None
    if numeric_analysis_performed:
        primary_aggregate, cluster_means = _aggregate_primary(numeric_primary, seeds)
        event_weighted = _aggregate_event_weighted(numeric_primary, seeds)
        seed_t = _seed_t_summaries(primary_aggregate, seeds)
        bootstrap = _bootstrap_clusters(
            cluster_means,
            seeds,
            samples=bootstrap_samples,
            rng_seed=rng_seed,
        )
    validation = validate_branch_rows(
        rows, target_window=window, expected_model_seeds=seeds
    )
    return {
        "protocol": PROTOCOL,
        "configuration": {
            "target_window": window,
            "bootstrap_samples": bootstrap_samples,
            "rng_seed": rng_seed,
            "primary_comparator": D_Q_BRANCH,
            "optional_secondary_comparator": W_BRANCH,
        },
        "validation": validation,
        "estimand": {
            "delta_orientation": "every reported delta is positive when R is better",
            "primary_weighting": primary_aggregate["weighting"],
            "secondary_event_weighting_is_primary": False,
            "training_seed_replication_unit": "model_seed (three exposed estimates)",
            "bootstrap_unit": "eligible model_seed x EpisodeInstance cluster",
            "events_are_independent_replicates": False,
            "comparison_identifies": (
                "a one-step-deviation value under the same frozen suffix policy, "
                "not the value of an always-never-reconfigure policy"
            ),
        },
        "pareto_protocol": {
            "safety_and_completion_are_prerequisites": True,
            "axes_positive_favors_R": [
                "dense_return_gain",
                "total_physical_rehandles_avoided",
                "steps_avoided",
                "sum_absolute_timing_error_reduced",
                "sum_tardiness_reduced",
                "within_window_deliveries_gained",
            ],
            "discounted_return_excluded_from_operational_dominance": True,
            "guard_counters_excluded_from_operational_dominance": True,
            "tolerance": _TOL,
        },
        "primary_R_vs_D_Q": {
            "event_count": len(primary),
            "safety_ledger": _safety_ledger(primary, seeds),
            "events": primary,
            "numeric_analysis_performed": numeric_analysis_performed,
            "numeric_complete_pair_count": len(numeric_primary),
            "numeric_complete_pair_denominator": len(primary),
            "model_seeds_with_numeric_pairs": sorted(numeric_seed_coverage),
            "instance_weighted_primary": primary_aggregate,
            "event_weighted_secondary": event_weighted,
            "three_seed_descriptive_t95": seed_t,
            "conditional_instance_cluster_bootstrap": bootstrap,
        },
        "secondary_R_vs_W": {
            "analysis_performed": bool(wait),
            "event_count": len(wait),
            "events": wait,
            "note": (
                "W is included only where the exact post-guard frontier contained "
                "an admissible defer candidate; no synthetic wait is imputed"
            ),
        },
        "secondary_R_vs_D_SELF": {
            "analysis_performed": bool(self_delivery),
            "event_count": len(self_delivery),
            "events": self_delivery,
            "note": (
                "D_SELF is the exactly-safe branch that directly delivers the "
                "same block selected by factual Reconfigure. It is diagnostic "
                "only and never substitutes for the precommitted D_Q comparator."
            ),
        },
        "interpretation_limits": {
            "development_mechanism_audit_only": True,
            "causal_scope": "matched branch boundary under deterministic replay",
            "event_selection_is_policy_conditioned": True,
            "no_claim_that_every_proactive_relocation_is_required": True,
            "no_posthoc_materiality_threshold": True,
        },
    }


# Concise public alias for callers that do not need the longer protocol name.
analyze_relocation_amortization = build_relocation_amortization_report


__all__ = [
    "PROTOCOL",
    "R_BRANCH",
    "D_Q_BRANCH",
    "W_BRANCH",
    "D_SELF_BRANCH",
    "DELTA_METRICS",
    "MECHANISM_FLAGS",
    "PARETO_VERDICTS",
    "VALUE_VERDICTS",
    "RelocationAmortizationAnalysisError",
    "validate_branch_rows",
    "build_relocation_amortization_report",
    "analyze_relocation_amortization",
]
