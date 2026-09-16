#!/usr/bin/env python3
"""Paired complete-system comparison for the exact-safe VCG controller.

The learned arm uses the canonical complete exact frontier.  The baseline arms
use one source-neutral duration-aware reserved-cell scheduler and swap only its
online assignment source.  All arms consume the same saved EpisodeInstance.

This is a complete-system comparison, not an assignment-source ablation: VCG
chooses Accept/Recover/Defer macros itself, whereas each baseline combines a
fixed temporal scheduler with a storage-assignment source.  Primitive return,
steps, timing, physical relocations, and strict success are comparable; macro
decision counts are intentionally not compared.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import numpy as np

from benchmark_viability_critic_priority import (
    EXACT_FULL,
    _load_controller_checkpoint,
    _liveness_rule,
    _make_env,
    _search_config,
    _sha256_file,
    run_arm,
)
from contention_metrics import (
    CANONICAL_CONTENTION_FIELDS,
    CONTENTION_METRIC_SCHEMA_VERSION,
    contention_metric_record,
    validate_contention_metric_record,
)
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.track_a import (
    TRACK_A_DYNAMIC,
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_ROLLING_COMPLETE,
    TRACK_A_NEAREST_FREE,
)
from track_b_urgency_evaluate import (
    DECISION_EPOCH_RESERVED,
    evaluate_assignment_ablation_one,
    resolve_device,
)
from train_viability_graph_smdp import SEALED_STRESS_V1_HOLDOUT_SEEDS
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS
from vcg_objective_audit import (
    DEFAULT_DENSE_B,
    DEFAULT_DENSE_WINDOW,
    DEFAULT_LAMBDA_ABS,
    DEFAULT_LAMBDA_OUTSIDE,
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    TimingObjectiveSpec,
    delivery_reward,
)


PROTOCOL = "paired_vcg_complete_system_online_baselines_v2"
VCG_METHOD = "vcg_smdp"
NEAREST_METHOD = "duration_aware_nearest_free"
DYNAMIC_METHOD = "duration_aware_dynamic_pslap"
ROLLING_GA_METHOD = "duration_aware_pslap_ga_2009_rolling"
ENHANCED_GA_METHOD = "duration_aware_enhanced_complete_rolling_ga"
METHOD_TO_SOURCE = {
    NEAREST_METHOD: TRACK_A_NEAREST_FREE,
    DYNAMIC_METHOD: TRACK_A_DYNAMIC,
    ROLLING_GA_METHOD: TRACK_A_GA_ROLLING,
    ENHANCED_GA_METHOD: TRACK_A_GA_ROLLING_COMPLETE,
}
METHODS = (VCG_METHOD, *METHOD_TO_SOURCE)
DEFAULT_METHODS = (
    VCG_METHOD,
    NEAREST_METHOD,
    DYNAMIC_METHOD,
    ROLLING_GA_METHOD,
)
RESULT_FIELDS = (
    "protocol",
    "method",
    "method_family",
    "information_regime",
    "instance_seed",
    "instance_id",
    "schedule_id",
    "return",
    "legacy_rescored_return",
    "dense_rescored_return",
    "success",
    "strict_method_success",
    "completion_rate",
    "steps",
    "delivery_count",
    "delivery_deviations",
    "first_two_mean_absolute_error",
    "first_two_within_target_window_rate",
    "positions_three_plus_mean_absolute_error",
    "positions_three_plus_within_target_window_rate",
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "p90_tardiness",
    "p90_absolute_error",
    *CANONICAL_CONTENTION_FIELDS,
    "relocations",
    "obstructive_moves",
    "illegal_drops",
    "invalid_assignments",
    "infeasible_epochs",
    "fallbacks",
    "method_failure_reason",
    "planning_seconds",
    "episode_wall_seconds",
)


def _dual_rescore_from_legacy_return(
    legacy_return: float,
    delivery_deviations: Sequence[float],
    spec: TimingObjectiveSpec,
) -> tuple[float, float]:
    """Exactly rescore a realized legacy trajectory under the dense objective.

    The environment dynamics and all non-delivery rewards are common to both
    objectives, so only the realized delivery terms need to be replaced.  This
    remains exact for incomplete baseline episodes and avoids inventing an
    unobserved placement count.
    """

    legacy_return = float(legacy_return)
    legacy_delivery = sum(
        delivery_reward(error, spec, LEGACY_CLIPPED)
        for error in delivery_deviations
    )
    dense_delivery = sum(
        delivery_reward(error, spec, DENSE_PIECEWISE)
        for error in delivery_deviations
    )
    return legacy_return, float(
        legacy_return - legacy_delivery + dense_delivery
    )


def _finite_mean(values) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    finite = [value for value in finite if math.isfinite(value)]
    return float(fmean(finite)) if finite else None


def _delivery_position_metrics(
    delivery_deviations: Sequence[float], target_window: float
) -> dict:
    values = tuple(float(value) for value in delivery_deviations)
    first_two = values[:2]
    later = values[2:]

    def summarize(selected: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
        if not selected:
            return None, None
        return (
            float(fmean(abs(value) for value in selected)),
            float(
                fmean(abs(value) <= float(target_window) for value in selected)
            ),
        )

    first_mae, first_window = summarize(first_two)
    later_mae, later_window = summarize(later)
    return {
        "delivery_deviations": values,
        "first_two_mean_absolute_error": first_mae,
        "first_two_within_target_window_rate": first_window,
        "positions_three_plus_mean_absolute_error": later_mae,
        "positions_three_plus_within_target_window_rate": later_window,
    }


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value):
    if isinstance(value, (dict, tuple, list)):
        return json.dumps(_json_safe(value), separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _load_instances(args, controller_payload) -> dict[int, EpisodeInstance]:
    result = {}
    for seed in args.seeds:
        env = _make_env(controller_payload)
        if args.instances_dir is None:
            instance = env.sample_episode_instance(int(seed))
        else:
            path = args.instances_dir / f"seed-{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
            if instance.seed is not None and int(instance.seed) != int(seed):
                raise ValueError(f"instance seed mismatch in {path}")
            instance.validate_for(env)
        result[int(seed)] = instance
    return result


def _baseline_runtime_args(args, controller_payload, device) -> SimpleNamespace:
    environment = controller_payload["environment"]
    return SimpleNamespace(
        lam=float(environment["arrival_rate"]),
        mu=float(environment["proc_mean"]),
        grid_rows=int(environment["grid_rows"]),
        grid_cols=int(environment["grid_cols"]),
        number_blocks=int(environment["number_blocks"]),
        exit_width=None,
        instance=None,
        device=str(device),
        max_steps=int(args.max_steps),
        max_defer_steps=int(args.max_defer_steps),
        lookahead_margin_steps=float(args.lookahead_margin_steps),
        assignment_commitment=DECISION_EPOCH_RESERVED,
        target_window=float(args.target_window),
        ga_seed_base=int(args.ga_seed_base),
        rolling_population=int(args.rolling_population),
        rolling_generations=int(args.rolling_generations),
        rolling_ga_egress_weight=int(args.rolling_ga_egress_weight),
        save_instances_dir=None,
        allow_selector_regime_shift=False,
    )


def _normalize_vcg(
    raw: dict,
    instance: EpisodeInstance,
    block_count: int,
    timing_spec: TimingObjectiveSpec,
) -> dict:
    delivery_count = len(raw["delivery_deviations"])
    legacy_return, dense_return = _dual_rescore_from_legacy_return(
        raw["return"], raw["delivery_deviations"], timing_spec
    )
    position_metrics = _delivery_position_metrics(
        raw["delivery_deviations"], timing_spec.window
    )
    contention_metrics = validate_contention_metric_record(raw)
    physical_relocations = contention_metrics[
        "physical_storage_relocations"
    ]
    target_bound_clearances = contention_metrics[
        "target_bound_obstruction_clearances"
    ]
    if int(raw["relocations"]) != physical_relocations:
        raise RuntimeError(
            "VCG legacy relocations is not an exact alias of physical storage "
            "relocations"
        )
    if int(raw.get("obstructive_moves", target_bound_clearances)) != (
        target_bound_clearances
    ):
        raise RuntimeError(
            "VCG obstructive_moves cannot be treated as physical relocation; "
            "it must alias target-bound obstruction clearances"
        )
    return {
        "protocol": PROTOCOL,
        "method": VCG_METHOD,
        "method_family": "learned_exact_viability_constrained_graph_smdp",
        "information_regime": "online_arrived_only_closed_admission_certificate",
        "instance_seed": int(raw["instance_seed"]),
        "instance_id": raw["instance_id"],
        "schedule_id": instance.schedule_id,
        "return": legacy_return,
        "legacy_rescored_return": legacy_return,
        "dense_rescored_return": dense_return,
        "success": float(raw["success"]),
        "strict_method_success": float(raw["strict_method_success"]),
        "completion_rate": delivery_count / block_count,
        "steps": int(raw["steps"]),
        "delivery_count": delivery_count,
        **position_metrics,
        "mean_signed_deviation": raw["mean_signed_deviation"],
        "mean_absolute_error": raw["mean_absolute_error"],
        "mean_tardiness": raw["mean_tardiness"],
        "mean_earliness": raw["mean_earliness"],
        "within_target_window_rate": raw["within_target_window_rate"],
        "p90_tardiness": raw["p90_tardiness"],
        "p90_absolute_error": raw["p90_absolute_error"],
        **contention_metrics,
        # Migration aliases.  They are checked again by downstream protocol
        # writers and may be removed only in a future schema version.
        "relocations": physical_relocations,
        "obstructive_moves": target_bound_clearances,
        "illegal_drops": int(raw["illegal_drops"]),
        "invalid_assignments": 0,
        "infeasible_epochs": 0,
        "fallbacks": 0,
        "method_failure_reason": raw["method_failure_reason"],
        "planning_seconds": float(raw["total_frontier_seconds"])
        + sum(float(item["selection_seconds"]) for item in raw["decisions"]),
        "episode_wall_seconds": float(raw["episode_wall_seconds"]),
        "method_audit": {
            "controller_action_interface": "exact_safe_accept_recover_defer_macros",
            "critic_used": False,
            "complete_frontier_exactly_verified": raw[
                "complete_frontier_exactly_verified"
            ],
            "exact_verifier_authoritative": True,
            "exact_cache_hits": raw["exact_cache_hits"],
            "exact_cache_misses": raw["exact_cache_misses"],
            "exact_search_seconds": raw["exact_search_seconds"],
            "macro_decisions": raw["macro_decisions"],
            "macro_failures": raw["macro_failures"],
            "contention_metric_schema_version": (
                CONTENTION_METRIC_SCHEMA_VERSION
            ),
            "decision_epochs_with_direct_delivery_available": int(
                raw["decision_epochs_with_direct_delivery_available"]
            ),
            "reconfiguration_decision_epochs": int(
                raw["reconfiguration_decision_epochs"]
            ),
            "reconfiguration_decision_epochs_with_direct_delivery_available": int(
                raw[
                    "reconfiguration_decision_epochs_with_direct_delivery_available"
                ]
            ),
            "reconfiguration_decision_epochs_without_direct_delivery_available": int(
                raw[
                    "reconfiguration_decision_epochs_without_direct_delivery_available"
                ]
            ),
            "directly_deliverable_self_reconfiguration_decision_epochs": int(
                raw[
                    "directly_deliverable_self_reconfiguration_decision_epochs"
                ]
            ),
            "frontiers": raw["frontiers"],
        },
    }


def _normalize_baseline(
    method: str,
    raw: dict,
    instance: EpisodeInstance,
    block_count: int,
    timing_spec: TimingObjectiveSpec,
) -> dict:
    selector = raw["selector_audit"]
    scheduler = raw["urgency_scheduler_audit"]
    options = raw["scheduler_audit"]
    planning_seconds = float(
        selector.get("assignment_total_seconds", selector["assignment_seconds"])
    ) + float(scheduler["preview_seconds"])
    completed_option_clearances = int(options["retrieve_relocations"])
    if "physical_storage_relocations" in raw:
        raw_schema = raw.get(
            "contention_metric_schema_version",
            raw.get("relocation_metric_schema"),
        )
        if raw_schema != CONTENTION_METRIC_SCHEMA_VERSION:
            raise RuntimeError("baseline event-time contention schema mismatch")
        contention_metrics = contention_metric_record(
            physical_storage_relocations=raw[
                "physical_storage_relocations"
            ],
            target_bound_obstruction_clearances=raw[
                "target_bound_obstruction_clearances"
            ],
            standalone_reconfigurations=raw["standalone_reconfigurations"],
            standalone_with_direct_delivery_available=raw[
                "standalone_with_direct_delivery_available"
            ],
            standalone_without_direct_delivery_available=raw[
                "standalone_without_direct_delivery_available"
            ],
            directly_deliverable_self_reconfigurations=raw[
                "directly_deliverable_self_reconfigurations"
            ],
        )
        physical_relocations = contention_metrics[
            "physical_storage_relocations"
        ]
        target_bound_clearances = contention_metrics[
            "target_bound_obstruction_clearances"
        ]
        if int(raw.get("relocations", physical_relocations)) != physical_relocations:
            raise RuntimeError(
                "baseline legacy relocations is not an exact alias of physical "
                "storage relocations"
            )
    else:
        # Legacy evaluators exposed only the generic environment event and a
        # completed-option audit.  In this controller interface every physical
        # storage-to-storage move is executed inside RetrieveDeliverOption, so
        # the event total supplies the lossless count even if truncation leaves
        # the final option absent from the completed-option audit.
        physical_relocations = int(
            raw.get("relocations", raw["obstructive_moves"])
        )
        target_bound_clearances = physical_relocations
        if (
            bool(raw.get("strict_method_success"))
            and physical_relocations != completed_option_clearances
        ):
            raise RuntimeError(
                "successful legacy baseline physical relocation events do not "
                "match the completed retrieval-option audit"
            )
        contention_metrics = contention_metric_record(
            physical_storage_relocations=physical_relocations,
            target_bound_obstruction_clearances=target_bound_clearances,
            standalone_reconfigurations=0,
            standalone_with_direct_delivery_available=0,
            standalone_without_direct_delivery_available=0,
            directly_deliverable_self_reconfigurations=0,
        )
    invalid = int(selector["invalid_assignment_count"])
    infeasible = int(selector["infeasible_epoch_count"])
    fallbacks = int(selector["fallback_count"])
    strict = bool(raw["strict_method_success"])
    strict = strict and invalid == 0 and fallbacks == 0
    legacy_return, dense_return = _dual_rescore_from_legacy_return(
        raw["return"], raw["delivery_deviations"], timing_spec
    )
    position_metrics = _delivery_position_metrics(
        raw["delivery_deviations"], timing_spec.window
    )
    return {
        "protocol": PROTOCOL,
        "method": method,
        "method_family": "deterministic_or_optimization_assignment_plus_duration_scheduler",
        "information_regime": "online_arrived_only",
        "instance_seed": int(raw["eval_seed"]),
        "instance_id": raw["instance_id"],
        "schedule_id": raw["schedule_id"],
        "return": legacy_return,
        "legacy_rescored_return": legacy_return,
        "dense_rescored_return": dense_return,
        "success": float(raw["success"]),
        "strict_method_success": float(strict),
        "completion_rate": int(raw["delivery_count"]) / block_count,
        "steps": int(raw["steps"]),
        "delivery_count": int(raw["delivery_count"]),
        **position_metrics,
        "mean_signed_deviation": raw["mean_signed_deviation"],
        "mean_absolute_error": raw["mean_absolute_error"],
        "mean_tardiness": raw["mean_tardiness"],
        "mean_earliness": raw["mean_earliness"],
        "within_target_window_rate": raw["within_target_window_rate"],
        "p90_tardiness": raw["p90_tardiness"],
        "p90_absolute_error": raw["p90_absolute_error"],
        **contention_metrics,
        "relocations": physical_relocations,
        "obstructive_moves": target_bound_clearances,
        "illegal_drops": int(raw["illegal_drops"]),
        "invalid_assignments": invalid,
        "infeasible_epochs": infeasible,
        "fallbacks": fallbacks,
        "method_failure_reason": raw["method_failure_reason"],
        "planning_seconds": planning_seconds,
        "episode_wall_seconds": float(raw["decision_seconds"]),
        "method_audit": {
            "assignment_source": raw["assignment_source"],
            "assignment_source_version": raw["assignment_source_version"],
            "controller_action_interface": raw["controller_action_interface"],
            "controller_architecture": raw["controller_architecture"],
            "assignment_commitment": raw["assignment_commitment"],
            "reservation_integrity": raw["reservation_integrity"],
            "retrieve_relocations": target_bound_clearances,
            "completed_option_retrieve_relocations": (
                completed_option_clearances
            ),
            "retrieve_replans": int(options["retrieve_replans"]),
            "inbound_failures": int(options["inbound_failures"]),
            "retrieve_failures": int(options["retrieve_failures"]),
            "preview_failures": int(scheduler["preview_failure_count"]),
            "exact_viability_certificate": False,
        },
    }


def _summary(method: str, runs: Sequence[dict]) -> dict:
    selected = [item for item in runs if item["method"] == method]
    returns = [float(item["return"]) for item in selected]
    total_physical_relocations = sum(
        item["physical_storage_relocations"] for item in selected
    )
    total_deliveries = sum(item["delivery_count"] for item in selected)
    return {
        "method": method,
        "episodes": len(selected),
        "mean_return": _finite_mean(returns),
        "mean_legacy_rescored_return": _finite_mean(
            item["legacy_rescored_return"] for item in selected
        ),
        "mean_dense_rescored_return": _finite_mean(
            item["dense_rescored_return"] for item in selected
        ),
        "return_std": float(pstdev(returns)) if returns else None,
        "success_rate": _finite_mean(item["success"] for item in selected),
        "strict_method_success_rate": _finite_mean(
            item["strict_method_success"] for item in selected
        ),
        "mean_completion_rate": _finite_mean(
            item["completion_rate"] for item in selected
        ),
        "mean_steps": _finite_mean(item["steps"] for item in selected),
        "mean_absolute_error": _finite_mean(
            item["mean_absolute_error"] for item in selected
        ),
        "first_two_mean_absolute_error": _finite_mean(
            item["first_two_mean_absolute_error"] for item in selected
        ),
        "positions_three_plus_mean_absolute_error": _finite_mean(
            item["positions_three_plus_mean_absolute_error"]
            for item in selected
        ),
        "mean_tardiness": _finite_mean(
            item["mean_tardiness"] for item in selected
        ),
        "mean_earliness": _finite_mean(
            item["mean_earliness"] for item in selected
        ),
        "within_target_window_rate": _finite_mean(
            item["within_target_window_rate"] for item in selected
        ),
        "first_two_within_target_window_rate": _finite_mean(
            item["first_two_within_target_window_rate"]
            for item in selected
        ),
        "positions_three_plus_within_target_window_rate": _finite_mean(
            item["positions_three_plus_within_target_window_rate"]
            for item in selected
        ),
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "total_physical_storage_relocations": total_physical_relocations,
        "physical_storage_relocations_per_100_deliveries": (
            100.0 * total_physical_relocations / total_deliveries
            if total_deliveries
            else None
        ),
        "total_target_bound_obstruction_clearances": sum(
            item["target_bound_obstruction_clearances"] for item in selected
        ),
        "total_standalone_reconfigurations": sum(
            item["standalone_reconfigurations"] for item in selected
        ),
        "total_standalone_with_direct_delivery_available": sum(
            item["standalone_with_direct_delivery_available"]
            for item in selected
        ),
        "total_standalone_without_direct_delivery_available": sum(
            item["standalone_without_direct_delivery_available"]
            for item in selected
        ),
        "total_directly_deliverable_self_reconfigurations": sum(
            item["directly_deliverable_self_reconfigurations"]
            for item in selected
        ),
        # Backward-compatible aggregate aliases.
        "total_relocations": total_physical_relocations,
        "relocations_per_100_deliveries": (
            100.0 * total_physical_relocations / total_deliveries
            if total_deliveries
            else None
        ),
        "total_illegal_drops": sum(item["illegal_drops"] for item in selected),
        "total_invalid_assignments": sum(
            item["invalid_assignments"] for item in selected
        ),
        "total_infeasible_epochs": sum(
            item["infeasible_epochs"] for item in selected
        ),
        "total_fallbacks": sum(item["fallbacks"] for item in selected),
        "mean_planning_seconds": _finite_mean(
            item["planning_seconds"] for item in selected
        ),
        "method_failures": [
            {
                "instance_seed": item["instance_seed"],
                "reason": item["method_failure_reason"],
            }
            for item in selected
            if item["method_failure_reason"] is not None
        ],
    }


def _bootstrap_ci(values: Sequence[float], samples: int, rng) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return [float(low), float(high)]


PAIR_METRICS = {
    "return_advantage": ("return", "reference_minus_baseline"),
    "legacy_rescored_return_advantage": (
        "legacy_rescored_return",
        "reference_minus_baseline",
    ),
    "dense_rescored_return_advantage": (
        "dense_rescored_return",
        "reference_minus_baseline",
    ),
    "strict_success_advantage": (
        "strict_method_success",
        "reference_minus_baseline",
    ),
    "completion_advantage": ("completion_rate", "reference_minus_baseline"),
    "step_reduction": ("steps", "baseline_minus_reference"),
    "absolute_error_reduction": (
        "mean_absolute_error",
        "baseline_minus_reference",
    ),
    "first_two_absolute_error_reduction": (
        "first_two_mean_absolute_error",
        "baseline_minus_reference",
    ),
    "positions_three_plus_absolute_error_reduction": (
        "positions_three_plus_mean_absolute_error",
        "baseline_minus_reference",
    ),
    "tardiness_reduction": ("mean_tardiness", "baseline_minus_reference"),
    "within_window_advantage": (
        "within_target_window_rate",
        "reference_minus_baseline",
    ),
    "first_two_within_window_advantage": (
        "first_two_within_target_window_rate",
        "reference_minus_baseline",
    ),
    "positions_three_plus_within_window_advantage": (
        "positions_three_plus_within_target_window_rate",
        "reference_minus_baseline",
    ),
    "physical_storage_relocation_reduction": (
        "physical_storage_relocations",
        "baseline_minus_reference",
    ),
    # Compatibility contrast; numerically identical to the canonical field.
    "relocation_reduction": ("relocations", "baseline_minus_reference"),
}


def _paired_comparisons(runs: Sequence[dict], methods, samples: int) -> list[dict]:
    by_method = {
        method: {
            item["instance_id"]: item for item in runs if item["method"] == method
        }
        for method in methods
    }
    reference = by_method[VCG_METHOD]
    rng = np.random.default_rng(20260806)
    output = []
    for method in methods:
        if method == VCG_METHOD:
            continue
        shared = sorted(set(reference) & set(by_method[method]))
        metrics = {}
        for metric, (field, orientation) in PAIR_METRICS.items():
            deltas = []
            for instance_id in shared:
                left = reference[instance_id][field]
                right = by_method[method][instance_id][field]
                if left is None or right is None:
                    continue
                left = float(left)
                right = float(right)
                if not math.isfinite(left) or not math.isfinite(right):
                    continue
                delta = left - right if orientation == "reference_minus_baseline" else right - left
                deltas.append(delta)
            metrics[metric] = {
                "n": len(deltas),
                "mean": _finite_mean(deltas),
                "bootstrap_95_ci": (
                    _bootstrap_ci(deltas, samples, rng) if deltas else None
                ),
                "per_instance": deltas,
            }
        output.append(
            {
                "reference_method": VCG_METHOD,
                "baseline_method": method,
                "n": len(shared),
                "instance_ids": shared,
                "metric_orientation": "positive_favors_vcg_smdp",
                "metrics": metrics,
            }
        )
    return output


def _checkpoint_readiness(payload: Mapping) -> dict:
    state = payload.get("agent_state", {})
    training_episodes = int(
        payload.get(
            "completed_training_episodes",
            len(tuple(payload.get("train_instance_seeds", ()))),
        )
    )
    smoke = bool(
        payload.get("instance_regime") == "calibration_only"
        or "smoke" in str(payload.get("protocol", ""))
    )
    best_record = payload.get("best_validation_record")
    best_validation_eligible = bool(
        isinstance(best_record, Mapping)
        and best_record.get("deployment_eligible") is True
    )
    proper_training_candidate = bool(
        payload.get("proper_training_confirmation_candidate") is True
    )
    protocol_training_complete = bool(
        payload.get("protocol_training_complete") is True
    )
    selection_finalized = bool(
        payload.get("selection_finalized_after_total_episodes") is True
    )
    finalized_best_artifact = bool(
        payload.get("checkpoint_role") == "best_deployment_finalized"
        and payload.get("trainer_resumable") is False
        and payload.get("deployment_checkpoint_eligible") is True
    )
    deployment_eligible = bool(
        best_validation_eligible
        and (
            not proper_training_candidate
            or (
                protocol_training_complete
                and selection_finalized
                and finalized_best_artifact
            )
        )
    )
    mechanism_screen = bool(
        payload.get("development_mechanism_screen_only") is True
        or "factorial_development_screen"
        in str(payload.get("matrix_protocol", ""))
    )
    if smoke:
        interpretation = "plumbing_and_mechanism_diagnostic_only"
    elif mechanism_screen:
        interpretation = (
            "development_objective_screen_requires_full_three_seed_"
            "confirmation_before_sealed_test"
        )
    elif proper_training_candidate and not (
        protocol_training_complete and selection_finalized
    ):
        interpretation = "dense_proper_training_not_finalized"
    elif proper_training_candidate and not finalized_best_artifact:
        interpretation = "dense_proper_use_finalized_best_checkpoint"
    elif proper_training_candidate and not deployment_eligible:
        interpretation = "dense_proper_finalized_best_ineligible"
    elif deployment_eligible:
        interpretation = (
            "eligible_single_training_seed_requires_replication_and_sealed_test"
        )
    else:
        interpretation = "requires_external_training_seed_and_validation_audit"
    return {
        "protocol": payload.get("protocol"),
        "instance_regime": payload.get("instance_regime"),
        "training_episodes": training_episodes,
        "training_transitions": int(state.get("transition_count", 0)),
        "gradient_steps": int(state.get("gradient_steps", 0)),
        "target_updates": int(state.get("target_updates", 0)),
        "saved_epsilon": state.get("epsilon"),
        "calibration_smoke_checkpoint": smoke,
        "development_mechanism_screen_checkpoint": mechanism_screen,
        "proper_training_confirmation_candidate": proper_training_candidate,
        "protocol_training_complete": protocol_training_complete,
        "selection_finalized_after_total_episodes": selection_finalized,
        "finalized_best_deployment_artifact": finalized_best_artifact,
        "best_validation_deployment_eligible": best_validation_eligible,
        "deployment_checkpoint_eligible": deployment_eligible,
        "performance_claim_authorized": False,
        "interpretation": interpretation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched complete-system VCG versus online baseline comparison"
    )
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--instances-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--lookahead-margin-steps", type=float, default=2.0)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument("--dense-b", type=float, default=DEFAULT_DENSE_B)
    parser.add_argument("--lambda-abs", type=float, default=DEFAULT_LAMBDA_ABS)
    parser.add_argument(
        "--lambda-outside", type=float, default=DEFAULT_LAMBDA_OUTSIDE
    )
    parser.add_argument(
        "--dense-window", type=float, default=DEFAULT_DENSE_WINDOW
    )
    parser.add_argument("--rolling-population", type=int, default=16)
    parser.add_argument("--rolling-generations", type=int, default=10)
    parser.add_argument("--rolling-ga-egress-weight", type=int, default=4)
    parser.add_argument("--ga-seed-base", type=int, default=310_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args) -> None:
    args.methods = tuple(dict.fromkeys(args.methods))
    args.seeds = tuple(dict.fromkeys(args.seeds))
    if VCG_METHOD not in args.methods or len(args.methods) < 2:
        raise ValueError("methods must include vcg_smdp and at least one baseline")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if args.max_steps <= 0 or args.max_defer_steps <= 0:
        raise ValueError("step limits must be positive")
    if args.rolling_population < 3 or args.rolling_generations <= 0:
        raise ValueError("rolling GA population/generations are invalid")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not math.isfinite(args.lookahead_margin_steps) or args.lookahead_margin_steps < 0:
        raise ValueError("lookahead margin must be finite and non-negative")
    for name in ("dense_b", "lambda_abs", "lambda_outside", "dense_window"):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"{name} must be finite")
    if args.lambda_abs < 0 or args.lambda_outside < 0 or args.dense_window <= 0:
        raise ValueError("dense timing slopes must be non-negative and window positive")
    sealed = set(args.seeds).intersection(
        SEALED_STRESS_V1_HOLDOUT_SEEDS | SEALED_IN_REGIME_TEST_SEEDS
    )
    if sealed:
        raise ValueError(f"development runner refuses sealed test seeds: {sorted(sealed)}")


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    results_path = args.output_dir / "comparison-results.csv"
    summary_path = args.output_dir / "comparison-summary.json"
    audit_path = args.output_dir / "comparison-audit.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists; use --overwrite or a new directory")

    device = resolve_device(args.device)
    controller_payload = _load_controller_checkpoint(args.controller_checkpoint)
    timing_spec = TimingObjectiveSpec.dense(
        dense_b=args.dense_b,
        lambda_abs=args.lambda_abs,
        lambda_outside=args.lambda_outside,
        window=args.dense_window,
    )
    authenticated_horizon = controller_payload.get("episode_horizon_steps")
    if authenticated_horizon is not None and int(args.max_steps) != int(
        authenticated_horizon
    ):
        raise ValueError(
            "--max-steps must equal the episodic checkpoint's authenticated "
            f"horizon ({authenticated_horizon})"
        )
    training_objective = controller_payload.get("timing_objective")
    resume_contract = controller_payload.get("resume_contract", {})
    authenticated_spec_payload = (
        resume_contract.get("objective_spec")
        if isinstance(resume_contract, Mapping)
        else None
    )
    if isinstance(authenticated_spec_payload, Mapping):
        authenticated_spec = TimingObjectiveSpec.from_dict(
            authenticated_spec_payload
        )
        if (
            training_objective == DENSE_PIECEWISE
            and authenticated_spec.to_dict() != timing_spec.to_dict()
        ):
            raise ValueError(
                "dense rescore coefficients do not match the checkpoint's "
                "authenticated training objective"
            )
    search_config = _search_config(controller_payload)
    liveness_rule = _liveness_rule(controller_payload)
    environment = controller_payload["environment"]
    block_count = int(environment["number_blocks"])
    instances = _load_instances(args, controller_payload)
    baseline_args = _baseline_runtime_args(args, controller_payload, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances_output = args.output_dir / "instances"
    instances_output.mkdir(parents=True, exist_ok=True)
    for seed, instance in instances.items():
        (instances_output / f"seed-{seed}.json").write_text(
            instance.to_json() + "\n", encoding="utf-8"
        )

    runs = []
    for seed in args.seeds:
        instance = instances[int(seed)]
        for method in args.methods:
            if method == VCG_METHOD:
                raw = run_arm(
                    arm=EXACT_FULL,
                    controller_payload=controller_payload,
                    instance=instance,
                    instance_seed=int(seed),
                    search_config=search_config,
                    liveness_rule=liveness_rule,
                    prioritizer=None,
                    max_steps=args.max_steps,
                    device=device,
                )
                run = _normalize_vcg(
                    raw, instance, block_count, timing_spec
                )
            else:
                raw = evaluate_assignment_ablation_one(
                    baseline_args,
                    int(seed),
                    None,
                    assignment_source=METHOD_TO_SOURCE[method],
                    episode_instance=instance,
                )
                run = _normalize_baseline(
                    method, raw, instance, block_count, timing_spec
                )
            if run["instance_id"] != instance.instance_id:
                raise RuntimeError(f"{method} did not consume the paired instance")
            if run["schedule_id"] != instance.schedule_id:
                raise RuntimeError(f"{method} schedule provenance mismatch")
            runs.append(run)
            print(
                f"[{method}] seed={seed} LegacyR={run['return']:.2f} "
                f"DenseR={run['dense_rescored_return']:.2f} "
                f"strict={int(run['strict_method_success'])} "
                f"steps={run['steps']} MAE={run['mean_absolute_error']} "
                f"First2={run['first_two_mean_absolute_error']} "
                f"physical_reloc={run['physical_storage_relocations']} "
                f"failure={run['method_failure_reason']}",
                flush=True,
            )

    expected = len(args.methods) * len(args.seeds)
    observed = {(item["method"], item["instance_id"]) for item in runs}
    if len(runs) != expected or len(observed) != expected:
        raise RuntimeError("method/instance comparison grid is incomplete")

    summaries = [_summary(method, runs) for method in args.methods]
    paired = _paired_comparisons(
        runs, args.methods, samples=args.bootstrap_samples
    )
    readiness = _checkpoint_readiness(controller_payload)
    result = {
        "protocol": PROTOCOL,
        "comparison_scope": "complete_system_not_assignment_only",
        "claim_status": readiness["interpretation"],
        "performance_claim_authorized": readiness[
            "performance_claim_authorized"
        ],
        "pairing_key": "EpisodeInstance.instance_id_and_schedule_id",
        "information_regime": "online_arrived_only",
        "vcg_frontier": "canonical_complete_exact_full_critic_disabled",
        "baseline_scheduler": (
            "source_neutral_duration_aware_reserved_cell_atomic_scheduler"
        ),
        "macro_count_comparable": False,
        "primitive_return_steps_timing_relocations_comparable": True,
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "common_relocation_estimand": (
            "completed_physical_storage_to_storage_moves"
        ),
        "mechanism_decomposition_interpretation": (
            "target_bound_retrieval_executor_clearances_and_standalone_vcg_"
            "reconfigurations_are_reported_separately"
        ),
        "rollout_reward": "legacy_clipped_v1",
        "dual_rescore_fixed_realized_trajectory": True,
        "timing_objective_rescore": timing_spec.to_dict(),
        "controller_training_objective": training_objective,
        "controller_checkpoint": str(args.controller_checkpoint.resolve()),
        "controller_checkpoint_sha256": _sha256_file(
            args.controller_checkpoint
        ),
        "checkpoint_readiness": readiness,
        "environment": dict(environment),
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "instance_ids": {
            str(seed): instances[int(seed)].instance_id for seed in args.seeds
        },
        "schedule_ids": {
            str(seed): instances[int(seed)].schedule_id for seed in args.seeds
        },
        "max_steps": args.max_steps,
        "device": str(device),
        "rolling_ga": {
            "population": args.rolling_population,
            "generations": args.rolling_generations,
            "ga_seed_base": args.ga_seed_base,
            "egress_weight": args.rolling_ga_egress_weight,
        },
        "summaries": summaries,
        "paired_comparisons": paired,
        "runs": [
            {key: value for key, value in run.items() if key != "method_audit"}
            for run in runs
        ],
    }

    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {name: _csv_value(run[name]) for name in RESULT_FIELDS}
            )
    summary_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "protocol": PROTOCOL,
                    "controller_checkpoint_sha256": result[
                        "controller_checkpoint_sha256"
                    ],
                    "checkpoint_readiness": readiness,
                    "runs": runs,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summaries), indent=2), flush=True)
    print(f"Results: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Audit: {audit_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
