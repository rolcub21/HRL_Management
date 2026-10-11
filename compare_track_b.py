#!/usr/bin/env python3
"""Paired complete-system comparison for Track B."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from example.helper.timing_metrics import (
    STORAGE_FLOW_METRIC_CONTRACT,
    summarize_delivery_timing,
    summarize_storage_flow_runs,
)
from example.small_rooms_env import SmallRoomsEnv
from example.urgency_scheduler import (
    DURATION_AWARE_ARCHITECTURE,
    DURATION_AWARE_METHOD,
    DURATION_AWARE_POLICY,
    URGENCY_FIRST_ACTION_INTERFACE,
    URGENCY_FIRST_ARCHITECTURE,
    URGENCY_FIRST_POLICY,
)
from gated_agent import (
    ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
    FOUNDATION_CONTROLLER_ARCHITECTURE,
    MODE_ONLY_CONTROLLER_ARCHITECTURE,
    SCHEDULER_CONTROLLER_ARCHITECTURE,
    SUPPORTED_CONTROLLER_ARCHITECTURES,
)
from PSLAP.ga_optimizer import GAConfig
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.ga_policy import DEFAULT_ROLLING_GA_CONFIG
from PSLAP.track_a import (
    TRACK_A_DYNAMIC,
    TRACK_A_GA_OFFLINE,
    TRACK_A_GA_ROLLING,
    TRACK_A_INFORMATION,
    TRACK_A_NEAREST_FREE,
    run_track_a_episode,
)
from track_b_evaluate import METHOD as LEARNED_METHOD
from track_b_evaluate import evaluate_one as evaluate_learned
from track_b_urgency_evaluate import METHOD as URGENCY_METHOD
from track_b_urgency_evaluate import (
    evaluate_duration_aware_one as evaluate_duration_aware,
)
from track_b_urgency_evaluate import evaluate_one as evaluate_urgency


TRACK = "B_complete_system"
ONLINE_PRIMARY = "online_primary"
OFFLINE_REFERENCE = "offline_reference"
BASELINE_METHODS = (
    TRACK_A_NEAREST_FREE,
    TRACK_A_DYNAMIC,
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_OFFLINE,
)
METHODS = (
    LEARNED_METHOD,
    URGENCY_METHOD,
    DURATION_AWARE_METHOD,
) + BASELINE_METHODS
DEFAULT_METHODS = (LEARNED_METHOD, URGENCY_METHOD) + BASELINE_METHODS

FIELDNAMES = [
    "track",
    "comparison_group",
    "method",
    "method_family",
    "information_regime",
    "lambda",
    "mu",
    "episode_seed",
    "ga_seed",
    "instance_id",
    "checkpoint_id",
    "checkpoint_episode",
    "controller_architecture",
    "policy_realization",
    "return",
    "success",
    "timeout",
    "completion_fraction",
    "steps",
    "delivery_count",
    "target_window",
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "tardy_delivery_rate",
    "mean_tardiness_when_tardy",
    "p90_tardiness",
    "p90_absolute_error",
    "storage_flow_metric_contract",
    "storage_flow_observation_end_step",
    "storage_flow_manifest_count",
    "storage_flow_arrived_count",
    "storage_flow_completed_count",
    "storage_flow_right_censored_count",
    "storage_flow_not_yet_arrived_count",
    "storage_flow_unfinished_count",
    "storage_flow_completion_rate",
    "storage_flow_arrived_completion_rate",
    "storage_flow_fully_observed",
    "mean_storage_flow_time",
    "median_storage_flow_time",
    "p90_storage_flow_time",
    "p95_storage_flow_time",
    "max_storage_flow_time",
    "completed_storage_flow_times",
    "right_censored_storage_flow_ages",
    "obstructive_moves",
    "illegal_drops",
    "assignment_decision_count",
    "valid_assignment_count",
    "invalid_assignment_count",
    "infeasible_epoch_count",
    "fallback_count",
    "fallback_contaminated",
    "strict_method_success",
    "source_setup_seconds",
    "online_assignment_seconds",
    "episode_loop_seconds",
    "total_method_seconds",
    "controller_option_decisions",
    "controller_primitive_decisions",
    "mean_option_probability",
    "delivery_deviations",
    "compact_audit",
]


STORAGE_FLOW_SCALAR_FIELDS = (
    "storage_flow_metric_contract",
    "storage_flow_observation_end_step",
    "storage_flow_manifest_count",
    "storage_flow_arrived_count",
    "storage_flow_completed_count",
    "storage_flow_right_censored_count",
    "storage_flow_not_yet_arrived_count",
    "storage_flow_unfinished_count",
    "storage_flow_completion_rate",
    "storage_flow_arrived_completion_rate",
    "storage_flow_fully_observed",
    "mean_storage_flow_time",
    "median_storage_flow_time",
    "p90_storage_flow_time",
    "p95_storage_flow_time",
    "max_storage_flow_time",
)


def format_storage_flow_fields(result):
    """Extract the common operational storage-flow schema for one CSV row."""

    fields = {name: result[name] for name in STORAGE_FLOW_SCALAR_FIELDS}
    for name in (
        "completed_storage_flow_times",
        "right_censored_storage_flow_ages",
    ):
        fields[name] = json.dumps(result[name], separators=(",", ":"))
    return fields


def make_env(lam, mu):
    return SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=lam,
        proc_mean=mu,
    )


def comparison_group(information_regime):
    return (
        OFFLINE_REFERENCE
        if information_regime == "offline_full_schedule"
        else ONLINE_PRIMARY
    )


def checkpoint_components(args):
    controller = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    architecture = controller.get("controller_architecture")
    if architecture not in SUPPORTED_CONTROLLER_ARCHITECTURES:
        raise ValueError(
            "Track B comparison requires a verified production gated checkpoint"
        )
    if architecture == ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE:
        if (
            controller.get("controller_backup")
            != "masked_macro_smdp_optimality_v1"
            or controller.get("within_mode_aggregation") != "masked_max"
            or controller.get("mode_aggregation")
            != "single_macro_mode_identity_v1"
        ):
            raise ValueError(
                "Atomic scheduler checkpoint aggregation metadata is invalid"
            )
    elif architecture in (
        MODE_ONLY_CONTROLLER_ARCHITECTURE,
        FOUNDATION_CONTROLLER_ARCHITECTURE,
        SCHEDULER_CONTROLLER_ARCHITECTURE,
    ) and (
        controller.get("within_mode_aggregation") != "masked_max"
        or controller.get("mode_aggregation") != "kl_regularized"
    ):
        raise ValueError("Mode-only checkpoint aggregation metadata is invalid")
    if controller.get("training_lambda") != args.lam or controller.get(
        "training_mu"
    ) != args.mu:
        raise ValueError(
            "Controller training regime does not match comparison regime"
        )
    if not controller.get("selector_frozen", False):
        raise ValueError("Expected a checkpoint trained with frozen REG-v5")
    if args.selector_checkpoint:
        selector = torch.load(
            args.selector_checkpoint, map_location="cpu", weights_only=False
        )
    else:
        selector = controller.get("selector_checkpoint")
    if selector is None:
        raise ValueError(
            "Checkpoint has no nested selector; provide --selector-checkpoint"
        )
    if selector.get("selector_feature_version") != 5:
        raise ValueError("Track B comparison requires REG-v5 selector features")
    if selector.get("training_lambda") != args.lam or selector.get(
        "training_mu"
    ) != args.mu:
        raise ValueError("Selector training regime does not match comparison")
    expected_selector_digest = controller.get("selector_deployment_digest")
    if (
        expected_selector_digest is not None
        and selector_deployment_digest(selector) != expected_selector_digest
    ):
        raise ValueError(
            "Selector checkpoint differs from the controller's frozen deployment"
        )
    return controller, selector


def learned_args(args, controller):
    mode_only = controller.get("controller_architecture") in {
        ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
        MODE_ONLY_CONTROLLER_ARCHITECTURE,
        FOUNDATION_CONTROLLER_ARCHITECTURE,
        SCHEDULER_CONTROLLER_ARCHITECTURE,
    }
    return SimpleNamespace(
        lam=args.lam,
        mu=args.mu,
        instance=None,
        device=args.device,
        policy=args.policy
        or controller.get(
            "validation_policy",
            "map"
            if controller.get("controller_architecture")
            == ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE
            else "mode_regularized"
            if mode_only
            else "regularized",
        ),
        tau_option=float(controller.get("tau_option", 0.1)),
        tau_primitive=float(controller.get("tau_primitive", 0.1)),
        tau_mode=float(controller["tau_mode"]),
        allow_legacy_controller=False,
        max_steps=args.max_steps,
        target_window=args.target_window,
        save_instances_dir=None,
    )


def urgency_args(args, controller):
    return SimpleNamespace(
        lam=args.lam,
        mu=args.mu,
        instance=None,
        device=args.device,
        max_steps=args.max_steps,
        max_defer_steps=int(controller.get("max_defer_steps", 10)),
        lookahead_margin_steps=args.lookahead_margin_steps,
        target_window=args.target_window,
        save_instances_dir=None,
    )


def format_learned_row(run, args, controller):
    selector = run["selector_audit"]
    gate = run["gate_audit"]
    decisions = run["controller_decisions"]
    valid = int(selector["valid_assignment_count"])
    invalid = int(selector["invalid_assignment_count"])
    infeasible = int(selector["infeasible_epoch_count"])
    loop_seconds = float(run["decision_seconds"])
    return {
        "track": TRACK,
        "comparison_group": ONLINE_PRIMARY,
        "method": LEARNED_METHOD,
        "method_family": "learned_gated_hrl",
        "information_regime": "online",
        "lambda": args.lam,
        "mu": args.mu,
        "episode_seed": run["eval_seed"],
        "ga_seed": "not_applicable",
        "instance_id": run["instance_id"],
        "checkpoint_id": Path(args.checkpoint).name,
        "checkpoint_episode": controller.get("completed_training_episodes"),
        "controller_architecture": controller["controller_architecture"],
        "policy_realization": args.policy
        or controller.get("validation_policy", "mode_regularized"),
        "return": run["return"],
        "success": run["success"],
        "timeout": int(not run["success"] and run["steps"] >= args.max_steps),
        "completion_fraction": run["delivery_count"]
        / float(controller.get("block_count", 40)),
        "steps": run["steps"],
        "delivery_count": run["delivery_count"],
        "target_window": args.target_window,
        "mean_signed_deviation": run["mean_signed_deviation"],
        "mean_absolute_error": run["mean_absolute_error"],
        "mean_tardiness": run["mean_tardiness"],
        "mean_earliness": run["mean_earliness"],
        "within_target_window_rate": run["within_target_window_rate"],
        "tardy_delivery_rate": run["tardy_delivery_rate"],
        "mean_tardiness_when_tardy": run["mean_tardiness_when_tardy"],
        "p90_tardiness": run["p90_tardiness"],
        "p90_absolute_error": run["p90_absolute_error"],
        **format_storage_flow_fields(run),
        "obstructive_moves": run["obstructive_moves"],
        "illegal_drops": run["illegal_drops"],
        "assignment_decision_count": selector["decision_count"],
        "valid_assignment_count": valid,
        "invalid_assignment_count": invalid,
        "infeasible_epoch_count": infeasible,
        "fallback_count": 0,
        "fallback_contaminated": 0,
        # Match Track A's strict protocol: infeasible decision epochs are an
        # environment condition, whereas invalid assignments are method
        # failures.  A complete episode with no invalid proposal passes.
        "strict_method_success": float(bool(run["success"]) and invalid == 0),
        "source_setup_seconds": 0.0,
        "online_assignment_seconds": selector["assignment_seconds"],
        "episode_loop_seconds": loop_seconds,
        "total_method_seconds": loop_seconds,
        "controller_option_decisions": decisions.get("option", 0),
        "controller_primitive_decisions": decisions.get("primitive", 0),
        "mean_option_probability": gate.get("mean_option_probability"),
        "delivery_deviations": json.dumps(
            run["delivery_deviations"], separators=(",", ":")
        ),
        "compact_audit": json.dumps(
            {
                "gate": gate,
                "controller_observation_version": run.get(
                    "controller_observation_version"
                ),
                "control_decisions": run.get(
                    "controller_control_decisions", {}
                ),
                "scheduler": run.get("scheduler_audit", {}),
                "selector": {
                    key: value
                    for key, value in selector.items()
                    if key != "decisions"
                },
            },
            separators=(",", ":"),
        ),
    }


def format_urgency_row(run, args, controller, selector_payload):
    """Map the no-training scheduler into the shared complete-system row."""

    selector = run["selector_audit"]
    scheduler = run["urgency_scheduler_audit"]
    option_audit = run["scheduler_audit"]
    valid = int(selector["valid_assignment_count"])
    invalid = int(selector["invalid_assignment_count"])
    infeasible = int(selector["infeasible_epoch_count"])
    loop_seconds = float(run["decision_seconds"])
    block_count = int(controller.get("block_count", 40))
    selector_digest = run["selector_deployment_digest"]
    return {
        "track": TRACK,
        "comparison_group": ONLINE_PRIMARY,
        "method": run["method"],
        "method_family": "learning_augmented_deterministic_scheduler",
        "information_regime": "online_arrived_only",
        "lambda": args.lam,
        "mu": args.mu,
        "episode_seed": run["eval_seed"],
        "ga_seed": "not_applicable",
        "instance_id": run["instance_id"],
        "checkpoint_id": f"reg-v5:{selector_digest[:16]}",
        "checkpoint_episode": selector_payload.get(
            "completed_training_episodes", "not_recorded"
        ),
        "controller_architecture": run["controller_architecture"],
        "policy_realization": run["policy_realization"],
        "return": run["return"],
        "success": run["success"],
        "timeout": int(run["truncated"]),
        "completion_fraction": run["delivery_count"] / float(block_count),
        "steps": run["steps"],
        "delivery_count": run["delivery_count"],
        "target_window": args.target_window,
        "mean_signed_deviation": run["mean_signed_deviation"],
        "mean_absolute_error": run["mean_absolute_error"],
        "mean_tardiness": run["mean_tardiness"],
        "mean_earliness": run["mean_earliness"],
        "within_target_window_rate": run["within_target_window_rate"],
        "tardy_delivery_rate": run["tardy_delivery_rate"],
        "mean_tardiness_when_tardy": run[
            "mean_tardiness_when_tardy"
        ],
        "p90_tardiness": run["p90_tardiness"],
        "p90_absolute_error": run["p90_absolute_error"],
        **format_storage_flow_fields(run),
        "obstructive_moves": run["obstructive_moves"],
        "illegal_drops": run["illegal_drops"],
        "assignment_decision_count": selector["decision_count"],
        "valid_assignment_count": valid,
        "invalid_assignment_count": invalid,
        "infeasible_epoch_count": infeasible,
        "fallback_count": 0,
        "fallback_contaminated": 0,
        "strict_method_success": run["strict_method_success"],
        # Model construction is excluded just as it is for the learned
        # controller; online inference remains inside the episode loop.
        "source_setup_seconds": 0.0,
        "online_assignment_seconds": (
            selector["assignment_seconds"]
            + float(scheduler.get("preview_seconds", 0.0))
        ),
        "episode_loop_seconds": loop_seconds,
        "total_method_seconds": loop_seconds,
        "controller_option_decisions": scheduler["decision_count"],
        "controller_primitive_decisions": 0,
        "mean_option_probability": "not_applicable",
        "delivery_deviations": json.dumps(
            run["delivery_deviations"], separators=(",", ":")
        ),
        "compact_audit": json.dumps(
            {
                "method_failure_reason": run["method_failure_reason"],
                "urgency_scheduler": {
                    key: value
                    for key, value in scheduler.items()
                    if key != "decisions"
                },
                "atomic_options": option_audit,
                "control_decisions": run.get(
                    "controller_control_decisions", {}
                ),
                "selector": {
                    key: value
                    for key, value in selector.items()
                    if key != "decisions"
                },
            },
            separators=(",", ":"),
        ),
    }


def ga_configs(args, seed):
    ga_seed = args.ga_seed_base + seed
    offline = GAConfig(
        population_size=args.offline_population,
        generations=args.offline_generations,
        elite_count=min(2, args.offline_population - 1),
        tournament_size=min(3, args.offline_population),
        seed=ga_seed,
    )
    rolling = replace(
        DEFAULT_ROLLING_GA_CONFIG,
        population_size=args.rolling_population,
        generations=args.rolling_generations,
        elite_count=min(2, args.rolling_population - 1),
        tournament_size=min(3, args.rolling_population),
        seed=ga_seed,
    )
    offline.validate()
    rolling.validate()
    return ga_seed, offline, rolling


def format_baseline_row(method, result, args, seed, ga_seed):
    timing = summarize_delivery_timing(
        result["delivery_errors"], args.target_window
    )
    information = TRACK_A_INFORMATION[method]
    source_seconds = float(result["source_setup_seconds"])
    assignment_seconds = float(result["assignment_planning_seconds"])
    loop_seconds = float(result["episode_loop_seconds"])
    return {
        "track": TRACK,
        "comparison_group": comparison_group(information),
        "method": method,
        "method_family": "deterministic_pslap_executor",
        "information_regime": information,
        "lambda": args.lam,
        "mu": args.mu,
        "episode_seed": seed,
        "ga_seed": ga_seed
        if method in (TRACK_A_GA_ROLLING, TRACK_A_GA_OFFLINE)
        else "not_applicable",
        "instance_id": result["instance_id"],
        "checkpoint_id": "not_applicable",
        "checkpoint_episode": "not_applicable",
        "controller_architecture": "not_applicable",
        "policy_realization": "deterministic",
        "return": result["return"],
        "success": result["success"],
        "timeout": int(
            not result["success"] and result["steps"] >= args.max_steps
        ),
        "completion_fraction": result["completion_fraction"],
        "steps": result["steps"],
        "delivery_count": result["delivery_count"],
        "target_window": args.target_window,
        "mean_signed_deviation": timing["mean_signed_deviation"],
        "mean_absolute_error": timing["mean_absolute_error"],
        "mean_tardiness": timing["mean_tardiness"],
        "mean_earliness": timing["mean_earliness"],
        "within_target_window_rate": timing["within_target_window_rate"],
        "tardy_delivery_rate": timing["tardy_delivery_rate"],
        "mean_tardiness_when_tardy": timing["mean_tardiness_when_tardy"],
        "p90_tardiness": timing["p90_tardiness"],
        "p90_absolute_error": timing["p90_absolute_error"],
        **format_storage_flow_fields(result),
        "obstructive_moves": result["obstructive_moves"],
        "illegal_drops": result["illegal_drops"],
        "assignment_decision_count": result["assignment_decision_count"],
        "valid_assignment_count": result["valid_assignment_count"],
        "invalid_assignment_count": result["invalid_assignment_count"],
        "infeasible_epoch_count": result["infeasible_epoch_count"],
        "fallback_count": result["fallback_count"],
        "fallback_contaminated": result["fallback_contaminated"],
        "strict_method_success": result["strict_method_success"],
        "source_setup_seconds": source_seconds,
        "online_assignment_seconds": assignment_seconds,
        "episode_loop_seconds": loop_seconds,
        "total_method_seconds": source_seconds + loop_seconds,
        "controller_option_decisions": "not_applicable",
        "controller_primitive_decisions": "not_applicable",
        "mean_option_probability": "not_applicable",
        "delivery_deviations": json.dumps(
            result["delivery_errors"], separators=(",", ":")
        ),
        "compact_audit": json.dumps(
            result["assignment_decisions_compact"], separators=(",", ":")
        ),
    }


def evaluate_baseline(method, args, seed, instance):
    random.seed(seed + 10_000)
    np.random.seed(seed + 20_000)
    env = make_env(args.lam, args.mu)
    ga_seed, offline, rolling = ga_configs(args, seed)
    result = run_track_a_episode(
        env,
        method,
        max_steps=args.max_steps,
        episode_instance=instance,
        offline_config=offline,
        rolling_config=rolling,
    )
    row = format_baseline_row(method, result, args, seed, ga_seed)
    return row, result


def finite(values):
    result = []
    for value in values:
        if value in (None, ""):
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            result.append(numeric)
    return np.asarray(result, dtype=float)


def optional_mean(values):
    values = finite(values)
    return float(values.mean()) if len(values) else None


def pooled_storage_flow_summary(rows):
    """Pool block-level flow times only when every manifest is complete."""

    normalized = []
    for row in rows:
        item = dict(row)
        raw = item["completed_storage_flow_times"]
        item["completed_storage_flow_times"] = (
            json.loads(raw) if isinstance(raw, str) else raw
        )
        normalized.append(item)
    return summarize_storage_flow_runs(normalized)


def method_summary(method, rows):
    selected = [row for row in rows if row["method"] == method]
    returns = finite([row["return"] for row in selected])
    return {
        "method": method,
        "comparison_group": selected[0]["comparison_group"],
        "information_regime": selected[0]["information_regime"],
        "controller_architecture": selected[0]["controller_architecture"],
        "episodes": len(selected),
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "success_rate": float(np.mean([row["success"] for row in selected])),
        "strict_method_success_rate": float(
            np.mean([row["strict_method_success"] for row in selected])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in selected])),
        "mean_absolute_error": optional_mean(
            [row["mean_absolute_error"] for row in selected]
        ),
        "mean_tardiness": optional_mean(
            [row["mean_tardiness"] for row in selected]
        ),
        "within_target_window_rate": optional_mean(
            [row["within_target_window_rate"] for row in selected]
        ),
        **pooled_storage_flow_summary(selected),
        "total_invalid_assignments": int(
            sum(row["invalid_assignment_count"] for row in selected)
        ),
        "total_fallbacks": int(sum(row["fallback_count"] for row in selected)),
        "mean_total_method_seconds": float(
            np.mean([row["total_method_seconds"] for row in selected])
        ),
    }


def bootstrap_ci(values, samples, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return [float(low), float(high)]


def paired_summary(
    rows,
    methods,
    samples,
    *,
    reference_method=LEARNED_METHOD,
):
    by_method = {
        method: {row["instance_id"]: row for row in rows if row["method"] == method}
        for method in methods
    }
    reference = by_method.get(reference_method, {})
    result = []
    rng = np.random.default_rng(20260804)
    for method in methods:
        if method == reference_method:
            continue
        shared = sorted(set(reference) & set(by_method[method]))
        if not shared:
            continue
        specifications = {
            "return_advantage": ("return", "reference_minus_comparison"),
            "step_reduction": ("steps", "comparison_minus_reference"),
            "absolute_error_reduction": (
                "mean_absolute_error",
                "comparison_minus_reference",
            ),
            "tardiness_reduction": (
                "mean_tardiness",
                "comparison_minus_reference",
            ),
            "success_advantage": (
                "success",
                "reference_minus_comparison",
            ),
            "storage_flow_completion_advantage": (
                "storage_flow_completion_rate",
                "reference_minus_comparison",
            ),
            "unfinished_storage_flow_reduction": (
                "storage_flow_unfinished_count",
                "comparison_minus_reference",
            ),
            "mean_storage_flow_time_reduction": (
                "mean_storage_flow_time",
                "comparison_minus_reference",
            ),
            "median_storage_flow_time_reduction": (
                "median_storage_flow_time",
                "comparison_minus_reference",
            ),
            "p90_storage_flow_time_reduction": (
                "p90_storage_flow_time",
                "comparison_minus_reference",
            ),
            "p95_storage_flow_time_reduction": (
                "p95_storage_flow_time",
                "comparison_minus_reference",
            ),
            "max_storage_flow_time_reduction": (
                "max_storage_flow_time",
                "comparison_minus_reference",
            ),
        }
        metrics = {}
        for name, (field, orientation) in specifications.items():
            values = []
            metric_instances = []
            for key in shared:
                reference_value = reference[key][field]
                baseline_value = by_method[method][key][field]
                if reference_value is None or baseline_value is None:
                    continue
                reference_value = float(reference_value)
                baseline_value = float(baseline_value)
                if not (
                    np.isfinite(reference_value)
                    and np.isfinite(baseline_value)
                ):
                    continue
                if orientation == "reference_minus_comparison":
                    values.append(reference_value - baseline_value)
                else:
                    values.append(baseline_value - reference_value)
                metric_instances.append(key)
            if not values:
                metrics[name] = {
                    "n": 0,
                    "mean": None,
                    "bootstrap_95_ci": None,
                    "instance_ids": [],
                    "per_instance": [],
                }
                continue
            metrics[name] = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "bootstrap_95_ci": bootstrap_ci(values, samples, rng),
                "instance_ids": metric_instances,
                "per_instance": [float(value) for value in values],
            }
        entry = {
            "reference_method": reference_method,
            "comparison_method": method,
            "comparison_group": by_method[method][shared[0]][
                "comparison_group"
            ],
            "n": len(shared),
            "instance_ids": shared,
            "metric_orientation": (
                "positive favors the reference method for every "
                "reported metric"
            ),
            "metrics": metrics,
        }
        if reference_method == LEARNED_METHOD:
            # Backward-compatible alias for existing result consumers.
            entry["learned_method"] = LEARNED_METHOD
        result.append(entry)
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[30000, 30001, 30002, 30003, 30004]
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--policy", choices=("map", "mode_regularized", "regularized")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--lookahead-margin-steps", type=float, default=0.0
    )
    parser.add_argument("--ga-seed-base", type=int, default=310000)
    parser.add_argument("--rolling-population", type=int, default=16)
    parser.add_argument("--rolling-generations", type=int, default=10)
    parser.add_argument("--offline-population", type=int, default=30)
    parser.add_argument("--offline-generations", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.methods = tuple(dict.fromkeys(args.methods))
    if LEARNED_METHOD not in args.methods:
        parser.error(f"--methods must include {LEARNED_METHOD}")
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    if (
        not np.isfinite(args.lookahead_margin_steps)
        or args.lookahead_margin_steps < 0.0
    ):
        parser.error("lookahead-margin-steps must be finite and nonnegative")
    for name in ("rolling_population", "offline_population"):
        if getattr(args, name) < 3:
            parser.error(f"{name.replace('_', '-')} must be at least 3")
    return args


def main():
    args = parse_args()
    results_path = args.output_dir / "track-b-results.csv"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{results_path} already exists; choose a new directory or --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = args.output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    controller, selector = checkpoint_components(args)
    runtime_args = learned_args(args, controller)
    runtime_urgency_args = urgency_args(args, controller)
    runtime_duration_args = urgency_args(args, controller)
    instances = {}
    for seed in args.seeds:
        instance = make_env(args.lam, args.mu).sample_episode_instance(seed)
        instances[seed] = instance
        (instances_dir / f"seed-{seed}.json").write_text(
            instance.to_json() + "\n"
        )

    rows = []
    audits = []
    for seed in args.seeds:
        instance = instances[seed]
        for method in args.methods:
            if method == LEARNED_METHOD:
                run = evaluate_learned(
                    runtime_args,
                    seed,
                    controller,
                    selector,
                    episode_instance=instance,
                )
                row = format_learned_row(run, args, controller)
                audit = run
            elif method == URGENCY_METHOD:
                run = evaluate_urgency(
                    runtime_urgency_args,
                    seed,
                    selector,
                    episode_instance=instance,
                )
                row = format_urgency_row(
                    run, args, controller, selector
                )
                audit = run
            elif method == DURATION_AWARE_METHOD:
                run = evaluate_duration_aware(
                    runtime_duration_args,
                    seed,
                    selector,
                    episode_instance=instance,
                )
                row = format_urgency_row(
                    run, args, controller, selector
                )
                audit = run
            else:
                row, audit = evaluate_baseline(method, args, seed, instance)
            if row["instance_id"] != instance.instance_id:
                raise RuntimeError("method did not consume the matched instance")
            rows.append(row)
            audits.append(
                {
                    "method": method,
                    "episode_seed": seed,
                    "instance_id": instance.instance_id,
                    "result": audit,
                }
            )
            print(
                f"[{method}] seed={seed} R={float(row['return']):.2f} "
                f"success={float(row['success']):.0f} steps={row['steps']} "
                f"invalid={row['invalid_assignment_count']} "
                f"fallback={row['fallback_count']}",
                flush=True,
            )

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summaries = [method_summary(method, rows) for method in args.methods]
    payload = json_safe(
        {
            "protocol": {
                "track": TRACK,
                "lambda": args.lam,
                "mu": args.mu,
                "seeds": args.seeds,
                "methods": args.methods,
                "max_steps": args.max_steps,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_episode": controller.get(
                    "completed_training_episodes"
                ),
                "controller_architecture": controller[
                    "controller_architecture"
                ],
                "controller_action_interface": controller.get(
                    "controller_action_interface",
                    "legacy_split_retrieval_v1",
                ),
                "learned_policy": runtime_args.policy,
                "selector_deployment_digest": selector_deployment_digest(
                    selector
                ),
                "urgency_scheduler": (
                    {
                        "method": URGENCY_METHOD,
                        "controller_architecture": URGENCY_FIRST_ARCHITECTURE,
                        "controller_action_interface": (
                            URGENCY_FIRST_ACTION_INTERFACE
                        ),
                        "policy_realization": URGENCY_FIRST_POLICY,
                        "selector_frozen": True,
                        "comparison_intervention": (
                            "urgency_policy_plus_complete_live_retrieval_start_v3"
                        ),
                        "learned_retrieval_contract": (
                            controller.get("retrieval_executor_version")
                        ),
                        "candidate_retrieval_contract": (
                            "named_atomic_retrieval_executor_v3"
                        ),
                        "candidate_retrieve_option_version": (
                            "retrieve_deliver_option_v3"
                        ),
                        "candidate_relocation_selector_version": (
                            "nearest_feasible_path_then_cell_v1"
                        ),
                        "ranking_eta_contract": (
                            "stored_yard_canonical_complete_plan_v1"
                        ),
                        "max_defer_steps": (
                            runtime_urgency_args.max_defer_steps
                        ),
                    }
                    if URGENCY_METHOD in args.methods
                    else None
                ),
                "duration_aware_scheduler": (
                    {
                        "method": DURATION_AWARE_METHOD,
                        "controller_architecture": (
                            DURATION_AWARE_ARCHITECTURE
                        ),
                        "controller_action_interface": (
                            URGENCY_FIRST_ACTION_INTERFACE
                        ),
                        "policy_realization": DURATION_AWARE_POLICY,
                        "selector_frozen": True,
                        "comparison_intervention": (
                            "current_epoch_reg_preview_plus_one_step_"
                            "post_accept_timing_projection_v1"
                        ),
                        "accept_assignment_contract": (
                            "post_pickup_frozen_reg_v5_assignment_v1"
                        ),
                        "preview_is_noncommitting_estimate": True,
                        "lookahead_margin_steps": (
                            runtime_duration_args.lookahead_margin_steps
                        ),
                        "max_defer_steps": (
                            runtime_duration_args.max_defer_steps
                        ),
                    }
                    if DURATION_AWARE_METHOD in args.methods
                    else None
                ),
                "online_primary_methods": [
                    method
                    for method in args.methods
                    if method != TRACK_A_GA_OFFLINE
                ],
                "offline_reference_methods": [
                    method
                    for method in args.methods
                    if method == TRACK_A_GA_OFFLINE
                ],
                "pairing_key": "instance_id",
                "storage_flow_metric_contract": (
                    STORAGE_FLOW_METRIC_CONTRACT
                ),
                "storage_flow_scope": "operational_episode",
                "storage_flow_censoring_contract": (
                    "arrived_unstored_right_censored_not_imputed_v1"
                ),
                "storage_flow_paired_eligibility": (
                    "time_reductions_require_both_manifests_fully_observed"
                ),
                "paired_metric_orientation": (
                    "positive favors learned for every paired metric"
                ),
                "online_runtime_timing_contract": (
                    "exclude_fixed_model_construction_include_online_"
                    "planning_inference_and_execution_v1"
                ),
            },
            "method_summaries": summaries,
            "paired_comparisons": paired_summary(
                rows, args.methods, args.bootstrap_samples
            ),
            "repaired_candidate_paired_comparisons": (
                paired_summary(
                    rows,
                    (URGENCY_METHOD,)
                    + tuple(
                        method
                        for method in args.methods
                        if method not in (URGENCY_METHOD, LEARNED_METHOD)
                    ),
                    args.bootstrap_samples,
                    reference_method=URGENCY_METHOD,
                )
                if URGENCY_METHOD in args.methods
                else []
            ),
            "duration_aware_candidate_paired_comparisons": (
                paired_summary(
                    rows,
                    (DURATION_AWARE_METHOD,)
                    + tuple(
                        method
                        for method in args.methods
                        if method
                        not in (DURATION_AWARE_METHOD, LEARNED_METHOD)
                    ),
                    args.bootstrap_samples,
                    reference_method=DURATION_AWARE_METHOD,
                )
                if DURATION_AWARE_METHOD in args.methods
                else []
            ),
        }
    )
    summary_path = args.output_dir / "track-b-summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    audit_path = args.output_dir / "track-b-audit.json"
    audit_path.write_text(
        json.dumps(json_safe(audits), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(payload["method_summaries"], indent=2))
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
