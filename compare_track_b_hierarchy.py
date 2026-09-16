#!/usr/bin/env python3
"""Paired Track-B experiment for decision-epoch cell commitment.

The primary method is the duration-aware scheduler with a frozen REG-v5
assignment proposal bound to ``AcceptStore`` at the scheduler decision epoch.
The experiment crosses assignment source, commitment contract, and geometry,
then adds a REG-only due-first control for interpreting scheduler coupling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compare_track_b_assignment_sources import (
    PAIR_METRIC_SPECIFICATIONS,
    STORAGE_FLOW_FIELDS,
    TIMING_FIELDS,
    bootstrap_ci,
    csv_value,
    finite_mean,
    invariant_signature,
    json_safe,
    retrieval_outcome_summary,
)
from example.Options.selector_v5 import TRACK_B_ASSIGNMENT_SOURCES
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import summarize_storage_flow_runs
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.track_a import (
    TRACK_A_DYNAMIC,
    TRACK_A_REG_SELECTOR_V5,
)
from track_b_urgency_evaluate import (
    ASSIGNMENT_COMMITMENTS,
    DECISION_EPOCH_RESERVED,
    DUE_ONLY_VARIANT,
    DURATION_AWARE_VARIANT,
    POST_PICKUP_RECOMPUTE,
    _evaluate_one,
    resolve_device,
    validate_selector_payload,
)


PROTOCOL_VERSION = "track_b_reserved_hierarchy_factorial_v1"
TRACK = "B_reserved_cell_hierarchy"
ORDINARY = "ordinary"
EGRESS_CONSTRAINED = "egress_constrained"
GEOMETRY_CONDITIONS = (ORDINARY, EGRESS_CONSTRAINED)
SCHEDULER_VARIANTS = (DURATION_AWARE_VARIANT, DUE_ONLY_VARIANT)

RESULTS_NAME = "hierarchy-results.csv"
FULL_NAME = "hierarchy-full.json"
SUMMARY_NAME = "hierarchy-summary.json"
AUDIT_NAME = "hierarchy-audit.json"

COMMITMENT_CONTRACTS = {
    POST_PICKUP_RECOMPUTE: "post_pickup_recompute_v1",
    DECISION_EPOCH_RESERVED: "decision_epoch_proposal_bound_once_v2",
}

COMMON_STRICT_AUDIT_FIELDS = (
    "retrieve_deliver_option_version",
    "retrieval_executor_version",
    "retrieval_start_contract",
    "ranking_eta_contract",
    "relocation_selector_version",
    "defer_option_version",
    "retrieval_termination_contract",
)

GROUP_AUDIT_FIELDS = (
    "decision_epoch_contract",
    "retrieve_deliver_option_version",
    "retrieval_executor_version",
    "retrieval_start_contract",
    "ranking_eta_contract",
    "relocation_selector_version",
    "accept_store_option_version",
    "defer_option_version",
    "retrieval_termination_contract",
    "assignment_commitment_contract",
    "reservation_contract",
)

EXTRA_PAIR_METRICS = {
    "delivery_count_advantage": (
        "delivery_count",
        "reference_minus_comparison",
    ),
    "storage_flow_completion_advantage": (
        "storage_flow_completion_rate",
        "reference_minus_comparison",
    ),
}
PAIR_METRICS = {**PAIR_METRIC_SPECIFICATIONS, **EXTRA_PAIR_METRICS}

CSV_FIELDS = (
    "track",
    "protocol_version",
    "geometry_condition",
    "scheduler_variant",
    "assignment_commitment",
    "assignment_source",
    "method",
    "method_variant",
    "controller_architecture",
    "controller_action_interface",
    "policy_realization",
    "scheduler_invariant_signature",
    "selector_checkpoint_id",
    "selector_checkpoint_digest",
    "episode_seed",
    "instance_id",
    "schedule_id",
    "geometry_regime",
    "geometry_signature",
    "grid_rows",
    "grid_cols",
    "requested_exit_width",
    "actual_exit_width",
    "lambda",
    "mu",
    "lookahead_margin_steps",
    "return",
    "success",
    "strict_method_success",
    "truncated",
    "steps",
    "delivery_count",
    "target_window",
    *TIMING_FIELDS,
    *STORAGE_FLOW_FIELDS,
    "obstructive_moves",
    "illegal_drops",
    "retrieve_relocations",
    "retrieve_replans",
    "mean_retrieval_initial_estimated_steps",
    "mean_retrieval_actual_steps",
    "mean_retrieval_path_expansion_steps",
    "p90_retrieval_path_expansion_steps",
    "selector_decision_count",
    "selector_valid_assignment_count",
    "selector_invalid_assignment_count",
    "selector_infeasible_epoch_count",
    "selector_fallback_count",
    "selector_reserved_commit_count",
    "selector_post_pickup_recompute_count",
    "reservation_integrity",
    "reservation_bound_count",
    "reservation_commit_count",
    "reservation_execution_match_count",
    "reservation_invalidation_count",
    "bound_proposal_ids",
    "committed_proposal_ids",
    "preview_call_count",
    "preview_failure_count",
    "preview_accept_count",
    "preview_cell_match_count",
    "preview_cell_comparison_count",
    "preview_cell_match_rate",
    "mean_accept_duration_actual_minus_estimated_steps",
    "source_setup_seconds",
    "online_assignment_seconds",
    "episode_loop_seconds",
    "method_failure_reason",
    "completed_storage_flow_times",
    "right_censored_storage_flow_ages",
    "delivery_deviations",
    "compact_audit",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired reserved-cell hierarchy factorial and its "
            "REG-only due-first control"
        )
    )
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument(
        "--constrained-exit-width",
        type=int,
        default=2,
        help="right-aligned bottom-gate width in the constrained condition",
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--lookahead-margin-steps", type=float, default=2.0)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.seeds = tuple(dict.fromkeys(args.seeds))
    if not args.seeds:
        parser.error("at least one seed is required")
    if args.max_steps <= 0 or args.max_defer_steps <= 0:
        parser.error("max-steps and max-defer-steps must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    if not math.isfinite(args.lam) or args.lam <= 0.0:
        parser.error("lambda must be finite and positive")
    if not math.isfinite(args.mu) or args.mu <= 0.0:
        parser.error("mu must be finite and positive")
    if (
        not math.isfinite(args.lookahead_margin_steps)
        or args.lookahead_margin_steps < 0.0
    ):
        parser.error("lookahead-margin-steps must be finite and nonnegative")
    if not math.isfinite(args.target_window) or args.target_window < 0.0:
        parser.error("target-window must be finite and nonnegative")
    try:
        ordinary = make_condition_env(args, ORDINARY)
        constrained = make_condition_env(args, EGRESS_CONSTRAINED)
    except ValueError as exc:
        parser.error(str(exc))
    if len(constrained.exit_cells) >= len(ordinary.exit_cells):
        parser.error(
            "constrained-exit-width must be narrower than the ordinary "
            f"gate ({len(ordinary.exit_cells)})"
        )
    return args


def requested_exit_width(args, condition: str) -> int | None:
    if condition == ORDINARY:
        return None
    if condition == EGRESS_CONSTRAINED:
        return int(args.constrained_exit_width)
    raise ValueError(f"unknown geometry condition: {condition!r}")


def make_condition_env(args, condition: str) -> SmallRoomsEnv:
    return make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=requested_exit_width(args, condition),
    )


def sample_matched_instances(
    args,
) -> dict[str, dict[int, EpisodeInstance]]:
    instances = {condition: {} for condition in GEOMETRY_CONDITIONS}
    for seed in args.seeds:
        ordinary = make_condition_env(args, ORDINARY).sample_episode_instance(seed)
        constrained = make_condition_env(
            args, EGRESS_CONSTRAINED
        ).sample_episode_instance(seed)
        ordinary.validate_for(make_condition_env(args, ORDINARY))
        constrained.validate_for(make_condition_env(args, EGRESS_CONSTRAINED))
        if ordinary.schedule_id != constrained.schedule_id:
            raise RuntimeError(
                f"geometry changed the stochastic schedule for seed {seed}"
            )
        if ordinary.instance_id == constrained.instance_id:
            raise RuntimeError(
                f"ordinary and constrained instances coincide for seed {seed}"
            )
        instances[ORDINARY][seed] = ordinary
        instances[EGRESS_CONSTRAINED][seed] = constrained
    return instances


def experiment_cells():
    """Yield scheduler, commitment, and source cells in stable order."""

    for commitment in ASSIGNMENT_COMMITMENTS:
        for source in TRACK_B_ASSIGNMENT_SOURCES:
            yield DURATION_AWARE_VARIANT, commitment, source
    for commitment in ASSIGNMENT_COMMITMENTS:
        yield DUE_ONLY_VARIANT, commitment, TRACK_A_REG_SELECTOR_V5


def _scheduler_preview_seconds(scheduler_audit: dict) -> float:
    if "preview_seconds" in scheduler_audit:
        return float(scheduler_audit["preview_seconds"])
    return float(
        sum(
            float(decision.get("preview_seconds", 0.0) or 0.0)
            for decision in scheduler_audit.get("decisions", ())
        )
    )


def _group_contract(run: dict) -> dict:
    audit = run["urgency_scheduler_audit"]
    result = {
        "scheduler_variant": run["scheduler_variant"],
        "scheduler_method": run["method"],
        "scheduler_architecture": run["controller_architecture"],
        "scheduler_policy": run["policy_realization"],
        "controller_action_interface": run["controller_action_interface"],
        **{name: audit.get(name) for name in GROUP_AUDIT_FIELDS},
    }
    if run["scheduler_variant"] == DURATION_AWARE_VARIANT:
        result.update(
            {
                "accept_duration_estimate_contract": audit.get(
                    "accept_duration_estimate_contract"
                ),
                "lookahead_cost_contract": audit.get(
                    "lookahead_cost_contract"
                ),
                "lookahead_margin_steps": audit.get(
                    "lookahead_margin_steps"
                ),
            }
        )
    return result


def _common_strict_contract(run: dict) -> dict:
    audit = run["urgency_scheduler_audit"]
    return {name: audit.get(name) for name in COMMON_STRICT_AUDIT_FIELDS}


def _source_contract(run: dict) -> dict:
    audit = run["selector_audit"]
    return {
        "assignment_source": run["assignment_source"],
        "assignment_source_family": run["assignment_source_family"],
        "assignment_source_version": run["assignment_source_version"],
        "assignment_source_learned": audit["assignment_source_learned"],
        "information_regime": audit["information_regime"],
        "preview_contract": audit["preview_contract"],
    }


def _reservation_integrity(run: dict) -> bool:
    options = run["scheduler_audit"]
    selector = run["selector_audit"]
    if run["assignment_commitment"] != DECISION_EPOCH_RESERVED:
        return bool(
            options["reservation_bound_count"] == 0
            and options["reservation_commit_count"] == 0
            and options["reservation_execution_match_count"] == 0
            and options["reservation_invalidation_count"] == 0
            and not options["bound_proposal_ids"]
            and not options["committed_proposal_ids"]
            and selector.get("reserved_commit_count", 0) == 0
            and selector.get("post_pickup_recompute_count", 0)
            == selector["valid_assignment_count"]
        )

    bound_ids = list(options["bound_proposal_ids"])
    committed_ids = list(options["committed_proposal_ids"])
    selector_ids = [
        decision.get("proposal_id")
        for decision in selector["decisions"]
        if decision.get("valid")
        and decision.get("commitment_contract")
        == COMMITMENT_CONTRACTS[DECISION_EPOCH_RESERVED]
    ]
    successes = int(options["inbound_successes"])
    return bool(
        options["reservation_invalidation_count"] == 0
        and options["reservation_bound_count"] == successes
        and options["reservation_commit_count"] == successes
        and options["reservation_execution_match_count"] == successes
        and len(bound_ids) == len(set(bound_ids))
        and bound_ids == committed_ids
        and committed_ids == selector_ids
        and selector.get("reserved_commit_count", 0) == successes
        and selector.get("post_pickup_recompute_count", 0) == 0
    )


def validate_run(
    run: dict,
    *,
    instance: EpisodeInstance,
    geometry: dict,
    scheduler_variant: str,
    commitment: str,
    source: str,
    selector_digest: str,
) -> None:
    if run["instance_id"] != instance.instance_id:
        raise RuntimeError("evaluator did not consume the matched instance")
    if run["schedule_id"] != instance.schedule_id:
        raise RuntimeError("evaluator did not consume the matched schedule")
    if run["geometry"] != geometry:
        raise RuntimeError("evaluator geometry provenance mismatch")
    if run["scheduler_variant"] != scheduler_variant:
        raise RuntimeError("scheduler-variant provenance mismatch")
    if run["assignment_commitment"] != commitment:
        raise RuntimeError("assignment-commitment provenance mismatch")
    if run["assignment_source"] != source:
        raise RuntimeError("assignment-source provenance mismatch")
    if run["selector_audit"]["fallback_count"] != 0:
        raise RuntimeError("strict hierarchy experiment used a fallback")
    expected_digest = (
        selector_digest if source == TRACK_A_REG_SELECTOR_V5 else None
    )
    if run["selector_deployment_digest"] != expected_digest:
        raise RuntimeError("selector checkpoint provenance mismatch")

    audit = run["urgency_scheduler_audit"]
    expected_contract = COMMITMENT_CONTRACTS[commitment]
    if audit.get("assignment_commitment_contract") != expected_contract:
        raise RuntimeError("scheduler commitment contract mismatch")
    computed_integrity = _reservation_integrity(run)
    if bool(run["reservation_integrity"]) != computed_integrity:
        raise RuntimeError("evaluator reservation-integrity audit mismatch")
    if (
        commitment == DECISION_EPOCH_RESERVED
        and not computed_integrity
        and run["method_failure_reason"] is None
        and not bool(run["truncated"])
    ):
        raise RuntimeError("silent reserved-cell integrity violation")

    options = run["scheduler_audit"]
    expected_strict = bool(
        run["success"]
        and run["method_failure_reason"] is None
        and run["selector_audit"]["invalid_assignment_count"] == 0
        and run["selector_audit"]["fallback_count"] == 0
        and options["inbound_failures"] == 0
        and options["retrieve_failures"] == 0
        and run["illegal_drops"] == 0
        and computed_integrity
    )
    if bool(run["strict_method_success"]) != expected_strict:
        raise RuntimeError("strict-success audit is internally inconsistent")


def _compact_audit(run: dict) -> dict:
    selector = {
        key: value
        for key, value in run["selector_audit"].items()
        if key != "decisions"
    }
    scheduler = {
        key: value
        for key, value in run["urgency_scheduler_audit"].items()
        if key not in ("decisions", "storage_flow_records")
    }
    options = {
        key: value
        for key, value in run["scheduler_audit"].items()
        if key not in ("inbound_outcomes", "retrieve_outcomes")
    }
    return {
        "selector": selector,
        "scheduler": scheduler,
        "atomic_options": options,
    }


def result_row(
    run: dict,
    *,
    geometry_condition: str,
    selector_checkpoint_id: str,
    selector_digest: str,
    scheduler_signature: str,
) -> dict:
    selector = run["selector_audit"]
    scheduler = run["urgency_scheduler_audit"]
    options = run["scheduler_audit"]
    preview_seconds = _scheduler_preview_seconds(scheduler)
    retrieval = retrieval_outcome_summary([options])
    return {
        "track": TRACK,
        "protocol_version": PROTOCOL_VERSION,
        "geometry_condition": geometry_condition,
        "scheduler_variant": run["scheduler_variant"],
        "assignment_commitment": run["assignment_commitment"],
        "assignment_source": run["assignment_source"],
        "method": run["method"],
        "method_variant": run["method_variant"],
        "controller_architecture": run["controller_architecture"],
        "controller_action_interface": run["controller_action_interface"],
        "policy_realization": run["policy_realization"],
        "scheduler_invariant_signature": scheduler_signature,
        "selector_checkpoint_id": (
            selector_checkpoint_id
            if run["assignment_source"] == TRACK_A_REG_SELECTOR_V5
            else "not_applicable"
        ),
        "selector_checkpoint_digest": (
            selector_digest
            if run["assignment_source"] == TRACK_A_REG_SELECTOR_V5
            else "not_applicable"
        ),
        "episode_seed": run["eval_seed"],
        "instance_id": run["instance_id"],
        "schedule_id": run["schedule_id"],
        "geometry_regime": run["geometry"]["geometry_regime"],
        "geometry_signature": run["geometry"]["geometry_signature"],
        "grid_rows": run["geometry"]["grid_rows"],
        "grid_cols": run["geometry"]["grid_cols"],
        "requested_exit_width": run["geometry"]["requested_exit_width"],
        "actual_exit_width": run["geometry"]["actual_exit_width"],
        "lambda": run["lambda"],
        "mu": run["mu"],
        "lookahead_margin_steps": (
            scheduler.get("lookahead_margin_steps")
            if run["scheduler_variant"] == DURATION_AWARE_VARIANT
            else None
        ),
        "return": run["return"],
        "success": run["success"],
        "strict_method_success": run["strict_method_success"],
        "truncated": run["truncated"],
        "steps": run["steps"],
        "delivery_count": run["delivery_count"],
        "target_window": run["target_window"],
        **{name: run[name] for name in TIMING_FIELDS},
        **{name: run[name] for name in STORAGE_FLOW_FIELDS},
        "obstructive_moves": run["obstructive_moves"],
        "illegal_drops": run["illegal_drops"],
        "retrieve_relocations": options["retrieve_relocations"],
        "retrieve_replans": options["retrieve_replans"],
        **retrieval,
        "selector_decision_count": selector["decision_count"],
        "selector_valid_assignment_count": selector["valid_assignment_count"],
        "selector_invalid_assignment_count": selector[
            "invalid_assignment_count"
        ],
        "selector_infeasible_epoch_count": selector["infeasible_epoch_count"],
        "selector_fallback_count": selector["fallback_count"],
        "selector_reserved_commit_count": selector.get(
            "reserved_commit_count", 0
        ),
        "selector_post_pickup_recompute_count": selector.get(
            "post_pickup_recompute_count", 0
        ),
        "reservation_integrity": run["reservation_integrity"],
        "reservation_bound_count": options["reservation_bound_count"],
        "reservation_commit_count": options["reservation_commit_count"],
        "reservation_execution_match_count": options[
            "reservation_execution_match_count"
        ],
        "reservation_invalidation_count": options[
            "reservation_invalidation_count"
        ],
        "bound_proposal_ids": options["bound_proposal_ids"],
        "committed_proposal_ids": options["committed_proposal_ids"],
        "preview_call_count": scheduler.get(
            "preview_call_count", options["reservation_bound_count"]
        ),
        "preview_failure_count": scheduler.get("preview_failure_count", 0),
        "preview_accept_count": scheduler.get(
            "preview_accept_count", options["reservation_bound_count"]
        ),
        "preview_cell_match_count": scheduler.get(
            "preview_cell_match_count",
            options["reservation_execution_match_count"],
        ),
        "preview_cell_comparison_count": scheduler.get(
            "preview_cell_comparison_count",
            options["reservation_commit_count"],
        ),
        "preview_cell_match_rate": scheduler.get(
            "preview_cell_match_rate",
            (
                options["reservation_execution_match_count"]
                / options["reservation_commit_count"]
                if options["reservation_commit_count"]
                else None
            ),
        ),
        "mean_accept_duration_actual_minus_estimated_steps": scheduler.get(
            "mean_accept_duration_actual_minus_estimated_steps"
        ),
        "source_setup_seconds": run["source_setup_seconds"],
        "online_assignment_seconds": float(
            selector.get("assignment_total_seconds", 0.0)
        )
        + preview_seconds,
        "episode_loop_seconds": run["decision_seconds"],
        "method_failure_reason": run["method_failure_reason"],
        "completed_storage_flow_times": run[
            "completed_storage_flow_times"
        ],
        "right_censored_storage_flow_ages": run[
            "right_censored_storage_flow_ages"
        ],
        "delivery_deviations": run["delivery_deviations"],
        "compact_audit": _compact_audit(run),
    }


def _metric_value(run: dict, field: str):
    if field in run:
        return run[field]
    if field in ("retrieve_relocations", "retrieve_replans"):
        return run["scheduler_audit"][field]
    raise KeyError(field)


def _favorable_delta(left: dict, right: dict, field: str, orientation: str):
    left_value = _metric_value(left, field)
    right_value = _metric_value(right, field)
    if left_value is None or right_value is None:
        return None
    left_value = float(left_value)
    right_value = float(right_value)
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        return None
    if orientation == "reference_minus_comparison":
        return left_value - right_value
    if orientation == "comparison_minus_reference":
        return right_value - left_value
    raise ValueError(f"unknown metric orientation: {orientation!r}")


def _filter_map(runs: list[dict], filters: dict) -> dict[str, dict]:
    selected = [
        run
        for run in runs
        if all(run.get(name) == value for name, value in filters.items())
    ]
    result = {}
    for run in selected:
        schedule_id = run["schedule_id"]
        if schedule_id in result:
            raise RuntimeError(f"duplicate contrast cell for {filters}")
        result[schedule_id] = run
    return result


def _bootstrap_rng(label: str, metric: str):
    digest = hashlib.sha256(f"{label}:{metric}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _metric_contrast_summary(
    *,
    label: str,
    metric: str,
    values: list[float],
    records: list[dict],
    samples: int,
) -> dict:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "bootstrap_95_ci": (
            bootstrap_ci(values, samples, _bootstrap_rng(label, metric))
            if values
            else None
        ),
        "wins_ties_losses": {
            "wins": sum(value > 1e-12 for value in values),
            "ties": sum(abs(value) <= 1e-12 for value in values),
            "losses": sum(value < -1e-12 for value in values),
        },
        "per_schedule": records,
    }


def paired_contrast(
    runs: list[dict],
    *,
    label: str,
    left_filters: dict,
    right_filters: dict,
    left_name: str,
    right_name: str,
    samples: int,
) -> dict:
    left = _filter_map(runs, left_filters)
    right = _filter_map(runs, right_filters)
    if set(left) != set(right):
        raise RuntimeError(f"unpaired schedules in contrast {label}")
    schedule_ids = sorted(left)
    metrics = {}
    for metric, (field, orientation) in PAIR_METRICS.items():
        values = []
        records = []
        for schedule_id in schedule_ids:
            left_run = left[schedule_id]
            right_run = right[schedule_id]
            if (
                left_run["geometry_condition"]
                == right_run["geometry_condition"]
                and left_run["instance_id"] != right_run["instance_id"]
            ):
                raise RuntimeError(f"instance mismatch in contrast {label}")
            delta = _favorable_delta(
                left_run, right_run, field, orientation
            )
            if delta is None:
                continue
            values.append(delta)
            records.append(
                {
                    "episode_seed": left_run["eval_seed"],
                    "schedule_id": schedule_id,
                    "left_instance_id": left_run["instance_id"],
                    "right_instance_id": right_run["instance_id"],
                    "delta": float(delta),
                }
            )
        metrics[metric] = _metric_contrast_summary(
            label=label,
            metric=metric,
            values=values,
            records=records,
            samples=samples,
        )
    return {
        "label": label,
        "left": left_name,
        "right": right_name,
        "left_filters": left_filters,
        "right_filters": right_filters,
        "pairing_key": "schedule_id",
        "n_schedules": len(schedule_ids),
        "metric_orientation": "positive favors the named left condition",
        "metrics": metrics,
    }


def interaction_contrast(
    runs: list[dict],
    *,
    label: str,
    effect_a: tuple[dict, dict],
    effect_b: tuple[dict, dict],
    effect_a_name: str,
    effect_b_name: str,
    samples: int,
) -> dict:
    a_left = _filter_map(runs, effect_a[0])
    a_right = _filter_map(runs, effect_a[1])
    b_left = _filter_map(runs, effect_b[0])
    b_right = _filter_map(runs, effect_b[1])
    maps = (a_left, a_right, b_left, b_right)
    schedule_sets = [set(item) for item in maps]
    if not schedule_sets or any(item != schedule_sets[0] for item in schedule_sets):
        raise RuntimeError(f"unpaired schedules in interaction {label}")
    schedule_ids = sorted(schedule_sets[0])
    metrics = {}
    for metric, (field, orientation) in PAIR_METRICS.items():
        values = []
        records = []
        for schedule_id in schedule_ids:
            effect_a_value = _favorable_delta(
                a_left[schedule_id],
                a_right[schedule_id],
                field,
                orientation,
            )
            effect_b_value = _favorable_delta(
                b_left[schedule_id],
                b_right[schedule_id],
                field,
                orientation,
            )
            if effect_a_value is None or effect_b_value is None:
                continue
            value = effect_a_value - effect_b_value
            values.append(value)
            records.append(
                {
                    "episode_seed": a_left[schedule_id]["eval_seed"],
                    "schedule_id": schedule_id,
                    "effect_a": float(effect_a_value),
                    "effect_b": float(effect_b_value),
                    "interaction": float(value),
                }
            )
        metrics[metric] = _metric_contrast_summary(
            label=label,
            metric=metric,
            values=values,
            records=records,
            samples=samples,
        )
    return {
        "label": label,
        "effect_a": effect_a_name,
        "effect_b": effect_b_name,
        "effect_a_filters": {"left": effect_a[0], "right": effect_a[1]},
        "effect_b_filters": {"left": effect_b[0], "right": effect_b[1]},
        "pairing_key": "schedule_id",
        "n_schedules": len(schedule_ids),
        "metric_orientation": (
            "positive means the favorable effect is larger for effect_a"
        ),
        "metrics": metrics,
    }


def cell_summary(runs: list[dict], filters: dict) -> dict:
    selected = [
        run
        for run in runs
        if all(run.get(name) == value for name, value in filters.items())
    ]
    if not selected:
        raise RuntimeError(f"empty experiment cell: {filters}")
    returns = np.asarray([run["return"] for run in selected], dtype=float)
    selectors = [run["selector_audit"] for run in selected]
    options = [run["scheduler_audit"] for run in selected]
    reservation_commits = sum(
        item["reservation_commit_count"] for item in options
    )
    reservation_matches = sum(
        item["reservation_execution_match_count"] for item in options
    )
    return {
        **filters,
        "episodes": len(selected),
        "mean_return": float(np.mean(returns)),
        "return_std": (
            float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        ),
        "success_rate": float(np.mean([run["success"] for run in selected])),
        "strict_method_success_rate": float(
            np.mean([run["strict_method_success"] for run in selected])
        ),
        "mean_delivery_count": float(
            np.mean([run["delivery_count"] for run in selected])
        ),
        "mean_steps": float(np.mean([run["steps"] for run in selected])),
        **{
            name: finite_mean(run[name] for run in selected)
            for name in TIMING_FIELDS
        },
        **summarize_storage_flow_runs(selected),
        "total_obstructive_moves": int(
            sum(run["obstructive_moves"] for run in selected)
        ),
        "total_retrieve_relocations": int(
            sum(item["retrieve_relocations"] for item in options)
        ),
        "episodes_with_retrieve_relocation": int(
            sum(item["retrieve_relocations"] > 0 for item in options)
        ),
        "mean_retrieve_relocations_per_episode": float(
            np.mean([item["retrieve_relocations"] for item in options])
        ),
        "total_retrieve_replans": int(
            sum(item["retrieve_replans"] for item in options)
        ),
        "total_invalid_assignments": int(
            sum(item["invalid_assignment_count"] for item in selectors)
        ),
        "total_infeasible_epochs": int(
            sum(item["infeasible_epoch_count"] for item in selectors)
        ),
        "total_fallbacks": int(
            sum(item["fallback_count"] for item in selectors)
        ),
        "total_illegal_drops": int(
            sum(run["illegal_drops"] for run in selected)
        ),
        "reservation_integrity_rate": float(
            np.mean([run["reservation_integrity"] for run in selected])
        ),
        "reservation_bound_count": int(
            sum(item["reservation_bound_count"] for item in options)
        ),
        "reservation_commit_count": int(reservation_commits),
        "reservation_execution_match_count": int(reservation_matches),
        "reservation_execution_match_rate": (
            reservation_matches / reservation_commits
            if reservation_commits
            else None
        ),
        "reservation_invalidation_count": int(
            sum(item["reservation_invalidation_count"] for item in options)
        ),
        "method_failures": [
            {
                "episode_seed": run["eval_seed"],
                "reason": run["method_failure_reason"],
            }
            for run in selected
            if run["method_failure_reason"] is not None
        ],
    }


def build_contrasts(runs: list[dict], samples: int) -> dict:
    reservations = []
    for scheduler in SCHEDULER_VARIANTS:
        sources = (
            TRACK_B_ASSIGNMENT_SOURCES
            if scheduler == DURATION_AWARE_VARIANT
            else (TRACK_A_REG_SELECTOR_V5,)
        )
        for geometry in GEOMETRY_CONDITIONS:
            for source in sources:
                common = {
                    "scheduler_variant": scheduler,
                    "geometry_condition": geometry,
                    "assignment_source": source,
                }
                reservations.append(
                    paired_contrast(
                        runs,
                        label=(
                            f"reserved_vs_recompute__{scheduler}__"
                            f"{geometry}__{source}"
                        ),
                        left_filters={
                            **common,
                            "assignment_commitment": DECISION_EPOCH_RESERVED,
                        },
                        right_filters={
                            **common,
                            "assignment_commitment": POST_PICKUP_RECOMPUTE,
                        },
                        left_name="decision_epoch_reserved",
                        right_name="post_pickup_recompute",
                        samples=samples,
                    )
                )

    reg_dynamic = []
    for commitment in ASSIGNMENT_COMMITMENTS:
        for geometry in GEOMETRY_CONDITIONS:
            common = {
                "scheduler_variant": DURATION_AWARE_VARIANT,
                "geometry_condition": geometry,
                "assignment_commitment": commitment,
            }
            reg_dynamic.append(
                paired_contrast(
                    runs,
                    label=f"reg_vs_dynamic__{commitment}__{geometry}",
                    left_filters={
                        **common,
                        "assignment_source": TRACK_A_REG_SELECTOR_V5,
                    },
                    right_filters={
                        **common,
                        "assignment_source": TRACK_A_DYNAMIC,
                    },
                    left_name="reg_selector_v5",
                    right_name="dynamic_pslap",
                    samples=samples,
                )
            )

    geometry_did = []
    for commitment in ASSIGNMENT_COMMITMENTS:
        constrained_common = {
            "scheduler_variant": DURATION_AWARE_VARIANT,
            "geometry_condition": EGRESS_CONSTRAINED,
            "assignment_commitment": commitment,
        }
        ordinary_common = {
            "scheduler_variant": DURATION_AWARE_VARIANT,
            "geometry_condition": ORDINARY,
            "assignment_commitment": commitment,
        }
        geometry_did.append(
            interaction_contrast(
                runs,
                label=f"geometry_did_reg_vs_dynamic__{commitment}",
                effect_a=(
                    {
                        **constrained_common,
                        "assignment_source": TRACK_A_REG_SELECTOR_V5,
                    },
                    {
                        **constrained_common,
                        "assignment_source": TRACK_A_DYNAMIC,
                    },
                ),
                effect_b=(
                    {
                        **ordinary_common,
                        "assignment_source": TRACK_A_REG_SELECTOR_V5,
                    },
                    {
                        **ordinary_common,
                        "assignment_source": TRACK_A_DYNAMIC,
                    },
                ),
                effect_a_name="REG-vs-dynamic under constrained egress",
                effect_b_name="REG-vs-dynamic under ordinary egress",
                samples=samples,
            )
        )

    contract_source = []
    for geometry in GEOMETRY_CONDITIONS:
        base = {
            "scheduler_variant": DURATION_AWARE_VARIANT,
            "geometry_condition": geometry,
        }
        contract_source.append(
            interaction_contrast(
                runs,
                label=f"contract_by_source_reg_vs_dynamic__{geometry}",
                effect_a=(
                    {
                        **base,
                        "assignment_source": TRACK_A_REG_SELECTOR_V5,
                        "assignment_commitment": DECISION_EPOCH_RESERVED,
                    },
                    {
                        **base,
                        "assignment_source": TRACK_A_REG_SELECTOR_V5,
                        "assignment_commitment": POST_PICKUP_RECOMPUTE,
                    },
                ),
                effect_b=(
                    {
                        **base,
                        "assignment_source": TRACK_A_DYNAMIC,
                        "assignment_commitment": DECISION_EPOCH_RESERVED,
                    },
                    {
                        **base,
                        "assignment_source": TRACK_A_DYNAMIC,
                        "assignment_commitment": POST_PICKUP_RECOMPUTE,
                    },
                ),
                effect_a_name="reservation effect for REG-v5",
                effect_b_name="reservation effect for dynamic PSLAP",
                samples=samples,
            )
        )

    contract_geometry = []
    for scheduler in SCHEDULER_VARIANTS:
        sources = (
            TRACK_B_ASSIGNMENT_SOURCES
            if scheduler == DURATION_AWARE_VARIANT
            else (TRACK_A_REG_SELECTOR_V5,)
        )
        for source in sources:
            common = {
                "scheduler_variant": scheduler,
                "assignment_source": source,
            }
            contract_geometry.append(
                interaction_contrast(
                    runs,
                    label=(
                        f"contract_by_geometry__{scheduler}__{source}"
                    ),
                    effect_a=(
                        {
                            **common,
                            "geometry_condition": EGRESS_CONSTRAINED,
                            "assignment_commitment": DECISION_EPOCH_RESERVED,
                        },
                        {
                            **common,
                            "geometry_condition": EGRESS_CONSTRAINED,
                            "assignment_commitment": POST_PICKUP_RECOMPUTE,
                        },
                    ),
                    effect_b=(
                        {
                            **common,
                            "geometry_condition": ORDINARY,
                            "assignment_commitment": DECISION_EPOCH_RESERVED,
                        },
                        {
                            **common,
                            "geometry_condition": ORDINARY,
                            "assignment_commitment": POST_PICKUP_RECOMPUTE,
                        },
                    ),
                    effect_a_name="reservation effect under constrained egress",
                    effect_b_name="reservation effect under ordinary egress",
                    samples=samples,
                )
            )

    scheduler_effects = []
    for geometry in GEOMETRY_CONDITIONS:
        for commitment in ASSIGNMENT_COMMITMENTS:
            common = {
                "geometry_condition": geometry,
                "assignment_commitment": commitment,
                "assignment_source": TRACK_A_REG_SELECTOR_V5,
            }
            scheduler_effects.append(
                paired_contrast(
                    runs,
                    label=f"duration_vs_due__{commitment}__{geometry}",
                    left_filters={
                        **common,
                        "scheduler_variant": DURATION_AWARE_VARIANT,
                    },
                    right_filters={
                        **common,
                        "scheduler_variant": DUE_ONLY_VARIANT,
                    },
                    left_name="duration_aware",
                    right_name="due_only",
                    samples=samples,
                )
            )

    contract_scheduler = []
    for geometry in GEOMETRY_CONDITIONS:
        common = {
            "geometry_condition": geometry,
            "assignment_source": TRACK_A_REG_SELECTOR_V5,
        }
        contract_scheduler.append(
            interaction_contrast(
                runs,
                label=f"contract_by_scheduler__reg__{geometry}",
                effect_a=(
                    {
                        **common,
                        "scheduler_variant": DURATION_AWARE_VARIANT,
                        "assignment_commitment": DECISION_EPOCH_RESERVED,
                    },
                    {
                        **common,
                        "scheduler_variant": DURATION_AWARE_VARIANT,
                        "assignment_commitment": POST_PICKUP_RECOMPUTE,
                    },
                ),
                effect_b=(
                    {
                        **common,
                        "scheduler_variant": DUE_ONLY_VARIANT,
                        "assignment_commitment": DECISION_EPOCH_RESERVED,
                    },
                    {
                        **common,
                        "scheduler_variant": DUE_ONLY_VARIANT,
                        "assignment_commitment": POST_PICKUP_RECOMPUTE,
                    },
                ),
                effect_a_name="reservation effect under duration-aware",
                effect_b_name="reservation effect under due-only",
                samples=samples,
            )
        )

    return {
        "reserved_minus_recompute": reservations,
        "reg_minus_dynamic_duration_aware": reg_dynamic,
        "geometry_did_reg_minus_dynamic": geometry_did,
        "contract_by_source": contract_source,
        "contract_by_geometry": contract_geometry,
        "duration_minus_due_reg": scheduler_effects,
        "contract_by_scheduler_reg": contract_scheduler,
    }


def _key_console_summary(contrasts: dict) -> dict:
    def metric(group, label, metric="return_advantage"):
        item = next(entry for entry in contrasts[group] if entry["label"] == label)
        return item["metrics"][metric]

    return {
        "reserved_REG_vs_dynamic_constrained_return": metric(
            "reg_minus_dynamic_duration_aware",
            "reg_vs_dynamic__decision_epoch_reserved__egress_constrained",
        ),
        "reserved_REG_vs_dynamic_geometry_DID_return": metric(
            "geometry_did_reg_minus_dynamic",
            "geometry_did_reg_vs_dynamic__decision_epoch_reserved",
        ),
        "REG_reservation_effect_ordinary_return": metric(
            "reserved_minus_recompute",
            "reserved_vs_recompute__duration_aware__ordinary__reg_selector_v5",
        ),
        "REG_reservation_effect_constrained_return": metric(
            "reserved_minus_recompute",
            (
                "reserved_vs_recompute__duration_aware__"
                "egress_constrained__reg_selector_v5"
            ),
        ),
        "REG_contract_by_scheduler_constrained_return": metric(
            "contract_by_scheduler_reg",
            "contract_by_scheduler__reg__egress_constrained",
        ),
    }


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "results": output_dir / RESULTS_NAME,
        "full": output_dir / FULL_NAME,
        "summary": output_dir / SUMMARY_NAME,
        "audit": output_dir / AUDIT_NAME,
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    args = parse_args()
    paths = _output_paths(args.output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; choose a new output directory "
            "or use --overwrite"
        )

    selector_payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    validate_selector_payload(selector_payload, lam=args.lam, mu=args.mu)
    selector_digest = selector_deployment_digest(selector_payload)
    device = resolve_device(args.device)
    instances = sample_matched_instances(args)
    geometries = {
        condition: geometry_metadata(
            make_condition_env(args, condition),
            requested_exit_width=requested_exit_width(args, condition),
        )
        for condition in GEOMETRY_CONDITIONS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_root = args.output_dir / "instances"
    for condition in GEOMETRY_CONDITIONS:
        destination = instance_root / condition
        destination.mkdir(parents=True, exist_ok=True)
        for seed, instance in instances[condition].items():
            path = destination / f"seed-{seed}.json"
            if path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{path} already exists; use --overwrite or a new directory"
                )
            path.write_text(instance.to_json() + "\n")

    runtime_args = {
        condition: SimpleNamespace(
            lam=args.lam,
            mu=args.mu,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            exit_width=requested_exit_width(args, condition),
            instance=None,
            device=str(device),
            max_steps=args.max_steps,
            max_defer_steps=args.max_defer_steps,
            lookahead_margin_steps=args.lookahead_margin_steps,
            target_window=args.target_window,
            save_instances_dir=None,
        )
        for condition in GEOMETRY_CONDITIONS
    }

    runs = []
    group_contracts = {}
    group_signatures = {}
    common_contract = None
    source_contracts = {}
    total_runs = len(args.seeds) * len(GEOMETRY_CONDITIONS) * sum(
        1 for _ in experiment_cells()
    )
    run_index = 0
    for seed in args.seeds:
        for condition in GEOMETRY_CONDITIONS:
            instance = instances[condition][seed]
            for scheduler_variant, commitment, source in experiment_cells():
                payload = (
                    selector_payload
                    if source == TRACK_A_REG_SELECTOR_V5
                    else None
                )
                run = _evaluate_one(
                    runtime_args[condition],
                    seed,
                    payload,
                    episode_instance=instance,
                    scheduler_variant=scheduler_variant,
                    assignment_source=source,
                    source_neutral_scheduler=(
                        scheduler_variant == DURATION_AWARE_VARIANT
                    ),
                    assignment_commitment=commitment,
                )
                run["geometry_condition"] = condition
                run["scheduler_variant"] = scheduler_variant
                validate_run(
                    run,
                    instance=instance,
                    geometry=geometries[condition],
                    scheduler_variant=scheduler_variant,
                    commitment=commitment,
                    source=source,
                    selector_digest=selector_digest,
                )

                strict_contract = _common_strict_contract(run)
                if common_contract is None:
                    common_contract = strict_contract
                elif common_contract != strict_contract:
                    raise RuntimeError(
                        "retrieval/defer strict contract changed across cells"
                    )
                group_key = (scheduler_variant, commitment)
                contract = _group_contract(run)
                previous = group_contracts.setdefault(group_key, contract)
                if previous != contract:
                    raise RuntimeError(
                        f"scheduler contract changed within group {group_key}"
                    )
                group_signatures[group_key] = invariant_signature(contract)
                source_contract = _source_contract(run)
                previous_source = source_contracts.setdefault(
                    source, source_contract
                )
                if previous_source != source_contract:
                    raise RuntimeError(
                        f"assignment-source contract changed for {source}"
                    )

                runs.append(run)
                run_index += 1
                print(
                    f"[{run_index:03d}/{total_runs}] "
                    f"seed={seed} geometry={condition} "
                    f"scheduler={scheduler_variant} "
                    f"commitment={commitment} source={source} "
                    f"R={run['return']:.2f} "
                    f"strict={run['strict_method_success']:.0f} "
                    f"steps={run['steps']} "
                    f"reloc={run['scheduler_audit']['retrieve_relocations']} "
                    f"MAE={run['mean_absolute_error']}",
                    flush=True,
                )

    expected = total_runs
    observed_keys = {
        (
            run["geometry_condition"],
            run["scheduler_variant"],
            run["assignment_commitment"],
            run["assignment_source"],
            run["schedule_id"],
        )
        for run in runs
    }
    if len(runs) != expected or len(observed_keys) != expected:
        raise RuntimeError("hierarchy experiment cell grid is incomplete")

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "track": TRACK,
        "design": (
            "paired_2_geometry_x_2_commitment_x_3_source_duration_aware_"
            "plus_REG_only_2_commitment_due_control_v1"
        ),
        "primary_method": {
            "scheduler_variant": DURATION_AWARE_VARIANT,
            "assignment_commitment": DECISION_EPOCH_RESERVED,
            "assignment_source": TRACK_A_REG_SELECTOR_V5,
        },
        "primary_comparator": TRACK_A_DYNAMIC,
        "assignment_sources": list(TRACK_B_ASSIGNMENT_SOURCES),
        "assignment_commitments": list(ASSIGNMENT_COMMITMENTS),
        "scheduler_variants": list(SCHEDULER_VARIANTS),
        "due_only_scope": "REG-v5 commitment mechanism control only",
        "geometry_conditions": geometries,
        "selector_checkpoint": str(args.selector_checkpoint.resolve()),
        "selector_checkpoint_digest": selector_digest,
        "selector_frozen": True,
        "lambda": args.lam,
        "mu": args.mu,
        "seeds": list(args.seeds),
        "instance_ids": {
            condition: {
                str(seed): instances[condition][seed].instance_id
                for seed in args.seeds
            }
            for condition in GEOMETRY_CONDITIONS
        },
        "schedule_ids": {
            str(seed): instances[ORDINARY][seed].schedule_id
            for seed in args.seeds
        },
        "within_geometry_pairing_key": "instance_id",
        "cross_geometry_pairing_key": "schedule_id",
        "max_steps": args.max_steps,
        "max_defer_steps": args.max_defer_steps,
        "lookahead_margin_steps": args.lookahead_margin_steps,
        "target_window": args.target_window,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "complete matched stochastic schedule",
        "device": str(device),
        "shared_candidate_mask": "shared_candidate_mask_v1",
        "assignment_failure_contract": (
            "invalid_or_missing_proposal_is_explicit_failure_no_fallback_v1"
        ),
        "primary_estimands": [
            (
                "reserved duration-aware REG-v5 minus reserved duration-aware "
                "dynamic PSLAP under constrained egress"
            ),
            (
                "the constrained-minus-ordinary interaction in the reserved "
                "REG-v5-minus-dynamic-PSLAP contrast"
            ),
        ],
        "interpretation_boundaries": [
            "all outcome contrasts are conditional on one frozen REG checkpoint",
            (
                "the due-only cells diagnose commitment versus scheduler "
                "coupling; they are not additional assignment-source baselines"
            ),
            (
                "the constrained geometry measures inventory-induced egress "
                "obstruction in a single-agent yard, not concurrent traffic"
            ),
        ],
    }

    rows = [
        result_row(
            run,
            geometry_condition=run["geometry_condition"],
            selector_checkpoint_id=args.selector_checkpoint.name,
            selector_digest=selector_digest,
            scheduler_signature=group_signatures[
                (run["scheduler_variant"], run["assignment_commitment"])
            ],
        )
        for run in runs
    ]
    with paths["results"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {name: csv_value(row[name]) for name in CSV_FIELDS}
            for row in rows
        )

    cell_summaries = []
    for condition in GEOMETRY_CONDITIONS:
        for scheduler_variant, commitment, source in experiment_cells():
            cell_summaries.append(
                cell_summary(
                    runs,
                    {
                        "geometry_condition": condition,
                        "scheduler_variant": scheduler_variant,
                        "assignment_commitment": commitment,
                        "assignment_source": source,
                    },
                )
            )
    contrasts = build_contrasts(runs, args.bootstrap_samples)
    key_summary = _key_console_summary(contrasts)

    reservation_runs = [
        run
        for run in runs
        if run["assignment_commitment"] == DECISION_EPOCH_RESERVED
    ]
    method_failures = [
        {
            "episode_seed": run["eval_seed"],
            "geometry_condition": run["geometry_condition"],
            "scheduler_variant": run["scheduler_variant"],
            "assignment_commitment": run["assignment_commitment"],
            "assignment_source": run["assignment_source"],
            "reason": run["method_failure_reason"],
        }
        for run in runs
        if run["method_failure_reason"] is not None
    ]
    audit = {
        "protocol": protocol,
        "expected_run_count": expected,
        "observed_run_count": len(runs),
        "complete_unique_cell_grid": len(observed_keys) == expected,
        "common_strict_contract": common_contract,
        "common_strict_contract_signature": invariant_signature(
            common_contract
        ),
        "scheduler_group_contracts": {
            f"{scheduler}__{commitment}": {
                "signature": group_signatures[(scheduler, commitment)],
                "contract": contract,
            }
            for (scheduler, commitment), contract in group_contracts.items()
        },
        "assignment_source_contracts": source_contracts,
        "pairing": {
            "all_geometry_schedule_ids_match": all(
                instances[ORDINARY][seed].schedule_id
                == instances[EGRESS_CONSTRAINED][seed].schedule_id
                for seed in args.seeds
            ),
            "all_geometry_instance_ids_differ": all(
                instances[ORDINARY][seed].instance_id
                != instances[EGRESS_CONSTRAINED][seed].instance_id
                for seed in args.seeds
            ),
            "runs_per_schedule": 2 * sum(1 for _ in experiment_cells()),
        },
        "strict_protocol": {
            "all_zero_fallbacks": all(
                run["selector_audit"]["fallback_count"] == 0 for run in runs
            ),
            "all_strict_success": all(
                bool(run["strict_method_success"]) for run in runs
            ),
            "strict_success_count": int(
                sum(bool(run["strict_method_success"]) for run in runs)
            ),
            "invalid_assignment_count": int(
                sum(
                    run["selector_audit"]["invalid_assignment_count"]
                    for run in runs
                )
            ),
            "illegal_drop_count": int(
                sum(run["illegal_drops"] for run in runs)
            ),
            "method_failures": method_failures,
        },
        "reserved_cell_integrity": {
            "all_reserved_runs_integrity": all(
                bool(run["reservation_integrity"])
                for run in reservation_runs
            ),
            "reserved_run_count": len(reservation_runs),
            "integrity_pass_count": int(
                sum(bool(run["reservation_integrity"]) for run in reservation_runs)
            ),
            "bound_count": int(
                sum(
                    run["scheduler_audit"]["reservation_bound_count"]
                    for run in reservation_runs
                )
            ),
            "commit_count": int(
                sum(
                    run["scheduler_audit"]["reservation_commit_count"]
                    for run in reservation_runs
                )
            ),
            "execution_match_count": int(
                sum(
                    run["scheduler_audit"][
                        "reservation_execution_match_count"
                    ]
                    for run in reservation_runs
                )
            ),
            "invalidation_count": int(
                sum(
                    run["scheduler_audit"]["reservation_invalidation_count"]
                    for run in reservation_runs
                )
            ),
        },
    }
    summary = {
        "protocol": protocol,
        "cell_summaries": cell_summaries,
        "paired_contrasts": contrasts,
        "key_summary": key_summary,
        "integrity_gate": {
            "all_reserved_runs_integrity": audit["reserved_cell_integrity"][
                "all_reserved_runs_integrity"
            ],
            "all_zero_fallbacks": audit["strict_protocol"][
                "all_zero_fallbacks"
            ],
            "all_strict_success": audit["strict_protocol"][
                "all_strict_success"
            ],
        },
    }
    full = {"protocol": protocol, "runs": runs}

    _write_json(paths["full"], full)
    _write_json(paths["summary"], summary)
    _write_json(paths["audit"], audit)

    print(json.dumps(json_safe(key_summary), indent=2), flush=True)
    print(f"Results: {paths['results']}", flush=True)
    print(f"Full: {paths['full']}", flush=True)
    print(f"Summary: {paths['summary']}", flush=True)
    print(f"Audit: {paths['audit']}", flush=True)


if __name__ == "__main__":
    main()
