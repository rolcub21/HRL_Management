#!/usr/bin/env python3
"""Matched assignment-source ablation under one fixed Track-B scheduler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from example.Options.selector_v5 import TRACK_B_ASSIGNMENT_SOURCES
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import summarize_storage_flow_runs
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.track_a import TRACK_A_REG_SELECTOR_V5
from track_b_urgency_evaluate import (
    ASSIGNMENT_COMMITMENTS,
    DECISION_EPOCH_RESERVED,
    evaluate_assignment_ablation_one,
    resolve_device,
    validate_selector_payload,
)


PROTOCOL_VERSION = "duration_aware_assignment_source_ablation_v3"
TRACK = "B_duration_aware_assignment_source_ablation"
RESULTS_NAME = "assignment-source-results.csv"
SUMMARY_NAME = "assignment-source-summary.json"
AUDIT_NAME = "assignment-source-audit.json"

TIMING_FIELDS = (
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "tardy_delivery_rate",
    "mean_tardiness_when_tardy",
    "p90_tardiness",
    "p90_absolute_error",
)

STORAGE_FLOW_FIELDS = (
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

INVARIANT_AUDIT_FIELDS = (
    "decision_epoch_contract",
    "retrieve_deliver_option_version",
    "retrieval_executor_version",
    "retrieval_start_contract",
    "ranking_eta_contract",
    "relocation_selector_version",
    "accept_store_option_version",
    "defer_option_version",
    "retrieval_termination_contract",
    "accept_duration_estimate_contract",
    "lookahead_cost_contract",
    "lookahead_margin_steps",
    "assignment_commitment_contract",
    "reservation_contract",
)

PAIR_METRIC_SPECIFICATIONS = {
    "return_advantage": ("return", "reference_minus_comparison"),
    "success_advantage": ("success", "reference_minus_comparison"),
    "strict_success_advantage": (
        "strict_method_success",
        "reference_minus_comparison",
    ),
    "step_reduction": ("steps", "comparison_minus_reference"),
    "absolute_error_reduction": (
        "mean_absolute_error",
        "comparison_minus_reference",
    ),
    "tardiness_reduction": (
        "mean_tardiness",
        "comparison_minus_reference",
    ),
    "within_window_advantage": (
        "within_target_window_rate",
        "reference_minus_comparison",
    ),
    "mean_storage_flow_time_reduction": (
        "mean_storage_flow_time",
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
    "obstructive_move_reduction": (
        "obstructive_moves",
        "comparison_minus_reference",
    ),
    "retrieval_relocation_reduction": (
        "retrieve_relocations",
        "comparison_minus_reference",
    ),
    "retrieval_replan_reduction": (
        "retrieve_replans",
        "comparison_minus_reference",
    ),
}

CSV_FIELDS = (
    "track",
    "protocol_version",
    "method_variant",
    "scheduler_method",
    "scheduler_architecture",
    "scheduler_policy",
    "controller_action_interface",
    "scheduler_invariant_signature",
    "assignment_commitment",
    "assignment_source",
    "assignment_source_family",
    "assignment_source_version",
    "assignment_source_learned",
    "assignment_checkpoint_id",
    "assignment_checkpoint_digest",
    "information_regime",
    "lambda",
    "mu",
    "episode_seed",
    "instance_id",
    "schedule_id",
    "geometry_regime",
    "geometry_signature",
    "grid_rows",
    "grid_cols",
    "requested_exit_width",
    "actual_exit_width",
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
    "commit_decision_count",
    "commit_valid_count",
    "commit_invalid_count",
    "infeasible_epoch_count",
    "fallback_count",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Swap only the assignment source under the fixed duration-aware "
            "Track-B scheduler"
        )
    )
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--assignment-sources",
        nargs="+",
        choices=TRACK_B_ASSIGNMENT_SOURCES,
        default=list(TRACK_B_ASSIGNMENT_SOURCES),
    )
    parser.add_argument(
        "--reference-assignment-source",
        choices=TRACK_B_ASSIGNMENT_SOURCES,
        default=TRACK_A_REG_SELECTOR_V5,
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--instances-dir",
        type=Path,
        help="load exact seed-<seed>.json instances instead of sampling",
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument(
        "--exit-width",
        type=int,
        help=(
            "right-aligned bottom-gate width; omit to use the environment's "
            "ordinary default gate"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--lookahead-margin-steps", type=float, default=2.0)
    parser.add_argument(
        "--assignment-commitment",
        choices=ASSIGNMENT_COMMITMENTS,
        default=DECISION_EPOCH_RESERVED,
    )
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=10_000
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--ga-seed-base", type=int, default=310_000)
    parser.add_argument("--rolling-population", type=int, default=16)
    parser.add_argument("--rolling-generations", type=int, default=10)
    parser.add_argument("--rolling-ga-egress-weight", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.assignment_sources = tuple(dict.fromkeys(args.assignment_sources))
    args.seeds = tuple(dict.fromkeys(args.seeds))
    if args.reference_assignment_source not in args.assignment_sources:
        parser.error("reference assignment source must be included")
    if len(args.assignment_sources) < 2:
        parser.error("at least two assignment sources are required")
    if not args.seeds:
        parser.error("at least one seed is required")
    if args.max_steps <= 0 or args.max_defer_steps <= 0:
        parser.error("max-steps and max-defer-steps must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    if args.rolling_population < 3:
        parser.error("rolling-population must be at least 3")
    if args.rolling_generations < 1:
        parser.error("rolling-generations must be positive")
    if args.rolling_ga_egress_weight < 1:
        parser.error("rolling-ga-egress-weight must be positive")
    try:
        make_env(args)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        not np.isfinite(args.lookahead_margin_steps)
        or args.lookahead_margin_steps < 0.0
    ):
        parser.error("lookahead-margin-steps must be finite and nonnegative")
    return args


def make_env(args) -> SmallRoomsEnv:
    return make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=args.exit_width,
    )


def load_instances(args) -> dict[int, EpisodeInstance]:
    instances = {}
    for seed in args.seeds:
        env = make_env(args)
        if args.instances_dir is None:
            instance = env.sample_episode_instance(seed)
        else:
            path = args.instances_dir / f"seed-{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing matched instance: {path}")
            instance = EpisodeInstance.from_json(path.read_text())
            if instance.seed is not None and instance.seed != seed:
                raise ValueError(
                    f"Instance seed mismatch for {path}: {instance.seed}"
                )
            instance.validate_for(env)
        instances[seed] = instance
    return instances


def scheduler_invariants(run: dict) -> dict:
    audit = run["urgency_scheduler_audit"]
    return {
        "scheduler_method": run["method"],
        "scheduler_architecture": run["controller_architecture"],
        "scheduler_policy": run["policy_realization"],
        "controller_action_interface": run["controller_action_interface"],
        **{name: audit[name] for name in INVARIANT_AUDIT_FIELDS},
    }


def invariant_signature(contract: dict) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def result_row(
    run: dict,
    args,
    *,
    selector_checkpoint_id: str,
    selector_digest: str,
    invariant_digest: str,
) -> dict:
    selector = run["selector_audit"]
    scheduler = run["urgency_scheduler_audit"]
    options = run["scheduler_audit"]
    is_reg = run["assignment_source"] == TRACK_A_REG_SELECTOR_V5
    online_assignment_seconds = float(
        selector.get("assignment_total_seconds", selector["assignment_seconds"])
    ) + float(scheduler["preview_seconds"])
    compact_audit = {
        "assignment_source": {
            key: value
            for key, value in selector.items()
            if key != "decisions"
        },
        "scheduler": {
            key: value
            for key, value in scheduler.items()
            if key not in ("decisions", "storage_flow_records")
        },
        "atomic_options": options,
    }
    retrieval = retrieval_outcome_summary([options])
    return {
        "track": TRACK,
        "protocol_version": PROTOCOL_VERSION,
        "method_variant": run["method_variant"],
        "scheduler_method": run["method"],
        "scheduler_architecture": run["controller_architecture"],
        "scheduler_policy": run["policy_realization"],
        "controller_action_interface": run["controller_action_interface"],
        "scheduler_invariant_signature": invariant_digest,
        "assignment_commitment": run["assignment_commitment"],
        "assignment_source": run["assignment_source"],
        "assignment_source_family": run["assignment_source_family"],
        "assignment_source_version": run["assignment_source_version"],
        "assignment_source_learned": selector[
            "assignment_source_learned"
        ],
        "assignment_checkpoint_id": (
            selector_checkpoint_id if is_reg else "not_applicable"
        ),
        "assignment_checkpoint_digest": (
            selector_digest if is_reg else "not_applicable"
        ),
        "information_regime": run["information_regime"],
        "lambda": args.lam,
        "mu": args.mu,
        "episode_seed": run["eval_seed"],
        "instance_id": run["instance_id"],
        "schedule_id": run["schedule_id"],
        "geometry_regime": run["geometry"]["geometry_regime"],
        "geometry_signature": run["geometry"]["geometry_signature"],
        "grid_rows": run["geometry"]["grid_rows"],
        "grid_cols": run["geometry"]["grid_cols"],
        "requested_exit_width": run["geometry"]["requested_exit_width"],
        "actual_exit_width": run["geometry"]["actual_exit_width"],
        "lookahead_margin_steps": scheduler["lookahead_margin_steps"],
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
        "commit_decision_count": selector["decision_count"],
        "commit_valid_count": selector["valid_assignment_count"],
        "commit_invalid_count": selector["invalid_assignment_count"],
        "infeasible_epoch_count": selector["infeasible_epoch_count"],
        "fallback_count": selector["fallback_count"],
        "preview_call_count": scheduler["preview_call_count"],
        "preview_failure_count": scheduler["preview_failure_count"],
        "preview_accept_count": scheduler["preview_accept_count"],
        "preview_cell_match_count": scheduler[
            "preview_cell_match_count"
        ],
        "preview_cell_comparison_count": scheduler[
            "preview_cell_comparison_count"
        ],
        "preview_cell_match_rate": scheduler["preview_cell_match_rate"],
        "mean_accept_duration_actual_minus_estimated_steps": scheduler[
            "mean_accept_duration_actual_minus_estimated_steps"
        ],
        "source_setup_seconds": run["source_setup_seconds"],
        "online_assignment_seconds": online_assignment_seconds,
        "episode_loop_seconds": run["decision_seconds"],
        "method_failure_reason": run["method_failure_reason"],
        "completed_storage_flow_times": run[
            "completed_storage_flow_times"
        ],
        "right_censored_storage_flow_ages": run[
            "right_censored_storage_flow_ages"
        ],
        "delivery_deviations": run["delivery_deviations"],
        "compact_audit": compact_audit,
    }


def csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            json_safe(value), separators=(",", ":"), allow_nan=False
        )
    return value


def finite_mean(values):
    finite = [float(value) for value in values if value is not None]
    finite = [value for value in finite if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def retrieval_outcome_summary(option_audits: list[dict]) -> dict:
    outcomes = [
        outcome
        for audit in option_audits
        for outcome in audit.get("retrieve_outcomes", ())
    ]

    def values(field):
        return [
            float(outcome[field])
            for outcome in outcomes
            if outcome.get(field) is not None
            and np.isfinite(float(outcome[field]))
        ]

    initial = values("initial_estimated_steps")
    actual = values("actual_steps")
    expansion = values("path_expansion_steps")
    return {
        "mean_retrieval_initial_estimated_steps": (
            float(np.mean(initial)) if initial else None
        ),
        "mean_retrieval_actual_steps": (
            float(np.mean(actual)) if actual else None
        ),
        "mean_retrieval_path_expansion_steps": (
            float(np.mean(expansion)) if expansion else None
        ),
        "p90_retrieval_path_expansion_steps": (
            float(np.quantile(expansion, 0.9)) if expansion else None
        ),
    }


def metric_value(run: dict, field: str):
    """Resolve top-level and atomic-option metrics uniformly."""

    if field in run:
        return run[field]
    if field in ("retrieve_relocations", "retrieve_replans"):
        return run["scheduler_audit"][field]
    raise KeyError(field)


def source_summary(source: str, runs: list[dict]) -> dict:
    selected = [run for run in runs if run["assignment_source"] == source]
    returns = np.asarray([run["return"] for run in selected], dtype=float)
    scheduler_audits = [run["urgency_scheduler_audit"] for run in selected]
    selector_audits = [run["selector_audit"] for run in selected]
    options = [run["scheduler_audit"] for run in selected]
    preview_comparisons = sum(
        audit["preview_cell_comparison_count"] for audit in scheduler_audits
    )
    preview_matches = sum(
        audit["preview_cell_match_count"] for audit in scheduler_audits
    )
    duration_errors = [
        value
        for audit in scheduler_audits
        for value in audit["accept_duration_actual_minus_estimated_steps"]
    ]
    return {
        "assignment_source": source,
        "assignment_source_family": selected[0]["assignment_source_family"],
        "assignment_source_version": selected[0][
            "assignment_source_version"
        ],
        "episodes": len(selected),
        "mean_return": float(returns.mean()),
        "return_std": (
            float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        ),
        "success_rate": float(np.mean([run["success"] for run in selected])),
        "strict_method_success_rate": float(
            np.mean([run["strict_method_success"] for run in selected])
        ),
        "mean_steps": float(np.mean([run["steps"] for run in selected])),
        "mean_absolute_error": finite_mean(
            run["mean_absolute_error"] for run in selected
        ),
        "mean_tardiness": finite_mean(
            run["mean_tardiness"] for run in selected
        ),
        "mean_earliness": finite_mean(
            run["mean_earliness"] for run in selected
        ),
        "within_target_window_rate": finite_mean(
            run["within_target_window_rate"] for run in selected
        ),
        **summarize_storage_flow_runs(selected),
        "total_obstructive_moves": int(
            sum(run["obstructive_moves"] for run in selected)
        ),
        "total_retrieve_relocations": int(
            sum(audit["retrieve_relocations"] for audit in options)
        ),
        "episodes_with_retrieve_relocation": int(
            sum(audit["retrieve_relocations"] > 0 for audit in options)
        ),
        "mean_retrieve_relocations_per_episode": float(
            np.mean([audit["retrieve_relocations"] for audit in options])
        ),
        "retrieve_relocations_per_delivery": (
            sum(audit["retrieve_relocations"] for audit in options)
            / sum(run["delivery_count"] for run in selected)
            if sum(run["delivery_count"] for run in selected)
            else None
        ),
        "total_retrieve_replans": int(
            sum(audit["retrieve_replans"] for audit in options)
        ),
        **retrieval_outcome_summary(options),
        "total_invalid_assignments": int(
            sum(audit["invalid_assignment_count"] for audit in selector_audits)
        ),
        "total_infeasible_epochs": int(
            sum(audit["infeasible_epoch_count"] for audit in selector_audits)
        ),
        "total_fallbacks": int(
            sum(audit["fallback_count"] for audit in selector_audits)
        ),
        "preview_call_count": int(
            sum(audit["preview_call_count"] for audit in scheduler_audits)
        ),
        "preview_failure_count": int(
            sum(audit["preview_failure_count"] for audit in scheduler_audits)
        ),
        "preview_cell_match_count": int(preview_matches),
        "preview_cell_comparison_count": int(preview_comparisons),
        "preview_cell_match_rate": (
            preview_matches / preview_comparisons
            if preview_comparisons
            else None
        ),
        "mean_accept_duration_actual_minus_estimated_steps": (
            float(np.mean(duration_errors)) if duration_errors else None
        ),
        "mean_online_assignment_seconds": float(
            np.mean(
                [
                    selector.get(
                        "assignment_total_seconds",
                        selector["assignment_seconds"],
                    )
                    + scheduler["preview_seconds"]
                    for selector, scheduler in zip(
                        selector_audits, scheduler_audits
                    )
                ]
            )
        ),
        "method_failures": [
            {
                "eval_seed": run["eval_seed"],
                "reason": run["method_failure_reason"],
            }
            for run in selected
            if run["method_failure_reason"] is not None
        ],
    }


def bootstrap_ci(values, samples: int, rng) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return [float(low), float(high)]


def paired_comparisons(
    runs: list[dict],
    sources: tuple[str, ...],
    reference: str,
    samples: int,
) -> list[dict]:
    by_source = {
        source: {
            run["instance_id"]: run
            for run in runs
            if run["assignment_source"] == source
        }
        for source in sources
    }
    reference_rows = by_source[reference]
    rng = np.random.default_rng(20260805)
    result = []
    for source in sources:
        if source == reference:
            continue
        shared = sorted(set(reference_rows) & set(by_source[source]))
        metrics = {}
        for metric, (field, orientation) in PAIR_METRIC_SPECIFICATIONS.items():
            deltas = []
            ids = []
            for instance_id in shared:
                reference_value = metric_value(reference_rows[instance_id], field)
                comparison_value = metric_value(
                    by_source[source][instance_id], field
                )
                if reference_value is None or comparison_value is None:
                    continue
                reference_value = float(reference_value)
                comparison_value = float(comparison_value)
                if not (
                    np.isfinite(reference_value)
                    and np.isfinite(comparison_value)
                ):
                    continue
                if orientation == "reference_minus_comparison":
                    delta = reference_value - comparison_value
                else:
                    delta = comparison_value - reference_value
                deltas.append(delta)
                ids.append(instance_id)
            metrics[metric] = {
                "n": len(deltas),
                "mean": float(np.mean(deltas)) if deltas else None,
                "bootstrap_95_ci": (
                    bootstrap_ci(deltas, samples, rng) if deltas else None
                ),
                "instance_ids": ids,
                "per_instance": [float(value) for value in deltas],
            }
        result.append(
            {
                "reference_assignment_source": reference,
                "comparison_assignment_source": source,
                "n": len(shared),
                "instance_ids": shared,
                "metric_orientation": (
                    "positive favors the reference assignment source"
                ),
                "metrics": metrics,
            }
        )
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    results_path = args.output_dir / RESULTS_NAME
    summary_path = args.output_dir / SUMMARY_NAME
    audit_path = args.output_dir / AUDIT_NAME
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{results_path} already exists; choose a new directory or "
            "use --overwrite"
        )

    selector_payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    validate_selector_payload(selector_payload, lam=args.lam, mu=args.mu)
    selector_digest = selector_deployment_digest(selector_payload)
    device = resolve_device(args.device)
    runtime_args = SimpleNamespace(
        lam=args.lam,
        mu=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=args.exit_width,
        instance=None,
        device=str(device),
        max_steps=args.max_steps,
        max_defer_steps=args.max_defer_steps,
        lookahead_margin_steps=args.lookahead_margin_steps,
        assignment_commitment=args.assignment_commitment,
        target_window=args.target_window,
        ga_seed_base=args.ga_seed_base,
        rolling_population=args.rolling_population,
        rolling_generations=args.rolling_generations,
        rolling_ga_egress_weight=args.rolling_ga_egress_weight,
        save_instances_dir=None,
    )

    instances = load_instances(args)
    experiment_geometry = geometry_metadata(
        make_env(args), requested_exit_width=args.exit_width
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_instances = args.output_dir / "instances"
    output_instances.mkdir(parents=True, exist_ok=True)
    for seed, instance in instances.items():
        (output_instances / f"seed-{seed}.json").write_text(
            instance.to_json() + "\n"
        )

    runs = []
    fixed_contract = None
    fixed_signature = None
    for seed in args.seeds:
        instance = instances[seed]
        for source in args.assignment_sources:
            run = evaluate_assignment_ablation_one(
                runtime_args,
                seed,
                selector_payload,
                assignment_source=source,
                episode_instance=instance,
            )
            if run["instance_id"] != instance.instance_id:
                raise RuntimeError(
                    f"{source} did not consume the matched EpisodeInstance"
                )
            if run["assignment_source"] != source:
                raise RuntimeError("assignment source provenance mismatch")
            if run["schedule_id"] != instance.schedule_id:
                raise RuntimeError("stochastic schedule provenance mismatch")
            if run["geometry"] != experiment_geometry:
                raise RuntimeError("geometry provenance mismatch")
            if run["selector_audit"]["fallback_count"] != 0:
                raise RuntimeError("Track-B assignment ablation used fallback")
            contract = scheduler_invariants(run)
            if fixed_contract is None:
                fixed_contract = contract
                fixed_signature = invariant_signature(contract)
            elif contract != fixed_contract:
                raise RuntimeError(
                    "non-assignment scheduler invariant changed across sources"
                )
            runs.append(run)
            print(
                f"[{source}] seed={seed} R={run['return']:.2f} "
                f"success={run['strict_method_success']:.0f} "
                f"steps={run['steps']} "
                f"MAE={run['mean_absolute_error']}",
                flush=True,
            )

    expected = len(args.seeds) * len(args.assignment_sources)
    observed_keys = {
        (run["assignment_source"], run["instance_id"]) for run in runs
    }
    if len(runs) != expected or len(observed_keys) != expected:
        raise RuntimeError("assignment-source/instance grid is incomplete")

    rows = [
        result_row(
            run,
            args,
            selector_checkpoint_id=args.selector_checkpoint.name,
            selector_digest=selector_digest,
            invariant_digest=fixed_signature,
        )
        for run in runs
    ]
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                key: csv_value(row[key])
                for key in CSV_FIELDS
            }
            for row in rows
        )

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "track": TRACK,
        "causal_intervention": "assignment_source_only",
        "assignment_commitment": args.assignment_commitment,
        "assignment_sources": list(args.assignment_sources),
        "reference_assignment_source": args.reference_assignment_source,
        "selector_checkpoint": str(args.selector_checkpoint.resolve()),
        "selector_deployment_digest": selector_digest,
        "lambda": args.lam,
        "mu": args.mu,
        "seeds": list(args.seeds),
        "instance_ids": {
            str(seed): instances[seed].instance_id for seed in args.seeds
        },
        "schedule_ids": {
            str(seed): instances[seed].schedule_id for seed in args.seeds
        },
        "geometry": experiment_geometry,
        "geometry_interpretation": (
            "egress geometry is fixed within this run; use the paired "
            "geometry runner for ordinary-versus-egress-constrained effects"
        ),
        "instance_origin": (
            str(args.instances_dir.resolve())
            if args.instances_dir is not None
            else "sampled_once_then_shared"
        ),
        "pairing_key": "instance_id",
        "max_steps": args.max_steps,
        "max_defer_steps": args.max_defer_steps,
        "lookahead_margin_steps": args.lookahead_margin_steps,
        "target_window": args.target_window,
        "device": str(device),
        "bootstrap_samples": args.bootstrap_samples,
        "rolling_ga_config": {
            "population_size": args.rolling_population,
            "generations": args.rolling_generations,
            "seed_contract": "ga_seed_base_plus_episode_seed",
            "ga_seed_base": args.ga_seed_base,
            "duration_aware_egress_weight": args.rolling_ga_egress_weight,
        },
        "scheduler_invariant_signature": fixed_signature,
        "fixed_scheduler_contract": fixed_contract,
        "shared_candidate_mask": "shared_candidate_mask_v1",
        "assignment_failure_contract": (
            "invalid_or_missing_proposal_is_explicit_failure_no_fallback_v1"
        ),
        "preview_commit_contract": (
            "decision_epoch_proposal_bound_once_v2"
            if args.assignment_commitment == DECISION_EPOCH_RESERVED
            else "same_source_noncommitting_preview_then_post_pickup_recompute_v1"
        ),
        "interpretation_boundary": (
            "differences estimate each assignment source's coupled preview-"
            "and-commit contribution under the fixed scheduler; they do not "
            "separately identify preview quality from final layout quality"
        ),
    }
    summary = {
        "protocol": protocol,
        "assignment_source_summaries": [
            source_summary(source, runs) for source in args.assignment_sources
        ],
        "paired_comparisons": paired_comparisons(
            runs,
            args.assignment_sources,
            args.reference_assignment_source,
            args.bootstrap_samples,
        ),
    }
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    audit_path.write_text(
        json.dumps(
            json_safe({"protocol": protocol, "runs": runs}),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(json_safe(summary["assignment_source_summaries"]), indent=2))
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
