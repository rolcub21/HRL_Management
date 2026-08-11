#!/usr/bin/env python3
"""Paired Track-B assignment-source test under ordinary and narrow egress.

The environment has one moving agent, so "contention" here means inventory-
induced path obstruction at a constrained egress, not concurrent traffic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compare_track_b_assignment_sources import (
    CSV_FIELDS as ASSIGNMENT_CSV_FIELDS,
    PAIR_METRIC_SPECIFICATIONS,
    bootstrap_ci,
    csv_value,
    invariant_signature,
    json_safe,
    metric_value,
    paired_comparisons,
    result_row,
    scheduler_invariants,
    source_summary,
)
from example.Options.selector_v5 import TRACK_B_ASSIGNMENT_SOURCES
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.track_a import TRACK_A_DYNAMIC, TRACK_A_REG_SELECTOR_V5
from track_b_urgency_evaluate import (
    ASSIGNMENT_COMMITMENTS,
    DECISION_EPOCH_RESERVED,
    evaluate_assignment_ablation_one,
    resolve_device,
    validate_selector_payload,
)


PROTOCOL_VERSION = "track_b_egress_assignment_interaction_v2"
TRACK = "B_egress_assignment_interaction"
ORDINARY = "ordinary"
EGRESS_CONSTRAINED = "egress_constrained"
GEOMETRY_CONDITIONS = (ORDINARY, EGRESS_CONSTRAINED)
RESULTS_NAME = "contention-results.csv"
SUMMARY_NAME = "contention-summary.json"
AUDIT_NAME = "contention-audit.json"
CSV_FIELDS = ("geometry_condition", *ASSIGNMENT_CSV_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired ordinary-versus-egress-constrained 2x3 Track-B "
            "assignment-source experiment"
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
    parser.add_argument(
        "--primary-comparison-source",
        choices=TRACK_B_ASSIGNMENT_SOURCES,
        default=TRACK_A_DYNAMIC,
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument(
        "--constrained-exit-width",
        type=int,
        default=2,
        help="right-aligned bottom-gate width for the constrained condition",
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
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.assignment_sources = tuple(dict.fromkeys(args.assignment_sources))
    args.seeds = tuple(dict.fromkeys(args.seeds))
    if args.reference_assignment_source not in args.assignment_sources:
        parser.error("reference assignment source must be included")
    if args.primary_comparison_source not in args.assignment_sources:
        parser.error("primary comparison source must be included")
    if args.primary_comparison_source == args.reference_assignment_source:
        parser.error("primary comparison source must differ from reference")
    if len(args.assignment_sources) < 2:
        parser.error("at least two assignment sources are required")
    if not args.seeds:
        parser.error("at least one seed is required")
    if args.max_steps <= 0 or args.max_defer_steps <= 0:
        parser.error("max-steps and max-defer-steps must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    if (
        not np.isfinite(args.lookahead_margin_steps)
        or args.lookahead_margin_steps < 0.0
    ):
        parser.error("lookahead-margin-steps must be finite and nonnegative")
    try:
        ordinary = make_condition_env(args, ORDINARY)
        constrained = make_condition_env(args, EGRESS_CONSTRAINED)
    except ValueError as exc:
        parser.error(str(exc))
    ordinary_width = len(ordinary.exit_cells)
    constrained_width = len(constrained.exit_cells)
    if constrained_width >= ordinary_width:
        parser.error(
            "constrained-exit-width must be narrower than the ordinary "
            f"gate ({ordinary_width})"
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
        for condition in GEOMETRY_CONDITIONS:
            instance = make_condition_env(args, condition).sample_episode_instance(
                seed
            )
            instances[condition][seed] = instance
        ordinary = instances[ORDINARY][seed]
        constrained = instances[EGRESS_CONSTRAINED][seed]
        if ordinary.schedule_id != constrained.schedule_id:
            raise RuntimeError(
                f"geometry changed the stochastic schedule for seed {seed}"
            )
        if ordinary.instance_id == constrained.instance_id:
            raise RuntimeError(
                f"ordinary and constrained geometries coincide for seed {seed}"
            )
    return instances


def oriented_delta(
    reference_run: dict,
    comparison_run: dict,
    field: str,
    orientation: str,
) -> float | None:
    reference_value = metric_value(reference_run, field)
    comparison_value = metric_value(comparison_run, field)
    if reference_value is None or comparison_value is None:
        return None
    reference_value = float(reference_value)
    comparison_value = float(comparison_value)
    if not np.isfinite(reference_value) or not np.isfinite(comparison_value):
        return None
    if orientation == "reference_minus_comparison":
        return reference_value - comparison_value
    if orientation == "comparison_minus_reference":
        return comparison_value - reference_value
    raise ValueError(f"unknown orientation: {orientation!r}")


def geometry_interactions(
    runs: list[dict],
    sources: tuple[str, ...],
    reference: str,
    samples: int,
) -> list[dict]:
    by_key = {
        (
            run["geometry_condition"],
            run["assignment_source"],
            run["schedule_id"],
        ): run
        for run in runs
    }
    schedule_sets = {
        (condition, source): {
            run["schedule_id"]
            for run in runs
            if run["geometry_condition"] == condition
            and run["assignment_source"] == source
        }
        for condition in GEOMETRY_CONDITIONS
        for source in sources
    }
    rng = np.random.default_rng(20260805)
    comparisons = []
    for source in sources:
        if source == reference:
            continue
        shared = set.intersection(
            *(
                schedule_sets[(condition, item)]
                for condition in GEOMETRY_CONDITIONS
                for item in (reference, source)
            )
        )
        schedule_ids = sorted(shared)
        metrics = {}
        for metric, (field, orientation) in PAIR_METRIC_SPECIFICATIONS.items():
            records = []
            interactions = []
            for schedule_id in schedule_ids:
                ordinary_reference = by_key[(ORDINARY, reference, schedule_id)]
                ordinary_comparison = by_key[(ORDINARY, source, schedule_id)]
                constrained_reference = by_key[
                    (EGRESS_CONSTRAINED, reference, schedule_id)
                ]
                constrained_comparison = by_key[
                    (EGRESS_CONSTRAINED, source, schedule_id)
                ]
                ordinary_delta = oriented_delta(
                    ordinary_reference,
                    ordinary_comparison,
                    field,
                    orientation,
                )
                constrained_delta = oriented_delta(
                    constrained_reference,
                    constrained_comparison,
                    field,
                    orientation,
                )
                if ordinary_delta is None or constrained_delta is None:
                    continue
                interaction = constrained_delta - ordinary_delta
                interactions.append(interaction)
                records.append(
                    {
                        "schedule_id": schedule_id,
                        "ordinary_instance_id": ordinary_reference[
                            "instance_id"
                        ],
                        "constrained_instance_id": constrained_reference[
                            "instance_id"
                        ],
                        "ordinary_reference_advantage": ordinary_delta,
                        "constrained_reference_advantage": constrained_delta,
                        "interaction": interaction,
                    }
                )
            metrics[metric] = {
                "n": len(interactions),
                "mean_interaction": (
                    float(np.mean(interactions)) if interactions else None
                ),
                "bootstrap_95_ci": (
                    bootstrap_ci(interactions, samples, rng)
                    if interactions
                    else None
                ),
                "interaction_wins_ties_losses": {
                    "wins": sum(value > 1e-12 for value in interactions),
                    "ties": sum(abs(value) <= 1e-12 for value in interactions),
                    "losses": sum(value < -1e-12 for value in interactions),
                },
                "per_schedule": records,
            }
        comparisons.append(
            {
                "reference_assignment_source": reference,
                "comparison_assignment_source": source,
                "primary_comparison": False,
                "n": len(schedule_ids),
                "schedule_ids": schedule_ids,
                "metric_orientation": (
                    "positive means the constrained geometry increases the "
                    "reference source's advantage"
                ),
                "metrics": metrics,
            }
        )
    return comparisons


def activation_summary(
    runs: list[dict], sources: tuple[str, ...], *, calibration_seed_count: int
) -> dict:
    cells = []
    for condition in GEOMETRY_CONDITIONS:
        for source in sources:
            selected = [
                run
                for run in runs
                if run["geometry_condition"] == condition
                and run["assignment_source"] == source
            ]
            relocations = [
                run["scheduler_audit"]["retrieve_relocations"]
                for run in selected
            ]
            replans = [
                run["scheduler_audit"]["retrieve_replans"]
                for run in selected
            ]
            delivered = sum(run["delivery_count"] for run in selected)
            manifests = sum(
                run["storage_flow_manifest_count"] for run in selected
            )
            cells.append(
                {
                    "geometry_condition": condition,
                    "assignment_source": source,
                    "episodes": len(selected),
                    "strict_method_success_rate": float(
                        np.mean(
                            [run["strict_method_success"] for run in selected]
                        )
                    ),
                    "delivery_completion_rate": delivered / max(1, manifests),
                    "total_retrieve_relocations": int(sum(relocations)),
                    "episodes_with_retrieve_relocation": int(
                        sum(value > 0 for value in relocations)
                    ),
                    "total_obstructive_moves": int(
                        sum(run["obstructive_moves"] for run in selected)
                    ),
                    "relocation_audit_matches_environment": bool(
                        sum(relocations)
                        == sum(run["obstructive_moves"] for run in selected)
                    ),
                    "total_retrieve_replans": int(sum(replans)),
                    "total_invalid_assignments": int(
                        sum(
                            run["selector_audit"]["invalid_assignment_count"]
                            for run in selected
                        )
                    ),
                    "total_fallbacks": int(
                        sum(
                            run["selector_audit"]["fallback_count"]
                            for run in selected
                        )
                    ),
                    "total_illegal_drops": int(
                        sum(run["illegal_drops"] for run in selected)
                    ),
                    "method_failure_count": int(
                        sum(
                            run["method_failure_reason"] is not None
                            for run in selected
                        )
                    ),
                }
            )

    constrained = [
        cell
        for cell in cells
        if cell["geometry_condition"] == EGRESS_CONSTRAINED
    ]
    minimum_relocation_episodes = max(
        1, math.ceil(0.6 * calibration_seed_count)
    )
    minimum_relocation_events = max(5, calibration_seed_count)
    integrity_pass = all(
        cell["strict_method_success_rate"] >= 0.8
        and cell["delivery_completion_rate"] >= 0.9
        and cell["total_invalid_assignments"] == 0
        and cell["total_fallbacks"] == 0
        and cell["total_illegal_drops"] == 0
        and cell["relocation_audit_matches_environment"]
        for cell in constrained
    )
    pooled_relocations = sum(
        cell["total_retrieve_relocations"] for cell in constrained
    )
    pooled_relocation_episodes = sum(
        cell["episodes_with_retrieve_relocation"] for cell in constrained
    )
    return {
        "cells": cells,
        "calibration_screen": {
            "selection_uses_performance_metrics": False,
            "minimum_strict_success_rate_per_source": 0.8,
            "minimum_delivery_completion_rate_per_source": 0.9,
            "require_zero_invalid_fallback_illegal_events": True,
            "minimum_pooled_relocation_events": minimum_relocation_events,
            "minimum_pooled_episodes_with_relocation": (
                minimum_relocation_episodes
            ),
            "pooled_relocation_events": pooled_relocations,
            "pooled_episodes_with_relocation": pooled_relocation_episodes,
            "sources_with_relocation": sum(
                cell["total_retrieve_relocations"] > 0
                for cell in constrained
            ),
            "integrity_pass": integrity_pass,
            "outcome_activation_pass": bool(
                pooled_relocations >= minimum_relocation_events
                and pooled_relocation_episodes >= minimum_relocation_episodes
            ),
            "overall_pass": bool(
                integrity_pass
                and pooled_relocations >= minimum_relocation_events
                and pooled_relocation_episodes
                >= minimum_relocation_episodes
            ),
            "interpretation": (
                "pooled source-blind activation allows a method that prevents "
                "obstruction to retain zero relocations; source-specific "
                "activation remains visible in cells"
            ),
        },
    }


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
        condition_dir = instance_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        for seed, instance in instances[condition].items():
            (condition_dir / f"seed-{seed}.json").write_text(
                instance.to_json() + "\n"
            )

    runs = []
    fixed_scheduler_contract = None
    fixed_scheduler_signature = None
    source_contracts = {}
    for condition in GEOMETRY_CONDITIONS:
        runtime_args = SimpleNamespace(
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
            assignment_commitment=args.assignment_commitment,
            target_window=args.target_window,
            save_instances_dir=None,
        )
        for seed in args.seeds:
            instance = instances[condition][seed]
            for source in args.assignment_sources:
                run = evaluate_assignment_ablation_one(
                    runtime_args,
                    seed,
                    selector_payload,
                    assignment_source=source,
                    episode_instance=instance,
                )
                run["geometry_condition"] = condition
                if run["instance_id"] != instance.instance_id:
                    raise RuntimeError("matched EpisodeInstance was not consumed")
                if run["assignment_source"] != source:
                    raise RuntimeError("assignment source provenance mismatch")
                if run["schedule_id"] != instance.schedule_id:
                    raise RuntimeError("matched stochastic schedule was not consumed")
                if run["geometry"] != geometries[condition]:
                    raise RuntimeError("geometry provenance mismatch")
                if run["selector_audit"]["fallback_count"] != 0:
                    raise RuntimeError("assignment source used a fallback")

                scheduler_contract = scheduler_invariants(run)
                if fixed_scheduler_contract is None:
                    fixed_scheduler_contract = scheduler_contract
                    fixed_scheduler_signature = invariant_signature(
                        scheduler_contract
                    )
                elif scheduler_contract != fixed_scheduler_contract:
                    raise RuntimeError(
                        "non-assignment scheduler invariant changed"
                    )
                source_contract = (
                    run["assignment_source_family"],
                    run["assignment_source_version"],
                )
                previous = source_contracts.setdefault(source, source_contract)
                if source_contract != previous:
                    raise RuntimeError(
                        f"assignment source contract changed for {source}"
                    )
                runs.append(run)
                print(
                    f"[{condition}/{source}] seed={seed} "
                    f"R={run['return']:.2f} "
                    f"strict={run['strict_method_success']:.0f} "
                    f"steps={run['steps']} "
                    f"reloc={run['scheduler_audit']['retrieve_relocations']} "
                    f"MAE={run['mean_absolute_error']}",
                    flush=True,
                )

    expected = (
        len(GEOMETRY_CONDITIONS)
        * len(args.assignment_sources)
        * len(args.seeds)
    )
    observed = {
        (
            run["geometry_condition"],
            run["assignment_source"],
            run["schedule_id"],
        )
        for run in runs
    }
    if len(runs) != expected or len(observed) != expected:
        raise RuntimeError("geometry/source/schedule grid is incomplete")

    rows = []
    for run in runs:
        row = result_row(
            run,
            args,
            selector_checkpoint_id=args.selector_checkpoint.name,
            selector_digest=selector_digest,
            invariant_digest=fixed_scheduler_signature,
        )
        row["track"] = TRACK
        row["protocol_version"] = PROTOCOL_VERSION
        rows.append({"geometry_condition": run["geometry_condition"], **row})
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {key: csv_value(row[key]) for key in CSV_FIELDS} for row in rows
        )

    schedule_ids = {
        str(seed): instances[ORDINARY][seed].schedule_id for seed in args.seeds
    }
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "track": TRACK,
        "design": "paired_2_geometry_x_assignment_source_factorial_v1",
        "causal_intervention": (
            "bottom_egress_width_only_within_each_assignment_source"
        ),
        "geometry_conditions": geometries,
        "assignment_sources": list(args.assignment_sources),
        "assignment_commitment": args.assignment_commitment,
        "reference_assignment_source": args.reference_assignment_source,
        "primary_comparison_source": args.primary_comparison_source,
        "selector_checkpoint": str(args.selector_checkpoint.resolve()),
        "selector_deployment_digest": selector_digest,
        "selector_geometry_status": (
            "frozen_REG_v5_trained_on_ordinary_geometry_zero_shot_under_"
            "egress_constraint"
        ),
        "lambda": args.lam,
        "mu": args.mu,
        "seeds": list(args.seeds),
        "schedule_ids": schedule_ids,
        "within_geometry_pairing_key": "instance_id",
        "cross_geometry_pairing_key": "schedule_id",
        "max_steps": args.max_steps,
        "max_defer_steps": args.max_defer_steps,
        "lookahead_margin_steps": args.lookahead_margin_steps,
        "target_window": args.target_window,
        "device": str(device),
        "bootstrap_samples": args.bootstrap_samples,
        "scheduler_invariant_signature": fixed_scheduler_signature,
        "fixed_scheduler_contract": fixed_scheduler_contract,
        "assignment_source_contracts": {
            source: {"family": contract[0], "version": contract[1]}
            for source, contract in source_contracts.items()
        },
        "shared_candidate_mask": "shared_candidate_mask_v1",
        "assignment_failure_contract": (
            "invalid_or_missing_proposal_is_explicit_failure_no_fallback_v1"
        ),
        "preview_commit_contract": (
            "decision_epoch_proposal_bound_once_v2"
            if args.assignment_commitment == DECISION_EPOCH_RESERVED
            else "same_source_noncommitting_preview_then_post_pickup_recompute_v1"
        ),
        "interaction_estimand": (
            "(reference-comparison advantage under constrained egress) - "
            "(reference-comparison advantage under ordinary egress), paired "
            "by stochastic schedule"
        ),
        "interpretation_boundaries": [
            (
                "the environment has one moving agent; egress contention is "
                "inventory-induced path obstruction, not concurrent traffic"
            ),
            (
                "REG is frozen and was trained on ordinary geometry, so its "
                "constrained result is a zero-shot geometry-generalization test"
            ),
            (
                "each assignment source controls both scheduling preview and "
                "commitment; effects are source-scheduler interactions, not a "
                "pure final-layout intervention"
            ),
        ],
    }

    interactions = geometry_interactions(
        runs,
        args.assignment_sources,
        args.reference_assignment_source,
        args.bootstrap_samples,
    )
    for comparison in interactions:
        comparison["primary_comparison"] = bool(
            comparison["comparison_assignment_source"]
            == args.primary_comparison_source
        )
    summaries = []
    within_geometry = []
    for condition in GEOMETRY_CONDITIONS:
        selected = [
            run for run in runs if run["geometry_condition"] == condition
        ]
        for source in args.assignment_sources:
            item = source_summary(source, selected)
            item["geometry_condition"] = condition
            item["geometry"] = geometries[condition]
            summaries.append(item)
        within_geometry.append(
            {
                "geometry_condition": condition,
                "paired_comparisons": paired_comparisons(
                    selected,
                    args.assignment_sources,
                    args.reference_assignment_source,
                    args.bootstrap_samples,
                ),
            }
        )
    activation = activation_summary(
        runs, args.assignment_sources, calibration_seed_count=len(args.seeds)
    )
    summary = {
        "protocol": protocol,
        "geometry_source_summaries": summaries,
        "within_geometry_paired_comparisons": within_geometry,
        "cross_geometry_interactions": interactions,
        "manipulation_check": activation,
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
    console_summaries = [
        {
            "geometry_condition": item["geometry_condition"],
            "assignment_source": item["assignment_source"],
            "episodes": item["episodes"],
            "mean_return": item["mean_return"],
            "strict_method_success_rate": item[
                "strict_method_success_rate"
            ],
            "mean_steps": item["mean_steps"],
            "mean_absolute_error": item["mean_absolute_error"],
            "total_retrieve_relocations": item[
                "total_retrieve_relocations"
            ],
            "total_retrieve_replans": item["total_retrieve_replans"],
        }
        for item in summaries
    ]
    console_interactions = []
    console_metric_names = (
        "return_advantage",
        "absolute_error_reduction",
        "step_reduction",
        "retrieval_relocation_reduction",
        "retrieval_replan_reduction",
    )
    for comparison in interactions:
        console_interactions.append(
            {
                "comparison_assignment_source": comparison[
                    "comparison_assignment_source"
                ],
                "primary_comparison": comparison["primary_comparison"],
                "metrics": {
                    metric: {
                        "n": comparison["metrics"][metric]["n"],
                        "mean_interaction": comparison["metrics"][metric][
                            "mean_interaction"
                        ],
                        "bootstrap_95_ci": comparison["metrics"][metric][
                            "bootstrap_95_ci"
                        ],
                        "wins_ties_losses": comparison["metrics"][metric][
                            "interaction_wins_ties_losses"
                        ],
                    }
                    for metric in console_metric_names
                },
            }
        )
    print(json.dumps(json_safe(console_summaries), indent=2), flush=True)
    print(
        json.dumps(
            json_safe(
                {
                    "calibration_screen": activation["calibration_screen"],
                    "cross_geometry_interactions": console_interactions,
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    print(f"Results: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Audit: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
