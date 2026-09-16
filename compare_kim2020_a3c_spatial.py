#!/usr/bin/env python3
"""Matched-instance Track-B comparison for the Kim (2020) adaptation.

Stochastic Kim rollouts are averaged within a fixed schedule before any
cross-schedule summary or paired contrast.  Policy seeds and schedule seeds
are separate experimental factors, so repeated policy samples never masquerade
as independent workload instances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.kim2020_a3c_spatial import (
    CHECKPOINT_SCHEMA_VERSION,
    DEPLOYMENT_MAP,
    DEPLOYMENT_STOCHASTIC,
    geometry_contract as kim2020_geometry_contract,
    kim2020_deployment_digest,
)
from PSLAP.online_policy import (
    EXACT_RECOVERY_FALLBACK_CONTRACT,
    EXACT_RECOVERY_MAX_NODES,
)
from PSLAP.track_a import (
    TRACK_A_DYNAMIC,
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_NEAREST_FREE,
    TRACK_A_REG_SELECTOR_V5,
)
from track_b_urgency_evaluate import (
    DECISION_EPOCH_RESERVED,
    DURATION_AWARE_VARIANT,
    _evaluate_one,
    resolve_device,
    validate_kim2020_payload,
    validate_selector_payload,
)


COMPARISON_VERSION = "kim2020_spatial_matched_track_b_v3"
LOCKED_KIM_PROVENANCE_CONTRACT = (
    "kim2020_best_checkpoint_metadata_geometry_and_optional_digest_v1"
)
# This comparison protocol deliberately names the trainer-side contracts it
# accepts.  A future trainer revision must update the comparison version and
# this mapping explicitly instead of silently changing the reported baseline.
LOCKED_KIM_REQUIRED_METADATA = {
    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
    "checkpoint_kind": "best",
    "trainer_version": "kim2020_a3c_spatial_adapted_trainer_v7",
    "adaptation_contract": (
        "kim_jeong_shin_2020_spatial_a3c_conceptual_adaptation_v6"
    ),
    "validation_contract": (
        "fixed_instances_map_once_stochastic_fixed_policy_seed_rollouts_v1"
    ),
    "model_selection_contract": (
        "primary_stochastic_strict_success_then_mean_obstructive_moves_v2"
    ),
    "exact_recovery_fallback_contract": EXACT_RECOVERY_FALLBACK_CONTRACT,
    "exact_recovery_max_nodes": EXACT_RECOVERY_MAX_NODES,
    "adaptation_status": "adaptation_not_exact_reproduction",
    "exact_paper_reproduction": False,
}
KIM_STOCHASTIC = f"{TRACK_A_KIM2020_A3C_SPATIAL}__stochastic"
KIM_MAP = f"{TRACK_A_KIM2020_A3C_SPATIAL}__map"
PRIMARY_METRICS = (
    "obstructive_moves",
    "obstructive_moves_per_delivered_block",
    "return",
    "mean_absolute_error",
    "steps",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen Kim et al. (2020)-inspired spatial policy "
            "against matched assignment baselines in reserved Track B."
        )
    )
    parser.add_argument("--kim-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-kim-deployment-digest",
        help=(
            "optional SHA-256 deployment identity to pin for a locked final "
            "run; a mismatch is fatal"
        ),
    )
    parser.add_argument(
        "--reg-checkpoint",
        type=Path,
        help="optional REG-v5 baseline checkpoint",
    )
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument(
        "--schedule-seeds", type=int, nargs="+", required=True
    )
    parser.add_argument("--stochastic-rollouts", type=int, default=5)
    parser.add_argument("--policy-seed-base", type=int, default=20_000_000)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--exit-width", type=int)
    parser.add_argument("--number-blocks", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--lookahead-margin-steps", type=float, default=0.0)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--allow-regime-shift",
        action="store_true",
        help="allow checkpoint training lambda/mu to differ from evaluation",
    )
    parser.add_argument(
        "--allow-seed-overlap",
        action="store_true",
        help=(
            "diagnostic only: permit final schedule seeds that overlap "
            "checkpoint training/validation seeds or unverifiable provenance"
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.lam < 0.0 or args.mu <= 0.0:
        parser.error("lambda must be nonnegative and mu must be positive")
    positive = {
        "stochastic-rollouts": args.stochastic_rollouts,
        "grid-rows": args.grid_rows,
        "grid-cols": args.grid_cols,
        "number-blocks": args.number_blocks,
        "max-steps": args.max_steps,
        "max-defer-steps": args.max_defer_steps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error("these arguments must be positive: " + ", ".join(invalid))
    if args.grid_rows < 4 or args.grid_cols < 4:
        parser.error("grid dimensions must both be at least 4")
    if args.exit_width is not None and not 1 <= args.exit_width <= args.grid_cols - 2:
        parser.error("exit-width must be in [1, grid-cols - 2]")
    if args.lookahead_margin_steps < 0.0 or not np.isfinite(
        args.lookahead_margin_steps
    ):
        parser.error("lookahead-margin-steps must be finite and nonnegative")
    if args.policy_seed_base < 0:
        parser.error("policy-seed-base must be nonnegative")
    if len(set(args.schedule_seeds)) != len(args.schedule_seeds):
        parser.error("schedule-seeds must be unique")
    if any(seed < 0 for seed in args.schedule_seeds):
        parser.error("schedule-seeds must be nonnegative")
    if args.expected_kim_deployment_digest is not None:
        digest = args.expected_kim_deployment_digest.lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            parser.error(
                "expected-kim-deployment-digest must be a 64-character "
                "hexadecimal SHA-256 digest"
            )
        args.expected_kim_deployment_digest = digest
    return args


def _make_template_env(args):
    return make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=args.exit_width,
        number_blocks=args.number_blocks,
    )


def _load_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return payload


def validate_locked_kim2020_checkpoint(
    payload: dict,
    *,
    env,
    requested_exit_width: int | None,
    expected_deployment_digest: str | None = None,
) -> dict:
    """Reject provenance drift before a registered matched comparison."""

    metadata_mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in LOCKED_KIM_REQUIRED_METADATA.items()
        if payload.get(key) != expected
    }
    expected_geometry_contract = kim2020_geometry_contract(env)
    observed_geometry_contract = payload.get("geometry_contract")
    geometry_contract_match = (
        observed_geometry_contract == expected_geometry_contract
    )
    expected_geometry_metadata = geometry_metadata(
        env, requested_exit_width=requested_exit_width
    )
    observed_geometry_metadata = payload.get("geometry")
    geometry_metadata_match = (
        observed_geometry_metadata == expected_geometry_metadata
    )
    observed_digest = kim2020_deployment_digest(payload)
    expected_digest = (
        expected_deployment_digest.lower()
        if expected_deployment_digest is not None
        else None
    )
    digest_match = (
        None if expected_digest is None else observed_digest == expected_digest
    )

    quality_mismatches = []
    training_seed = payload.get("training_seed")
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        quality_mismatches.append("training_seed")
    completed_training_episodes = payload.get("completed_training_episodes")
    if (
        not isinstance(completed_training_episodes, int)
        or isinstance(completed_training_episodes, bool)
        or completed_training_episodes <= 0
    ):
        quality_mismatches.append("completed_training_episodes")
    validation_seeds = payload.get("validation_instance_seeds")
    if (
        not isinstance(validation_seeds, (list, tuple))
        or not validation_seeds
        or len(set(validation_seeds)) != len(validation_seeds)
    ):
        quality_mismatches.append("validation_instance_seeds")

    validation = payload.get("validation_evaluation")
    if not isinstance(validation, dict):
        quality_mismatches.append("validation_evaluation")
    else:
        selection = validation.get("selection", {})
        if selection.get("contract") != LOCKED_KIM_REQUIRED_METADATA[
            "model_selection_contract"
        ]:
            quality_mismatches.append("validation.selection.contract")
        if selection.get("eligible") is not True:
            quality_mismatches.append("validation.selection.eligible")
        if selection.get("primary_deployment_mode") != DEPLOYMENT_STOCHASTIC:
            quality_mismatches.append(
                "validation.selection.primary_deployment_mode"
            )
        if validation.get("validation_instance_seeds") != validation_seeds:
            quality_mismatches.append("validation.validation_instance_seeds")
        for mode in (DEPLOYMENT_STOCHASTIC, DEPLOYMENT_MAP):
            mode_summary = validation.get(mode, {})
            if mode_summary.get("all_strict_method_success") is not True:
                quality_mismatches.append(
                    f"validation.{mode}.all_strict_method_success"
                )
            for metric in (
                "mean_invalid_assignment_count",
                "mean_fallback_count",
                "mean_retrieval_live_plan_failure_count",
                "mean_exact_recovery_failure_count",
            ):
                if mode_summary.get(metric) != 0.0:
                    quality_mismatches.append(
                        f"validation.{mode}.{metric}"
                    )

    problems = []
    if metadata_mismatches:
        problems.append(
            "metadata=" + ",".join(sorted(metadata_mismatches))
        )
    if not geometry_contract_match:
        problems.append("geometry_contract")
    if not geometry_metadata_match:
        problems.append("geometry_metadata")
    if digest_match is False:
        problems.append("deployment_digest")
    if quality_mismatches:
        problems.append(
            "quality=" + ",".join(sorted(quality_mismatches))
        )
    if problems:
        raise ValueError(
            "Kim checkpoint failed locked comparison provenance: "
            + "; ".join(problems)
        )

    return {
        "contract": LOCKED_KIM_PROVENANCE_CONTRACT,
        "verified": True,
        "required_metadata": dict(LOCKED_KIM_REQUIRED_METADATA),
        "metadata_mismatches": metadata_mismatches,
        "quality_mismatches": quality_mismatches,
        "selected_training_episode": completed_training_episodes,
        "training_seed": training_seed,
        "validation_instance_seeds": validation_seeds,
        "geometry_contract_match": geometry_contract_match,
        "geometry_metadata_match": geometry_metadata_match,
        "expected_deployment_digest": expected_digest,
        "observed_deployment_digest": observed_digest,
        "deployment_digest_pinned": expected_digest is not None,
        "deployment_digest_match": digest_match,
    }


def checkpoint_seed_audit(
    payload: dict,
    schedule_seeds: Iterable[int],
    *,
    default_training_seed_base: int | None = None,
) -> dict:
    """Audit final schedules against one checkpoint's train/validation seeds."""

    requested = tuple(int(seed) for seed in schedule_seeds)
    training_seed = payload.get("training_seed")
    completed = payload.get("completed_training_episodes")
    base = payload.get("training_instance_seed_base", default_training_seed_base)
    training_provenance_complete = all(
        value is not None for value in (training_seed, completed, base)
    )
    training_overlap = []
    training_range = None
    if training_provenance_complete:
        start = int(base) + int(training_seed) * 1_000_000 + 1
        end = start + int(completed) - 1
        training_range = [start, end]
        training_overlap = [seed for seed in requested if start <= seed <= end]

    validation_seeds = payload.get("validation_instance_seeds")
    if validation_seeds is None:
        validation_seeds = payload.get("validation_evaluation", {}).get("seeds")
    validation_provenance_complete = validation_seeds is not None
    validation_set = (
        {int(seed) for seed in validation_seeds}
        if validation_provenance_complete
        else set()
    )
    validation_overlap = [seed for seed in requested if seed in validation_set]
    verified = bool(
        training_provenance_complete and validation_provenance_complete
    )
    return {
        "verified": verified,
        "training_seed_range": training_range,
        "validation_seeds": sorted(validation_set),
        "training_overlap": training_overlap,
        "validation_overlap": validation_overlap,
        "disjoint": bool(
            verified and not training_overlap and not validation_overlap
        ),
    }


def enforce_seed_disjointness(audits: dict[str, dict], *, allow: bool) -> None:
    invalid = {
        name: audit
        for name, audit in audits.items()
        if not audit["disjoint"]
    }
    if invalid and not allow:
        details = "; ".join(
            f"{name}: verified={audit['verified']}, "
            f"training_overlap={audit['training_overlap']}, "
            f"validation_overlap={audit['validation_overlap']}"
            for name, audit in invalid.items()
        )
        raise ValueError(
            "final schedule seeds are not verified disjoint from checkpoint "
            f"training/validation data ({details}); use --allow-seed-overlap "
            "only for a diagnostic run"
        )


def _method_label(source: str, deployment_mode: str | None) -> str:
    if source == TRACK_A_KIM2020_A3C_SPATIAL:
        return KIM_STOCHASTIC if deployment_mode == DEPLOYMENT_STOCHASTIC else KIM_MAP
    return source


def _protocol_valid(run: dict) -> bool:
    selector = run["selector_audit"]
    return bool(
        run["strict_method_success"] == 1.0
        and run["reservation_integrity"]
        and run["illegal_drops"] == 0
        and selector["invalid_assignment_count"] == 0
        and selector.get("fallback_count", 0) == 0
    )


def _run_one(
    args,
    *,
    schedule_seed: int,
    instance,
    assignment_source: str,
    payload: dict | None,
    deployment_mode: str | None = None,
    policy_seed: int | None = None,
    rollout_index: int | None = None,
) -> dict:
    run = _evaluate_one(
        args,
        schedule_seed,
        payload,
        episode_instance=instance,
        scheduler_variant=DURATION_AWARE_VARIANT,
        assignment_source=assignment_source,
        source_neutral_scheduler=True,
        assignment_commitment=DECISION_EPOCH_RESERVED,
        deployment_mode=(
            DEPLOYMENT_MAP if deployment_mode is None else deployment_mode
        ),
        policy_seed=policy_seed,
    )
    run["comparison_method"] = _method_label(
        assignment_source, deployment_mode
    )
    run["schedule_seed"] = int(schedule_seed)
    run["rollout_index"] = rollout_index
    run["protocol_valid"] = _protocol_valid(run)
    run["obstructive_moves_per_delivered_block"] = (
        float(run["obstructive_moves"]) / max(1, int(run["delivery_count"]))
    )
    return run


def _mean(rows: Iterable[dict], key: str):
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def aggregate_within_instance(raw_runs: list[dict]) -> list[dict]:
    """Collapse policy rolls before treating a schedule as one observation."""

    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in raw_runs:
        key = (run["comparison_method"], run["instance_id"])
        grouped.setdefault(key, []).append(run)
    rows = []
    for (method, instance_id), group in sorted(grouped.items()):
        schedule_seeds = {int(item["schedule_seed"]) for item in group}
        if len(schedule_seeds) != 1:
            raise RuntimeError("one instance_id mapped to multiple schedule seeds")
        policy_seeds = sorted(
            int(item["assignment_policy_seed"])
            for item in group
            if item.get("assignment_policy_seed") is not None
        )
        row = {
            "comparison_method": method,
            "instance_id": instance_id,
            "schedule_id": group[0]["schedule_id"],
            "schedule_seed": next(iter(schedule_seeds)),
            "rollout_count": len(group),
            "policy_seeds": policy_seeds,
            "protocol_valid": bool(all(item["protocol_valid"] for item in group)),
            "success": _mean(group, "success"),
            "strict_method_success": _mean(group, "strict_method_success"),
            "reservation_integrity_rate": float(
                np.mean([item["reservation_integrity"] for item in group])
            ),
            "invalid_assignment_count": int(
                sum(
                    item["selector_audit"]["invalid_assignment_count"]
                    for item in group
                )
            ),
            "fallback_count": int(
                sum(item["selector_audit"].get("fallback_count", 0) for item in group)
            ),
            "illegal_drops": int(sum(item["illegal_drops"] for item in group)),
        }
        for metric in PRIMARY_METRICS:
            row[metric] = _mean(group, metric)
        rows.append(row)
    return rows


def summarize_method(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty method")
    eligible_rows = [row for row in rows if row["protocol_valid"]]
    summary = {
        "schedule_count": len(rows),
        "eligible_schedule_count": len(eligible_rows),
        "censored_schedule_count": len(rows) - len(eligible_rows),
        "all_protocol_valid": bool(all(row["protocol_valid"] for row in rows)),
        "protocol_valid_rate": float(
            np.mean([row["protocol_valid"] for row in rows])
        ),
        "strict_method_success_rate": _mean(rows, "strict_method_success"),
        "success_rate": _mean(rows, "success"),
        "total_invalid_assignments": int(
            sum(row["invalid_assignment_count"] for row in rows)
        ),
        "total_fallbacks": int(sum(row["fallback_count"] for row in rows)),
        "total_illegal_drops": int(sum(row["illegal_drops"] for row in rows)),
    }
    for metric in PRIMARY_METRICS:
        values = np.asarray(
            [
                row[metric]
                for row in eligible_rows
                if row.get(metric) is not None
                and np.isfinite(float(row[metric]))
            ],
            dtype=float,
        )
        mean_key = metric if metric.startswith("mean_") else f"mean_{metric}"
        summary[mean_key] = (
            float(values.mean()) if len(values) else None
        )
        summary[f"std_{metric}"] = (
            float(values.std(ddof=1))
            if len(values) > 1
            else (0.0 if len(values) else None)
        )
    return summary


def paired_contrast(
    per_instance: list[dict], reference: str, comparator: str
) -> dict:
    reference_rows = {
        row["instance_id"]: row
        for row in per_instance
        if row["comparison_method"] == reference
    }
    comparator_rows = {
        row["instance_id"]: row
        for row in per_instance
        if row["comparison_method"] == comparator
    }
    if set(reference_rows) != set(comparator_rows):
        raise RuntimeError(
            f"paired instance mismatch: {reference} versus {comparator}"
        )
    schedule_rows = []
    for instance_id in sorted(reference_rows):
        left = reference_rows[instance_id]
        right = comparator_rows[instance_id]
        if left["schedule_id"] != right["schedule_id"]:
            raise RuntimeError("paired rows have different schedule IDs")
        row = {
            "instance_id": instance_id,
            "schedule_id": left["schedule_id"],
            "schedule_seed": left["schedule_seed"],
            "reference_protocol_valid": bool(left["protocol_valid"]),
            "comparator_protocol_valid": bool(right["protocol_valid"]),
            "pair_eligible": bool(
                left["protocol_valid"] and right["protocol_valid"]
            ),
        }
        for metric in PRIMARY_METRICS:
            left_value = left.get(metric)
            right_value = right.get(metric)
            row[f"delta_{metric}"] = (
                float(left_value - right_value)
                if row["pair_eligible"]
                and left_value is not None
                and right_value is not None
                else None
            )
        schedule_rows.append(row)
    eligible_rows = [row for row in schedule_rows if row["pair_eligible"]]
    relocation_deltas = np.asarray(
        [row["delta_obstructive_moves"] for row in eligible_rows], dtype=float
    )
    result = {
        "reference": reference,
        "comparator": comparator,
        "delta_definition": "reference_minus_comparator",
        "schedule_count": len(schedule_rows),
        "eligible_schedule_count": len(eligible_rows),
        "censored_schedule_count": len(schedule_rows) - len(eligible_rows),
        "reference_wins": int(np.sum(relocation_deltas < 0.0)),
        "ties": int(np.sum(relocation_deltas == 0.0)),
        "reference_losses": int(np.sum(relocation_deltas > 0.0)),
        "schedule_rows": schedule_rows,
    }
    for metric in PRIMARY_METRICS:
        values = np.asarray(
            [
                row[f"delta_{metric}"]
                for row in eligible_rows
                if row[f"delta_{metric}"] is not None
            ],
            dtype=float,
        )
        result[f"mean_delta_{metric}"] = (
            float(values.mean()) if len(values) else None
        )
        result[f"std_delta_{metric}"] = (
            float(values.std(ddof=1))
            if len(values) > 1
            else (0.0 if len(values) else None)
        )
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    args = parse_args(argv)
    # Attributes consumed by the shared evaluator but intentionally fixed for
    # this registered comparison protocol.
    args.assignment_commitment = DECISION_EPOCH_RESERVED
    args.allow_selector_regime_shift = args.allow_regime_shift
    args.instance = None
    args.save_instances_dir = None

    kim_payload = _load_payload(args.kim_checkpoint)
    validate_kim2020_payload(
        kim_payload,
        lam=args.lam,
        mu=args.mu,
        allow_regime_shift=args.allow_regime_shift,
    )
    template_env = _make_template_env(args)
    kim_provenance_audit = validate_locked_kim2020_checkpoint(
        kim_payload,
        env=template_env,
        requested_exit_width=args.exit_width,
        expected_deployment_digest=args.expected_kim_deployment_digest,
    )
    reg_payload = None
    if args.reg_checkpoint is not None:
        reg_payload = _load_payload(args.reg_checkpoint)
        validate_selector_payload(
            reg_payload,
            lam=args.lam,
            mu=args.mu,
            allow_regime_shift=args.allow_regime_shift,
        )
    seed_audits = {
        "kim2020": checkpoint_seed_audit(
            kim_payload, args.schedule_seeds
        )
    }
    if reg_payload is not None:
        # REG-v5's trainer uses the historical fixed 200,000 seed namespace.
        seed_audits["reg_v5"] = checkpoint_seed_audit(
            reg_payload,
            args.schedule_seeds,
            default_training_seed_base=200_000,
        )
    enforce_seed_disjointness(
        seed_audits, allow=bool(args.allow_seed_overlap)
    )
    resolve_device(args.device)

    instances = [
        (seed, template_env.sample_episode_instance(seed))
        for seed in args.schedule_seeds
    ]
    for _, instance in instances:
        instance.validate_for(template_env)

    raw_runs = []
    for schedule_seed, instance in instances:
        for rollout_index in range(args.stochastic_rollouts):
            raw_runs.append(
                _run_one(
                    args,
                    schedule_seed=schedule_seed,
                    instance=instance,
                    assignment_source=TRACK_A_KIM2020_A3C_SPATIAL,
                    payload=kim_payload,
                    deployment_mode=DEPLOYMENT_STOCHASTIC,
                    policy_seed=args.policy_seed_base + rollout_index,
                    rollout_index=rollout_index,
                )
            )
        raw_runs.append(
            _run_one(
                args,
                schedule_seed=schedule_seed,
                instance=instance,
                assignment_source=TRACK_A_KIM2020_A3C_SPATIAL,
                payload=kim_payload,
                deployment_mode=DEPLOYMENT_MAP,
                policy_seed=args.policy_seed_base + args.stochastic_rollouts,
                rollout_index=0,
            )
        )
        for source, payload in (
            (TRACK_A_DYNAMIC, None),
            (TRACK_A_NEAREST_FREE, None),
            (TRACK_A_REG_SELECTOR_V5, reg_payload),
        ):
            if source == TRACK_A_REG_SELECTOR_V5 and payload is None:
                continue
            raw_runs.append(
                _run_one(
                    args,
                    schedule_seed=schedule_seed,
                    instance=instance,
                    assignment_source=source,
                    payload=payload,
                    rollout_index=0,
                )
            )

    per_instance = aggregate_within_instance(raw_runs)
    methods = sorted({row["comparison_method"] for row in per_instance})
    summaries = {
        method: summarize_method(
            [row for row in per_instance if row["comparison_method"] == method]
        )
        for method in methods
    }
    contrasts = {
        comparator: paired_contrast(per_instance, KIM_STOCHASTIC, comparator)
        for comparator in methods
        if comparator != KIM_STOCHASTIC
    }
    geometry = geometry_metadata(
        template_env, requested_exit_width=args.exit_width
    )
    payload = json_safe(
        {
            "config": {
                "comparison_version": COMPARISON_VERSION,
                "adaptation_not_exact_reproduction": True,
                "paper_doi": "10.1080/00207543.2020.1748247",
                "primary_method": KIM_STOCHASTIC,
                "secondary_method": KIM_MAP,
                "scheduler_variant": DURATION_AWARE_VARIANT,
                "source_neutral_scheduler": True,
                "assignment_commitment": DECISION_EPOCH_RESERVED,
                "stochastic_aggregation": "within_instance_mean_before_pairing_v1",
                "kim_checkpoint": str(args.kim_checkpoint.resolve()),
                "kim_deployment_digest": kim2020_deployment_digest(kim_payload),
                "kim_checkpoint_provenance_audit": kim_provenance_audit,
                "reg_checkpoint": (
                    str(args.reg_checkpoint.resolve())
                    if args.reg_checkpoint is not None
                    else None
                ),
                "reg_deployment_digest": (
                    selector_deployment_digest(reg_payload)
                    if reg_payload is not None
                    else None
                ),
                "lambda": args.lam,
                "mu": args.mu,
                "geometry": geometry,
                "schedule_seeds": args.schedule_seeds,
                "policy_seeds": [
                    args.policy_seed_base + index
                    for index in range(args.stochastic_rollouts)
                ],
                "stochastic_rollouts": args.stochastic_rollouts,
                "max_steps": args.max_steps,
                "max_defer_steps": args.max_defer_steps,
                "lookahead_margin_steps": args.lookahead_margin_steps,
                "target_window": args.target_window,
                "device": str(resolve_device(args.device)),
                "allow_regime_shift": args.allow_regime_shift,
                "seed_disjointness_audit": seed_audits,
                "seed_overlap_override": bool(args.allow_seed_overlap),
            },
            "summaries": summaries,
            "paired_contrasts": contrasts,
            "per_instance": per_instance,
            "raw_runs": raw_runs,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "raw_run_count": len(raw_runs),
                "independent_schedule_count": len(args.schedule_seeds),
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
