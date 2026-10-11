#!/usr/bin/env python3
"""Evaluate frozen REG-v5 with deterministic urgency-first atomic scheduling."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from contention_metrics import contention_metric_record
from example.Options.selector_v5 import (
    make_track_b_assignment_option,
)
from example.Options.RetrieveDeliverOption import RetrieveDeliverOption
from example.controller_options import (
    build_controller_options,
    scheduler_episode_audit,
)
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import (
    summarize_block_storage_flow,
    summarize_delivery_timing,
    summarize_storage_flow_runs,
)
from example.helper.occupancy_pressure import (
    OccupancyPressureTracker,
    summarize_occupancy_pressure_runs,
)
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from example.urgency_scheduler import (
    DURATION_AWARE_ARCHITECTURE,
    DURATION_AWARE_METHOD,
    DURATION_AWARE_POLICY,
    AssignmentSourceReservedCellDurationAwareAtomicScheduler,
    AssignmentSourceDurationAwareAtomicScheduler,
    DurationAwareAtomicScheduler,
    SchedulerInfeasibleError,
    RESERVED_CELL_ACTION_INTERFACE,
    ReservedCellDurationAwareAtomicScheduler,
    ReservedCellUrgencyFirstAtomicScheduler,
    URGENCY_FIRST_ACTION_INTERFACE,
    URGENCY_FIRST_ARCHITECTURE,
    URGENCY_FIRST_METHOD,
    URGENCY_FIRST_POLICY,
    UrgencyFirstAtomicScheduler,
)
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.ga_policy import DEFAULT_ROLLING_GA_CONFIG
from PSLAP.kim2020_a3c_spatial import (
    DEPLOYMENT_MAP as KIM2020_DEPLOYMENT_MAP,
    DEPLOYMENT_STOCHASTIC as KIM2020_DEPLOYMENT_STOCHASTIC,
    METHOD_NAME as KIM2020_METHOD,
    kim2020_deployment_digest,
)
from PSLAP.track_a import (
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_REG_SELECTOR_V5,
)


METHOD = URGENCY_FIRST_METHOD
DUE_ONLY_VARIANT = "due_only"
DURATION_AWARE_VARIANT = "duration_aware"
SCHEDULER_VARIANTS = (DUE_ONLY_VARIANT, DURATION_AWARE_VARIANT)
POST_PICKUP_RECOMPUTE = "post_pickup_recompute"
DECISION_EPOCH_RESERVED = "decision_epoch_reserved"
ASSIGNMENT_COMMITMENTS = (
    POST_PICKUP_RECOMPUTE,
    DECISION_EPOCH_RESERVED,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_instance(path, env, seed: int) -> EpisodeInstance:
    if path is None:
        return env.sample_episode_instance(seed)
    instance = EpisodeInstance.from_json(Path(path).read_text())
    if instance.seed is not None and instance.seed != seed:
        raise ValueError("instance seed does not match evaluation seed")
    instance.validate_for(env)
    return instance


def validate_selector_payload(
    payload, *, lam: float, mu: float, allow_regime_shift: bool = False
) -> None:
    if payload.get("selector_feature_version") != 5:
        raise ValueError("Urgency scheduler requires a REG-v5 selector")
    regime = (payload.get("training_lambda"), payload.get("training_mu"))
    if regime != (lam, mu) and not allow_regime_shift:
        raise ValueError(
            "Frozen selector regime does not match evaluation: "
            f"selector={regime}, evaluation={(lam, mu)}"
        )


def validate_kim2020_payload(
    payload, *, lam: float, mu: float, allow_regime_shift: bool = False
) -> None:
    if payload.get("assignment_source") != KIM2020_METHOD:
        raise ValueError("Kim spatial evaluation requires a Kim checkpoint")
    regime = (payload.get("training_lambda"), payload.get("training_mu"))
    if regime != (lam, mu) and not allow_regime_shift:
        raise ValueError(
            "Frozen Kim policy regime does not match evaluation: "
            f"policy={regime}, evaluation={(lam, mu)}"
        )


def _failed_macro_outcome(option):
    outcome = getattr(option, "last_outcome", None)
    return (
        outcome
        if isinstance(outcome, dict) and outcome.get("success") is False
        else None
    )


def _evaluate_one(
    args,
    seed,
    selector_payload,
    *,
    episode_instance=None,
    scheduler_variant=DUE_ONLY_VARIANT,
    assignment_source=TRACK_A_REG_SELECTOR_V5,
    source_neutral_scheduler=False,
    assignment_commitment=None,
    deployment_mode=KIM2020_DEPLOYMENT_MAP,
    policy_seed=None,
):
    if assignment_commitment is None:
        assignment_commitment = getattr(
            args, "assignment_commitment", POST_PICKUP_RECOMPUTE
        )
    if assignment_commitment not in ASSIGNMENT_COMMITMENTS:
        raise ValueError(
            f"Unknown assignment commitment: {assignment_commitment!r}"
        )
    if assignment_source == TRACK_A_REG_SELECTOR_V5:
        if selector_payload is None:
            raise ValueError("REG-v5 assignment requires a selector payload")
        validate_selector_payload(
            selector_payload,
            lam=args.lam,
            mu=args.mu,
            allow_regime_shift=bool(
                getattr(args, "allow_selector_regime_shift", False)
            ),
        )
    elif assignment_source == TRACK_A_KIM2020_A3C_SPATIAL:
        if selector_payload is None:
            raise ValueError("Kim spatial assignment requires a checkpoint payload")
        validate_kim2020_payload(
            selector_payload,
            lam=args.lam,
            mu=args.mu,
            allow_regime_shift=bool(
                getattr(args, "allow_selector_regime_shift", False)
            ),
        )
    elif selector_payload is not None:
        raise ValueError(
            "deterministic assignment sources may not receive a learned payload"
        )
    seed_everything(seed)
    requested_exit_width = getattr(args, "exit_width", None)
    env = make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=getattr(args, "grid_rows", 10),
        grid_cols=getattr(args, "grid_cols", 10),
        exit_width=requested_exit_width,
        number_blocks=getattr(args, "number_blocks", 40),
    )
    instance = (
        load_instance(getattr(args, "instance", None), env, seed)
        if episode_instance is None
        else episode_instance
    )
    instance.validate_for(env)
    device = resolve_device(args.device)

    setup_started = perf_counter()
    selector = make_track_b_assignment_option(
        env,
        assignment_source,
        selector_payload=selector_payload,
        device=device,
        seed=seed,
        policy_seed=seed if policy_seed is None else policy_seed,
        deployment_mode=deployment_mode,
        rolling_ga_config=replace(
            DEFAULT_ROLLING_GA_CONFIG,
            population_size=int(getattr(args, "rolling_population", 16)),
            generations=int(getattr(args, "rolling_generations", 10)),
            elite_count=min(
                2, int(getattr(args, "rolling_population", 16)) - 1
            ),
            tournament_size=min(
                3, int(getattr(args, "rolling_population", 16))
            ),
            seed=int(getattr(args, "ga_seed_base", 310_000)) + int(seed),
        ),
        rolling_ga_egress_weight=int(
            getattr(args, "rolling_ga_egress_weight", 4)
        ),
    )
    controller_action_interface = (
        RESERVED_CELL_ACTION_INTERFACE
        if assignment_commitment == DECISION_EPOCH_RESERVED
        else URGENCY_FIRST_ACTION_INTERFACE
    )
    build_controller_options(
        env,
        selector,
        controller_action_interface=controller_action_interface,
        max_defer_steps=args.max_defer_steps,
    )
    if scheduler_variant == DUE_ONLY_VARIANT:
        scheduler = (
            ReservedCellUrgencyFirstAtomicScheduler(env)
            if assignment_commitment == DECISION_EPOCH_RESERVED
            else UrgencyFirstAtomicScheduler(env)
        )
    elif scheduler_variant == DURATION_AWARE_VARIANT:
        if assignment_commitment == DECISION_EPOCH_RESERVED:
            scheduler_class = (
                AssignmentSourceReservedCellDurationAwareAtomicScheduler
                if source_neutral_scheduler
                else ReservedCellDurationAwareAtomicScheduler
            )
        else:
            scheduler_class = (
                AssignmentSourceDurationAwareAtomicScheduler
                if source_neutral_scheduler
                else DurationAwareAtomicScheduler
            )
        scheduler = scheduler_class(
            env,
            margin_steps=float(
                getattr(args, "lookahead_margin_steps", 0.0)
            ),
        )
    else:
        raise ValueError(f"Unknown scheduler variant: {scheduler_variant!r}")
    setup_seconds = perf_counter() - setup_started

    state = env.reset(instance=instance)
    pressure = OccupancyPressureTracker(env)
    seed_everything(seed)
    current_option = None
    errors = []
    total_return = 0.0
    # ``relocated_block`` is a physical storage-to-storage PUTDOWN event.  In
    # this source-neutral atomic interface every such event must occur inside a
    # named RetrieveDeliverOption.  Count the event when it happens so a
    # truncation after the PUTDOWN cannot erase it from the audit.
    physical_storage_relocations = 0
    target_bound_obstruction_clearances = 0
    illegal_drops = 0
    steps = 0
    done = False
    method_failure_reason = None
    loop_started = perf_counter()

    while steps < args.max_steps and not done and method_failure_reason is None:
        if current_option is None:
            try:
                current_option = scheduler.select_action(state)
            except SchedulerInfeasibleError as exc:
                method_failure_reason = f"scheduler_infeasible:{exc}"
                break
        try:
            action = current_option.policy(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            pressure.observe_transition()
            selector.on_step(reward, info)
            total_return += float(reward)
            if "delivery_error_time" in info:
                errors.append(float(info["delivery_error_time"]))
            relocation_event = int(bool(info.get("relocated_block")))
            physical_storage_relocations += relocation_event
            if relocation_event:
                if not isinstance(current_option, RetrieveDeliverOption):
                    raise RuntimeError(
                        "physical storage relocation occurred outside the "
                        "named retrieval executor"
                    )
                target_bound_obstruction_clearances += relocation_event
            illegal_drops += int(bool(info.get("illegal_drop")))
            terminated = current_option.termination(next_state)
        except RuntimeError as exc:
            method_failure_reason = f"macro_runtime:{exc}"
            break
        state = next_state
        if terminated:
            failed_outcome = _failed_macro_outcome(current_option)
            if failed_outcome is not None:
                method_failure_reason = (
                    "macro_failure:"
                    f"{type(current_option).__name__}:"
                    f"{failed_outcome.get('reason', 'unknown')}"
                )
            current_option = None

    loop_seconds = perf_counter() - loop_started
    truncated = bool(steps >= args.max_steps and not done)
    selector.on_episode_end(success=done, truncated=truncated)
    timing = summarize_delivery_timing(errors, args.target_window)
    storage_flow = summarize_block_storage_flow(env.blocks, env.time_steps)
    selector_audit = selector.audit()
    for decision in selector_audit.get("decisions", ()):
        candidate_count = decision.get("candidate_count")
        if candidate_count is not None:
            # Deterministic/REG baselines expose the physical assignment set,
            # but not a learned hierarchy-wide candidate set.  Keep the two
            # notions separate rather than pretending they are cardinally
            # comparable.
            pressure.observe_candidate_set(
                None, accept_count=int(candidate_count)
            )
    option_audit = scheduler_episode_audit(env)
    scheduler_audit = scheduler.audit()
    if (
        done
        and method_failure_reason is None
        and target_bound_obstruction_clearances
        != int(option_audit["retrieve_relocations"])
    ):
        raise RuntimeError(
            "event-time target-bound obstruction count disagrees with the "
            "completed retrieval-option audit"
        )
    if assignment_commitment == DECISION_EPOCH_RESERVED:
        bound_ids = option_audit["bound_proposal_ids"]
        committed_ids = option_audit["committed_proposal_ids"]
        selector_committed_ids = [
            decision.get("proposal_id")
            for decision in selector_audit["decisions"]
            if decision.get("valid")
            and decision.get("commitment_contract")
            == "decision_epoch_proposal_bound_once_v2"
        ]
        reservation_integrity = bool(
            option_audit["reservation_invalidation_count"] == 0
            and option_audit["reservation_bound_count"]
            == option_audit["inbound_successes"]
            and option_audit["reservation_commit_count"]
            == option_audit["inbound_successes"]
            and option_audit["reservation_execution_match_count"]
            == option_audit["inbound_successes"]
            and len(bound_ids) == len(set(bound_ids))
            and bound_ids == committed_ids
            and committed_ids == selector_committed_ids
        )
    else:
        reservation_integrity = True
    strict_success = bool(
        done
        and method_failure_reason is None
        and selector_audit["invalid_assignment_count"] == 0
        and selector_audit.get("fallback_count", 0) == 0
        and option_audit["inbound_failures"] == 0
        and option_audit["retrieve_failures"] == 0
        and illegal_drops == 0
        and reservation_integrity
    )
    contention_metrics = contention_metric_record(
        physical_storage_relocations=physical_storage_relocations,
        target_bound_obstruction_clearances=(
            target_bound_obstruction_clearances
        ),
        standalone_reconfigurations=0,
        standalone_with_direct_delivery_available=0,
        standalone_without_direct_delivery_available=0,
        directly_deliverable_self_reconfigurations=0,
    )

    save_dir = getattr(args, "save_instances_dir", None)
    if save_dir:
        destination = Path(save_dir) / f"seed-{seed}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(instance.to_json() + "\n")

    return {
        "method": scheduler.METHOD,
        "method_variant": f"{scheduler.METHOD}__{assignment_source}",
        "assignment_commitment": assignment_commitment,
        "assignment_source": assignment_source,
        "assignment_source_family": selector_audit[
            "assignment_source_family"
        ],
        "assignment_source_version": selector_audit[
            "assignment_source_version"
        ],
        "track": "B",
        "evaluation_scope": "repaired_learning_augmented_scheduler_candidate",
        "controller_architecture": scheduler.VERSION,
        "controller_action_interface": controller_action_interface,
        "policy_realization": scheduler.POLICY,
        "information_regime": "online_arrived_only",
        "lambda": args.lam,
        "mu": args.mu,
        "eval_seed": seed,
        "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "geometry": geometry_metadata(
            env, requested_exit_width=requested_exit_width
        ),
        "return": total_return,
        "success": float(done),
        "strict_method_success": float(strict_success),
        "method_failure_reason": method_failure_reason,
        "truncated": float(truncated),
        "steps": steps,
        "delivery_count": len(errors),
        "source_setup_seconds": setup_seconds,
        "decision_seconds": loop_seconds,
        # Migration aliases remain available to older result readers.  New
        # protocols must use the explicit decomposition below.
        "obstructive_moves": physical_storage_relocations,
        "relocations": physical_storage_relocations,
        **contention_metrics,
        "illegal_drops": illegal_drops,
        **timing,
        **storage_flow,
        **pressure.metrics(),
        "target_window": args.target_window,
        "delivery_deviations": errors,
        "controller_decisions": {"option": scheduler.decision_count},
        "controller_control_decisions": dict(scheduler.control_decisions),
        "urgency_scheduler_audit": scheduler_audit,
        "scheduler_audit": option_audit,
        "reservation_integrity": reservation_integrity,
        "selector_audit": selector_audit,
        "selector_deployment_digest": (
            kim2020_deployment_digest(selector_payload)
            if assignment_source == TRACK_A_KIM2020_A3C_SPATIAL
            else (
                selector_deployment_digest(selector_payload)
                if selector_payload is not None
                else None
            )
        ),
        "assignment_policy_realization": (
            deployment_mode
            if assignment_source == TRACK_A_KIM2020_A3C_SPATIAL
            else "deterministic_map"
        ),
        "assignment_policy_seed": (
            int(seed if policy_seed is None else policy_seed)
            if assignment_source == TRACK_A_KIM2020_A3C_SPATIAL
            else None
        ),
    }


def evaluate_one(
    args,
    seed,
    selector_payload,
    *,
    episode_instance=None,
):
    """Backward-compatible due-only evaluator entry point."""

    return _evaluate_one(
        args,
        seed,
        selector_payload,
        episode_instance=episode_instance,
        scheduler_variant=getattr(
            args, "scheduler_variant", DUE_ONLY_VARIANT
        ),
    )


def evaluate_duration_aware_one(
    args,
    seed,
    selector_payload,
    *,
    episode_instance=None,
):
    return _evaluate_one(
        args,
        seed,
        selector_payload,
        episode_instance=episode_instance,
        scheduler_variant=DURATION_AWARE_VARIANT,
    )


def evaluate_assignment_ablation_one(
    args,
    seed,
    selector_payload,
    *,
    assignment_source,
    episode_instance=None,
):
    """Evaluate one source under the fixed source-neutral scheduler stack."""

    payload = (
        selector_payload
        if assignment_source in (
            TRACK_A_REG_SELECTOR_V5,
            TRACK_A_KIM2020_A3C_SPATIAL,
        )
        else None
    )
    return _evaluate_one(
        args,
        seed,
        payload,
        episode_instance=episode_instance,
        scheduler_variant=DURATION_AWARE_VARIANT,
        assignment_source=assignment_source,
        source_neutral_scheduler=True,
        assignment_commitment=getattr(
            args, "assignment_commitment", POST_PICKUP_RECOMPUTE
        ),
    )


def summarize(runs):
    returns = np.asarray([run["return"] for run in runs], dtype=float)
    return {
        "n": len(runs),
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(runs) > 1 else 0.0,
        "success_rate": float(np.mean([run["success"] for run in runs])),
        "strict_method_success_rate": float(
            np.mean([run["strict_method_success"] for run in runs])
        ),
        "mean_delivery_count": float(
            np.mean([run["delivery_count"] for run in runs])
        ),
        "mean_steps": float(np.mean([run["steps"] for run in runs])),
        "mean_absolute_error": float(
            np.nanmean(
                np.asarray([run["mean_absolute_error"] for run in runs], dtype=float)
            )
        ),
        "total_invalid_assignments": int(
            sum(
                run["selector_audit"]["invalid_assignment_count"]
                for run in runs
            )
        ),
        "total_infeasible_epochs": int(
            sum(
                run["selector_audit"]["infeasible_epoch_count"]
                for run in runs
            )
        ),
        "method_failures": [
            {
                "eval_seed": run["eval_seed"],
                "reason": run["method_failure_reason"],
            }
            for run in runs
            if run["method_failure_reason"] is not None
        ],
        **summarize_storage_flow_runs(runs),
        **summarize_occupancy_pressure_runs(runs),
    }


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
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--instance")
    parser.add_argument("--save-instances-dir")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument(
        "--scheduler-variant",
        choices=SCHEDULER_VARIANTS,
        default=DUE_ONLY_VARIANT,
    )
    parser.add_argument(
        "--lookahead-margin-steps", type=float, default=0.0
    )
    parser.add_argument(
        "--assignment-commitment",
        choices=ASSIGNMENT_COMMITMENTS,
        default=POST_PICKUP_RECOMPUTE,
    )
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.instance and len(args.eval_seeds) != 1:
        parser.error("--instance requires exactly one --eval-seeds value")
    if args.max_steps <= 0 or args.max_defer_steps <= 0:
        parser.error("max-steps and max-defer-steps must be positive")
    if (
        not np.isfinite(args.lookahead_margin_steps)
        or args.lookahead_margin_steps < 0.0
    ):
        parser.error("lookahead-margin-steps must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    selector_payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    validate_selector_payload(selector_payload, lam=args.lam, mu=args.mu)
    runs = [
        evaluate_one(args, seed, selector_payload)
        for seed in args.eval_seeds
    ]
    payload = json_safe(
        {
            "config": {
                "method": runs[0]["method"],
                "scheduler_variant": args.scheduler_variant,
                "scheduler_architecture": runs[0][
                    "controller_architecture"
                ],
                "policy_realization": runs[0]["policy_realization"],
                "controller_action_interface": runs[0][
                    "controller_action_interface"
                ],
                "assignment_commitment": args.assignment_commitment,
                "selector_checkpoint": str(args.selector_checkpoint.resolve()),
                "selector_deployment_digest": selector_deployment_digest(
                    selector_payload
                ),
                "selector_frozen": True,
                "lambda": args.lam,
                "mu": args.mu,
                "eval_seeds": args.eval_seeds,
                "max_steps": args.max_steps,
                "max_defer_steps": args.max_defer_steps,
                "lookahead_margin_steps": args.lookahead_margin_steps,
                "device": str(resolve_device(args.device)),
                "method_contract": {
                    key: runs[0]["urgency_scheduler_audit"][key]
                    for key in (
                        "decision_epoch_contract",
                        "retrieve_deliver_option_version",
                        "retrieval_executor_version",
                        "retrieval_start_contract",
                        "ranking_eta_contract",
                        "relocation_selector_version",
                        "accept_store_option_version",
                        "defer_option_version",
                        "retrieval_termination_contract",
                        "assignment_preview_contract",
                        "accept_duration_estimate_contract",
                        "accept_assignment_contract",
                        "lookahead_cost_contract",
                        "assignment_commitment_contract",
                        "reservation_contract",
                    )
                    if key in runs[0]["urgency_scheduler_audit"]
                },
            },
            "summary": summarize(runs),
            "runs": runs,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
