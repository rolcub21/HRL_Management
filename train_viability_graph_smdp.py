"""Calibration-only smoke training for the viability-constrained graph SMDP.

This entry point is intentionally self contained and deliberately small.  It
executes only the parameterized macros returned by
``PSLAP.viability_candidates`` and learns only from the environment reward
stream.  No assignment, scheduling, or behavior-policy baseline is imported.

The program is a smoke/integration experiment, not a sealed-holdout runner.
Every stochastic episode is identified by an explicit calibration seed in the
saved audit.  A candidate frontier is exact-verifier authoritative: ``UNSAFE``
and computational ``UNKNOWN`` successors are excluded, and an empty frontier
ends the episode as a method failure rather than invoking a fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev
from time import perf_counter
from typing import Optional, Sequence

import numpy as np
import torch

from example.helper.timing_metrics import summarize_delivery_timing
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    StrictDecisionBoundaryError,
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.viability_prioritizer import (
    RecoveryStatePrioritizer,
    ViabilityCriticPrioritizer,
)
from viability_graph_hierarchy import (
    NoCertifiedViableAction,
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
)


VCG_SMOKE_PROTOCOL = "vcg_smdp_calibration_only_smoke_v1"
SMDP_RETURN_CONTRACT = "sum_i_0_to_k_minus_1_gamma_pow_i_environment_reward_v1"
NO_FALLBACK_CONTRACT = "empty_exact_safe_frontier_is_method_failure_v1"
CERTIFICATE_SCOPE = (
    "current_admitted_work_closed_admission_geometric_recoverability_v1"
)
# ``stress_v1`` declares this namespace untouched in
# compare_track_b_generalization.py.  This smoke entry point must never open it.
SEALED_STRESS_V1_HOLDOUT_SEEDS = frozenset(range(69_000, 69_010))


def _finite_float(value, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` without silently accepting an unavailable CUDA device."""

    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class MacroExecution:
    """One observed strict-macro SMDP transition."""

    candidate_key: str
    action_type: str
    mode: str
    discounted_return: float
    raw_return: float
    duration: int
    env_terminal: bool
    truncated: bool
    option_terminated: bool
    option_success: bool
    failure_reason: Optional[str]
    delivery_deviations: tuple[float, ...]
    relocations: int
    illegal_drops: int

    @property
    def replay_terminal(self) -> bool:
        return bool(
            self.env_terminal
            or self.truncated
            or not self.option_success
        )


def _option_failure(option) -> Optional[str]:
    outcome = getattr(option, "last_outcome", None)
    if isinstance(outcome, dict) and outcome.get("success") is False:
        return str(outcome.get("reason", "option_reported_failure"))
    if bool(getattr(option, "failed", False)):
        return str(getattr(option, "failure_reason", "option_failed"))
    return None


def execute_certified_macro(
    env,
    candidate: ViabilityActionCandidate,
    *,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
) -> MacroExecution:
    """Execute exactly the one-shot option bound in ``candidate``.

    Rewards are accumulated once, directly from ``env.step``.  The discounted
    value is the SMDP reward used in replay; the raw value is used for episode
    reporting.  The function never chooses or substitutes another macro.
    """

    gamma = _finite_float(gamma, name="gamma")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if (
        isinstance(remaining_steps, bool)
        or not isinstance(remaining_steps, int)
        or remaining_steps <= 0
    ):
        raise ValueError("remaining_steps must be a positive integer")
    if candidate.certificate.status is not ViabilityStatus.SAFE:
        raise ValueError("only exactly SAFE candidates may be executed")

    option = candidate.option
    if not bool(option.initiation(env.get_current_state())):
        return MacroExecution(
            candidate_key=candidate.key,
            action_type=candidate.action_type.value,
            mode=candidate.mode.value,
            discounted_return=0.0,
            raw_return=0.0,
            duration=0,
            env_terminal=False,
            truncated=False,
            option_terminated=False,
            option_success=False,
            failure_reason="bound_option_not_initiable",
            delivery_deviations=(),
            relocations=0,
            illegal_drops=0,
        )

    discounted = 0.0
    raw = 0.0
    deviations = []
    relocations = 0
    illegal_drops = 0
    terminal = False
    terminated = False
    runtime_failure = None
    duration = 0

    while duration < remaining_steps and not terminal and not terminated:
        state = env.get_current_state()
        try:
            action = option.policy(state, test=evaluation)
            next_state, reward, terminal, info = env.step(action)
            duration += 1
            reward = float(reward)
            discounted += (gamma ** (duration - 1)) * reward
            raw += reward
            if "delivery_error_time" in info:
                deviations.append(float(info["delivery_error_time"]))
            relocations += int(bool(info.get("relocated_block")))
            illegal_drops += int(bool(info.get("illegal_drop")))
            terminated = bool(option.termination(next_state))
        except RuntimeError as error:
            runtime_failure = f"macro_runtime:{type(error).__name__}:{error}"
            break

    truncated = bool(
        duration >= remaining_steps and not terminal and not terminated
    )
    failure = runtime_failure or _option_failure(option)
    if terminal and not terminated and failure is None:
        failure = "environment_terminated_before_option_contract"
    if truncated and failure is None:
        failure = "step_limit_during_bound_macro"
    return MacroExecution(
        candidate_key=candidate.key,
        action_type=candidate.action_type.value,
        mode=candidate.mode.value,
        discounted_return=float(discounted),
        raw_return=float(raw),
        duration=duration,
        env_terminal=bool(terminal),
        truncated=truncated,
        option_terminated=terminated,
        option_success=failure is None and (terminated or terminal),
        failure_reason=failure,
        delivery_deviations=tuple(deviations),
        relocations=relocations,
        illegal_drops=illegal_drops,
    )


def _update_frontier_audit(counter: Counter, snapshot: ViabilityCandidateSnapshot):
    audit = snapshot.audit
    counter["frontier_calls"] += 1
    counter["safe_candidates"] += audit.candidate_count
    counter["unsafe_candidates_rejected"] += (
        audit.unsafe_accept_count + audit.unsafe_recovery_count
    )
    counter["unknown_candidates_rejected"] += (
        audit.unknown_accept_count + audit.unknown_recovery_count
    )
    counter["fail_closed_rejections"] += audit.fail_closed_rejection_count
    counter["exact_cache_hits"] += audit.cache_hits
    counter["exact_cache_misses"] += audit.cache_misses
    counter["exact_analysis_seconds"] += float(audit.exact_analysis_seconds)
    counter["frontier_analysis_seconds"] += float(audit.analysis_seconds)
    counter["priority_states_scored"] += audit.priority_states_scored
    counter["priority_pass_count"] += audit.priority_pass_count
    counter["priority_inference_seconds"] += float(
        audit.priority_inference_seconds
    )
    counter["epochs_with_no_viable_acceptance"] += int(
        audit.physical_accept_count > 0 and audit.safe_accept_count == 0
    )
    counter["current_safe"] += int(
        audit.current_recovery_status is ViabilityStatus.SAFE
    )
    counter["current_unsafe"] += int(
        audit.current_recovery_status is ViabilityStatus.UNSAFE
    )
    counter["current_unknown"] += int(
        audit.current_recovery_status is ViabilityStatus.UNKNOWN
    )


def _mean(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else math.nan


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if values else math.nan


def epsilon_at(
    decision: int,
    *,
    start: float,
    end: float,
    decay_decisions: int,
    warmup_decisions: int = 0,
) -> float:
    if decision < 0:
        raise ValueError("decision must be non-negative")
    start = float(start)
    end = float(end)
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("epsilon endpoints must be in [0, 1]")
    if decay_decisions <= 0:
        raise ValueError("epsilon decay must be positive")
    if warmup_decisions < 0:
        raise ValueError("epsilon warmup must be non-negative")
    elapsed = max(int(decision) - int(warmup_decisions), 0)
    fraction = min(float(elapsed) / float(decay_decisions), 1.0)
    return start + fraction * (end - start)


def _failure_name(prefix: str, error: Exception) -> str:
    return f"{prefix}:{type(error).__name__}:{error}"


def run_calibration_episode(
    agent: ViabilityGraphHierarchyAgent,
    env: SmallRoomsEnv,
    *,
    instance_seed: int,
    training: bool,
    max_steps: int,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    certificate_cache: ViabilityCertificateCache,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_decisions: int,
    updates_per_macro: int,
    state_prioritizer: Optional[RecoveryStatePrioritizer] = None,
    epsilon_warmup_decisions: int = 0,
) -> dict:
    """Run one seeded instance without exposing its future schedule to policy."""

    instance = env.sample_episode_instance(int(instance_seed))
    env.reset(instance=instance)
    agent.reset_episode_state()
    total_return = 0.0
    steps = 0
    macro_count = 0
    consecutive_defer = 0
    delivery_deviations = []
    method_failure_reason = None
    losses = []
    selected = Counter()
    decision_audit = []
    audit = Counter()
    pending_snapshot = None
    started = perf_counter()

    while steps < max_steps and not env.is_state_terminal(env.current_state):
        if pending_snapshot is None:
            try:
                pending_snapshot = enumerate_viability_candidates(
                    env,
                    consecutive_defer_decisions=consecutive_defer,
                    search_config=search_config,
                    liveness_rule=liveness_rule,
                    cache=certificate_cache,
                    state_prioritizer=state_prioritizer,
                )
            except (StrictDecisionBoundaryError, RuntimeError, ValueError) as error:
                method_failure_reason = _failure_name(
                    "frontier_construction", error
                )
                audit["frontier_construction_failures"] += 1
                break
            _update_frontier_audit(audit, pending_snapshot)

        snapshot = pending_snapshot
        pending_snapshot = None
        if not snapshot.candidates:
            method_failure_reason = "no_exact_safe_candidate"
            audit["empty_safe_frontiers"] += 1
            break

        epsilon = (
            epsilon_at(
                agent.decision_count,
                start=epsilon_start,
                end=epsilon_end,
                decay_decisions=epsilon_decay_decisions,
                warmup_decisions=epsilon_warmup_decisions,
            )
            if training
            else 0.0
        )
        if training:
            agent.set_epsilon(epsilon)
        try:
            decision = agent.select(
                snapshot,
                training=training,
                epsilon=epsilon,
            )
        except (NoCertifiedViableAction, RuntimeError, ValueError) as error:
            method_failure_reason = _failure_name("safe_selection", error)
            audit["selection_failures"] += 1
            audit["empty_safe_frontiers"] += int(
                isinstance(error, NoCertifiedViableAction)
            )
            break

        selected[decision.candidate.action_type.value] += 1
        execution = execute_certified_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=max_steps - steps,
            evaluation=not training,
        )
        macro_count += 1
        steps += execution.duration
        total_return += execution.raw_return
        delivery_deviations.extend(execution.delivery_deviations)
        audit["relocations"] += execution.relocations
        audit["illegal_drops"] += execution.illegal_drops
        audit["macro_failures"] += int(not execution.option_success)
        audit["truncated_macros"] += int(execution.truncated)
        if decision.candidate.mode.value == "recover":
            if decision.exact_rank_progress:
                audit["exact_rank_reducing_recovery_selections"] += 1
            else:
                audit["nonprogress_or_nonexact_recovery_selections"] += 1
        certificate = decision.candidate.certificate
        decision_audit.append(
            {
                "decision_index": macro_count - 1,
                "decision_epoch": int(snapshot.decision_epoch),
                "candidate_key": decision.candidate.key,
                "mode": decision.candidate.mode.value,
                "action_type": decision.candidate.action_type.value,
                "target_label": decision.candidate.target_label,
                "source": decision.candidate.source,
                "destination": decision.candidate.destination,
                "certificate_status": certificate.status.value,
                "certificate_witness_macros": len(certificate.witness),
                "certificate_witness_primitive_steps": (
                    certificate.witness_primitive_steps
                ),
                "certificate_search_order": certificate.search_order,
                "certificate_exhaustive": bool(certificate.exhaustive),
                "recovery_rank_exact": bool(
                    snapshot.audit.recovery_rank_exact
                ),
                "recovery_rank_before": (
                    decision.candidate.recovery_rank_before
                ),
                "recovery_rank_after": (
                    decision.candidate.recovery_rank_after
                ),
                "rank_delta": decision.candidate.rank_delta,
                "liveness_forced": bool(decision.liveness_forced),
                "exact_rank_progress": bool(decision.exact_rank_progress),
                "selection_source": decision.selection_source,
                "duration": execution.duration,
                "discounted_macro_return": execution.discounted_return,
                "raw_macro_return": execution.raw_return,
                "option_success": execution.option_success,
                "failure_reason": execution.failure_reason,
            }
        )

        if execution.action_type == ViabilityActionType.DEFER.value:
            outcome = getattr(decision.option, "last_outcome", None)
            observed_event = bool(
                isinstance(outcome, dict)
                and outcome.get("reason") == "observed_event"
            )
            consecutive_defer = 0 if observed_event else consecutive_defer + 1
        else:
            consecutive_defer = 0

        terminal_boundary = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
        )
        if execution.duration == 0:
            terminal_boundary = True

        if not terminal_boundary:
            try:
                pending_snapshot = enumerate_viability_candidates(
                    env,
                    consecutive_defer_decisions=consecutive_defer,
                    search_config=search_config,
                    liveness_rule=liveness_rule,
                    cache=certificate_cache,
                    state_prioritizer=state_prioritizer,
                )
                _update_frontier_audit(audit, pending_snapshot)
                if not pending_snapshot.candidates:
                    terminal_boundary = True
                    method_failure_reason = "no_exact_safe_candidate"
                    audit["empty_safe_frontiers"] += 1
            except (StrictDecisionBoundaryError, RuntimeError, ValueError) as error:
                terminal_boundary = True
                method_failure_reason = _failure_name(
                    "next_frontier_construction", error
                )
                audit["frontier_construction_failures"] += 1

        outcome_observed = False
        if execution.duration > 0:
            try:
                agent.observe_outcome(
                    decision,
                    next_snapshot=(None if terminal_boundary else pending_snapshot),
                    done=terminal_boundary,
                )
                outcome_observed = True
            except (NoCertifiedViableAction, RuntimeError, ValueError) as error:
                # The liveness shield is part of admissibility.  A witness
                # mismatch or empty shielded continuation is a method failure,
                # never permission to fall back to the unshielded frontier.
                terminal_boundary = True
                outcome_observed = True
                pending_snapshot = None
                method_failure_reason = _failure_name(
                    "liveness_outcome", error
                )
                audit["liveness_guard_failures"] += 1

        if training and execution.duration > 0:
            try:
                agent.remember(
                    decision,
                    reward=execution.discounted_return,
                    duration=execution.duration,
                    next_snapshot=(None if terminal_boundary else pending_snapshot),
                    done=terminal_boundary,
                    outcome_already_observed=outcome_observed,
                )
                for _ in range(updates_per_macro):
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(float(loss))
            except (NoCertifiedViableAction, RuntimeError, ValueError) as error:
                terminal_boundary = True
                method_failure_reason = _failure_name("learning", error)
                audit["learning_failures"] += 1

        if not execution.option_success and method_failure_reason is None:
            method_failure_reason = (
                f"macro_failure:{execution.action_type}:"
                f"{execution.failure_reason or 'unknown'}"
            )
        if terminal_boundary:
            break

    terminal = bool(env.is_state_terminal(env.current_state))
    if steps >= max_steps and not terminal and method_failure_reason is None:
        method_failure_reason = "episode_step_limit"
    success = bool(terminal and method_failure_reason is None)
    timing = summarize_delivery_timing(
        delivery_deviations,
        env.DELIVERY_TARGET_WINDOW,
    )
    return {
        "protocol": VCG_SMOKE_PROTOCOL,
        "split": "train_calibration" if training else "evaluation_calibration",
        "instance_seed": int(instance_seed),
        "episode_instance_id": instance.instance_id,
        "return": float(total_return),
        "success": success,
        "strict_method_success": bool(
            success
            and audit["illegal_drops"] == 0
            and audit["macro_failures"] == 0
            and audit["frontier_construction_failures"] == 0
            and audit["empty_safe_frontiers"] == 0
        ),
        "steps": int(steps),
        "macro_decisions": int(macro_count),
        "method_failure_reason": method_failure_reason,
        "selected_action_counts": dict(selected),
        "decisions": tuple(decision_audit),
        "loss_mean": _mean(losses),
        "loss_updates": len(losses),
        "delivery_deviations": tuple(delivery_deviations),
        **timing,
        "audit": {
            **dict(audit),
            "certificate_cache_entries": len(certificate_cache),
            "baseline_viability_teacher": False,
            "baseline_policy_query": False,
            "exact_verifier_authoritative": True,
            "future_schedule_visible_to_policy": False,
            "certificate_scope": CERTIFICATE_SCOPE,
            "complete_episode_certificate": False,
            "no_fallback_contract": NO_FALLBACK_CONTRACT,
        },
        "wall_seconds": perf_counter() - started,
    }


def summarize_runs(runs: Sequence[dict]) -> dict:
    runs = tuple(runs)
    returns = [float(item["return"]) for item in runs]
    timing_errors = [
        float(item["mean_absolute_error"])
        for item in runs
        if math.isfinite(float(item["mean_absolute_error"]))
    ]
    return {
        "episodes": len(runs),
        "mean_return": _mean(returns),
        "return_std": _std(returns),
        "success_rate": _mean([float(item["success"]) for item in runs]),
        "strict_method_success_rate": _mean(
            [float(item["strict_method_success"]) for item in runs]
        ),
        "mean_absolute_error": _mean(timing_errors),
        "total_macro_decisions": sum(item["macro_decisions"] for item in runs),
        "total_relocations": sum(item["audit"].get("relocations", 0) for item in runs),
        "total_fail_closed_rejections": sum(
            item["audit"].get("fail_closed_rejections", 0) for item in runs
        ),
        "total_empty_safe_frontiers": sum(
            item["audit"].get("empty_safe_frontiers", 0) for item in runs
        ),
        "total_epochs_with_no_viable_acceptance": sum(
            item["audit"].get("epochs_with_no_viable_acceptance", 0)
            for item in runs
        ),
        "total_liveness_guard_failures": sum(
            item["audit"].get("liveness_guard_failures", 0)
            for item in runs
        ),
        "total_exact_analysis_seconds": sum(
            item["audit"].get("exact_analysis_seconds", 0.0)
            for item in runs
        ),
        "total_frontier_analysis_seconds": sum(
            item["audit"].get("frontier_analysis_seconds", 0.0)
            for item in runs
        ),
        "total_priority_states_scored": sum(
            item["audit"].get("priority_states_scored", 0)
            for item in runs
        ),
        "total_priority_pass_count": sum(
            item["audit"].get("priority_pass_count", 0)
            for item in runs
        ),
        "total_priority_inference_seconds": sum(
            item["audit"].get("priority_inference_seconds", 0.0)
            for item in runs
        ),
        "method_failures": [
            {
                "instance_seed": item["instance_seed"],
                "reason": item["method_failure_reason"],
            }
            for item in runs
            if item["method_failure_reason"] is not None
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-free calibration smoke run for the exact-safe VCG-SMDP"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="network/RNG seed")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--train-instance-seed-base", type=int, default=73000)
    parser.add_argument("--eval-instance-seed-base", type=int, default=73100)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=1)
    parser.add_argument("--arrival-rate", type=float, default=1.5)
    parser.add_argument("--proc-mean", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--viability-critic-checkpoint",
        type=Path,
        default=None,
        help=(
            "optional authenticated calibrated critic used only to order "
            "complete exact-frontier certification"
        ),
    )
    parser.add_argument(
        "--viability-critic-device",
        default="cpu",
        help="device for the frozen prioritization ensemble",
    )
    parser.add_argument(
        "--viability-critic-dataset-manifest",
        type=Path,
        default=None,
        help="optional explicit authenticated dataset manifest",
    )
    parser.add_argument(
        "--viability-critic-expected-sha256",
        default=None,
        help="optional pinned SHA-256 identity for the critic checkpoint",
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replay-size", type=int, default=500)
    parser.add_argument("--updates-per-macro", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=20)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-warmup-decisions", type=int, default=0)
    parser.add_argument("--epsilon-decay-decisions", type=int, default=100)
    parser.add_argument("--tau-accept", type=float, default=0.1)
    parser.add_argument("--tau-recover", type=float, default=0.1)
    parser.add_argument("--tau-defer", type=float, default=0.1)
    parser.add_argument("--tau-mode", type=float, default=1.0)

    parser.add_argument("--graph-hidden-dim", type=int, default=8)
    parser.add_argument("--graph-embedding-dim", type=int, default=8)
    parser.add_argument("--message-passing-steps", type=int, default=1)
    parser.add_argument("--action-embedding-dim", type=int, default=4)
    parser.add_argument("--head-hidden-dim", type=int, default=16)

    parser.add_argument("--search-max-depth", type=int, default=None)
    parser.add_argument("--search-max-nodes", type=int, default=20_000)
    parser.add_argument("--search-max-primitive-steps", type=int, default=None)
    parser.add_argument(
        "--search-order",
        choices=("goal_directed", "breadth_first"),
        default="goal_directed",
    )
    parser.add_argument(
        "--reserve-queue-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--max-consecutive-defers", type=int, default=16)
    parser.add_argument(
        "--max-nonprogress-recovery-decisions", type=int, default=2
    )
    parser.add_argument(
        "--force-recovery-witness-when-due",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "episodes",
        "grid_rows",
        "grid_cols",
        "number_blocks",
        "proc_mean",
        "max_steps",
        "batch_size",
        "replay_size",
        "updates_per_macro",
        "target_update_every",
        "epsilon_decay_decisions",
        "graph_hidden_dim",
        "graph_embedding_dim",
        "action_embedding_dim",
        "head_hidden_dim",
        "max_defer_steps",
        "max_consecutive_defers",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.eval_episodes < 0:
        raise ValueError("--eval-episodes must be non-negative")
    if args.epsilon_warmup_decisions < 0:
        raise ValueError("--epsilon-warmup-decisions must be non-negative")
    if args.message_passing_steps < 0:
        raise ValueError("--message-passing-steps must be non-negative")
    if args.max_nonprogress_recovery_decisions < 0:
        raise ValueError(
            "--max-nonprogress-recovery-decisions must be non-negative"
        )
    if args.train_instance_seed_base < 0 or args.eval_instance_seed_base < 0:
        raise ValueError("instance seed bases must be non-negative")
    if args.arrival_rate < 0.0 or not math.isfinite(args.arrival_rate):
        raise ValueError("--arrival-rate must be finite and non-negative")
    if args.viability_critic_checkpoint is None and (
        args.viability_critic_dataset_manifest is not None
        or args.viability_critic_expected_sha256 is not None
    ):
        raise ValueError(
            "critic manifest/identity requires --viability-critic-checkpoint"
        )
    train_seeds = set(
        range(args.train_instance_seed_base, args.train_instance_seed_base + args.episodes)
    )
    eval_seeds = set(
        range(args.eval_instance_seed_base, args.eval_instance_seed_base + args.eval_episodes)
    )
    if train_seeds.intersection(eval_seeds):
        raise ValueError("training and evaluation calibration seeds must be disjoint")
    sealed = (train_seeds | eval_seeds).intersection(
        SEALED_STRESS_V1_HOLDOUT_SEEDS
    )
    if sealed:
        raise ValueError(
            "calibration-only runner refuses sealed stress_v1 holdout seeds: "
            f"{tuple(sorted(sealed))}"
        )


def _json_safe(value):
    if isinstance(value, dict):
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


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = resolve_device(args.device)
    critic_device = resolve_device(args.viability_critic_device)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_prioritizer = None
    if args.viability_critic_checkpoint is not None:
        state_prioritizer = ViabilityCriticPrioritizer.from_checkpoint(
            args.viability_critic_checkpoint,
            device=critic_device,
            dataset_manifest=args.viability_critic_dataset_manifest,
            require_test_confirmation=True,
            expected_checkpoint_sha256=(
                args.viability_critic_expected_sha256
            ),
        )

    env = SmallRoomsEnv(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        number_blocks=args.number_blocks,
        choose_storage=False,
        arrival_rate=args.arrival_rate,
        proc_mean=args.proc_mean,
    )
    search_config = ViabilitySearchConfig(
        max_depth=args.search_max_depth,
        max_nodes=args.search_max_nodes,
        max_primitive_steps=args.search_max_primitive_steps,
        reserve_queue_cells=args.reserve_queue_cells,
        search_order=args.search_order,
    )
    liveness_rule = BoundedEventDeferRule(
        max_option_steps=args.max_defer_steps,
        max_consecutive_defer_decisions=args.max_consecutive_defers,
    )
    graph_config = ViabilityGraphConfig(
        graph_hidden_dim=args.graph_hidden_dim,
        graph_embedding_dim=args.graph_embedding_dim,
        message_passing_steps=args.message_passing_steps,
        action_embedding_dim=args.action_embedding_dim,
        head_hidden_dim=args.head_hidden_dim,
        tau_accept=args.tau_accept,
        tau_recover=args.tau_recover,
        tau_defer=args.tau_defer,
        tau_mode=args.tau_mode,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_size,
        target_update_every=args.target_update_every,
        max_nonprogress_recovery_decisions=(
            args.max_nonprogress_recovery_decisions
        ),
        force_recovery_witness_when_due=(
            args.force_recovery_witness_when_due
        ),
    )
    agent = ViabilityGraphHierarchyAgent(
        config=graph_config,
        seed=args.seed,
        device=device,
        epsilon=args.epsilon_start,
    )
    cache = ViabilityCertificateCache()
    print(
        "VCG-SMDP smoke | "
        f"device={device} | train={args.episodes} | eval={args.eval_episodes} | "
        f"grid={args.grid_rows}x{args.grid_cols} | blocks={args.number_blocks} | "
        "exact-safe=true | baseline-teacher=false",
        flush=True,
    )
    if state_prioritizer is not None:
        print(
            "Viability priority | "
            f"protocol={state_prioritizer.protocol} | "
            f"device={state_prioritizer.device} | "
            f"threshold={state_prioritizer.threshold:.6f} | "
            f"sha256={state_prioritizer.checkpoint_sha256}",
            flush=True,
        )

    train_runs = []
    for episode in range(args.episodes):
        instance_seed = args.train_instance_seed_base + episode
        env.current_episode = episode + 1
        run = run_calibration_episode(
            agent,
            env,
            instance_seed=instance_seed,
            training=True,
            max_steps=args.max_steps,
            search_config=search_config,
            liveness_rule=liveness_rule,
            certificate_cache=cache,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_decisions=args.epsilon_decay_decisions,
            updates_per_macro=args.updates_per_macro,
            state_prioritizer=state_prioritizer,
            epsilon_warmup_decisions=args.epsilon_warmup_decisions,
        )
        train_runs.append(run)
        print(
            f"VCG Ep {episode + 1:3d} | R {run['return']:8.2f} | "
            f"Succ {int(run['success'])} | Dec {run['macro_decisions']:3d} | "
            f"Safe {run['audit'].get('safe_candidates', 0):4d} | "
            f"Reject {run['audit'].get('fail_closed_rejections', 0):3d} | "
            f"Eps {agent.epsilon:.3f}",
            flush=True,
        )

    training_audit = agent.audit(include_decisions=False)
    train_seeds = tuple(
        args.train_instance_seed_base + index for index in range(args.episodes)
    )
    checkpoint_path = args.output_dir / "checkpoint.pth"
    checkpoint = agent.checkpoint(
        include_replay=args.save_replay,
        protocol=VCG_SMOKE_PROTOCOL,
        instance_regime="calibration_only",
        smdp_return_contract=SMDP_RETURN_CONTRACT,
        no_fallback_contract=NO_FALLBACK_CONTRACT,
        certificate_scope=CERTIFICATE_SCOPE,
        complete_episode_certificate=False,
        model_seed=args.seed,
        train_instance_seeds=train_seeds,
        train_episode_instance_ids=tuple(
            item["episode_instance_id"] for item in train_runs
        ),
        environment={
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "number_blocks": args.number_blocks,
            "arrival_rate": args.arrival_rate,
            "proc_mean": args.proc_mean,
        },
        viability_search=asdict(search_config),
        liveness_rule=asdict(liveness_rule),
        viability_prioritization=(
            None
            if state_prioritizer is None
            else state_prioritizer.audit_dict()
        ),
    )
    torch.save(checkpoint, checkpoint_path)

    # Evaluate a fresh object loaded from the exact saved weights.  This keeps
    # deployment decisions from changing resumable training clocks/audits.
    eval_agent = ViabilityGraphHierarchyAgent.from_checkpoint(
        checkpoint,
        device=device,
        resumable=False,
        seed=args.seed,
    )
    eval_cache = ViabilityCertificateCache()
    eval_runs = []
    for episode in range(args.eval_episodes):
        instance_seed = args.eval_instance_seed_base + episode
        env.current_episode = args.episodes + episode + 1
        run = run_calibration_episode(
            eval_agent,
            env,
            instance_seed=instance_seed,
            training=False,
            max_steps=args.max_steps,
            search_config=search_config,
            liveness_rule=liveness_rule,
            certificate_cache=eval_cache,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            updates_per_macro=1,
            state_prioritizer=state_prioritizer,
        )
        eval_runs.append(run)
        print(
            f"VCG Eval {episode + 1:2d} | R {run['return']:8.2f} | "
            f"Succ {int(run['success'])} | Dec {run['macro_decisions']:3d} | "
            f"Failure {run['method_failure_reason']}",
            flush=True,
        )

    result = {
        "protocol": VCG_SMOKE_PROTOCOL,
        "scope": "calibration_only_not_sealed_holdout",
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "certificate_scope": CERTIFICATE_SCOPE,
        "complete_episode_certificate": False,
        "exact_verifier_authoritative": True,
        "unsafe_unknown_fail_closed": True,
        "smdp_return_contract": SMDP_RETURN_CONTRACT,
        "no_fallback_contract": NO_FALLBACK_CONTRACT,
        "viability_prioritization": (
            None
            if state_prioritizer is None
            else state_prioritizer.audit_dict()
        ),
        "device": str(device),
        "arguments": vars(args),
        "agent_config": graph_config.to_dict(),
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "checkpoint": str(checkpoint_path.resolve()),
        "training": {
            "runs": train_runs,
            "summary": summarize_runs(train_runs),
            "agent_audit": training_audit,
        },
        "evaluation": {
            "runs": eval_runs,
            "summary": summarize_runs(eval_runs),
            "deployment_decisions": (
                eval_agent.decision_count - agent.decision_count
            ),
            "agent_audit": eval_agent.audit(include_decisions=False),
        },
    }
    result_path = args.output_dir / "results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {result_path}")
    return result


if __name__ == "__main__":
    main()
