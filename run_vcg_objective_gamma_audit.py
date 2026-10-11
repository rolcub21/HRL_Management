"""Paired 2x2 VCG timing-objective/discount mechanism audit.

This runner is intentionally outside the VCG-v1 replication entry point.  It
crosses the legacy clipped timing reward and a dense piecewise timing reward
with ``gamma`` 0.99 and 1.0.  Every arm receives the same initialization seed,
EpisodeInstances, validation panel, architecture, exact-SAFE frontier, and
training hyperparameters.  Objective and discount are authenticated in each
arm's resume contract, so an arm (or a VCG-v1 checkpoint) cannot be resumed
under another condition.

Raw returns from unlike reward arms are not compared.  Every validation
trajectory is instead rescored under both objectives while holding its
realized actions fixed.  This is a development-only mechanism screen, never a
formal test or performance-confirmation protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean, pstdev
from time import perf_counter
from typing import Optional, Sequence

import numpy as np
import torch

from example.helper.timing_metrics import summarize_delivery_timing
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    StrictDecisionBoundaryError,
    ViabilityActionCandidate,
    ViabilityCertificateCache,
    ViabilityActionType,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from train_viability_graph_smdp import (
    CERTIFICATE_SCOPE,
    MacroExecution,
    NO_FALLBACK_CONTRACT,
    SEALED_STRESS_V1_HOLDOUT_SEEDS,
    SMDP_RETURN_CONTRACT,
    _option_failure,
    _update_frontier_audit,
    epsilon_at,
    resolve_device,
    seed_everything,
)
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS
from viability_graph_episodic_audit import (
    EPISODIC_TERMINAL_BOUNDARY_CONTRACT,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)
from viability_graph_hierarchy import NoCertifiedViableAction
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    ObjectiveAuditSmallRoomsEnv,
    TimingObjectiveSpec,
    delivery_reward,
)


MATRIX_PROTOCOL = "vcg_objective_gamma_factorial_development_screen_v1"
ARM_TRAINING_PROTOCOL = "vcg_objective_gamma_factorial_arm_training_v1"
TRAINER_SCHEMA_VERSION = 1
MATRIX_SCHEMA_VERSION = 1
DEFAULT_TOTAL_EPISODES = 200
DEFAULT_VALIDATION_SEEDS = tuple(range(76_000, 76_010))
TRAIN_SEED_ORIGIN = 40_000_000
TRAIN_SEED_STRIDE = 1_000_000
OBJECTIVES = (LEGACY_CLIPPED, DENSE_PIECEWISE)
GAMMAS = (0.99, 1.0)
OBJECTIVE_LABELS = {
    LEGACY_CLIPPED: "legacy",
    DENSE_PIECEWISE: "dense",
}
PROTECTED_SEEDS = frozenset(
    SEALED_IN_REGIME_TEST_SEEDS | SEALED_STRESS_V1_HOLDOUT_SEEDS
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


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_hash(contract: dict) -> str:
    encoded = json.dumps(
        _json_safe(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_global_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_global_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    saved_cuda = state.get("cuda")
    if saved_cuda is not None and torch.cuda.is_available():
        for index, rng_state in enumerate(
            saved_cuda[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(rng_state, device=index)


def _optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _derived_train_seed_base(model_seed: int) -> int:
    return TRAIN_SEED_ORIGIN + int(model_seed) * TRAIN_SEED_STRIDE


def _gamma_label(gamma: float) -> str:
    if float(gamma) == 0.99:
        return "gamma0p99"
    if float(gamma) == 1.0:
        return "gamma1p00"
    raise ValueError("factorial gamma must be exactly 0.99 or 1.0")


def _arm_id(objective: str, gamma: float) -> str:
    return f"{OBJECTIVE_LABELS[objective]}-{_gamma_label(gamma)}"


def _objective_spec(args, objective: str) -> TimingObjectiveSpec:
    return TimingObjectiveSpec(
        objective=objective,
        legacy_base=args.legacy_base,
        legacy_max_bonus=args.legacy_max_bonus,
        dense_b=args.dense_b,
        lambda_abs=args.lambda_abs,
        lambda_outside=args.lambda_outside,
        window=args.target_window,
        legacy_window=args.target_window,
    )


def _new_env(args, objective_spec: TimingObjectiveSpec):
    return ObjectiveAuditSmallRoomsEnv(
        timing_objective=objective_spec,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        number_blocks=args.number_blocks,
        choose_storage=False,
        arrival_rate=args.arrival_rate,
        proc_mean=args.proc_mean,
    )


def _graph_config(args, gamma: float) -> EpisodicViabilityGraphConfig:
    return EpisodicViabilityGraphConfig(
        graph_hidden_dim=args.graph_hidden_dim,
        graph_embedding_dim=args.graph_embedding_dim,
        message_passing_steps=args.message_passing_steps,
        action_embedding_dim=args.action_embedding_dim,
        head_hidden_dim=args.head_hidden_dim,
        tau_accept=args.tau_accept,
        tau_recover=args.tau_recover,
        tau_defer=args.tau_defer,
        tau_mode=args.tau_mode,
        gamma=gamma,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_size,
        update_every=args.update_every,
        target_update_every=args.target_update_every,
        grad_clip=args.grad_clip,
        huber_delta=args.huber_delta,
        max_nonprogress_recovery_decisions=(
            args.max_nonprogress_recovery_decisions
        ),
        force_recovery_witness_when_due=(
            args.force_recovery_witness_when_due
        ),
        episode_horizon_steps=args.max_steps,
    )


def _search_config(args) -> ViabilitySearchConfig:
    return ViabilitySearchConfig(
        max_depth=args.search_max_depth,
        max_nodes=args.search_max_nodes,
        max_primitive_steps=args.search_max_primitive_steps,
        reserve_queue_cells=args.reserve_queue_cells,
        search_order=args.search_order,
    )


def _liveness_rule(args) -> BoundedEventDeferRule:
    return BoundedEventDeferRule(
        max_option_steps=args.max_defer_steps,
        max_consecutive_defer_decisions=args.max_consecutive_defers,
    )


def _execute_audit_macro(
    env,
    candidate: ViabilityActionCandidate,
    *,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
) -> MacroExecution:
    """VCG-v1 macro execution extended only to finite-episode gamma=1."""

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
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
            runtime_failure = (
                f"macro_runtime:{type(error).__name__}:{error}"
            )
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


def _rescore_run(run: dict, spec: TimingObjectiveSpec) -> None:
    """Attach both reward scores to one fixed realized trajectory."""

    deviations = tuple(float(value) for value in run["delivery_deviations"])
    selected_delivery = sum(
        delivery_reward(value, spec, objective=spec.objective)
        for value in deviations
    )
    non_delivery_return = float(run["return"]) - selected_delivery
    legacy = non_delivery_return + sum(
        delivery_reward(value, spec, objective=LEGACY_CLIPPED)
        for value in deviations
    )
    dense = non_delivery_return + sum(
        delivery_reward(value, spec, objective=DENSE_PIECEWISE)
        for value in deviations
    )
    run.update(
        {
            "selected_objective": spec.objective,
            "selected_objective_return": float(run["return"]),
            "non_delivery_return": float(non_delivery_return),
            "legacy_rescored_return": float(legacy),
            "dense_rescored_return": float(dense),
            "dual_rescore_uses_fixed_realized_trajectory": True,
        }
    )


def _mean(values) -> float:
    values = tuple(float(value) for value in values)
    return float(fmean(values)) if values else math.nan


def _std(values) -> float:
    values = tuple(float(value) for value in values)
    return float(pstdev(values)) if values else math.nan


def _timing_slice(values: Sequence[float], window: float) -> dict:
    values = tuple(float(value) for value in values)
    summary = summarize_delivery_timing(values, window)
    return {
        "n": len(values),
        "mean_signed_deviation": summary["mean_signed_deviation"],
        "mean_absolute_error": summary["mean_absolute_error"],
        "within_target_window_rate": summary[
            "within_target_window_rate"
        ],
    }


def _delivery_position_summary(
    runs: Sequence[dict], window: float
) -> dict:
    sequences = [
        tuple(float(value) for value in run.get("delivery_deviations", ()))
        for run in runs
    ]
    longest = max((len(values) for values in sequences), default=0)
    by_position = []
    for position in range(longest):
        values = [
            sequence[position]
            for sequence in sequences
            if position < len(sequence)
        ]
        by_position.append(
            {"delivery_position": position + 1, **_timing_slice(values, window)}
        )
    first_two = [
        value for sequence in sequences for value in sequence[:2]
    ]
    positions_three_plus = [
        value for sequence in sequences for value in sequence[2:]
    ]
    return {
        "by_delivery_position": tuple(by_position),
        "first_two": _timing_slice(first_two, window),
        "positions_three_plus": _timing_slice(
            positions_three_plus, window
        ),
    }


def _failure_name(prefix: str, error: Exception) -> str:
    return f"{prefix}:{type(error).__name__}:{error}"


def run_objective_audit_episode(
    agent: EpisodicViabilityGraphHierarchyAgent,
    env,
    *,
    objective_spec: TimingObjectiveSpec,
    instance_seed: int,
    training: bool,
    max_steps: int,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_decisions: int,
    updates_per_macro: int,
    epsilon_warmup_decisions: int = 0,
    run_protocol: str = ARM_TRAINING_PROTOCOL,
) -> dict:
    """Run one arm episode with an explicit terminal time-limit boundary."""

    instance = env.sample_episode_instance(int(instance_seed))
    env.reset(instance=instance)
    agent.reset_episode_state()
    certificate_cache = ViabilityCertificateCache()
    total_return = 0.0
    episode_discounted_return = 0.0
    steps = 0
    macro_count = 0
    consecutive_defer = 0
    delivery_deviations = []
    method_failure_reason = None
    losses = []
    selected = Counter()
    audit = Counter()
    selection_sources = Counter()
    previous_liveness_forced = False
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
                    state_prioritizer=None,
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
                snapshot, training=training, epsilon=epsilon
            )
        except (NoCertifiedViableAction, RuntimeError, ValueError) as error:
            method_failure_reason = _failure_name("safe_selection", error)
            audit["selection_failures"] += 1
            audit["empty_safe_frontiers"] += int(
                isinstance(error, NoCertifiedViableAction)
            )
            break

        selected[decision.candidate.action_type.value] += 1
        selection_sources[decision.selection_source] += 1
        if decision.liveness_forced:
            audit["liveness_forced_decisions"] += 1
            audit["liveness_forced_delivery_decisions"] += int(
                decision.candidate.action_type is ViabilityActionType.DELIVER
            )
            if not previous_liveness_forced:
                audit["liveness_guard_activations"] += 1
        previous_liveness_forced = bool(decision.liveness_forced)
        steps_before_macro = steps
        execution = _execute_audit_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=max_steps - steps,
            evaluation=not training,
        )
        macro_count += 1
        steps += execution.duration
        total_return += execution.raw_return
        episode_discounted_return += (
            (agent.config.gamma ** steps_before_macro)
            * execution.discounted_return
        )
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

        if execution.action_type == ViabilityActionType.DEFER.value:
            outcome = getattr(decision.option, "last_outcome", None)
            observed_event = bool(
                isinstance(outcome, dict)
                and outcome.get("reason") == "observed_event"
            )
            consecutive_defer = 0 if observed_event else consecutive_defer + 1
        else:
            consecutive_defer = 0

        time_limit = bool(
            steps >= max_steps
            and not env.is_state_terminal(env.current_state)
        )
        terminal_boundary = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or time_limit
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
                    state_prioritizer=None,
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
                    next_snapshot=(
                        None if terminal_boundary else pending_snapshot
                    ),
                    done=terminal_boundary,
                )
                outcome_observed = True
            except (NoCertifiedViableAction, RuntimeError, ValueError) as error:
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
                    next_snapshot=(
                        None if terminal_boundary else pending_snapshot
                    ),
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
        if time_limit and method_failure_reason is None:
            method_failure_reason = "episode_step_limit"
        if terminal_boundary:
            break

    terminal = bool(env.is_state_terminal(env.current_state))
    if steps >= max_steps and not terminal and method_failure_reason is None:
        method_failure_reason = "episode_step_limit"
    success = bool(terminal and method_failure_reason is None)
    horizon_cap_hit = bool(steps >= max_steps and not terminal)
    # Report one common operational window across both reward arms.  The
    # dense breakpoint remains separately authenticated in the objective spec.
    timing = summarize_delivery_timing(
        delivery_deviations, objective_spec.legacy_window
    )
    run = {
        "protocol": str(run_protocol),
        "split": "training" if training else "development_validation",
        "instance_seed": int(instance_seed),
        "episode_instance_id": instance.instance_id,
        "return": float(total_return),
        "episode_start_discounted_return": float(
            episode_discounted_return
        ),
        "success": success,
        "horizon_cap_hit": horizon_cap_hit,
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
        "selection_source_counts": dict(selection_sources),
        "loss_mean": _mean(losses),
        "loss_updates": len(losses),
        "delivery_deviations": tuple(delivery_deviations),
        "target_window": float(objective_spec.legacy_window),
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
            "finite_horizon_terminal_boundary": True,
        },
        "wall_seconds": perf_counter() - started,
    }
    _rescore_run(run, objective_spec)
    objective_integrity = env.objective_audit_summary()
    selected_logged = float(
        objective_integrity["return_by_objective"][objective_spec.objective]
    )
    if not math.isclose(
        selected_logged, total_return, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError(
            "selected timing-objective component total does not match the "
            "observed environment return"
        )
    if not math.isclose(
        float(objective_integrity["return_by_objective"][LEGACY_CLIPPED]),
        float(run["legacy_rescored_return"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ) or not math.isclose(
        float(objective_integrity["return_by_objective"][DENSE_PIECEWISE]),
        float(run["dense_rescored_return"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("offline dual rescore disagrees with step records")
    run["objective_audit_integrity"] = objective_integrity
    component_records = env.objective_audit_records
    legacy_discounted = sum(
        (agent.config.gamma**index)
        * float(record["step_reward_by_objective"][LEGACY_CLIPPED])
        for index, record in enumerate(component_records)
    )
    dense_discounted = sum(
        (agent.config.gamma**index)
        * float(record["step_reward_by_objective"][DENSE_PIECEWISE])
        for index, record in enumerate(component_records)
    )
    selected_discounted = (
        legacy_discounted
        if objective_spec.objective == LEGACY_CLIPPED
        else dense_discounted
    )
    if not math.isclose(
        selected_discounted,
        episode_discounted_return,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "macro-composed episode discount disagrees with primitive-step "
            "objective records"
        )
    run["legacy_episode_start_discounted_return"] = float(
        legacy_discounted
    )
    run["dense_episode_start_discounted_return"] = float(dense_discounted)
    return run


def summarize_objective_runs(runs: Sequence[dict]) -> dict:
    runs = tuple(runs)
    deviations = [
        float(value)
        for run in runs
        for value in run.get("delivery_deviations", ())
    ]
    deliveries = len(deviations)
    relocations = sum(
        int(run.get("audit", {}).get("relocations", 0)) for run in runs
    )
    timing = summarize_delivery_timing(
        deviations,
        float(runs[0]["target_window"]) if runs else 20.0,
    )
    target_window = (
        float(runs[0]["target_window"]) if runs else 20.0
    )
    forced = sum(
        int(run.get("audit", {}).get("liveness_forced_decisions", 0))
        for run in runs
    )
    forced_deliveries = sum(
        int(
            run.get("audit", {}).get(
                "liveness_forced_delivery_decisions", 0
            )
        )
        for run in runs
    )
    macro_decisions = sum(int(run["macro_decisions"]) for run in runs)
    selected_actions = sum(
        (
            Counter(run.get("selected_action_counts", {}))
            for run in runs
        ),
        Counter(),
    )
    position_summary = _delivery_position_summary(runs, target_window)
    return {
        "episodes": len(runs),
        "mean_selected_objective_return": _mean(
            run["selected_objective_return"] for run in runs
        ),
        "selected_objective_return_std": _std(
            run["selected_objective_return"] for run in runs
        ),
        "mean_episode_start_discounted_return": _mean(
            run["episode_start_discounted_return"] for run in runs
        ),
        "mean_legacy_episode_start_discounted_return": _mean(
            run["legacy_episode_start_discounted_return"] for run in runs
        ),
        "mean_dense_episode_start_discounted_return": _mean(
            run["dense_episode_start_discounted_return"] for run in runs
        ),
        "mean_legacy_rescored_return": _mean(
            run["legacy_rescored_return"] for run in runs
        ),
        "mean_dense_rescored_return": _mean(
            run["dense_rescored_return"] for run in runs
        ),
        "strict_method_success_rate": _mean(
            float(run["strict_method_success"]) for run in runs
        ),
        "completion_rate": _mean(float(run["success"]) for run in runs),
        "horizon_cap_hit_rate": _mean(
            float(run.get("horizon_cap_hit", False)) for run in runs
        ),
        "horizon_empirically_nonbinding": not any(
            bool(run.get("horizon_cap_hit", False)) for run in runs
        ),
        "mean_steps": _mean(run["steps"] for run in runs),
        "total_macro_decisions": macro_decisions,
        "total_deliveries": deliveries,
        "total_relocations": relocations,
        "relocations_per_100_deliveries": (
            100.0 * relocations / deliveries if deliveries else math.nan
        ),
        "total_liveness_forced_decisions": forced,
        "liveness_forced_decision_rate": (
            forced / macro_decisions if macro_decisions else math.nan
        ),
        "total_liveness_guard_activations": sum(
            int(
                run.get("audit", {}).get(
                    "liveness_guard_activations", 0
                )
            )
            for run in runs
        ),
        "forced_action_delivery_share": (
            forced_deliveries / forced if forced else math.nan
        ),
        "guard_forced_share_of_deliveries": (
            forced_deliveries / selected_actions.get("deliver", 0)
            if selected_actions.get("deliver", 0)
            else math.nan
        ),
        "selected_action_counts": dict(selected_actions),
        "selection_source_counts": dict(
            sum(
                (
                    Counter(run.get("selection_source_counts", {}))
                    for run in runs
                ),
                Counter(),
            )
        ),
        "method_failures": tuple(
            {
                "instance_seed": int(run["instance_seed"]),
                "reason": run["method_failure_reason"],
            }
            for run in runs
            if run.get("method_failure_reason") is not None
        ),
        "first_two_mean_signed_deviation": position_summary["first_two"][
            "mean_signed_deviation"
        ],
        "first_two_mean_absolute_error": position_summary["first_two"][
            "mean_absolute_error"
        ],
        "first_two_within_target_window_rate": position_summary[
            "first_two"
        ]["within_target_window_rate"],
        "positions_three_plus_mean_signed_deviation": position_summary[
            "positions_three_plus"
        ]["mean_signed_deviation"],
        "positions_three_plus_mean_absolute_error": position_summary[
            "positions_three_plus"
        ]["mean_absolute_error"],
        "positions_three_plus_within_target_window_rate": position_summary[
            "positions_three_plus"
        ]["within_target_window_rate"],
        **timing,
        **position_summary,
    }


def _training_agent_signature(
    agent: EpisodicViabilityGraphHierarchyAgent,
) -> tuple:
    return (
        agent.transition_count,
        agent.decision_count,
        agent.gradient_steps,
        agent.target_updates,
        len(agent.replay),
        agent.rng.getstate(),
        tuple(agent.mode_decisions.items()),
        tuple(agent.selection_sources.items()),
    )


def _run_validation(
    agent: EpisodicViabilityGraphHierarchyAgent,
    args,
    *,
    objective_spec: TimingObjectiveSpec,
    device: torch.device,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    run_protocol: str = ARM_TRAINING_PROTOCOL,
) -> tuple[list[dict], dict]:
    """Evaluate a frozen audit clone without advancing training state/RNG."""

    training_signature = _training_agent_signature(agent)
    global_rng = _capture_global_rng_state()
    clone_payload = agent.checkpoint(
        include_replay=False,
        training_protocol=str(run_protocol),
        validation_clone_source=True,
    )
    evaluator = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        clone_payload,
        device=device,
        resumable=False,
        seed=args.model_seed,
    )
    evaluator.set_epsilon(0.0)
    evaluator.Q_local.eval()
    evaluator.Q_target.eval()
    initial_decisions = evaluator.decision_count
    env = _new_env(args, objective_spec)
    runs = []
    try:
        for index, instance_seed in enumerate(args.validation_seeds):
            env.current_episode = index + 1
            run = run_objective_audit_episode(
                evaluator,
                env,
                objective_spec=objective_spec,
                instance_seed=int(instance_seed),
                training=False,
                max_steps=args.max_steps,
                search_config=search_config,
                liveness_rule=liveness_rule,
                epsilon_start=0.0,
                epsilon_end=0.0,
                epsilon_decay_decisions=1,
                updates_per_macro=1,
                epsilon_warmup_decisions=0,
                run_protocol=run_protocol,
            )
            runs.append(run)
    finally:
        _restore_global_rng_state(global_rng)
    if _training_agent_signature(agent) != training_signature:
        raise RuntimeError("validation mutated the training agent")
    summary = summarize_objective_runs(runs)
    summary.update(
        {
            "instance_seeds": tuple(args.validation_seeds),
            "deployment_decisions": (
                evaluator.decision_count - initial_decisions
            ),
            "evaluation_epsilon": 0.0,
            "evaluation_learning": False,
            "replay_size": len(evaluator.replay),
            "q_local_training_mode": bool(evaluator.Q_local.training),
            "fresh_certificate_cache_per_episode": True,
            "viability_critic_enabled": False,
            "dual_rescoring": True,
        }
    )
    return runs, summary


def _validation_record(
    runs: Sequence[dict],
    summary: dict,
    *,
    checkpoint_episode: int,
    number_blocks: int,
) -> dict:
    mean_return_per_block = (
        float(summary["mean_selected_objective_return"]) / number_blocks
    )
    mae = float(summary["mean_absolute_error"])
    relocation_rate = float(summary["relocations_per_100_deliveries"])
    score = (
        float(summary["strict_method_success_rate"]),
        float(summary["completion_rate"]),
        mean_return_per_block,
        -mae,
        -relocation_rate,
        -int(checkpoint_episode),
    )
    authoritative = all(
        bool(run["audit"].get("exact_verifier_authoritative"))
        and not bool(run["audit"].get("baseline_viability_teacher"))
        and not bool(run["audit"].get("baseline_policy_query"))
        for run in runs
    )
    all_delivered = all(
        int(run.get("delivery_count", 0)) == int(number_blocks)
        for run in runs
    )
    eligible = bool(
        float(summary["strict_method_success_rate"]) == 1.0
        and float(summary["completion_rate"]) == 1.0
        and not summary["method_failures"]
        and all_delivered
        and authoritative
    )
    return {
        "checkpoint_episode": int(checkpoint_episode),
        "selection_score": score,
        "selection_score_fields": (
            "strict_success_rate",
            "completion_rate",
            "selected_objective_return_per_block",
            "negative_mean_absolute_error",
            "negative_relocations_per_100_deliveries",
            "negative_checkpoint_episode",
        ),
        "deployment_eligible": eligible,
        "development_selection_only": True,
        "summary": summary,
        "runs": tuple(runs),
    }


def _score_tuple(record: Optional[dict]) -> Optional[tuple]:
    if record is None:
        return None
    return tuple(float(value) for value in record["selection_score"])


def _common_matrix_contract(args, train_seed_base: int) -> dict:
    shared_config = _graph_config(args, 0.99).to_dict()
    shared_config.pop("gamma")
    return {
        "matrix_protocol": MATRIX_PROTOCOL,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "mechanism_screen_not_performance_confirmation": True,
        "model_seed": int(args.model_seed),
        "total_episodes_per_arm": int(args.total_episodes),
        "train_instance_seed_base": int(train_seed_base),
        "validation_instance_seeds": tuple(args.validation_seeds),
        "conditions": tuple(
            {
                "arm_id": _arm_id(objective, gamma),
                "objective": objective,
                "gamma": gamma,
            }
            for objective in OBJECTIVES
            for gamma in GAMMAS
        ),
        "environment": {
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "number_blocks": args.number_blocks,
            "arrival_rate": args.arrival_rate,
            "proc_mean": args.proc_mean,
            "max_steps": args.max_steps,
        },
        "timing_objective_coefficients": {
            name: value
            for name, value in _objective_spec(
                args, LEGACY_CLIPPED
            ).to_dict().items()
            if name != "objective"
        },
        "shared_graph_and_training_config_excluding_factor_gamma": (
            shared_config
        ),
        "search_config": asdict(_search_config(args)),
        "liveness_rule": asdict(_liveness_rule(args)),
        "updates_per_macro": args.updates_per_macro,
        "epsilon_schedule": {
            "start": args.epsilon_start,
            "end": args.epsilon_end,
            "warmup_decisions": args.epsilon_warmup_decisions,
            "decay_decisions": args.epsilon_decay_decisions,
        },
        "validation_every_episodes": args.eval_every,
        "checkpoint_every_episodes": args.checkpoint_every,
        "same_initialization_seed_across_arms": True,
        "same_episode_instances_across_arms": True,
        "same_validation_panel_across_arms": True,
        "same_network_architecture_across_arms": True,
        "internal_factorial_control_not_canonical_vcg_v1_replication": True,
        "exact_safe_frontier_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "future_schedule_visible_to_policy": False,
        "sealed_test_panels_opened": False,
    }


def _arm_resume_contract(
    args,
    *,
    arm_id: str,
    objective_spec: TimingObjectiveSpec,
    gamma: float,
    train_seed_base: int,
    graph_config: EpisodicViabilityGraphConfig,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    matrix_contract_hash: str,
) -> dict:
    return {
        "training_protocol": ARM_TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "matrix_protocol": MATRIX_PROTOCOL,
        "matrix_contract_sha256": matrix_contract_hash,
        "arm_id": arm_id,
        "objective_spec": objective_spec.to_dict(),
        "gamma": float(gamma),
        "model_seed": int(args.model_seed),
        "total_episodes": int(args.total_episodes),
        "train_instance_seed_base": int(train_seed_base),
        "validation_instance_seeds": tuple(args.validation_seeds),
        "environment": {
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "number_blocks": args.number_blocks,
            "arrival_rate": args.arrival_rate,
            "proc_mean": args.proc_mean,
            "max_steps": args.max_steps,
        },
        "graph_config": graph_config.to_dict(),
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "updates_per_macro": args.updates_per_macro,
        "epsilon_schedule": {
            "start": args.epsilon_start,
            "end": args.epsilon_end,
            "warmup_decisions": args.epsilon_warmup_decisions,
            "decay_decisions": args.epsilon_decay_decisions,
        },
        "validation_every_episodes": args.eval_every,
        "exact_verifier_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "terminal_boundary_contract": EPISODIC_TERMINAL_BOUNDARY_CONTRACT,
        "dual_rescore_fixed_realized_trajectory": True,
        "sealed_test_panels_opened": False,
    }


def _build_arm_checkpoint(
    agent: EpisodicViabilityGraphHierarchyAgent,
    *,
    include_replay: bool,
    resumable: bool,
    completed_episodes: int,
    train_seed_base: int,
    resume_contract: dict,
    train_history: Sequence[dict],
    validation_history: Sequence[dict],
    best_record: Optional[dict],
    best_checkpoint_sha256: Optional[str],
) -> dict:
    objective_spec = resume_contract["objective_spec"]
    return agent.checkpoint(
        include_replay=include_replay,
        protocol=ARM_TRAINING_PROTOCOL,
        training_protocol=ARM_TRAINING_PROTOCOL,
        trainer_schema_version=TRAINER_SCHEMA_VERSION,
        trainer_resumable=bool(resumable),
        completed_training_episodes=int(completed_episodes),
        next_training_episode=int(completed_episodes) + 1,
        next_train_instance_seed=int(train_seed_base + completed_episodes),
        resume_contract=resume_contract,
        matrix_protocol=MATRIX_PROTOCOL,
        matrix_contract_sha256=resume_contract["matrix_contract_sha256"],
        factorial_arm_id=resume_contract["arm_id"],
        timing_objective=objective_spec["objective"],
        timing_objective_spec=dict(objective_spec),
        train_instance_seeds=tuple(
            range(train_seed_base, train_seed_base + completed_episodes)
        ),
        validation_instance_seeds=tuple(
            resume_contract["validation_instance_seeds"]
        ),
        environment=dict(resume_contract["environment"]),
        viability_search=dict(resume_contract["search_config"]),
        liveness_rule=dict(resume_contract["liveness_rule"]),
        smdp_return_contract=SMDP_RETURN_CONTRACT,
        no_fallback_contract=NO_FALLBACK_CONTRACT,
        certificate_scope=CERTIFICATE_SCOPE,
        training_history=tuple(train_history),
        validation_history=tuple(validation_history),
        best_validation_record=best_record,
        best_checkpoint_sha256=best_checkpoint_sha256,
        global_rng_state=(
            _capture_global_rng_state() if resumable else None
        ),
        exact_full=True,
        viability_critic_enabled=False,
        future_schedule_visible_to_policy=False,
        sealed_in_regime_test_seeds=tuple(
            sorted(SEALED_IN_REGIME_TEST_SEEDS)
        ),
        sealed_stress_test_seeds=tuple(
            sorted(SEALED_STRESS_V1_HOLDOUT_SEEDS)
        ),
        test_panels_opened=False,
        development_mechanism_screen_only=True,
    )


def _validate_arm_resume_payload(payload: dict, contract: dict) -> None:
    expected = {
        "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
        "training_protocol": ARM_TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "trainer_resumable": True,
        "matrix_protocol": MATRIX_PROTOCOL,
        "factorial_arm_id": contract["arm_id"],
        "timing_objective": contract["objective_spec"]["objective"],
        "gamma": contract["gamma"],
        "exact_full": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "test_panels_opened": False,
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"incompatible objective-audit resume checkpoint: {mismatches!r}"
        )
    if payload.get("resume_contract") != contract:
        raise ValueError("objective-audit arm resume contract mismatch")
    state = payload.get("agent_state", {})
    required = ("optimizer", "replay", "rng_state")
    missing = tuple(name for name in required if name not in state)
    if missing:
        raise ValueError(f"resumable checkpoint is missing {missing!r}")


def _arm_result_payload(
    args,
    *,
    arm_dir: Path,
    completed_episodes: int,
    train_seed_base: int,
    contract: dict,
    train_history: Sequence[dict],
    validation_history: Sequence[dict],
    best_record: Optional[dict],
    best_checkpoint_sha256: Optional[str],
    agent: EpisodicViabilityGraphHierarchyAgent,
) -> dict:
    complete = int(completed_episodes) == int(args.total_episodes)
    final_record = validation_history[-1] if validation_history else None
    return {
        "training_protocol": ARM_TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "matrix_protocol": MATRIX_PROTOCOL,
        "status": "complete" if complete else "paused",
        "mechanism_screen_not_performance_confirmation": True,
        "formal_test_authorized": False,
        "test_panels_opened": False,
        "arm_id": contract["arm_id"],
        "objective_spec": contract["objective_spec"],
        "gamma": contract["gamma"],
        "completed_training_episodes": int(completed_episodes),
        "total_training_episodes": int(args.total_episodes),
        "model_seed": int(args.model_seed),
        "train_instance_seed_base": int(train_seed_base),
        "validation_instance_seeds": tuple(args.validation_seeds),
        "resume_contract": contract,
        "training": {
            "runs": tuple(train_history),
            "summary": summarize_objective_runs(train_history),
        },
        "validation_history": tuple(validation_history),
        "final_validation_record": final_record,
        "best_validation_record": best_record,
        "deployment_checkpoint_eligible": bool(
            best_record and best_record["deployment_eligible"]
        ),
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "agent_audit": agent.audit(include_decisions=False),
        "latest_checkpoint": str((arm_dir / "latest.pth").resolve()),
        "best_checkpoint": (
            str((arm_dir / "best.pth").resolve())
            if best_record is not None
            else None
        ),
    }


def _run_arm(
    args,
    *,
    objective: str,
    gamma: float,
    train_seed_base: int,
    matrix_contract_hash: str,
    device: torch.device,
) -> dict:
    arm_id = _arm_id(objective, gamma)
    arm_dir = args.output_dir / arm_id
    latest_path = arm_dir / "latest.pth"
    best_path = arm_dir / "best.pth"
    result_path = arm_dir / "training-summary.json"
    objective_spec = _objective_spec(args, objective)
    graph_config = _graph_config(args, gamma)
    search_config = _search_config(args)
    liveness_rule = _liveness_rule(args)
    contract = _arm_resume_contract(
        args,
        arm_id=arm_id,
        objective_spec=objective_spec,
        gamma=gamma,
        train_seed_base=train_seed_base,
        graph_config=graph_config,
        search_config=search_config,
        liveness_rule=liveness_rule,
        matrix_contract_hash=matrix_contract_hash,
    )

    if arm_dir.exists() and any(arm_dir.iterdir()):
        if not args.resume_existing:
            raise FileExistsError(
                f"arm {arm_id} is nonempty; pass --resume-existing or use "
                "a fresh matrix output directory"
            )
        if not latest_path.is_file():
            raise ValueError(f"arm {arm_id} has no resumable latest.pth")
        payload = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, dict):
            raise ValueError("resume checkpoint must contain a mapping")
        _validate_arm_resume_payload(payload, contract)
        completed = int(payload["completed_training_episodes"])
        if completed == args.total_episodes:
            if not result_path.is_file():
                raise ValueError(f"complete arm {arm_id} has no summary")
            with result_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        agent = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
            payload,
            device=device,
            resumable=True,
            seed=args.model_seed,
        )
        _optimizer_to(agent.optimizer, device)
        train_history = list(payload.get("training_history", ()))
        validation_history = list(payload.get("validation_history", ()))
        best_record = payload.get("best_validation_record")
        best_checkpoint_sha256 = payload.get("best_checkpoint_sha256")
        if len(train_history) != completed:
            raise ValueError("resume training history length is inconsistent")
        if best_record is not None:
            if not best_path.is_file() or not best_checkpoint_sha256:
                raise ValueError("resume is missing its authenticated best.pth")
            if _sha256(best_path) != best_checkpoint_sha256:
                raise ValueError("resume best.pth SHA-256 mismatch")
        _restore_global_rng_state(payload["global_rng_state"])
        print(f"[{arm_id}] resumed at episode {completed}", flush=True)
    else:
        arm_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(args.model_seed)
        agent = EpisodicViabilityGraphHierarchyAgent(
            config=graph_config,
            seed=args.model_seed,
            device=device,
            epsilon=args.epsilon_start,
        )
        completed = 0
        train_history = []
        validation_history = []
        best_record = None
        best_checkpoint_sha256 = None

    print(
        f"[{arm_id}] objective={objective} gamma={gamma:.2f} "
        f"episodes={args.total_episodes} device={device} "
        "exact_full=true critic=false teacher=false",
        flush=True,
    )
    env = _new_env(args, objective_spec)
    stop_at = args.total_episodes
    if args.stop_after_episode is not None:
        stop_at = min(stop_at, args.stop_after_episode)
    if completed > stop_at:
        raise ValueError(
            f"arm {arm_id} resume is beyond --stop-after-episode"
        )

    for episode_index in range(completed, stop_at):
        episode_number = episode_index + 1
        instance_seed = train_seed_base + episode_index
        env.current_episode = episode_number
        run = run_objective_audit_episode(
            agent,
            env,
            objective_spec=objective_spec,
            instance_seed=instance_seed,
            training=True,
            max_steps=args.max_steps,
            search_config=search_config,
            liveness_rule=liveness_rule,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_decisions=args.epsilon_decay_decisions,
            updates_per_macro=args.updates_per_macro,
            epsilon_warmup_decisions=args.epsilon_warmup_decisions,
        )
        train_history.append(run)
        agent.decision_log.clear()
        completed = episode_number
        agent.set_epsilon(
            epsilon_at(
                agent.decision_count,
                start=args.epsilon_start,
                end=args.epsilon_end,
                warmup_decisions=args.epsilon_warmup_decisions,
                decay_decisions=args.epsilon_decay_decisions,
            )
        )

        if episode_number == 1 or episode_number % args.log_every == 0:
            recent = train_history[-min(10, len(train_history)) :]
            print(
                f"[{arm_id}] Ep {episode_number:4d} | "
                f"R {_mean(item['return'] for item in recent):8.2f} | "
                f"Strict {int(run['strict_method_success'])} | "
                f"Dec {run['macro_decisions']:3d} | "
                f"Replay {len(agent.replay):5d} | "
                f"Grad {agent.gradient_steps:6d} | "
                f"Eps {agent.epsilon:.3f}",
                flush=True,
            )

        validation_due = bool(
            episode_number % args.eval_every == 0
            or episode_number == args.total_episodes
        )
        if validation_due:
            validation_runs, validation_summary = _run_validation(
                agent,
                args,
                objective_spec=objective_spec,
                device=device,
                search_config=search_config,
                liveness_rule=liveness_rule,
            )
            record = _validation_record(
                validation_runs,
                validation_summary,
                checkpoint_episode=episode_number,
                number_blocks=args.number_blocks,
            )
            validation_history.append(record)
            print(
                f"[{arm_id}] Val Ep {episode_number:4d} | "
                f"Strict {validation_summary['strict_method_success_rate']:.3f} | "
                f"Complete {validation_summary['completion_rate']:.3f} | "
                f"MAE {validation_summary['mean_absolute_error']:.3f} | "
                f"LegacyR {validation_summary['mean_legacy_rescored_return']:.2f} | "
                f"DenseR {validation_summary['mean_dense_rescored_return']:.2f}",
                flush=True,
            )
            if best_record is None or _score_tuple(record) > _score_tuple(
                best_record
            ):
                best_record = record
                best_payload = _build_arm_checkpoint(
                    agent,
                    include_replay=False,
                    resumable=False,
                    completed_episodes=completed,
                    train_seed_base=train_seed_base,
                    resume_contract=contract,
                    train_history=train_history,
                    validation_history=validation_history,
                    best_record=best_record,
                    best_checkpoint_sha256=None,
                )
                _atomic_torch_save(best_payload, best_path)
                best_checkpoint_sha256 = _sha256(best_path)

        checkpoint_due = bool(
            validation_due
            or episode_number % args.checkpoint_every == 0
            or episode_number == stop_at
        )
        if checkpoint_due:
            latest = _build_arm_checkpoint(
                agent,
                include_replay=True,
                resumable=True,
                completed_episodes=completed,
                train_seed_base=train_seed_base,
                resume_contract=contract,
                train_history=train_history,
                validation_history=validation_history,
                best_record=best_record,
                best_checkpoint_sha256=best_checkpoint_sha256,
            )
            _atomic_torch_save(latest, latest_path)
            result = _arm_result_payload(
                args,
                arm_dir=arm_dir,
                completed_episodes=completed,
                train_seed_base=train_seed_base,
                contract=contract,
                train_history=train_history,
                validation_history=validation_history,
                best_record=best_record,
                best_checkpoint_sha256=best_checkpoint_sha256,
                agent=agent,
            )
            _atomic_json_save(result, result_path)

    if completed == 0:
        raise RuntimeError(f"arm {arm_id} completed no training episode")
    return _arm_result_payload(
        args,
        arm_dir=arm_dir,
        completed_episodes=completed,
        train_seed_base=train_seed_base,
        contract=contract,
        train_history=train_history,
        validation_history=validation_history,
        best_record=best_record,
        best_checkpoint_sha256=best_checkpoint_sha256,
        agent=agent,
    )


def _arm_report(result: dict) -> dict:
    final_record = result.get("final_validation_record")
    best_record = result.get("best_validation_record")
    final_summary = final_record["summary"] if final_record else None
    return {
        "arm_id": result["arm_id"],
        "objective": result["objective_spec"]["objective"],
        "gamma": result["gamma"],
        "status": result["status"],
        "completed_training_episodes": result[
            "completed_training_episodes"
        ],
        "final_checkpoint_episode": (
            final_record["checkpoint_episode"] if final_record else None
        ),
        "best_checkpoint_episode": (
            best_record["checkpoint_episode"] if best_record else None
        ),
        "final_validation": final_summary,
        "best_validation": best_record["summary"] if best_record else None,
        "best_checkpoint": result.get("best_checkpoint"),
        "latest_checkpoint": result.get("latest_checkpoint"),
    }


def _paired_final_rows(arm_results: Sequence[dict]) -> tuple[dict, ...]:
    by_arm = {}
    for result in arm_results:
        record = result.get("final_validation_record")
        if record is None:
            continue
        by_arm[result["arm_id"]] = {
            int(run["instance_seed"]): run for run in record["runs"]
        }
    if len(by_arm) != len(tuple(arm_results)) or not by_arm:
        return ()
    rows = []
    for seed in sorted(set.intersection(*(set(value) for value in by_arm.values()))):
        row = {"instance_seed": seed}
        for arm_id in sorted(by_arm):
            run = by_arm[arm_id][seed]
            row[arm_id] = {
                "strict_method_success": bool(run["strict_method_success"]),
                "completion": bool(run["success"]),
                "legacy_rescored_return": run["legacy_rescored_return"],
                "dense_rescored_return": run["dense_rescored_return"],
                "legacy_episode_start_discounted_return": run[
                    "legacy_episode_start_discounted_return"
                ],
                "dense_episode_start_discounted_return": run[
                    "dense_episode_start_discounted_return"
                ],
                "steps": run["steps"],
                "mean_signed_deviation": run["mean_signed_deviation"],
                "mean_absolute_error": run["mean_absolute_error"],
                "mean_tardiness": run["mean_tardiness"],
                "within_target_window_rate": run[
                    "within_target_window_rate"
                ],
                "relocations": run["audit"].get("relocations", 0),
            }
        rows.append(row)
    return tuple(rows)


def _factorial_contrasts(arm_results: Sequence[dict]) -> tuple[dict, ...]:
    reports = {result["arm_id"]: _arm_report(result) for result in arm_results}
    metrics = (
        "strict_method_success_rate",
        "completion_rate",
        "horizon_cap_hit_rate",
        "mean_legacy_rescored_return",
        "mean_dense_rescored_return",
        "mean_legacy_episode_start_discounted_return",
        "mean_dense_episode_start_discounted_return",
        "mean_steps",
        "mean_signed_deviation",
        "mean_absolute_error",
        "mean_tardiness",
        "within_target_window_rate",
        "relocations_per_100_deliveries",
        "liveness_forced_decision_rate",
        "forced_action_delivery_share",
        "guard_forced_share_of_deliveries",
        "first_two_mean_signed_deviation",
        "first_two_mean_absolute_error",
        "first_two_within_target_window_rate",
        "positions_three_plus_mean_signed_deviation",
        "positions_three_plus_mean_absolute_error",
        "positions_three_plus_within_target_window_rate",
    )
    pairs = []
    for gamma in GAMMAS:
        pairs.append(
            (
                f"dense_minus_legacy_at_{_gamma_label(gamma)}",
                _arm_id(DENSE_PIECEWISE, gamma),
                _arm_id(LEGACY_CLIPPED, gamma),
            )
        )
    for objective in OBJECTIVES:
        pairs.append(
            (
                f"gamma1_minus_gamma0p99_for_{OBJECTIVE_LABELS[objective]}",
                _arm_id(objective, 1.0),
                _arm_id(objective, 0.99),
            )
        )
    contrasts = []
    def difference(left_value, right_value):
        if left_value is None or right_value is None:
            return None
        left_value = float(left_value)
        right_value = float(right_value)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            return None
        return left_value - right_value

    for name, left, right in pairs:
        left_summary = reports[left]["final_validation"]
        right_summary = reports[right]["final_validation"]
        if left_summary is None or right_summary is None:
            continue
        contrasts.append(
            {
                "contrast": name,
                "left_arm": left,
                "right_arm": right,
                "difference_left_minus_right": {
                    metric: difference(
                        left_summary[metric], right_summary[metric]
                    )
                    for metric in metrics
                },
                "descriptive_mechanism_screen_only": True,
            }
        )
    return tuple(contrasts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired development-only 2x2 VCG objective/gamma mechanism audit"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--total-episodes", "--episodes", type=int, default=DEFAULT_TOTAL_EPISODES
    )
    parser.add_argument("--train-instance-seed-base", type=int)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--stop-after-episode", type=int)

    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--proc-mean", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    parser.add_argument("--legacy-base", type=float, default=10.0)
    parser.add_argument("--legacy-max-bonus", type=float, default=30.0)
    parser.add_argument("--dense-b", type=float, default=40.0)
    parser.add_argument("--lambda-abs", type=float, default=1.5)
    parser.add_argument("--lambda-outside", type=float, default=0.5)
    parser.add_argument("--target-window", type=float, default=20.0)

    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-macro", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.90)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-warmup-decisions", type=int, default=0)
    parser.add_argument("--epsilon-decay-decisions", type=int, default=3_000)

    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--graph-embedding-dim", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--action-embedding-dim", type=int, default=32)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--tau-accept", type=float, default=0.1)
    parser.add_argument("--tau-recover", type=float, default=0.1)
    parser.add_argument("--tau-defer", type=float, default=0.1)
    parser.add_argument("--tau-mode", type=float, default=1.0)

    parser.add_argument("--search-max-depth", type=int)
    parser.add_argument("--search-max-nodes", type=int, default=20_000)
    parser.add_argument("--search-max-primitive-steps", type=int)
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
    return parser


def _validate_args(args) -> int:
    positive = (
        "total_episodes",
        "eval_every",
        "checkpoint_every",
        "log_every",
        "grid_rows",
        "grid_cols",
        "number_blocks",
        "proc_mean",
        "max_steps",
        "batch_size",
        "replay_size",
        "update_every",
        "updates_per_macro",
        "target_update_every",
        "epsilon_decay_decisions",
        "graph_hidden_dim",
        "graph_embedding_dim",
        "action_embedding_dim",
        "head_hidden_dim",
        "search_max_nodes",
        "max_defer_steps",
        "max_consecutive_defers",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.model_seed < 0:
        raise ValueError("--model-seed must be non-negative")
    if args.total_episodes >= TRAIN_SEED_STRIDE:
        raise ValueError("episode budget exceeds model-seed namespace")
    if args.replay_size < args.batch_size:
        raise ValueError("--replay-size must be at least --batch-size")
    if args.message_passing_steps < 0:
        raise ValueError("message-passing steps must be non-negative")
    if args.max_nonprogress_recovery_decisions < 0:
        raise ValueError("nonprogress recovery bound must be non-negative")
    if args.epsilon_warmup_decisions < 0:
        raise ValueError("epsilon warmup must be non-negative")
    if not args.validation_seeds or len(set(args.validation_seeds)) != len(
        args.validation_seeds
    ):
        raise ValueError("validation seeds must be nonempty and unique")
    if any(seed < 0 for seed in args.validation_seeds):
        raise ValueError("validation seeds must be non-negative")
    if args.stop_after_episode is not None and not (
        1 <= args.stop_after_episode <= args.total_episodes
    ):
        raise ValueError("--stop-after-episode must lie inside the budget")
    if not math.isfinite(args.arrival_rate) or args.arrival_rate < 0.0:
        raise ValueError("arrival rate must be finite and non-negative")
    if not 0.0 <= args.epsilon_start <= 1.0 or not 0.0 <= args.epsilon_end <= 1.0:
        raise ValueError("epsilon endpoints must be in [0, 1]")

    # The legacy factor must remain the unmodified VCG-v1 objective.  The
    # dense coefficients are the configurable mechanism under study.
    if (
        float(args.legacy_base) != 10.0
        or float(args.legacy_max_bonus) != 30.0
        or float(args.target_window) != 20.0
    ):
        raise ValueError(
            "the factorial legacy arm is frozen at base=10, bonus=30, "
            "window=20"
        )
    for objective in OBJECTIVES:
        _objective_spec(args, objective)

    train_seed_base = (
        _derived_train_seed_base(args.model_seed)
        if args.train_instance_seed_base is None
        else int(args.train_instance_seed_base)
    )
    if train_seed_base < 0:
        raise ValueError("training seed base must be non-negative")
    train_seeds = set(
        range(train_seed_base, train_seed_base + args.total_episodes)
    )
    validation_seeds = set(args.validation_seeds)
    if train_seeds.intersection(validation_seeds):
        raise ValueError("training and validation EpisodeInstance seeds overlap")
    opened = (train_seeds | validation_seeds).intersection(PROTECTED_SEEDS)
    if opened:
        raise ValueError(
            f"objective audit refuses sealed test seeds: {tuple(sorted(opened))}"
        )
    return train_seed_base


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    train_seed_base = _validate_args(args)
    device = resolve_device(args.device)
    matrix_contract = _common_matrix_contract(args, train_seed_base)
    matrix_hash = _contract_hash(matrix_contract)
    manifest_path = args.output_dir / "matrix-contract.json"
    result_path = args.output_dir / "matrix-summary.json"

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume_existing:
            raise FileExistsError(
                "fresh matrix refuses a nonempty output directory; pass "
                "--resume-existing or choose a new directory"
            )
        if not manifest_path.is_file():
            raise ValueError("matrix output has no authenticated contract")
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("matrix_contract_sha256") != matrix_hash:
            raise ValueError("matrix resume contract hash mismatch")
        if existing.get("contract") != _json_safe(matrix_contract):
            raise ValueError("matrix resume contract payload mismatch")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_save(
            {
                "matrix_contract_sha256": matrix_hash,
                "contract": matrix_contract,
            },
            manifest_path,
        )

    arm_results = []
    for objective in OBJECTIVES:
        for gamma in GAMMAS:
            arm_results.append(
                _run_arm(
                    args,
                    objective=objective,
                    gamma=gamma,
                    train_seed_base=train_seed_base,
                    matrix_contract_hash=matrix_hash,
                    device=device,
                )
            )

    complete = all(result["status"] == "complete" for result in arm_results)
    result = {
        "matrix_protocol": MATRIX_PROTOCOL,
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_contract_sha256": matrix_hash,
        "status": "complete" if complete else "paused",
        "mechanism_screen_not_performance_confirmation": True,
        "formal_test_authorized": False,
        "sealed_test_panels_opened": False,
        "comparison_warning": (
            "Do not compare selected-objective raw returns across reward arms; "
            "use common metrics and both fixed-trajectory rescored returns."
        ),
        "arms": tuple(_arm_report(item) for item in arm_results),
        "factorial_contrasts": _factorial_contrasts(arm_results),
        "paired_final_validation": _paired_final_rows(arm_results),
        "matrix_contract": matrix_contract,
    }
    _atomic_json_save(result, result_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": {
                    arm["arm_id"]: {
                        "episodes": arm["completed_training_episodes"],
                        "final": arm["final_validation"],
                    }
                    for arm in result["arms"]
                },
                "summary": str(result_path.resolve()),
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
