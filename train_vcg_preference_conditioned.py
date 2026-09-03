#!/usr/bin/env python3
"""Lean development trainer for the preference-conditioned vector VCG.

This runner compares two learning architectures while keeping the physical
environment, exact fail-closed VCG candidate frontier, liveness guard, macro
execution, training instances, and per-episode preference schedule paired:

``conditioned`` (B)
    The vector critic receives the handling preference as an input.

``unconditioned`` (C)
    The same vector critic masks that input.  Lambda still scalarizes the two
    learned components when an action is selected.

One lambda is sampled from a balanced, reproducibly shuffled grid at the
start of each episode and remains fixed until that episode ends.  Replay
stores the preference-independent vector outcome.  In particular, handling
cost is recorded both as an exact raw count and as its exact primitive-time
discounted intra-macro cumulant; it is never reconstructed from macro length.

The trainer is deliberately development-only.  It does no interval
validation or checkpoint selection and never opens an 85k/86k/87k/88k/89k
panel.  A complete run emits one fixed terminal model checkpoint.  ``latest``
is resumable at clean episode boundaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean, pstdev
from time import perf_counter
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from example.helper.timing_metrics import summarize_delivery_timing
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    StrictDecisionBoundaryError,
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from train_viability_graph_smdp import (
    CERTIFICATE_SCOPE,
    NO_FALLBACK_CONTRACT,
    _option_failure,
    _update_frontier_audit,
    epsilon_at,
    resolve_device,
    seed_everything,
)
from vcg_objective_audit import (
    ObjectiveAuditSmallRoomsEnv,
    TimingObjectiveSpec,
)
from viability_graph_hierarchy import NoCertifiedViableAction
from viability_graph_preference_conditioned import (
    PreferenceConditionedVectorAgent,
    PreferenceConditionedVectorConfig,
)


TRAINING_PROTOCOL = "vcg_preference_conditioned_vector_development_v1"
TRAINER_SCHEMA_VERSION = 1
TERMINAL_CHECKPOINT_ROLE = "fixed_terminal_development_model"
LATEST_CHECKPOINT_ROLE = "resumable_latest_development_state"
ARCHITECTURES = ("conditioned", "unconditioned")
ARCHITECTURE_ARM_LABELS = {"conditioned": "B", "unconditioned": "C"}
PREFERENCE_GRID = (0.0, 0.025, 0.05, 0.1, 0.2)
DEFAULT_EPISODES = 500
ALLOWED_MODEL_SEEDS = (0, 1, 2)
TRAIN_SEED_ORIGIN = 50_000_000
TRAIN_SEED_STRIDE = 1_000_000
PREFERENCE_SCHEDULE_SEED_ORIGIN = 510_000_000
CHECKPOINT_EVERY_EPISODES = 5
PROTECTED_DEVELOPMENT_AND_TEST_SEEDS = frozenset(
    set(range(79_000, 80_000)) | set(range(85_000, 90_000))
)
HANDLING_EVENT_DEFINITION = "info.relocated_block_true_once_per_physical_rehandle_v1"
HANDLING_RETURN_CONTRACT = (
    "sum_i_0_to_k_minus_1_gamma_pow_i_indicator_relocated_block_v1"
)
OPERATIONAL_RETURN_CONTRACT = (
    "sum_i_0_to_k_minus_1_gamma_pow_i_dense_environment_reward_v1"
)
OBJECTIVE_SPEC = TimingObjectiveSpec.dense(
    dense_b=40.0,
    lambda_abs=1.5,
    lambda_outside=0.5,
    window=20.0,
)


@dataclass(frozen=True)
class PreferenceMacroExecution:
    """One exact-bound macro outcome at primitive transition resolution."""

    candidate_key: str
    action_type: str
    mode: str
    discounted_operational_return: float
    raw_operational_return: float
    discounted_rehandles: float
    raw_rehandles: int
    duration: int
    env_terminal: bool
    truncated: bool
    option_terminated: bool
    option_success: bool
    failure_reason: Optional[str]
    delivery_deviations: tuple[float, ...]
    illegal_drops: int


def _finite(value, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def execute_preference_macro(
    env,
    candidate: ViabilityActionCandidate,
    *,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
) -> PreferenceMacroExecution:
    """Execute one certified macro and observe both vector components.

    Rehandle discounting is performed at the primitive step where the
    environment reports ``relocated_block``.  Consequently two macros with
    equal duration and equal raw rehandle count need not have the same
    discounted handling outcome.
    """

    gamma = _finite(gamma, name="gamma")
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
        return PreferenceMacroExecution(
            candidate_key=candidate.key,
            action_type=candidate.action_type.value,
            mode=candidate.mode.value,
            discounted_operational_return=0.0,
            raw_operational_return=0.0,
            discounted_rehandles=0.0,
            raw_rehandles=0,
            duration=0,
            env_terminal=False,
            truncated=False,
            option_terminated=False,
            option_success=False,
            failure_reason="bound_option_not_initiable",
            delivery_deviations=(),
            illegal_drops=0,
        )

    discounted_operational = 0.0
    raw_operational = 0.0
    discounted_rehandles = 0.0
    raw_rehandles = 0
    deviations: list[float] = []
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
            primitive_discount = gamma ** (duration - 1)
            discounted_operational += primitive_discount * reward
            raw_operational += reward
            relocated = int(bool(info.get("relocated_block")))
            discounted_rehandles += primitive_discount * relocated
            raw_rehandles += relocated
            if "delivery_error_time" in info:
                deviations.append(float(info["delivery_error_time"]))
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
    return PreferenceMacroExecution(
        candidate_key=candidate.key,
        action_type=candidate.action_type.value,
        mode=candidate.mode.value,
        discounted_operational_return=float(discounted_operational),
        raw_operational_return=float(raw_operational),
        discounted_rehandles=float(discounted_rehandles),
        raw_rehandles=int(raw_rehandles),
        duration=int(duration),
        env_terminal=bool(terminal),
        truncated=truncated,
        option_terminated=terminated,
        option_success=failure is None and (terminated or terminal),
        failure_reason=failure,
        delivery_deviations=tuple(deviations),
        illegal_drops=int(illegal_drops),
    )


def _failure_name(prefix: str, error: Exception) -> str:
    return f"{prefix}:{type(error).__name__}:{error}"


def _mean(values) -> float:
    values = tuple(float(value) for value in values)
    return float(fmean(values)) if values else math.nan


def _std(values) -> float:
    values = tuple(float(value) for value in values)
    return float(pstdev(values)) if values else math.nan


def _loss_value(loss) -> Optional[float]:
    if loss is None:
        return None
    if isinstance(loss, Mapping):
        for key in ("loss", "total_loss"):
            if key in loss:
                return _finite(loss[key], name=key)
        raise ValueError("learn() returned a mapping without a total loss")
    return _finite(loss, name="loss")


def run_preference_episode(
    agent: PreferenceConditionedVectorAgent,
    env,
    *,
    instance_seed: int,
    preference_lambda: float,
    preference_schedule_record: Mapping,
    max_steps: int,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_decisions: int,
    epsilon_warmup_decisions: int,
    updates_per_macro: int,
) -> dict:
    """Run one dynamic 5x5 episode at one fixed behavior preference."""

    preference_lambda = _finite(
        preference_lambda, name="preference_lambda"
    )
    instance = env.sample_episode_instance(int(instance_seed))
    env.reset(instance=instance)
    agent.reset_episode_state()
    certificate_cache = ViabilityCertificateCache()
    total_return = 0.0
    episode_discounted_return = 0.0
    raw_rehandles = 0
    episode_discounted_rehandles = 0.0
    sum_intra_macro_discounted_rehandles = 0.0
    steps = 0
    macro_count = 0
    consecutive_defer = 0
    delivery_deviations: list[float] = []
    method_failure_reason = None
    losses: list[float] = []
    selected = Counter()
    selection_sources = Counter()
    macro_outcomes: list[dict] = []
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
                    state_prioritizer=None,
                )
            except (
                StrictDecisionBoundaryError,
                RuntimeError,
                ValueError,
            ) as error:
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

        epsilon = epsilon_at(
            agent.decision_count,
            start=epsilon_start,
            end=epsilon_end,
            decay_decisions=epsilon_decay_decisions,
            warmup_decisions=epsilon_warmup_decisions,
        )
        agent.set_epsilon(epsilon)
        try:
            decision = agent.select(
                snapshot,
                preference_lambda=preference_lambda,
                training=True,
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
        selection_sources[decision.selection_source] += 1
        steps_before_macro = steps
        execution = execute_preference_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=max_steps - steps,
            evaluation=False,
        )
        macro_count += 1
        steps += execution.duration
        total_return += execution.raw_operational_return
        episode_discount = agent.config.gamma ** steps_before_macro
        episode_discounted_return += (
            episode_discount * execution.discounted_operational_return
        )
        raw_rehandles += execution.raw_rehandles
        sum_intra_macro_discounted_rehandles += (
            execution.discounted_rehandles
        )
        episode_discounted_rehandles += (
            episode_discount * execution.discounted_rehandles
        )
        delivery_deviations.extend(execution.delivery_deviations)
        audit["relocations"] += execution.raw_rehandles
        audit["illegal_drops"] += execution.illegal_drops
        audit["macro_failures"] += int(not execution.option_success)
        audit["truncated_macros"] += int(execution.truncated)
        if decision.candidate.mode.value == "recover":
            if decision.exact_rank_progress:
                audit["exact_rank_reducing_recovery_selections"] += 1
            else:
                audit["nonprogress_or_nonexact_recovery_selections"] += 1

        macro_outcomes.append(
            {
                "macro_index": macro_count - 1,
                "decision_epoch": int(snapshot.decision_epoch),
                "candidate_key": execution.candidate_key,
                "mode": execution.mode,
                "action_type": execution.action_type,
                "duration": execution.duration,
                "discounted_operational_return": (
                    execution.discounted_operational_return
                ),
                "raw_operational_return": execution.raw_operational_return,
                "discounted_rehandles": execution.discounted_rehandles,
                "raw_rehandles": execution.raw_rehandles,
                "behavior_lambda": preference_lambda,
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

        time_limit = bool(
            steps >= max_steps
            and not env.is_state_terminal(env.current_state)
        )
        terminal_boundary = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or time_limit
            or execution.duration == 0
        )

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
            except (
                StrictDecisionBoundaryError,
                RuntimeError,
                ValueError,
            ) as error:
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

        if execution.duration > 0:
            try:
                agent.remember(
                    decision,
                    operational_return=(
                        execution.discounted_operational_return
                    ),
                    discounted_rehandles=execution.discounted_rehandles,
                    raw_rehandles=execution.raw_rehandles,
                    duration=execution.duration,
                    next_snapshot=(
                        None if terminal_boundary else pending_snapshot
                    ),
                    done=terminal_boundary,
                    outcome_already_observed=outcome_observed,
                )
                for _ in range(updates_per_macro):
                    value = _loss_value(agent.learn())
                    if value is not None:
                        losses.append(value)
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
    timing = summarize_delivery_timing(
        delivery_deviations, OBJECTIVE_SPEC.window
    )
    return {
        "protocol": TRAINING_PROTOCOL,
        "split": "development_training",
        "instance_seed": int(instance_seed),
        "episode_instance_id": instance.instance_id,
        "behavior_lambda": preference_lambda,
        "preference_schedule": dict(preference_schedule_record),
        "return": float(total_return),
        "episode_start_discounted_return": float(
            episode_discounted_return
        ),
        "physical_rehandles": int(raw_rehandles),
        "episode_start_discounted_rehandles": float(
            episode_discounted_rehandles
        ),
        "sum_intra_macro_discounted_rehandles": float(
            sum_intra_macro_discounted_rehandles
        ),
        "handling_return_contract": HANDLING_RETURN_CONTRACT,
        "success": success,
        "horizon_cap_hit": bool(steps >= max_steps and not terminal),
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
        **timing,
        "macro_outcomes": tuple(macro_outcomes),
        "audit": {
            **dict(audit),
            "certificate_cache_entries": len(certificate_cache),
            "exact_verifier_authoritative": True,
            "unsafe_unknown_fail_closed": True,
            "baseline_viability_teacher": False,
            "baseline_policy_query": False,
            "future_schedule_visible_to_policy": False,
            "certificate_scope": CERTIFICATE_SCOPE,
            "complete_episode_certificate": False,
            "no_fallback_contract": NO_FALLBACK_CONTRACT,
            "finite_horizon_terminal_boundary": True,
        },
        "wall_seconds": perf_counter() - started,
    }


def build_balanced_preference_schedule(
    *,
    episodes: int,
    model_seed: int,
) -> tuple[dict, ...]:
    """Create paired block-balanced preference draws for one model seed."""

    if isinstance(episodes, bool) or int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    if model_seed not in ALLOWED_MODEL_SEEDS:
        raise ValueError(f"model_seed must be one of {ALLOWED_MODEL_SEEDS!r}")
    rng_seed = PREFERENCE_SCHEDULE_SEED_ORIGIN + int(model_seed)
    rng = random.Random(rng_seed)
    records = []
    block_size = len(PREFERENCE_GRID)
    for block_start in range(0, int(episodes), block_size):
        permutation = list(PREFERENCE_GRID)
        rng.shuffle(permutation)
        block_index = block_start // block_size
        for offset, value in enumerate(permutation):
            episode_index = block_start + offset
            if episode_index >= int(episodes):
                break
            records.append(
                {
                    "episode_number": episode_index + 1,
                    "preference_block_number": block_index + 1,
                    "position_in_preference_block": offset + 1,
                    "behavior_lambda": float(value),
                    "block_permutation": tuple(float(x) for x in permutation),
                    "schedule_rng_seed": rng_seed,
                    "sampling_without_replacement_within_block": True,
                    "fixed_for_complete_episode": True,
                }
            )
    counts = Counter(item["behavior_lambda"] for item in records)
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("preference schedule is not block balanced")
    if tuple(item["episode_number"] for item in records) != tuple(
        range(1, int(episodes) + 1)
    ):
        raise RuntimeError("preference schedule episode clock is inconsistent")
    return tuple(records)


def summarize_training_runs(runs: Sequence[Mapping]) -> dict:
    runs = tuple(runs)

    def finite_metric(items, name):
        return [
            float(item[name])
            for item in items
            if item.get(name) is not None
            and math.isfinite(float(item[name]))
        ]

    def summarize_group(items):
        items = tuple(items)
        return {
            "episodes": len(items),
            "mean_return": _mean(item["return"] for item in items),
            "return_std": _std(item["return"] for item in items),
            "mean_absolute_error": _mean(
                finite_metric(items, "mean_absolute_error")
            ),
            "strict_method_success_rate": _mean(
                float(item["strict_method_success"]) for item in items
            ),
            "completion_rate": _mean(
                float(item["success"]) for item in items
            ),
            "raw_physical_rehandles": int(
                sum(int(item["physical_rehandles"]) for item in items)
            ),
            "mean_raw_physical_rehandles": _mean(
                item["physical_rehandles"] for item in items
            ),
            "mean_episode_start_discounted_rehandles": _mean(
                item["episode_start_discounted_rehandles"]
                for item in items
            ),
            "total_deliveries": int(
                sum(int(item["delivery_count"]) for item in items)
            ),
            "physical_rehandles_per_100_deliveries": (
                100.0
                * sum(int(item["physical_rehandles"]) for item in items)
                / sum(int(item["delivery_count"]) for item in items)
                if sum(int(item["delivery_count"]) for item in items) > 0
                else math.nan
            ),
            "method_failures": tuple(
                {
                    "episode_number": item.get("episode_number"),
                    "instance_seed": item["instance_seed"],
                    "reason": item["method_failure_reason"],
                }
                for item in items
                if item.get("method_failure_reason") is not None
            ),
        }

    by_lambda = {}
    for value in PREFERENCE_GRID:
        selected = [
            item
            for item in runs
            if float(item["behavior_lambda"]) == float(value)
        ]
        by_lambda[str(value)] = summarize_group(selected)
    return {
        **summarize_group(runs),
        "preference_counts": {
            str(value): sum(
                float(item["behavior_lambda"]) == float(value)
                for item in runs
            )
            for value in PREFERENCE_GRID
        },
        "by_behavior_lambda": by_lambda,
    }


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_hash(contract: Mapping) -> str:
    encoded = json.dumps(
        _json_safe(dict(contract)),
        sort_keys=True,
        separators=(",", ":"),
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


def _restore_global_rng_state(state: Mapping) -> None:
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


def _new_env(args):
    return ObjectiveAuditSmallRoomsEnv(
        timing_objective=OBJECTIVE_SPEC,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        number_blocks=args.number_blocks,
        choose_storage=False,
        arrival_rate=args.arrival_rate,
        proc_mean=args.proc_mean,
    )


def _build_search_config(args) -> ViabilitySearchConfig:
    return ViabilitySearchConfig(
        max_depth=args.search_max_depth,
        max_nodes=args.search_max_nodes,
        max_primitive_steps=args.search_max_primitive_steps,
        reserve_queue_cells=args.reserve_queue_cells,
        search_order=args.search_order,
    )


def _build_liveness_rule(args) -> BoundedEventDeferRule:
    return BoundedEventDeferRule(
        max_option_steps=args.max_defer_steps,
        max_consecutive_defer_decisions=args.max_consecutive_defers,
    )


def _build_agent_config(args) -> PreferenceConditionedVectorConfig:
    return PreferenceConditionedVectorConfig(
        graph_hidden_dim=args.graph_hidden_dim,
        graph_embedding_dim=args.graph_embedding_dim,
        message_passing_steps=args.message_passing_steps,
        action_embedding_dim=args.action_embedding_dim,
        candidate_hidden_dim=args.head_hidden_dim,
        consequence_hidden_dim=args.head_hidden_dim,
        tau_accept=args.tau_accept,
        tau_recover=args.tau_recover,
        tau_defer=args.tau_defer,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        lambda_max=max(PREFERENCE_GRID),
        condition_on_preference=args.architecture == "conditioned",
        preference_relabels=args.preference_relabels,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_size,
        update_every=args.update_every,
        target_update_every=args.target_update_every,
        grad_clip=args.grad_clip,
        huber_delta=args.huber_delta,
        handling_loss_weight=args.handling_loss_weight,
        max_nonprogress_recovery_decisions=(
            args.max_nonprogress_recovery_decisions
        ),
        force_recovery_witness_when_due=(
            args.force_recovery_witness_when_due
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train conditioned (B) or masked-unconditioned (C) vector VCG"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=ARCHITECTURES, required=True
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        choices=ALLOWED_MODEL_SEEDS,
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--stop-after-episode", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)

    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--proc-mean", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=2_000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-macro", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--handling-loss-weight", type=float, default=1.0)
    parser.add_argument("--preference-relabels", type=int, default=1)
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


def _apply_smoke_recipe(args) -> None:
    """Install a fast deterministic integration recipe, not a result run."""

    if not args.smoke:
        return
    args.episodes = len(PREFERENCE_GRID)
    args.grid_rows = 5
    args.grid_cols = 5
    args.number_blocks = 1
    args.arrival_rate = 1.5
    args.proc_mean = 5
    args.max_steps = 100
    args.batch_size = 2
    args.replay_size = 64
    args.target_update_every = 20
    args.epsilon_decay_decisions = 100
    args.graph_hidden_dim = 8
    args.graph_embedding_dim = 8
    args.message_passing_steps = 1
    args.action_embedding_dim = 4
    args.head_hidden_dim = 16
    args.log_every = 1


def _validate_args(args) -> int:
    positive_ints = (
        "episodes",
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
        "preference_relabels",
        "epsilon_decay_decisions",
        "graph_hidden_dim",
        "graph_embedding_dim",
        "action_embedding_dim",
        "head_hidden_dim",
        "search_max_nodes",
        "max_defer_steps",
        "max_consecutive_defers",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.episodes >= TRAIN_SEED_STRIDE:
        raise ValueError("episode count exceeds paired model-seed namespace")
    if args.replay_size < args.batch_size:
        raise ValueError("--replay-size must be at least --batch-size")
    if args.message_passing_steps < 0:
        raise ValueError("--message-passing-steps must be non-negative")
    if args.max_nonprogress_recovery_decisions < 0:
        raise ValueError("nonprogress recovery bound must be non-negative")
    if args.epsilon_warmup_decisions < 0:
        raise ValueError("epsilon warmup must be non-negative")
    if args.stop_after_episode is not None and not (
        1 <= args.stop_after_episode <= args.episodes
    ):
        raise ValueError("--stop-after-episode must lie in the episode budget")
    if not math.isfinite(args.arrival_rate) or args.arrival_rate < 0.0:
        raise ValueError("--arrival-rate must be finite and non-negative")
    if not 0.0 <= float(args.gamma) < 1.0:
        raise ValueError("--gamma must be in [0, 1)")
    for name in (
        "learning_rate",
        "reward_scale",
        "grad_clip",
        "huber_delta",
        "handling_loss_weight",
        "tau_accept",
        "tau_recover",
        "tau_defer",
    ):
        if not math.isfinite(float(getattr(args, name))) or float(
            getattr(args, name)
        ) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("epsilon_start", "epsilon_end"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    train_seed_base = TRAIN_SEED_ORIGIN + args.model_seed * TRAIN_SEED_STRIDE
    train_seeds = set(range(train_seed_base, train_seed_base + args.episodes))
    protected_overlap = train_seeds.intersection(
        PROTECTED_DEVELOPMENT_AND_TEST_SEEDS
    )
    if protected_overlap:
        raise ValueError(
            "training runner refuses protected 79k/85k--89k panel seeds: "
            f"{tuple(sorted(protected_overlap))}"
        )
    return int(train_seed_base)


def _resume_contract(
    args,
    *,
    train_seed_base: int,
    agent_config: PreferenceConditionedVectorConfig,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    preference_schedule: Sequence[Mapping],
) -> dict:
    semantic = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "development_only": True,
        "architecture": args.architecture,
        "architecture_screen_arm": ARCHITECTURE_ARM_LABELS[
            args.architecture
        ],
        "condition_on_preference": args.architecture == "conditioned",
        "model_seed": int(args.model_seed),
        "model_and_agent_rng_seed": int(args.model_seed),
        "architecture_flag_used_in_rng_derivation": False,
        "episodes": int(args.episodes),
        "train_instance_seed_base": int(train_seed_base),
        "paired_instance_namespace_across_architectures": True,
        "environment": {
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "number_blocks": int(args.number_blocks),
            "arrival_rate": float(args.arrival_rate),
            "proc_mean": int(args.proc_mean),
            "max_steps": int(args.max_steps),
        },
        "objective": OBJECTIVE_SPEC.to_dict(),
        "agent_config": agent_config.to_dict(),
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "updates_per_macro": int(args.updates_per_macro),
        "epsilon_schedule": {
            "start": float(args.epsilon_start),
            "end": float(args.epsilon_end),
            "warmup_decisions": int(args.epsilon_warmup_decisions),
            "decay_decisions": int(args.epsilon_decay_decisions),
        },
        "preference_grid": PREFERENCE_GRID,
        "preference_schedule_seed": (
            PREFERENCE_SCHEDULE_SEED_ORIGIN + int(args.model_seed)
        ),
        "preference_schedule_protocol": (
            "seeded_shuffle_without_replacement_per_five_episode_block_v1"
        ),
        "preference_fixed_for_complete_episode": True,
        "preference_schedule": tuple(dict(item) for item in preference_schedule),
        "common_gamma_for_operational_and_handling_components": float(
            args.gamma
        ),
        "operational_return_contract": OPERATIONAL_RETURN_CONTRACT,
        "handling_return_contract": HANDLING_RETURN_CONTRACT,
        "handling_event_definition": HANDLING_EVENT_DEFINITION,
        "raw_rehandles_stored_separately": True,
        "exact_verifier_authoritative": True,
        "unsafe_unknown_fail_closed": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "certificate_scope": CERTIFICATE_SCOPE,
        "no_fallback_contract": NO_FALLBACK_CONTRACT,
        "interval_validation": False,
        "checkpoint_selection": False,
        "terminal_checkpoint_rule": "fixed_final_episode_only_v1",
        "resumable_checkpoint_every_episodes": CHECKPOINT_EVERY_EPISODES,
        "evaluation_panels_opened": False,
        "protected_79k_and_85k_through_89k_panels_opened": False,
        "smoke_recipe": bool(args.smoke),
    }
    return {**semantic, "contract_sha256": _contract_hash(semantic)}


def _build_checkpoint(
    agent: PreferenceConditionedVectorAgent,
    *,
    include_replay: bool,
    resumable: bool,
    role: str,
    completed_episodes: int,
    contract: Mapping,
    training_history: Sequence[Mapping],
) -> dict:
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": str(role),
        "trainer_resumable": bool(resumable),
        "completed_training_episodes": int(completed_episodes),
        "next_training_episode": int(completed_episodes) + 1,
        "resume_contract": dict(contract),
        "resume_contract_sha256": contract["contract_sha256"],
        "agent_checkpoint": agent.checkpoint(include_replay=include_replay),
        "training_history": tuple(training_history),
        "global_rng_state": (
            _capture_global_rng_state() if resumable else None
        ),
        "fixed_terminal_checkpoint": role == TERMINAL_CHECKPOINT_ROLE,
        "development_only": True,
        "evaluation_panels_opened": False,
    }


def _validate_resume_payload(payload: Mapping, contract: Mapping) -> None:
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": LATEST_CHECKPOINT_ROLE,
        "trainer_resumable": True,
        "development_only": True,
        "evaluation_panels_opened": False,
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"incompatible resume checkpoint: {mismatches!r}")
    if payload.get("resume_contract") != contract:
        raise ValueError("resume contract, architecture, or recipe changed")
    if payload.get("resume_contract_sha256") != contract["contract_sha256"]:
        raise ValueError("resume contract SHA-256 mismatch")
    if not isinstance(payload.get("agent_checkpoint"), Mapping):
        raise ValueError("resume checkpoint is missing its agent checkpoint")
    if payload.get("global_rng_state") is None:
        raise ValueError("resume checkpoint is missing global RNG state")
    completed = int(payload.get("completed_training_episodes", -1))
    history = tuple(payload.get("training_history", ()))
    if completed < 0 or len(history) != completed:
        raise ValueError("resume episode clock/history is inconsistent")
    if payload.get("next_training_episode") != completed + 1:
        raise ValueError("resume next-episode clock is inconsistent")


def _validate_terminal_payload(payload: Mapping, contract: Mapping) -> None:
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": TERMINAL_CHECKPOINT_ROLE,
        "trainer_resumable": False,
        "fixed_terminal_checkpoint": True,
        "development_only": True,
        "evaluation_panels_opened": False,
        "completed_training_episodes": int(contract["episodes"]),
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"incompatible terminal checkpoint: {mismatches!r}")
    if payload.get("resume_contract") != contract:
        raise ValueError("terminal checkpoint contract changed")
    agent_checkpoint = payload.get("agent_checkpoint")
    if not isinstance(agent_checkpoint, Mapping):
        raise ValueError("terminal checkpoint is missing its agent model")
    if "replay" in agent_checkpoint.get("agent_state", {}):
        raise ValueError("fixed terminal checkpoint must be replay-free")


def _result_payload(
    args,
    *,
    completed_episodes: int,
    train_seed_base: int,
    contract: Mapping,
    training_history: Sequence[Mapping],
    agent: PreferenceConditionedVectorAgent,
    terminal_sha256: Optional[str],
) -> dict:
    complete = int(completed_episodes) == int(args.episodes)
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "status": "complete" if complete else "paused",
        "development_only": True,
        "architecture": args.architecture,
        "architecture_screen_arm": ARCHITECTURE_ARM_LABELS[
            args.architecture
        ],
        "condition_on_preference": args.architecture == "conditioned",
        "model_seed": int(args.model_seed),
        "completed_training_episodes": int(completed_episodes),
        "total_training_episodes": int(args.episodes),
        "train_instance_seed_base": int(train_seed_base),
        "last_completed_train_instance_seed": (
            int(train_seed_base + completed_episodes - 1)
            if completed_episodes
            else None
        ),
        "preference_grid": PREFERENCE_GRID,
        "preference_schedule_seed": (
            PREFERENCE_SCHEDULE_SEED_ORIGIN + int(args.model_seed)
        ),
        "resume_contract": dict(contract),
        "training": {
            "runs": tuple(training_history),
            "summary": summarize_training_runs(training_history),
        },
        "agent_audit": agent.audit(include_decisions=False),
        "latest_checkpoint": str(
            (args.output_dir / "latest.pth").resolve()
        ),
        "terminal_checkpoint": (
            str((args.output_dir / "terminal.pth").resolve())
            if complete
            else None
        ),
        "terminal_checkpoint_sha256": terminal_sha256,
        "fixed_terminal_checkpoint": complete,
        "checkpoint_selection_used": False,
        "interval_validation_used": False,
        "evaluation_panels_opened": False,
        "smoke_recipe": bool(args.smoke),
    }


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    _apply_smoke_recipe(args)
    train_seed_base = _validate_args(args)
    device = resolve_device(args.device)
    agent_config = _build_agent_config(args)
    search_config = _build_search_config(args)
    liveness_rule = _build_liveness_rule(args)
    preference_schedule = build_balanced_preference_schedule(
        episodes=args.episodes,
        model_seed=args.model_seed,
    )
    contract = _resume_contract(
        args,
        train_seed_base=train_seed_base,
        agent_config=agent_config,
        search_config=search_config,
        liveness_rule=liveness_rule,
        preference_schedule=preference_schedule,
    )

    latest_path = args.output_dir / "latest.pth"
    terminal_path = args.output_dir / "terminal.pth"
    summary_path = args.output_dir / "training-summary.json"
    if not args.resume_existing:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                "fresh training refuses a nonempty output directory; pass "
                "--resume-existing or choose a new directory"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(args.model_seed)
        agent = PreferenceConditionedVectorAgent(
            config=agent_config,
            seed=args.model_seed,
            device=device,
            epsilon=args.epsilon_start,
        )
        completed = 0
        training_history: list[dict] = []
    else:
        if not latest_path.is_file():
            raise FileNotFoundError(
                "--resume-existing requires output-dir/latest.pth"
            )
        payload = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise ValueError("latest.pth must contain a mapping")
        _validate_resume_payload(payload, contract)
        agent = PreferenceConditionedVectorAgent.from_checkpoint(
            payload["agent_checkpoint"],
            device=device,
            resumable=True,
            seed=args.model_seed,
        )
        _optimizer_to(agent.optimizer, device)
        completed = int(payload["completed_training_episodes"])
        training_history = list(payload["training_history"])
        if completed > args.episodes:
            raise ValueError("resume checkpoint exceeds planned episodes")
        _restore_global_rng_state(payload["global_rng_state"])
        print(
            f"Resumed {latest_path} at completed episode {completed}",
            flush=True,
        )

    print(
        "Preference-conditioned vector VCG | "
        f"architecture={args.architecture} | device={device} | "
        f"model_seed={args.model_seed} | episodes={args.episodes} | "
        f"train_seed_base={train_seed_base} | gamma={args.gamma:.4f} | "
        f"lambda_grid={PREFERENCE_GRID} | smoke={str(args.smoke).lower()} | "
        "exact_safe=true | teacher=false | interval_validation=false",
        flush=True,
    )
    env = _new_env(args)
    stop_at = int(args.episodes)
    if args.stop_after_episode is not None:
        stop_at = min(stop_at, int(args.stop_after_episode))
    if completed > stop_at:
        raise ValueError("resume is already beyond --stop-after-episode")

    for episode_index in range(completed, stop_at):
        episode_number = episode_index + 1
        schedule_record = preference_schedule[episode_index]
        preference_lambda = float(schedule_record["behavior_lambda"])
        instance_seed = train_seed_base + episode_index
        env.current_episode = episode_number
        run = run_preference_episode(
            agent,
            env,
            instance_seed=instance_seed,
            preference_lambda=preference_lambda,
            preference_schedule_record=schedule_record,
            max_steps=args.max_steps,
            search_config=search_config,
            liveness_rule=liveness_rule,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_decisions=args.epsilon_decay_decisions,
            epsilon_warmup_decisions=args.epsilon_warmup_decisions,
            updates_per_macro=args.updates_per_macro,
        )
        run.update(
            {
                "episode_number": episode_number,
                "architecture": args.architecture,
                "architecture_screen_arm": ARCHITECTURE_ARM_LABELS[
                    args.architecture
                ],
                "condition_on_preference": (
                    args.architecture == "conditioned"
                ),
            }
        )
        training_history.append(run)
        completed = episode_number
        if hasattr(agent, "decision_log"):
            agent.decision_log.clear()

        if episode_number == 1 or episode_number % args.log_every == 0:
            recent = training_history[-min(10, len(training_history)) :]
            print(
                f"Ep {episode_number:4d} | "
                f"Lambda {preference_lambda:6.3f} | "
                f"TrainR {fmean(item['return'] for item in recent):8.2f} | "
                f"Strict {int(run['strict_method_success'])} | "
                f"Reh {run['physical_rehandles']:3d} | "
                f"DReh {run['episode_start_discounted_rehandles']:7.3f} | "
                f"Dec {run['macro_decisions']:3d} | "
                f"Replay {len(agent.replay):5d} | "
                f"Grad {agent.gradient_steps:6d} | "
                f"Eps {agent.epsilon:.3f}",
                flush=True,
            )

        checkpoint_due = bool(
            episode_number % CHECKPOINT_EVERY_EPISODES == 0
            or episode_number == stop_at
            or episode_number == args.episodes
        )
        if checkpoint_due:
            latest = _build_checkpoint(
                agent,
                include_replay=True,
                resumable=True,
                role=LATEST_CHECKPOINT_ROLE,
                completed_episodes=completed,
                contract=contract,
                training_history=training_history,
            )
            _atomic_torch_save(latest, latest_path)
            interim = _result_payload(
                args,
                completed_episodes=completed,
                train_seed_base=train_seed_base,
                contract=contract,
                training_history=training_history,
                agent=agent,
                terminal_sha256=(
                    _file_sha256(terminal_path)
                    if terminal_path.is_file()
                    else None
                ),
            )
            _atomic_json_save(interim, summary_path)

    if completed == 0:
        raise RuntimeError("no training episode was completed")

    terminal_sha256 = None
    if completed == args.episodes:
        if terminal_path.is_file():
            terminal = torch.load(
                terminal_path, map_location="cpu", weights_only=False
            )
            if not isinstance(terminal, Mapping):
                raise ValueError("terminal.pth must contain a mapping")
            _validate_terminal_payload(terminal, contract)
        else:
            terminal = _build_checkpoint(
                agent,
                include_replay=False,
                resumable=False,
                role=TERMINAL_CHECKPOINT_ROLE,
                completed_episodes=completed,
                contract=contract,
                training_history=training_history,
            )
            _atomic_torch_save(terminal, terminal_path)
        terminal_sha256 = _file_sha256(terminal_path)

    result = _result_payload(
        args,
        completed_episodes=completed,
        train_seed_base=train_seed_base,
        contract=contract,
        training_history=training_history,
        agent=agent,
        terminal_sha256=terminal_sha256,
    )
    _atomic_json_save(result, summary_path)
    print(
        json.dumps(
            _json_safe(
                {
                    "status": result["status"],
                    "architecture": args.architecture,
                    "model_seed": args.model_seed,
                    "completed_training_episodes": completed,
                    "latest_checkpoint": result["latest_checkpoint"],
                    "terminal_checkpoint": result["terminal_checkpoint"],
                    "terminal_checkpoint_sha256": terminal_sha256,
                    "summary": str(summary_path.resolve()),
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
