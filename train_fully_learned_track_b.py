#!/usr/bin/env python3
"""Train the versioned fully learned reserved-macro Track-B hierarchy.

This entry point is intentionally separate from ``train_track_b.py`` and
``train_relational_track_b.py``.  Its replay records span complete, explicitly
parameterised macros and its four online phases are part of the checkpoint
contract:

``imitation``
    Execute the duration-aware + REG-v5 teacher and train hierarchical behavior
    cloning at every macro decision.
``temporal``
    Learn the common SMDP objective while exposing only the teacher's accepted
    storage cell.
``spatial``
    Expand to every feasible cell and distil the frozen REG ranking without TD
    updates, bridging the candidate-set distribution shift.
``joint``
    Learn over every feasible ``AcceptStore(block, cell)`` candidate while
    annealing training-only teacher retention.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import (
    build_controller_options,
    scheduler_episode_audit,
)
from example.Options.selector_v5 import ExplicitCellAssignmentRegistry
from example.episode_instance import EpisodeInstance
from example.helper.occupancy_pressure import (
    OccupancyPressureTracker,
    summarize_occupancy_pressure_runs,
)
from example.helper.timing_metrics import (
    summarize_block_storage_flow,
    summarize_delivery_timing,
    summarize_storage_flow_runs,
)
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.reg_selector_v5 import REGV5AssignmentSource
from fully_learned_hierarchy import (
    FULLY_LEARNED_ACTION_INTERFACE,
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    TRAINING_PHASES,
    FullyLearnedConfig,
    FullyLearnedHierarchyAgent,
    FullyLearnedInfeasible,
    validate_fully_learned_checkpoint_metadata,
)


FULLY_LEARNED_TRAINER_VERSION = 4
METHOD = "fully_learned_reserved_macro_hierarchy_v4_1"
RAW_MACRO_RETURN_CONTRACT = "raw_environment_sum_gamma_i_once_v1"
TRUNCATION_CONTRACT = (
    "time_limit_terminal_training_boundary_penalty_no_bootstrap_v2"
)
FAILURE_CONTRACT = "strict_terminal_no_fallback_v1"
FAILURE_PENALTY_CONTRACT = (
    "separate_gamma_k_failure_or_horizon_boundary_penalty_v2"
)
CURRICULUM_CONTRACT = (
    "spatial_calibration_replay_warmup_joint_retention_quality_gated_v4_1"
)
MODEL_SELECTION_CONTRACT = (
    "hard_safety_warm_performance_gated_joint_lexicographic_v3"
)
QUALITY_GATE_CONTRACT = (
    "immediate_safety_warm_return_mae_patience_relative_v3"
)
TEMPORAL_WARM_START_CONTRACT = "authenticated_v3_temporal_weights_only_v1"
SPATIAL_WARM_START_CONTRACT = "authenticated_v4_spatial_weights_only_v1"
HISTORY_CONTRACT = "complete_episode_and_validation_history_v1"
PHASE_ORDER = ("imitation", "temporal", "spatial", "joint")

V3_TEMPORAL_WARM_START_METADATA = {
    "fully_learned_checkpoint_schema_version": 3,
    "fully_learned_trainer_version": 2,
    "controller_architecture": "relational_parameterized_mode_regularized_smdp_v3",
    "network_architecture": "shared_deepset_candidate_q_v2",
    "replay_version": "variable_candidate_mode_balanced_common_continuation_smdp_v3",
    "backup_version": "nested_logmeanexp_raw_q_common_smdp_v2",
    "curriculum_contract": (
        "phase_local_exploration_teacher_mix_retained_bc_mode_balanced_v3"
    ),
    "checkpoint_kind": "resumable_phase_end",
    "training_phase": "temporal",
}

V4_SPATIAL_WARM_START_METADATA = {
    "fully_learned_checkpoint_schema_version": 4,
    "fully_learned_trainer_version": 3,
    "controller_architecture": "relational_parameterized_mode_regularized_smdp_v4",
    "network_architecture": "shared_deepset_candidate_q_v2",
    "replay_version": "variable_candidate_mode_balanced_common_continuation_smdp_v4",
    "backup_version": "nested_logmeanexp_raw_q_common_smdp_v2",
    "curriculum_contract": (
        "spatial_calibration_joint_retention_quality_gated_v4"
    ),
    "quality_gate_contract": (
        "deployment_mode_return_mae_strict_reservation_relative_v2"
    ),
    "checkpoint_kind": "resumable_phase_end",
    "training_phase": "spatial",
}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    saved_cuda = state.get("cuda")
    if saved_cuda is not None and torch.cuda.is_available():
        for index, rng_state in enumerate(
            saved_cuda[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(rng_state, device=index)


def optimizer_to(optimizer, device: torch.device) -> None:
    """Move restored optimizer tensors to the model device."""

    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace one checkpoint only after a complete same-directory write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def phase_episode_counts(args) -> dict[str, int]:
    return {
        "imitation": int(args.imitation_episodes),
        "temporal": int(args.temporal_episodes),
        "spatial": int(args.spatial_episodes),
        "joint": int(args.joint_episodes),
    }


def total_training_episodes(args) -> int:
    return sum(phase_episode_counts(args).values())


def phase_for_episode(episode: int, counts: dict[str, int]) -> str:
    if episode <= 0:
        raise ValueError("episode must be positive")
    boundary = 0
    for phase in PHASE_ORDER:
        boundary += counts[phase]
        if episode <= boundary:
            return phase
    raise ValueError(f"episode {episode} exceeds configured phase schedule")


def linear_decision_schedule(
    decision_count: int,
    *,
    start: float,
    end: float,
    warmup: int,
    decay: int,
) -> float:
    """Piecewise-linear schedule on macro decisions, never primitive steps."""

    decision_count = int(decision_count)
    if decision_count <= int(warmup):
        return float(start)
    if decision_count >= int(warmup) + int(decay):
        return float(end)
    fraction = min(
        1.0,
        (decision_count - int(warmup)) / int(decay),
    )
    return float(start + fraction * (end - start))


def scheduled_epsilon(decision_count: int, args, phase: str = "temporal") -> float:
    """Phase-local epsilon; imitation is always teacher forced."""

    if phase in {"imitation", "spatial"}:
        return 0.0
    return linear_decision_schedule(
        decision_count,
        start=getattr(args, f"{phase}_epsilon_start"),
        end=getattr(args, f"{phase}_epsilon_end"),
        warmup=getattr(args, f"{phase}_epsilon_warmup_decisions"),
        decay=getattr(args, f"{phase}_epsilon_decay_decisions"),
    )


def scheduled_teacher_mixture(decision_count: int, args, phase: str) -> float:
    if phase not in {"temporal", "joint"}:
        return 0.0
    return linear_decision_schedule(
        decision_count,
        start=getattr(args, f"{phase}_teacher_mixture_start"),
        end=getattr(args, f"{phase}_teacher_mixture_end"),
        warmup=getattr(args, f"{phase}_teacher_mixture_warmup_decisions"),
        decay=getattr(args, f"{phase}_teacher_mixture_decay_decisions"),
    )


def scheduled_bc_weight(decision_count: int, args, phase: str) -> float:
    if phase in {"imitation", "spatial"}:
        return 1.0
    if phase not in {"temporal", "joint"}:
        return 0.0
    return linear_decision_schedule(
        decision_count,
        start=getattr(args, f"{phase}_bc_start"),
        end=getattr(args, f"{phase}_bc_end"),
        warmup=getattr(args, f"{phase}_bc_warmup_decisions"),
        decay=getattr(args, f"{phase}_bc_decay_decisions"),
    )


def td_learning_enabled(
    phase: str,
    phase_decisions: int,
    *,
    joint_warmup_decisions: int,
) -> bool:
    """Keep the first joint replay records free of narrow-buffer TD updates."""

    return bool(
        phase != "joint"
        or int(phase_decisions) > int(joint_warmup_decisions)
    )


def _failed_macro_outcome(option):
    outcome = getattr(option, "last_outcome", None)
    return (
        outcome
        if isinstance(outcome, dict) and outcome.get("success") is False
        else None
    )


def reservation_integrity(option_audit: dict, selector_audit: dict) -> bool:
    """Verify the exact proposal -> commit -> execution join for one episode."""

    bound = list(option_audit.get("bound_proposal_ids", ()))
    committed = list(option_audit.get("committed_proposal_ids", ()))
    selector_ids = [
        decision.get("proposal_id")
        for decision in selector_audit.get("decisions", ())
        if decision.get("valid")
        and decision.get("commitment_contract")
        == "decision_epoch_proposal_bound_once_v2"
    ]
    successes = int(option_audit.get("inbound_successes", 0))
    return bool(
        option_audit.get("accept_store_option_version")
        == "accept_store_reserved_cell_v2"
        and option_audit.get("reservation_invalidation_count", 0) == 0
        and option_audit.get("reservation_bound_count", 0) == successes
        and option_audit.get("reservation_commit_count", 0) == successes
        and option_audit.get("reservation_execution_match_count", 0)
        == successes
        and len(bound) == len(set(bound))
        and bound == committed
        and committed == selector_ids
    )


def selector_snapshot(selector) -> dict:
    return {
        "decision_count": int(getattr(selector, "decision_count", 0)),
        "infeasible_epoch_count": int(
            getattr(selector, "infeasible_epoch_count", 0)
        ),
        "invalid_assignment_count": int(
            getattr(selector, "invalid_assignment_count", 0)
        ),
        "assignment_seconds": float(getattr(selector, "assignment_seconds", 0.0)),
        "assignment_total_seconds": float(
            getattr(selector, "assignment_total_seconds", 0.0)
        ),
        "decisions_length": len(getattr(selector, "decisions", ())),
    }


def selector_delta(selector, start: dict) -> dict:
    audit = selector.audit()
    decisions = list(getattr(selector, "decisions", ()))[
        start["decisions_length"] :
    ]
    return {
        **audit,
        "decision_count": int(getattr(selector, "decision_count", 0))
        - start["decision_count"],
        "infeasible_epoch_count": int(
            getattr(selector, "infeasible_epoch_count", 0)
        )
        - start["infeasible_epoch_count"],
        "invalid_assignment_count": int(
            getattr(selector, "invalid_assignment_count", 0)
        )
        - start["invalid_assignment_count"],
        "assignment_seconds": float(getattr(selector, "assignment_seconds", 0.0))
        - start["assignment_seconds"],
        "assignment_total_seconds": float(
            getattr(selector, "assignment_total_seconds", 0.0)
        )
        - start["assignment_total_seconds"],
        "decisions": decisions,
    }


def summarize_runs(runs: list[dict]) -> dict:
    returns = np.asarray([item["return"] for item in runs], dtype=float)
    errors = np.asarray(
        [item.get("mean_absolute_error", np.nan) for item in runs], dtype=float
    )
    finite_errors = errors[np.isfinite(errors)]
    controls = Counter()
    modes = Counter()
    for run in runs:
        audit = run.get("fully_learned_audit", {})
        controls.update(audit.get("control_decisions", {}))
        modes.update(audit.get("mode_decisions", {}))
    return {
        "episodes": len(runs),
        "mean_return": float(returns.mean()) if len(returns) else float("nan"),
        "return_std": (
            float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        ),
        "success_rate": float(np.mean([item["success"] for item in runs])),
        "strict_method_success_rate": float(
            np.mean([item["strict_method_success"] for item in runs])
        ),
        "reservation_integrity_rate": float(
            np.mean([item["reservation_integrity"] for item in runs])
        ),
        "selector_independent_replay_isolation_rate": float(
            np.mean(
                [item["selector_independent_replay_isolated"] for item in runs]
            )
        ),
        "mean_steps": float(np.mean([item["steps"] for item in runs])),
        "mean_delivery_count": float(
            np.mean([item["delivery_count"] for item in runs])
        ),
        "mean_absolute_error": (
            float(finite_errors.mean()) if len(finite_errors) else float("nan")
        ),
        "total_invalid_assignments": int(
            sum(
                item["selector_audit"].get("invalid_assignment_count", 0)
                for item in runs
            )
        ),
        "total_fallbacks": int(
            sum(
                item["selector_audit"].get("fallback_count", 0)
                for item in runs
            )
        ),
        "total_reservation_invalidations": int(
            sum(
                item["scheduler_audit"].get(
                    "reservation_invalidation_count", 0
                )
                for item in runs
            )
        ),
        "method_failures": [
            {
                "eval_seed": item.get("eval_seed"),
                "reason": item["method_failure_reason"],
            }
            for item in runs
            if item.get("method_failure_reason") is not None
        ],
        "controller_control_decisions": dict(controls),
        "controller_mode_decisions": dict(modes),
        **summarize_storage_flow_runs(runs),
        **summarize_occupancy_pressure_runs(runs),
    }


def make_env(args):
    return make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=args.exit_width,
        number_blocks=args.number_blocks,
    )


def _config_from_args(args) -> FullyLearnedConfig:
    values = {
        "block_embedding_dim": args.block_embedding_dim,
        "global_embedding_dim": args.global_embedding_dim,
        "candidate_embedding_dim": args.candidate_embedding_dim,
        "context_dim": args.context_dim,
        "history_length": args.history_length,
        "tau_accept": args.tau_accept,
        "tau_retrieve": args.tau_retrieve,
        "tau_defer": args.tau_defer,
        "tau_mode": args.tau_mode,
        "teacher_coefficient": args.teacher_coefficient,
        "teacher_score_temperature": args.teacher_score_temperature,
        "spatial_distillation_weight": args.spatial_distillation_weight,
        "joint_accept_exploration_top_k": (
            args.joint_accept_exploration_top_k
        ),
        "huber_delta": args.huber_delta,
        "failure_penalty": args.failure_penalty,
        "lookahead_margin_steps": args.lookahead_margin_steps,
    }
    allowed = getattr(FullyLearnedConfig, "__dataclass_fields__", {})
    return FullyLearnedConfig(
        **{key: value for key, value in values.items() if key in allowed}
    )


def _config_from_payload(payload: dict) -> FullyLearnedConfig:
    values = payload.get("fully_learned_config")
    if not isinstance(values, dict):
        raise ValueError("checkpoint has no fully learned config")
    factory = getattr(FullyLearnedConfig, "from_dict", None)
    if callable(factory):
        return factory(values)
    allowed = getattr(FullyLearnedConfig, "__dataclass_fields__", {})
    return FullyLearnedConfig(
        **{key: value for key, value in values.items() if key in allowed}
    )


def _build_stack(
    args,
    selector_payload: dict,
    *,
    device: torch.device,
    config: FullyLearnedConfig | None = None,
):
    """Construct one independent env/registry/agent stack."""

    seed_everything(args.seed)
    env = make_env(args)
    # Network construction must not make controller initialization depend on
    # the temporary frozen-teacher objects.
    state = capture_rng_state()
    source = REGV5AssignmentSource.from_checkpoint(
        env,
        selector_payload,
        learning_enabled=False,
        device=device,
        seed=args.seed,
    )
    selector = ExplicitCellAssignmentRegistry(env, source)
    restore_rng_state(state)
    build_controller_options(
        env,
        selector,
        controller_action_interface=FULLY_LEARNED_ACTION_INTERFACE,
        max_defer_steps=args.max_defer_steps,
    )
    initial = env.sample_episode_instance(
        args.training_instance_seed_base + args.seed * 1_000_000
    )
    env.reset(instance=initial)
    encoder = OnlineManifestTimingObservationEncoder(env)
    config = config or _config_from_args(args)
    # The agent owns a copy of the deployment/EMA spatial teacher. The registry
    # remains frozen and exists only for proposal provenance and execution.
    spatial_network = deepcopy(source.network)
    agent = FullyLearnedHierarchyAgent(
        env,
        encoder,
        config=config,
        spatial_network=spatial_network,
        seed=args.seed,
        device=device,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        spatial_learning_rate=args.spatial_learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_every=args.update_every,
        target_tau=args.target_tau,
        grad_clip=args.grad_clip,
        reward_scale=args.reward_scale,
        epsilon=args.temporal_epsilon_start,
        replay_sampling=args.replay_sampling,
    )
    return env, selector, agent


def _agent_checkpoint_state(agent, *, include_replay: bool) -> dict:
    callback = getattr(agent, "checkpoint_state", None)
    if callback is None:
        raise RuntimeError("fully learned agent has no checkpoint_state API")
    try:
        return callback(include_replay=include_replay)
    except TypeError:
        state = callback()
        if not include_replay:
            state = dict(state)
            state.pop("replay", None)
            state.pop("replay_memory", None)
        return state


def _load_agent_checkpoint_state(
    agent, state: dict, *, resumable: bool
) -> None:
    callback = getattr(agent, "load_checkpoint_state", None)
    if callback is None:
        raise RuntimeError("fully learned agent has no load_checkpoint_state API")
    try:
        callback(state, resumable=resumable)
    except TypeError:
        callback(state)
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None:
        optimizer_to(optimizer, agent.device)


def _completed_phase_counts(episode: int, counts: dict[str, int]) -> dict:
    remaining = int(episode)
    result = {}
    for phase in PHASE_ORDER:
        result[phase] = min(counts[phase], max(0, remaining))
        remaining -= result[phase]
    return result


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_reference(validation: dict, *, source: str) -> dict:
    required = (
        "mean_return",
        "mean_absolute_error",
        "strict_method_success_rate",
        "reservation_integrity_rate",
    )
    missing = [key for key in required if key not in validation]
    if missing:
        raise ValueError(f"validation reference lacks fields: {missing}")
    result = {key: float(validation[key]) for key in required}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("validation reference must be finite")
    result["source"] = str(source)
    return result


def validation_quality_gate(
    validation: dict,
    reference: dict,
    *,
    min_return_ratio: float,
    max_mae_ratio: float,
) -> dict:
    """Compare deployment-mode validation to one immutable phase baseline."""

    observed_return = float(validation.get("mean_return", float("nan")))
    observed_mae = float(validation.get("mean_absolute_error", float("nan")))
    reference_return = float(reference["mean_return"])
    reference_mae = float(reference["mean_absolute_error"])
    return_threshold = (
        reference_return * float(min_return_ratio)
        if reference_return >= 0.0
        else reference_return / float(min_return_ratio)
    )
    # A zero-error reference must remain exact rather than making every
    # positive error fail through an accidental zero ratio threshold.
    mae_threshold = max(1.0e-6, reference_mae * float(max_mae_ratio))
    finite = bool(np.isfinite(observed_return) and np.isfinite(observed_mae))
    return_ok = bool(finite and observed_return >= return_threshold)
    mae_ok = bool(finite and observed_mae <= mae_threshold)
    strict_reference = reference.get("strict_method_success_rate")
    reservation_reference = reference.get("reservation_integrity_rate")
    observed_strict = float(
        validation.get("strict_method_success_rate", float("nan"))
    )
    observed_reservation = float(
        validation.get("reservation_integrity_rate", float("nan"))
    )
    strict_ok = bool(
        strict_reference is None
        or (
            np.isfinite(observed_strict)
            and observed_strict >= float(strict_reference)
        )
    )
    reservation_ok = bool(
        reservation_reference is None
        or (
            np.isfinite(observed_reservation)
            and observed_reservation >= float(reservation_reference)
        )
    )
    return {
        "passed": bool(return_ok and mae_ok and strict_ok and reservation_ok),
        "return_ok": return_ok,
        "mae_ok": mae_ok,
        "strict_ok": strict_ok,
        "reservation_ok": reservation_ok,
        "observed_return": observed_return,
        "observed_mae": observed_mae,
        "reference_return": reference_return,
        "reference_mae": reference_mae,
        "observed_strict_method_success_rate": observed_strict,
        "reference_strict_method_success_rate": strict_reference,
        "observed_reservation_integrity_rate": observed_reservation,
        "reference_reservation_integrity_rate": reservation_reference,
        "return_threshold": return_threshold,
        "mae_threshold": mae_threshold,
        "contract": QUALITY_GATE_CONTRACT,
    }


def joint_gate_decision(
    gate: dict,
    *,
    phase_decisions: int,
    warmup_decisions: int,
    prior_bad_validations: int,
    collapse_patience: int,
) -> dict:
    """Apply hard safety immediately and patience only to mature performance.

    Return/MAE observations remain visible during replay warmup, but they cannot
    terminate training until the decision-clocked gate warmup has completed.
    Strict method success and reservation integrity are never covered by that
    grace period.
    """

    safety_failed = bool(
        not gate.get("strict_ok", False)
        or not gate.get("reservation_ok", False)
    )
    performance_failed = bool(
        not gate.get("return_ok", False) or not gate.get("mae_ok", False)
    )
    performance_enforced = bool(
        int(phase_decisions) >= int(warmup_decisions)
    )
    if safety_failed:
        bad_validations = int(prior_bad_validations)
    elif not performance_enforced:
        bad_validations = 0
    elif performance_failed:
        bad_validations = int(prior_bad_validations) + 1
    else:
        bad_validations = 0
    performance_blocked = bool(
        performance_enforced
        and bad_validations >= int(collapse_patience)
    )
    return {
        "safety_failed": safety_failed,
        "performance_failed": performance_failed,
        "performance_enforced": performance_enforced,
        "phase_decisions": int(phase_decisions),
        "warmup_decisions": int(warmup_decisions),
        "bad_validations": bad_validations,
        "collapse_patience": int(collapse_patience),
        "blocked": bool(safety_failed or performance_blocked),
        "block_reason": (
            "safety"
            if safety_failed
            else "performance"
            if performance_blocked
            else None
        ),
    }


def _configure_phase_optimization(agent, args, phase: str) -> None:
    if not callable(getattr(agent, "set_learning_rate_scales", None)):
        return
    if phase == "joint":
        agent.set_learning_rate_scales(
            main=args.joint_main_lr_scale,
            spatial=args.joint_spatial_lr_scale,
        )
    else:
        agent.set_learning_rate_scales(main=1.0, spatial=1.0)


def checkpoint_payload(
    agent,
    selector_payload: dict,
    args,
    *,
    episode: int,
    phase: str,
    validation: dict,
    best_score,
    exploration_state: dict,
    checkpoint_kind: str,
    history=None,
    validations=None,
) -> dict:
    resumable = checkpoint_kind in {
        "resumable_latest",
        "resumable_phase_end",
    }
    counts = phase_episode_counts(args)
    completed = _completed_phase_counts(episode, counts)
    payload = {
        "fully_learned_checkpoint_schema_version": (
            FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
        ),
        "fully_learned_trainer_version": FULLY_LEARNED_TRAINER_VERSION,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "checkpoint_kind": checkpoint_kind,
        "resumable": resumable,
        **agent.checkpoint_metadata(),
        "agent_checkpoint_state": _agent_checkpoint_state(
            agent, include_replay=resumable
        ),
        "fully_learned_config": (
            agent.config.to_dict()
            if callable(getattr(agent.config, "to_dict", None))
            else {
                key: getattr(agent.config, key)
                for key in agent.config.__dataclass_fields__
            }
        ),
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "selector_frozen_independent_replay_disabled": True,
        "selector_checkpoint": selector_payload,
        "selector_deployment_digest": selector_deployment_digest(
            selector_payload
        ),
        "selector_warm_start_state": "deployment_ema_copy_v1",
        "training_phase": phase,
        "phase_episode_counts": counts,
        "completed_imitation_episodes": completed["imitation"],
        "completed_temporal_episodes": completed["temporal"],
        "completed_spatial_episodes": completed["spatial"],
        "completed_joint_episodes": completed["joint"],
        "completed_training_episodes": int(episode),
        "training_seed": args.seed,
        "python_hash_seed": args.python_hash_seed,
        "training_instance_seed_base": args.training_instance_seed_base,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "geometry": geometry_metadata(
            agent.env, requested_exit_width=args.exit_width
        ),
        "max_steps": args.max_steps,
        "max_defer_steps": args.max_defer_steps,
        "lookahead_margin_steps": args.lookahead_margin_steps,
        "learning_rate": args.learning_rate,
        "spatial_learning_rate": args.spatial_learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "update_every": args.update_every,
        "updates_per_macro": args.updates_per_macro,
        "joint_updates_per_macro": args.joint_updates_per_macro,
        "joint_td_warmup_decisions": args.joint_td_warmup_decisions,
        "gamma": args.gamma,
        "target_tau": args.target_tau,
        "grad_clip": args.grad_clip,
        "reward_scale": args.reward_scale,
        "failure_penalty": args.failure_penalty,
        "replay_sampling": args.replay_sampling,
        "temporal_epsilon_start": args.temporal_epsilon_start,
        "temporal_epsilon_end": args.temporal_epsilon_end,
        "temporal_epsilon_warmup_decisions": (
            args.temporal_epsilon_warmup_decisions
        ),
        "temporal_epsilon_decay_decisions": (
            args.temporal_epsilon_decay_decisions
        ),
        "joint_epsilon_start": args.joint_epsilon_start,
        "joint_epsilon_end": args.joint_epsilon_end,
        "joint_epsilon_warmup_decisions": args.joint_epsilon_warmup_decisions,
        "joint_epsilon_decay_decisions": args.joint_epsilon_decay_decisions,
        "temporal_teacher_mixture_start": (
            args.temporal_teacher_mixture_start
        ),
        "temporal_teacher_mixture_end": args.temporal_teacher_mixture_end,
        "temporal_teacher_mixture_warmup_decisions": (
            args.temporal_teacher_mixture_warmup_decisions
        ),
        "temporal_teacher_mixture_decay_decisions": (
            args.temporal_teacher_mixture_decay_decisions
        ),
        "joint_teacher_mixture_start": args.joint_teacher_mixture_start,
        "joint_teacher_mixture_end": args.joint_teacher_mixture_end,
        "joint_teacher_mixture_warmup_decisions": (
            args.joint_teacher_mixture_warmup_decisions
        ),
        "joint_teacher_mixture_decay_decisions": (
            args.joint_teacher_mixture_decay_decisions
        ),
        "temporal_bc_start": args.temporal_bc_start,
        "temporal_bc_end": args.temporal_bc_end,
        "temporal_bc_warmup_decisions": args.temporal_bc_warmup_decisions,
        "temporal_bc_decay_decisions": args.temporal_bc_decay_decisions,
        "joint_bc_start": args.joint_bc_start,
        "joint_bc_end": args.joint_bc_end,
        "joint_bc_warmup_decisions": args.joint_bc_warmup_decisions,
        "joint_bc_decay_decisions": args.joint_bc_decay_decisions,
        "joint_main_lr_scale": args.joint_main_lr_scale,
        "joint_spatial_lr_scale": args.joint_spatial_lr_scale,
        "joint_spatial_freeze_decisions": args.joint_spatial_freeze_decisions,
        "reset_optimizer_on_phase_transition": (
            args.reset_optimizer_on_phase_transition
        ),
        "temporal_min_validation_strict": args.temporal_min_validation_strict,
        "temporal_collapse_patience": args.temporal_collapse_patience,
        "spatial_min_return_ratio": args.spatial_min_return_ratio,
        "spatial_max_mae_ratio": args.spatial_max_mae_ratio,
        "joint_min_return_ratio": args.joint_min_return_ratio,
        "joint_max_mae_ratio": args.joint_max_mae_ratio,
        "joint_gate_warmup_decisions": args.joint_gate_warmup_decisions,
        "joint_collapse_patience": args.joint_collapse_patience,
        "quality_gate_contract": QUALITY_GATE_CONTRACT,
        "epsilon_clock_contract": (
            "independent_all_macro_decisions_per_phase_v2"
        ),
        "macro_decisions": int(exploration_state["decisions"]),
        "phase_macro_decisions": dict(exploration_state["phase_decisions"]),
        "temporal_bad_validations": int(
            exploration_state.get("temporal_bad_validations", 0)
        ),
        "joint_bad_validations": int(
            exploration_state.get("joint_bad_validations", 0)
        ),
        "quality_gate_blocked": bool(
            exploration_state.get("quality_gate_blocked", False)
        ),
        "temporal_reference_validation": exploration_state.get(
            "temporal_reference_validation"
        ),
        "spatial_reference_validation": exploration_state.get(
            "spatial_reference_validation"
        ),
        "warm_start_provenance": exploration_state.get(
            "warm_start_provenance"
        ),
        "epsilon_macro_decisions": int(
            sum(exploration_state["phase_decisions"].values())
        ),
        "epsilon": float(agent.epsilon),
        "validation_seeds": tuple(args.validation_seeds),
        "validation_steps": args.validation_steps,
        "eval_every": args.eval_every,
        "target_window": args.target_window,
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
        "validation": validation,
        "best_score": tuple(best_score),
    }
    if resumable:
        payload["rng_state"] = capture_rng_state()
        payload["training_history"] = list(history or ())
        payload["validation_history"] = list(validations or ())
        payload["history_contract"] = HISTORY_CONTRACT
    return payload


def _validate_resume_payload(payload: dict, args, selector_payload: dict) -> None:
    if payload.get("fully_learned_checkpoint_schema_version") != (
        FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("resume checkpoint uses an incompatible schema")
    if payload.get("fully_learned_trainer_version") != FULLY_LEARNED_TRAINER_VERSION:
        raise ValueError("resume checkpoint uses an incompatible trainer version")
    if payload.get("checkpoint_kind") not in {
        "resumable_latest",
        "resumable_phase_end",
    } or not payload.get("resumable"):
        raise ValueError("--resume requires a resumable fully learned checkpoint")
    if payload.get("quality_gate_blocked"):
        raise ValueError(
            "resume checkpoint is a gate-failed diagnostic and cannot advance "
            "the curriculum"
        )
    validate_fully_learned_checkpoint_metadata(payload)
    requested_config = _config_from_args(args).to_dict()
    expected = {
        "training_seed": args.seed,
        "python_hash_seed": args.python_hash_seed,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "phase_episode_counts": phase_episode_counts(args),
        "training_instance_seed_base": args.training_instance_seed_base,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "fully_learned_config": requested_config,
        "max_steps": args.max_steps,
        "max_defer_steps": args.max_defer_steps,
        "learning_rate": args.learning_rate,
        "spatial_learning_rate": args.spatial_learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "update_every": args.update_every,
        "updates_per_macro": args.updates_per_macro,
        "joint_updates_per_macro": args.joint_updates_per_macro,
        "joint_td_warmup_decisions": args.joint_td_warmup_decisions,
        "gamma": args.gamma,
        "target_tau": args.target_tau,
        "grad_clip": args.grad_clip,
        "reward_scale": args.reward_scale,
        "failure_penalty": args.failure_penalty,
        "replay_sampling": args.replay_sampling,
        "temporal_epsilon_start": args.temporal_epsilon_start,
        "temporal_epsilon_end": args.temporal_epsilon_end,
        "temporal_epsilon_warmup_decisions": (
            args.temporal_epsilon_warmup_decisions
        ),
        "temporal_epsilon_decay_decisions": (
            args.temporal_epsilon_decay_decisions
        ),
        "joint_epsilon_start": args.joint_epsilon_start,
        "joint_epsilon_end": args.joint_epsilon_end,
        "joint_epsilon_warmup_decisions": args.joint_epsilon_warmup_decisions,
        "joint_epsilon_decay_decisions": args.joint_epsilon_decay_decisions,
        "temporal_teacher_mixture_start": (
            args.temporal_teacher_mixture_start
        ),
        "temporal_teacher_mixture_end": args.temporal_teacher_mixture_end,
        "temporal_teacher_mixture_warmup_decisions": (
            args.temporal_teacher_mixture_warmup_decisions
        ),
        "temporal_teacher_mixture_decay_decisions": (
            args.temporal_teacher_mixture_decay_decisions
        ),
        "joint_teacher_mixture_start": args.joint_teacher_mixture_start,
        "joint_teacher_mixture_end": args.joint_teacher_mixture_end,
        "joint_teacher_mixture_warmup_decisions": (
            args.joint_teacher_mixture_warmup_decisions
        ),
        "joint_teacher_mixture_decay_decisions": (
            args.joint_teacher_mixture_decay_decisions
        ),
        "temporal_bc_start": args.temporal_bc_start,
        "temporal_bc_end": args.temporal_bc_end,
        "temporal_bc_warmup_decisions": args.temporal_bc_warmup_decisions,
        "temporal_bc_decay_decisions": args.temporal_bc_decay_decisions,
        "joint_bc_start": args.joint_bc_start,
        "joint_bc_end": args.joint_bc_end,
        "joint_bc_warmup_decisions": args.joint_bc_warmup_decisions,
        "joint_bc_decay_decisions": args.joint_bc_decay_decisions,
        "joint_main_lr_scale": args.joint_main_lr_scale,
        "joint_spatial_lr_scale": args.joint_spatial_lr_scale,
        "joint_spatial_freeze_decisions": args.joint_spatial_freeze_decisions,
        "reset_optimizer_on_phase_transition": (
            args.reset_optimizer_on_phase_transition
        ),
        "temporal_min_validation_strict": args.temporal_min_validation_strict,
        "temporal_collapse_patience": args.temporal_collapse_patience,
        "spatial_min_return_ratio": args.spatial_min_return_ratio,
        "spatial_max_mae_ratio": args.spatial_max_mae_ratio,
        "joint_min_return_ratio": args.joint_min_return_ratio,
        "joint_max_mae_ratio": args.joint_max_mae_ratio,
        "joint_gate_warmup_decisions": args.joint_gate_warmup_decisions,
        "joint_collapse_patience": args.joint_collapse_patience,
        "quality_gate_contract": QUALITY_GATE_CONTRACT,
        "validation_seeds": tuple(args.validation_seeds),
        "validation_steps": args.validation_steps,
        "eval_every": args.eval_every,
        "target_window": args.target_window,
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
        "history_contract": HISTORY_CONTRACT,
    }
    mismatches = {
        key: {"saved": payload.get(key), "requested": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if selector_deployment_digest(selector_payload) != payload.get(
        "selector_deployment_digest"
    ):
        mismatches["selector_deployment_digest"] = "mismatch"
    saved_geometry = payload.get("geometry", {})
    requested_geometry = {
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "requested_exit_width": args.exit_width,
        "block_count": args.number_blocks,
    }
    geometry_mismatches = {
        key: {"saved": saved_geometry.get(key), "requested": value}
        for key, value in requested_geometry.items()
        if saved_geometry.get(key, 40 if key == "block_count" else None)
        != value
    }
    if geometry_mismatches:
        mismatches["geometry"] = geometry_mismatches
    saved_history = payload.get("training_history")
    saved_validations = payload.get("validation_history")
    completed = int(payload.get("completed_training_episodes", -1))
    if not 1 <= completed <= total_training_episodes(args):
        mismatches["completed_training_episodes"] = completed
    else:
        expected_phase = phase_for_episode(completed, phase_episode_counts(args))
        if payload.get("training_phase") != expected_phase:
            mismatches["training_phase"] = {
                "saved": payload.get("training_phase"),
                "expected": expected_phase,
            }
        agent_phase = payload.get("agent_checkpoint_state", {}).get(
            "training_phase"
        )
        if agent_phase != expected_phase:
            mismatches["agent_training_phase"] = {
                "saved": agent_phase,
                "expected": expected_phase,
            }
        completed_counts = _completed_phase_counts(
            completed, phase_episode_counts(args)
        )
        for phase, value in completed_counts.items():
            key = f"completed_{phase}_episodes"
            if payload.get(key) != value:
                mismatches[key] = {
                    "saved": payload.get(key),
                    "expected": value,
                }
    phase_clocks = payload.get("phase_macro_decisions")
    if (
        not isinstance(phase_clocks, dict)
        or set(phase_clocks) != set(PHASE_ORDER)
        or any(
            not isinstance(value, int) or value < 0
            for value in phase_clocks.values()
        )
        or sum(phase_clocks.values()) != int(payload.get("macro_decisions", -1))
    ):
        mismatches["phase_macro_decisions"] = phase_clocks
    for key in ("temporal_reference_validation", "spatial_reference_validation"):
        value = payload.get(key)
        if value is not None and not isinstance(value, dict):
            mismatches[key] = value
    if not isinstance(payload.get("joint_bad_validations", 0), int):
        mismatches["joint_bad_validations"] = payload.get(
            "joint_bad_validations"
        )
    if not isinstance(saved_history, list) or len(saved_history) != completed:
        mismatches["training_history"] = {
            "saved_length": (
                len(saved_history) if isinstance(saved_history, list) else None
            ),
            "completed_training_episodes": completed,
        }
    if not isinstance(saved_validations, list) or not saved_validations:
        mismatches["validation_history"] = "missing_or_empty"
    else:
        latest_validation = payload.get("validation")
        if not isinstance(latest_validation, dict) or any(
            saved_validations[-1].get(key) != latest_validation.get(key)
            for key in ("episode", "phase")
        ):
            mismatches["validation_history"] = "latest_validation_mismatch"
    if mismatches:
        raise ValueError(f"resume contract mismatch: {mismatches!r}")


def _validate_v3_temporal_warm_start(
    payload: dict,
    args,
    selector_payload: dict,
    source_path: Path,
) -> tuple[dict, dict]:
    """Authenticate one legacy v3 temporal-end artifact for weights-only import."""

    mismatches = {
        key: {"expected": expected, "found": payload.get(key)}
        for key, expected in V3_TEMPORAL_WARM_START_METADATA.items()
        if payload.get(key) != expected
    }
    if not payload.get("resumable"):
        mismatches["resumable"] = payload.get("resumable")
    if int(payload.get("completed_temporal_episodes", 0)) <= 0:
        mismatches["completed_temporal_episodes"] = payload.get(
            "completed_temporal_episodes"
        )
    if int(payload.get("completed_joint_episodes", 0)) != 0:
        mismatches["completed_joint_episodes"] = payload.get(
            "completed_joint_episodes"
        )
    expected = {
        "training_seed": args.seed,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "gamma": args.gamma,
        "reward_scale": args.reward_scale,
        "validation_seeds": tuple(args.validation_seeds),
        "validation_steps": args.validation_steps,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            mismatches[key] = {"expected": value, "found": payload.get(key)}
    if selector_deployment_digest(selector_payload) != payload.get(
        "selector_deployment_digest"
    ):
        mismatches["selector_deployment_digest"] = "mismatch"
    saved_geometry = payload.get("geometry", {})
    requested_geometry = {
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "requested_exit_width": args.exit_width,
        "block_count": args.number_blocks,
    }
    geometry_mismatches = {
        key: {"expected": value, "found": saved_geometry.get(key)}
        for key, value in requested_geometry.items()
        if saved_geometry.get(key, 40 if key == "block_count" else None)
        != value
    }
    if geometry_mismatches:
        mismatches["geometry"] = geometry_mismatches
    source_config = payload.get("fully_learned_config")
    requested_config = _config_from_args(args).to_dict()
    if not isinstance(source_config, dict):
        mismatches["fully_learned_config"] = "missing"
    else:
        config_mismatches = {
            key: {"expected": requested_config[key], "found": value}
            for key, value in source_config.items()
            if key in requested_config and requested_config[key] != value
        }
        if config_mismatches:
            mismatches["fully_learned_config"] = config_mismatches
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        mismatches["validation"] = "missing"
    elif float(validation.get("strict_method_success_rate", -1.0)) < (
        args.temporal_min_validation_strict
    ):
        mismatches["validation"] = {
            "strict_method_success_rate": validation.get(
                "strict_method_success_rate"
            ),
            "required": args.temporal_min_validation_strict,
        }
    agent_state = payload.get("agent_checkpoint_state")
    if not isinstance(agent_state, dict) or "Q_local" not in agent_state:
        mismatches["agent_checkpoint_state"] = "missing Q_local"
    if mismatches:
        raise ValueError(f"v3 temporal warm-start mismatch: {mismatches!r}")
    reference = _validation_reference(
        validation,
        source="v3_temporal_end",
    )
    provenance = {
        "contract": TEMPORAL_WARM_START_CONTRACT,
        "source_path": str(source_path.resolve()),
        "source_sha256": _checkpoint_sha256(source_path),
        "source_schema_version": payload["fully_learned_checkpoint_schema_version"],
        "source_trainer_version": payload["fully_learned_trainer_version"],
        "source_curriculum_contract": payload["curriculum_contract"],
        "source_completed_training_episodes": int(
            payload["completed_training_episodes"]
        ),
        "source_selector_deployment_digest": payload[
            "selector_deployment_digest"
        ],
    }
    return reference, provenance


def _validate_v4_spatial_warm_start(
    payload: dict,
    args,
    selector_payload: dict,
    source_path: Path,
) -> tuple[dict, dict, dict]:
    """Authenticate the immutable v4 spatial endpoint for weights-only import."""

    mismatches = {
        key: {"expected": expected, "found": payload.get(key)}
        for key, expected in V4_SPATIAL_WARM_START_METADATA.items()
        if payload.get(key) != expected
    }
    if not payload.get("resumable"):
        mismatches["resumable"] = payload.get("resumable")
    if payload.get("quality_gate_blocked"):
        mismatches["quality_gate_blocked"] = True
    if int(payload.get("completed_spatial_episodes", 0)) <= 0:
        mismatches["completed_spatial_episodes"] = payload.get(
            "completed_spatial_episodes"
        )
    if int(payload.get("completed_joint_episodes", 0)) != 0:
        mismatches["completed_joint_episodes"] = payload.get(
            "completed_joint_episodes"
        )
    expected = {
        "training_seed": args.seed,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "max_steps": args.max_steps,
        "max_defer_steps": args.max_defer_steps,
        "gamma": args.gamma,
        "reward_scale": args.reward_scale,
        "failure_penalty": args.failure_penalty,
        "validation_seeds": tuple(args.validation_seeds),
        "validation_steps": args.validation_steps,
        "target_window": args.target_window,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            mismatches[key] = {"expected": value, "found": payload.get(key)}
    if selector_deployment_digest(selector_payload) != payload.get(
        "selector_deployment_digest"
    ):
        mismatches["selector_deployment_digest"] = "mismatch"
    saved_geometry = payload.get("geometry", {})
    requested_geometry = {
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "requested_exit_width": args.exit_width,
        "block_count": args.number_blocks,
    }
    geometry_mismatches = {
        key: {"expected": value, "found": saved_geometry.get(key)}
        for key, value in requested_geometry.items()
        if saved_geometry.get(key, 40 if key == "block_count" else None)
        != value
    }
    if geometry_mismatches:
        mismatches["geometry"] = geometry_mismatches
    source_config = payload.get("fully_learned_config")
    requested_config = _config_from_args(args).to_dict()
    if not isinstance(source_config, dict):
        mismatches["fully_learned_config"] = "missing"
    else:
        config_mismatches = {
            key: {"expected": requested_config[key], "found": value}
            for key, value in source_config.items()
            if key in requested_config and requested_config[key] != value
        }
        if config_mismatches:
            mismatches["fully_learned_config"] = config_mismatches
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        mismatches["validation"] = "missing"
    else:
        gate = validation.get("quality_gate")
        if not isinstance(gate, dict) or not gate.get("passed"):
            mismatches["validation_quality_gate"] = gate
        if validation.get("phase") != "spatial":
            mismatches["validation_phase"] = validation.get("phase")
        if float(validation.get("strict_method_success_rate", -1.0)) < 1.0:
            mismatches["validation_strict_method_success_rate"] = validation.get(
                "strict_method_success_rate"
            )
        if float(validation.get("reservation_integrity_rate", -1.0)) < 1.0:
            mismatches["validation_reservation_integrity_rate"] = validation.get(
                "reservation_integrity_rate"
            )
    temporal_reference = payload.get("temporal_reference_validation")
    saved_spatial_reference = payload.get("spatial_reference_validation")
    if not isinstance(temporal_reference, dict):
        mismatches["temporal_reference_validation"] = temporal_reference
    if not isinstance(saved_spatial_reference, dict):
        mismatches["spatial_reference_validation"] = saved_spatial_reference
    elif isinstance(validation, dict):
        for key in (
            "mean_return",
            "mean_absolute_error",
            "strict_method_success_rate",
            "reservation_integrity_rate",
        ):
            if float(saved_spatial_reference.get(key, float("nan"))) != float(
                validation.get(key, float("nan"))
            ):
                mismatches.setdefault("spatial_reference_validation", {})[key] = {
                    "reference": saved_spatial_reference.get(key),
                    "validation": validation.get(key),
                }
    agent_state = payload.get("agent_checkpoint_state")
    if not isinstance(agent_state, dict) or "Q_local" not in agent_state:
        mismatches["agent_checkpoint_state"] = "missing Q_local"
    elif agent_state.get("training_phase") != "spatial":
        mismatches["agent_training_phase"] = agent_state.get("training_phase")
    if mismatches:
        raise ValueError(f"v4 spatial warm-start mismatch: {mismatches!r}")
    spatial_reference = _validation_reference(
        validation,
        source="v4_spatial_end",
    )
    provenance = {
        "contract": SPATIAL_WARM_START_CONTRACT,
        "source_path": str(source_path.resolve()),
        "source_sha256": _checkpoint_sha256(source_path),
        "source_schema_version": payload["fully_learned_checkpoint_schema_version"],
        "source_trainer_version": payload["fully_learned_trainer_version"],
        "source_curriculum_contract": payload["curriculum_contract"],
        "source_completed_training_episodes": int(
            payload["completed_training_episodes"]
        ),
        "source_selector_deployment_digest": payload[
            "selector_deployment_digest"
        ],
    }
    return dict(temporal_reference), spatial_reference, provenance


def _import_weights_only(agent, payload: dict) -> None:
    source = payload["agent_checkpoint_state"]["Q_local"]
    agent.Q_local.load_state_dict(source, strict=True)
    agent.Q_target.load_state_dict(agent.Q_local.state_dict(), strict=True)
    agent.optimizer.state.clear()
    agent.replay.clear()
    agent.step_count = 0
    agent.decision_count = 0
    agent.gradient_steps = 0
    agent.imitation_steps = 0
    agent.retention_steps = 0
    agent.spatial_distillation_steps = 0


def _import_v3_temporal_weights(agent, payload: dict) -> None:
    _import_weights_only(agent, payload)


def _import_v4_spatial_weights(agent, payload: dict) -> None:
    _import_weights_only(agent, payload)


def _validate_resume_best_artifact(payload: dict, output_dir: Path):
    """Ensure resume cannot silently lose or substitute deployment-best weights."""

    best_path = output_dir / "best.pth"
    joint_started = int(payload.get("completed_joint_episodes", 0)) > 0
    saved_best_score = tuple(
        payload.get("best_score", (-1.0, -float("inf")))
    )
    passing_joint_best_exists = bool(
        len(saved_best_score) == 2 and float(saved_best_score[0]) >= 0.0
    )
    if not best_path.is_file():
        if passing_joint_best_exists:
            raise FileNotFoundError(
                "resume metadata records a passing joint checkpoint, so the "
                "existing deployment best.pth is required"
            )
        return None
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(best, dict):
        raise ValueError("deployment best checkpoint must be a mapping")
    validate_fully_learned_checkpoint_metadata(best)
    expected = {
        "fully_learned_checkpoint_schema_version": (
            FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
        ),
        "fully_learned_trainer_version": FULLY_LEARNED_TRAINER_VERSION,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "checkpoint_kind": "deployment_best",
        "resumable": False,
        "training_phase": "joint",
        "training_seed": payload["training_seed"],
        "python_hash_seed": payload["python_hash_seed"],
        "training_lambda": payload["training_lambda"],
        "training_mu": payload["training_mu"],
        "phase_episode_counts": payload["phase_episode_counts"],
        "selector_deployment_digest": payload["selector_deployment_digest"],
        "geometry": payload["geometry"],
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
    }
    mismatches = {
        key: {"best": best.get(key), "latest": value}
        for key, value in expected.items()
        if best.get(key) != value
    }
    best_episode = int(best.get("completed_training_episodes", -1))
    latest_episode = int(payload["completed_training_episodes"])
    if not joint_started and best_episode <= latest_episode:
        mismatches["completed_training_episodes"] = (
            "pre_joint_latest_has_nonfuture_best"
        )
    if int(best.get("completed_joint_episodes", 0)) <= 0:
        mismatches["completed_joint_episodes"] = best.get(
            "completed_joint_episodes"
        )
    if best_episode <= latest_episode and tuple(best.get("best_score", ())) != tuple(
        payload["best_score"]
    ):
        mismatches["best_score"] = {
            "best": best.get("best_score"),
            "latest": payload["best_score"],
        }
    if best_episode > latest_episode + int(payload["eval_every"]):
        mismatches["completed_training_episodes"] = (
            "best_is_too_far_ahead_of_latest_commit"
        )
    forbidden = set(best.get("agent_checkpoint_state", {})).intersection(
        {"optimizer", "replay", "rng_state"}
    )
    if forbidden:
        mismatches["deployment_state"] = {
            "unexpected_resumable_fields": sorted(forbidden)
        }
    if mismatches:
        raise ValueError(f"resume deployment-best mismatch: {mismatches!r}")
    return best


def _validate_existing_joint_init(
    path: Path,
    agent,
    args,
    selector_payload: dict,
    *,
    completed_episode: int,
) -> None:
    """Authenticate a boundary artifact left by an interrupted transition."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("joint-init checkpoint must be a mapping")
    validate_fully_learned_checkpoint_metadata(payload)
    expected = {
        "fully_learned_checkpoint_schema_version": (
            FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
        ),
        "fully_learned_trainer_version": FULLY_LEARNED_TRAINER_VERSION,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "checkpoint_kind": "joint_init_baseline",
        "resumable": False,
        "training_phase": "joint",
        "completed_training_episodes": int(completed_episode),
        "training_seed": args.seed,
        "phase_episode_counts": phase_episode_counts(args),
        "selector_deployment_digest": selector_deployment_digest(
            selector_payload
        ),
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    saved_state = payload.get("agent_checkpoint_state", {}).get("Q_local")
    current_state = agent.Q_local.state_dict()
    if not isinstance(saved_state, dict) or set(saved_state) != set(current_state):
        mismatches["Q_local"] = "missing_or_incompatible"
    elif any(
        not torch.equal(saved_state[name].cpu(), value.detach().cpu())
        for name, value in current_state.items()
    ):
        mismatches["Q_local"] = "weights_do_not_match_boundary"
    if mismatches:
        raise ValueError(f"joint-init checkpoint mismatch: {mismatches!r}")


def build_stack_from_payload(args, payload, *, device, for_evaluation=False):
    """Reconstruct the exact versioned stack used by evaluator or resume."""

    validate_fully_learned_checkpoint_metadata(payload)
    selector_payload = payload.get("selector_checkpoint")
    if not isinstance(selector_payload, dict):
        raise ValueError("fully learned checkpoint does not embed its REG teacher")
    digest = selector_deployment_digest(selector_payload)
    if digest != payload.get("selector_deployment_digest"):
        raise ValueError("embedded selector deployment digest mismatch")
    # Evaluation CLI does not carry training-only fields; materialize them from
    # the checkpoint without weakening regime/geometry checks performed by the
    # caller.
    defaults = {
        "seed": int(payload["training_seed"]),
        "training_instance_seed_base": int(
            payload.get("training_instance_seed_base", 600_000)
        ),
        "max_defer_steps": int(payload.get("max_defer_steps", 10)),
        "batch_size": int(payload.get("batch_size", 128)),
        "buffer_size": int(payload.get("buffer_size", 20_000)),
        "update_every": int(payload.get("update_every", 100)),
        "learning_rate": float(payload.get("learning_rate", 5e-5)),
        "spatial_learning_rate": float(
            payload.get("spatial_learning_rate", 5e-6)
        ),
        "gamma": float(payload.get("gamma", 0.99)),
        "target_tau": float(payload.get("target_tau", 1e-3)),
        "grad_clip": float(payload.get("grad_clip", 5.0)),
        "reward_scale": float(payload.get("reward_scale", 0.01)),
        "temporal_epsilon_start": (
            0.0 if for_evaluation else float(payload["temporal_epsilon_start"])
        ),
        "replay_sampling": payload.get("replay_sampling", "mode_balanced"),
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    env, selector, agent = _build_stack(
        args,
        selector_payload,
        device=device,
        config=_config_from_payload(payload),
    )
    _load_agent_checkpoint_state(
        agent,
        payload["agent_checkpoint_state"],
        resumable=not for_evaluation,
    )
    if for_evaluation:
        agent.epsilon = 0.0
        agent.set_training_phase("joint")
        for network_name in ("Q_local", "Q_target"):
            network = getattr(agent, network_name, None)
            if network is not None:
                network.eval()
    return env, selector, agent


def _select_action(
    agent,
    state,
    *,
    epsilon: float,
    teacher_forcing: bool,
    teacher_probability: float = 0.0,
    retain_teacher_label: bool = False,
):
    parameters = inspect.signature(agent.select_action).parameters
    kwargs = {}
    if "eps" in parameters:
        kwargs["eps"] = float(epsilon)
    elif "epsilon" in parameters:
        kwargs["epsilon"] = float(epsilon)
    if "teacher_forcing" in parameters:
        kwargs["teacher_forcing"] = bool(teacher_forcing)
    if "teacher_probability" in parameters:
        kwargs["teacher_probability"] = float(teacher_probability)
    if "retain_teacher_label" in parameters:
        kwargs["retain_teacher_label"] = bool(retain_teacher_label)
    return agent.select_action(state, **kwargs)


def _flush_runtime_failure(agent, state, *, failure_penalty, store_transition):
    callback = getattr(agent, "flush_failure", None)
    if callback is None:
        # A runtime failure before the first environment transition has no valid
        # positive-duration SMDP record. Resetting at the next episode clears the
        # ephemeral candidate; the strict episode still fails explicitly.
        return False
    parameters = inspect.signature(callback).parameters
    kwargs = {}
    if "failure_penalty" in parameters:
        kwargs["failure_penalty"] = float(failure_penalty)
    if "store_transition" in parameters:
        kwargs["store_transition"] = bool(store_transition)
    return bool(callback(state, **kwargs))


def run_episode(
    agent,
    selector,
    env,
    instance: EpisodeInstance,
    *,
    max_steps: int,
    target_window: float,
    training: bool,
    phase: str,
    exploration_state=None,
    epsilon_args=None,
    failure_penalty: float = -50.0,
    updates_per_macro: int = 1,
    joint_td_warmup_decisions: int = 0,
    include_decisions: bool = False,
):
    """Execute one episode and create at most one replay record per macro.

    Environment reward reaches the agent exactly once per environment step.
    ``FullyLearnedHierarchyAgent`` owns the raw discounted accumulator and
    applies ``reward_scale`` only inside its learner.
    """

    if phase not in TRAINING_PHASES:
        raise ValueError(f"unsupported training phase: {phase!r}")
    state = env.reset(instance=instance)
    pressure = OccupancyPressureTracker(env)
    agent.reset_episode()
    if getattr(agent, "training_phase", None) != phase:
        agent.set_training_phase(phase)
    selector_start = selector_snapshot(selector)
    total_return = 0.0
    errors = []
    obstructive_moves = 0
    illegal_drops = 0
    losses = []
    imitation_losses = []
    method_failure_reason = None
    done = False
    steps = 0
    macro_count = 0
    active_macro_steps = 0
    decision_start = (
        int(exploration_state["decisions"])
        if exploration_state is not None
        else 0
    )
    phase_clock_start = (
        int(exploration_state["phase_decisions"].get(phase, 0))
        if exploration_state is not None
        else 0
    )
    teacher_mixture_values = []
    bc_weight_values = []
    started = perf_counter()
    store_td_transition = bool(
        training and phase not in {"imitation", "spatial"}
    )

    def finalize_closed_macro():
        """Advance the macro clock and consume any newly closed replay record."""

        nonlocal macro_count, active_macro_steps
        macro_count += 1
        active_macro_steps = 0
        if not training:
            return
        agent.step_count += 1
        if phase in {"imitation", "spatial"}:
            return
        phase_decisions = (
            int(exploration_state["phase_decisions"].get(phase, 0))
            if exploration_state is not None
            else macro_count
        )
        if not td_learning_enabled(
            phase,
            phase_decisions,
            joint_warmup_decisions=joint_td_warmup_decisions,
        ):
            return
        for _ in range(int(updates_per_macro)):
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

    while steps < max_steps and not done and method_failure_reason is None:
        if agent.current_option is None:
            teacher_forcing = bool(
                training and phase in {"imitation", "spatial"}
            )
            phase_decision = (
                int(exploration_state["phase_decisions"].get(phase, 0))
                if exploration_state is not None
                else 0
            )
            epsilon = (
                scheduled_epsilon(phase_decision, epsilon_args, phase)
                if training and not teacher_forcing
                else 0.0
            )
            teacher_probability = (
                scheduled_teacher_mixture(phase_decision, epsilon_args, phase)
                if training
                else 0.0
            )
            bc_weight = (
                scheduled_bc_weight(phase_decision, epsilon_args, phase)
                if training
                else 0.0
            )
            if training and phase == "joint":
                network = getattr(agent, "Q_local", None)
                callback = getattr(network, "set_spatial_trainable", None)
                if callable(callback):
                    callback(
                        phase_decision
                        >= epsilon_args.joint_spatial_freeze_decisions
                    )
            agent.epsilon = float(epsilon)
            agent.current_teacher_mixture = float(teacher_probability)
            teacher_supervision = bool(
                training
                and (
                    phase == "spatial"
                    or (
                        phase == "joint"
                        and (teacher_probability > 0.0 or bc_weight > 0.0)
                    )
                )
            )
            if callable(getattr(agent, "set_teacher_supervision", None)):
                agent.set_teacher_supervision(teacher_supervision)
            try:
                option = _select_action(
                    agent,
                    state,
                    epsilon=epsilon,
                    teacher_forcing=teacher_forcing,
                    teacher_probability=teacher_probability,
                    retain_teacher_label=bool(
                        training
                        and phase in {"temporal", "joint"}
                        and bc_weight > 0.0
                    ),
                )
            except FullyLearnedInfeasible as exc:
                method_failure_reason = f"scheduler_infeasible:{exc}"
                break
            agent.current_option = option
            decisions = getattr(agent, "decisions", ())
            if decisions:
                pressure.observe_decision(decisions[-1])
            if training:
                exploration_state["decisions"] += 1
                exploration_state["phase_decisions"][phase] = phase_decision + 1
                teacher_mixture_values.append(float(teacher_probability))
                bc_weight_values.append(float(bc_weight))
                if bc_weight > 0.0:
                    imitation_loss = agent.learn_imitation(weight=bc_weight)
                    if imitation_loss is not None:
                        imitation_losses.append(imitation_loss)

        option = agent.current_option
        try:
            action = option.policy(state)
            next_state, reward, done, info = env.step(action)
        except RuntimeError as exc:
            _flush_runtime_failure(
                agent,
                state,
                failure_penalty=failure_penalty,
                store_transition=store_td_transition,
            )
            if active_macro_steps > 0:
                finalize_closed_macro()
            method_failure_reason = (
                f"macro_runtime:{type(option).__name__}:{exc}"
            )
            agent.current_option = None
            break

        steps += 1
        pressure.observe_transition()
        active_macro_steps += 1
        total_return += float(reward)
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        obstructive_moves += int(bool(info.get("relocated_block")))
        illegal_drops += int(bool(info.get("illegal_drop")))
        try:
            macro_complete = bool(option.termination(next_state))
        except RuntimeError as exc:
            # The environment transition already happened, so its reward must
            # enter the macro accumulator exactly once even though lifecycle
            # validation failed afterward.
            agent.process_step(
                next_state,
                float(reward),
                done=True,
                terminated=True,
                failed=True,
                env_terminal=bool(done),
                truncated=False,
                store_transition=store_td_transition,
            )
            finalize_closed_macro()
            state = next_state
            agent.current_option = None
            method_failure_reason = (
                f"macro_runtime:{type(option).__name__}:{exc}"
            )
            break

        failed_outcome = _failed_macro_outcome(option) if macro_complete else None
        macro_failed = failed_outcome is not None
        truncated = bool(steps >= max_steps and not done)
        close_macro = bool(macro_complete or done or truncated or macro_failed)
        replay_terminal = bool(done or truncated or macro_failed)
        agent.process_step(
            next_state,
            float(reward),
            done=replay_terminal,
            terminated=close_macro,
            failed=macro_failed,
            env_terminal=bool(done),
            truncated=truncated,
            store_transition=store_td_transition,
        )
        state = next_state
        if close_macro:
            finalize_closed_macro()
            agent.current_option = None
        if macro_failed:
            method_failure_reason = (
                f"macro_failure:{type(option).__name__}:"
                f"{failed_outcome.get('reason', 'unknown')}"
            )

    elapsed = perf_counter() - started
    success = bool(done and method_failure_reason is None)
    truncated = bool(steps >= max_steps and not done)
    # The wrapped source is frozen, so this lifecycle callback cannot create an
    # independent replay record. Assert that invariant below.
    selector.on_episode_end(success=success, truncated=truncated or not success)
    source_replay = getattr(selector.source, "replay", ())
    source_isolated = bool(
        not selector.source.learning_enabled
        and len(source_replay) == 0
        and getattr(selector.source, "pending", None) is None
    )
    timing = summarize_delivery_timing(errors, target_window)
    storage_flow = summarize_block_storage_flow(env.blocks, env.time_steps)
    selector_audit = selector_delta(selector, selector_start)
    option_audit = scheduler_episode_audit(env)
    exact_reservation = reservation_integrity(option_audit, selector_audit)
    strict_success = bool(
        success
        and method_failure_reason is None
        and selector_audit.get("invalid_assignment_count", 0) == 0
        and selector_audit.get("fallback_count", 0) == 0
        and option_audit.get("inbound_failures", 0) == 0
        and option_audit.get("retrieve_failures", 0) == 0
        and illegal_drops == 0
        and exact_reservation
        and source_isolated
    )
    agent_audit = agent.audit(include_decisions=include_decisions)
    return {
        "return": total_return,
        "success": float(success),
        "strict_method_success": float(strict_success),
        "reservation_integrity": float(exact_reservation),
        "selector_independent_replay_isolated": float(source_isolated),
        "truncated": float(truncated),
        "method_failure_reason": method_failure_reason,
        "phase": phase,
        "steps": steps,
        "macro_count": macro_count,
        "delivery_count": len(errors),
        "delivery_deviations": errors,
        "illegal_drops": illegal_drops,
        "obstructive_moves": obstructive_moves,
        "decision_seconds": elapsed,
        "selector_audit": selector_audit,
        "scheduler_audit": option_audit,
        "fully_learned_audit": agent_audit,
        "controller_decision_count": (
            exploration_state["decisions"] - decision_start
            if exploration_state is not None
            else macro_count
        ),
        "epsilon_decision_count": (
            exploration_state["phase_decisions"].get(phase, 0)
            - phase_clock_start
            if exploration_state is not None
            else 0
        ),
        "phase_macro_decision_count": (
            exploration_state["phase_decisions"].get(phase, 0)
            - phase_clock_start
            if exploration_state is not None
            else 0
        ),
        "mean_teacher_mixture": (
            float(np.mean(teacher_mixture_values))
            if teacher_mixture_values
            else 0.0
        ),
        "mean_bc_weight": (
            float(np.mean(bc_weight_values)) if bc_weight_values else 0.0
        ),
        "loss": losses[-1] if losses else None,
        "td_update_count": len(losses),
        "imitation_loss": (
            imitation_losses[-1] if imitation_losses else None
        ),
        **pressure.metrics(),
        **timing,
        **storage_flow,
    }


def validate(agent, selector_payload: dict, args, *, phase: str, device):
    """Evaluate in a separately constructed stack with no training mutation."""

    rng = capture_rng_state()
    training_state = _agent_checkpoint_state(agent, include_replay=False)
    try:
        eval_env, eval_selector, eval_agent = _build_stack(
            args,
            selector_payload,
            device=device,
            config=deepcopy(agent.config),
        )
        _load_agent_checkpoint_state(
            eval_agent, training_state, resumable=False
        )
        eval_agent.epsilon = 0.0
        eval_agent.set_training_phase(phase)
        for network_name in ("Q_local", "Q_target"):
            network = getattr(eval_agent, network_name, None)
            if network is not None:
                network.eval()
        runs = []
        with torch.inference_mode():
            for seed in args.validation_seeds:
                seed_everything(seed)
                instance = eval_env.sample_episode_instance(seed)
                result = run_episode(
                    eval_agent,
                    eval_selector,
                    eval_env,
                    instance,
                    max_steps=args.validation_steps,
                    target_window=args.target_window,
                    training=False,
                    phase=phase,
                    include_decisions=False,
                )
                result.update(
                    {
                        "eval_seed": int(seed),
                        "instance_id": instance.instance_id,
                        "schedule_id": instance.schedule_id,
                    }
                )
                runs.append(result)
        return {**summarize_runs(runs), "runs": runs}
    finally:
        restore_rng_state(rng)


def _phase_end_episodes(counts: dict[str, int]) -> set[int]:
    result = set()
    total = 0
    for phase in PHASE_ORDER:
        total += counts[phase]
        if counts[phase]:
            result.add(total)
    return result


def _write_summary(
    args,
    *,
    device,
    best_score,
    history,
    validations,
    agent,
):
    summary = json_safe(
        {
            "config": vars(args),
            "device": str(device),
            "phase_episode_counts": phase_episode_counts(args),
            "total_training_episodes": total_training_episodes(args),
            "best_score": best_score,
            "final_epsilon": agent.epsilon,
            "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
            "latest_checkpoint": str(
                (args.output_dir / "latest.pth").resolve()
            ),
            "phase_checkpoints": {
                phase: str((args.output_dir / f"{phase}-end.pth").resolve())
                for phase in PHASE_ORDER
                if (args.output_dir / f"{phase}-end.pth").is_file()
            },
            "history": history,
            "validations": validations,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


def main(argv=None):
    args = parse_args(argv)
    hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        python_hash_seed = int(hash_seed) if hash_seed is not None else None
    except ValueError:
        python_hash_seed = None
    if (
        python_hash_seed is None
        or not 0 <= python_hash_seed <= 4_294_967_295
    ):
        raise RuntimeError(
            "PYTHONHASHSEED must be a fixed integer in [0, 4294967295] "
            "before interpreter startup; 'random' is not reproducible"
        )
    args.python_hash_seed = python_hash_seed
    if args.resume is not None:
        if args.output_dir.resolve() != args.resume.resolve().parent:
            raise ValueError(
                "--resume must write back to the checkpoint's directory so "
                "the authenticated deployment best is preserved"
            )
    else:
        warm_start_path = (
            args.warm_start_temporal or args.warm_start_spatial
        )
        if (
            warm_start_path is not None
            and args.output_dir.resolve() == warm_start_path.resolve().parent
        ):
            raise ValueError(
                "a weights-only warm start must write to a new output directory"
            )
        conflicts = [
            path.name
            for path in (
                args.output_dir / "latest.pth",
                args.output_dir / "best.pth",
                args.output_dir / "training-summary.json",
                args.output_dir / "joint-init.pth",
                *(args.output_dir / f"{phase}-end.pth" for phase in PHASE_ORDER),
            )
            if path.exists()
        ]
        if conflicts:
            raise FileExistsError(
                "refusing to overwrite an existing training run: "
                + ", ".join(conflicts)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selector_payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(selector_payload, dict):
        raise ValueError("selector checkpoint payload must be a mapping")
    # Compare named fields independently. This deliberately avoids the malformed
    # tuple regime check that appeared in an older experimental trainer.
    if selector_payload.get("training_lambda") != args.lam:
        raise ValueError("selector training lambda does not match Track B")
    if selector_payload.get("training_mu") != args.mu:
        raise ValueError("selector training mu does not match Track B")

    device = resolve_device(args.device)
    counts = phase_episode_counts(args)
    total_episodes = total_training_episodes(args)
    history = []
    validations = []
    rolling_returns = deque(maxlen=25)

    if args.resume is not None:
        resume_payload = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if not isinstance(resume_payload, dict):
            raise ValueError("resume checkpoint payload must be a mapping")
        _validate_resume_payload(resume_payload, args, selector_payload)
        resume_best = _validate_resume_best_artifact(
            resume_payload, args.output_dir
        )
        env, selector, agent = build_stack_from_payload(
            args, resume_payload, device=device, for_evaluation=False
        )
        completed = int(resume_payload["completed_training_episodes"])
        start_episode = completed + 1
        exploration_state = {
            "decisions": int(resume_payload["macro_decisions"]),
            "phase_decisions": {
                phase: int(
                    resume_payload.get("phase_macro_decisions", {}).get(phase, 0)
                )
                for phase in PHASE_ORDER
            },
            "temporal_bad_validations": int(
                resume_payload.get("temporal_bad_validations", 0)
            ),
            "joint_bad_validations": int(
                resume_payload.get("joint_bad_validations", 0)
            ),
            "quality_gate_blocked": bool(
                resume_payload.get("quality_gate_blocked", False)
            ),
            "temporal_reference_validation": resume_payload.get(
                "temporal_reference_validation"
            ),
            "spatial_reference_validation": resume_payload.get(
                "spatial_reference_validation"
            ),
            "warm_start_provenance": resume_payload.get(
                "warm_start_provenance"
            ),
        }
        best_score = tuple(
            (
                resume_best.get("best_score")
                if resume_best is not None
                else resume_payload.get(
                    "best_score", (-1.0, -float("inf"))
                )
            )
        )
        history = list(resume_payload["training_history"])
        validations = list(resume_payload["validation_history"])
        rolling_returns.extend(
            item["return"] for item in history[-rolling_returns.maxlen :]
        )
        restore_rng_state(resume_payload["rng_state"])
        print(
            f"Resuming fully learned Track B at episode {start_episode} "
            f"from {args.resume}",
            flush=True,
        )
    elif args.warm_start_temporal is not None:
        source_payload = torch.load(
            args.warm_start_temporal,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(source_payload, dict):
            raise ValueError("v3 temporal warm-start payload must be a mapping")
        temporal_reference, warm_start_provenance = (
            _validate_v3_temporal_warm_start(
                source_payload,
                args,
                selector_payload,
                args.warm_start_temporal,
            )
        )
        seed_everything(args.seed)
        env, selector, agent = _build_stack(
            args, selector_payload, device=device
        )
        _import_v3_temporal_weights(agent, source_payload)
        del source_payload
        start_episode = 1
        exploration_state = {
            "decisions": 0,
            "phase_decisions": {phase: 0 for phase in PHASE_ORDER},
            "temporal_bad_validations": 0,
            "joint_bad_validations": 0,
            "quality_gate_blocked": False,
            "temporal_reference_validation": temporal_reference,
            "spatial_reference_validation": None,
            "warm_start_provenance": warm_start_provenance,
        }
        best_score = (-1.0, -float("inf"))
        print(
            "Imported authenticated v3 temporal weights from "
            f"{args.warm_start_temporal} (source ValR "
            f"{temporal_reference['mean_return']:.2f}, MAE "
            f"{temporal_reference['mean_absolute_error']:.2f}); optimizer, "
            "replay, clocks, and RNG start fresh",
            flush=True,
        )
    elif args.warm_start_spatial is not None:
        source_payload = torch.load(
            args.warm_start_spatial,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(source_payload, dict):
            raise ValueError("v4 spatial warm-start payload must be a mapping")
        (
            temporal_reference,
            spatial_reference,
            warm_start_provenance,
        ) = _validate_v4_spatial_warm_start(
            source_payload,
            args,
            selector_payload,
            args.warm_start_spatial,
        )
        seed_everything(args.seed)
        env, selector, agent = _build_stack(
            args, selector_payload, device=device
        )
        _import_v4_spatial_weights(agent, source_payload)
        del source_payload
        start_episode = 1
        exploration_state = {
            "decisions": 0,
            "phase_decisions": {phase: 0 for phase in PHASE_ORDER},
            "temporal_bad_validations": 0,
            "joint_bad_validations": 0,
            "quality_gate_blocked": False,
            "temporal_reference_validation": temporal_reference,
            "spatial_reference_validation": spatial_reference,
            "warm_start_provenance": warm_start_provenance,
        }
        best_score = (-1.0, -float("inf"))
        print(
            "Imported authenticated v4 spatial weights from "
            f"{args.warm_start_spatial} (source ValR "
            f"{spatial_reference['mean_return']:.2f}, MAE "
            f"{spatial_reference['mean_absolute_error']:.2f}); optimizer, "
            "replay, clocks, history, and RNG start fresh",
            flush=True,
        )
    else:
        seed_everything(args.seed)
        env, selector, agent = _build_stack(
            args, selector_payload, device=device
        )
        start_episode = 1
        exploration_state = {
            "decisions": 0,
            "phase_decisions": {phase: 0 for phase in PHASE_ORDER},
            "temporal_bad_validations": 0,
            "joint_bad_validations": 0,
            "quality_gate_blocked": False,
            "temporal_reference_validation": None,
            "spatial_reference_validation": None,
            "warm_start_provenance": None,
        }
        best_score = (-1.0, -float("inf"))

    if start_episode > total_episodes:
        raise ValueError("resume checkpoint already completed this phase schedule")
    phase_ends = _phase_end_episodes(counts)
    previous_phase = None
    print(
        f"Fully learned reserved hierarchy | device={device} | seed={args.seed} "
        f"| phases={counts} | total={total_episodes}",
        flush=True,
    )

    for episode in range(start_episode, total_episodes + 1):
        phase = phase_for_episode(episode, counts)
        if phase != previous_phase:
            transitioned = getattr(agent, "training_phase", None) != phase
            if transitioned:
                agent.set_training_phase(phase)
                if args.reset_optimizer_on_phase_transition:
                    agent.reset_optimizer_state(sync_target=True)
            _configure_phase_optimization(agent, args, phase)
            if phase == "joint" and transitioned:
                reference = (
                    exploration_state.get("spatial_reference_validation")
                    or exploration_state.get("temporal_reference_validation")
                )
                if reference is None:
                    raise RuntimeError(
                        "joint training requires a deployment-mode spatial baseline"
                    )
                joint_init = checkpoint_payload(
                    agent,
                    selector_payload,
                    args,
                    episode=episode - 1,
                    phase="joint",
                    validation={
                        **reference,
                        "episode": episode - 1,
                        "phase": "joint_init",
                    },
                    best_score=best_score,
                    exploration_state=exploration_state,
                    checkpoint_kind="joint_init_baseline",
                )
                joint_init_path = args.output_dir / "joint-init.pth"
                if joint_init_path.exists():
                    if args.resume is None:
                        raise FileExistsError(
                            "refusing to overwrite joint baseline: "
                            f"{joint_init_path}"
                        )
                    _validate_existing_joint_init(
                        joint_init_path,
                        agent,
                        args,
                        selector_payload,
                        completed_episode=episode - 1,
                    )
                else:
                    atomic_torch_save(joint_init, joint_init_path)
            print(
                f"Phase -> {phase} at episode {episode} | "
                f"macro decisions={exploration_state['decisions']} | "
                f"phase clock={exploration_state['phase_decisions'][phase]}",
                flush=True,
            )
            previous_phase = phase
        instance_seed = (
            args.training_instance_seed_base
            + args.seed * 1_000_000
            + episode
        )
        instance = env.sample_episode_instance(instance_seed)
        result = run_episode(
            agent,
            selector,
            env,
            instance,
            max_steps=args.max_steps,
            target_window=args.target_window,
            training=True,
            phase=phase,
            exploration_state=exploration_state,
            epsilon_args=args,
            failure_penalty=args.failure_penalty,
            updates_per_macro=(
                args.joint_updates_per_macro
                if phase == "joint"
                else args.updates_per_macro
            ),
            joint_td_warmup_decisions=args.joint_td_warmup_decisions,
            include_decisions=False,
        )
        result.update(
            {
                "episode": episode,
                "instance_seed": instance_seed,
                "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "epsilon": agent.epsilon,
            }
        )
        history.append(result)
        rolling_returns.append(result["return"])

        should_validate = bool(
            episode % args.eval_every == 0
            or episode in phase_ends
            or episode == total_episodes
        )
        if should_validate:
            validation = validate(
                agent,
                selector_payload,
                args,
                phase=phase,
                device=device,
            )
            validation.update({"episode": episode, "phase": phase})
            exploration_state["quality_gate_blocked"] = False
            spatial_gate_failed = False
            temporal_gate_failed = False
            joint_gate_state = None
            if phase == "temporal" and episode in phase_ends:
                exploration_state["temporal_reference_validation"] = (
                    _validation_reference(validation, source="temporal_end")
                )
            if phase == "spatial":
                reference = exploration_state.get(
                    "temporal_reference_validation"
                )
                if reference is None:
                    raise RuntimeError(
                        "spatial calibration has no temporal validation baseline"
                    )
                gate = validation_quality_gate(
                    validation,
                    reference,
                    min_return_ratio=args.spatial_min_return_ratio,
                    max_mae_ratio=args.spatial_max_mae_ratio,
                )
                validation["quality_gate"] = gate
                if episode in phase_ends:
                    exploration_state["spatial_reference_validation"] = (
                        _validation_reference(validation, source="spatial_end")
                    )
                    spatial_gate_failed = not gate["passed"]
            elif phase == "joint":
                reference = exploration_state.get(
                    "spatial_reference_validation"
                )
                if reference is None:
                    raise RuntimeError(
                        "joint validation has no immutable spatial baseline"
                    )
                gate = validation_quality_gate(
                    validation,
                    reference,
                    min_return_ratio=args.joint_min_return_ratio,
                    max_mae_ratio=args.joint_max_mae_ratio,
                )
                validation["quality_gate"] = gate
                joint_gate_state = joint_gate_decision(
                    gate,
                    phase_decisions=exploration_state["phase_decisions"][
                        "joint"
                    ],
                    warmup_decisions=args.joint_gate_warmup_decisions,
                    prior_bad_validations=exploration_state[
                        "joint_bad_validations"
                    ],
                    collapse_patience=args.joint_collapse_patience,
                )
                validation["quality_gate_enforcement"] = joint_gate_state
                exploration_state["joint_bad_validations"] = (
                    joint_gate_state["bad_validations"]
                )
            validations.append(validation)
            if phase == "temporal":
                if validation["strict_method_success_rate"] < (
                    args.temporal_min_validation_strict
                ):
                    exploration_state["temporal_bad_validations"] += 1
                    temporal_gate_failed = bool(episode in phase_ends)
                else:
                    exploration_state["temporal_bad_validations"] = 0
            if (
                phase == "temporal"
                and (
                    temporal_gate_failed
                    or exploration_state["temporal_bad_validations"]
                    >= args.temporal_collapse_patience
                )
            ):
                exploration_state["quality_gate_blocked"] = True
            elif spatial_gate_failed:
                exploration_state["quality_gate_blocked"] = True
            elif phase == "joint" and joint_gate_state["blocked"]:
                exploration_state["quality_gate_blocked"] = True
            score = (
                validation["strict_method_success_rate"],
                validation["mean_return"],
            )
            # A deployable "best" must have reached the unrestricted joint
            # action set. Teacher-forced or locked-cell checkpoints cannot win.
            improved = bool(
                phase == "joint"
                and validation.get("quality_gate", {}).get("passed", False)
                and score > best_score
            )
            if improved:
                best_score = score
            if improved:
                deployment = checkpoint_payload(
                    agent,
                    selector_payload,
                    args,
                    episode=episode,
                    phase=phase,
                    validation=validation,
                    best_score=best_score,
                    exploration_state=exploration_state,
                    checkpoint_kind="deployment_best",
                )
                # Commit best first. If the process dies before latest is
                # replaced, resume accepts this one-validation-ahead artifact
                # and retains its score while replaying from the older state.
                atomic_torch_save(deployment, args.output_dir / "best.pth")
            latest = checkpoint_payload(
                agent,
                selector_payload,
                args,
                episode=episode,
                phase=phase,
                validation=validation,
                best_score=best_score,
                exploration_state=exploration_state,
                checkpoint_kind="resumable_latest",
                history=history,
                validations=validations,
            )
            atomic_torch_save(latest, args.output_dir / "latest.pth")
            if episode in phase_ends:
                phase_path = args.output_dir / f"{phase}-end.pth"
                if phase_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite immutable phase checkpoint: {phase_path}"
                    )
                phase_checkpoint = checkpoint_payload(
                    agent,
                    selector_payload,
                    args,
                    episode=episode,
                    phase=phase,
                    validation=validation,
                    best_score=best_score,
                    exploration_state=exploration_state,
                    checkpoint_kind="resumable_phase_end",
                    history=history,
                    validations=validations,
                )
                atomic_torch_save(phase_checkpoint, phase_path)
            _write_summary(
                args,
                device=device,
                best_score=best_score,
                history=history,
                validations=validations,
                agent=agent,
            )
            print(
                f"Ep {episode:4d} | Phase {phase:9s} | "
                f"TrainR {np.mean(rolling_returns):8.2f} | "
                f"ValR {validation['mean_return']:8.2f} | "
                f"ValStrict {validation['strict_method_success_rate']:.3f} | "
                f"ValReserve {validation['reservation_integrity_rate']:.3f} | "
                f"Eps {agent.epsilon:.3f} | Replay {len(agent.replay)} | "
                f"Mix {result['mean_teacher_mixture']:.3f} | "
                f"BCw {result['mean_bc_weight']:.3f} | "
                f"Loss {result['loss']} | BC {result['imitation_loss']}",
                flush=True,
            )
            if (
                phase == "temporal"
                and (
                    temporal_gate_failed
                    or exploration_state["temporal_bad_validations"]
                    >= args.temporal_collapse_patience
                )
            ):
                raise RuntimeError(
                    "temporal validation gate stopped training after "
                    f"{exploration_state['temporal_bad_validations']} consecutive "
                    "strict-success failures (the temporal endpoint must always "
                    "pass); restart from imitation-end.pth"
                )
            if spatial_gate_failed:
                gate = validation["quality_gate"]
                raise RuntimeError(
                    "spatial calibration gate blocked joint training after its "
                    f"diagnostic checkpoint: ValR {gate['observed_return']:.2f} "
                    f"(required {gate['return_threshold']:.2f}), MAE "
                    f"{gate['observed_mae']:.2f} (maximum "
                    f"{gate['mae_threshold']:.2f}), Strict "
                    f"{gate['observed_strict_method_success_rate']:.3f}, "
                    "Reserve "
                    f"{gate['observed_reservation_integrity_rate']:.3f}"
                )
            if phase == "joint" and joint_gate_state["blocked"]:
                gate = validation["quality_gate"]
                if joint_gate_state["block_reason"] == "safety":
                    raise RuntimeError(
                        "joint safety gate stopped training immediately: Strict "
                        f"{gate['observed_strict_method_success_rate']:.3f} "
                        f"(required {gate['reference_strict_method_success_rate']:.3f}), "
                        "Reserve "
                        f"{gate['observed_reservation_integrity_rate']:.3f} "
                        f"(required {gate['reference_reservation_integrity_rate']:.3f}). "
                        "The performance warmup and patience never override safety."
                    )
                raise RuntimeError(
                    "joint quality gate stopped training after "
                    f"{exploration_state['joint_bad_validations']} consecutive "
                    "post-warmup deployment-mode performance failures; last ValR "
                    f"{gate['observed_return']:.2f} (required "
                    f"{gate['return_threshold']:.2f}), MAE "
                    f"{gate['observed_mae']:.2f} (maximum "
                    f"{gate['mae_threshold']:.2f}), Strict "
                    f"{gate['observed_strict_method_success_rate']:.3f}, "
                    "Reserve "
                    f"{gate['observed_reservation_integrity_rate']:.3f}. "
                    "Preserve this run for "
                    "diagnosis, then warm-start the authenticated spatial endpoint "
                    "into a new v4.1 output directory."
                )
        elif episode % args.log_every == 0 or episode == start_episode:
            print(
                f"Ep {episode:4d} | Phase {phase:9s} | "
                f"TrainR {np.mean(rolling_returns):8.2f} | "
                f"Strict {result['strict_method_success']:.0f} | "
                f"Reserve {result['reservation_integrity']:.0f} | "
                f"Eps {agent.epsilon:.3f} | Dec {result['macro_count']} | "
                f"Replay {len(agent.replay)} | "
                f"Mix {result['mean_teacher_mixture']:.3f} | "
                f"BCw {result['mean_bc_weight']:.3f} | "
                f"Loss {result['loss']} | "
                f"BC {result['imitation_loss']}",
                flush=True,
            )

    final_latest = torch.load(
        args.output_dir / "latest.pth", map_location="cpu", weights_only=False
    )
    _validate_resume_best_artifact(final_latest, args.output_dir)
    print(
        json.dumps(
            {
                "best_score": best_score,
                "final_epsilon": agent.epsilon,
                "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
                "latest_checkpoint": str(
                    (args.output_dir / "latest.pth").resolve()
                ),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", type=Path)
    restart.add_argument("--warm-start-temporal", type=Path)
    restart.add_argument("--warm-start-spatial", type=Path)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--exit-width", type=int)
    parser.add_argument("--number-blocks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imitation-episodes", type=int, default=50)
    parser.add_argument("--temporal-episodes", type=int, default=150)
    parser.add_argument("--spatial-episodes", type=int, default=50)
    parser.add_argument("--joint-episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--update-every", type=int, default=100)
    parser.add_argument("--updates-per-macro", type=int, default=1)
    parser.add_argument("--joint-updates-per-macro", type=int, default=1)
    parser.add_argument(
        "--joint-td-warmup-decisions", type=int, default=2_000
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--spatial-learning-rate", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--failure-penalty", type=float, default=-50.0)
    parser.add_argument(
        "--replay-sampling",
        choices=("mode_balanced", "regime_mode_balanced", "uniform"),
        default="mode_balanced",
    )
    parser.add_argument("--temporal-epsilon-start", type=float, default=0.25)
    parser.add_argument("--temporal-epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--temporal-epsilon-warmup-decisions", type=int, default=0
    )
    parser.add_argument(
        "--temporal-epsilon-decay-decisions", type=int, default=10_000
    )
    parser.add_argument("--joint-epsilon-start", type=float, default=0.05)
    parser.add_argument("--joint-epsilon-end", type=float, default=0.05)
    parser.add_argument("--joint-epsilon-warmup-decisions", type=int, default=0)
    parser.add_argument(
        "--joint-epsilon-decay-decisions", type=int, default=20_000
    )
    parser.add_argument(
        "--temporal-teacher-mixture-start", type=float, default=0.50
    )
    parser.add_argument(
        "--temporal-teacher-mixture-end", type=float, default=0.0
    )
    parser.add_argument(
        "--temporal-teacher-mixture-warmup-decisions", type=int, default=0
    )
    parser.add_argument(
        "--temporal-teacher-mixture-decay-decisions", type=int, default=10_000
    )
    parser.add_argument(
        "--joint-teacher-mixture-start", type=float, default=0.25
    )
    parser.add_argument("--joint-teacher-mixture-end", type=float, default=0.0)
    parser.add_argument(
        "--joint-teacher-mixture-warmup-decisions", type=int, default=2_000
    )
    parser.add_argument(
        "--joint-teacher-mixture-decay-decisions", type=int, default=8_000
    )
    parser.add_argument("--temporal-bc-start", type=float, default=0.50)
    parser.add_argument("--temporal-bc-end", type=float, default=0.05)
    parser.add_argument("--temporal-bc-warmup-decisions", type=int, default=0)
    parser.add_argument(
        "--temporal-bc-decay-decisions", type=int, default=10_000
    )
    parser.add_argument("--joint-bc-start", type=float, default=0.20)
    parser.add_argument("--joint-bc-end", type=float, default=0.02)
    parser.add_argument("--joint-bc-warmup-decisions", type=int, default=5_000)
    parser.add_argument(
        "--joint-bc-decay-decisions", type=int, default=20_000
    )
    parser.add_argument("--joint-main-lr-scale", type=float, default=0.10)
    parser.add_argument("--joint-spatial-lr-scale", type=float, default=0.10)
    parser.add_argument(
        "--joint-spatial-freeze-decisions", type=int, default=5_000
    )
    parser.add_argument(
        "--reset-optimizer-on-phase-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--temporal-min-validation-strict", type=float, default=1.0
    )
    parser.add_argument("--temporal-collapse-patience", type=int, default=2)
    parser.add_argument("--spatial-min-return-ratio", type=float, default=0.90)
    parser.add_argument("--spatial-max-mae-ratio", type=float, default=2.0)
    parser.add_argument("--joint-min-return-ratio", type=float, default=0.90)
    parser.add_argument("--joint-max-mae-ratio", type=float, default=2.0)
    parser.add_argument(
        "--joint-gate-warmup-decisions", type=int, default=3_000
    )
    parser.add_argument("--joint-collapse-patience", type=int, default=2)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--block-embedding-dim", type=int, default=64)
    parser.add_argument("--global-embedding-dim", type=int, default=64)
    parser.add_argument("--candidate-embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--tau-accept", type=float, default=0.1)
    parser.add_argument("--tau-retrieve", type=float, default=0.1)
    parser.add_argument("--tau-defer", type=float, default=0.1)
    parser.add_argument("--tau-mode", type=float, default=1.0)
    parser.add_argument("--teacher-coefficient", type=float, default=0.0)
    parser.add_argument("--teacher-score-temperature", type=float, default=1.0)
    parser.add_argument("--spatial-distillation-weight", type=float, default=1.0)
    parser.add_argument(
        "--joint-accept-exploration-top-k", type=int, default=8
    )
    parser.add_argument("--lookahead-margin-steps", type=float, default=2.0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[10000, 10001, 10002, 10003, 10004],
    )
    parser.add_argument("--validation-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--training-instance-seed-base", type=int, default=600_000)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    counts = phase_episode_counts(args)
    if any(value < 0 for value in counts.values()) or not sum(counts.values()):
        parser.error("phase episode counts must be nonnegative with a positive total")
    if counts["joint"] <= 0:
        parser.error(
            "joint-episodes must be positive for a deployable fully learned run"
        )
    if args.warm_start_spatial is None and counts["spatial"] <= 0:
        parser.error(
            "spatial-episodes must be positive to calibrate the full cell set"
        )
    if args.warm_start_temporal is not None and (
        counts["imitation"] != 0
        or counts["temporal"] != 0
        or counts["spatial"] <= 0
    ):
        parser.error(
            "--warm-start-temporal requires imitation-episodes=0, "
            "temporal-episodes=0, and spatial-episodes>0"
        )
    if args.warm_start_spatial is not None and (
        counts["imitation"] != 0
        or counts["temporal"] != 0
        or counts["spatial"] != 0
    ):
        parser.error(
            "--warm-start-spatial requires imitation-episodes=0, "
            "temporal-episodes=0, and spatial-episodes=0"
        )
    if (
        args.warm_start_temporal is None
        and args.warm_start_spatial is None
        and args.resume is None
        and counts["temporal"] <= 0
    ):
        parser.error(
            "a fresh v4 run requires temporal-episodes>0 before spatial calibration"
        )
    unknown = set(counts).difference(TRAINING_PHASES)
    if unknown:
        parser.error(f"core does not support training phases: {sorted(unknown)}")
    positive = (
        "max_steps",
        "batch_size",
        "buffer_size",
        "update_every",
        "updates_per_macro",
        "joint_updates_per_macro",
        "temporal_epsilon_decay_decisions",
        "joint_epsilon_decay_decisions",
        "temporal_teacher_mixture_decay_decisions",
        "joint_teacher_mixture_decay_decisions",
        "temporal_bc_decay_decisions",
        "joint_bc_decay_decisions",
        "max_defer_steps",
        "eval_every",
        "log_every",
        "validation_steps",
        "temporal_collapse_patience",
        "joint_collapse_patience",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("step, buffer, update, and logging arguments must be positive")
    if args.buffer_size < args.batch_size:
        parser.error("buffer-size must be at least batch-size")
    schedules = (
        ("temporal epsilon", args.temporal_epsilon_start, args.temporal_epsilon_end),
        ("joint epsilon", args.joint_epsilon_start, args.joint_epsilon_end),
        (
            "temporal teacher mixture",
            args.temporal_teacher_mixture_start,
            args.temporal_teacher_mixture_end,
        ),
        (
            "joint teacher mixture",
            args.joint_teacher_mixture_start,
            args.joint_teacher_mixture_end,
        ),
    )
    for label, start, end in schedules:
        if not 0 <= end <= start <= 1:
            parser.error(f"{label} must satisfy 0 <= end <= start <= 1")
    if not 0 <= args.temporal_bc_end <= args.temporal_bc_start:
        parser.error("temporal BC must satisfy 0 <= end <= start")
    if not 0 <= args.joint_bc_end <= args.joint_bc_start:
        parser.error("joint BC must satisfy 0 <= end <= start")
    warmups = (
        args.temporal_epsilon_warmup_decisions,
        args.joint_epsilon_warmup_decisions,
        args.temporal_teacher_mixture_warmup_decisions,
        args.joint_teacher_mixture_warmup_decisions,
        args.temporal_bc_warmup_decisions,
        args.joint_bc_warmup_decisions,
        args.joint_td_warmup_decisions,
        args.joint_gate_warmup_decisions,
    )
    if any(value < 0 for value in warmups):
        parser.error("phase schedule warmups must be nonnegative")
    if args.joint_spatial_freeze_decisions < 0:
        parser.error("joint-spatial-freeze-decisions must be nonnegative")
    if not args.validation_seeds:
        parser.error("at least one validation seed is required")
    if not 0.0 <= args.temporal_min_validation_strict <= 1.0:
        parser.error("temporal-min-validation-strict must be in [0, 1]")
    for name in ("spatial_min_return_ratio", "joint_min_return_ratio"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 < value <= 1:
            parser.error(f"{name.replace('_', '-')} must be in (0, 1]")
    for name in ("spatial_max_mae_ratio", "joint_max_mae_ratio"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 1:
            parser.error(f"{name.replace('_', '-')} must be at least 1")
    if not 0 < args.gamma <= 1:
        parser.error("gamma must be in (0, 1]")
    if not 0 < args.target_tau <= 1:
        parser.error("target-tau must be in (0, 1]")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (
            args.learning_rate,
            args.spatial_learning_rate,
            args.grad_clip,
            args.reward_scale,
            args.joint_main_lr_scale,
            args.joint_spatial_lr_scale,
        )
    ):
        parser.error("learning rates, grad-clip, and reward-scale must be positive")
    if not np.isfinite(args.failure_penalty) or args.failure_penalty > 0:
        parser.error("failure-penalty must be finite and nonpositive")
    if not np.isfinite(args.target_window) or args.target_window < 0:
        parser.error("target-window must be finite and nonnegative")
    try:
        _config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    main()
