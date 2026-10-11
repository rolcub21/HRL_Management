#!/usr/bin/env python3
"""Mixed-layout/load adaptation for the fully learned Track-B hierarchy.

This is a separate experiment contract.  It imports an authenticated v4.1
deployment checkpoint, keeps the neural/action semantics unchanged, and trains
one agent over a deterministic round-robin distribution of environment
regimes.  A larger regime must not dominate replay merely because it produces
more macro decisions, so sampling balances regimes before temporal modes.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from example.Options.selector_v5 import ExplicitCellAssignmentRegistry
from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import build_controller_options
from example.track_b_regimes import MIXED_TRAIN_V1, canonical_manifest
from fully_learned_hierarchy import FULLY_LEARNED_ACTION_INTERFACE
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.reg_selector_v5 import REGV5AssignmentSource
from track_b_fully_learned_evaluate import validate_deployment_payload
from train_fully_learned_track_b import (
    CURRICULUM_CONTRACT,
    FAILURE_CONTRACT,
    FAILURE_PENALTY_CONTRACT,
    FULLY_LEARNED_TRAINER_VERSION,
    METHOD,
    MODEL_SELECTION_CONTRACT,
    RAW_MACRO_RETURN_CONTRACT,
    TRUNCATION_CONTRACT,
    _agent_checkpoint_state,
    atomic_torch_save,
    build_stack_from_payload,
    capture_rng_state,
    json_safe,
    resolve_device,
    restore_rng_state,
    run_episode,
    seed_everything,
)


MIXED_CHECKPOINT_SCHEMA_VERSION = 2
MIXED_TRAINER_VERSION = 2
MIXED_METHOD = "fully_learned_reserved_macro_hierarchy_v5_mixed"
MIXED_CURRICULUM = "v4_1_synced_weight_transfer_joint_mixed_regime_v2"
MIXED_REPLAY_CONTRACT = "regime_then_mode_balanced_without_replacement_v1"
MIXED_SAMPLER_CONTRACT = "seed_shuffled_round_robin_equal_episode_v1"
MIXED_INSTANCE_SEED_CONTRACT = "cycle_paired_across_regimes_disjoint_by_train_seed_v1"
MIXED_TRANSFER_CONTRACT = "q_local_import_target_hard_sync_adam_reset_v1"
MIXED_MODEL_SELECTION_CONTRACT = (
    "strict_reservation_retention_sign_safe_macro_return_mae_v2"
)
MIXED_BASE_DEPLOYMENT_CONTRACT = "embedded_authenticated_v4_1_semantics_v1"
MIXED_TARGET_SYNC_CONTRACT = "Q_target_equals_imported_Q_local_v1"


BASE_PROVENANCE_FIELDS = (
    "fully_learned_checkpoint_schema_version",
    "fully_learned_trainer_version",
    "curriculum_contract",
    "checkpoint_kind",
    "resumable",
    "controller_architecture",
    "controller_action_interface",
    "network_architecture",
    "replay_version",
    "backup_version",
    "candidate_feature_version",
    "macro_return_contract",
    "truncation_contract",
    "failure_contract",
    "failure_penalty_contract",
    "model_selection_contract",
    "common_continuation",
    "deployment_reg_policy_query",
    "selector_frozen_independent_replay_disabled",
    "gamma",
    "reward_scale",
    "failure_penalty",
    "training_lambda",
    "training_mu",
    "training_seed",
    "selector_deployment_digest",
    "geometry",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    encoded = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def base_deployment_provenance(payload):
    """Capture the exact authenticated v4.1 semantics used for transfer."""

    missing = [key for key in BASE_PROVENANCE_FIELDS if key not in payload]
    if missing:
        raise ValueError(
            f"base deployment checkpoint lacks provenance fields: {missing}"
        )
    contract = json_safe(
        {key: payload[key] for key in BASE_PROVENANCE_FIELDS}
    )
    return {
        "contract": MIXED_BASE_DEPLOYMENT_CONTRACT,
        "fields": contract,
        "sha256": _json_digest(contract),
    }


def validate_anchor_payload(payload, regime):
    """Require the imported v4.1 deployment to be the manifest anchor."""

    provenance = regime.provenance()
    expected = {
        "training_lambda": regime.arrival_rate,
        "training_mu": regime.proc_mean,
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    saved_geometry = payload.get("geometry")
    expected_geometry = provenance["geometry"]
    geometry_fields = (
        "geometry_signature",
        "grid_rows",
        "grid_cols",
        "requested_exit_width",
        "block_count",
    )
    if not isinstance(saved_geometry, dict):
        mismatches["geometry"] = "missing"
    else:
        geometry_mismatches = {
            key: {
                "expected": expected_geometry[key],
                "found": saved_geometry.get(key),
            }
            for key in geometry_fields
            if saved_geometry.get(key) != expected_geometry[key]
        }
        if geometry_mismatches:
            mismatches["geometry"] = geometry_mismatches
    if mismatches:
        raise ValueError(
            f"warm-start does not match the mixed manifest anchor: {mismatches}"
        )


def synchronize_imported_target(agent):
    """Start mixed adaptation from one critic, without inherited Adam lag."""

    agent.optimizer.state.clear()
    agent.Q_target.load_state_dict(agent.Q_local.state_dict(), strict=True)
    agent.Q_target.requires_grad_(False)
    agent.Q_target.eval()
    local = agent.Q_local.state_dict()
    target = agent.Q_target.state_dict()
    if set(local) != set(target) or any(
        not torch.equal(local[name].detach().cpu(), target[name].detach().cpu())
        for name in local
    ):
        raise RuntimeError("mixed warm-start target synchronization failed")


def _runtime(regime, selector_payload, *, device, seed):
    env = regime.make_env()
    source = REGV5AssignmentSource.from_checkpoint(
        env,
        selector_payload,
        learning_enabled=False,
        device=device,
        seed=seed,
    )
    selector = ExplicitCellAssignmentRegistry(env, source)
    build_controller_options(
        env,
        selector,
        controller_action_interface=FULLY_LEARNED_ACTION_INTERFACE,
        max_defer_steps=10,
    )
    env.reset(instance=env.sample_episode_instance(0))
    return env, selector, OnlineManifestTimingObservationEncoder(env)


def _agent_args(regime, seed):
    return SimpleNamespace(
        lam=regime.arrival_rate,
        mu=regime.proc_mean,
        grid_rows=regime.grid_rows,
        grid_cols=regime.grid_cols,
        exit_width=regime.exit_width,
        number_blocks=regime.number_blocks,
        seed=int(seed),
    )


def _compact_summary(runs):
    returns = np.asarray([item["return"] for item in runs], dtype=float)
    errors = np.asarray(
        [item["mean_absolute_error"] for item in runs], dtype=float
    )
    return {
        "episodes": len(runs),
        "mean_return": float(returns.mean()),
        "return_std": (
            float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        ),
        "strict_method_success_rate": float(
            np.mean([item["strict_method_success"] for item in runs])
        ),
        "reservation_integrity_rate": float(
            np.mean([item["reservation_integrity"] for item in runs])
        ),
        "mean_absolute_error": float(errors.mean()),
        "mean_steps": float(np.mean([item["steps"] for item in runs])),
        "total_relocations": int(
            sum(item["obstructive_moves"] for item in runs)
        ),
        "episodes_with_relocation": int(
            sum(item["obstructive_moves"] > 0 for item in runs)
        ),
        "method_failures": [
            item["method_failure_reason"]
            for item in runs
            if item["method_failure_reason"] is not None
        ],
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
    }


def evaluate(agent, runtimes, regimes, *, seeds, max_steps, target_window):
    local_training = agent.Q_local.training
    target_training = agent.Q_target.training
    agent.Q_local.eval()
    agent.Q_target.eval()
    summaries = {}
    all_runs = {}
    with torch.inference_mode():
        for regime in regimes:
            env, selector, encoder = runtimes[regime.regime_id]
            agent.bind_runtime(env, encoder, regime_id=regime.regime_id)
            runs = []
            for seed in seeds:
                seed_everything(seed)
                instance = env.sample_episode_instance(seed)
                result = run_episode(
                    agent,
                    selector,
                    env,
                    instance,
                    max_steps=max_steps,
                    target_window=target_window,
                    training=False,
                    phase="joint",
                )
                result.update(
                    {
                        "eval_seed": int(seed),
                        "instance_id": instance.instance_id,
                        "schedule_id": instance.schedule_id,
                    }
                )
                runs.append(result)
            summaries[regime.regime_id] = _compact_summary(runs)
            all_runs[regime.regime_id] = runs
    agent.Q_local.train(local_training)
    agent.Q_target.train(target_training)
    agent.Q_target.eval()
    return {"per_regime": summaries, "runs": all_runs}


def score_validation(validation, reference):
    current = validation["per_regime"]
    baseline = reference["per_regime"]
    if not current:
        raise ValueError("validation must contain at least one regime")
    if set(current) != set(baseline):
        raise ValueError(
            "validation/reference regime mismatch: "
            f"validation={sorted(current)}, reference={sorted(baseline)}"
        )
    ids = sorted(current)

    def metric(source, regime_id, name):
        try:
            value = float(source[regime_id][name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {name!r} for regime {regime_id!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"nonfinite {name!r} for regime {regime_id!r}"
            )
        return value

    # A raw ``current / reference`` ratio reverses improvement whenever the
    # reference return is negative.  The signed, scale-normalized score below
    # preserves the familiar ratio for positive references while always
    # assigning a value above one to an improvement and below one to a loss.
    def return_retention(regime_id):
        value = metric(current, regime_id, "mean_return")
        anchor = metric(baseline, regime_id, "mean_return")
        return 1.0 + (value - anchor) / max(abs(anchor), 1.0)

    min_strict = min(
        metric(current, key, "strict_method_success_rate") for key in ids
    )
    min_reservation = min(
        metric(current, key, "reservation_integrity_rate") for key in ids
    )
    return_retentions = [return_retention(key) for key in ids]
    worst_return_ratio = min(return_retentions)
    worst_mae_ratio = max(
        metric(current, key, "mean_absolute_error")
        / max(1.0e-6, metric(baseline, key, "mean_absolute_error"))
        for key in ids
    )
    macro_return_ratio = float(np.mean(return_retentions))
    macro_mae = float(
        np.mean(
            [metric(current, key, "mean_absolute_error") for key in ids]
        )
    )
    retention_gate = float(
        worst_return_ratio >= 0.90 and worst_mae_ratio <= 2.0
    )
    return (
        float(min_strict),
        float(min_reservation),
        retention_gate,
        macro_return_ratio,
        -macro_mae,
        float(worst_return_ratio),
        float(-worst_mae_ratio),
    )


def checkpoint_payload(
    agent,
    base_payload,
    args,
    manifest,
    *,
    episode,
    cycle,
    validation,
    reference,
    best_score,
    history,
    exposure_counts,
    include_replay,
    checkpoint_kind,
):
    return {
        "mixed_checkpoint_schema_version": MIXED_CHECKPOINT_SCHEMA_VERSION,
        "mixed_trainer_version": MIXED_TRAINER_VERSION,
        "method": MIXED_METHOD,
        "curriculum_contract": MIXED_CURRICULUM,
        "checkpoint_kind": checkpoint_kind,
        "resumable": bool(include_replay),
        **agent.checkpoint_metadata(),
        "fully_learned_config": agent.config.to_dict(),
        "agent_checkpoint_state": _agent_checkpoint_state(
            agent, include_replay=include_replay
        ),
        "selector_checkpoint": base_payload["selector_checkpoint"],
        "selector_deployment_digest": base_payload[
            "selector_deployment_digest"
        ],
        "base_checkpoint": str(args.warm_start.resolve()),
        "base_checkpoint_sha256": _sha256(args.warm_start),
        "base_training_seed": base_payload["training_seed"],
        "base_deployment_provenance": base_deployment_provenance(base_payload),
        "warm_start_target_sync": MIXED_TARGET_SYNC_CONTRACT,
        "transfer_contract": MIXED_TRANSFER_CONTRACT,
        "model_selection_contract": MIXED_MODEL_SELECTION_CONTRACT,
        "instance_seed_contract": MIXED_INSTANCE_SEED_CONTRACT,
        "macro_return_contract": base_payload["macro_return_contract"],
        "truncation_contract": base_payload["truncation_contract"],
        "failure_contract": base_payload["failure_contract"],
        "failure_penalty_contract": base_payload[
            "failure_penalty_contract"
        ],
        "training_seed": args.seed,
        "training_manifest": manifest,
        "replay_sampling_contract": MIXED_REPLAY_CONTRACT,
        "regime_sampler_contract": MIXED_SAMPLER_CONTRACT,
        "episodes_per_regime": args.episodes_per_regime,
        "completed_training_episodes": int(episode),
        "completed_cycles": int(cycle),
        "exposure_counts": dict(exposure_counts),
        "validation_seeds": list(args.validation_seeds),
        "validation": validation,
        "initial_reference_validation": reference,
        "best_score": list(best_score),
        "max_steps": args.max_steps,
        "target_window": args.target_window,
        "learning_rate_scale": args.learning_rate_scale,
        "spatial_learning_rate_scale": args.spatial_learning_rate_scale,
        "epsilon": args.epsilon,
        "joint_td_warmup_decisions": args.joint_td_warmup_decisions,
        "teacher_retention_schedule": {
            "mixture_start": args.teacher_mixture_start,
            "mixture_hold_decisions": args.teacher_mixture_hold_decisions,
            "mixture_decay_decisions": args.teacher_mixture_decay_decisions,
            "bc_start": args.bc_start,
            "bc_end": args.bc_end,
            "bc_hold_decisions": args.bc_hold_decisions,
            "bc_decay_decisions": args.bc_decay_decisions,
            "spatial_freeze_decisions": args.spatial_freeze_decisions,
        },
        "history": history,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-regime", type=int, default=40)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=(62000, 62001))
    parser.add_argument("--eval-every-cycles", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--target-window", type=float, default=20.0)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--joint-td-warmup-decisions", type=int, default=2000)
    parser.add_argument("--teacher-mixture-start", type=float, default=0.25)
    parser.add_argument(
        "--teacher-mixture-hold-decisions", type=int, default=2000
    )
    parser.add_argument(
        "--teacher-mixture-decay-decisions", type=int, default=8000
    )
    parser.add_argument("--bc-start", type=float, default=0.20)
    parser.add_argument("--bc-end", type=float, default=0.02)
    parser.add_argument("--bc-hold-decisions", type=int, default=5000)
    parser.add_argument("--bc-decay-decisions", type=int, default=20000)
    parser.add_argument("--spatial-freeze-decisions", type=int, default=5000)
    parser.add_argument("--learning-rate-scale", type=float, default=0.10)
    parser.add_argument(
        "--spatial-learning-rate-scale", type=float, default=0.10
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=30000)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--training-instance-seed-base", type=int, default=700000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes_per_regime <= 0 or args.eval_every_cycles <= 0:
        parser.error("episode and evaluation counts must be positive")
    if args.max_steps <= 0 or args.batch_size <= 0 or args.buffer_size <= 0:
        parser.error("step, batch, and buffer sizes must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("epsilon must be in [0, 1]")
    if not 0.0 <= args.teacher_mixture_start <= 1.0:
        parser.error("teacher-mixture-start must be in [0, 1]")
    if not 0.0 <= args.bc_end <= args.bc_start:
        parser.error("BC weights must satisfy 0 <= end <= start")
    decision_counts = (
        args.joint_td_warmup_decisions,
        args.teacher_mixture_hold_decisions,
        args.teacher_mixture_decay_decisions,
        args.bc_hold_decisions,
        args.bc_decay_decisions,
        args.spatial_freeze_decisions,
    )
    if any(value < 0 for value in decision_counts):
        parser.error("decision schedules must be nonnegative")
    if args.teacher_mixture_decay_decisions <= 0 or args.bc_decay_decisions <= 0:
        parser.error("schedule decay decisions must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be absent or empty")
    args.validation_seeds = tuple(dict.fromkeys(args.validation_seeds))
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    base_payload = torch.load(
        args.warm_start, map_location="cpu", weights_only=False
    )
    if not isinstance(base_payload, dict):
        raise ValueError("warm-start checkpoint must be a mapping")
    validate_deployment_payload(base_payload)
    validate_anchor_payload(base_payload, MIXED_TRAIN_V1[0])
    base_deployment_provenance(base_payload)
    manifest = canonical_manifest(MIXED_TRAIN_V1)
    selector_payload = base_payload["selector_checkpoint"]
    if selector_deployment_digest(selector_payload) != base_payload[
        "selector_deployment_digest"
    ]:
        raise ValueError("embedded selector digest mismatch")

    anchor = MIXED_TRAIN_V1[0]
    anchor_args = _agent_args(anchor, args.seed)
    env, selector, agent = build_stack_from_payload(
        anchor_args, base_payload, device=device, for_evaluation=True
    )
    runtimes = {
        anchor.regime_id: (
            env,
            selector,
            OnlineManifestTimingObservationEncoder(env),
        )
    }
    for index, regime in enumerate(MIXED_TRAIN_V1[1:], start=1):
        runtimes[regime.regime_id] = _runtime(
            regime,
            selector_payload,
            device=device,
            seed=args.seed * 100 + index,
        )

    agent.batch_size = args.batch_size
    if agent.replay.capacity != args.buffer_size:
        agent.replay = type(agent.replay)(args.buffer_size)
    else:
        agent.replay.clear()
    agent.update_every = args.update_every
    agent.replay_sampling = "regime_mode_balanced"
    # A deployment checkpoint may retain a deliberately lagged target critic.
    # Mixed adaptation starts a new optimizer/replay process, so importing that
    # lag would create an unversioned initialization difference across seeds.
    synchronize_imported_target(agent)
    agent.step_count = 0
    agent.decision_count = 0
    agent.gradient_steps = 0
    agent.imitation_steps = 0
    agent.retention_steps = 0
    agent.spatial_distillation_steps = 0
    agent.set_training_phase("joint", clear_replay=False)
    agent.set_learning_rate_scales(
        main=args.learning_rate_scale,
        spatial=args.spatial_learning_rate_scale,
    )
    agent.Q_local.train()
    agent.Q_local.set_spatial_trainable(True)
    agent.Q_target.eval()

    epsilon_args = SimpleNamespace(
        joint_epsilon_start=args.epsilon,
        joint_epsilon_end=args.epsilon,
        joint_epsilon_warmup_decisions=0,
        joint_epsilon_decay_decisions=1,
        joint_teacher_mixture_start=args.teacher_mixture_start,
        joint_teacher_mixture_end=0.0,
        joint_teacher_mixture_warmup_decisions=(
            args.teacher_mixture_hold_decisions
        ),
        joint_teacher_mixture_decay_decisions=(
            args.teacher_mixture_decay_decisions
        ),
        joint_bc_start=args.bc_start,
        joint_bc_end=args.bc_end,
        joint_bc_warmup_decisions=args.bc_hold_decisions,
        joint_bc_decay_decisions=args.bc_decay_decisions,
        joint_spatial_freeze_decisions=args.spatial_freeze_decisions,
    )
    exploration_state = {
        "decisions": 0,
        "phase_decisions": {"joint": 0},
    }
    rng_state = capture_rng_state()
    reference = evaluate(
        agent,
        runtimes,
        MIXED_TRAIN_V1,
        seeds=args.validation_seeds,
        max_steps=args.max_steps,
        target_window=args.target_window,
    )
    restore_rng_state(rng_state)
    best_score = score_validation(reference, reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    exposure_counts = Counter()
    initial_payload = checkpoint_payload(
        agent,
        base_payload,
        args,
        manifest,
        episode=0,
        cycle=0,
        validation=reference,
        reference=reference,
        best_score=best_score,
        history=history,
        exposure_counts=exposure_counts,
        include_replay=False,
        checkpoint_kind="imported_initial",
    )
    atomic_torch_save(initial_payload, args.output_dir / "initial.pth")
    atomic_torch_save(
        {**initial_payload, "checkpoint_kind": "deployment_best"},
        args.output_dir / "best.pth",
    )
    print(
        f"Mixed hierarchy | device={device} | seed={args.seed} | "
        f"regimes={len(MIXED_TRAIN_V1)} | "
        f"episodes={args.episodes_per_regime * len(MIXED_TRAIN_V1)}",
        flush=True,
    )
    print(
        "Initial macro validation: "
        + json.dumps(reference["per_regime"], sort_keys=True),
        flush=True,
    )

    episode = 0
    rolling = deque(maxlen=20)
    for cycle in range(1, args.episodes_per_regime + 1):
        order = list(MIXED_TRAIN_V1)
        random.Random(args.seed * 1_000_003 + cycle).shuffle(order)
        for regime in order:
            episode += 1
            env, selector, encoder = runtimes[regime.regime_id]
            agent.bind_runtime(env, encoder, regime_id=regime.regime_id)
            instance_seed = (
                args.training_instance_seed_base
                + args.seed * 1_000_000
                + cycle
            )
            seed_everything(instance_seed)
            instance = env.sample_episode_instance(instance_seed)
            result = run_episode(
                agent,
                selector,
                env,
                instance,
                max_steps=args.max_steps,
                target_window=args.target_window,
                training=True,
                phase="joint",
                exploration_state=exploration_state,
                epsilon_args=epsilon_args,
                updates_per_macro=1,
                joint_td_warmup_decisions=args.joint_td_warmup_decisions,
            )
            if result["strict_method_success"] != 1.0:
                raise RuntimeError(
                    f"strict mixed training failure in {regime.regime_id}: "
                    f"{result['method_failure_reason']}"
                )
            exposure_counts[regime.regime_id] += 1
            rolling.append(result["return"])
            history.append(
                {
                    "episode": episode,
                    "cycle": cycle,
                    "regime_id": regime.regime_id,
                    "instance_seed": instance_seed,
                    "instance_id": instance.instance_id,
                    "return": result["return"],
                    "mean_absolute_error": result["mean_absolute_error"],
                    "steps": result["steps"],
                    "macro_count": result["macro_count"],
                    "relocations": result["obstructive_moves"],
                    "loss": result["loss"],
                    "epsilon": agent.epsilon,
                }
            )

        should_validate = bool(
            cycle % args.eval_every_cycles == 0
            or cycle == args.episodes_per_regime
        )
        if should_validate:
            rng_state = capture_rng_state()
            validation = evaluate(
                agent,
                runtimes,
                MIXED_TRAIN_V1,
                seeds=args.validation_seeds,
                max_steps=args.max_steps,
                target_window=args.target_window,
            )
            restore_rng_state(rng_state)
            score = score_validation(validation, reference)
            improved = score > best_score
            if improved:
                best_score = score
                best_payload = checkpoint_payload(
                    agent,
                    base_payload,
                    args,
                    manifest,
                    episode=episode,
                    cycle=cycle,
                    validation=validation,
                    reference=reference,
                    best_score=best_score,
                    history=history,
                    exposure_counts=exposure_counts,
                    include_replay=False,
                    checkpoint_kind="deployment_best",
                )
                atomic_torch_save(best_payload, args.output_dir / "best.pth")
            latest = checkpoint_payload(
                agent,
                base_payload,
                args,
                manifest,
                episode=episode,
                cycle=cycle,
                validation=validation,
                reference=reference,
                best_score=best_score,
                history=history,
                exposure_counts=exposure_counts,
                include_replay=True,
                checkpoint_kind="resumable_latest",
            )
            atomic_torch_save(latest, args.output_dir / "latest.pth")
            print(
                f"Cycle {cycle:3d} | Ep {episode:4d} | "
                f"TrainR {np.mean(rolling):8.2f} | "
                f"Replay {len(agent.replay):5d} | "
                f"Loss {history[-1]['loss']} | improved={int(improved)} | "
                f"score={score}",
                flush=True,
            )
            for key, value in validation["per_regime"].items():
                print(
                    f"  {key}: R={value['mean_return']:.2f} "
                    f"MAE={value['mean_absolute_error']:.3f} "
                    f"strict={value['strict_method_success_rate']:.3f} "
                    f"reloc={value['total_relocations']}",
                    flush=True,
                )

    summary = {
        "method": MIXED_METHOD,
        "training_seed": args.seed,
        "training_manifest": manifest,
        "episodes_per_regime": args.episodes_per_regime,
        "total_training_episodes": episode,
        "exposure_counts": dict(exposure_counts),
        "best_score": list(best_score),
        "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
        "latest_checkpoint": str((args.output_dir / "latest.pth").resolve()),
        "history": history,
    }
    (args.output_dir / "training-summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: summary[key] for key in (
        "training_seed", "total_training_episodes", "exposure_counts",
        "best_score", "best_checkpoint", "latest_checkpoint")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
