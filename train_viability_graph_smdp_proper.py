"""Resumable, validation-selected training for the exact-safe VCG-SMDP.

This is deliberately separate from ``train_viability_graph_smdp.py``.  The
latter remains the calibration-only smoke harness that produced the early VCG
artifacts.  This entry point adds the lifecycle needed for a real training
replicate while retaining the same action interface, environment reward,
exact verifier, SMDP return, and teacher-free controller.

The script trains on unique EpisodeInstances, selects a deployment checkpoint
on a fixed development-validation panel, and never opens either declared test
panel.  ``latest.pth`` is resumable; ``best.pth`` is a frozen deployment
candidate and intentionally excludes replay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean
from typing import Optional, Sequence

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    ViabilityCertificateCache,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from train_viability_graph_smdp import (
    CERTIFICATE_SCOPE,
    NO_FALLBACK_CONTRACT,
    SEALED_STRESS_V1_HOLDOUT_SEEDS,
    SMDP_RETURN_CONTRACT,
    epsilon_at,
    resolve_device,
    run_calibration_episode,
    seed_everything,
    summarize_runs,
)
from viability_graph_hierarchy import (
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
)


TRAINING_PROTOCOL = "vcg_smdp_contention_proper_training_v1"
TRAINER_SCHEMA_VERSION = 1
CHECKPOINT_SELECTION_VERSION = (
    "strict_completion_return_mae_relocation_earlier_lexicographic_v1"
)
DEFAULT_TOTAL_EPISODES = 500
TRAIN_SEED_ORIGIN = 30_000_000
TRAIN_SEED_STRIDE = 1_000_000
DEFAULT_VALIDATION_SEEDS = tuple(range(75_000, 75_020))
SEALED_IN_REGIME_TEST_SEEDS = frozenset(range(77_000, 77_050))
DEVELOPMENT_ONLY_EPISODE_SEEDS = frozenset(
    tuple(range(68_000, 68_003)) + tuple(range(73_000, 75_000))
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


def _build_graph_config(args) -> ViabilityGraphConfig:
    return ViabilityGraphConfig(
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


def _new_env(args) -> SmallRoomsEnv:
    return SmallRoomsEnv(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        number_blocks=args.number_blocks,
        choose_storage=False,
        arrival_rate=args.arrival_rate,
        proc_mean=args.proc_mean,
    )


def _resume_contract(
    args,
    *,
    train_seed_base: int,
    graph_config: ViabilityGraphConfig,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
) -> dict:
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
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
        },
        "max_steps": args.max_steps,
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
        "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
        "exact_verifier_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "certificate_scope": CERTIFICATE_SCOPE,
        "smdp_return_contract": SMDP_RETURN_CONTRACT,
        "no_fallback_contract": NO_FALLBACK_CONTRACT,
    }


def _compact_run(run: dict) -> dict:
    """Drop decision-level payloads while preserving every reported metric."""

    return {name: value for name, value in run.items() if name != "decisions"}


def _training_agent_signature(agent: ViabilityGraphHierarchyAgent) -> tuple:
    """Cheap state signature used to prove validation did not touch training."""

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
    agent: ViabilityGraphHierarchyAgent,
    args,
    *,
    device: torch.device,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
) -> tuple[list[dict], dict]:
    """Evaluate a frozen clone without mutating any training state or RNG."""

    training_signature = _training_agent_signature(agent)
    global_rng = _capture_global_rng_state()
    checkpoint = agent.checkpoint(
        include_replay=False,
        training_protocol=TRAINING_PROTOCOL,
        validation_clone_source=True,
    )
    evaluator = ViabilityGraphHierarchyAgent.from_checkpoint(
        checkpoint,
        device=device,
        resumable=False,
        seed=args.model_seed,
    )
    evaluator.set_epsilon(0.0)
    evaluator.Q_local.eval()
    evaluator.Q_target.eval()
    initial_decisions = evaluator.decision_count
    env = _new_env(args)
    runs = []
    try:
        for index, instance_seed in enumerate(args.validation_seeds):
            env.current_episode = index + 1
            # Cache reuse is retained within an episode but never leaks across
            # validation instances or into training.
            run = run_calibration_episode(
                evaluator,
                env,
                instance_seed=int(instance_seed),
                training=False,
                max_steps=args.max_steps,
                search_config=search_config,
                liveness_rule=liveness_rule,
                certificate_cache=ViabilityCertificateCache(),
                epsilon_start=0.0,
                epsilon_end=0.0,
                epsilon_decay_decisions=1,
                updates_per_macro=1,
                state_prioritizer=None,
                epsilon_warmup_decisions=0,
            )
            run["protocol"] = TRAINING_PROTOCOL
            run["split"] = "checkpoint_selection_validation"
            runs.append(_compact_run(run))
    finally:
        _restore_global_rng_state(global_rng)
    if _training_agent_signature(agent) != training_signature:
        raise RuntimeError("validation mutated the training agent")
    summary = summarize_runs(runs)
    summary.update(
        {
            "instance_seeds": tuple(args.validation_seeds),
            "deployment_decisions": (
                evaluator.decision_count - initial_decisions
            ),
            "evaluation_epsilon": 0.0,
            "replay_size": len(evaluator.replay),
            "q_local_training_mode": bool(evaluator.Q_local.training),
            "fresh_certificate_cache_per_episode": True,
            "viability_critic_enabled": False,
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
    delivery_count = sum(int(item.get("delivery_count", 0)) for item in runs)
    relocations = sum(
        int(item.get("audit", {}).get("relocations", 0)) for item in runs
    )
    relocation_rate = (
        100.0 * relocations / delivery_count
        if delivery_count > 0
        else math.inf
    )
    mae = float(summary["mean_absolute_error"])
    if not math.isfinite(mae):
        mae = math.inf
    mean_return_per_block = float(summary["mean_return"]) / number_blocks
    score = (
        float(summary["strict_method_success_rate"]),
        float(summary["strict_method_success_rate"]),
        float(summary["success_rate"]),
        float(summary["success_rate"]),
        mean_return_per_block,
        -mae,
        -relocation_rate,
        -int(checkpoint_episode),
    )
    no_failures = not summary.get("method_failures")
    all_delivered = all(
        int(item.get("delivery_count", 0)) == int(number_blocks)
        for item in runs
    )
    authoritative = all(
        bool(item.get("audit", {}).get("exact_verifier_authoritative"))
        and not bool(item.get("audit", {}).get("baseline_viability_teacher"))
        and not bool(item.get("audit", {}).get("baseline_policy_query"))
        for item in runs
    )
    eligible = bool(
        float(summary["strict_method_success_rate"]) == 1.0
        and float(summary["success_rate"]) == 1.0
        and no_failures
        and all_delivered
        and authoritative
    )
    return {
        "checkpoint_episode": int(checkpoint_episode),
        "selection_score": score,
        "selection_score_fields": (
            "min_regime_strict_success_rate",
            "macro_mean_strict_success_rate",
            "min_regime_completion_rate",
            "macro_mean_completion_rate",
            "macro_mean_return_per_manifest_block",
            "negative_macro_mean_absolute_error",
            "negative_relocations_per_100_deliveries",
            "negative_checkpoint_episode",
        ),
        "deployment_eligible": eligible,
        "eligibility_contract": (
            "100pct_strict_and_complete_each_validation_regime_no_method_"
            "failure_exact_authority_no_fallback_teacher_or_critic_v1"
        ),
        "mean_return_per_block": mean_return_per_block,
        "relocations_per_100_deliveries": relocation_rate,
        "summary": summary,
        "runs": tuple(runs),
    }


def _score_tuple(record: Optional[dict]) -> Optional[tuple]:
    if record is None:
        return None
    return tuple(float(value) for value in record["selection_score"])


def _build_checkpoint(
    agent: ViabilityGraphHierarchyAgent,
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
    return agent.checkpoint(
        include_replay=include_replay,
        protocol=TRAINING_PROTOCOL,
        training_protocol=TRAINING_PROTOCOL,
        trainer_schema_version=TRAINER_SCHEMA_VERSION,
        trainer_resumable=bool(resumable),
        completed_training_episodes=int(completed_episodes),
        next_training_episode=int(completed_episodes) + 1,
        next_train_instance_seed=int(train_seed_base + completed_episodes),
        resume_contract=resume_contract,
        instance_regime="contention_proper_training",
        model_seed=int(resume_contract["model_seed"]),
        train_instance_seeds=tuple(
            range(train_seed_base, train_seed_base + completed_episodes)
        ),
        environment=dict(resume_contract["environment"]),
        viability_search=dict(resume_contract["search_config"]),
        liveness_rule=dict(resume_contract["liveness_rule"]),
        smdp_return_contract=SMDP_RETURN_CONTRACT,
        no_fallback_contract=NO_FALLBACK_CONTRACT,
        certificate_scope=CERTIFICATE_SCOPE,
        complete_episode_certificate=False,
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
        sealed_in_regime_test_seeds=tuple(sorted(SEALED_IN_REGIME_TEST_SEEDS)),
        sealed_stress_test_seeds=tuple(
            sorted(SEALED_STRESS_V1_HOLDOUT_SEEDS)
        ),
        test_panels_opened=False,
    )


def _validate_resume_payload(payload: dict, contract: dict) -> None:
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "trainer_resumable": True,
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
        raise ValueError(f"incompatible resume checkpoint: {mismatches!r}")
    if payload.get("resume_contract") != contract:
        raise ValueError("resume contract mismatch")
    state = payload.get("agent_state", {})
    required = ("optimizer", "replay", "rng_state")
    missing = tuple(name for name in required if name not in state)
    if missing:
        raise ValueError(f"resumable checkpoint is missing {missing!r}")
    replay = state["replay"]
    capacity = int(replay["capacity"])
    if len(replay["memory"]) > capacity:
        raise ValueError("resume replay exceeds its declared capacity")
    for name in (
        "transition_count",
        "decision_count",
        "gradient_steps",
        "target_updates",
    ):
        if int(state.get(name, 0)) < 0:
            raise ValueError(f"resume agent clock {name!r} is negative")
    if int(state.get("transition_count", 0)) < len(replay["memory"]):
        raise ValueError("resume transition clock is smaller than replay")


def _result_payload(
    args,
    *,
    completed_episodes: int,
    train_seed_base: int,
    contract: dict,
    train_history: Sequence[dict],
    validation_history: Sequence[dict],
    best_record: Optional[dict],
    best_checkpoint_sha256: Optional[str],
    agent: ViabilityGraphHierarchyAgent,
) -> dict:
    complete = int(completed_episodes) == int(args.total_episodes)
    canonical = bool(
        complete
        and args.total_episodes == DEFAULT_TOTAL_EPISODES
        and tuple(args.validation_seeds) == DEFAULT_VALIDATION_SEEDS
        and train_seed_base == _derived_train_seed_base(args.model_seed)
        and args.grid_rows == 5
        and args.grid_cols == 5
        and args.number_blocks == 8
        and float(args.arrival_rate) == 10.0
        and args.proc_mean == 80
        and args.max_steps == 2_000
        and contract["graph_config"] == ViabilityGraphConfig().to_dict()
        and contract["search_config"]
        == asdict(ViabilitySearchConfig(max_nodes=20_000))
        and contract["liveness_rule"] == asdict(BoundedEventDeferRule())
        and contract["updates_per_macro"] == 1
        and contract["epsilon_schedule"]
        == {
            "start": 0.90,
            "end": 0.05,
            "warmup_decisions": 0,
            "decay_decisions": 10_000,
        }
        and contract["validation_every_episodes"] == 25
    )
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "status": "complete" if complete else "paused",
        "canonical_500_episode_replication_complete": canonical,
        "formal_test_authorized": False,
        "formal_test_authorization_note": (
            "This trainer never opens a test panel. Freeze all three selected "
            "training-seed checkpoints before running the separate evaluator."
        ),
        "test_panels_opened": False,
        "completed_training_episodes": int(completed_episodes),
        "total_training_episodes": int(args.total_episodes),
        "model_seed": int(args.model_seed),
        "train_instance_seed_base": int(train_seed_base),
        "last_completed_train_instance_seed": (
            int(train_seed_base + completed_episodes - 1)
            if completed_episodes
            else None
        ),
        "validation_instance_seeds": tuple(args.validation_seeds),
        "sealed_in_regime_test_seeds": tuple(
            sorted(SEALED_IN_REGIME_TEST_SEEDS)
        ),
        "sealed_stress_test_seeds": tuple(
            sorted(SEALED_STRESS_V1_HOLDOUT_SEEDS)
        ),
        "resume_contract": contract,
        "training": {
            "runs": tuple(train_history),
            "summary": summarize_runs(train_history),
        },
        "validation_history": tuple(validation_history),
        "best_validation_record": best_record,
        "deployment_checkpoint_eligible": bool(
            best_record and best_record["deployment_eligible"]
        ),
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "agent_audit": agent.audit(include_decisions=False),
        "latest_checkpoint": str((args.output_dir / "latest.pth").resolve()),
        "best_checkpoint": (
            str((args.output_dir / "best.pth").resolve())
            if best_record is not None
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Proper resumable training for the exact-safe teacher-free VCG-SMDP"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--total-episodes", type=int, default=500)
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
    parser.add_argument(
        "--stop-after-episode",
        type=int,
        help="clean episode-boundary pause; total training contract is unchanged",
    )

    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--proc-mean", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

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
    parser.add_argument("--epsilon-start", type=float, default=0.90)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-warmup-decisions", type=int, default=0)
    parser.add_argument("--epsilon-decay-decisions", type=int, default=10_000)

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
    parser.add_argument("--max-nonprogress-recovery-decisions", type=int, default=2)
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
        raise ValueError("episode budget exceeds the reserved model-seed namespace")
    if args.replay_size < args.batch_size:
        raise ValueError("--replay-size must be at least --batch-size")
    if args.message_passing_steps < 0:
        raise ValueError("--message-passing-steps must be non-negative")
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
        raise ValueError("--stop-after-episode must lie inside the training budget")
    if not math.isfinite(args.arrival_rate) or args.arrival_rate < 0.0:
        raise ValueError("--arrival-rate must be finite and non-negative")

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
    protected = (
        SEALED_IN_REGIME_TEST_SEEDS | SEALED_STRESS_V1_HOLDOUT_SEEDS
    )
    opened = (train_seeds | validation_seeds).intersection(protected)
    if opened:
        raise ValueError(
            f"training runner refuses sealed test seeds: {tuple(sorted(opened))}"
        )
    development_overlap = (train_seeds | validation_seeds).intersection(
        DEVELOPMENT_ONLY_EPISODE_SEEDS
    )
    if development_overlap:
        raise ValueError(
            "proper training/validation may not reuse development-only "
            f"seeds: {tuple(sorted(development_overlap))}"
        )
    return train_seed_base


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    train_seed_base = _validate_args(args)
    device = resolve_device(args.device)
    graph_config = _build_graph_config(args)
    search_config = _build_search_config(args)
    liveness_rule = _build_liveness_rule(args)
    contract = _resume_contract(
        args,
        train_seed_base=train_seed_base,
        graph_config=graph_config,
        search_config=search_config,
        liveness_rule=liveness_rule,
    )

    latest_path = args.output_dir / "latest.pth"
    best_path = args.output_dir / "best.pth"
    result_path = args.output_dir / "training-summary.json"
    if args.resume is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                "fresh training refuses a nonempty output directory; use "
                "--resume with its latest.pth or choose a new directory"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(args.model_seed)
        agent = ViabilityGraphHierarchyAgent(
            config=graph_config,
            seed=args.model_seed,
            device=device,
            epsilon=args.epsilon_start,
        )
        completed = 0
        train_history: list[dict] = []
        validation_history: list[dict] = []
        best_record = None
        best_checkpoint_sha256 = None
    else:
        if args.resume.resolve().parent != args.output_dir.resolve():
            raise ValueError("--resume must belong to --output-dir")
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("resume checkpoint must contain a mapping")
        _validate_resume_payload(payload, contract)
        agent = ViabilityGraphHierarchyAgent.from_checkpoint(
            payload,
            device=device,
            resumable=True,
            seed=args.model_seed,
        )
        _optimizer_to(agent.optimizer, device)
        completed = int(payload["completed_training_episodes"])
        if payload.get("next_train_instance_seed") != train_seed_base + completed:
            raise ValueError("resume next EpisodeInstance seed is inconsistent")
        if completed > args.total_episodes:
            raise ValueError("resume checkpoint exceeds the planned budget")
        train_history = list(payload.get("training_history", ()))
        validation_history = list(payload.get("validation_history", ()))
        if len(train_history) != completed:
            raise ValueError("resume training history length is inconsistent")
        best_record = payload.get("best_validation_record")
        best_checkpoint_sha256 = payload.get("best_checkpoint_sha256")
        if best_record is not None:
            if not best_path.is_file() or not best_checkpoint_sha256:
                raise ValueError("resume is missing its authenticated best.pth")
            if _sha256(best_path) != best_checkpoint_sha256:
                raise ValueError("resume best.pth SHA-256 mismatch")
        _restore_global_rng_state(payload["global_rng_state"])
        print(
            f"Resumed {args.resume} at completed episode {completed}",
            flush=True,
        )

    print(
        "Proper VCG-SMDP | "
        f"device={device} | model_seed={args.model_seed} | "
        f"episodes={args.total_episodes} | train_seed_base={train_seed_base} | "
        f"validation={len(args.validation_seeds)}x every {args.eval_every} | "
        "exact_full=true | critic=false | teacher=false",
        flush=True,
    )
    env = _new_env(args)
    stop_at = args.total_episodes
    if args.stop_after_episode is not None:
        stop_at = min(stop_at, args.stop_after_episode)
    if completed > stop_at:
        raise ValueError("resume is already beyond --stop-after-episode")

    for episode_index in range(completed, stop_at):
        episode_number = episode_index + 1
        instance_seed = train_seed_base + episode_index
        env.current_episode = episode_number
        run = run_calibration_episode(
            agent,
            env,
            instance_seed=instance_seed,
            training=True,
            max_steps=args.max_steps,
            search_config=search_config,
            liveness_rule=liveness_rule,
            # Deliberately fresh: only within-episode exact analyses are reused.
            certificate_cache=ViabilityCertificateCache(),
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_decisions=args.epsilon_decay_decisions,
            updates_per_macro=args.updates_per_macro,
            state_prioritizer=None,
            epsilon_warmup_decisions=args.epsilon_warmup_decisions,
        )
        run["protocol"] = TRAINING_PROTOCOL
        run["split"] = "training"
        train_history.append(_compact_run(run))
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
            guarded_fraction = (
                agent.selection_sources.get(
                    "exact_recovery_witness_guard", 0
                )
                / max(agent.decision_count, 1)
            )
            print(
                f"Ep {episode_number:4d} | "
                f"TrainR {fmean(item['return'] for item in recent):8.2f} | "
                f"Strict {int(run['strict_method_success'])} | "
                f"Dec {run['macro_decisions']:3d} | "
                f"Replay {len(agent.replay):5d} | "
                f"Grad {agent.gradient_steps:6d} | "
                f"Target {agent.target_updates:3d} | "
                f"Guard {guarded_fraction:.3f} | "
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
                f"Validation Ep {episode_number:4d} | "
                f"R {validation_summary['mean_return']:8.2f} | "
                f"Strict {validation_summary['strict_method_success_rate']:.3f} | "
                f"Complete {validation_summary['success_rate']:.3f} | "
                f"MAE {validation_summary['mean_absolute_error']:.3f} | "
                f"Reloc/100 {record['relocations_per_100_deliveries']:.2f} | "
                f"Eligible {int(record['deployment_eligible'])}",
                flush=True,
            )
            if best_record is None or _score_tuple(record) > _score_tuple(best_record):
                best_record = record
                best_payload = _build_checkpoint(
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
                print(
                    f"New best at episode {episode_number} | "
                    f"sha256={best_checkpoint_sha256}",
                    flush=True,
                )

        checkpoint_due = bool(
            validation_due
            or episode_number % args.checkpoint_every == 0
            or episode_number == stop_at
        )
        if checkpoint_due:
            latest = _build_checkpoint(
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
            result = _result_payload(
                args,
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
        raise RuntimeError("no training episode was completed")
    result = _result_payload(
        args,
        completed_episodes=completed,
        train_seed_base=train_seed_base,
        contract=contract,
        train_history=train_history,
        validation_history=validation_history,
        best_record=best_record,
        best_checkpoint_sha256=best_checkpoint_sha256,
        agent=agent,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "status": result["status"],
                    "completed_training_episodes": completed,
                    "best_episode": (
                        best_record["checkpoint_episode"]
                        if best_record is not None
                        else None
                    ),
                    "deployment_checkpoint_eligible": result[
                        "deployment_checkpoint_eligible"
                    ],
                    "best_checkpoint": result["best_checkpoint"],
                    "latest_checkpoint": result["latest_checkpoint"],
                    "summary": str(result_path.resolve()),
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
