#!/usr/bin/env python3
"""Train one conservative VCG 1.1-anchored preference residual.

This development trainer never changes the authenticated VCG 1.1 controller.
Lambda zero always executes that controller directly.  The two small residual
towers are initialized so that positive-lambda deployment initially reproduces
the completed nested handling pilot, then are adapted on fresh dynamic 5x5
episodes with a balanced fixed-per-episode lambda schedule.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_preference_conditioned as common
from train_viability_graph_smdp import resolve_device, seed_everything
from vcg_v11_anchored_preference import (
    AnchoredPreferenceAgent,
    AnchoredPreferenceConfig,
)


TRAINING_PROTOCOL = "vcg_v1_1_anchored_preference_seed0_development_v1"
TRAINER_SCHEMA_VERSION = 1
MODEL_SEED = 0
DEFAULT_EPISODES = 200
TRAIN_SEED_BASE = 53_000_000
PREFERENCE_GRID = (0.0, 0.025, 0.05, 0.1, 0.2)
PREFERENCE_SCHEDULE_SEED = 531_000_000
CHECKPOINT_EVERY_EPISODES = 5
LATEST_CHECKPOINT_ROLE = "resumable_latest_anchored_development_state"
TERMINAL_CHECKPOINT_ROLE = "fixed_terminal_anchored_development_model"
COST_RELATIVE_PATH = Path(
    "results/vcg-v1-1-nested-handling-seed0-85k-development/handling-cost-head.pth"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            common._json_safe(dict(value)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_preference_schedule(episodes: int) -> tuple[dict, ...]:
    if isinstance(episodes, bool) or int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    import random

    rng = random.Random(PREFERENCE_SCHEDULE_SEED)
    result = []
    for start in range(0, int(episodes), len(PREFERENCE_GRID)):
        values = list(PREFERENCE_GRID)
        rng.shuffle(values)
        for offset, value in enumerate(values):
            index = start + offset
            if index >= episodes:
                break
            result.append(
                {
                    "episode_number": index + 1,
                    "preference_block_number": start // len(PREFERENCE_GRID) + 1,
                    "position_in_preference_block": offset + 1,
                    "behavior_lambda": float(value),
                    "block_permutation": tuple(float(x) for x in values),
                    "schedule_rng_seed": PREFERENCE_SCHEDULE_SEED,
                    "sampling_without_replacement_within_block": True,
                    "fixed_for_complete_episode": True,
                }
            )
    return tuple(result)


def _authenticated_inputs(project_root: Path, device: torch.device):
    arm, _latest, _sources, q_digest, _records = pilot._authenticate_inputs(
        project_root
    )
    cost_path = project_root / COST_RELATIVE_PATH
    if not cost_path.is_file():
        raise FileNotFoundError(f"missing completed handling head: {cost_path}")
    base = pilot._fresh_base(arm, device)
    cost = pilot._load_bound_cost(
        cost_path, arm, device=device, config=base.config
    )
    return arm, base, cost, q_digest, cost_path


def _agent_config(base, args) -> AnchoredPreferenceConfig:
    return AnchoredPreferenceConfig(
        feature_dim=(
            3 * base.config.graph_embedding_dim
            + base.config.action_embedding_dim
        ),
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=max(PREFERENCE_GRID),
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_size,
        update_every=1,
        target_update_every=args.target_update_every,
        grad_clip=base.config.grad_clip,
        huber_delta=base.config.huber_delta,
        operational_loss_weight=1.0,
        handling_loss_weight=args.handling_loss_weight,
    )


def _contract(
    args,
    *,
    arm,
    q_digest: str,
    cost_path: Path,
    config: AnchoredPreferenceConfig,
    search_config,
    liveness_rule,
    schedule: Sequence[Mapping],
) -> dict:
    semantic = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "development_only": True,
        "model_seed": MODEL_SEED,
        "episodes": int(args.episodes),
        "train_instance_seed_base": TRAIN_SEED_BASE,
        "fresh_training_instances": True,
        "protected_evaluation_panels_opened": False,
        "preference_grid": PREFERENCE_GRID,
        "preference_schedule_seed": PREFERENCE_SCHEDULE_SEED,
        "preference_schedule": tuple(dict(item) for item in schedule),
        "preference_fixed_for_complete_episode": True,
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": 8,
            "max_steps": int(args.max_steps),
        },
        "agent_config": config.to_dict(),
        "updates_per_macro": int(args.updates_per_macro),
        "epsilon_schedule": {
            "start": float(args.epsilon_start),
            "end": float(args.epsilon_end),
            "warmup_decisions": 0,
            "decay_decisions": int(args.epsilon_decay_decisions),
        },
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "base_checkpoint_sha256": arm.checkpoint_sha256,
        "base_policy_digest": arm.deployment_policy_digest,
        "base_q_state_sha256": q_digest,
        "source_cost_checkpoint": str(cost_path.resolve()),
        "source_cost_sha256": _sha256(cost_path),
        "base_operational_controller_frozen": True,
        "lambda_zero_direct_vcg_v1_1_delegation": True,
        "positive_lambda_initial_policy_equals_nested_vcg": True,
        "operational_residual_zero_initialized": True,
        "handling_head_warm_started_exactly": True,
        "handling_objective": "undiscounted_physical_rehandles",
        "operational_objective": "vcg_v1_1_primitive_time_discounted_dense_return",
        "shared_online_selected_successor_action_for_vector_backup": True,
        "cached_frozen_v1_features_in_replay": True,
        "teacher_policy_queries": False,
        "exact_verifier_authoritative": True,
        "unsafe_unknown_fail_closed": True,
        "interval_validation": False,
        "checkpoint_selection": False,
        "terminal_checkpoint_rule": "fixed_episode_budget_only",
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def _checkpoint(
    agent: AnchoredPreferenceAgent,
    *,
    arm,
    cost_path: Path,
    role: str,
    resumable: bool,
    completed: int,
    contract: Mapping,
    history: Sequence[Mapping],
) -> dict:
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": role,
        "trainer_resumable": resumable,
        "fixed_terminal_checkpoint": role == TERMINAL_CHECKPOINT_ROLE,
        "completed_training_episodes": completed,
        "next_training_episode": completed + 1,
        "resume_contract": dict(contract),
        "resume_contract_sha256": contract["contract_sha256"],
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=arm.checkpoint_sha256,
            base_policy_digest=arm.deployment_policy_digest,
            source_cost_sha256=_sha256(cost_path),
            include_replay=resumable,
        ),
        "training_history": tuple(history),
        "global_rng_state": common._capture_global_rng_state() if resumable else None,
        "development_only": True,
        "evaluation_panels_opened": False,
    }


def _validate_checkpoint(payload: Mapping, contract: Mapping, *, resumable: bool):
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": (
            LATEST_CHECKPOINT_ROLE if resumable else TERMINAL_CHECKPOINT_ROLE
        ),
        "trainer_resumable": resumable,
        "fixed_terminal_checkpoint": not resumable,
        "development_only": True,
        "evaluation_panels_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"incompatible anchored checkpoint: {mismatch}")
    if payload.get("resume_contract") != contract:
        raise ValueError("anchored resume contract changed")
    completed = int(payload.get("completed_training_episodes", -1))
    if len(tuple(payload.get("training_history", ()))) != completed:
        raise ValueError("anchored checkpoint history clock changed")
    if not resumable and completed != int(contract["episodes"]):
        raise ValueError("terminal anchored checkpoint is not at fixed budget")


def _summary(history: Sequence[Mapping]) -> dict:
    result = common.summarize_training_runs(history)
    result["all_strict_safe_complete"] = all(
        bool(row["strict_method_success"]) for row in history
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--stop-after-episode", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--target-update-every", type=int, default=200)
    parser.add_argument("--handling-loss-weight", type=float, default=1.0)
    parser.add_argument("--updates-per-macro", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-decisions", type=int, default=1_500)
    return parser


def _validate_args(args) -> None:
    for name in (
        "episodes", "log_every", "max_steps", "batch_size", "replay_size",
        "target_update_every", "updates_per_macro", "epsilon_decay_decisions",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.replay_size < args.batch_size:
        raise ValueError("replay size must be at least batch size")
    for name in ("learning_rate", "handling_loss_weight"):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("epsilon_start", "epsilon_end"):
        if not 0 <= float(getattr(args, name)) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.stop_after_episode is not None and not 1 <= args.stop_after_episode <= args.episodes:
        raise ValueError("stop episode must lie within the fixed budget")


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    args.output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parent
    device = resolve_device(args.device)
    arm, base, cost, q_digest, cost_path = _authenticated_inputs(
        project_root, device
    )
    config = _agent_config(base, args)
    search_config = benchmark._search_config(arm.payload)
    liveness_rule = benchmark._liveness_rule(arm.payload)
    schedule = build_preference_schedule(args.episodes)
    contract = _contract(
        args,
        arm=arm,
        q_digest=q_digest,
        cost_path=cost_path,
        config=config,
        search_config=search_config,
        liveness_rule=liveness_rule,
        schedule=schedule,
    )
    latest_path = args.output_dir / "latest.pth"
    terminal_path = args.output_dir / "terminal.pth"
    summary_path = args.output_dir / "training-summary.json"

    if args.resume_existing:
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(payload, contract, resumable=True)
        fresh_base = pilot._fresh_base(arm, device)
        agent = AnchoredPreferenceAgent.from_checkpoint(
            payload["agent_checkpoint"],
            base_agent=fresh_base,
            expected_base_checkpoint_sha256=arm.checkpoint_sha256,
            expected_base_policy_digest=arm.deployment_policy_digest,
            expected_source_cost_sha256=_sha256(cost_path),
            seed=MODEL_SEED,
            resumable=True,
        )
        common._optimizer_to(agent.optimizer, device)
        completed = int(payload["completed_training_episodes"])
        history = list(payload["training_history"])
        common._restore_global_rng_state(payload["global_rng_state"])
        print(f"Resumed anchored pilot after episode {completed}", flush=True)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                "fresh anchored training refuses a nonempty directory; use "
                "--resume-existing or choose another directory"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(MODEL_SEED)
        agent = AnchoredPreferenceAgent(
            base,
            config=config,
            seed=MODEL_SEED,
            warm_start_cost=cost,
            epsilon=args.epsilon_start,
        )
        completed = 0
        history = []

    stop_at = args.episodes
    if args.stop_after_episode is not None:
        stop_at = min(stop_at, args.stop_after_episode)
    if completed > stop_at:
        raise ValueError("resume checkpoint is already beyond requested stop")
    print(
        "Anchored VCG pilot | seed=0 | "
        f"device={device} | episodes={args.episodes} | "
        f"lambda_grid={PREFERENCE_GRID} | frozen_v1=true | raw_QN=true",
        flush=True,
    )
    env = benchmark._make_env(arm.payload)
    for index in range(completed, stop_at):
        episode_number = index + 1
        record = schedule[index]
        value = float(record["behavior_lambda"])
        env.current_episode = episode_number
        run = common.run_preference_episode(
            agent,
            env,
            instance_seed=TRAIN_SEED_BASE + index,
            preference_lambda=value,
            preference_schedule_record=record,
            max_steps=args.max_steps,
            search_config=search_config,
            liveness_rule=liveness_rule,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_decisions=args.epsilon_decay_decisions,
            epsilon_warmup_decisions=0,
            updates_per_macro=args.updates_per_macro,
        )
        run.update(
            {
                "episode_number": episode_number,
                "architecture": "vcg_v1_1_anchored_preference_residual",
                "model_seed": MODEL_SEED,
            }
        )
        history.append(run)
        completed = episode_number
        agent.base_agent.decision_log.clear()
        if episode_number == 1 or episode_number % args.log_every == 0:
            recent = history[-min(10, len(history)):]
            print(
                f"Ep {episode_number:4d} | Lambda {value:5.3f} | "
                f"TrainR {fmean(float(x['return']) for x in recent):8.2f} | "
                f"Strict {int(run['strict_method_success'])} | "
                f"Reh {run['physical_rehandles']:3d} | "
                f"Replay {len(agent.replay):5d} | Grad {agent.gradient_steps:6d} | "
                f"Eps {agent.epsilon:.3f}",
                flush=True,
            )
        if (
            episode_number % CHECKPOINT_EVERY_EPISODES == 0
            or episode_number == stop_at
            or episode_number == args.episodes
        ):
            latest = _checkpoint(
                agent,
                arm=arm,
                cost_path=cost_path,
                role=LATEST_CHECKPOINT_ROLE,
                resumable=True,
                completed=completed,
                contract=contract,
                history=history,
            )
            common._atomic_torch_save(latest, latest_path)
            common._atomic_json_save(
                {
                    "status": "complete" if completed == args.episodes else "paused",
                    "completed_training_episodes": completed,
                    "total_training_episodes": args.episodes,
                    "training_summary": _summary(history),
                    "agent_audit": agent.audit(),
                    "latest_checkpoint": str(latest_path),
                    "terminal_checkpoint": None,
                },
                summary_path,
            )

    terminal_sha = None
    if completed == args.episodes:
        if terminal_path.is_file():
            terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
            _validate_checkpoint(terminal, contract, resumable=False)
        else:
            terminal = _checkpoint(
                agent,
                arm=arm,
                cost_path=cost_path,
                role=TERMINAL_CHECKPOINT_ROLE,
                resumable=False,
                completed=completed,
                contract=contract,
                history=history,
            )
            common._atomic_torch_save(terminal, terminal_path)
        terminal_sha = _sha256(terminal_path)
    result = {
        "status": "complete" if completed == args.episodes else "paused",
        "training_protocol": TRAINING_PROTOCOL,
        "model_seed": MODEL_SEED,
        "completed_training_episodes": completed,
        "total_training_episodes": args.episodes,
        "training_summary": _summary(history),
        "agent_audit": agent.audit(),
        "resume_contract": contract,
        "latest_checkpoint": str(latest_path),
        "terminal_checkpoint": str(terminal_path) if terminal_sha else None,
        "terminal_checkpoint_sha256": terminal_sha,
        "fixed_terminal_checkpoint": terminal_sha is not None,
        "checkpoint_selection_used": False,
        "evaluation_panels_opened": False,
    }
    common._atomic_json_save(result, summary_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_training_episodes": completed,
                "terminal_checkpoint": result["terminal_checkpoint"],
                "summary": str(summary_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
