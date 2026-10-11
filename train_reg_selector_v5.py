#!/usr/bin/env python3
"""Train REG-v5 under the neutral matched-instance Track A protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.track_a import TRACK_A_REG_SELECTOR_V5, run_track_a_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the assignment-epoch Double-DQN REG-v5 selector"
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-replay-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--updates-per-episode", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--truncation-penalty", type=float, default=0.0)
    parser.add_argument("--target-update-every", type=int, default=250)
    parser.add_argument("--ema-tau", type=float, default=0.01)
    parser.add_argument("--epsilon-start", type=float, default=0.90)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-warmup", type=int, default=300)
    parser.add_argument("--epsilon-decay", type=int, default=10000)
    parser.add_argument("--block-embedding-dim", type=int, default=64)
    parser.add_argument("--candidate-embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--validation-seeds",
        nargs="+",
        type=int,
        default=[10_000, 10_001, 10_002, 10_003, 10_004],
        help="fixed model-selection seeds; never reuse them for final testing",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reg-selector-v5"),
    )
    return parser.parse_args()


def make_env(lam: float, mu: float) -> SmallRoomsEnv:
    return SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=lam,
        proc_mean=mu,
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def evaluate(
    training_source: REGV5AssignmentSource,
    *,
    lam: float,
    mu: float,
    seeds: list[int],
    max_steps: int,
) -> dict:
    payload = training_source.checkpoint()
    rows = []
    for seed in seeds:
        env = make_env(lam, mu)
        instance = env.sample_episode_instance(seed)
        source = REGV5AssignmentSource.from_checkpoint(
            env,
            payload,
            learning_enabled=False,
            device="cpu",
            seed=seed,
        )
        result = run_track_a_episode(
            env,
            TRACK_A_REG_SELECTOR_V5,
            max_steps=max_steps,
            episode_instance=instance,
            assignment_source=source,
        )
        errors = np.asarray(result["delivery_errors"], dtype=float)
        rows.append(
            {
                "seed": seed,
                "instance_id": result["instance_id"],
                "return": result["return"],
                "success": result["success"],
                "strict_method_success": result["strict_method_success"],
                "steps": result["steps"],
                "completion_fraction": result["completion_fraction"],
                "mean_absolute_error": (
                    float(np.abs(errors).mean()) if errors.size else None
                ),
                "mean_tardiness": (
                    float(np.maximum(errors, 0.0).mean()) if errors.size else None
                ),
                "invalid_assignment_count": result["invalid_assignment_count"],
                "fallback_count": result["fallback_count"],
                "obstructive_moves": result["obstructive_moves"],
                "infeasible_epoch_count": result["infeasible_epoch_count"],
            }
        )
    absolute_errors = [
        row["mean_absolute_error"]
        for row in rows
        if row["mean_absolute_error"] is not None
    ]
    tardiness = [
        row["mean_tardiness"]
        for row in rows
        if row["mean_tardiness"] is not None
    ]
    return {
        "seeds": list(seeds),
        "episode_count": len(rows),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "strict_method_success_rate": float(
            np.mean([row["strict_method_success"] for row in rows])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_absolute_error": (
            float(np.mean(absolute_errors)) if absolute_errors else None
        ),
        "mean_tardiness": float(np.mean(tardiness)) if tardiness else None,
        "mean_infeasible_epoch_count": float(
            np.mean([row["infeasible_epoch_count"] for row in rows])
        ),
        "rows": rows,
    }


def save_checkpoint(
    source: REGV5AssignmentSource,
    path: Path,
    *,
    args: argparse.Namespace,
    completed_episodes: int,
    evaluation: dict,
) -> None:
    payload = source.checkpoint(
        training_seed=args.seed,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        training_lambda=args.lam,
        training_mu=args.mu,
        completed_training_episodes=completed_episodes,
        max_steps=args.max_steps,
        validation_evaluation=evaluation,
    )
    torch.save(payload, path)


def mean_tail(values: list[float], count: int = 100):
    return float(np.mean(values[-count:])) if values else None


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.eval_every <= 0:
        raise ValueError("episodes, max-steps, and eval-every must be positive")
    if not args.validation_seeds:
        raise ValueError("at least one validation seed is required")
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None:
        raise RuntimeError(
            "PYTHONHASHSEED must be fixed before interpreter startup"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = REGV5Config(
        block_embedding_dim=args.block_embedding_dim,
        candidate_embedding_dim=args.candidate_embedding_dim,
        context_dim=args.context_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        updates_per_episode=args.updates_per_episode,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        grad_clip=args.grad_clip,
        huber_delta=args.huber_delta,
        truncation_penalty=args.truncation_penalty,
        target_update_every=args.target_update_every,
        ema_tau=args.ema_tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_warmup_assignments=args.epsilon_warmup,
        epsilon_decay_assignments=args.epsilon_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.lam, args.mu)
    source = REGV5AssignmentSource(
        env,
        config,
        seed=args.seed,
        learning_enabled=True,
        device=device,
    )
    history = []
    best_key = None
    best_episode = None
    best_evaluation = None
    recent_returns = []
    recent_successes = []
    print(
        f"REG-v5 | device={device} | seed={args.seed} | "
        f"episodes={args.episodes} | gamma={config.gamma} | "
        f"reward_scale={config.reward_scale}",
        flush=True,
    )

    for episode in range(1, args.episodes + 1):
        instance_seed = 200_000 + args.seed * 1_000_000 + episode
        instance = env.sample_episode_instance(instance_seed)
        result = run_track_a_episode(
            env,
            TRACK_A_REG_SELECTOR_V5,
            max_steps=args.max_steps,
            episode_instance=instance,
            assignment_source=source,
        )
        recent_returns.append(float(result["return"]))
        recent_successes.append(float(result["success"]))
        recent_returns = recent_returns[-100:]
        recent_successes = recent_successes[-100:]
        if episode % args.eval_every != 0 and episode != args.episodes:
            continue

        validation = evaluate(
            source,
            lam=args.lam,
            mu=args.mu,
            seeds=args.validation_seeds,
            max_steps=args.max_steps,
        )
        record = {
            "episode": episode,
            "training_instance_seed": instance_seed,
            "training_return_ma100": float(np.mean(recent_returns)),
            "training_success_ma100": float(np.mean(recent_successes)),
            "selector_epsilon": source.epsilon,
            "assignment_count": source.assignment_count,
            "replay_size": len(source.replay),
            "gradient_steps": source.gradient_steps,
            "target_updates": source.target_updates,
            "completed_intervals": source.completed_intervals,
            "terminal_intervals": source.terminal_intervals,
            "truncated_intervals": source.truncated_intervals,
            "loss_ma100": mean_tail(source.loss_history),
            "q_ma100": mean_tail(source.q_history),
            "target_ma100": mean_tail(source.target_history),
            "validation": validation,
        }
        history.append(record)
        save_checkpoint(
            source,
            args.output_dir / "latest.pth",
            args=args,
            completed_episodes=episode,
            evaluation=validation,
        )
        key = (
            validation["strict_method_success_rate"],
            validation["mean_return"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_episode = episode
            best_evaluation = validation
            save_checkpoint(
                source,
                args.output_dir / "best.pth",
                args=args,
                completed_episodes=episode,
                evaluation=validation,
            )
        (args.output_dir / "training-history.json").write_text(
            json.dumps(json_safe(history), indent=2) + "\n"
        )
        print(
            f"Ep {episode:4d} | TrainR {record['training_return_ma100']:8.2f} "
            f"| ValR {validation['mean_return']:8.2f} "
            f"| ValSucc {validation['success_rate']:.3f} "
            f"| ValMAE {validation['mean_absolute_error']:.2f} "
            f"| Eps {source.epsilon:.3f} | Loss {record['loss_ma100']}",
            flush=True,
        )

    summary = {
        "status": "completed",
        "selector_feature_version": source.feature_version,
        "selector_architecture": source.architecture_name,
        "config": asdict(config),
        "lambda": args.lam,
        "mu": args.mu,
        "training_seed": args.seed,
        "python_hash_seed": hash_seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "training_instance_seed_start": 200_001 + args.seed * 1_000_000,
        "training_instance_seed_end": 200_000 + args.seed * 1_000_000 + args.episodes,
        "validation_seeds": args.validation_seeds,
        "best_episode": best_episode,
        "best_validation": best_evaluation,
        "final_validation": history[-1]["validation"],
        "assignment_count": source.assignment_count,
        "completed_intervals": source.completed_intervals,
        "terminal_intervals": source.terminal_intervals,
        "truncated_intervals": source.truncated_intervals,
        "gradient_steps": source.gradient_steps,
        "target_updates": source.target_updates,
        "final_epsilon": source.epsilon,
        "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
        "latest_checkpoint": str((args.output_dir / "latest.pth").resolve()),
    }
    (args.output_dir / "training-summary.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n"
    )
    print(json.dumps(json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
