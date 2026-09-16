#!/usr/bin/env python3
"""Train the opt-in relational residual Track-B scheduler.

The original ``train_track_b.py --controller-variant atomic_inbound_scheduler``
path is intentionally unchanged.  This trainer writes a different checkpoint
schema which can only be consumed by ``track_b_relational_evaluate.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from example.Options.selector_v5 import StorageSelectOptionV5
from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import build_controller_options, scheduler_episode_audit
from example.helper.timing_metrics import (
    summarize_block_storage_flow,
    summarize_delivery_timing,
    summarize_storage_flow_runs,
)
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.checkpoint_identity import selector_deployment_digest
from relational_scheduler import (
    RELATIONAL_ACTION_INTERFACE,
    RELATIONAL_POLICY_REALIZATIONS,
    RelationalResidualSchedulerAgent,
    RelationalSchedulerConfig,
    RelationalSchedulerInfeasible,
)


RELATIONAL_CHECKPOINT_SCHEMA_VERSION = 1
RELATIONAL_TRAINER_VERSION = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def scheduled_epsilon(decision_count, args):
    if decision_count <= args.epsilon_warmup_decisions:
        return args.epsilon_start
    fraction = min(
        1.0,
        (decision_count - args.epsilon_warmup_decisions)
        / args.epsilon_decay_decisions,
    )
    return args.epsilon_start + fraction * (
        args.epsilon_end - args.epsilon_start
    )


def selector_snapshot(selector):
    return {
        "decision_count": selector.decision_count,
        "infeasible_epoch_count": selector.infeasible_epoch_count,
        "invalid_assignment_count": selector.invalid_assignment_count,
        "assignment_seconds": selector.assignment_seconds,
        "decisions_length": len(selector.decisions),
    }


def restore_selector(selector, snapshot):
    selector.decision_count = snapshot["decision_count"]
    selector.infeasible_epoch_count = snapshot["infeasible_epoch_count"]
    selector.invalid_assignment_count = snapshot["invalid_assignment_count"]
    selector.assignment_seconds = snapshot["assignment_seconds"]
    del selector.decisions[snapshot["decisions_length"] :]


def selector_delta(selector, snapshot):
    return {
        "decision_count": selector.decision_count - snapshot["decision_count"],
        "infeasible_epoch_count": (
            selector.infeasible_epoch_count
            - snapshot["infeasible_epoch_count"]
        ),
        "invalid_assignment_count": (
            selector.invalid_assignment_count
            - snapshot["invalid_assignment_count"]
        ),
        "assignment_seconds": (
            selector.assignment_seconds - snapshot["assignment_seconds"]
        ),
    }


def _failed_macro_outcome(option):
    outcome = getattr(option, "last_outcome", None)
    return (
        outcome
        if isinstance(outcome, dict) and outcome.get("success") is False
        else None
    )


def run_episode(
    agent,
    selector,
    env,
    instance,
    *,
    max_steps,
    target_window,
    training,
    exploration_state=None,
    epsilon_args=None,
    include_decisions=False,
):
    state = env.reset(instance=instance)
    agent.reset_episode()
    selector_start = selector_snapshot(selector)
    total_return = 0.0
    errors = []
    obstructive_moves = 0
    illegal_drops = 0
    losses = []
    method_failure_reason = None
    done = False
    steps = 0
    decision_start = (
        int(exploration_state["decisions"])
        if exploration_state is not None
        else 0
    )
    epsilon_start = (
        int(exploration_state["epsilon_decisions"])
        if exploration_state is not None
        else 0
    )
    started = perf_counter()

    while steps < max_steps and not done and method_failure_reason is None:
        if agent.current_option is None:
            try:
                agent.current_option = agent.select_action(
                    state, eps=agent.epsilon if training else 0.0
                )
            except RelationalSchedulerInfeasible as exc:
                method_failure_reason = f"scheduler_infeasible:{exc}"
                break
            if training:
                exploration_state["decisions"] += 1
                if not agent.last_decision_was_forced:
                    exploration_state["epsilon_decisions"] += 1
                agent.epsilon = scheduled_epsilon(
                    exploration_state["epsilon_decisions"], epsilon_args
                )

        option = agent.current_option
        try:
            action = option.policy(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            selector.on_step(reward, info)
            total_return += float(reward)
            if "delivery_error_time" in info:
                errors.append(float(info["delivery_error_time"]))
            obstructive_moves += int(bool(info.get("relocated_block")))
            illegal_drops += int(bool(info.get("illegal_drop")))
            terminated = bool(option.termination(next_state))
        except RuntimeError as exc:
            method_failure_reason = f"macro_runtime:{type(option).__name__}:{exc}"
            break

        failed_outcome = _failed_macro_outcome(option) if terminated else None
        macro_failed = failed_outcome is not None
        truncated = bool(steps >= max_steps and not done)
        effective_termination = bool(terminated or done or truncated)
        replay_done = bool(done or truncated or macro_failed)
        if training:
            agent.step_count += 1
        agent.process_step(
            next_state,
            reward,
            done=replay_done,
            terminated=effective_termination,
            failed=macro_failed,
            store_transition=training,
        )
        if effective_termination:
            agent.current_option = None
        if training:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        state = next_state
        if macro_failed:
            method_failure_reason = (
                f"macro_failure:{type(option).__name__}:"
                f"{failed_outcome.get('reason', 'unknown')}"
            )

    elapsed = perf_counter() - started
    success = bool(done and method_failure_reason is None)
    selector.on_episode_end(success=success, truncated=not success)
    timing = summarize_delivery_timing(errors, target_window)
    storage_flow = summarize_block_storage_flow(env.blocks, env.time_steps)
    selector_audit = selector_delta(selector, selector_start)
    option_audit = scheduler_episode_audit(env)
    strict_success = bool(
        success
        and selector_audit["invalid_assignment_count"] == 0
        and option_audit["inbound_failures"] == 0
        and option_audit["retrieve_failures"] == 0
        and illegal_drops == 0
    )
    return {
        "return": total_return,
        "success": float(success),
        "strict_method_success": float(strict_success),
        "truncated": float(steps >= max_steps and not done),
        "method_failure_reason": method_failure_reason,
        "steps": steps,
        "delivery_count": len(errors),
        "delivery_deviations": errors,
        "illegal_drops": illegal_drops,
        "obstructive_moves": obstructive_moves,
        "decision_seconds": elapsed,
        "selector_audit": selector_audit,
        "scheduler_audit": option_audit,
        "relational_audit": agent.audit(include_decisions=include_decisions),
        "controller_decision_count": (
            exploration_state["decisions"] - decision_start
            if exploration_state is not None
            else agent.decision_counts["option"]
        ),
        "epsilon_decision_count": (
            exploration_state["epsilon_decisions"] - epsilon_start
            if exploration_state is not None
            else 0
        ),
        "loss": losses[-1] if losses else None,
        **timing,
        **storage_flow,
    }


def summarize_runs(runs):
    returns = np.asarray([item["return"] for item in runs], dtype=float)
    errors = np.asarray(
        [item["mean_absolute_error"] for item in runs], dtype=float
    )
    finite_errors = errors[np.isfinite(errors)]
    controls = Counter()
    gates = Counter()
    for run in runs:
        controls.update(run["relational_audit"]["control_decisions"])
        gates.update(run["relational_audit"]["gate_decisions"])
    return {
        "episodes": len(runs),
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(runs) > 1 else 0.0,
        "success_rate": float(np.mean([item["success"] for item in runs])),
        "strict_method_success_rate": float(
            np.mean([item["strict_method_success"] for item in runs])
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
                item["selector_audit"]["invalid_assignment_count"]
                for item in runs
            )
        ),
        "total_infeasible_assignment_epochs": int(
            sum(
                item["selector_audit"]["infeasible_epoch_count"]
                for item in runs
            )
        ),
        "total_inbound_failures": int(
            sum(item["scheduler_audit"]["inbound_failures"] for item in runs)
        ),
        "total_retrieval_failures": int(
            sum(item["scheduler_audit"]["retrieve_failures"] for item in runs)
        ),
        "method_failures": [
            {
                "eval_seed": item.get("eval_seed"),
                "reason": item["method_failure_reason"],
            }
            for item in runs
            if item["method_failure_reason"] is not None
        ],
        "controller_control_decisions": dict(controls),
        "gate_decisions": dict(gates),
        **summarize_storage_flow_runs(runs),
    }


def validate(agent, selector, env, args):
    random_state = rng_state()
    selector_state = selector_snapshot(selector)
    saved_epsilon = agent.epsilon
    saved_policy = agent.policy_realization
    try:
        agent.set_policy_realization(args.validation_policy)
        runs = []
        for seed in args.validation_seeds:
            seed_everything(seed)
            instance = env.sample_episode_instance(seed)
            result = run_episode(
                agent,
                selector,
                env,
                instance,
                max_steps=args.validation_steps,
                target_window=args.target_window,
                training=False,
                include_decisions=False,
            )
            result.update({"eval_seed": seed, "instance_id": instance.instance_id})
            runs.append(result)
        return {**summarize_runs(runs), "runs": runs}
    finally:
        restore_selector(selector, selector_state)
        restore_rng_state(random_state)
        agent.epsilon = saved_epsilon
        agent.set_policy_realization(saved_policy)
        agent.reset_episode()


def build_stack(args, selector_payload, device):
    seed_everything(args.seed)
    env = SmallRoomsEnv(
        choose_storage=False, arrival_rate=args.lam, proc_mean=args.mu
    )
    state = rng_state()
    selector = StorageSelectOptionV5.from_checkpoint(
        env,
        selector_payload,
        device=device,
        seed=args.seed,
        learning_enabled=False,
    )
    restore_rng_state(state)
    build_controller_options(
        env,
        selector,
        controller_action_interface=RELATIONAL_ACTION_INTERFACE,
        max_defer_steps=args.max_defer_steps,
    )
    initial = env.sample_episode_instance(
        args.training_instance_seed_base + args.seed * 1_000_000
    )
    env.reset(instance=initial)
    encoder = OnlineManifestTimingObservationEncoder(env)
    config = RelationalSchedulerConfig(
        block_embedding_dim=args.block_embedding_dim,
        global_embedding_dim=args.global_embedding_dim,
        candidate_embedding_dim=args.candidate_embedding_dim,
        context_dim=args.context_dim,
        residual_scale=args.residual_scale,
        uncertainty_penalty=args.uncertainty_penalty,
        override_margin=args.override_margin,
        history_length=args.history_length,
        risk_loss_weight=args.risk_loss_weight,
    )
    agent = RelationalResidualSchedulerAgent(
        env,
        encoder,
        config=config,
        seed=args.seed,
        device=device,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_every=args.update_every,
        target_tau=args.target_tau,
        grad_clip=args.grad_clip,
        reward_clip=args.reward_clip,
        epsilon=args.epsilon_start,
    )
    agent.set_policy_realization(args.training_policy)
    return env, selector, agent


def checkpoint_payload(
    agent,
    selector_payload,
    args,
    episode,
    validation,
    best_score,
    exploration_state,
):
    return {
        "relational_checkpoint_schema_version": (
            RELATIONAL_CHECKPOINT_SCHEMA_VERSION
        ),
        "relational_trainer_version": RELATIONAL_TRAINER_VERSION,
        **agent.checkpoint_metadata(),
        "relational_q_state_dict": agent.Q_local.state_dict(),
        "relational_target_state_dict": agent.Q_target.state_dict(),
        "relational_optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "step_count": agent.step_count,
        "exploration_decisions": exploration_state["decisions"],
        "epsilon_exploration_decisions": exploration_state["epsilon_decisions"],
        "completed_training_episodes": episode,
        "training_seed": args.seed,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "max_defer_steps": args.max_defer_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "update_every": args.update_every,
        "target_tau": args.target_tau,
        "grad_clip": args.grad_clip,
        "reward_clip": args.reward_clip,
        "validation_policy": args.validation_policy,
        "selector_checkpoint": selector_payload,
        "selector_deployment_digest": selector_deployment_digest(
            selector_payload
        ),
        "selector_frozen": True,
        "validation": validation,
        "best_score": best_score,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--update-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--reward-clip", type=float, default=100.0)
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-warmup-decisions", type=int, default=500)
    parser.add_argument("--epsilon-decay-decisions", type=int, default=20_000)
    parser.add_argument("--max-defer-steps", type=int, default=10)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--block-embedding-dim", type=int, default=64)
    parser.add_argument("--global-embedding-dim", type=int, default=64)
    parser.add_argument("--candidate-embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.05)
    parser.add_argument("--risk-loss-weight", type=float, default=0.1)
    parser.add_argument("--override-margin", type=float, default=1.0)
    parser.add_argument(
        "--training-policy",
        choices=RELATIONAL_POLICY_REALIZATIONS,
        default="residual_map",
    )
    parser.add_argument(
        "--validation-policy",
        choices=RELATIONAL_POLICY_REALIZATIONS,
        default="residual_map",
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[10000, 10001, 10002],
    )
    parser.add_argument("--validation-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--training-instance-seed-base", type=int, default=0)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    if args.batch_size <= 0 or args.buffer_size < args.batch_size:
        parser.error("buffer-size must be at least batch-size, both positive")
    if args.eval_every <= 0 or args.log_every <= 0:
        parser.error("evaluation and logging intervals must be positive")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon must satisfy 0 <= end <= start <= 1")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selector_payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector_regime = (
        selector_payload.get("training_lambda"),
        selector_payload.get("training_mu"),
    )
    if selector_regime != (args.lam, args.mu):
        raise ValueError(
            "Frozen selector regime does not match relational training: "
            f"selector={selector_regime}, training={(args.lam, args.mu)}"
        )
    device = resolve_device(args.device)
    env, selector, agent = build_stack(args, selector_payload, device)
    history = []
    validations = []
    rolling_returns = deque(maxlen=25)
    best_score = (-1.0, -float("inf"))
    exploration_state = {"decisions": 0, "epsilon_decisions": 0}
    print(
        "Track-B relational-residual+frozen-REG-v5 | "
        f"device={device} | seed={args.seed} | episodes={args.episodes}",
        flush=True,
    )

    for episode in range(1, args.episodes + 1):
        instance_seed = (
            args.training_instance_seed_base + args.seed * 1_000_000 + episode
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
            exploration_state=exploration_state,
            epsilon_args=args,
            include_decisions=False,
        )
        result.update(
            {
                "episode": episode,
                "instance_seed": instance_seed,
                "instance_id": instance.instance_id,
                "epsilon": agent.epsilon,
            }
        )
        history.append(result)
        rolling_returns.append(result["return"])
        should_validate = episode % args.eval_every == 0 or episode == args.episodes
        if should_validate:
            validation = validate(agent, selector, env, args)
            validation["episode"] = episode
            validations.append(validation)
            score = (
                validation["strict_method_success_rate"],
                validation["mean_return"],
            )
            improved = score > best_score
            if improved:
                best_score = score
            payload = checkpoint_payload(
                agent,
                selector_payload,
                args,
                episode,
                validation,
                best_score,
                exploration_state,
            )
            torch.save(payload, args.output_dir / "latest.pth")
            if improved:
                torch.save(payload, args.output_dir / "best.pth")
            summary = json_safe(
                {
                    "config": vars(args),
                    "device": str(device),
                    "best_score": best_score,
                    "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
                    "latest_checkpoint": str(
                        (args.output_dir / "latest.pth").resolve()
                    ),
                    "history": history,
                    "validations": validations,
                }
            )
            (args.output_dir / "training-summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"Ep {episode:4d} | TrainR {np.mean(rolling_returns):8.2f} | "
                f"ValR {validation['mean_return']:8.2f} | "
                f"ValSucc {validation['strict_method_success_rate']:.3f} | "
                f"Eps {agent.epsilon:.3f} | Replay {len(agent.replay)} | "
                f"Loss {result['loss']}",
                flush=True,
            )
        elif episode % args.log_every == 0 or episode == 1:
            print(
                f"Ep {episode:4d} | TrainR {np.mean(rolling_returns):8.2f} | "
                f"Succ {result['strict_method_success']:.0f} | "
                f"Eps {agent.epsilon:.3f} | Replay {len(agent.replay)} | "
                f"Loss {result['loss']}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "best_score": best_score,
                "final_epsilon": agent.epsilon,
                "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
                "latest_checkpoint": str((args.output_dir / "latest.pth").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
