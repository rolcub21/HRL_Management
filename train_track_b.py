#!/usr/bin/env python3
"""Train the gated Track B controller around a frozen REG-v5 selector."""

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
from example.controller_options import (
    build_controller_options,
    scheduler_episode_audit,
)
from example.controller_observation import (
    LEGACY_CONTROLLER_OBSERVATION,
    ONLINE_MANIFEST_TIMING_OBSERVATION,
    ONLINE_SIGNED_TIMING_OBSERVATION,
    SUPPORTED_CONTROLLER_OBSERVATIONS,
    make_controller_observation_encoder,
)
from example.helper.timing_metrics import summarize_delivery_timing
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import (
    ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE,
    GatedAtomicInboundSchedulerAgent,
    GatedModeOnlyAgent,
    GatedModeOnlyFoundationAgent,
    GatedRegularizedAgent,
    GatedSchedulingAgent,
    LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE,
    SCHEDULER_CONTROLLER_ACTION_INTERFACE,
)
from options_agent import (
    CHECKPOINT_SCHEMA_VERSION,
    controller_ids,
)
from PSLAP.checkpoint_identity import selector_deployment_digest


TRAINER_VERSION = 5
CONTROLLER_VARIANTS = {
    "full_regularized": GatedRegularizedAgent,
    "mode_only": GatedModeOnlyAgent,
    "mode_only_foundation": GatedModeOnlyFoundationAgent,
    "mode_only_scheduler": GatedSchedulingAgent,
    "atomic_inbound_scheduler": GatedAtomicInboundSchedulerAgent,
}


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


def make_env(args):
    return SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=args.lam,
        proc_mean=args.mu,
    )


def build_training_stack(args, selector_payload, device):
    seed_everything(args.seed)
    env = make_env(args)

    # Loading the frozen selector constructs temporary network parameters. Keep
    # that from changing the seeded initialization of the controller critics.
    state = rng_state()
    selector = StorageSelectOptionV5.from_checkpoint(
        env,
        selector_payload,
        device=device,
        seed=args.seed,
        learning_enabled=False,
    )
    restore_rng_state(state)
    action_interface = {
        "mode_only_scheduler": SCHEDULER_CONTROLLER_ACTION_INTERFACE,
        "atomic_inbound_scheduler": (
            ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
        ),
    }.get(
        args.controller_variant,
        LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE,
    )
    build_controller_options(
        env,
        selector,
        retrieval_lead_time=args.retrieval_lead_time,
        include_retrieval_handling_steps=(
            args.retrieval_duration_definition == "complete_plan_v1"
        ),
        controller_action_interface=action_interface,
        max_defer_steps=args.max_defer_steps,
    )
    initial_instance = env.sample_episode_instance(
        args.training_instance_seed_base + args.seed * 1_000_000
    )
    initial_state = env.reset(instance=initial_instance)
    observation_encoder = make_controller_observation_encoder(
        env, args.controller_observation
    )
    initial_features = observation_encoder(initial_state)
    agent_class = CONTROLLER_VARIANTS[args.controller_variant]
    agent = agent_class(
        env=env,
        state_size=len(initial_features),
        action_size=len([option for option in env.options if option.is_primitive]),
        n_episodes=args.episodes,
        n_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        lr_manager=args.lr_manager,
        lr_worker=args.lr_worker,
        gamma=args.gamma,
        tau_soft=args.target_tau,
        update_every=args.update_every,
        grad_clip=args.grad_clip,
        reward_clip=args.reward_clip,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_end,
        tau_option=args.tau_option,
        tau_primitive=args.tau_primitive,
        tau_mode=args.tau_mode,
        training_policy=args.training_policy,
        seed=args.seed,
        state_encoder=observation_encoder,
        controller_observation_metadata=observation_encoder.metadata(),
        disable_tensorboard=True,
        device=device,
        verbose=True,
        max_defer_steps=args.max_defer_steps,
    )
    return env, selector, agent


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


def run_training_episode(
    agent, selector, env, instance, max_steps, exploration_state, args
):
    exploration_state.setdefault(
        "epsilon_decisions", int(exploration_state.get("decisions", 0))
    )
    decision_start = int(exploration_state["decisions"])
    epsilon_decision_start = int(exploration_state["epsilon_decisions"])
    state = env.reset(instance=instance)
    agent.current_option = None
    total_return = 0.0
    errors = []
    manager_losses = []
    worker_losses = []
    illegal_drops = 0
    obstructive_moves = 0
    done = False

    for step in range(max_steps):
        state_features = agent.encode_state(state)
        if agent.current_option is None:
            agent.current_option = agent.select_action(
                state,
                agent.epsilon,
                state_features=state_features,
            )
            exploration_state["decisions"] += 1
            if not getattr(agent, "last_decision_was_forced", False):
                exploration_state["epsilon_decisions"] += 1
            agent.epsilon = scheduled_epsilon(
                exploration_state["epsilon_decisions"], args
            )
            if not agent.current_option.is_primitive:
                agent.option_start_state = state
                agent.option_start_features = state_features.copy()
                agent.option_reward_traj = []
        current_option = agent.current_option
        action = current_option.policy(state)
        next_state, reward, done, info = env.step(action)
        next_state_features = agent.encode_state(next_state)
        selector.on_step(reward, info)
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        illegal_drops += int(bool(info.get("illegal_drop")))
        obstructive_moves += int(bool(info.get("relocated_block")))
        terminated = current_option.termination(next_state)
        truncated = bool(step + 1 >= max_steps and not done)
        replay_done = bool(
            done or (truncated and agent.terminal_on_truncation)
        )
        replay_termination = bool(
            terminated
            or (
                replay_done
                and agent.close_options_on_episode_end
            )
        )
        agent.step_count += 1
        agent.process_step(
            state,
            action,
            reward,
            next_state,
            replay_done,
            replay_termination,
            state_features=state_features,
            next_state_features=next_state_features,
        )
        if replay_termination:
            agent.current_option = None
        manager_loss, worker_loss = agent.learn()
        if manager_loss is not None:
            manager_losses.append(float(manager_loss))
        if worker_loss is not None:
            worker_losses.append(float(worker_loss))
        total_return += float(reward)
        state = next_state
        if done:
            break

    steps = step + 1
    truncated = bool(steps >= max_steps and not done)
    selector.on_episode_end(
        success=done,
        truncated=truncated,
    )
    return {
        "return": total_return,
        "success": float(done),
        "truncated": float(truncated),
        "steps": steps,
        "delivery_count": len(errors),
        "delivery_deviations": errors,
        "illegal_drops": illegal_drops,
        "obstructive_moves": obstructive_moves,
        "scheduler_audit": scheduler_episode_audit(env),
        "controller_decision_count": int(
            exploration_state["decisions"] - decision_start
        ),
        "epsilon_decision_count": int(
            exploration_state["epsilon_decisions"]
            - epsilon_decision_start
        ),
        "manager_loss": manager_losses[-1] if manager_losses else None,
        "worker_loss": worker_losses[-1] if worker_losses else None,
    }


def run_validation_episode(
    agent, selector, env, instance, seed, max_steps, policy, target_window
):
    seed_everything(seed)
    state = env.reset(instance=instance)
    agent.current_option = None
    agent.set_policy_realization(policy)
    agent.decision_counts.clear()
    agent.control_decisions.clear()
    agent.gate_decisions.clear()
    selector_start = {
        "decision_count": selector.decision_count,
        "infeasible": selector.infeasible_epoch_count,
        "invalid": selector.invalid_assignment_count,
        "seconds": selector.assignment_seconds,
        "decision_length": len(selector.decisions),
    }
    total_return = 0.0
    errors = []
    illegal_drops = 0
    obstructive_moves = 0
    done = False
    started = perf_counter()
    for step in range(max_steps):
        if agent.current_option is None:
            agent.current_option = agent.select_action(state, eps=0.0)
        action = agent.current_option.policy(state)
        next_state, reward, done, info = env.step(action)
        selector.on_step(reward, info)
        total_return += float(reward)
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        illegal_drops += int(bool(info.get("illegal_drop")))
        obstructive_moves += int(bool(info.get("relocated_block")))
        if agent.current_option.termination(next_state):
            agent.current_option = None
        state = next_state
        if done:
            break
    elapsed = perf_counter() - started
    steps = step + 1
    selector.on_episode_end(
        success=done,
        truncated=bool(steps >= max_steps and not done),
    )
    timing = summarize_delivery_timing(errors, target_window)
    result = {
        "eval_seed": seed,
        "instance_id": instance.instance_id,
        "return": total_return,
        "success": float(done),
        "steps": steps,
        "delivery_count": len(errors),
        "illegal_drops": illegal_drops,
        "obstructive_moves": obstructive_moves,
        "decision_seconds": elapsed,
        "selector_decisions": selector.decision_count
        - selector_start["decision_count"],
        "selector_infeasible_epochs": selector.infeasible_epoch_count
        - selector_start["infeasible"],
        "selector_invalid_assignments": selector.invalid_assignment_count
        - selector_start["invalid"],
        "selector_assignment_seconds": selector.assignment_seconds
        - selector_start["seconds"],
        "controller_decisions": dict(agent.decision_counts),
        "controller_control_decisions": dict(agent.control_decisions),
        "scheduler_audit": scheduler_episode_audit(env),
        **timing,
    }
    # Validation must not contaminate training-side selector audit state.
    selector.decision_count = selector_start["decision_count"]
    selector.infeasible_epoch_count = selector_start["infeasible"]
    selector.invalid_assignment_count = selector_start["invalid"]
    selector.assignment_seconds = selector_start["seconds"]
    del selector.decisions[selector_start["decision_length"] :]
    return result


def validate(agent, selector, env, args):
    state = rng_state()
    training_policy = agent.policy_realization
    training_decisions = agent.decision_counts.copy()
    training_control_decisions = agent.control_decisions.copy()
    training_gate = agent.gate_decisions.copy()
    try:
        runs = []
        for seed in args.validation_seeds:
            instance = env.sample_episode_instance(seed)
            runs.append(
                run_validation_episode(
                    agent,
                    selector,
                    env,
                    instance,
                    seed,
                    args.validation_steps,
                    args.validation_policy,
                    args.target_window,
                )
            )
    finally:
        restore_rng_state(state)
        agent.set_policy_realization(training_policy)
        agent.decision_counts = training_decisions
        agent.control_decisions = training_control_decisions
        agent.gate_decisions = training_gate
        agent.current_option = None
    returns = np.asarray([run["return"] for run in runs], dtype=float)
    absolute_errors = np.asarray(
        [run["mean_absolute_error"] for run in runs], dtype=float
    )
    finite_absolute_errors = absolute_errors[np.isfinite(absolute_errors)]
    control_decisions = Counter()
    defer_outcomes = Counter()
    for run in runs:
        control_decisions.update(run["controller_control_decisions"])
        defer_outcomes.update(
            run["scheduler_audit"].get("defer_outcomes", {})
        )
    return {
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(runs) > 1 else 0.0,
        "success_rate": float(np.mean([run["success"] for run in runs])),
        "mean_steps": float(np.mean([run["steps"] for run in runs])),
        "mean_absolute_error": (
            float(finite_absolute_errors.mean())
            if len(finite_absolute_errors)
            else float("nan")
        ),
        "illegal_drops": int(sum(run["illegal_drops"] for run in runs)),
        "selector_invalid_assignments": int(
            sum(run["selector_invalid_assignments"] for run in runs)
        ),
        "selector_infeasible_epochs": int(
            sum(run["selector_infeasible_epochs"] for run in runs)
        ),
        "inbound_successes": int(
            sum(
                run["scheduler_audit"].get("inbound_successes", 0)
                for run in runs
            )
        ),
        "inbound_failures": int(
            sum(
                run["scheduler_audit"].get("inbound_failures", 0)
                for run in runs
            )
        ),
        "retrieve_successes": int(
            sum(
                run["scheduler_audit"].get("retrieve_successes", 0)
                for run in runs
            )
        ),
        "retrieve_failures": int(
            sum(
                run["scheduler_audit"].get("retrieve_failures", 0)
                for run in runs
            )
        ),
        "defer_outcomes": dict(defer_outcomes),
        "controller_control_decisions": dict(control_decisions),
        "runs": runs,
    }


def checkpoint_payload(
    agent,
    selector_payload,
    args,
    episode,
    validation,
    *,
    best_score,
    exploration_decisions,
    epsilon_exploration_decisions,
):
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "trainer_version": TRAINER_VERSION,
        **agent.checkpoint_metadata(),
        "manager_option_ids": controller_ids(agent.manager_options),
        "primitive_action_ids": controller_ids(agent.primitive_options),
        "manager_state_dict": agent.Q_manager_local.state_dict(),
        "manager_target_state_dict": agent.Q_manager_target.state_dict(),
        "worker_state_dict": agent.Q_worker_local.state_dict(),
        "worker_target_state_dict": agent.Q_worker_target.state_dict(),
        "manager_opt_state": agent.optimizer_manager.state_dict(),
        "worker_opt_state": agent.optimizer_worker.state_dict(),
        "epsilon": agent.epsilon,
        "exploration_decisions": exploration_decisions,
        "epsilon_exploration_decisions": epsilon_exploration_decisions,
        "step_count": agent.step_count,
        "completed_training_episodes": episode,
        "training_seed": args.seed,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "validation_policy": args.validation_policy,
        "gamma": args.gamma,
        "target_tau": args.target_tau,
        "retrieval_lead_time": args.retrieval_lead_time,
        "retrieval_duration_definition": args.retrieval_duration_definition,
        "max_defer_steps": args.max_defer_steps,
        "selector_kind": selector_payload["selector_architecture"],
        "selector_feature_version": selector_payload["selector_feature_version"],
        "selector_return_definition": "assignment_epoch_double_dqn_v5",
        "selector_gamma": float(selector_payload["config"]["gamma"]),
        "selector_checkpoint": selector_payload,
        "selector_deployment_digest": selector_deployment_digest(
            selector_payload
        ),
        "selector_frozen": True,
        "validation": validation,
        "best_score": best_score,
    }
    return payload


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
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controller-variant",
        choices=tuple(CONTROLLER_VARIANTS),
        default="atomic_inbound_scheduler",
        help=(
            "atomic_inbound_scheduler is the v5 scheduler-only controller; "
            "mode_only_scheduler preserves the v4 split-inbound ablation; "
            "mode_only_foundation preserves the corrected split-option ablation; "
            "mode_only and full_regularized preserve earlier ablations"
        ),
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--update-every", type=int, default=100)
    parser.add_argument("--lr-manager", type=float, default=5e-5)
    parser.add_argument("--lr-worker", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--reward-clip", type=float, default=100.0)
    parser.add_argument("--epsilon-start", type=float, default=0.90)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-warmup-decisions", type=int, default=None)
    parser.add_argument(
        "--epsilon-decay-decisions",
        type=int,
        default=None,
        help=(
            "linear decay length in non-forced controller decisions; defaults "
            "to 20000 for atomic_inbound_scheduler, 600000 for earlier "
            "mode-only variants, and 100000 for full_regularized"
        ),
    )
    parser.add_argument("--tau-option", type=float, default=0.1)
    parser.add_argument("--tau-primitive", type=float, default=0.1)
    parser.add_argument("--tau-mode", type=float, default=1.0)
    parser.add_argument(
        "--max-defer-steps",
        type=int,
        default=10,
        help="maximum WAIT actions in one event-driven defer option",
    )
    parser.add_argument(
        "--retrieval-lead-time",
        type=float,
        default=None,
        help=(
            "dispatch safety margin in environment steps; defaults to 0 for "
            "the corrected foundation and 20 for legacy controller variants"
        ),
    )
    parser.add_argument(
        "--controller-observation",
        choices=SUPPORTED_CONTROLLER_OBSERVATIONS,
        default=None,
        help=(
            "defaults to online_signed_timing_v2 for the corrected foundation "
            "and legacy_flat_v1 for historical controller variants"
        ),
    )
    parser.add_argument(
        "--training-policy",
        choices=("map", "mode_regularized", "regularized"),
        default=None,
    )
    parser.add_argument(
        "--validation-policy",
        choices=("map", "mode_regularized", "regularized"),
        default=None,
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[10000, 10001, 10002, 10003, 10004],
    )
    parser.add_argument("--validation-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--training-instance-seed-base", type=int, default=0
    )
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
    default_policy = (
        "map"
        if args.controller_variant == "atomic_inbound_scheduler"
        else "mode_regularized"
        if args.controller_variant in (
            "mode_only",
            "mode_only_foundation",
            "mode_only_scheduler",
        )
        else "regularized"
    )
    args.training_policy = args.training_policy or default_policy
    args.validation_policy = args.validation_policy or default_policy
    if args.controller_variant == "atomic_inbound_scheduler":
        non_map = [
            name
            for name in ("training_policy", "validation_policy")
            if getattr(args, name) != "map"
        ]
        if non_map:
            parser.error(
                "atomic_inbound_scheduler has one active macro mode and "
                "requires map policy for " + ", ".join(non_map)
            )
    if args.controller_variant in (
        "mode_only_foundation",
        "mode_only_scheduler",
        "atomic_inbound_scheduler",
    ):
        args.controller_observation = args.controller_observation or (
            ONLINE_MANIFEST_TIMING_OBSERVATION
            if args.controller_variant in (
                "mode_only_scheduler",
                "atomic_inbound_scheduler",
            )
            else ONLINE_SIGNED_TIMING_OBSERVATION
        )
        args.retrieval_duration_definition = (
            "atomic_named_retrieval_v1"
            if args.controller_variant in (
                "mode_only_scheduler",
                "atomic_inbound_scheduler",
            )
            else "complete_plan_v1"
        )
        if args.retrieval_lead_time is None:
            args.retrieval_lead_time = 0.0
    else:
        args.controller_observation = (
            args.controller_observation or LEGACY_CONTROLLER_OBSERVATION
        )
        args.retrieval_duration_definition = "legacy_movement_only"
        if args.retrieval_lead_time is None:
            args.retrieval_lead_time = 20.0
    if args.epsilon_warmup_decisions is None:
        args.epsilon_warmup_decisions = (
            500
            if args.controller_variant == "atomic_inbound_scheduler"
            else 5_000
        )
    if args.epsilon_decay_decisions is None:
        args.epsilon_decay_decisions = (
            20_000
            if args.controller_variant == "atomic_inbound_scheduler"
            else
            600_000
            if args.controller_variant in (
                "mode_only",
                "mode_only_foundation",
                "mode_only_scheduler",
            )
            else 100_000
        )
    if args.controller_variant in (
        "mode_only",
        "mode_only_foundation",
        "mode_only_scheduler",
        "atomic_inbound_scheduler",
    ):
        incompatible = [
            name
            for name in ("training_policy", "validation_policy")
            if getattr(args, name) == "regularized"
        ]
        if incompatible:
            parser.error(
                "mode_only does not sample controls within a mode; use map or "
                "mode_regularized for " + ", ".join(incompatible)
            )
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    if args.eval_every <= 0 or args.log_every <= 0:
        parser.error("eval-every and log-every must be positive")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon must satisfy 0 <= end <= start <= 1")
    if args.epsilon_warmup_decisions < 0 or args.epsilon_decay_decisions <= 0:
        parser.error("epsilon decision schedule must be non-negative/positive")
    if args.max_defer_steps <= 0:
        parser.error("max-defer-steps must be positive")
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
            "Frozen selector regime does not match Track B training: "
            f"selector={selector_regime}, training={(args.lam, args.mu)}"
        )
    device = resolve_device(args.device)
    env, selector, agent = build_training_stack(args, selector_payload, device)
    history = []
    validations = []
    rolling_returns = deque(maxlen=25)
    best_score = (-1.0, -float("inf"))
    exploration_state = {"decisions": 0, "epsilon_decisions": 0}
    print(
        f"Track-B {args.controller_variant}+REG-v5 | device={device} | "
        f"seed={args.seed} | episodes={args.episodes} | selector=frozen | "
        f"observation={args.controller_observation}",
        flush=True,
    )

    for episode in range(1, args.episodes + 1):
        instance_seed = (
            args.training_instance_seed_base
            + args.seed * 1_000_000
            + episode
        )
        instance = env.sample_episode_instance(instance_seed)
        result = run_training_episode(
            agent,
            selector,
            env,
            instance,
            args.max_steps,
            exploration_state,
            args,
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
            score = (validation["success_rate"], validation["mean_return"])
            improved = score > best_score
            if improved:
                best_score = score
            payload = checkpoint_payload(
                agent,
                selector_payload,
                args,
                episode,
                validation,
                best_score=best_score,
                exploration_decisions=exploration_state["decisions"],
                epsilon_exploration_decisions=(
                    exploration_state["epsilon_decisions"]
                ),
            )
            torch.save(payload, args.output_dir / "latest.pth")
            if improved:
                torch.save(payload, args.output_dir / "best.pth")
            print(
                f"Ep {episode:4d} | TrainR {np.mean(rolling_returns):8.2f} | "
                f"ValR {validation['mean_return']:8.2f} | "
                f"ValSucc {validation['success_rate']:.3f} | "
                f"ValAcc {validation['inbound_successes']} | "
                f"ValRet {validation['retrieve_successes']} | "
                f"Eps {agent.epsilon:.3f} | "
                f"Dec {result['controller_decision_count']} | "
                f"Exp {result['epsilon_decision_count']} | "
                f"Acc {result['scheduler_audit']['inbound_successes']} | "
                f"Ret {result['scheduler_audit']['retrieve_successes']} | "
                f"Def {sum(result['scheduler_audit']['defer_outcomes'].values())} | "
                f"MgrL {result['manager_loss']} | WrkL {result['worker_loss']}",
                flush=True,
            )

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
            # Path objects in argparse config need explicit string conversion.
            summary["config"]["selector_checkpoint"] = str(
                args.selector_checkpoint.resolve()
            )
            summary["config"]["output_dir"] = str(args.output_dir.resolve())
            (args.output_dir / "training-summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n"
            )
        elif episode % args.log_every == 0 or episode == 1:
            print(
                f"Ep {episode:4d} | TrainR {np.mean(rolling_returns):8.2f} | "
                f"Succ {result['success']:.0f} | Eps {agent.epsilon:.3f} | "
                f"Dec {result['controller_decision_count']} | "
                f"Exp {result['epsilon_decision_count']} | "
                f"Acc {result['scheduler_audit']['inbound_successes']} | "
                f"Ret {result['scheduler_audit']['retrieve_successes']} | "
                f"Def {sum(result['scheduler_audit']['defer_outcomes'].values())} | "
                f"MgrL {result['manager_loss']} | WrkL {result['worker_loss']}",
                flush=True,
            )

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


if __name__ == "__main__":
    main()
