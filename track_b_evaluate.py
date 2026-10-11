#!/usr/bin/env python3
"""Matched-instance Track B evaluation with a frozen REG-v5 selector."""

from __future__ import annotations

import argparse
from collections import Counter
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
    controller_observation_encoder_from_checkpoint,
)
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import (
    summarize_block_storage_flow,
    summarize_delivery_timing,
    summarize_storage_flow_runs,
)
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import (
    ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE,
    ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
    FOUNDATION_CONTROLLER_ARCHITECTURE,
    FULL_REGULARIZED_CONTROLLER_ARCHITECTURE,
    MODE_ONLY_CONTROLLER_ARCHITECTURE,
    SCHEDULER_CONTROLLER_ARCHITECTURE,
    SUPPORTED_CONTROLLER_ARCHITECTURES,
    GatedModeOnlyAgent,
    GatedModeOnlyFoundationAgent,
    GatedAtomicInboundSchedulerAgent,
    GatedRegularizedAgent,
    GatedSchedulingAgent,
    LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE,
    SCHEDULER_CONTROLLER_ACTION_INTERFACE,
)
from options_agent import resolve_q_network_config, validate_checkpoint_metadata
from PSLAP.checkpoint_identity import selector_deployment_digest


METHOD = "gated_regularized_reg_v5_frozen"


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


def load_instance(path: str | None, env, seed: int) -> EpisodeInstance:
    if path is None:
        return env.sample_episode_instance(seed)
    instance = EpisodeInstance.from_json(Path(path).read_text())
    if instance.seed is not None and instance.seed != seed:
        raise ValueError("instance seed does not match evaluation seed")
    instance.validate_for(env)
    return instance


def build_agent(
    env,
    initial_state,
    controller_payload,
    selector_payload,
    *,
    seed,
    device,
    policy,
    tau_option,
    tau_primitive,
    tau_mode,
    allow_legacy_controller,
):
    saved_architecture = controller_payload.get("controller_architecture")
    if saved_architecture is None:
        if not allow_legacy_controller:
            raise ValueError(
                "Controller checkpoint has no architecture metadata"
            )
        runtime_architecture = FULL_REGULARIZED_CONTROLLER_ARCHITECTURE
    elif saved_architecture not in SUPPORTED_CONTROLLER_ARCHITECTURES:
        raise ValueError(
            f"unsupported controller architecture: {saved_architecture!r}"
        )
    else:
        runtime_architecture = saved_architecture
    if (
        runtime_architecture == ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE
        and policy != "map"
    ):
        raise ValueError(
            "Atomic inbound scheduler has one active macro mode and must be "
            "evaluated with policy='map'"
        )
    agent_class = {
        ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE: (
            GatedAtomicInboundSchedulerAgent
        ),
        SCHEDULER_CONTROLLER_ARCHITECTURE: GatedSchedulingAgent,
        FOUNDATION_CONTROLLER_ARCHITECTURE: GatedModeOnlyFoundationAgent,
        MODE_ONLY_CONTROLLER_ARCHITECTURE: GatedModeOnlyAgent,
        FULL_REGULARIZED_CONTROLLER_ARCHITECTURE: GatedRegularizedAgent,
    }[runtime_architecture]
    network_config = resolve_q_network_config(controller_payload)
    selector = StorageSelectOptionV5.from_checkpoint(
        env,
        selector_payload,
        device=device,
        seed=seed,
        learning_enabled=False,
    )
    retrieval_definition = controller_payload.get(
        "retrieval_duration_definition", "legacy_movement_only"
    )
    if retrieval_definition not in (
        "legacy_movement_only",
        "complete_plan_v1",
        "atomic_named_retrieval_v1",
    ):
        raise ValueError(
            f"Unknown retrieval duration definition: {retrieval_definition!r}"
        )
    action_interface = controller_payload.get(
        "controller_action_interface",
        LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE,
    )
    if runtime_architecture in (
        SCHEDULER_CONTROLLER_ARCHITECTURE,
        ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
    ):
        scheduler_contract = {
            "block_identity_contract": "episode_manifest_slot_v1",
            "block_count": len(env.blocks),
            "block_slot_ids": tuple(block.label for block in env.blocks),
            "retrieve_deliver_option_version": "retrieve_deliver_option_v1",
            "retrieval_executor_version": "named_atomic_retrieval_executor_v1",
            "retrieval_termination_contract": (
                "complete_delivery_or_safe_explicit_failure_v1"
            ),
            "defer_option_version": "strategic_defer_until_event_v1",
            "relocation_selector_version": (
                "nearest_feasible_path_then_cell_v1"
            ),
            "retrieval_duration_definition": "atomic_named_retrieval_v1",
            "controller_observation_version": "online_manifest_timing_v3",
            "controller_observation_block_row_contract": (
                "episode_manifest_slot_v1"
            ),
            "selector_deployment_digest": selector_deployment_digest(
                selector_payload
            ),
        }
        if runtime_architecture == ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE:
            scheduler_contract.update(
                {
                    "controller_action_interface": (
                        ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
                    ),
                    "controller_backup": (
                        "masked_macro_smdp_optimality_v1"
                    ),
                    "mode_aggregation": (
                        "single_macro_mode_identity_v1"
                    ),
                    "decision_epoch_contract": (
                        "mandatory_inbound_then_schedule_v1"
                    ),
                    "scheduler_decision_state_contract": (
                        "forced_inbound_else_retrieve_or_defer_v1"
                    ),
                    "primitive_decisions_enabled": False,
                    "accept_store_option_version": (
                        "accept_store_option_v1"
                    ),
                    "selector_assignment_duration": (
                        "zero_environment_steps_v1"
                    ),
                }
            )
        else:
            scheduler_contract["controller_action_interface"] = (
                SCHEDULER_CONTROLLER_ACTION_INTERFACE
            )
        mismatches = {
            key: controller_payload.get(key)
            for key, expected in scheduler_contract.items()
            if controller_payload.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Scheduler checkpoint contract mismatch: "
                f"{mismatches!r}"
            )
    elif action_interface != LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE:
        raise ValueError(
            "Legacy controller architecture cannot use action interface "
            f"{action_interface!r}"
        )
    max_defer_steps = int(controller_payload.get("max_defer_steps", 10))
    build_controller_options(
        env,
        selector,
        retrieval_lead_time=float(
            controller_payload.get("retrieval_lead_time", 20.0)
        ),
        include_retrieval_handling_steps=(
            retrieval_definition == "complete_plan_v1"
        ),
        controller_action_interface=action_interface,
        max_defer_steps=max_defer_steps,
    )
    observation_encoder = controller_observation_encoder_from_checkpoint(
        env, controller_payload
    )
    initial_features = observation_encoder(initial_state)
    agent = agent_class(
        env=env,
        state_size=len(initial_features),
        action_size=len([option for option in env.options if option.is_primitive]),
        n_episodes=1,
        n_steps=1,
        gamma=float(controller_payload.get("gamma", 0.99)),
        tau_option=tau_option,
        tau_primitive=tau_primitive,
        tau_mode=tau_mode,
        training_policy=policy,
        seed=network_config["controller_seed"],
        network_normalization=network_config["normalization"],
        dueling_centering=network_config["dueling_centering"],
        state_encoder=observation_encoder,
        controller_observation_metadata=observation_encoder.metadata(),
        wait_training_penalty=float(
            controller_payload.get("wait_training_penalty", 0.5)
        ),
        terminal_on_truncation=bool(
            controller_payload.get("terminal_on_truncation", False)
        ),
        close_options_on_episode_end=bool(
            controller_payload.get("close_options_on_episode_end", False)
        ),
        disable_tensorboard=True,
        device=device,
        verbose=False,
        max_defer_steps=max_defer_steps,
    )
    validate_checkpoint_metadata(
        controller_payload,
        agent.manager_options,
        agent.primitive_options,
        selector=None,
        allow_legacy=allow_legacy_controller,
    )
    agent.Q_manager_local.load_state_dict(
        controller_payload["manager_state_dict"]
    )
    agent.Q_manager_target.load_state_dict(
        controller_payload["manager_state_dict"]
    )
    agent.Q_worker_local.load_state_dict(controller_payload["worker_state_dict"])
    agent.Q_worker_target.load_state_dict(controller_payload["worker_state_dict"])
    agent.epsilon = 0.0
    agent.current_option = None
    return agent, selector


def normalized_gate_audit(counter: Counter) -> dict:
    result = dict(counter)
    scored = result.get("scored_states", 0)
    for source, destination in (
        ("u_option_sum", "mean_u_option"),
        ("u_primitive_sum", "mean_u_primitive"),
        ("option_probability_sum", "mean_option_probability"),
    ):
        result[destination] = (
            result.get(source, 0.0) / scored if scored else None
        )
    return result


def evaluate_one(
    args,
    seed,
    controller_payload,
    selector_payload,
    *,
    episode_instance=None,
):
    seed_everything(seed)
    env = SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=args.lam,
        proc_mean=args.mu,
    )
    instance = (
        load_instance(args.instance, env, seed)
        if episode_instance is None
        else episode_instance
    )
    instance.validate_for(env)
    initial_state = env.reset(instance=instance)
    device = resolve_device(args.device)
    agent, selector = build_agent(
        env,
        initial_state,
        controller_payload,
        selector_payload,
        seed=seed,
        device=device,
        policy=args.policy,
        tau_option=args.tau_option,
        tau_primitive=args.tau_primitive,
        tau_mode=args.tau_mode,
        allow_legacy_controller=args.allow_legacy_controller,
    )
    # Reset again after attaching options so every option receives its reset hook.
    state = env.reset(instance=instance)
    # Model/option construction initializes temporary parameters before loading
    # their checkpoint values and therefore consumes Python/Torch RNG state.
    # Establish the rollout seed only after construction so stochastic policy
    # realizations exactly reproduce train_track_b.run_validation_episode.
    seed_everything(seed)
    agent.current_option = None
    agent.decision_counts.clear()
    agent.control_decisions.clear()
    agent.gate_decisions.clear()
    errors = []
    total_return = 0.0
    obstructive_moves = 0
    illegal_drops = 0
    done = False
    started = perf_counter()
    for step in range(args.max_steps):
        if agent.current_option is None:
            agent.current_option = agent.select_action(state, eps=0.0)
        action = agent.current_option.policy(state)
        next_state, reward, done, info = env.step(action)
        selector.on_step(reward, info)
        total_return += float(reward)
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        obstructive_moves += int(bool(info.get("relocated_block")))
        illegal_drops += int(bool(info.get("illegal_drop")))
        if agent.current_option.termination(next_state):
            agent.current_option = None
        state = next_state
        agent.step_count += 1
        if done:
            break
    elapsed = perf_counter() - started
    steps = step + 1
    selector.on_episode_end(
        success=done,
        truncated=bool(steps >= args.max_steps and not done),
    )
    timing = summarize_delivery_timing(errors, args.target_window)
    storage_flow = summarize_block_storage_flow(env.blocks, env.time_steps)
    if args.save_instances_dir:
        destination = Path(args.save_instances_dir) / f"seed-{seed}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(instance.to_json() + "\n")
    return {
        "method": METHOD,
        "track": "B",
        "evaluation_scope": "frozen_zero_shot_adapter",
        "lambda": args.lam,
        "mu": args.mu,
        "eval_seed": seed,
        "instance_id": instance.instance_id,
        "return": total_return,
        "success": float(done),
        "steps": steps,
        "delivery_count": len(errors),
        "decision_seconds": elapsed,
        "obstructive_moves": obstructive_moves,
        "illegal_drops": illegal_drops,
        **timing,
        **storage_flow,
        "target_window": args.target_window,
        "delivery_deviations": errors,
        "controller_decisions": dict(agent.decision_counts),
        "controller_control_decisions": dict(agent.control_decisions),
        "scheduler_audit": scheduler_episode_audit(env),
        "controller_observation_version": agent.controller_observation_metadata.get(
            "controller_observation_version", "unversioned_legacy"
        ),
        "gate_audit": normalized_gate_audit(agent.gate_decisions),
        "selector_audit": selector.audit(),
    }


def summarize(runs):
    returns = np.asarray([run["return"] for run in runs], dtype=float)
    success = np.asarray([run["success"] for run in runs], dtype=float)
    delivered = np.asarray([run["delivery_count"] for run in runs], dtype=float)
    return {
        "n": len(runs),
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(runs) > 1 else 0.0,
        "success_rate": float(success.mean()),
        "mean_delivery_count": float(delivered.mean()),
        "mean_steps": float(np.mean([run["steps"] for run in runs])),
        "selector_invalid_assignments": int(
            sum(
                run["selector_audit"]["invalid_assignment_count"] for run in runs
            )
        ),
        "selector_infeasible_epochs": int(
            sum(run["selector_audit"]["infeasible_epoch_count"] for run in runs)
        ),
        **summarize_storage_flow_runs(runs),
    }


def json_safe(value):
    """Replace non-finite scalar metrics with JSON ``null`` recursively."""

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
    parser.add_argument("--controller-checkpoint", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--instance",
        help="exact EpisodeInstance JSON; allowed only with one evaluation seed",
    )
    parser.add_argument("--save-instances-dir")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument(
        "--policy",
        choices=("map", "mode_regularized", "regularized"),
        help="defaults to the checkpoint validation policy",
    )
    parser.add_argument("--tau-option", type=float)
    parser.add_argument("--tau-primitive", type=float)
    parser.add_argument("--tau-mode", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-legacy-controller",
        action="store_true",
        help="accept a controller checkpoint without the current gated metadata",
    )
    parser.add_argument("--controller-training-lambda", type=float)
    parser.add_argument("--controller-training-mu", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.instance and len(args.eval_seeds) != 1:
        parser.error("--instance requires exactly one --eval-seeds value")
    return args


def main():
    args = parse_args()
    controller_payload = torch.load(
        args.controller_checkpoint, map_location="cpu", weights_only=False
    )
    mode_only_architectures = {
        ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
        MODE_ONLY_CONTROLLER_ARCHITECTURE,
        FOUNDATION_CONTROLLER_ARCHITECTURE,
        SCHEDULER_CONTROLLER_ARCHITECTURE,
    }
    args.policy = args.policy or controller_payload.get(
        "validation_policy",
        "map"
        if controller_payload.get("controller_architecture")
        == ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE
        else "mode_regularized"
        if controller_payload.get("controller_architecture") in mode_only_architectures
        else "regularized",
    )
    args.tau_option = (
        float(args.tau_option)
        if args.tau_option is not None
        else float(controller_payload.get("tau_option", 0.1))
    )
    args.tau_primitive = (
        float(args.tau_primitive)
        if args.tau_primitive is not None
        else float(controller_payload.get("tau_primitive", 0.1))
    )
    args.tau_mode = (
        float(args.tau_mode)
        if args.tau_mode is not None
        else float(controller_payload.get("tau_mode", 1.0))
    )
    selector_payload = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    runs = [
        evaluate_one(args, seed, controller_payload, selector_payload)
        for seed in args.eval_seeds
    ]
    selector_regime = {
        "lambda": selector_payload.get("training_lambda"),
        "mu": selector_payload.get("training_mu"),
    }
    controller_regime = {
        "lambda": args.controller_training_lambda,
        "mu": args.controller_training_mu,
    }
    payload = {
        "config": {
            "method": METHOD,
            "controller_checkpoint": str(Path(args.controller_checkpoint).resolve()),
            "selector_checkpoint": str(Path(args.selector_checkpoint).resolve()),
            "controller_checkpoint_architecture": controller_payload.get(
                "controller_architecture", "unverified_legacy"
            ),
            "runtime_controller_architecture": controller_payload.get(
                "controller_architecture",
                FULL_REGULARIZED_CONTROLLER_ARCHITECTURE,
            ),
            "controller_training_regime": controller_regime,
            "selector_training_regime": selector_regime,
            "evaluation_regime": {"lambda": args.lam, "mu": args.mu},
            "controller_regime_matches_evaluation": controller_regime
            == {"lambda": args.lam, "mu": args.mu},
            "selector_regime_matches_evaluation": selector_regime
            == {"lambda": args.lam, "mu": args.mu},
            "policy": args.policy,
            "tau_option": args.tau_option,
            "tau_primitive": args.tau_primitive,
            "tau_mode": args.tau_mode,
            "eval_seeds": args.eval_seeds,
            "max_steps": args.max_steps,
            "device": str(resolve_device(args.device)),
            "frozen_selector": True,
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(payload)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
