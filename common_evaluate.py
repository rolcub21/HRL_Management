#!/usr/bin/env python3
"""Greedy, seed-controlled evaluator for the full PSLAP comparison."""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from primitive_option import PrimitiveOption
from example.Options.storeOption import StoreOption
from example.Options.DeliverOption import DeliverOption
from example.Options.pickupOption import PickupOption
from example.Options.PickupRipeOption import PickupRipeOption
from example.Options.selector import StorageSelectOption, TinyQ
from example.Options.GAStorageSelectOption import GAStorageSelectOption
from example.episode_instance import EpisodeInstance
from example.helper.tools import flat
from example.helper.timing_metrics import summarize_delivery_timing
from options_agent import DQNAgent, validate_checkpoint_metadata
from PSLAP.baselines import (
    ACCEPTED_BASELINES,
    BASELINE_INFORMATION,
    normalize_baseline,
)
from PSLAP.run_pslap import run_pslap_experiment


FIELDNAMES = [
    "method", "information_regime", "lambda", "mu", "train_seed",
    "eval_seed", "instance_id",
    "checkpoint_id",
    "return", "delivery_error", "success", "steps", "decision_seconds",
    "obstructive_moves", "illegal_drops", "assignment_fallbacks",
    "delivery_count", "target_window", "mean_signed_deviation",
    "mean_absolute_error", "mean_tardiness", "mean_earliness",
    "within_target_window_rate", "tardy_delivery_rate",
    "mean_tardiness_when_tardy", "p90_tardiness", "p90_absolute_error",
    "signed_deviation_std", "absolute_error_std", "delivery_deviations",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(lam, mu):
    return SmallRoomsEnv(choose_storage=False, arrival_rate=lam, proc_mean=mu)


def format_evaluation_result(
    total_return, errors, success, steps, elapsed, target_window,
    obstructive_moves=0, illegal_drops=0, instance_id=None,
    assignment_fallbacks=0,
):
    """Build one evaluation row without conflating bias and accuracy."""
    errors = [float(value) for value in errors]
    timing = summarize_delivery_timing(errors, target_window)
    return {
        "return": float(total_return),
        # Backward-compatible alias. It is a signed bias, not an error norm.
        "delivery_error": timing["mean_signed_deviation"],
        "success": float(success),
        "steps": int(steps),
        "decision_seconds": float(elapsed),
        "instance_id": instance_id,
        "obstructive_moves": int(obstructive_moves),
        "illegal_drops": int(illegal_drops),
        "assignment_fallbacks": int(assignment_fallbacks),
        **timing,
        "target_window": float(target_window),
        "delivery_deviations": json.dumps(errors, separators=(",", ":")),
    }


def build_agent(
    env,
    method,
    checkpoint,
    assignment_path,
    initial_state,
    allow_legacy_checkpoint=False,
):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for action in env.get_action_space():
        env.options.add(PrimitiveOption(action, env))
    option_classes = [PickupOption, PickupRipeOption, DeliverOption, StoreOption]
    for cls in option_classes:
        env.options.add(cls(env))
    if method == "hrl":
        saved_definition = payload.get("selector_return_definition")
        known_definitions = {
            (
                f"selector_event_{mode}_v"
                f"{StorageSelectOption.RETURN_DEFINITION_VERSION}"
            ): mode
            for mode in StorageSelectOption.RETURN_MODES
        }
        if saved_definition is None:
            selector_return_mode = StorageSelectOption.RETURN_FULL_ENVIRONMENT
        elif saved_definition in known_definitions:
            selector_return_mode = known_definitions[saved_definition]
        else:
            raise ValueError(
                "Unknown checkpoint selector return definition: "
                f"{saved_definition!r}"
            )
        selector_gamma = float(payload.get("selector_gamma", 0.99))
        env.options.add(
            StorageSelectOption(
                env,
                gamma=selector_gamma,
                return_mode=selector_return_mode,
            )
        )
    else:
        if not assignment_path:
            raise ValueError("GA evaluation requires --ga-assignment")
        env.options.add(GAStorageSelectOption(env, assignment_path))

    agent = DQNAgent(
        env=env,
        state_size=len(flat(initial_state)),
        action_size=len([o for o in env.options if o.is_primitive]),
        n_episodes=1,
        n_steps=4000,
        gamma=(selector_gamma if method == "hrl" else 0.99),
        disable_tensorboard=True,
    )
    selector = (
        next(o for o in env.options if isinstance(o, StorageSelectOption))
        if method == "hrl"
        else None
    )
    validate_checkpoint_metadata(
        payload,
        agent.manager_options,
        agent.primitive_options,
        selector,
        allow_legacy=allow_legacy_checkpoint,
    )
    agent.Q_manager_local.load_state_dict(payload["manager_state_dict"])
    agent.Q_worker_local.load_state_dict(payload["worker_state_dict"])
    if method == "hrl":
        selector_weights = payload["selector_state_dict"]
        hidden_1 = selector_weights["net.0.weight"].shape[0]
        hidden_2 = selector_weights["net.3.weight"].shape[0]
        selector.q = TinyQ(selector_weights["net.0.weight"].shape[1], selector.n_cells,
                           hidden_1, hidden_2).to(selector.device)
        selector.q.load_state_dict(selector_weights)
        selector.set_learning_enabled(False)
    agent.epsilon = 0.0
    agent.current_option = None
    agent.step_count = 0
    return agent


def evaluate_learned(
    method, lam, mu, eval_seed, checkpoint, assignment_path, max_steps,
    target_window, allow_legacy_checkpoint=False, episode_instance=None,
):
    seed_everything(eval_seed)
    env = make_env(lam, mu)
    instance = (
        env.sample_episode_instance(eval_seed)
        if episode_instance is None
        else episode_instance
    )
    state = env.reset(instance=instance)
    agent = build_agent(
        env,
        method,
        checkpoint,
        assignment_path,
        state,
        allow_legacy_checkpoint,
    )
    total_return, errors = 0.0, []
    obstructive_moves = 0
    illegal_drops = 0
    started = time.perf_counter()
    for step in range(max_steps):
        if agent.current_option is None:
            agent.current_option = agent.select_action(state, eps=0.0)
        action = agent.current_option.policy(state)
        next_state, reward, done, info = env.step(action)
        total_return += float(reward)
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        if info.get("relocated_block"):
            obstructive_moves += 1
        if info.get("illegal_drop"):
            illegal_drops += 1
        if agent.current_option.termination(next_state):
            agent.current_option = None
        state = next_state
        agent.step_count += 1
        if done:
            break
    return format_evaluation_result(
        total_return,
        errors,
        done,
        step + 1,
        time.perf_counter() - started,
        target_window,
        obstructive_moves,
        illegal_drops,
        instance.instance_id,
    )


def evaluate_pslap(
    lam, mu, eval_seed, max_steps, target_window, baseline,
    episode_instance=None,
):
    seed_everything(eval_seed)
    env = make_env(lam, mu)
    instance = (
        env.sample_episode_instance(eval_seed)
        if episode_instance is None
        else episode_instance
    )
    started = time.perf_counter()
    logs = run_pslap_experiment(
        env,
        n_episodes=1,
        max_steps=max_steps,
        baseline=baseline,
        episode_instances=(instance,),
    )
    elapsed = time.perf_counter() - started
    errors = [float(record[1]) for record in env.delivery_error_times]
    return format_evaluation_result(
        logs["episode_returns"][0],
        errors,
        logs["episode_success"][0],
        env.time_steps,
        elapsed,
        target_window,
        logs["episode_obstructive_moves"][0],
        logs["episode_illegal_drops"][0],
        instance.instance_id,
        logs["episode_assignment_fallbacks"][0],
    )


def main():
    parser = argparse.ArgumentParser()
    pslap_methods = ACCEPTED_BASELINES
    parser.add_argument(
        "--method",
        choices=("hrl", "ga") + pslap_methods,
        required=True,
    )
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument(
        "--instance",
        help="JSON EpisodeInstance to replay instead of sampling from eval seed",
    )
    parser.add_argument(
        "--save-instance",
        help="write the exact sampled/replayed EpisodeInstance to this JSON path",
    )
    parser.add_argument("--train-seed", type=int, default=-1)
    parser.add_argument("--checkpoint")
    parser.add_argument("--ga-assignment")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help=(
            "load checkpoints without complete controller/selector metadata; "
            "results are not guaranteed comparable"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
        help="absolute delivery-deviation tolerance used only for reporting",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.instance:
        with open(args.instance) as handle:
            episode_instance = EpisodeInstance.from_json(handle.read())
        if (
            episode_instance.seed is not None
            and episode_instance.seed != args.eval_seed
        ):
            parser.error(
                "--eval-seed must match the seed recorded in --instance"
            )
    else:
        episode_instance = make_env(args.lam, args.mu).sample_episode_instance(
            args.eval_seed
        )

    # Validate before checkpoint loading or policy execution so a mismatched
    # experimental condition fails without producing a partial result.
    episode_instance.validate_for(make_env(args.lam, args.mu))
    if args.save_instance:
        with open(args.save_instance, "w") as handle:
            handle.write(episode_instance.to_json() + "\n")

    canonical_method = args.method
    if args.method in pslap_methods:
        canonical_method = normalize_baseline(args.method)
        result = evaluate_pslap(
            args.lam,
            args.mu,
            args.eval_seed,
            args.max_steps,
            args.target_window,
            canonical_method,
            episode_instance,
        )
        checkpoint_id = "not_applicable"
        information_regime = BASELINE_INFORMATION[canonical_method]
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for hrl and ga")
        result = evaluate_learned(args.method, args.lam, args.mu, args.eval_seed,
                                  args.checkpoint, args.ga_assignment, args.max_steps,
                                  args.target_window, args.allow_legacy_checkpoint,
                                  episode_instance)
        checkpoint_id = os.path.basename(args.checkpoint)
        information_regime = "online"

    row = {
        "method": canonical_method,
        "information_regime": information_regime,
        "lambda": args.lam,
        "mu": args.mu,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "checkpoint_id": checkpoint_id,
        **result,
    }
    new_file = not os.path.exists(args.output)
    if not new_file:
        with open(args.output, newline="") as handle:
            existing_header = next(csv.reader(handle), None)
        if existing_header != FIELDNAMES:
            raise ValueError(
                "Existing evaluation CSV uses a different schema; choose a "
                "new --output path"
            )
    with open(args.output, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
