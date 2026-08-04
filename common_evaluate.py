#!/usr/bin/env python3
"""Greedy, seed-controlled evaluator for the full PSLAP comparison."""

import argparse
import csv
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
from example.helper.tools import flat
from options_agent import DQNAgent
from PSLAP.run_pslap import run_pslap_experiment


FIELDNAMES = [
    "method", "lambda", "mu", "train_seed", "eval_seed", "checkpoint_id",
    "return", "delivery_error", "success", "steps", "decision_seconds",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(lam, mu):
    return SmallRoomsEnv(choose_storage=False, arrival_rate=lam, proc_mean=mu)


def build_agent(env, method, checkpoint, assignment_path):
    for action in env.get_action_space():
        env.options.add(PrimitiveOption(action, env))
    option_classes = [PickupOption, PickupRipeOption, DeliverOption, StoreOption]
    for cls in option_classes:
        env.options.add(cls(env))
    if method == "hrl":
        env.options.add(StorageSelectOption(env))
    else:
        if not assignment_path:
            raise ValueError("GA evaluation requires --ga-assignment")
        env.options.add(GAStorageSelectOption(env, assignment_path))

    agent = DQNAgent(
        env=env,
        state_size=len(flat(env.reset())),
        action_size=len([o for o in env.options if o.is_primitive]),
        n_episodes=1,
        n_steps=4000,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent.Q_manager_local.load_state_dict(payload["manager_state_dict"])
    agent.Q_worker_local.load_state_dict(payload["worker_state_dict"])
    if method == "hrl":
        selector = next(o for o in env.options if isinstance(o, StorageSelectOption))
        selector_weights = payload["selector_state_dict"]
        hidden_1 = selector_weights["net.0.weight"].shape[0]
        hidden_2 = selector_weights["net.3.weight"].shape[0]
        selector.q = TinyQ(selector_weights["net.0.weight"].shape[1], selector.n_cells,
                           hidden_1, hidden_2).to(selector.device)
        selector.q.load_state_dict(selector_weights)
        selector.eps = 0.0
    agent.epsilon = 0.0
    agent.current_option = None
    agent.step_count = 0
    return agent


def evaluate_learned(method, lam, mu, eval_seed, checkpoint, assignment_path, max_steps):
    seed_everything(eval_seed)
    env = make_env(lam, mu)
    agent = build_agent(env, method, checkpoint, assignment_path)
    state = env.reset()
    total_return, errors = 0.0, []
    started = time.perf_counter()
    for step in range(max_steps):
        if agent.current_option is None:
            agent.current_option = agent.select_action(state, eps=0.0)
        action = agent.current_option.policy(state)
        next_state, reward, done, info = env.step(action)
        total_return += float(reward)
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        if agent.current_option.termination(next_state):
            agent.current_option = None
        state = next_state
        agent.step_count += 1
        if done:
            break
    return total_return, (float(np.mean(errors)) if errors else np.nan), float(done), step + 1, time.perf_counter() - started


def evaluate_pslap(lam, mu, eval_seed, max_steps):
    seed_everything(eval_seed)
    env = make_env(lam, mu)
    started = time.perf_counter()
    logs = run_pslap_experiment(env, n_episodes=1, max_steps=max_steps)
    elapsed = time.perf_counter() - started
    return (logs["episode_returns"][0], logs["episode_avg_error"][0],
            logs["episode_success"][0], max_steps, elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("hrl", "ga", "pslap"), required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--train-seed", type=int, default=-1)
    parser.add_argument("--checkpoint")
    parser.add_argument("--ga-assignment")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.method == "pslap":
        result = evaluate_pslap(args.lam, args.mu, args.eval_seed, args.max_steps)
        checkpoint_id = "not_applicable"
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for hrl and ga")
        result = evaluate_learned(args.method, args.lam, args.mu, args.eval_seed,
                                  args.checkpoint, args.ga_assignment, args.max_steps)
        checkpoint_id = os.path.basename(args.checkpoint)

    row = dict(zip(FIELDNAMES, [args.method, args.lam, args.mu, args.train_seed,
                                args.eval_seed, checkpoint_id, *result]))
    new_file = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
