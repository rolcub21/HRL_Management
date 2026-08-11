#!/usr/bin/env python3
"""Small matched-seed comparison for independent PSLAP-family baselines."""

import argparse
import csv
import random

import numpy as np

from example.helper.timing_metrics import summarize_delivery_timing
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.baselines import (
    ACCEPTED_BASELINES,
    BASELINE_INFORMATION,
    DYNAMIC_PSLAP,
    PSLAP_GA_2009_OFFLINE,
    PSLAP_GA_2009_ROLLING,
    normalize_baseline,
)
from PSLAP.run_pslap import run_pslap_episode


FIELDS = [
    "method",
    "information_regime",
    "lambda",
    "mu",
    "seed",
    "instance_id",
    "exit_width",
    "return",
    "success",
    "steps",
    "delivery_count",
    "mean_signed_deviation",
    "mean_absolute_error",
    "within_target_window_rate",
    "tardy_delivery_rate",
    "obstructive_moves",
    "illegal_drops",
    "assignment_fallbacks",
    "ga_predicted_infeasible_events",
    "ga_predicted_obstructive_moves",
    "ga_predicted_route_steps",
]


def make_env(lam, mu, exit_width):
    kwargs = {
        "choose_storage": False,
        "arrival_rate": lam,
        "proc_mean": mu,
    }
    if exit_width is not None:
        if not 1 <= exit_width <= 8:
            raise ValueError("exit_width must be in [1, 8]")
        right = 8
        left = right - exit_width + 1
        kwargs["exit_cells"] = [(9, col) for col in range(left, right + 1)]
    return SmallRoomsEnv(**kwargs)


def evaluate(
    method, lam, mu, seed, exit_width, max_steps, episode_instance
):
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(lam, mu, exit_width)
    result = run_pslap_episode(
        env,
        max_steps=max_steps,
        baseline=method,
        episode_instance=episode_instance,
    )
    timing = summarize_delivery_timing(
        result["errors"], env.DELIVERY_TARGET_WINDOW
    )
    return {
        "method": method,
        "information_regime": BASELINE_INFORMATION[method],
        "lambda": lam,
        "mu": mu,
        "seed": seed,
        "instance_id": episode_instance.instance_id,
        "exit_width": "environment_default" if exit_width is None else exit_width,
        "return": result["return"],
        "success": result["success"],
        "steps": result["steps"],
        "delivery_count": timing["delivery_count"],
        "mean_signed_deviation": timing["mean_signed_deviation"],
        "mean_absolute_error": timing["mean_absolute_error"],
        "within_target_window_rate": timing["within_target_window_rate"],
        "tardy_delivery_rate": timing["tardy_delivery_rate"],
        "obstructive_moves": result["obstructive_moves"],
        "illegal_drops": result["illegal_drops"],
        "assignment_fallbacks": result["assignment_fallbacks"],
        "ga_predicted_infeasible_events": result[
            "ga_predicted_infeasible_events"
        ],
        "ga_predicted_obstructive_moves": result[
            "ga_predicted_obstructive_moves"
        ],
        "ga_predicted_route_steps": result["ga_predicted_route_steps"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ACCEPTED_BASELINES,
        default=[
            DYNAMIC_PSLAP,
            PSLAP_GA_2009_ROLLING,
            PSLAP_GA_2009_OFFLINE,
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=80.0)
    parser.add_argument(
        "--exit-width",
        type=int,
        help="right-aligned bottom gate width; omit for environment default",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    instances = {
        seed: make_env(args.lam, args.mu, args.exit_width)
        .sample_episode_instance(seed)
        for seed in args.seeds
    }
    methods = tuple(normalize_baseline(method) for method in args.methods)
    rows = [
        evaluate(
            method,
            args.lam,
            args.mu,
            seed,
            args.exit_width,
            args.max_steps,
            instances[seed],
        )
        for method in methods
        for seed in args.seeds
    ]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
