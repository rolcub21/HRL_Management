#!/usr/bin/env python3
"""Cheap dynamics/interface smoke test; this does not train an executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "farama_modified"))
from reach_goal_ant import GoalReachAnt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command-distance-bank",
        type=Path,
        default=HERE / "artifacts/command_distance_bank_r47.npz",
    )
    args = parser.parse_args()
    env = GoalReachAnt(command_distance_bank=str(args.command_distance_bank.resolve()))
    events = []
    try:
        observation, _ = env.reset(seed=549)
        if observation.shape != (29,) or env.action_space.shape != (8,):
            raise ValueError("E549 spaces changed")
        for _ in range(205):
            action = np.zeros(8, dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise ValueError("E549 smoke produced non-finite values")
            if info["command_event"] is not None:
                events.append(info["command_event"])
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(events) < 2 or any(int(event["duration"]) > 100 for event in events):
        raise ValueError("E549 did not enforce the 100-step command contract")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "observation_shape": [29],
                "action_shape": [8],
                "steps_executed": 205,
                "command_events": len(events),
                "event_durations": [int(event["duration"]) for event in events],
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

