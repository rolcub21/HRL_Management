#!/usr/bin/env python3
"""Executor-only adaptive chained-command screen for E549.

Unlike ``evaluate.py``, this screen does not replay an absolute target
sequence.  After every command, the environment samples the next command from
the endpoint the executor actually reached.  This is the command-handoff
contract used during E549 training and matches TF-SN's use of the executor:
every new command begins at the live state rather than at an imagined path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import SAC


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "farama_modified"))
from reach_goal_ant import GoalReachAnt  # noqa: E402


PROJECT = HERE.parents[1]
DEFAULT_BANK = HERE / "artifacts/command_distance_bank_r47.npz"
DEFAULT_CHECKPOINT = (
    PROJECT
    / "results/experiments/549_antmaze_chained_sac_executor/checkpoints"
    / "GoalAntMazeChained_model_5000000_steps.zip"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "results/experiments/549_antmaze_chained_sac_executor"
    / "evaluation_5m_adaptive"
)
PROTOCOL = HERE / "protocol.json"
RESET_SEEDS = list(range(549_000, 549_032))
COMMANDS_PER_EPISODE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def position_summary(rows: list[dict]) -> list[dict]:
    result = []
    for command_index in range(COMMANDS_PER_EPISODE):
        selected = [row for row in rows if row["command_index"] == command_index]
        if not selected:
            raise ValueError(f"No rows for command position {command_index}")
        result.append(
            {
                "command_index_zero_based": command_index,
                "command_number_one_based": command_index + 1,
                "commands": len(selected),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "inversion_rate": float(np.mean([row["inverted"] for row in selected])),
                "requested_distance_mean_m": float(
                    np.mean([row["requested_distance_m"] for row in selected])
                ),
                "endpoint_error_mean_m": float(
                    np.mean([row["endpoint_error_m"] for row in selected])
                ),
                "displacement_mean_m": float(
                    np.mean([row["displacement_m"] for row in selected])
                ),
            }
        )
    return result


def distance_summary(rows: list[dict]) -> tuple[list[dict], list[float]]:
    """Report empirical quintiles; the bands introduce no method setting."""
    distance = np.asarray([row["requested_distance_m"] for row in rows], dtype=np.float64)
    boundaries = np.quantile(distance, np.linspace(0.0, 1.0, 6))
    result = []
    for band in range(5):
        low = float(boundaries[band])
        high = float(boundaries[band + 1])
        if band < 4:
            mask = (distance >= low) & (distance < high)
        else:
            mask = (distance >= low) & (distance <= high)
        indices = np.flatnonzero(mask)
        selected = [rows[int(index)] for index in indices]
        result.append(
            {
                "empirical_quintile": band + 1,
                "low_m_inclusive": low,
                "high_m_inclusive_only_for_last_band": high,
                "commands": len(selected),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "inversion_rate": float(np.mean([row["inverted"] for row in selected])),
                "endpoint_error_mean_m": float(
                    np.mean([row["endpoint_error_m"] for row in selected])
                ),
                "displacement_mean_m": float(
                    np.mean([row["displacement_m"] for row in selected])
                ),
            }
        )
    return result, [float(value) for value in boundaries]


def evaluate(
    checkpoint: Path,
    bank: Path,
    output: Path,
    device: str,
    reset_seeds: list[int],
) -> dict:
    model = SAC.load(str(checkpoint), device=device)
    if tuple(model.observation_space.shape) != (29,):
        raise ValueError("The E549 checkpoint has the wrong observation space")
    if tuple(model.action_space.shape) != (8,):
        raise ValueError("The E549 checkpoint has the wrong action space")

    rows: list[dict] = []
    episodes: list[dict] = []
    for episode, seed in enumerate(reset_seeds):
        env = GoalReachAnt(command_distance_bank=str(bank))
        try:
            observation, _ = env.reset(seed=seed)
            primitive_steps = 0
            events = []
            while len(events) < COMMANDS_PER_EPISODE and primitive_steps < 1000:
                action, _ = model.predict(observation, deterministic=True)
                observation, _reward, terminated, truncated, info = env.step(action)
                primitive_steps += 1
                event = info["command_event"]
                if event is not None:
                    row = {
                        "model": "e549_5m",
                        "episode": episode,
                        "reset_seed": seed,
                        "command_index": int(event["command_index"]),
                        "reason": str(event["reason"]),
                        "duration": int(event["duration"]),
                        "start": np.asarray(event["start"]).tolist(),
                        "goal": np.asarray(event["goal"]).tolist(),
                        "endpoint": np.asarray(event["endpoint"]).tolist(),
                        "requested_distance_m": float(
                            np.linalg.norm(
                                np.asarray(event["goal"]) - np.asarray(event["start"])
                            )
                        ),
                        "endpoint_error_m": float(event["endpoint_error_m"]),
                        "displacement_m": float(event["displacement_m"]),
                        "torso_height_m": float(env.torso_height),
                        "torso_uprightness": float(env.torso_uprightness),
                        # Diagnostic only; never an executor or planner rule.
                        "inverted": bool(env.torso_uprightness < 0.0),
                        "success": event["reason"] == "reached",
                    }
                    rows.append(row)
                    events.append(row)
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "episode": episode,
                    "reset_seed": seed,
                    "commands": len(events),
                    "primitive_steps": primitive_steps,
                    "any_inversion": any(row["inverted"] for row in events),
                }
            )
            print(
                f"e549 adaptive episode={episode + 1}/{len(reset_seeds)} "
                f"commands={len(events)} success={sum(row['success'] for row in events)} "
                f"flipped={int(any(row['inverted'] for row in events))}",
                flush=True,
            )
        finally:
            env.close()

    expected = len(reset_seeds) * COMMANDS_PER_EPISODE
    if len(rows) != expected:
        raise ValueError(f"Adaptive screen produced {len(rows)} rows, expected {expected}")

    by_position = position_summary(rows)
    by_distance, distance_boundaries = distance_summary(rows)
    early = [row for row in rows if row["command_index"] < COMMANDS_PER_EPISODE // 2]
    late = [row for row in rows if row["command_index"] >= COMMANDS_PER_EPISODE // 2]
    report = {
        "schema_version": 1,
        "experiment_id": "antmaze_chained_sac_executor_e549_adaptive_5m",
        "method_reference": str(PROTOCOL.resolve()),
        "method_reference_sha256": sha256(PROTOCOL.resolve()),
        "method_revision": "antmaze_e549_5m_adaptive_chained_executor_screen_r1",
        "claim_level": "executor_only_adaptive_chained_command_development_screen",
        "declared_deviations": [],
        "oracle_inputs": [],
        "adaptive_contract": (
            "each_next_command_is_sampled_from_the_actual_previous_endpoint_using_"
            "the_frozen_R47_distance_bank_and_Farama_uniform_direction"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_num_timesteps": int(model.num_timesteps),
        "command_distance_bank": str(bank),
        "command_distance_bank_sha256": sha256(bank),
        "reset_seeds": reset_seeds,
        "episodes": len(episodes),
        "commands_per_episode": COMMANDS_PER_EPISODE,
        "commands": len(rows),
        "fixed_command_horizon": 100,
        "physical_episode_horizon": 1000,
        "command_success_rate": float(np.mean([row["success"] for row in rows])),
        "command_inversion_rate": float(np.mean([row["inverted"] for row in rows])),
        "episodes_with_inversion_rate": float(
            np.mean([row["any_inversion"] for row in episodes])
        ),
        "early_commands_1_to_5_success_rate": float(
            np.mean([row["success"] for row in early])
        ),
        "late_commands_6_to_10_success_rate": float(
            np.mean([row["success"] for row in late])
        ),
        "mean_displacement_m": float(np.mean([row["displacement_m"] for row in rows])),
        "median_displacement_m": float(
            np.median([row["displacement_m"] for row in rows])
        ),
        "mean_endpoint_error_m": float(
            np.mean([row["endpoint_error_m"] for row in rows])
        ),
        "success_by_command_position": by_position,
        "distance_band_definition": "empirical_quintiles_of_this_frozen_screen",
        "distance_band_boundaries_m": distance_boundaries,
        "success_by_requested_distance": by_distance,
        "task_reward_used_by_evaluator": False,
        "executor_training_reward_used": True,
        "hidden_goals_used": False,
        "tf_sn_planner_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "commands.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--command-distance-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=len(RESET_SEEDS))
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    bank = args.command_distance_bank.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not bank.is_file():
        raise FileNotFoundError(bank)
    if not 1 <= args.episodes <= len(RESET_SEEDS):
        raise ValueError(f"episodes must be in [1, {len(RESET_SEEDS)}]")
    evaluate(checkpoint, bank, output, args.device, RESET_SEEDS[: args.episodes])


if __name__ == "__main__":
    main()
