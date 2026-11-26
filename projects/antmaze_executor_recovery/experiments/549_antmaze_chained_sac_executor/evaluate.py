#!/usr/bin/env python3
"""Paired chained-command screen for Farama and E549 checkpoints."""

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
FARAMA_CHECKPOINT = PROJECT / "artifacts/farama_goalreach_sac/GoalReachAnt_model.zip"
FARAMA_SHA256 = "f47b729c0966c8389d8474eef135a6fdb806ace9b10a51b49161a130b7322a54"
DEFAULT_BANK = HERE / "artifacts/command_distance_bank_r47.npz"
DEFAULT_OUTPUT = PROJECT / "results/experiments/549_antmaze_chained_sac_executor/evaluation"
RESET_SEEDS = list(range(549_000, 549_032))
COMMANDS_PER_EPISODE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sequence(env: GoalReachAnt, start: np.ndarray, seed: int) -> np.ndarray:
    """Freeze one exact absolute command sequence, shared by all policies."""
    rng = np.random.default_rng(seed)
    anchor = np.asarray(start, dtype=np.float64).copy()
    goals = []
    for _ in range(COMMANDS_PER_EPISODE):
        for _attempt in range(10_000):
            theta = 2.0 * np.pi * rng.uniform()
            distance = float(rng.choice(env.command_distances))
            goal = anchor + distance * np.asarray([np.cos(theta), np.sin(theta)])
            if env._is_free(goal):
                goals.append(goal.copy())
                anchor = goal
                break
        else:
            raise RuntimeError("Could not build a frozen free-space command sequence")
    return np.asarray(goals, dtype=np.float64)


def frozen_panel(bank: Path, reset_seeds: list[int]) -> dict[int, np.ndarray]:
    panel = {}
    for seed in reset_seeds:
        env = GoalReachAnt(command_distance_bank=str(bank))
        try:
            env.reset(seed=seed)
            panel[seed] = make_sequence(env, env.command_start, seed + 1_000_000)
        finally:
            env.close()
    return panel


def evaluate_model(
    label: str,
    checkpoint: Path,
    bank: Path,
    panel: dict[int, np.ndarray],
    reset_seeds: list[int],
    device: str,
) -> tuple[dict, list[dict]]:
    model = SAC.load(str(checkpoint), device=device)
    if tuple(model.observation_space.shape) != (29,):
        raise ValueError(f"{label} has the wrong observation space")
    if tuple(model.action_space.shape) != (8,):
        raise ValueError(f"{label} has the wrong action space")
    rows = []
    episodes = []
    for episode, seed in enumerate(reset_seeds):
        env = GoalReachAnt(command_distance_bank=str(bank))
        try:
            _initial, _ = env.reset(seed=seed)
            observation = env.install_goal_sequence(panel[seed])
            primitive_steps = 0
            events = []
            while len(events) < COMMANDS_PER_EPISODE and primitive_steps < 1000:
                action, _ = model.predict(observation, deterministic=True)
                observation, _reward, terminated, truncated, info = env.step(action)
                primitive_steps += 1
                event = info["command_event"]
                if event is not None:
                    row = {
                        "model": label,
                        "episode": episode,
                        "reset_seed": seed,
                        "command_index": int(event["command_index"]),
                        "reason": str(event["reason"]),
                        "duration": int(event["duration"]),
                        "start": np.asarray(event["start"]).tolist(),
                        "goal": np.asarray(event["goal"]).tolist(),
                        "endpoint": np.asarray(event["endpoint"]).tolist(),
                        "requested_distance_m": float(
                            np.linalg.norm(np.asarray(event["goal"]) - np.asarray(event["start"]))
                        ),
                        "endpoint_error_m": float(event["endpoint_error_m"]),
                        "displacement_m": float(event["displacement_m"]),
                        "torso_height_m": float(env.torso_height),
                        "torso_uprightness": float(env.torso_uprightness),
                        # Inversion is a physical diagnostic, never a reward or policy rule.
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
                f"e549 evaluate {label} episode={episode + 1}/{len(reset_seeds)} "
                f"commands={len(events)} success={sum(r['success'] for r in events)} "
                f"flipped={int(any(r['inverted'] for r in events))}",
                flush=True,
            )
        finally:
            env.close()
    if len(rows) != len(reset_seeds) * COMMANDS_PER_EPISODE:
        raise ValueError(f"{label} did not complete the paired command screen")
    summary = {
        "label": label,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint.resolve()),
        "checkpoint_num_timesteps": int(model.num_timesteps),
        "episodes": len(episodes),
        "commands": len(rows),
        "command_success_rate": float(np.mean([row["success"] for row in rows])),
        "command_inversion_rate": float(np.mean([row["inverted"] for row in rows])),
        "episodes_with_inversion_rate": float(
            np.mean([row["any_inversion"] for row in episodes])
        ),
        "mean_displacement_m": float(np.mean([row["displacement_m"] for row in rows])),
        "median_displacement_m": float(np.median([row["displacement_m"] for row in rows])),
        "mean_endpoint_error_m": float(np.mean([row["endpoint_error_m"] for row in rows])),
    }
    return summary, rows


def parse_candidate(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("Candidate must be LABEL=/path/to/checkpoint.zip")
    label, raw_path = specification.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise ValueError(f"Invalid candidate: {specification}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-distance-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=len(RESET_SEEDS))
    args = parser.parse_args()
    bank = args.command_distance_bank.resolve()
    if not bank.is_file():
        raise FileNotFoundError(bank)
    if sha256(FARAMA_CHECKPOINT) != FARAMA_SHA256:
        raise ValueError("The official Farama comparator changed")

    if not 1 <= args.episodes <= len(RESET_SEEDS):
        raise ValueError(f"episodes must be in [1, {len(RESET_SEEDS)}]")
    reset_seeds = RESET_SEEDS[: args.episodes]
    models = [("farama_18m_open_plane", FARAMA_CHECKPOINT)]
    models.extend(parse_candidate(specification) for specification in args.candidate)
    panel = frozen_panel(bank, reset_seeds)
    summaries = []
    all_rows = []
    for label, checkpoint in models:
        summary, rows = evaluate_model(
            label, checkpoint, bank, panel, reset_seeds, args.device
        )
        summaries.append(summary)
        all_rows.extend(rows)

    baseline = summaries[0]
    comparisons = []
    for candidate in summaries[1:]:
        comparisons.append(
            {
                "candidate": candidate["label"],
                "baseline": baseline["label"],
                "command_success_rate_delta": (
                    candidate["command_success_rate"] - baseline["command_success_rate"]
                ),
                "command_inversion_rate_delta": (
                    candidate["command_inversion_rate"] - baseline["command_inversion_rate"]
                ),
                "matches_or_exceeds_farama_on_both_headline_metrics": bool(
                    candidate["command_success_rate"] >= baseline["command_success_rate"]
                    and candidate["command_inversion_rate"] <= baseline["command_inversion_rate"]
                ),
            }
        )
    report = {
        "schema_version": 1,
        "experiment_id": "antmaze_chained_sac_executor_e549",
        "claim_level": "executor_only_paired_chained_command_development_screen",
        "reset_seeds": reset_seeds,
        "commands_per_episode": COMMANDS_PER_EPISODE,
        "fixed_command_horizon": 100,
        "same_absolute_command_sequences_across_models": True,
        "summaries": summaries,
        "comparisons": comparisons,
        "task_reward_used_by_evaluator": False,
        "hidden_goals_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "commands.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

