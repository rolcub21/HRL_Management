"""Combine R11 supplied-goal blocks and compare with frozen 2.9M MEGA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import atomic_json, load_json, resolve_repository, sha256


PROTOCOL_PATH = Path(__file__).with_name("supplied_goal_protocol.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    contract = load_json(PROTOCOL_PATH)
    protocol_sha = sha256(PROTOCOL_PATH)
    frames = []
    for seed in contract["panel"]["environment_seeds"]:
        directory = args.root / f"seed{seed}"
        summary = load_json(directory / "summary.json")
        if summary["protocol_sha256"] != protocol_sha:
            raise ValueError(f"R11 supplied-goal protocol mismatch for seed {seed}")
        frame = pd.read_csv(directory / "episodes.csv")
        if len(frame) != 26 * 20:
            raise ValueError(f"incomplete R11 supplied-goal seed {seed}")
        frames.append(frame)
    r11 = pd.concat(frames, ignore_index=True)
    if len(r11) != int(contract["panel"]["rows"]):
        raise ValueError("R11 supplied-goal matrix is incomplete")

    mega_contract = contract["mega_comparator"]
    mega_path = resolve_repository(mega_contract["episodes"])
    if sha256(mega_path) != mega_contract["episodes_sha256"]:
        raise ValueError("frozen MEGA episodes changed")
    mega = pd.read_csv(mega_path)
    keys = ["seed", "goal_name", "episode"]
    paired = r11.merge(
        mega[keys + ["success", "steps"]],
        on=keys,
        suffixes=("_r11", "_mega"),
        validate="one_to_one",
    )
    if len(paired) != int(contract["panel"]["rows"]):
        raise ValueError("R11 and MEGA scenarios do not pair")

    report = {
        "schema_version": 1,
        "experiment_id": contract["experiment_id"],
        "claim_level": contract["claim_level"],
        "episodes": int(len(paired)),
        "r11": {
            "success_rate": float(paired.success_r11.mean()),
            "mean_capped_steps": float(paired.steps_r11.mean()),
            "reward_free_transitions": int(
                contract["reward_free_budget"]["r11_total_transitions"]
            ),
        },
        "mega": {
            "training_seed": int(mega_contract["training_seed"]),
            "success_rate": float(paired.success_mega.mean()),
            "mean_capped_steps": float(paired.steps_mega.mean()),
            "reward_free_transitions": int(
                contract["reward_free_budget"]["mega_total_transitions"]
            ),
        },
        "r11_minus_mega_success_rate": float(
            paired.success_r11.mean() - paired.success_mega.mean()
        ),
        "paired_outcomes": {
            "both_succeed": int(
                ((paired.success_r11 == 1) & (paired.success_mega == 1)).sum()
            ),
            "r11_only": int(
                ((paired.success_r11 == 1) & (paired.success_mega == 0)).sum()
            ),
            "mega_only": int(
                ((paired.success_r11 == 0) & (paired.success_mega == 1)).sum()
            ),
            "neither": int(
                ((paired.success_r11 == 0) & (paired.success_mega == 0)).sum()
            ),
        },
        "equal_budget": False,
        "interpretation": "descriptive_supplied_goal_comparison_because_R11_uses_4p1m_and_MEGA_uses_2p9m_reward_free_transitions",
    }
    output = args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_episodes.csv", index=False)
    atomic_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
