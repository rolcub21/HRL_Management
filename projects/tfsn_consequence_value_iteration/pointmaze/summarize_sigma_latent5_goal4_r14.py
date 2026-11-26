"""Summarize the bounded R14 fixed-sigma versus latent-radius comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import atomic_json, load_json, sha256


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "sigma_latent5_goal4_protocol_r14.json"
ARMS = (
    "sigma_kernel__nearest_recorded",
    "latent_radius__nearest_recorded",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(PROTOCOL)
    rows, sources, arm_metrics = [], {}, {}
    for arm in ARMS:
        path = args.root / arm / "summary.json"
        summary = load_json(path)
        if summary.get("arm") != arm:
            raise ValueError(f"R14 arm mismatch: {path}")
        if summary.get("protocol_sha256") != sha256(PROTOCOL):
            raise ValueError(f"stale R14 arm summary: {path}")
        sources[arm] = str(path)
        local = []
        for task in summary["rows"]:
            metric = task["metrics"]
            closest = min(
                float(episode["closest_evaluator_distance_m"])
                for episode in task["episodes"]
            )
            row = {
                "arm": arm,
                "goal_name": task["goal_name"],
                "acquired": int(metric["first_contact_episode"] is not None),
                "first_contact_episode_ordinal": (
                    metric.get("first_contact_episode_ordinal")
                ),
                "mean_e2e": float(metric["mean_e2e"]),
                "conditional_return": metric["conditional_return"],
                "capped_contact_commands": float(metric["contact_latency"]) / 100.0,
                "minimum_evaluator_distance_m": closest,
            }
            rows.append(row)
            local.append(row)
        returns = [
            float(row["conditional_return"])
            for row in local if row["conditional_return"] is not None
        ]
        arm_metrics[arm] = {
            "acquired_goals": int(sum(row["acquired"] for row in local)),
            "acquisition_fraction": float(np.mean([
                row["acquired"] for row in local
            ])),
            "mean_e2e": float(np.mean([row["mean_e2e"] for row in local])),
            "conditional_return_across_defined_tasks": (
                float(np.mean(returns)) if returns else None
            ),
            "mean_capped_contact_commands": float(np.mean([
                row["capped_contact_commands"] for row in local
            ])),
            "mean_minimum_evaluator_distance_m": float(np.mean([
                row["minimum_evaluator_distance_m"] for row in local
            ])),
        }
    fixed = arm_metrics[ARMS[0]]
    latent = arm_metrics[ARMS[1]]
    report = {
        "schema_version": 1,
        "method_revision": protocol["method_revision"],
        "claim_level": protocol["claim_level"],
        "protocol_sha256": sha256(PROTOCOL),
        "sources": sources,
        "arm_metrics": arm_metrics,
        "latent_minus_sigma": {
            "acquisition_fraction": (
                latent["acquisition_fraction"] - fixed["acquisition_fraction"]
            ),
            "mean_e2e": latent["mean_e2e"] - fixed["mean_e2e"],
            "mean_capped_contact_commands": (
                latent["mean_capped_contact_commands"]
                - fixed["mean_capped_contact_commands"]
            ),
            "mean_minimum_evaluator_distance_m": (
                latent["mean_minimum_evaluator_distance_m"]
                - fixed["mean_minimum_evaluator_distance_m"]
            ),
        },
        "task_rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "comparison.json", report)
    with (args.output / "task_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ("fixed sigma", "latent rho")
    axes[0].bar(
        np.arange(2) - 0.17,
        [arm_metrics[arm]["acquisition_fraction"] for arm in ARMS],
        width=0.34,
        label="acquisition",
    )
    axes[0].bar(
        np.arange(2) + 0.17,
        [arm_metrics[arm]["mean_e2e"] for arm in ARMS],
        width=0.34,
        label="mean e2e",
    )
    axes[0].set_xticks(np.arange(2), labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Five-command outcomes")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    goals = protocol["goal_names"]
    x = np.arange(len(goals))
    for offset, arm, label in zip((-0.18, 0.18), ARMS, labels):
        lookup = {
            row["goal_name"]: row["minimum_evaluator_distance_m"]
            for row in rows if row["arm"] == arm
        }
        axes[1].bar(
            x + offset,
            [lookup[goal] for goal in goals],
            width=0.36,
            label=label,
        )
    axes[1].set_xticks(x, [goal.rsplit("_", 2)[-2] + "_" + goal.rsplit("_", 1)[-1] for goal in goals], rotation=25)
    axes[1].set_ylabel("minimum evaluator distance (m)")
    axes[1].set_title("Closest approach over eight episodes")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(args.output / "comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
