"""Summarize the isolated PointMaze 2x2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBSERVATIONS = ("sigma_kernel", "latent_radius")
PASSAGES = ("nearest_recorded", "shared_learned")


def effect(rows, metric):
    cell = {
        (o, p): float(np.mean([
            row[metric] for row in rows
            if row["observation_memory"] == o
            and row["passage_prediction"] == p
        ]))
        for o in OBSERVATIONS for p in PASSAGES
    }
    return {
        "cells": {f"{o}__{p}": cell[(o, p)] for o in OBSERVATIONS for p in PASSAGES},
        "latent_radius_main_effect": 0.5 * (
            cell[("latent_radius", "nearest_recorded")]
            - cell[("sigma_kernel", "nearest_recorded")]
            + cell[("latent_radius", "shared_learned")]
            - cell[("sigma_kernel", "shared_learned")]
        ),
        "shared_learned_passage_main_effect": 0.5 * (
            cell[("sigma_kernel", "shared_learned")]
            - cell[("sigma_kernel", "nearest_recorded")]
            + cell[("latent_radius", "shared_learned")]
            - cell[("latent_radius", "nearest_recorded")]
        ),
        "interaction_difference_in_differences": (
            cell[("latent_radius", "shared_learned")]
            - cell[("latent_radius", "nearest_recorded")]
            - cell[("sigma_kernel", "shared_learned")]
            + cell[("sigma_kernel", "nearest_recorded")]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, sources = [], {}
    for observation in OBSERVATIONS:
        for passage in PASSAGES:
            arm = f"{observation}__{passage}"
            path = args.root / arm / "summary.json"
            value = json.loads(path.read_text())
            if value.get("arm") != arm:
                raise ValueError(f"factorial arm mismatch: {path}")
            sources[arm] = str(path)
            for task in value["rows"]:
                rows.append({
                    "goal_name": task["goal_name"],
                    "observation_memory": observation,
                    "passage_prediction": passage,
                    "acquired": float(
                        task["metrics"]["first_contact_episode"] is not None
                    ),
                    "mean_e2e": float(task["metrics"]["mean_e2e"]),
                })
    report = {
        "schema_version": 1,
        "claim_level": "consumed_development_factorial_only",
        "sources": sources,
        "acquisition": effect(rows, "acquired"),
        "mean_e2e": effect(rows, "mean_e2e"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "factorial_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output / "factorial_task_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    for passage, marker in zip(PASSAGES, ("o", "s")):
        y = [
            np.mean([
                row["acquired"] for row in rows
                if row["observation_memory"] == observation
                and row["passage_prediction"] == passage
            ])
            for observation in OBSERVATIONS
        ]
        axis.plot((0, 1), y, marker=marker, linewidth=2, label=passage)
    axis.set_xticks((0, 1), ("sigma kernel", "latent radius"))
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("goal acquisition fraction")
    axis.set_title("PointMaze exposed corner goals")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(args.output / "factorial_acquisition.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
