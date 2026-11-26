"""Summarize the R12 two-by-two as main effects and an interaction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBSERVATIONS = ("sigma_kernel", "latent_radius")
PASSAGES = ("nearest_recorded", "shared_learned")


def load_arm(root, domain, observation, passage):
    path = root / domain / f"{observation}__{passage}" / "summary.json"
    value = json.loads(path.read_text())
    expected = f"{observation}__{passage}"
    if value.get("arm", expected) != expected:
        raise ValueError(f"factorial arm mismatch: {path}")
    return value, path


def task_rows(arm, observation, passage):
    rows = []
    for row in arm["rows"]:
        task = f"task_{int(row['task_id'])}"
        acquired = float(row["first_contact_episode"] is not None)
        mean_e2e = float(row["mean_e2e"])
        rows.append({
            "domain": "antmaze",
            "task": task,
            "observation_memory": observation,
            "passage_prediction": passage,
            "acquired": acquired,
            "mean_e2e": mean_e2e,
        })
    return rows


def effect(table, metric):
    cell = {
        (o, p): float(np.mean([
            row[metric] for row in table
            if row["observation_memory"] == o and row["passage_prediction"] == p
        ]))
        for o in OBSERVATIONS for p in PASSAGES
    }
    memory = 0.5 * (
        cell[("latent_radius", "nearest_recorded")]
        - cell[("sigma_kernel", "nearest_recorded")]
        + cell[("latent_radius", "shared_learned")]
        - cell[("sigma_kernel", "shared_learned")]
    )
    passage = 0.5 * (
        cell[("sigma_kernel", "shared_learned")]
        - cell[("sigma_kernel", "nearest_recorded")]
        + cell[("latent_radius", "shared_learned")]
        - cell[("latent_radius", "nearest_recorded")]
    )
    interaction = (
        cell[("latent_radius", "shared_learned")]
        - cell[("latent_radius", "nearest_recorded")]
        - cell[("sigma_kernel", "shared_learned")]
        + cell[("sigma_kernel", "nearest_recorded")]
    )
    return {
        "cells": {f"{o}__{p}": cell[(o, p)] for o in OBSERVATIONS for p in PASSAGES},
        "latent_radius_main_effect": memory,
        "shared_learned_passage_main_effect": passage,
        "interaction_difference_in_differences": interaction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rows, sources = [], {"antmaze": {}}
    for observation in OBSERVATIONS:
        for passage in PASSAGES:
            arm, path = load_arm(
                args.root, "antmaze", observation, passage
            )
            sources["antmaze"][f"{observation}__{passage}"] = str(path)
            all_rows.extend(task_rows(arm, observation, passage))
    report = {
        "schema_version": 1,
        "claim_level": "consumed_development_factorial_only",
        "sources": sources,
        "domains": {
            "antmaze": {
                "acquisition": effect(all_rows, "acquired"),
                "mean_e2e": effect(all_rows, "mean_e2e"),
            }
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "factorial_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output / "factorial_task_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    figure, axis = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    for passage, marker in zip(PASSAGES, ("o", "s")):
        y = [
            np.mean([
                row["acquired"] for row in all_rows
                if row["observation_memory"] == observation
                and row["passage_prediction"] == passage
            ])
            for observation in OBSERVATIONS
        ]
        axis.plot((0, 1), y, marker=marker, linewidth=2, label=passage)
    axis.set_xticks((0, 1), ("sigma kernel", "latent radius"))
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("task acquisition fraction")
    axis.set_title("AntMaze tasks 1 and 4")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(args.output / "factorial_acquisition.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
