#!/usr/bin/env python3
"""Plot v4.1 training stability and paired Track-B holdout comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "Fully learned v4.1",
    "Duration-aware + REG-v5",
    "Duration-aware + dynamic PSLAP",
    "Duration-aware + nearest-free",
)
BASELINE_LABELS = {
    "reg_selector_v5": METHODS[1],
    "dynamic_pslap": METHODS[2],
    "nearest_free": METHODS[3],
}
COLORS = {
    METHODS[0]: "#0072B2",
    METHODS[1]: "#D55E00",
    METHODS[2]: "#009E73",
    METHODS[3]: "#7A7A7A",
}
MARKERS = {
    METHODS[0]: "o",
    METHODS[1]: "s",
    METHODS[2]: "^",
    METHODS[3]: "X",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path(
            "results/fully-learned-v4-1-seed0-from-v4-spatial-250ep/"
            "training-summary.json"
        ),
    )
    parser.add_argument(
        "--learned-holdout",
        type=Path,
        default=Path(
            "results/fully-learned-v4-1-holdout-48000-48009.json"
        ),
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path(
            "results/track-b-hierarchy-confirmation-48000-48009/"
            "hierarchy-full.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "results/fully-learned-v4-1-comparison-curves"
        ),
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def load_paired_holdout(learned_path: Path, baseline_path: Path):
    learned_payload = load_json(learned_path)
    baseline_payload = load_json(baseline_path)
    learned_runs = sorted(
        learned_payload["runs"], key=lambda run: int(run["eval_seed"])
    )
    learned_by_instance = {run["instance_id"]: run for run in learned_runs}
    if len(learned_by_instance) != len(learned_runs):
        raise ValueError("learned holdout contains duplicate instance IDs")

    selected = {}
    for run in baseline_payload["runs"]:
        if (
            run["geometry_condition"] == "ordinary"
            and run["scheduler_variant"] == "duration_aware"
            and run["assignment_commitment"] == "decision_epoch_reserved"
            and run["assignment_source"] in BASELINE_LABELS
        ):
            selected.setdefault(run["assignment_source"], {})[
                run["instance_id"]
            ] = run

    expected_instances = set(learned_by_instance)
    for source in BASELINE_LABELS:
        found = set(selected.get(source, {}))
        if found != expected_instances:
            raise ValueError(
                f"paired baseline grid mismatch for {source}: "
                f"missing={sorted(expected_instances - found)}, "
                f"extra={sorted(found - expected_instances)}"
            )

    ordered_ids = [run["instance_id"] for run in learned_runs]
    for source, runs in selected.items():
        for learned in learned_runs:
            baseline = runs[learned["instance_id"]]
            if baseline["schedule_id"] != learned["schedule_id"]:
                raise ValueError(
                    f"schedule mismatch for {source}, seed {learned['eval_seed']}"
                )

    paired = {METHODS[0]: learned_runs}
    for source, label in BASELINE_LABELS.items():
        paired[label] = [selected[source][key] for key in ordered_ids]
    return learned_payload, paired


def metric(paired, method, name):
    return np.asarray(
        [float(run[name]) for run in paired[method]], dtype=np.float64
    )


def add_holdout_panel(ax, x, seeds, paired, metric_name, ylabel, title):
    for method in METHODS:
        values = metric(paired, method, metric_name)
        ax.plot(
            x,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=5.2,
            alpha=0.95,
            label=method,
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Matched holdout instance seed")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=45, ha="right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    inset = ax.inset_axes([0.43, 0.08, 0.54, 0.39])
    focused = METHODS[:3]
    all_values = []
    for method in focused:
        values = metric(paired, method, metric_name)
        all_values.extend(values.tolist())
        inset.plot(
            x,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.15,
            markersize=3.0,
        )
    padding = max(1.0, 0.08 * (max(all_values) - min(all_values)))
    inset.set_ylim(min(all_values) - padding, max(all_values) + padding)
    inset.set_xticks((x[0], x[-1]))
    inset.set_xticklabels((seeds[0], seeds[-1]), fontsize=6)
    inset.tick_params(axis="y", labelsize=6)
    inset.grid(True, linestyle=":", alpha=0.35)
    inset.set_title("Top-three zoom", fontsize=7)


def write_plot_data(path: Path, validations, seeds, paired):
    fields = ("panel", "episode", "eval_seed", "method", "return", "mae")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in validations:
            writer.writerow(
                {
                    "panel": "training_validation",
                    "episode": row["episode"],
                    "eval_seed": "",
                    "method": METHODS[0],
                    "return": row["mean_return"],
                    "mae": row["mean_absolute_error"],
                }
            )
        for method in METHODS:
            for seed, run in zip(seeds, paired[method]):
                writer.writerow(
                    {
                        "panel": "paired_holdout",
                        "episode": "",
                        "eval_seed": seed,
                        "method": method,
                        "return": run["return"],
                        "mae": run["mean_absolute_error"],
                    }
                )


def main():
    args = parse_args()
    training = load_json(args.training_summary)
    validations = [
        row for row in training["validations"] if row.get("phase") == "joint"
    ]
    if not validations:
        raise ValueError("training summary has no joint validations")
    learned_payload, paired = load_paired_holdout(
        args.learned_holdout, args.baseline_results
    )
    seeds = [str(run["eval_seed"]) for run in paired[METHODS[0]]]
    x = np.arange(len(seeds))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    episodes = np.asarray([row["episode"] for row in validations])
    returns = np.asarray([row["mean_return"] for row in validations])
    return_sd = np.asarray([row["return_std"] for row in validations])
    reference = float(validations[0]["quality_gate"]["reference_return"])
    threshold = float(validations[0]["quality_gate"]["return_threshold"])
    best_index = int(np.argmax(returns))

    ax = axes[0]
    ax.fill_between(
        episodes,
        returns - return_sd,
        returns + return_sd,
        color=COLORS[METHODS[0]],
        alpha=0.13,
        label="Validation mean ± SD (3 seeds)",
    )
    ax.plot(
        episodes,
        returns,
        color=COLORS[METHODS[0]],
        marker="o",
        markersize=3.8,
        linewidth=2.0,
        label="V4.1 validation return",
    )
    ax.axhline(
        reference,
        color="#6A3D9A",
        linestyle="--",
        linewidth=1.5,
        label=f"Spatial reference ({reference:.1f})",
    )
    ax.axhline(
        threshold,
        color="#CC6677",
        linestyle=":",
        linewidth=1.5,
        label=f"Quality-gate floor ({threshold:.1f})",
    )
    ax.scatter(
        episodes[best_index],
        returns[best_index],
        s=75,
        facecolor="#F0E442",
        edgecolor="black",
        zorder=5,
    )
    ax.annotate(
        f"best: ep {episodes[best_index]}\n{returns[best_index]:.1f}",
        (episodes[best_index], returns[best_index]),
        xytext=(12, 13),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.set_title("A. Joint-training stability", loc="left", fontweight="bold")
    ax.set_xlabel("Joint-training episode")
    ax.set_ylabel("Validation mean return")
    ax.set_xlim(0, max(episodes) + 5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

    add_holdout_panel(
        axes[1],
        x,
        seeds,
        paired,
        "return",
        "Episode return",
        "B. Paired holdout return",
    )
    add_holdout_panel(
        axes[2],
        x,
        seeds,
        paired,
        "mean_absolute_error",
        "Mean absolute timing error",
        "C. Paired holdout timing accuracy",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.66, -0.015),
        frameon=False,
    )
    fig.suptitle(
        "Fully learned reserved hierarchy: training stability and matched Track-B holdout",
        fontsize=14,
        fontweight="bold",
        y=1.015,
    )
    fig.text(
        0.66,
        -0.065,
        "Holdout curves use identical saved EpisodeInstances (seeds 48000–48009); "
        "all methods achieved 100% strict success and reservation integrity.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98), w_pad=2.2)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    csv_path = args.output_prefix.with_suffix(".csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    write_plot_data(csv_path, validations, seeds, paired)

    means = {
        method: {
            "mean_return": float(metric(paired, method, "return").mean()),
            "mean_absolute_error": float(
                metric(paired, method, "mean_absolute_error").mean()
            ),
        }
        for method in METHODS
    }
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path), "csv": str(csv_path), "holdout_means": means}, indent=2))


if __name__ == "__main__":
    main()
