#!/usr/bin/env python3
"""Plot E191's development curve against the matched E192 heuristics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    ROOT / "results/vcg-conditioned-e192-e191-matched-longstay-heuristics-251k-v1"
)
E191_COLOR = "#0072B2"
DYNAMIC_COLOR = "#D55E00"
GA_COLOR = "#009E73"


METRICS = (
    ("J", "Objective J", True),
    ("storage_clock_mae", "Storage-clock MAE", True),
    ("rehandles_per_100", "Rehandles / 100 deliveries", True),
    ("peak_stored", "Peak blocks stored", False),
)


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _style_axis(axis, *, lower_better: bool) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    if lower_better:
        axis.text(
            0.99,
            0.97,
            "lower is better",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )


def _learning_panel(axis, report: dict, metric: str, title: str, lower: bool) -> None:
    checkpoints = sorted(int(value) for value in report["e191_checkpoint_summaries"])
    values = [
        report["e191_checkpoint_summaries"][str(episode)]["overall"]["metrics"][metric]
        for episode in checkpoints
    ]
    dynamic = report["baseline_summaries"]["dynamic_pslap"]["overall"]["metrics"][metric]
    axis.plot(
        checkpoints,
        values,
        marker="o",
        markersize=6,
        linewidth=2.4,
        color=E191_COLOR,
        label="E191",
    )
    axis.axhline(
        dynamic,
        color=DYNAMIC_COLOR,
        linestyle="--",
        linewidth=2.0,
        label="Dynamic PSLAP",
    )
    for episode, value in zip(checkpoints, values):
        axis.annotate(
            f"{value:,.0f}" if metric == "J" else f"{value:.1f}",
            (episode, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=E191_COLOR,
        )
    axis.annotate(
        f"Dynamic {dynamic:,.0f}" if metric == "J" else f"Dynamic {dynamic:.1f}",
        (checkpoints[-1], dynamic),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=DYNAMIC_COLOR,
    )
    axis.set_title(title, fontsize=11, weight="semibold")
    axis.set_xlabel("E191 training episode")
    axis.set_xticks(checkpoints)
    _style_axis(axis, lower_better=lower)


def _workload_panel(axis, report: dict, metric: str, title: str, lower: bool) -> None:
    workloads = ("20%", "50%", "80%")
    x = np.arange(len(workloads), dtype=float)
    width = 0.25
    e191 = [
        report["e191_checkpoint_summaries"]["60"]["by_workload"][level]["metrics"][metric]
        for level in workloads
    ]
    dynamic = [
        report["baseline_summaries"]["dynamic_pslap"]["by_workload"][level]["metrics"][metric]
        for level in workloads
    ]
    ga_groups = report["baseline_summaries"]["ga"]["by_workload"]
    ga = [
        ga_groups[level]["metrics"].get(metric)
        if ga_groups[level]["all_instances_complete"]
        else np.nan
        for level in workloads
    ]

    axis.bar(x - width, dynamic, width, color=DYNAMIC_COLOR, label="Dynamic PSLAP")
    axis.bar(x, e191, width, color=E191_COLOR, label="E191 ep60")
    axis.bar(x + width, ga, width, color=GA_COLOR, label="GA")
    for index, level in enumerate(workloads):
        if np.isnan(ga[index]):
            completed = ga_groups[level]["strict_complete_instances"]
            required = ga_groups[level]["required_instances"]
            axis.scatter(
                [x[index] + width],
                [0],
                marker="x",
                s=48,
                linewidth=2,
                color=GA_COLOR,
                clip_on=False,
                zorder=4,
            )
            axis.annotate(
                f"GA {completed}/{required}",
                (x[index] + width, 0),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=GA_COLOR,
            )
    axis.set_title(title, fontsize=11, weight="semibold")
    axis.set_xlabel("Nominal expected occupancy workload")
    axis.set_xticks(x, workloads)
    _style_axis(axis, lower_better=lower)


def plot(report_path: Path, output: Path) -> Path:
    report = _read(report_path)
    if report.get("status") != "complete":
        raise RuntimeError("E192 report is incomplete")
    if "60" not in report.get("e191_checkpoint_summaries", {}):
        raise RuntimeError("E191 episode-60 validation is absent")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlepad": 9,
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    })
    figure, axes = plt.subplots(2, 4, figsize=(16, 9.2))
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.89,
        bottom=0.14,
        hspace=0.32,
        wspace=0.12,
    )
    for column, (metric, title, lower) in enumerate(METRICS):
        _learning_panel(axes[0, column], report, metric, title, lower)
        _workload_panel(axes[1, column], report, metric, title, lower)

    figure.suptitle(
        "E191 learning progress and matched heuristic comparison",
        y=0.985,
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.5,
        0.945,
        "Top: nine-day development-panel means.  Bottom: episode-60 workload means.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    handles = [
        plt.Line2D([0], [0], color=E191_COLOR, marker="o", linewidth=2.4, label="E191"),
        plt.Line2D([0], [0], color=DYNAMIC_COLOR, linestyle="--", linewidth=2.0, label="Dynamic PSLAP"),
        plt.Rectangle((0, 0), 1, 1, color=GA_COLOR, label="GA (complete workloads only)"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncols=3,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.01,
        0.014,
        "Development data only; all E191 and Dynamic rows completed 9/9. "
        "GA completed 7/9 instances, so 50% and 80% GA costs are intentionally omitted. "
        "Certified Dynamic is not included.",
        ha="left",
        fontsize=8.5,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_RESULTS / "e192-report.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "e191-learning-and-baselines.png",
    )
    args = parser.parse_args()
    print(plot(args.report.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
