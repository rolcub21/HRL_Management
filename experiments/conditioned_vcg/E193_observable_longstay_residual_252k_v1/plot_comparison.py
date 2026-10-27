#!/usr/bin/env python3
"""Plot E191 and E193 development learning against plain Dynamic PSLAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
E191_RESULTS = ROOT / "results/vcg-conditioned-e191-fresh-long-stay-crowding-250k-v1"
E193_RESULTS = ROOT / "results/vcg-conditioned-e193-observable-longstay-residual-252k-v1"
E192_RESULTS = ROOT / "results/vcg-conditioned-e192-e191-matched-longstay-heuristics-251k-v1"

E191_SUMMARY = (
    E191_RESULTS
    / "training/operational/empty_8x8_n36_longstay_mixedrho/seed-0/training-summary.json"
)
E193_SUMMARY = (
    E193_RESULTS
    / "training/operational/empty_8x8_n36_longstay_mixedrho/seed-0/training-summary.json"
)
DYNAMIC_REPORT = E192_RESULTS / "e192-report.json"
DEFAULT_OUTPUT = E193_RESULTS / "e191-e193-dynamic-comparison.png"

E191_COLOR = "#0072B2"
E193_COLOR = "#CC79A7"
DYNAMIC_COLOR = "#D55E00"
GA_COLOR = "#009E73"

LEARNING_METRICS = (
    ("mean_e187_J", "J", "Objective J", True),
    ("mean_storage_clock_mae", "storage_clock_mae", "Storage-clock MAE", True),
    ("mean_admission_wait", "mean_admission_wait", "Admission wait", True),
    (
        "relocations_per_100_deliveries",
        "rehandles_per_100",
        "Rehandles / 100 deliveries",
        True,
    ),
    ("mean_peak_stored", "peak_stored", "Peak blocks stored", False),
)

WORKLOAD_METRICS = (
    ("J", "J", "Objective J", True),
    ("storage_clock_mae", "storage_clock_mae", "Storage-clock MAE", True),
    ("mean_admission_wait", "mean_admission_wait", "Admission wait", True),
    ("peak_stored", "peak_stored", "Peak blocks stored", False),
    ("mean_stored", "mean_stored", "Mean blocks stored", False),
)


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validation_records(summary: dict) -> list[dict]:
    return sorted(
        summary["validation_history"], key=lambda item: item["checkpoint_episode"]
    )


def _format(metric: str, value: float) -> str:
    if metric == "J":
        return f"{value:,.0f}"
    return f"{value:.1f}"


def _style(axis, *, lower_better: bool) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_ylim(bottom=0)
    if lower_better:
        axis.text(
            0.99,
            0.97,
            "lower is better",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#555555",
        )


def _learning_panel(
    axis,
    e191_records: list[dict],
    e193_records: list[dict],
    dynamic: dict,
    summary_key: str,
    baseline_key: str,
    title: str,
    lower_better: bool,
) -> None:
    all_episodes = sorted({
        record["checkpoint_episode"]
        for record in (*e191_records, *e193_records)
    })
    for records, color, label in (
        (e191_records, E191_COLOR, "E191"),
        (e193_records, E193_COLOR, "E193"),
    ):
        episodes = [record["checkpoint_episode"] for record in records]
        values = [float(record["summary"][summary_key]) for record in records]
        axis.plot(
            episodes,
            values,
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=color,
            label=label,
        )
        label_offset = 6 if label == "E191" else -10
        label_vertical_alignment = "bottom" if label == "E191" else "top"
        for episode, value in zip(episodes, values):
            axis.annotate(
                _format("J" if baseline_key == "J" else baseline_key, value),
                (episode, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va=label_vertical_alignment,
                fontsize=7.2,
                color=color,
            )

    dynamic_value = float(dynamic[baseline_key])
    axis.axhline(
        dynamic_value,
        color=DYNAMIC_COLOR,
        linestyle="--",
        linewidth=1.9,
        label="Dynamic PSLAP",
    )
    axis.annotate(
        f"Dynamic {_format('J' if baseline_key == 'J' else baseline_key, dynamic_value)}",
        (all_episodes[-1], dynamic_value),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        fontsize=7.2,
        color=DYNAMIC_COLOR,
    )
    axis.set_title(title, fontsize=10.5, weight="semibold")
    axis.set_xlabel("Training episode")
    axis.set_xticks(all_episodes)
    axis.set_xlim(all_episodes[0] - 4, all_episodes[-1] + 4)
    _style(axis, lower_better=lower_better)


def _workload_panel(
    axis,
    e191_record: dict,
    e193_record: dict,
    dynamic_groups: dict,
    ga_groups: dict,
    summary_key: str,
    baseline_key: str,
    title: str,
    lower_better: bool,
) -> None:
    workloads = ("20%", "50%", "80%")
    x = np.arange(len(workloads), dtype=float)
    width = 0.20
    e191 = [
        float(e191_record["summary"]["e191_by_workload"][level]["means"][summary_key])
        for level in workloads
    ]
    e193 = [
        float(e193_record["summary"]["e191_by_workload"][level]["means"][summary_key])
        for level in workloads
    ]
    dynamic = [
        float(dynamic_groups[level]["metrics"][baseline_key]) for level in workloads
    ]
    ga = [
        float(ga_groups[level]["metrics"][baseline_key])
        if ga_groups[level]["all_instances_complete"]
        else np.nan
        for level in workloads
    ]

    bars = (
        axis.bar(x - 1.5 * width, dynamic, width, color=DYNAMIC_COLOR),
        axis.bar(x - 0.5 * width, e191, width, color=E191_COLOR),
        axis.bar(x + 0.5 * width, e193, width, color=E193_COLOR),
        axis.bar(x + 1.5 * width, ga, width, color=GA_COLOR),
    )
    for container, values in zip(bars, (dynamic, e191, e193, ga)):
        labels = [
            "" if np.isnan(value) else _format(
                "J" if baseline_key == "J" else baseline_key, value
            )
            for value in values
        ]
        axis.bar_label(container, labels=labels, padding=2, fontsize=6.8)
    for index, level in enumerate(workloads):
        if np.isnan(ga[index]):
            completed = ga_groups[level]["strict_complete_instances"]
            required = ga_groups[level]["required_instances"]
            location = x[index] + 1.5 * width
            axis.scatter(
                [location],
                [0],
                marker="x",
                s=42,
                linewidth=2,
                color=GA_COLOR,
                clip_on=False,
                zorder=4,
            )
            axis.annotate(
                f"GA {completed}/{required}",
                (location, 0),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=GA_COLOR,
            )

    axis.set_title(title, fontsize=10.5, weight="semibold")
    axis.set_xlabel("Nominal expected occupancy workload")
    axis.set_xticks(x, workloads)
    _style(axis, lower_better=lower_better)


def plot(
    e191_summary_path: Path,
    e193_summary_path: Path,
    dynamic_report_path: Path,
    output: Path,
) -> Path:
    e191_summary = _read(e191_summary_path)
    e193_summary = _read(e193_summary_path)
    report = _read(dynamic_report_path)
    e191_records = _validation_records(e191_summary)
    e193_records = _validation_records(e193_summary)
    e191_episodes = [record["checkpoint_episode"] for record in e191_records]
    e193_episodes = [record["checkpoint_episode"] for record in e193_records]
    if e191_episodes[:5] != [20, 40, 60, 80, 100]:
        raise RuntimeError("E191 validation history lacks checkpoints 20 through 100")
    if e193_episodes[:3] != [20, 40, 60]:
        raise RuntimeError("E193 validation history lacks checkpoints 20 through 60")
    if any(episode % 20 for episode in (*e191_episodes, *e193_episodes)):
        raise RuntimeError("validation history contains a non-20-episode checkpoint")
    e191_training = e191_summary["training"]["runs"][e191_episodes[-1] - 1]
    e193_training = e193_summary["training"]["runs"][e193_episodes[-1] - 1]

    dynamic_summary = report["baseline_summaries"]["dynamic_pslap"]
    if not dynamic_summary["overall"]["all_instances_complete"]:
        raise RuntimeError("plain Dynamic baseline panel is incomplete")
    dynamic_overall = dynamic_summary["overall"]["metrics"]
    dynamic_groups = dynamic_summary["by_workload"]
    ga_groups = report["baseline_summaries"]["ga"]["by_workload"]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlepad": 8,
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    })
    figure, axes = plt.subplots(2, 5, figsize=(18, 9.4))
    figure.subplots_adjust(
        left=0.045,
        right=0.99,
        top=0.88,
        bottom=0.15,
        hspace=0.34,
        wspace=0.20,
    )

    for column, metric in enumerate(LEARNING_METRICS):
        _learning_panel(
            axes[0, column],
            e191_records,
            e193_records,
            dynamic_overall,
            *metric,
        )
    for column, metric in enumerate(WORKLOAD_METRICS):
        _workload_panel(
            axes[1, column],
            e191_records[-1],
            e193_records[-1],
            dynamic_groups,
            ga_groups,
            *metric,
        )

    figure.suptitle(
        "Long-stay workload learning: E191 versus E193 and matched heuristics",
        y=0.982,
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.5,
        0.944,
        "Top: greedy nine-day development-panel learning curves.  "
        f"Bottom: workload means at E191 episode {e191_episodes[-1]} and "
        f"E193 episode {e193_episodes[-1]}, with GA where complete.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    handles = [
        plt.Line2D([0], [0], color=E191_COLOR, marker="o", linewidth=2.2, label="E191"),
        plt.Line2D([0], [0], color=E193_COLOR, marker="o", linewidth=2.2, label="E193"),
        plt.Line2D(
            [0], [0], color=DYNAMIC_COLOR, linestyle="--", linewidth=1.9,
            label="Dynamic PSLAP",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=GA_COLOR,
            label="GA (bottom; complete workloads only)",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.054),
        ncols=4,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.01,
        0.014,
        "Development data only; all panels use the same nine cases and evaluation epsilon 0.  "
        f"Episode is not an equal update budget: E191 ep{e191_episodes[-1]} = "
        f"{e191_training['training_decision_count_after']:,} training decisions "
        f"(epsilon {e191_training['scheduled_epsilon_after']:.3f}); "
        f"E193 ep{e193_episodes[-1]} = "
        f"{e193_training['training_decision_count_after']:,} decisions "
        f"(epsilon {e193_training['scheduled_epsilon_after']:.3f}).  "
        "GA completed 7/9 cases; its incomplete 50% and 80% costs are marked and omitted.  "
        "Certified Dynamic is not included.",
        ha="left",
        fontsize=8.2,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e191-summary", type=Path, default=E191_SUMMARY)
    parser.add_argument("--e193-summary", type=Path, default=E193_SUMMARY)
    parser.add_argument("--dynamic-report", type=Path, default=DYNAMIC_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        plot(
            args.e191_summary.resolve(),
            args.e193_summary.resolve(),
            args.dynamic_report.resolve(),
            args.output.resolve(),
        )
    )


if __name__ == "__main__":
    main()
