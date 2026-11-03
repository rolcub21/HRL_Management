#!/usr/bin/env python3
"""Plot the completed E203 sealed 8x8 test-panel comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    ROOT / "results/vcg-conditioned-e203-e191-matched-8x8-test-panel-262k-v1"
)
WORKLOADS = ("20%", "50%", "80%")
MODEL_SEEDS = ("0", "1", "2")

VCG_COLOR = "#0072B2"
DYNAMIC_COLOR = "#D55E00"
GA_COLOR = "#009E73"
SEED_COLORS = ("#56B4E9", "#332288", "#CC79A7")


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"E203 report is not a JSON object: {path}")
    return value


def workload_values(report: dict, metric: str) -> dict:
    """Return workload means and visible seed-level VCG values."""

    dynamic = [
        float(report["baseline_by_workload"]["dynamic_pslap"][level]["metrics"][metric])
        for level in WORKLOADS
    ]
    ga = [
        float(report["baseline_by_workload"]["ga"][level]["metrics"][metric])
        for level in WORKLOADS
    ]
    vcg_seeds = [
        [
            float(report["e191_selected_by_workload"][seed][level]["metrics"][metric])
            for level in WORKLOADS
        ]
        for seed in MODEL_SEEDS
    ]
    vcg = [
        float(fmean(vcg_seeds[index][column] for index in range(len(MODEL_SEEDS))))
        for column in range(len(WORKLOADS))
    ]
    return {
        "dynamic_pslap": dynamic,
        "ga": ga,
        "vcg": vcg,
        "vcg_seeds": vcg_seeds,
    }


def raw_workload_values(results: Path, metric: str) -> dict:
    """Aggregate KPIs retained in episode rows but not the compact report."""

    def rows(pattern: str) -> list[dict]:
        values = [_read(path) for path in sorted(results.glob(pattern))]
        if not values or not all(value.get("strict_complete") for value in values):
            raise RuntimeError(f"E203 raw KPI rows are incomplete: {pattern}")
        return values

    def level(row: dict) -> str:
        return str(row["metrics"]["e191_workload"]["target_label"])

    dynamic_rows = rows("rows/dynamic_pslap/*.json")
    ga_rows = rows("rows/ga/*.json")
    vcg_rows = {
        seed: rows(f"rows/e191/model-seed-{seed}/*.json") for seed in MODEL_SEEDS
    }
    dynamic = []
    ga = []
    vcg_seeds = [[] for _ in MODEL_SEEDS]
    for workload in WORKLOADS:
        dynamic_group = [row for row in dynamic_rows if level(row) == workload]
        if len(dynamic_group) != 10:
            raise RuntimeError(f"E203 Dynamic raw KPI group differs: {workload}")
        dynamic.append(fmean(float(row["metrics"][metric]) for row in dynamic_group))

        ga_group = [row for row in ga_rows if level(row) == workload]
        by_instance: dict[int, list[float]] = {}
        for row in ga_group:
            by_instance.setdefault(int(row["instance_seed"]), []).append(
                float(row["metrics"][metric])
            )
        if len(by_instance) != 10 or any(len(values) != 4 for values in by_instance.values()):
            raise RuntimeError(f"E203 GA raw KPI group differs: {workload}")
        ga.append(fmean(fmean(values) for values in by_instance.values()))

        for seed_index, seed in enumerate(MODEL_SEEDS):
            group = [row for row in vcg_rows[seed] if level(row) == workload]
            if len(group) != 10:
                raise RuntimeError(f"E203 VCG raw KPI group differs: seed={seed} {workload}")
            vcg_seeds[seed_index].append(
                fmean(float(row["metrics"][metric]) for row in group)
            )
    vcg = [
        fmean(vcg_seeds[seed_index][column] for seed_index in range(len(MODEL_SEEDS)))
        for column in range(len(WORKLOADS))
    ]
    return {
        "dynamic_pslap": [float(value) for value in dynamic],
        "ga": [float(value) for value in ga],
        "vcg": [float(value) for value in vcg],
        "vcg_seeds": [[float(value) for value in seed] for seed in vcg_seeds],
    }


def _style(axis, *, direction: str | None) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    if direction is not None:
        axis.text(
            0.99,
            0.97,
            direction,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1},
            zorder=6,
        )


def _panel(
    axis,
    report: dict,
    metric: str,
    title: str,
    *,
    direction: str | None,
    percentage: bool = False,
    values: dict | None = None,
) -> None:
    values = workload_values(report, metric) if values is None else values
    x = np.arange(len(WORKLOADS), dtype=float)
    width = 0.24
    axis.bar(
        x - width,
        values["dynamic_pslap"],
        width,
        color=DYNAMIC_COLOR,
        label="Dynamic PSLAP",
    )
    axis.bar(x, values["ga"], width, color=GA_COLOR, label="Rolling GA")
    axis.bar(
        x + width,
        values["vcg"],
        width,
        color=VCG_COLOR,
        label="VCG (E191 pooled)",
    )

    seed_offsets = (-0.045, 0.0, 0.045)
    for seed_index, (seed, color, offset) in enumerate(
        zip(MODEL_SEEDS, SEED_COLORS, seed_offsets)
    ):
        axis.scatter(
            x + width + offset,
            values["vcg_seeds"][seed_index],
            s=27,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
            label=f"VCG seed {seed}",
        )

    if metric == "J":
        for column in range(len(WORKLOADS)):
            for position, key in (
                (x[column] - width, "dynamic_pslap"),
                (x[column], "ga"),
                (x[column] + width, "vcg"),
            ):
                value = values[key][column]
                axis.annotate(
                    f"{value:.0f}",
                    (position, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

    axis.set_title(title, fontsize=11, weight="semibold")
    axis.set_xlabel("Nominal workload")
    axis.set_xticks(x, WORKLOADS)
    if percentage:
        axis.set_ylim(0, 1.08)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    _style(axis, direction=direction)


def plot(report_path: Path, output: Path) -> Path:
    report = _read(report_path)
    if report.get("status") != "complete" or not report.get("paper_evidence"):
        raise RuntimeError("E203 plot requires the complete sealed test panel")
    if int(report.get("observed_rows", -1)) != int(report.get("expected_rows", -2)):
        raise RuntimeError("E203 report does not contain every declared row")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlepad": 9,
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    })
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 5.3))
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.78,
        bottom=0.24,
        wspace=0.19,
    )
    _panel(axes[0], report, "J", "Objective J", direction="lower is better")
    _panel(
        axes[1],
        report,
        "storage_within_20_rate",
        "Deliveries within storage window (±20)",
        direction="higher is better",
        percentage=True,
    )
    _panel(
        axes[2],
        report,
        "peak_stored",
        "Peak blocks stored (capacity 35)",
        direction=None,
    )

    vcg_j = float(report["e191_selected_pooled"]["metrics"]["J"])
    dynamic_j = float(report["baseline_summaries"]["dynamic_pslap"]["metrics"]["J"])
    ga_j = float(report["baseline_summaries"]["ga"]["metrics"]["J"])
    dynamic_change = 100.0 * (vcg_j - dynamic_j) / dynamic_j
    ga_change = 100.0 * (vcg_j - ga_j) / ga_j

    figure.suptitle(
        "8×8 sealed test panel: matched methods under workload",
        y=0.975,
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.91,
        "30 unseen instances (10/workload); VCG bars pool three frozen seeds and dots show seed means",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncols=6,
        frameon=False,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.06,
        f"All methods completed 100%. Overall J: VCG {vcg_j:.1f}, "
        f"Dynamic {dynamic_j:.1f}, GA {ga_j:.1f}. VCG change: "
        f"{dynamic_change:+.1f}% vs Dynamic, {ga_change:+.1f}% vs GA "
        "(lower is better).",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    figure.text(
        0.01,
        0.014,
        "GA averages four nuisance replicates within each instance. Lower J is better. "
        "Certified Dynamic is excluded as predeclared.",
        ha="left",
        fontsize=8,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return output


def plot_kpis(report_path: Path, output: Path) -> Path:
    """Render the broader quality, objective-driver, and operations dashboard."""

    report = _read(report_path)
    if report.get("status") != "complete" or not report.get("paper_evidence"):
        raise RuntimeError("E203 KPI plot requires the complete sealed test panel")
    if int(report.get("observed_rows", -1)) != int(report.get("expected_rows", -2)):
        raise RuntimeError("E203 report does not contain every declared row")

    panels = (
        ("J", "Objective J", "lower is better", False, False),
        ("storage_clock_mae", "Storage-clock MAE", "lower is better", False, False),
        (
            "storage_within_20_rate",
            "Deliveries within ±20",
            "higher is better",
            True,
            False,
        ),
        (
            "component_admission_wait_accrual",
            "Admission-wait accrual W",
            "lower is better",
            False,
            False,
        ),
        (
            "component_storage_earliness_at_delivery",
            "Earliness accrual E",
            "lower is better",
            False,
            False,
        ),
        (
            "component_storage_tardiness_accrual",
            "Tardiness accrual T",
            "lower is better",
            False,
            False,
        ),
        (
            "component_equipment_work",
            "Equipment work L",
            "lower is better",
            False,
            False,
        ),
        (
            "rehandles_per_100",
            "Rehandles / 100 deliveries",
            "lower is better",
            False,
            False,
        ),
        (
            "steps_per_delivery",
            "Primitive steps / delivery",
            "lower is better",
            False,
            False,
        ),
        (
            "mean_stored",
            "Mean blocks stored",
            None,
            False,
            True,
        ),
        ("peak_stored", "Peak blocks stored (capacity 35)", None, False, False),
        (
            "transporter_utilization",
            "Transporter utilization",
            None,
            True,
            True,
        ),
    )
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlepad": 7,
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    })
    figure, axes = plt.subplots(4, 3, figsize=(14.8, 14.2))
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.89,
        bottom=0.115,
        hspace=0.49,
        wspace=0.21,
    )
    for axis, (metric, title, direction, percentage, raw) in zip(axes.flat, panels):
        _panel(
            axis,
            report,
            metric,
            title,
            direction=direction,
            percentage=percentage,
            values=(
                raw_workload_values(report_path.parent, metric) if raw else None
            ),
        )
        axis.set_xlabel("")

    figure.suptitle(
        "8×8 sealed test panel: matched KPI dashboard",
        y=0.975,
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.935,
        "Outcome quality · objective drivers · operational effort · realized workload",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncols=6,
        frameon=False,
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.029,
        "All methods completed 100%. Component panels show raw accumulations; "
        "J = 2T + E + 0.5W + 0.1L + 10R.",
        ha="center",
        fontsize=8.7,
        color="#333333",
    )
    figure.text(
        0.01,
        0.009,
        "30 unseen instances (10/workload). VCG bars pool three frozen seeds; dots show seed means. "
        "GA averages four nuisance replicates within instance. Mean inventory includes empty-start ramp and terminal drain.",
        ha="left",
        fontsize=7.8,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_RESULTS / "e203-report.json"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RESULTS / "e203-test-panel.png"
    )
    parser.add_argument(
        "--kpi-output",
        type=Path,
        default=DEFAULT_RESULTS / "e203-kpi-dashboard.png",
    )
    args = parser.parse_args()
    print(plot(args.report.resolve(), args.output.resolve()))
    print(plot_kpis(args.report.resolve(), args.kpi_output.resolve()))


if __name__ == "__main__":
    main()
