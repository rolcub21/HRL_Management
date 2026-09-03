#!/usr/bin/env python3
"""Render the predeclared final90 cumulative and trade-off figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_vcg_conditioned_final_comparison_90k as final90


COLORS = {
    "lambda0": "#0072B2",
    "lambda02": "#009E73",
    final90.V23_METHOD: "#56B4E9",
    final90.DYNAMIC_METHOD: "#CC79A7",
    final90.GA_METHOD: "#D55E00",
    final90.KIM_METHOD: "#6F4E7C",
}
MARKERS = {
    "lambda0": "o",
    "lambda02": "D",
    final90.V23_METHOD: "X",
    final90.DYNAMIC_METHOD: "^",
    final90.GA_METHOD: "v",
    final90.KIM_METHOD: "P",
}
LABELS = {
    "lambda0": "VCG (λ = 0)",
    "lambda02": "VCG + handling (λ = 0.2)",
    final90.V23_METHOD: "Historical VCG 2.3",
    final90.DYNAMIC_METHOD: "Dynamic PSLAP",
    final90.GA_METHOD: "Capacity-aware GA",
    final90.KIM_METHOD: "Kim2020 adaptation",
}
CURVE_METRICS = (
    ("dense_return", "Cumulative dense return ↑"),
    ("mean_absolute_error", "Cumulative MAE ↓"),
    (
        "physical_rehandles_per_100_required_deliveries",
        "Cumulative physical rehandles / 100 ↓",
    ),
)


class Plot90Error(RuntimeError):
    pass


def _metric(row: Mapping, name: str) -> float:
    key = "dense_objective_return" if (
        name == "dense_return" and "dense_return" not in row
    ) else name
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Plot90Error(f"missing metric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise Plot90Error(f"nonfinite metric {name}")
    return result


def _prefix_curve(rows, *, expected_per_instance: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["instance_seed"])].append(row)
    points = []
    failed_at = None
    for position, seed in enumerate(final90.INSTANCE_SEEDS, start=1):
        group = grouped.get(seed, [])
        if (
            len(group) != expected_per_instance
            or not all(row.get("strict_safe_complete") is True for row in group)
        ):
            failed_at = position
            break
        points.append(
            {
                metric: fmean(_metric(row, metric) for row in group)
                for metric, _label in CURVE_METRICS
            }
        )
    curves = {}
    for metric, _label in CURVE_METRICS:
        values = [point[metric] for point in points]
        mean = []
        lower = []
        upper = []
        for count in range(1, len(values) + 1):
            prefix = values[:count]
            center = fmean(prefix)
            half = (
                0.0
                if count == 1
                else 1.96 * stdev(prefix) / math.sqrt(count)
            )
            mean.append(center)
            lower.append(center - half)
            upper.append(center + half)
        curves[metric] = {"mean": mean, "lower": lower, "upper": upper}
    return {"curves": curves, "failed_at": failed_at, "safe_prefix": len(points)}


def _load_curves(project_root: Path, output_dir: Path):
    manifest = final90.authenticate_manifest(project_root, output_dir)
    conditioned = final90._load_conditioned_rows(output_dir, manifest)
    methods = {
        "lambda0": _prefix_curve(
            [row for row in conditioned if row["preference_lambda"] == 0.0],
            expected_per_instance=len(final90.MODEL_SEEDS),
        ),
        "lambda02": _prefix_curve(
            [row for row in conditioned if row["preference_lambda"] == 0.2],
            expected_per_instance=len(final90.MODEL_SEEDS),
        ),
        final90.V23_METHOD: _prefix_curve(
            final90._load_baseline_rows(output_dir, final90.V23_METHOD),
            expected_per_instance=(
                len(final90.final86.V23_MODEL_SEEDS)
                * len(final90.final86.RNG_INDICES)
            ),
        ),
        final90.DYNAMIC_METHOD: _prefix_curve(
            final90._load_baseline_rows(output_dir, final90.DYNAMIC_METHOD),
            expected_per_instance=1,
        ),
        final90.GA_METHOD: _prefix_curve(
            final90._load_baseline_rows(output_dir, final90.GA_METHOD),
            expected_per_instance=len(final90.final86.RNG_INDICES),
        ),
        final90.KIM_METHOD: _prefix_curve(
            final90._load_kim_rows(project_root, output_dir),
            expected_per_instance=(
                len(final90.kim_final.MODEL_SEEDS)
                * len(final90.kim_final.ROLLOUTS)
            ),
        ),
    }
    return methods


def render_cumulative(project_root: Path, output_dir: Path) -> list[str]:
    methods = _load_curves(project_root, output_dir)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    for axis, (metric, ylabel) in zip(axes, CURVE_METRICS):
        for key, payload in methods.items():
            curve = payload["curves"][metric]
            count = len(curve["mean"])
            if not count:
                continue
            x = np.arange(1, count + 1)
            axis.plot(
                x,
                curve["mean"],
                color=COLORS[key],
                marker=MARKERS[key],
                markevery=max(1, count // 6),
                linewidth=2.0,
                markersize=5,
                label=LABELS[key],
            )
            axis.fill_between(
                x,
                curve["lower"],
                curve["upper"],
                color=COLORS[key],
                alpha=0.11,
                linewidth=0,
            )
            if payload["failed_at"] is not None:
                axis.scatter(
                    [payload["failed_at"]],
                    [curve["mean"][-1]],
                    marker="x",
                    s=65,
                    linewidths=2,
                    color=COLORS[key],
                    zorder=5,
                )
        axis.set_xlabel("Cumulative common EpisodeInstances $k$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, len(final90.INSTANCE_SEEDS))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle("Frozen-policy comparison on the prospective common 90k panel")
    figure.tight_layout(rect=(0, 0.11, 1, 0.94))
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"vcg-conditioned-final90-cumulative.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def render_frontier(output_dir: Path, report: Mapping) -> list[str]:
    eligible = [
        record for record in report["methods"] if record["whole_method_eligible"]
    ]
    conditioned = sorted(
        (
            record
            for record in eligible
            if record["method"] == final90.CONDITIONED_METHOD
        ),
        key=lambda record: float(record["preference_lambda"]),
    )
    figure, axis = plt.subplots(figsize=(7.3, 5.6))
    if conditioned:
        x = [
            record["metrics"]["physical_rehandles_per_100_required_deliveries"]
            for record in conditioned
        ]
        y = [record["metrics"]["mean_absolute_error"] for record in conditioned]
        axis.plot(x, y, color="#0072B2", linewidth=2.2, alpha=0.8)
        axis.scatter(x, y, color="#0072B2", s=42, zorder=3, label="Conditioned VCG")
        for record, x_value, y_value in zip(conditioned, x, y):
            axis.annotate(
                f"{record['preference_lambda']:g}",
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    for record in eligible:
        if record["method"] == final90.CONDITIONED_METHOD:
            continue
        key = record["method"]
        axis.scatter(
            [record["metrics"]["physical_rehandles_per_100_required_deliveries"]],
            [record["metrics"]["mean_absolute_error"]],
            color=COLORS[key],
            marker=MARKERS[key],
            s=75,
            label=LABELS[key],
            zorder=4,
        )
    axis.set_xlabel("Physical rehandles / 100 required deliveries ↓")
    axis.set_ylabel("Mean absolute delivery error ↓")
    axis.set_title("Timing–handling operating points on the common 90k panel")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"vcg-conditioned-final90-tradeoff.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def render(project_root: Path, output_dir: Path) -> dict:
    report = final90.analyze(project_root, output_dir)
    return {
        "status": "complete",
        "cumulative": render_cumulative(project_root, output_dir),
        "tradeoff": render_frontier(output_dir, report),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path, default=final90.DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = render(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
