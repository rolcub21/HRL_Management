#!/usr/bin/env python3
"""Plot cumulative performance for the two nested VCG confirmation arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_vcg_v11_nested_handling_confirmation_88k as confirmation


ARM_SPECS = (
    ("lambda0", "VCG  (λ = 0)", "#0072B2", "o"),
    ("lambda0025", "VCG + handling  (λ = 0.025)", "#009E73", "D"),
)
METRICS = (
    ("dense_return", "Cumulative dense return ↑"),
    ("mean_absolute_error", "Cumulative MAE ↓"),
    ("physical_rehandles_per_100", "Cumulative physical rehandles / 100 ↓"),
)


class ConfirmationPlotError(RuntimeError):
    pass


def _finite(value, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfirmationPlotError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfirmationPlotError(f"{label} must be finite")
    return result


def build_curve_data(report: Mapping) -> dict:
    confirmation._verify_hash(report, "report_sha256", label="confirmation report")
    if (
        report.get("protocol") != confirmation.PROTOCOL
        or report.get("row_count") != confirmation.EXPECTED_ROWS
        or report.get("strict_safe_row_count") != confirmation.EXPECTED_ROWS
    ):
        raise ConfirmationPlotError("confirmation report is incomplete")
    points = report.get("instance_points")
    if not isinstance(points, list) or len(points) != len(confirmation.INSTANCE_SEEDS):
        raise ConfirmationPlotError("confirmation report has no exact 30-instance grid")
    if [point.get("instance_seed") for point in points] != list(
        confirmation.INSTANCE_SEEDS
    ):
        raise ConfirmationPlotError("confirmation points are not in frozen panel order")

    seed_summaries = report.get("seed_summaries")
    # Per-seed cumulative paths are reconstructed from the authenticated ledgers
    # in render(); this function owns the equal-seed curve in the report itself.
    curves = {}
    for arm, _, _, _ in ARM_SPECS:
        curves[arm] = {}
        for metric, _ in METRICS:
            values = np.asarray(
                [_finite(point[arm][metric], label=f"{arm}/{metric}") for point in points],
                dtype=float,
            )
            curves[arm][metric] = {
                "mean": (np.cumsum(values) / np.arange(1, values.size + 1)).tolist()
            }
            endpoint = curves[arm][metric]["mean"][-1]
            expected = _finite(
                report["arm_summaries"][arm][metric],
                label=f"reported endpoint {arm}/{metric}",
            )
            if not math.isclose(endpoint, expected, rel_tol=0.0, abs_tol=1e-10):
                raise ConfirmationPlotError(f"endpoint mismatch: {arm}/{metric}")
    return curves


def _seed_curves(project_root: Path, output_dir: Path, report: Mapping) -> dict:
    contract = confirmation.authenticate_contract(project_root, output_dir)
    manifest = confirmation.authenticate_instance_manifest(project_root, output_dir)
    rows = confirmation._read_all_rows(project_root, output_dir, contract, manifest)
    result = {}
    for arm, _, _, _ in ARM_SPECS:
        result[arm] = {}
        for metric, _ in METRICS:
            paths = []
            for seed in confirmation.MODEL_SEEDS:
                values = np.asarray(
                    [
                        _finite(rows[arm][seed][instance][metric], label=f"row {metric}")
                        for instance in confirmation.INSTANCE_SEEDS
                    ],
                    dtype=float,
                )
                paths.append(np.cumsum(values) / np.arange(1, values.size + 1))
            matrix = np.vstack(paths)
            result[arm][metric] = {
                "minimum": matrix.min(axis=0).tolist(),
                "maximum": matrix.max(axis=0).tolist(),
            }
    return result


def render(project_root: Path, output_dir: Path) -> dict:
    report_path = output_dir / confirmation.REPORT_NAME
    report = confirmation._load_json(report_path, label="confirmation report")
    curves = build_curve_data(report)
    seed_curves = _seed_curves(project_root, output_dir, report)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharex=True)
    x = np.arange(1, len(confirmation.INSTANCE_SEEDS) + 1)
    for arm, label, color, marker in ARM_SPECS:
        for axis, (metric, ylabel) in zip(axes, METRICS):
            mean = np.asarray(curves[arm][metric]["mean"], dtype=float)
            low = np.asarray(seed_curves[arm][metric]["minimum"], dtype=float)
            high = np.asarray(seed_curves[arm][metric]["maximum"], dtype=float)
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            axis.plot(
                x,
                mean,
                color=color,
                linewidth=2.35,
                marker=marker,
                markersize=4.5,
                markevery=(0, 4, 9, 14, 19, 24, 29),
                label=label if axis is axes[0] else None,
                zorder=3,
            )
            axis.scatter([30], [mean[-1]], color=color, marker=marker, s=42, zorder=4)
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xlim(1, 30)
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_xlabel("Cumulative unseen EpisodeInstances  k")
    figure.suptitle(
        "Frozen nested-VCG policies as the unseen 88k panel accumulates",
        y=0.99,
        fontsize=13,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    png = output_dir / "confirmation-curves.png"
    pdf = output_dir / "confirmation-curves.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    result = confirmation._with_hash(
        {
            "schema_version": confirmation.SCHEMA_VERSION,
            "protocol": confirmation.PROTOCOL,
            "figure_role": "cumulative_frozen_policy_confirmation",
            "not_a_training_learning_curve": True,
            "panel_order": list(confirmation.INSTANCE_SEEDS),
            "methods": [arm for arm, _, _, _ in ARM_SPECS],
            "baseline_data_included": False,
            "band": "descriptive pointwise min-max across three fixed model seeds",
            "endpoint": {
                arm: {
                    metric: curves[arm][metric]["mean"][-1]
                    for metric, _ in METRICS
                }
                for arm, _, _, _ in ARM_SPECS
            },
            "confirmation_report_sha256": report["report_sha256"],
            "png": str(png),
            "pdf": str(pdf),
        },
        "figure_summary_sha256",
    )
    confirmation._atomic_json(output_dir / "confirmation-curve-summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    output = (
        args.output_dir.absolute()
        if args.output_dir is not None
        else root / "results/vcg-v1-1-nested-handling-confirmation-88k"
    )
    result = render(root, output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
