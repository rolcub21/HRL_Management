#!/usr/bin/env python3
"""Render the confirmed sampled frontier and cumulative 89k panel curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_vcg_v11_nested_lambda_frontier_confirmation_89k as confirmation


FRONTIER_STEM = "nested-vcg-lambda-frontier-confirmation"
CURVES_STEM = "nested-vcg-lambda-confirmation-curves"
SUMMARY_NAME = "plot-summary.json"
COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#7A5195")
MARKERS = ("o", "D", "s", "^", "P")
METRICS = (
    ("dense_return", "Cumulative dense return ↑"),
    ("mean_absolute_error", "Cumulative MAE ↓"),
    ("physical_rehandles_per_100", "Cumulative physical rehandles / 100 ↓"),
)
REPORT_METRIC_NAMES = {
    "dense_return": "mean_dense_return",
    "mean_absolute_error": "mean_absolute_error",
    "physical_rehandles_per_100": "physical_rehandles_per_100",
}


class PlotError(RuntimeError):
    pass


def _cumulative(values: Sequence[float]) -> list[float]:
    total = 0.0
    result = []
    for index, value in enumerate(values, start=1):
        total += float(value)
        result.append(total / index)
    return result


def _curve_data(rows: Mapping) -> dict:
    result = {}
    for value in confirmation.LAMBDA_GRID:
        result[value] = {}
        for metric, _ in METRICS:
            seed_curves = []
            for seed in confirmation.MODEL_SEEDS:
                sequence = [
                    float(rows[value][seed][instance][metric])
                    for instance in confirmation.INSTANCE_SEEDS
                ]
                seed_curves.append(_cumulative(sequence))
            matrix = np.asarray(seed_curves, dtype=float)
            result[value][metric] = {
                "mean": matrix.mean(axis=0).tolist(),
                "minimum": matrix.min(axis=0).tolist(),
                "maximum": matrix.max(axis=0).tolist(),
            }
    return result


def _report_metric(report: Mapping, value: float, metric: str) -> float:
    key = confirmation._lambda_key(value)
    report_metric = REPORT_METRIC_NAMES.get(metric)
    if report_metric is None:
        raise PlotError(f"unknown curve metric: {metric}")
    try:
        return float(report["aggregate_metrics"][key][report_metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlotError(f"report is missing {key}/{report_metric}") from exc


def _check_endpoints(curves: Mapping, report: Mapping) -> dict:
    checked = {}
    for value in confirmation.LAMBDA_GRID:
        checked[confirmation._lambda_key(value)] = {}
        for metric, _ in METRICS:
            observed = float(curves[value][metric]["mean"][-1])
            expected = _report_metric(report, value, metric)
            if not np.isclose(observed, expected, rtol=0.0, atol=1e-10):
                raise PlotError(
                    f"curve/report endpoint mismatch for lambda={value}, {metric}: "
                    f"{observed} != {expected}"
                )
            checked[confirmation._lambda_key(value)][metric] = observed
    return checked


def _save(fig, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _frontier_figure(report: Mapping):
    fig, axis = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    xs = [
        _report_metric(report, value, "physical_rehandles_per_100")
        for value in confirmation.LAMBDA_GRID
    ]
    ys = [
        _report_metric(report, value, "mean_absolute_error")
        for value in confirmation.LAMBDA_GRID
    ]
    axis.plot(xs, ys, color="#6B7280", linewidth=1.4, zorder=1)
    for value, x, y, color, marker in zip(
        confirmation.LAMBDA_GRID, xs, ys, COLORS, MARKERS
    ):
        axis.scatter(
            x,
            y,
            s=58,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
            label=f"λ = {value:g}",
        )
    axis.set_xlabel("Physical rehandles / 100 required deliveries ↓")
    axis.set_ylabel("Mean absolute timing error ↓")
    axis.set_title("Frozen nested VCG operating points on unseen 89k")
    axis.grid(axis="both", alpha=0.25)
    axis.legend(frameon=False, ncol=1)
    return fig


def _curves_figure(curves: Mapping):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    x = np.arange(1, len(confirmation.INSTANCE_SEEDS) + 1)
    for axis, (metric, ylabel) in zip(axes, METRICS):
        for value, color, marker in zip(
            confirmation.LAMBDA_GRID, COLORS, MARKERS
        ):
            curve = curves[value][metric]
            mean = np.asarray(curve["mean"])
            low = np.asarray(curve["minimum"])
            high = np.asarray(curve["maximum"])
            axis.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0)
            axis.plot(
                x,
                mean,
                color=color,
                linewidth=1.8,
                marker=marker,
                markevery=(4, 5),
                markersize=4.0,
                label=f"λ = {value:g}",
            )
        axis.set_xlabel("Cumulative common EpisodeInstances k")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, len(confirmation.INSTANCE_SEEDS))
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(confirmation.LAMBDA_GRID),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("Frozen nested VCG as the common 89k panel accumulates")
    return fig


def render(project_root: Path, output_dir: Path) -> dict:
    output = output_dir.absolute()
    contract = confirmation.authenticate_contract(project_root, output)
    manifest = confirmation.authenticate_instance_manifest(project_root, output)
    report = confirmation._load_json(
        output / confirmation.REPORT_NAME, label="confirmation report"
    )
    confirmation._verify_hash(report, "report_sha256", label="confirmation report")
    if (
        report.get("status") not in ("passed", "failed")
        or report.get("whole_method_metrics_suppressed") is True
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise PlotError("confirmation report is not eligible for plotting")
    rows = confirmation._read_all_rows(output, contract, manifest)
    curves = _curve_data(rows)
    endpoints = _check_endpoints(curves, report)
    files = []
    files.extend(_save(_frontier_figure(report), output, FRONTIER_STEM))
    files.extend(_save(_curves_figure(curves), output, CURVES_STEM))
    summary = {
        "schema_version": 1,
        "protocol": confirmation.PROTOCOL,
        "status": "complete",
        "figures": files,
        "curve_endpoint_matches_authenticated_report": True,
        "checked_endpoints": endpoints,
        "cumulative_center": "equal mean over three model-seed cumulative means",
        "band": "minimum-to-maximum across the three model-seed curves",
        "frontier_axes": ["physical_rehandles_per_100", "mean_absolute_error"],
        "report_sha256": report["report_sha256"],
    }
    confirmation._atomic_json(output / SUMMARY_NAME, summary)
    return summary


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
        else root / "results/vcg-v1-1-nested-lambda-frontier-confirmation-89k"
    )
    print(json.dumps(render(root, output), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
