#!/usr/bin/env python3
"""Render paper-facing E11 distribution-shift figures and paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
import run_vcg_conditioned_final_comparison_90k as final90


OPERATING_STEM = "e11-operating-points-by-shift"
SHIFT_STEM = "e11-paired-shift-effects"
DATA_NAME = "e11-figure-data.json"
STATS_NAME = "e11-paired-shift-statistics.json"
TABLE_NAME = "e11-paired-shift-table.md"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 93_011

REGIME_ORDER = (
    "reference",
    "arrival_spread",
    "dwell_short",
    "dwell_long",
    "dwell_bimodal",
    "mirrored_entry",
    "combined_shift",
)
REGIME_LABELS = {
    "reference": "Reference",
    "arrival_spread": "Spread arrivals",
    "dwell_short": "Short dwell",
    "dwell_long": "Long dwell",
    "dwell_bimodal": "Bimodal dwell",
    "mirrored_entry": "Mirrored access",
    "combined_shift": "Combined shift",
}
COORDINATES = (
    (e11.QOP, 0.0, r"$Q_{op}$ ($\lambda=0$)", "#0072B2", "o"),
    (e11.CONDITIONED, 0.05, r"$\lambda=.05$", "#E69F00", "D"),
    (e11.CONDITIONED, 0.10, r"$\lambda=.10$", "#009E73", "s"),
    (e11.CONDITIONED, 0.20, r"$\lambda=.20$", "#CC79A7", "^"),
)
METRICS = (
    "mean_absolute_error",
    "physical_rehandles_per_100_required_deliveries",
    "dense_return",
    "within_target_window_rate",
    "steps",
)


class E11FigureError(RuntimeError):
    pass


def _coordinate_key(method: str, value: float) -> str:
    return e11._coordinate(method, value)


def _rows_for(
    rows: Sequence[Mapping],
    regime_id: str,
    method: str,
    value: float,
    model_seed: int | None = None,
) -> list[Mapping]:
    selected = [
        row
        for row in rows
        if row["regime_id"] == regime_id
        and row["method"] == method
        and float(row["lambda"]) == value
        and (model_seed is None or int(row["model_seed"]) == model_seed)
    ]
    expected = len(e11.INSTANCE_SEEDS) * (
        len(e11.MODEL_SEEDS) if model_seed is None else 1
    )
    if len(selected) != expected:
        raise E11FigureError(
            f"incomplete E11 coordinate {regime_id}/{method}/{value}/"
            f"{model_seed}: {len(selected)}/{expected}"
        )
    return selected


def _point(rows: Sequence[Mapping]) -> dict:
    return {
        "rows": len(rows),
        "mae": float(fmean(float(row["mean_absolute_error"]) for row in rows)),
        "rehandles_per_100": float(
            fmean(
                float(row["physical_rehandles_per_100_required_deliveries"])
                for row in rows
            )
        ),
        "return": float(fmean(float(row["dense_return"]) for row in rows)),
        "within_window": float(
            fmean(float(row["within_target_window_rate"]) for row in rows)
        ),
        "steps": float(fmean(float(row["steps"]) for row in rows)),
    }


def _check_point_against_report(
    point: Mapping, report_item: Mapping, *, label: str
) -> None:
    mapping = {
        "mae": "mean_absolute_error",
        "rehandles_per_100": "physical_rehandles_per_100_required_deliveries",
        "return": "dense_return",
        "within_window": "within_target_window_rate",
        "steps": "steps",
    }
    for point_metric, report_metric in mapping.items():
        if abs(float(point[point_metric]) - float(report_item[report_metric])) > 1e-10:
            raise E11FigureError(
                f"E11 figure/report mismatch for {label}/{report_metric}"
            )


def build_operating_data(rows: Sequence[Mapping], report: Mapping) -> dict:
    regimes = []
    for regime_id in REGIME_ORDER:
        aggregate = []
        per_seed = {str(seed): [] for seed in e11.MODEL_SEEDS}
        for method, value, label, color, marker in COORDINATES:
            point = {
                "method": method,
                "lambda": value,
                "label": label,
                "color": color,
                "marker": marker,
                **_point(_rows_for(rows, regime_id, method, value)),
            }
            _check_point_against_report(
                point,
                report["aggregate"][regime_id][_coordinate_key(method, value)],
                label=f"{regime_id}/{method}/{value}",
            )
            aggregate.append(point)
            for seed in e11.MODEL_SEEDS:
                per_seed[str(seed)].append(
                    {
                        "method": method,
                        "lambda": value,
                        **_point(
                            _rows_for(
                                rows,
                                regime_id,
                                method,
                                value,
                                model_seed=seed,
                            )
                        ),
                    }
                )
        regimes.append(
            {
                "regime_id": regime_id,
                "label": REGIME_LABELS[regime_id],
                "aggregate": aggregate,
                "per_model_seed": per_seed,
            }
        )
    return {
        "schema_version": 1,
        "protocol": e11.PROTOCOL,
        "report_sha256": report["report_sha256"],
        "figure_contract": (
            "seven_shared_axis_shift_panels_with_three_seed_paths_v1"
        ),
        "coordinate_order": [item[1] for item in COORDINATES],
        "regimes": regimes,
    }


def bootstrap_indices(
    *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(
        0, len(e11.MODEL_SEEDS), size=(replicates, len(e11.MODEL_SEEDS))
    )
    instance_draws = rng.integers(
        0,
        len(e11.INSTANCE_SEEDS),
        size=(replicates, len(e11.MODEL_SEEDS), len(e11.INSTANCE_SEEDS)),
    )
    return seed_draws, instance_draws


def hierarchical_interval(
    matrix: np.ndarray,
    seed_draws: np.ndarray,
    instance_draws: np.ndarray,
) -> tuple[float, float, float]:
    expected_shape = (len(e11.MODEL_SEEDS), len(e11.INSTANCE_SEEDS))
    if matrix.shape != expected_shape:
        raise E11FigureError(
            f"paired matrix has shape {matrix.shape}; expected {expected_shape}"
        )
    sampled = np.empty(
        (
            seed_draws.shape[0],
            len(e11.MODEL_SEEDS),
            len(e11.INSTANCE_SEEDS),
        ),
        dtype=float,
    )
    for slot in range(len(e11.MODEL_SEEDS)):
        sampled[:, slot, :] = matrix[
            seed_draws[:, slot][:, None], instance_draws[:, slot, :]
        ]
    bootstrap_means = sampled.mean(axis=(1, 2))
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(matrix.mean()), float(lower), float(upper)


def _paired_matrix(
    rows: Sequence[Mapping],
    regime_id: str,
    method: str,
    value: float,
    metric: str,
) -> np.ndarray:
    reference = {
        (int(row["model_seed"]), int(row["instance_seed"])): row
        for row in _rows_for(rows, "reference", method, value)
    }
    shifted = {
        (int(row["model_seed"]), int(row["instance_seed"])): row
        for row in _rows_for(rows, regime_id, method, value)
    }
    matrix = np.empty((len(e11.MODEL_SEEDS), len(e11.INSTANCE_SEEDS)))
    for seed_index, seed in enumerate(e11.MODEL_SEEDS):
        for instance_index, instance_seed in enumerate(e11.INSTANCE_SEEDS):
            key = (seed, instance_seed)
            if key not in reference or key not in shifted:
                raise E11FigureError(f"missing paired E11 coordinate: {key}")
            matrix[seed_index, instance_index] = float(shifted[key][metric]) - float(
                reference[key][metric]
            )
    return matrix


def build_paired_statistics(rows: Sequence[Mapping], report: Mapping) -> dict:
    seed_draws, instance_draws = bootstrap_indices()
    shifts = []
    for regime_id in REGIME_ORDER[1:]:
        coordinates = []
        for method, value, label, color, marker in COORDINATES:
            metrics = {}
            for metric in METRICS:
                matrix = _paired_matrix(rows, regime_id, method, value, metric)
                mean, lower, upper = hierarchical_interval(
                    matrix, seed_draws, instance_draws
                )
                metrics[metric] = {
                    "mean_difference": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "model_seed_mean_differences": [
                        float(matrix[index].mean())
                        for index in range(len(e11.MODEL_SEEDS))
                    ],
                }
                report_value = report["paired_shift_minus_reference"][regime_id][
                    _coordinate_key(method, value)
                ][f"{metric}_shift_minus_reference"]
                if abs(mean - float(report_value)) > 1e-10:
                    raise E11FigureError(
                        f"paired-statistics/report mismatch for "
                        f"{regime_id}/{method}/{value}/{metric}"
                    )
            coordinates.append(
                {
                    "method": method,
                    "lambda": value,
                    "label": label,
                    "color": color,
                    "marker": marker,
                    "pairs": len(e11.MODEL_SEEDS) * len(e11.INSTANCE_SEEDS),
                    "metrics": metrics,
                }
            )
        shifts.append(
            {
                "regime_id": regime_id,
                "label": REGIME_LABELS[regime_id],
                "coordinates": coordinates,
            }
        )
    return {
        "schema_version": 1,
        "protocol": e11.PROTOCOL,
        "report_sha256": report["report_sha256"],
        "comparison": "shift_minus_paired_reference",
        "pairs_per_coordinate": len(e11.MODEL_SEEDS) * len(e11.INSTANCE_SEEDS),
        "bootstrap": {
            "method": (
                "hierarchical_percentile_bootstrap_resampling_model_seeds_"
                "then_paired_instances_within_seed"
            ),
            "replicates": BOOTSTRAP_REPLICATES,
            "rng_seed": BOOTSTRAP_SEED,
            "interval": 0.95,
            "qualification": (
                "pointwise_descriptive_uncertainty_with_only_three_model_seeds"
            ),
        },
        "shifts": shifts,
    }


def render_operating_points(output_dir: Path, data: Mapping) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(14.1, 7.2), sharex=True, sharey=True)
    panel_axes = list(axes.flat)
    for axis, regime in zip(panel_axes, data["regimes"]):
        for seed_points in regime["per_model_seed"].values():
            x = [point["rehandles_per_100"] for point in seed_points]
            y = [point["mae"] for point in seed_points]
            axis.plot(x, y, color="#9AA3AD", linewidth=1.05, alpha=0.48, zorder=1)
            for point, coordinate in zip(seed_points, COORDINATES):
                axis.scatter(
                    point["rehandles_per_100"],
                    point["mae"],
                    s=22,
                    marker=coordinate[4],
                    facecolor=coordinate[3],
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.42,
                    zorder=2,
                )

        aggregate = regime["aggregate"]
        x = [point["rehandles_per_100"] for point in aggregate]
        y = [point["mae"] for point in aggregate]
        axis.plot(x, y, color="#243746", linewidth=2.5, zorder=3)
        for point in aggregate:
            axis.scatter(
                point["rehandles_per_100"],
                point["mae"],
                s=70,
                marker=point["marker"],
                facecolor=point["color"],
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
        axis.set_title(regime["label"], loc="left", weight="semibold")
        axis.grid(alpha=0.19, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-3.5, 71.5)
        axis.set_ylim(5.5, 33.2)

    legend_axis = panel_axes[-1]
    legend_axis.axis("off")
    handles = [
        Line2D(
            [0], [0], color="#9AA3AD", linewidth=1.1, alpha=0.65,
            marker="o", markerfacecolor="#9AA3AD", markeredgecolor="white",
            markersize=5, label="Model-seed path",
        )
    ]
    handles.extend(
        Line2D(
            [0], [0], linestyle="none", marker=marker, markersize=7,
            markerfacecolor=color, markeredgecolor="white", label=label,
        )
        for _method, _value, label, color, marker in COORDINATES
    )
    legend_axis.legend(handles=handles, loc="upper left", frameon=False)
    legend_axis.text(
        0.0,
        0.43,
        "2,730 / 2,730 strict completions\n\n"
        "Bold path: three-model-seed mean\n"
        "Light paths: individual model-seed means\n\n"
        "Lines connect prespecified preferences;\n"
        "they are not temporal trajectories.",
        transform=legend_axis.transAxes,
        fontsize=9,
        color="#505B66",
        va="top",
        linespacing=1.35,
    )
    figure.suptitle(
        "Frozen VCG operating points under distribution shift",
        x=0.065,
        ha="left",
        fontsize=15,
    )
    figure.supxlabel("Physical rehandles / 100 required deliveries ↓", fontsize=11)
    figure.supylabel("Mean absolute delivery error ↓", fontsize=11)
    figure.tight_layout(rect=(0.035, 0.04, 1, 0.95))
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{OPERATING_STEM}.{suffix}"
        figure.savefig(path, dpi=260, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def render_shift_effects(output_dir: Path, statistics: Mapping) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    plotted = (
        ("mean_absolute_error", r"$\Delta$ MAE (shift − reference)"),
        (
            "physical_rehandles_per_100_required_deliveries",
            r"$\Delta$ physical rehandles / 100 (shift − reference)",
        ),
    )
    x = np.arange(len(statistics["shifts"]), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(COORDINATES))
    for axis, (metric, ylabel) in zip(axes, plotted):
        for coordinate_index, coordinate in enumerate(COORDINATES):
            means = []
            lower = []
            upper = []
            for shift in statistics["shifts"]:
                item = shift["coordinates"][coordinate_index]["metrics"][metric]
                means.append(item["mean_difference"])
                lower.append(item["ci95_lower"])
                upper.append(item["ci95_upper"])
            means_array = np.asarray(means)
            yerr = np.vstack(
                (means_array - np.asarray(lower), np.asarray(upper) - means_array)
            )
            axis.errorbar(
                x + offsets[coordinate_index],
                means_array,
                yerr=yerr,
                fmt=coordinate[4],
                markersize=6,
                color=coordinate[3],
                markeredgecolor="white",
                markeredgewidth=0.7,
                capsize=2.5,
                elinewidth=1.15,
                linewidth=0,
                label=coordinate[2],
                zorder=3,
            )
        axis.axhline(0, color="#59636E", linewidth=1.0, alpha=0.75, zorder=1)
        axis.set_xticks(
            x,
            [shift["label"].replace(" ", "\n", 1) for shift in statistics["shifts"]],
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.20, linewidth=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) Timing sensitivity", loc="left")
    axes[1].set_title("(b) Handling sensitivity", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.suptitle(
        "Zero-shot change from the paired in-distribution reference",
        x=0.065,
        ha="left",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.105,
        "Points: paired means; bars: pointwise hierarchical 95% bootstrap intervals",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#626B75",
    )
    figure.tight_layout(rect=(0, 0.19, 1, 0.93))
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{SHIFT_STEM}.{suffix}"
        figure.savefig(path, dpi=260, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def render_statistics_table(output_dir: Path, statistics: Mapping) -> str:
    lines = [
        "# E11 paired distribution-shift effects",
        "",
        "Differences are shift minus the paired in-distribution reference. "
        "Intervals are pointwise hierarchical 95% bootstrap intervals over model "
        "seeds and paired instances.",
        "",
        "| Shift | Frozen policy | ΔMAE [95% CI] | Δ rehandles/100 [95% CI] | Δ return [95% CI] | Δ steps [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def display(item: Mapping, digits: int = 2) -> str:
        return (
            f"{item['mean_difference']:.{digits}f} "
            f"[{item['ci95_lower']:.{digits}f}, {item['ci95_upper']:.{digits}f}]"
        )

    for shift in statistics["shifts"]:
        for coordinate in shift["coordinates"]:
            metric = coordinate["metrics"]
            label = (
                r"$Q_{op}$ ($\lambda=0$)"
                if coordinate["method"] == e11.QOP
                else f"VCG ($\\lambda={coordinate['lambda']:.2f}$)"
            )
            lines.append(
                f"| {shift['label']} | {label} | "
                f"{display(metric['mean_absolute_error'])} | "
                f"{display(metric['physical_rehandles_per_100_required_deliveries'])} | "
                f"{display(metric['dense_return'])} | "
                f"{display(metric['steps'])} |"
            )
    path = output_dir / TABLE_NAME
    final90._atomic_text(path, "\n".join(lines) + "\n")
    return str(path.resolve())


def render(project_root: Path, output_dir: Path) -> dict:
    contract, _manifest = e11.authenticate(project_root, output_dir)
    report = e11._load_json(output_dir / e11.REPORT_NAME, label="E11 report")
    e11._verify_hash(report, "report_sha256", label="E11 report")
    if (
        report.get("status") != "complete"
        or report.get("paper_evidence") is not True
        or report.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise E11FigureError(
            "E11 paper rendering requires the authenticated complete report"
        )
    rows = e11._all_rows(output_dir)
    if len(rows) != int(contract["expected_full_rows"]):
        raise E11FigureError("E11 paper rendering requires all declared rows")
    if not all(
        row.get("strict_safe_complete") is True
        and row.get("all_selected_candidates_exact_safe") is True
        for row in rows
    ):
        raise E11FigureError("E11 paper rendering requires strict exact-SAFE rows")

    operating_data = build_operating_data(rows, report)
    statistics = build_paired_statistics(rows, report)
    final90._atomic_json(output_dir / DATA_NAME, operating_data)
    final90._atomic_json(output_dir / STATS_NAME, statistics)
    return {
        "status": "complete",
        "rows": len(rows),
        "report_sha256": report["report_sha256"],
        "operating_point_figures": render_operating_points(
            output_dir, operating_data
        ),
        "paired_shift_figures": render_shift_effects(output_dir, statistics),
        "paired_shift_table": render_statistics_table(output_dir, statistics),
        "figure_data": str((output_dir / DATA_NAME).resolve()),
        "paired_statistics": str((output_dir / STATS_NAME).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=e11.DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
