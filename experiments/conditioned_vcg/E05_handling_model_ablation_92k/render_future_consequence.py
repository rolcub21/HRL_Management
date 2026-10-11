#!/usr/bin/env python3
"""Render the paper-facing E5(c) future-consequence ablation figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    run_future_consequence as e05c,
)
import run_vcg_conditioned_final_comparison_90k as final90


FIGURE_STEM = "e05c-future-consequence-effect"
DATA_NAME = "e05c-figure-data.json"

LAMBDA_COLORS = {
    0.05: "#E69F00",
    0.10: "#009E73",
    0.20: "#0072B2",
}


class E5CFigureError(RuntimeError):
    pass


def _mean(rows: Sequence[Mapping], metric: str) -> float:
    if not rows:
        raise E5CFigureError(f"cannot aggregate an empty group for {metric}")
    return float(fmean(float(row[metric]) for row in rows))


def _point(rows: Sequence[Mapping]) -> dict:
    return {
        "rows": len(rows),
        "mae": _mean(rows, "mean_absolute_error"),
        "rehandles_per_100": _mean(
            rows, "physical_rehandles_per_100_required_deliveries"
        ),
        "steps": _mean(rows, "steps"),
        "return": _mean(rows, "dense_return"),
        "within_window": _mean(rows, "within_target_window_rate"),
    }


def collect_pairs(
    immediate_rows: Sequence[Mapping],
    control_lookup: Callable[[int, float, int], Mapping],
) -> list[dict]:
    expected = (
        len(e05c.MODEL_SEEDS)
        * len(e05c.DEPLOYMENT_LAMBDAS)
        * len(e05c.INSTANCE_SEEDS)
    )
    if len(immediate_rows) != expected:
        raise E5CFigureError(
            f"E5(c) figure requires the complete immediate-only grid: "
            f"{len(immediate_rows)}/{expected}"
        )

    observed = set()
    pairs = []
    for immediate in immediate_rows:
        coordinate = (
            int(immediate["model_seed"]),
            float(immediate["deployment_lambda"]),
            int(immediate["instance_seed"]),
        )
        if coordinate in observed:
            raise E5CFigureError(f"duplicate E5(c) coordinate: {coordinate}")
        observed.add(coordinate)
        control = control_lookup(*coordinate)
        if not (
            immediate.get("strict_safe_complete") is True
            and control.get("strict_safe_complete") is True
        ):
            raise E5CFigureError(
                "E5(c) paper figure requires strict completion in both arms"
            )
        pairs.append(
            {
                "model_seed": coordinate[0],
                "lambda": coordinate[1],
                "instance_seed": coordinate[2],
                "immediate": immediate,
                "future": control,
            }
        )

    required = {
        (seed, value, instance_seed)
        for seed in e05c.MODEL_SEEDS
        for value in e05c.DEPLOYMENT_LAMBDAS
        for instance_seed in e05c.INSTANCE_SEEDS
    }
    if observed != required:
        raise E5CFigureError("E5(c) coordinate set does not match the declared grid")
    return pairs


def build_figure_data(pairs: Sequence[Mapping], report: Mapping) -> dict:
    coordinates = []
    for value in e05c.DEPLOYMENT_LAMBDAS:
        selected = [pair for pair in pairs if float(pair["lambda"]) == value]
        if len(selected) != len(e05c.MODEL_SEEDS) * len(e05c.INSTANCE_SEEDS):
            raise E5CFigureError(f"incomplete E5(c) lambda coordinate: {value}")
        aggregate = {
            "lambda": value,
            "pairs": len(selected),
            "immediate": _point([pair["immediate"] for pair in selected]),
            "future": _point([pair["future"] for pair in selected]),
            "identical_behavior": sum(
                pair["immediate"].get("behavior_digest")
                == pair["future"].get("behavior_digest")
                for pair in selected
            ),
            "per_seed": [],
        }
        for seed in e05c.MODEL_SEEDS:
            seed_pairs = [
                pair for pair in selected if int(pair["model_seed"]) == seed
            ]
            if len(seed_pairs) != len(e05c.INSTANCE_SEEDS):
                raise E5CFigureError(
                    f"incomplete E5(c) seed/lambda coordinate: {seed}/{value}"
                )
            aggregate["per_seed"].append(
                {
                    "model_seed": seed,
                    "pairs": len(seed_pairs),
                    "immediate": _point(
                        [pair["immediate"] for pair in seed_pairs]
                    ),
                    "future": _point([pair["future"] for pair in seed_pairs]),
                }
            )
        coordinates.append(aggregate)

    return {
        "schema_version": 1,
        "protocol": e05c.PROTOCOL,
        "report_sha256": report["report_sha256"],
        "figure_contract": (
            "paired_immediate_to_future_operating_points_with_model_seed_means_v1"
        ),
        "direction": "arrows_run_from_immediate_only_to_learned_future",
        "coordinates": coordinates,
    }


def _check_report(data: Mapping, report: Mapping) -> None:
    metric_names = {
        "mae": "mean_absolute_error",
        "rehandles_per_100": "physical_rehandles_per_100_required_deliveries",
        "steps": "steps",
        "return": "dense_return",
        "within_window": "within_target_window_rate",
    }
    for coordinate in data["coordinates"]:
        value = float(coordinate["lambda"])
        for arm, condition in (
            ("future", e05c.FIXED_FUTURE),
            ("immediate", e05c.IMMEDIATE_ONLY),
        ):
            expected = report["aggregate"][f"{condition}_lambda_{value:.2f}"]
            for point_metric, report_metric in metric_names.items():
                observed = float(coordinate[arm][point_metric])
                if abs(observed - float(expected[report_metric])) > 1e-10:
                    raise E5CFigureError(
                        f"figure/report mismatch for {value}/{arm}/{report_metric}"
                    )


def _arrow(axis, start: Mapping, end: Mapping, *, color: str, width: float,
           alpha: float, mutation_scale: float, zorder: int) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (start["rehandles_per_100"], start["mae"]),
            (end["rehandles_per_100"], end["mae"]),
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=width,
            color=color,
            alpha=alpha,
            shrinkA=5,
            shrinkB=7,
            zorder=zorder,
        )
    )


def render_figure(output_dir: Path, data: Mapping) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.25))

    all_points = []
    label_offsets = {
        0.05: (2, 10),
        0.10: (3, -17),
        0.20: (5, 8),
    }
    for coordinate in data["coordinates"]:
        value = float(coordinate["lambda"])
        color = LAMBDA_COLORS[value]
        for seed_coordinate in coordinate["per_seed"]:
            immediate = seed_coordinate["immediate"]
            future = seed_coordinate["future"]
            all_points.extend((immediate, future))
            _arrow(
                axis,
                immediate,
                future,
                color=color,
                width=1.05,
                alpha=0.25,
                mutation_scale=8,
                zorder=1,
            )
            axis.scatter(
                immediate["rehandles_per_100"],
                immediate["mae"],
                s=25,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
                alpha=0.50,
                zorder=2,
            )
            axis.scatter(
                future["rehandles_per_100"],
                future["mae"],
                s=27,
                marker="D",
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.50,
                zorder=2,
            )

        immediate = coordinate["immediate"]
        future = coordinate["future"]
        all_points.extend((immediate, future))
        _arrow(
            axis,
            immediate,
            future,
            color=color,
            width=2.8,
            alpha=0.95,
            mutation_scale=14,
            zorder=4,
        )
        axis.scatter(
            immediate["rehandles_per_100"],
            immediate["mae"],
            s=105,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            zorder=5,
        )
        axis.scatter(
            future["rehandles_per_100"],
            future["mae"],
            s=100,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        midpoint = (
            (immediate["rehandles_per_100"] + future["rehandles_per_100"]) / 2,
            (immediate["mae"] + future["mae"]) / 2,
        )
        axis.annotate(
            f"$\\lambda={value:.2f}$",
            midpoint,
            xytext=label_offsets[value],
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            weight="semibold",
            zorder=6,
        )

    x_values = [point["rehandles_per_100"] for point in all_points]
    y_values = [point["mae"] for point in all_points]
    axis.set_xlim(min(-0.6, min(x_values) - 1.1), max(x_values) + 1.5)
    axis.set_ylim(min(y_values) - 0.45, max(y_values) + 0.55)
    axis.set_xlabel("Physical rehandles / 100 required deliveries ↓")
    axis.set_ylabel("Mean absolute delivery error ↓")
    axis.set_title("Effect of modeling future handling consequences", loc="left")
    axis.grid(alpha=0.20, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D(
            [0], [0], linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor="#4B5563", markeredgewidth=1.6, markersize=8,
            label="Immediate reconfigure cost only",
        ),
        Line2D(
            [0], [0], linestyle="none", marker="D", markerfacecolor="#4B5563",
            markeredgecolor="white", markersize=8,
            label="Learned future consequence",
        ),
        Line2D(
            [0], [0], color="#8A929C", linewidth=1.1, alpha=0.65,
            marker="o", markerfacecolor="white", markeredgecolor="#8A929C",
            markersize=4, label="Model-seed means",
        ),
    ]
    axis.legend(handles=handles, loc="upper right", frameon=False)
    axis.text(
        0.01,
        0.015,
        "Arrows: immediate-only → learned future; bold = 90-pair mean",
        transform=axis.transAxes,
        color="#606873",
        fontsize=8.5,
        va="bottom",
    )
    figure.tight_layout()

    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{FIGURE_STEM}.{suffix}"
        figure.savefig(path, dpi=260, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def render(
    project_root: Path,
    output_dir: Path,
    e04_output: Path,
    e05_output: Path,
) -> dict:
    contract = e05c.authenticate_contract(
        project_root, output_dir, e04_output, e05_output
    )
    report = e05c._load_json(output_dir / e05c.REPORT_NAME, label="E5(c) report")
    e05c._verify_hash(report, "report_sha256", label="E5(c) report")
    if (
        report.get("status") != "complete"
        or report.get("paper_evidence") is not True
        or report.get("contract_sha256") != contract["contract_sha256"]
    ):
        raise E5CFigureError(
            "paper rendering requires the authenticated complete E5(c) report"
        )

    immediate_rows = e05c._all_immediate_rows(output_dir)
    pairs = collect_pairs(
        immediate_rows,
        lambda seed, value, instance_seed: e05c._future_control(
            e04_output, e05_output, seed, value, instance_seed
        ),
    )
    data = build_figure_data(pairs, report)
    _check_report(data, report)
    final90._atomic_json(output_dir / DATA_NAME, data)
    figure_paths = render_figure(output_dir, data)
    return {
        "status": "complete",
        "pairs": len(pairs),
        "report_sha256": report["report_sha256"],
        "figure_data": str((output_dir / DATA_NAME).resolve()),
        "figures": figure_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=e05c.DEFAULT_OUTPUT)
    parser.add_argument("--e04-output", type=Path, default=e05c.E04_OUTPUT)
    parser.add_argument("--e05-output", type=Path, default=e05c.E05_OUTPUT)
    args = parser.parse_args()
    result = render(
        args.project_root.resolve(),
        args.output_dir.resolve(),
        args.e04_output.resolve(),
        args.e05_output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
