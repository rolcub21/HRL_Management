#!/usr/bin/env python3
"""Render the paper-facing E4 ranking figure and tightened result table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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

from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04
import run_vcg_conditioned_final_comparison_90k as final90


FIGURE_STEM = "e04-certified-frontier-ranking"
TABLE_STEM = "e04-paper-results-table"
DATA_NAME = "e04-figure-data.json"

QOP_COLOR = "#0072B2"
LAMBDA_COLORS = {
    0.05: "#35A89A",
    0.10: "#009E73",
    0.20: "#006B57",
}
RANDOM_COLOR = "#7A7F87"
HEURISTIC_COLOR = "#E69F00"
SEED_COLOR = "#A5ACB5"

LEARNED_COORDINATES = (
    (e04.QOP_SAFE, 0.0, r"VCG $Q_{op}$ ($\lambda=0$)"),
    (e04.CONDITIONED_SAFE, 0.05, r"VCG ($\lambda=.05$)"),
    (e04.CONDITIONED_SAFE, 0.10, r"VCG ($\lambda=.10$)"),
    (e04.CONDITIONED_SAFE, 0.20, r"VCG ($\lambda=.20$)"),
)


class E4PlotError(RuntimeError):
    pass


def _mean(rows: Sequence[Mapping], metric: str) -> float:
    if not rows:
        raise E4PlotError(f"cannot aggregate an empty group for {metric}")
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


def _learned_rows(rows: Sequence[Mapping], method: str, value: float, seed=None):
    selected = [
        row
        for row in rows
        if row["ranking_signal"] == method
        and float(row["preference_lambda"]) == value
        and (seed is None or int(row["model_seed"]) == seed)
    ]
    expected = (len(e04.MODEL_SEEDS) if seed is None else 1) * len(
        e04.INSTANCE_SEEDS
    )
    if len(selected) != expected:
        raise E4PlotError(
            f"incomplete learned coordinate {method}/{value}/{seed}: "
            f"{len(selected)}/{expected}"
        )
    return selected


def build_figure_data(rows: Sequence[Mapping], report: Mapping) -> dict:
    if len(rows) != 540 or not all(
        row.get("strict_safe_complete") is True
        and row.get("all_selected_candidates_exact_safe") is True
        for row in rows
    ):
        raise E4PlotError("E4 figure requires all 540 strict exact-SAFE rows")

    aggregate = []
    per_seed = {str(seed): [] for seed in e04.MODEL_SEEDS}
    for method, value, label in LEARNED_COORDINATES:
        aggregate.append(
            {
                "method": method,
                "lambda": value,
                "label": label,
                **_point(_learned_rows(rows, method, value)),
            }
        )
        for seed in e04.MODEL_SEEDS:
            per_seed[str(seed)].append(
                {
                    "method": method,
                    "lambda": value,
                    **_point(_learned_rows(rows, method, value, seed)),
                }
            )

    random_replicates = []
    for ranking_seed in e04.RANDOM_RANKING_SEEDS:
        group = [
            row
            for row in rows
            if row["ranking_signal"] == e04.RANDOM_SAFE
            and int(row["ranking_seed"]) == ranking_seed
        ]
        if len(group) != len(e04.INSTANCE_SEEDS):
            raise E4PlotError("random-safe ranking replicate is incomplete")
        random_replicates.append(
            {"ranking_seed": ranking_seed, **_point(group)}
        )
    random_rows = [row for row in rows if row["ranking_signal"] == e04.RANDOM_SAFE]
    heuristic_rows = [
        row for row in rows if row["ranking_signal"] == e04.HEURISTIC_SAFE
    ]
    if len(heuristic_rows) != len(e04.INSTANCE_SEEDS):
        raise E4PlotError("heuristic-safe rows are incomplete")

    return {
        "schema_version": 1,
        "protocol": e04.PROTOCOL,
        "report_sha256": report["report_sha256"],
        "figure_contract": (
            "full_safe_ranking_landscape_plus_three_seed_learned_zoom_v1"
        ),
        "learned_coordinate_order": [item[1] for item in LEARNED_COORDINATES],
        "learned_aggregate": aggregate,
        "learned_per_model_seed": per_seed,
        "random_safe_aggregate": _point(random_rows),
        "random_safe_per_ranking_seed": random_replicates,
        "heuristic_safe": _point(heuristic_rows),
    }


def _plot_point(axis, point, *, color, marker, size, label=None, zorder=4):
    axis.scatter(
        [point["rehandles_per_100"]],
        [point["mae"]],
        color=color,
        edgecolor="white",
        linewidth=0.8,
        marker=marker,
        s=size,
        label=label,
        zorder=zorder,
    )


def _learned_xy(points):
    return (
        [point["rehandles_per_100"] for point in points],
        [point["mae"] for point in points],
    )


def render_figure(output_dir: Path, data: Mapping) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, (full_axis, zoom_axis) = plt.subplots(
        1, 2, figsize=(11.4, 4.65), gridspec_kw={"width_ratios": (1.03, 1.18)}
    )

    aggregate = data["learned_aggregate"]
    x, y = _learned_xy(aggregate)
    full_axis.plot(x, y, color="#3E5968", linewidth=2.1, zorder=2)
    for index, point in enumerate(aggregate):
        color = QOP_COLOR if index == 0 else LAMBDA_COLORS[point["lambda"]]
        marker = "o" if index == 0 else "D"
        _plot_point(full_axis, point, color=color, marker=marker, size=68)

    for point in data["random_safe_per_ranking_seed"]:
        full_axis.scatter(
            point["rehandles_per_100"],
            point["mae"],
            marker="x",
            s=34,
            linewidth=1.2,
            color=RANDOM_COLOR,
            alpha=0.48,
            zorder=2,
        )
    _plot_point(
        full_axis,
        data["random_safe_aggregate"],
        color=RANDOM_COLOR,
        marker="X",
        size=90,
    )
    _plot_point(
        full_axis,
        data["heuristic_safe"],
        color=HEURISTIC_COLOR,
        marker="s",
        size=78,
    )

    for seed in e04.MODEL_SEEDS:
        seed_points = data["learned_per_model_seed"][str(seed)]
        seed_x, seed_y = _learned_xy(seed_points)
        zoom_axis.plot(
            seed_x,
            seed_y,
            color=SEED_COLOR,
            linewidth=1.15,
            alpha=0.72,
            zorder=1,
        )
        zoom_axis.scatter(
            seed_x,
            seed_y,
            facecolor="white",
            edgecolor=SEED_COLOR,
            linewidth=1.0,
            s=30,
            zorder=2,
        )
        zoom_axis.annotate(
            f"seed {seed}",
            (seed_x[-1], seed_y[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color="#747C86",
            va="center",
        )

    zoom_axis.plot(x, y, color="#253E4A", linewidth=2.5, zorder=3)
    annotation_offsets = ((7, -15), (4, 9), (5, 9), (-7, 10))
    for index, (point, offset) in enumerate(zip(aggregate, annotation_offsets)):
        color = QOP_COLOR if index == 0 else LAMBDA_COLORS[point["lambda"]]
        marker = "o" if index == 0 else "D"
        _plot_point(zoom_axis, point, color=color, marker=marker, size=82, zorder=4)
        label = r"$Q_{op}$" if index == 0 else f"$\\lambda={point['lambda']:.2f}$"
        zoom_axis.annotate(
            label,
            (point["rehandles_per_100"], point["mae"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            weight="semibold",
        )

    for axis in (full_axis, zoom_axis):
        axis.set_xlabel("Physical rehandles / 100 required deliveries ↓")
        axis.set_ylabel("Mean absolute delivery error ↓")
        axis.grid(alpha=0.20, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    full_axis.set_title("(a) All certified-frontier ranking signals", loc="left")
    zoom_axis.set_title("(b) Learned VCG operating family", loc="left")
    full_axis.margins(x=0.08, y=0.10)
    zoom_axis.set_xlim(-1.5, 31.5)
    observed_y = [
        point["mae"]
        for values in data["learned_per_model_seed"].values()
        for point in values
    ]
    zoom_axis.set_ylim(min(observed_y) - 0.45, max(observed_y) + 0.60)

    handles = [
        Line2D(
            [0], [0], color="#253E4A", marker="o", markerfacecolor=QOP_COLOR,
            markeredgecolor="white", linewidth=2.2, label=r"VCG $Q_{op}$ and handling path"
        ),
        Line2D(
            [0], [0], color="none", marker="X", markerfacecolor=RANDOM_COLOR,
            markeredgecolor="white", markersize=8, label="Random SAFE"
        ),
        Line2D(
            [0], [0], color="none", marker="s", markerfacecolor=HEURISTIC_COLOR,
            markeredgecolor="white", markersize=8, label="Heuristic SAFE"
        ),
        Line2D(
            [0], [0], color=SEED_COLOR, marker="o", markerfacecolor="white",
            markeredgecolor=SEED_COLOR, linewidth=1.1, label="Model-seed mean"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{FIGURE_STEM}.{suffix}"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


TABLE_ORDER = (
    (e04.QOP_SAFE, r"VCG $Q_{op}$ ($\lambda=0$)"),
    (f"{e04.CONDITIONED_SAFE}_lambda_0.05", r"VCG, $\lambda=.05$"),
    (f"{e04.CONDITIONED_SAFE}_lambda_0.10", r"VCG, $\lambda=.10$"),
    (f"{e04.CONDITIONED_SAFE}_lambda_0.20", r"VCG, $\lambda=.20$"),
    (e04.RANDOM_SAFE, "Random SAFE"),
    (e04.HEURISTIC_SAFE, "Heuristic SAFE"),
)


def _mean_sd(item: Mapping, metric: str, digits: int = 2) -> str:
    mean = item[metric]
    sd = item[f"{metric}_sd"]
    if mean is None:
        return "—"
    if sd is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ({float(sd):.{digits}f})"


def _paired_effects(rows: Sequence[Mapping]) -> list[dict]:
    indexed = {
        (
            row["ranking_signal"],
            int(row["model_seed"]),
            float(row["preference_lambda"]),
            int(row["instance_seed"]),
        ): row
        for row in rows
        if row["ranking_signal"] in (e04.QOP_SAFE, e04.CONDITIONED_SAFE)
    }
    effects = []
    for value in e04.CONDITIONED_LAMBDAS:
        pairs = []
        for seed in e04.MODEL_SEEDS:
            for instance_seed in e04.INSTANCE_SEEDS:
                qop = indexed[(e04.QOP_SAFE, seed, 0.0, instance_seed)]
                conditioned = indexed[
                    (e04.CONDITIONED_SAFE, seed, value, instance_seed)
                ]
                pairs.append((conditioned, qop))
        effects.append(
            {
                "lambda": value,
                "pairs": len(pairs),
                "mae_difference": fmean(
                    float(conditioned["mean_absolute_error"])
                    - float(qop["mean_absolute_error"])
                    for conditioned, qop in pairs
                ),
                "rehandle_difference": fmean(
                    float(conditioned["physical_rehandles_per_100_required_deliveries"])
                    - float(qop["physical_rehandles_per_100_required_deliveries"])
                    for conditioned, qop in pairs
                ),
                "step_difference": fmean(
                    float(conditioned["steps"]) - float(qop["steps"])
                    for conditioned, qop in pairs
                ),
                "return_difference": fmean(
                    float(conditioned["dense_return"])
                    - float(qop["dense_return"])
                    for conditioned, qop in pairs
                ),
            }
        )
    return effects


def render_tables(output_dir: Path, report: Mapping, rows: Sequence[Mapping]):
    records = []
    lines = [
        "# E4 certified-frontier ranking results",
        "",
        "| Ranking signal | Rows | Strict completion | MAE ↓ | Rehandles/100 ↓ | Within ±20 ↑ | Steps ↓ | Return ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in TABLE_ORDER:
        item = report["aggregate"][key]
        record = {
            "ranking_signal": label.replace("$", ""),
            "rows": item["rows"],
            "strict_complete": item["strict_safe_complete"],
            "mae_mean_sd": _mean_sd(item, "mean_absolute_error"),
            "rehandles_per_100_mean_sd": _mean_sd(
                item, "physical_rehandles_per_100_required_deliveries"
            ),
            "within_20_rate": item["within_target_window_rate"],
            "steps_mean_sd": _mean_sd(item, "steps"),
            "return_mean_sd": _mean_sd(item, "dense_return"),
        }
        records.append(record)
        lines.append(
            f"| {label} | {item['rows']} | "
            f"{item['strict_safe_complete']}/{item['rows']} | "
            f"{record['mae_mean_sd']} | {record['rehandles_per_100_mean_sd']} | "
            f"{100.0 * item['within_target_window_rate']:.1f}% | "
            f"{record['steps_mean_sd']} | {record['return_mean_sd']} |"
        )

    effects = _paired_effects(rows)
    lines.extend(
        [
            "",
            "Values are mean (SD) across rollout rows. Learned rankings contain 3 model seeds × 30 instances; Random SAFE contains 5 fixed ranking seeds × 30 instances; Heuristic SAFE is deterministic on 30 instances.",
            "",
            "## Matched conditioned-minus-$Q_{op}$ differences",
            "",
            "| $\\lambda$ | Matched pairs | ΔMAE | ΔRehandles/100 | ΔSteps | ΔReturn |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for effect in effects:
        lines.append(
            f"| {effect['lambda']:.2f} | {effect['pairs']} | "
            f"{effect['mae_difference']:+.2f} | {effect['rehandle_difference']:+.2f} | "
            f"{effect['step_difference']:+.2f} | {effect['return_difference']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "All 540 rows completed strictly and every selected candidate was exact-SAFE. Paired differences use identical model-seed/EpisodeInstance coordinates and are descriptive; the figure exposes model-seed variation directly.",
        ]
    )

    markdown_path = output_dir / f"{TABLE_STEM}.md"
    final90._atomic_text(markdown_path, "\n".join(lines) + "\n")
    csv_path = output_dir / f"{TABLE_STEM}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return [str(markdown_path.resolve()), str(csv_path.resolve())], effects


def render(project_root: Path, output_dir: Path) -> dict:
    contract = e04.authenticate_contract(project_root, output_dir)
    manifest = e04.authenticate_manifest(project_root, output_dir)
    report = e04._load_json(output_dir / e04.REPORT_NAME, label="E4 report")
    e04._verify_hash(report, "report_sha256", label="E4 report")
    if (
        report.get("status") != "complete"
        or report.get("paper_evidence") is not True
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise E4PlotError("paper rendering requires the authenticated complete E4 report")
    rows = e04._all_ledgers(output_dir)
    data = build_figure_data(rows, report)
    final90._atomic_json(output_dir / DATA_NAME, data)
    figure_paths = render_figure(output_dir, data)
    table_paths, effects = render_tables(output_dir, report, rows)
    return {
        "status": "complete",
        "rows": len(rows),
        "report_sha256": report["report_sha256"],
        "figure_data": str((output_dir / DATA_NAME).resolve()),
        "figures": figure_paths,
        "tables": table_paths,
        "paired_effects": effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=e04.DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
