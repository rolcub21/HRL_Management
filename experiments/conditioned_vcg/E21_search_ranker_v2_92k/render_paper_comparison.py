#!/usr/bin/env python3
"""Render a derived E4+E21-v2 comparison without modifying E4 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import (
    render_results as e4_render,
)
from experiments.conditioned_vcg.E21_search_ranker_v2_92k import run as e21


FIGURE_STEM = "e04-e21-certified-frontier-ranking"
TABLE_NAME = "e04-e21-ranking-table.md"
DATA_NAME = "e04-e21-ranking-figure-data.json"
SEARCH_COLOR = "#8C5A9E"


class E21FigureError(RuntimeError):
    pass


def _load_data(output_dir: Path) -> dict:
    contract, manifest, parent = e21.authenticate(output_dir)
    report = e21._load_json(output_dir / e21.REPORT_NAME, "E21-v2 report")
    e21._verify_hash(report, "report_sha256", "E21-v2 report")
    if (
        report.get("status") != "complete"
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("new_search_rows") != len(e21.INSTANCE_SEEDS)
    ):
        raise E21FigureError("combined rendering requires complete E21-v2 results")

    e4_rows = e21.v1.e4._all_ledgers(e21.v1.E4_OUTPUT)
    e4_data = e4_render.build_figure_data(e4_rows, parent["report"])
    search_rows = e21._new_rows(output_dir)
    if len(search_rows) != len(e21.INSTANCE_SEEDS) or not all(
        row.get("strict_safe_complete") is True
        and row.get("all_selected_candidates_exact_safe") is True
        for row in search_rows
    ):
        raise E21FigureError("E21-v2 search rows are incomplete or not exact-SAFE")

    return {
        "schema_version": 1,
        "figure_contract": "authenticated_E4_plus_post_hoc_E21_v2_v1",
        "e4_report_sha256": parent["report"]["report_sha256"],
        "e21_report_sha256": report["report_sha256"],
        "e21_post_hoc": True,
        "e4": e4_data,
        "search": e4_render._point(search_rows),
        "e4_aggregate": parent["report"]["aggregate"],
        "e21_aggregate": report["aggregate"],
        "paired_search_minus_control": report["paired_search_minus_control"],
    }


def _search_point(axis, point: Mapping, *, annotate: bool) -> None:
    axis.scatter(
        [point["rehandles_per_100"]],
        [point["mae"]],
        facecolor="white",
        edgecolor=SEARCH_COLOR,
        linewidth=1.8,
        marker="^",
        s=105,
        zorder=5,
    )
    if annotate:
        axis.annotate(
            "Depth-2 search",
            (point["rehandles_per_100"], point["mae"]),
            xytext=(8, 2),
            textcoords="offset points",
            fontsize=8.5,
            color=SEARCH_COLOR,
            weight="semibold",
            va="center",
        )


def _render_figure(output_dir: Path, data: Mapping) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, (full_axis, zoom_axis) = plt.subplots(
        1, 2, figsize=(12.0, 4.7), gridspec_kw={"width_ratios": (1.03, 1.18)}
    )
    e4_data = data["e4"]
    aggregate = e4_data["learned_aggregate"]
    x, y = e4_render._learned_xy(aggregate)

    full_axis.plot(x, y, color="#3E5968", linewidth=2.1, zorder=2)
    for index, point in enumerate(aggregate):
        color = (
            e4_render.QOP_COLOR
            if index == 0
            else e4_render.LAMBDA_COLORS[point["lambda"]]
        )
        marker = "o" if index == 0 else "D"
        e4_render._plot_point(
            full_axis, point, color=color, marker=marker, size=68
        )

    for point in e4_data["random_safe_per_ranking_seed"]:
        full_axis.scatter(
            point["rehandles_per_100"],
            point["mae"],
            marker="x",
            s=34,
            linewidth=1.2,
            color=e4_render.RANDOM_COLOR,
            alpha=0.48,
            zorder=2,
        )
    e4_render._plot_point(
        full_axis,
        e4_data["random_safe_aggregate"],
        color=e4_render.RANDOM_COLOR,
        marker="X",
        size=90,
    )
    e4_render._plot_point(
        full_axis,
        e4_data["heuristic_safe"],
        color=e4_render.HEURISTIC_COLOR,
        marker="s",
        size=78,
    )
    _search_point(full_axis, data["search"], annotate=True)

    for seed in e21.v1.e4.MODEL_SEEDS:
        seed_points = e4_data["learned_per_model_seed"][str(seed)]
        seed_x, seed_y = e4_render._learned_xy(seed_points)
        zoom_axis.plot(
            seed_x,
            seed_y,
            color=e4_render.SEED_COLOR,
            linewidth=1.15,
            alpha=0.72,
            zorder=1,
        )
        zoom_axis.scatter(
            seed_x,
            seed_y,
            facecolor="white",
            edgecolor=e4_render.SEED_COLOR,
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
        color = (
            e4_render.QOP_COLOR
            if index == 0
            else e4_render.LAMBDA_COLORS[point["lambda"]]
        )
        marker = "o" if index == 0 else "D"
        e4_render._plot_point(
            zoom_axis, point, color=color, marker=marker, size=82, zorder=4
        )
        label = (
            r"$Q_{op}$"
            if index == 0
            else f"$\\lambda={point['lambda']:.2f}$"
        )
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
    full_axis.set_title("(a) Certified-frontier ranking signals", loc="left")
    zoom_axis.set_title("(b) Learned VCG operating family", loc="left")
    full_axis.margins(x=0.08, y=0.10)
    zoom_axis.set_xlim(-1.5, 31.5)
    observed_y = [
        point["mae"]
        for values in e4_data["learned_per_model_seed"].values()
        for point in values
    ]
    zoom_axis.set_ylim(min(observed_y) - 0.45, max(observed_y) + 0.60)

    handles = [
        Line2D(
            [0],
            [0],
            color="#253E4A",
            marker="o",
            markerfacecolor=e4_render.QOP_COLOR,
            markeredgecolor="white",
            linewidth=2.2,
            label=r"VCG $Q_{op}$ and handling path",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="^",
            markerfacecolor="white",
            markeredgecolor=SEARCH_COLOR,
            markeredgewidth=1.5,
            markersize=8,
            label="Depth-2 search (post-hoc)",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="X",
            markerfacecolor=e4_render.RANDOM_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="Random SAFE",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="s",
            markerfacecolor=e4_render.HEURISTIC_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="Heuristic SAFE",
        ),
        Line2D(
            [0],
            [0],
            color=e4_render.SEED_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=e4_render.SEED_COLOR,
            linewidth=1.1,
            label="Model-seed mean",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))

    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{FIGURE_STEM}.{suffix}"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def _mean_sd(item: Mapping, metric: str, digits: int = 2) -> str:
    mean = item.get(metric)
    sd = item.get(f"{metric}_sd")
    if mean is None:
        return "—"
    if sd is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ({float(sd):.{digits}f})"


def _interval(item: Mapping, metric: str, digits: int = 2) -> str:
    result = item[metric]
    mean = result["mean_search_minus_control"]
    interval = result["ci95"]
    if mean is None or interval is None:
        return "—"
    return (
        f"{float(mean):+.{digits}f} "
        f"[{float(interval[0]):.{digits}f}, {float(interval[1]):.{digits}f}]"
    )


def _render_table(output_dir: Path, data: Mapping) -> str:
    e4_aggregate = data["e4_aggregate"]
    e21_aggregate = data["e21_aggregate"]
    rows = (
        (e21.v1.e4.QOP_SAFE, r"VCG $Q_{op}$ ($\lambda=0$)"),
        (
            f"{e21.v1.e4.CONDITIONED_SAFE}_lambda_0.05",
            r"VCG, $\lambda=.05$",
        ),
        (
            f"{e21.v1.e4.CONDITIONED_SAFE}_lambda_0.10",
            r"VCG, $\lambda=.10$",
        ),
        (
            f"{e21.v1.e4.CONDITIONED_SAFE}_lambda_0.20",
            r"VCG, $\lambda=.20$",
        ),
        (e21.SEARCH_RANKER, "Certified depth-2 search v2*"),
        (e21.v1.e4.RANDOM_SAFE, "Random SAFE"),
        (e21.v1.e4.HEURISTIC_SAFE, "Heuristic SAFE"),
    )
    lines = [
        "# E4 ranking results with the post-hoc E21-v2 search baseline",
        "",
        "| Ranking signal | Rows | Strict completion | MAE ↓ | Rehandles/100 ↓ | Within ±20 ↑ | Steps ↓ | Return ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in rows:
        item = (
            e21_aggregate[key]
            if key == e21.SEARCH_RANKER
            else e4_aggregate[key]
        )
        lines.append(
            f"| {label} | {item['rows']} | "
            f"{item['strict_safe_complete']}/{item['rows']} | "
            f"{_mean_sd(item, 'mean_absolute_error')} | "
            f"{_mean_sd(item, 'physical_rehandles_per_100_required_deliveries')} | "
            f"{100.0 * item['within_target_window_rate']:.1f}% | "
            f"{_mean_sd(item, 'steps')} | {_mean_sd(item, 'dense_return')} |"
        )

    labels = {
        "heuristic_safe": "Heuristic SAFE",
        "frozen_qop": r"Frozen $Q_{op}$",
        "conditioned_lambda_0.10": r"VCG, $\lambda=.10$",
    }
    lines.extend(
        [
            "",
            "*E21-v2 was designed after diagnosing E21-v1 and is a post-hoc sensitivity baseline. The E4 rows are authenticated reanalyses, not independent replications. Learned E4 rows contain three model seeds on each of 30 instances; search and heuristic rows are deterministic on the 30 instances.",
            "",
            "## Paired E21-v2 search-minus-control differences",
            "",
            "| Comparator | ΔReturn | ΔMAE | ΔWithin ±20 | ΔSteps | ΔRehandles/100 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("heuristic_safe", "frozen_qop", "conditioned_lambda_0.10"):
        item = data["paired_search_minus_control"][key]
        lines.append(
            f"| {labels[key]} | {_interval(item, 'dense_return')} | "
            f"{_interval(item, 'mean_absolute_error')} | "
            f"{_interval(item, 'within_target_window_rate', 3)} | "
            f"{_interval(item, 'steps')} | "
            f"{_interval(item, 'physical_rehandles_per_100_required_deliveries')} |"
        )
    lines.extend(
        [
            "",
            "Differences are search minus comparator with unadjusted paired 95% t intervals across the 30 common EpisodeInstances. For learned controls, the three model-seed outcomes are averaged within each instance before differencing.",
        ]
    )
    path = output_dir / TABLE_NAME
    e21._atomic_text(path, "\n".join(lines) + "\n")
    return str(path.resolve())


def render(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    data = _load_data(output_dir)
    e21._atomic_json(output_dir / DATA_NAME, data)
    figures = _render_figure(output_dir, data)
    table = _render_table(output_dir, data)
    return {
        "status": "complete",
        "e4_report_sha256": data["e4_report_sha256"],
        "e21_report_sha256": data["e21_report_sha256"],
        "figure_data": str((output_dir / DATA_NAME).resolve()),
        "figures": figures,
        "table": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=e21.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
