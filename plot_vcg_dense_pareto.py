#!/usr/bin/env python3
"""Plot the frozen VCG-Dense timing--relocation development diagnostic.

The figure is derived exclusively from ``pareto-report.json``.  Methods that
failed the report's lexicographic safety gate are named in the figure but are
never assigned numeric coordinates.  Episode-500 weights remain visibly
diagnostic and are not presented as deployment checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPORT_PROTOCOL = "vcg_dense_timing_relocation_pareto_analysis_v1"
SELECTED_GROUP = "selected_best"
FINAL_GROUP = "episode500_final_diagnostic"
DEFAULT_REPORT = Path(
    "results/vcg-dense-v1-1-pareto-development-30seed/pareto-report.json"
)
DEFAULT_FIGURE_NAME = "vcg-dense-v1-1-pareto"

SELECTED_COLOR = "#0072B2"
FINAL_COLOR = "#D55E00"
ENHANCED_GA_COLOR = "#009E73"
NEAREST_COLOR = "#666666"
SHIFT_COLOR = "#3A3A3A"
SEED_SHIFT_COLOR = "#8C8C8C"
MARGIN_COLOR = "#56B4E9"

BASELINE_LABELS = {
    "duration_aware_dynamic_pslap": "Dynamic PSLAP",
    "duration_aware_enhanced_complete_rolling_ga": "Enhanced rolling GA",
    "duration_aware_nearest_free": "Nearest-free",
    "duration_aware_pslap_ga_2009_rolling": "Rolling GA (2009)",
}


@dataclass(frozen=True)
class ParetoPoint:
    point_id: str
    label: str
    mae: float
    relocations_per_100: float
    policy_group: str
    diagnostic_only: bool
    model_seed: Optional[int] = None


@dataclass(frozen=True)
class SeedShift:
    model_seed: int
    selected: ParetoPoint
    final: ParetoPoint
    zero_length: bool


@dataclass(frozen=True)
class PlotSpec:
    aggregate_points: tuple[ParetoPoint, ...]
    selected_aggregate: ParetoPoint
    final_aggregate: ParetoPoint
    seed_shifts: tuple[SeedShift, ...]
    excluded_baselines: tuple[str, ...]
    mae_margin: float
    instance_count: int
    scope: str


def _mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)


def _baseline_label(method_id: str) -> str:
    return BASELINE_LABELS.get(
        method_id,
        method_id.replace("duration_aware_", "").replace("_", " ").title(),
    )


def load_report(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Pareto report must contain a JSON object")
    return payload


def _aggregate_point(point_id: str, payload: Mapping) -> ParetoPoint:
    diagnostic = bool(payload.get("diagnostic_only", False))
    if point_id == SELECTED_GROUP:
        label = "Selected VCG (3-seed mean)"
    elif point_id == FINAL_GROUP:
        label = "Episode-500 VCG (diagnostic mean)"
    else:
        label = _baseline_label(point_id)
    return ParetoPoint(
        point_id=point_id,
        label=label,
        mae=_finite(payload.get("mean_absolute_error"), f"{point_id}.MAE"),
        relocations_per_100=_finite(
            payload.get("relocations_per_100_deliveries"),
            f"{point_id}.relocations_per_100",
        ),
        policy_group=str(payload.get("policy_group")),
        diagnostic_only=diagnostic,
    )


def _seed_point(
    method_id: str,
    model_seed: int,
    policy_group: str,
    method_points: Mapping,
) -> ParetoPoint:
    method = _mapping(method_points.get(method_id), f"method_points.{method_id}")
    point = _mapping(method.get("point"), f"method_points.{method_id}.point")
    if int(method.get("model_seed")) != model_seed:
        raise ValueError(f"model-seed mismatch for {method_id}")
    if method.get("policy_group") != policy_group:
        raise ValueError(f"policy-group mismatch for {method_id}")
    if point.get("pareto_safety_eligible") is not True:
        raise ValueError(f"learned point is safety-ineligible: {method_id}")
    return ParetoPoint(
        point_id=method_id,
        label=f"Seed {model_seed}",
        mae=_finite(point.get("mean_absolute_error"), f"{method_id}.MAE"),
        relocations_per_100=_finite(
            point.get("relocations_per_100_deliveries"),
            f"{method_id}.relocations_per_100",
        ),
        policy_group=policy_group,
        diagnostic_only=policy_group == FINAL_GROUP,
        model_seed=model_seed,
    )


def build_plot_spec(report: Mapping) -> PlotSpec:
    """Validate the report and extract only safety-eligible plot coordinates."""

    if report.get("protocol") != REPORT_PROTOCOL:
        raise ValueError(f"unexpected report protocol: {report.get('protocol')!r}")
    if report.get("performance_claim_authorized") is not False:
        raise ValueError("this plot requires a development-only report")
    limits = _mapping(report.get("interpretation_limits"), "interpretation_limits")
    if limits.get("development_diagnostic_only") is not True:
        raise ValueError("development-diagnostic interpretation is not authenticated")
    if limits.get("final_weights_are_diagnostic_only") is not True:
        raise ValueError("episode-500 diagnostic status is not authenticated")

    validation = _mapping(report.get("validation"), "validation")
    instance_count = int(validation.get("instance_count", 0))
    if instance_count <= 0 or validation.get("complete_pairing_grid") is not True:
        raise ValueError("report does not contain a complete paired instance grid")
    if validation.get("deterministic_baselines_counted_once") is not True:
        raise ValueError("deterministic baselines were not counted exactly once")

    gate = _mapping(report.get("safety_gate"), "safety_gate")
    if gate.get("numeric_pareto_authorized") is not True:
        raise ValueError("numeric Pareto plotting is not authorized by the safety gate")
    eligible_baselines = tuple(str(value) for value in gate.get("eligible_baseline_method_ids", ()))
    excluded_baselines = tuple(str(value) for value in gate.get("excluded_baseline_method_ids", ()))
    if set(eligible_baselines) & set(excluded_baselines):
        raise ValueError("a baseline is both safety-eligible and excluded")
    ledgers = _mapping(gate.get("method_ledgers"), "safety_gate.method_ledgers")
    for method_id in excluded_baselines:
        ledger = _mapping(ledgers.get(method_id), f"method_ledgers.{method_id}")
        if (
            int(ledger.get("row_count", 0)) != 30
            or int(ledger.get("strict_successful_row_count", 0)) != 29
            or ledger.get("whole_method_excluded") is not True
        ):
            raise ValueError(
                f"unexpected safety-exclusion ledger for {method_id}"
            )

    pareto = _mapping(report.get("pareto"), "pareto")
    participants = _mapping(pareto.get("participants"), "pareto.participants")
    expected_participants = {SELECTED_GROUP, FINAL_GROUP, *eligible_baselines}
    if set(participants) != expected_participants:
        raise ValueError(
            "Pareto participants do not match the safety-eligible methods: "
            f"expected={sorted(expected_participants)}, found={sorted(participants)}"
        )
    if set(participants) & set(excluded_baselines):
        raise ValueError("a safety-excluded baseline has numeric Pareto coordinates")

    ordered_ids = (SELECTED_GROUP, FINAL_GROUP, *eligible_baselines)
    aggregates = tuple(
        _aggregate_point(point_id, _mapping(participants[point_id], point_id))
        for point_id in ordered_ids
    )
    aggregate_by_id = {point.point_id: point for point in aggregates}
    selected_aggregate = aggregate_by_id[SELECTED_GROUP]
    final_aggregate = aggregate_by_id[FINAL_GROUP]
    if selected_aggregate.diagnostic_only:
        raise ValueError("selected checkpoint group cannot be diagnostic-only")
    if not final_aggregate.diagnostic_only:
        raise ValueError("episode-500 group must be diagnostic-only")

    learned_ids = _mapping(validation.get("learned_method_ids"), "learned_method_ids")
    selected_ids = _mapping(learned_ids.get(SELECTED_GROUP), SELECTED_GROUP)
    final_ids = _mapping(learned_ids.get(FINAL_GROUP), FINAL_GROUP)
    if set(selected_ids) != {"0", "1", "2"} or set(final_ids) != {"0", "1", "2"}:
        raise ValueError("expected learned points for model seeds 0, 1, and 2")
    method_points = _mapping(report.get("method_points"), "method_points")
    per_seed_delta = _mapping(
        _mapping(report.get("best_vs_final"), "best_vs_final").get("per_model_seed"),
        "best_vs_final.per_model_seed",
    )

    shifts = []
    for seed in (0, 1, 2):
        key = str(seed)
        selected = _seed_point(
            str(selected_ids[key]), seed, SELECTED_GROUP, method_points
        )
        final = _seed_point(str(final_ids[key]), seed, FINAL_GROUP, method_points)
        delta = _mapping(per_seed_delta.get(key), f"per_model_seed.{key}")
        mae_cost = _finite(delta.get("mae_cost"), f"seed{seed}.mae_cost")
        relocation_saving = _finite(
            delta.get("relocation_saving_per_100_deliveries"),
            f"seed{seed}.relocation_saving",
        )
        if not _close(final.mae - selected.mae, mae_cost):
            raise ValueError(f"seed {seed} MAE shift disagrees with the report")
        if not _close(
            selected.relocations_per_100 - final.relocations_per_100,
            relocation_saving,
        ):
            raise ValueError(f"seed {seed} relocation shift disagrees with the report")
        zero = _close(selected.mae, final.mae) and _close(
            selected.relocations_per_100, final.relocations_per_100
        )
        shifts.append(SeedShift(seed, selected, final, zero))

    if not shifts[0].zero_length:
        raise ValueError("seed 0 must authenticate the selected/final zero shift")

    best_vs_final = _mapping(report.get("best_vs_final"), "best_vs_final")
    estimate = _mapping(best_vs_final.get("point_estimate"), "point_estimate")
    if not _close(
        final_aggregate.mae - selected_aggregate.mae,
        _finite(estimate.get("mae_cost"), "aggregate.mae_cost"),
    ):
        raise ValueError("aggregate MAE shift disagrees with the report")
    if not _close(
        selected_aggregate.relocations_per_100 - final_aggregate.relocations_per_100,
        _finite(
            estimate.get("relocation_saving_per_100_deliveries"),
            "aggregate.relocation_saving",
        ),
    ):
        raise ValueError("aggregate relocation shift disagrees with the report")

    guardrails = _mapping(report.get("guardrails"), "guardrails")
    margin = _finite(guardrails.get("mae_noninferiority_margin"), "MAE margin")
    if margin <= 0.0:
        raise ValueError("MAE noninferiority margin must be positive")

    return PlotSpec(
        aggregate_points=aggregates,
        selected_aggregate=selected_aggregate,
        final_aggregate=final_aggregate,
        seed_shifts=tuple(shifts),
        excluded_baselines=excluded_baselines,
        mae_margin=margin,
        instance_count=instance_count,
        scope=str(report.get("scope")),
    )


def _coordinates(point: ParetoPoint) -> tuple[float, float]:
    return point.mae, point.relocations_per_100


def _draw_shift(
    ax,
    shift: SeedShift,
    *,
    label_seed: bool,
    marker_size: float,
) -> None:
    start = _coordinates(shift.selected)
    end = _coordinates(shift.final)
    # A two-point Line2D is retained even for seed 0, whose endpoints coincide.
    ax.plot(
        (start[0], end[0]),
        (start[1], end[1]),
        color=SEED_SHIFT_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=1.25,
        alpha=0.9,
        zorder=2,
    )
    if not shift.zero_length:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": SEED_SHIFT_COLOR,
                "linestyle": (0, (3, 2)),
                "linewidth": 1.1,
                "mutation_scale": 9,
            },
            zorder=2,
        )
    ax.scatter(
        *start,
        s=marker_size,
        marker="o",
        facecolor=SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        *end,
        s=marker_size * 0.92,
        marker="d",
        facecolor="none",
        edgecolor=FINAL_COLOR,
        linewidth=1.3,
        zorder=5,
    )
    if shift.zero_length:
        ax.scatter(
            *start,
            s=marker_size * 1.75,
            marker="o",
            facecolor="none",
            edgecolor=FINAL_COLOR,
            linewidth=1.0,
            zorder=3,
        )
    if label_seed:
        offsets = {0: (-28, -16), 1: (-24, 9), 2: (-25, 9)}
        suffix = " (Δ=0)" if shift.zero_length else ""
        ax.annotate(
            f"s{shift.model_seed}{suffix}",
            start,
            xytext=offsets[shift.model_seed],
            textcoords="offset points",
            fontsize=7.5,
            color=SELECTED_COLOR,
            fontweight="bold",
            zorder=7,
        )


def _draw_aggregate_shift(ax, spec: PlotSpec, *, linewidth: float) -> None:
    start = _coordinates(spec.selected_aggregate)
    end = _coordinates(spec.final_aggregate)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": SHIFT_COLOR,
            "linewidth": linewidth,
            "mutation_scale": 13,
            "shrinkA": 7,
            "shrinkB": 7,
        },
        zorder=6,
    )


def _draw_aggregate_points(ax, spec: PlotSpec, *, annotate: bool) -> None:
    selected = spec.selected_aggregate
    final = spec.final_aggregate
    ax.scatter(
        *_coordinates(selected),
        s=220,
        marker="*",
        facecolor=SELECTED_COLOR,
        edgecolor="black",
        linewidth=0.8,
        zorder=9,
    )
    ax.scatter(
        *_coordinates(final),
        s=128,
        marker="D",
        facecolor="white",
        edgecolor=FINAL_COLOR,
        linewidth=2.0,
        zorder=9,
    )

    for point in spec.aggregate_points:
        if point.point_id in (SELECTED_GROUP, FINAL_GROUP):
            continue
        if point.point_id == "duration_aware_enhanced_complete_rolling_ga":
            color, marker = ENHANCED_GA_COLOR, "s"
        elif point.point_id == "duration_aware_nearest_free":
            color, marker = NEAREST_COLOR, "X"
        else:
            color, marker = "#CC79A7", "P"
        ax.scatter(
            *_coordinates(point),
            s=105,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=8,
        )

    if not annotate:
        return
    annotations = {
        SELECTED_GROUP: (-4, 15, "left"),
        FINAL_GROUP: (14, -24, "left"),
        "duration_aware_enhanced_complete_rolling_ga": (10, 11, "left"),
        "duration_aware_nearest_free": (-8, 15, "right"),
    }
    for point in spec.aggregate_points:
        dx, dy, alignment = annotations.get(point.point_id, (8, 8, "left"))
        ax.annotate(
            point.label,
            _coordinates(point),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=8.4,
            fontweight="bold" if point.point_id in (SELECTED_GROUP, FINAL_GROUP) else "normal",
            color=FINAL_COLOR if point.diagnostic_only else "#202020",
            zorder=10,
        )


def _draw_inset(ax, spec: PlotSpec) -> None:
    inset = ax.inset_axes([0.51, 0.39, 0.46, 0.53])
    selected_x = spec.selected_aggregate.mae
    margin_x = selected_x + spec.mae_margin
    inset.axvspan(
        selected_x,
        margin_x,
        color=MARGIN_COLOR,
        alpha=0.11,
        zorder=0,
    )
    inset.axvline(
        margin_x,
        color=MARGIN_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=1.1,
        zorder=1,
    )
    for shift in spec.seed_shifts:
        _draw_shift(inset, shift, label_seed=True, marker_size=31)
    _draw_aggregate_shift(inset, spec, linewidth=2.0)
    _draw_aggregate_points(inset, spec, annotate=False)

    learned = [
        point
        for shift in spec.seed_shifts
        for point in (shift.selected, shift.final)
    ]
    detail = learned + [spec.selected_aggregate, spec.final_aggregate]
    eligible_nearby = [
        point
        for point in spec.aggregate_points
        if point.point_id == "duration_aware_enhanced_complete_rolling_ga"
    ]
    detail.extend(eligible_nearby)
    x_values = [point.mae for point in detail] + [margin_x]
    y_values = [point.relocations_per_100 for point in detail]
    inset.set_xlim(min(x_values) - 0.45, max(x_values) + 0.45)
    inset.set_ylim(max(0.0, min(y_values) - 2.1), max(y_values) + 2.2)
    inset.set_title(
        "Learned-policy detail and MAE margin",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=5,
    )
    inset.text(
        margin_x - 0.04,
        0.97,
        f"+{spec.mae_margin:.1f} MAE",
        transform=inset.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.2,
        color="#236B8E",
    )
    inset.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
    inset.tick_params(axis="both", labelsize=7)
    inset.set_xlabel("MAE", fontsize=7.5)
    inset.set_ylabel("Relocations / 100", fontsize=7.5)
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)


def render_plot(
    spec: PlotSpec,
    output_prefix: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 6.35))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.205)

    for shift in spec.seed_shifts:
        _draw_shift(ax, shift, label_seed=False, marker_size=38)
    _draw_aggregate_shift(ax, spec, linewidth=2.2)
    _draw_aggregate_points(ax, spec, annotate=True)

    all_points = list(spec.aggregate_points)
    all_points.extend(
        point
        for shift in spec.seed_shifts
        for point in (shift.selected, shift.final)
    )
    x_values = [point.mae for point in all_points]
    y_values = [point.relocations_per_100 for point in all_points]
    ax.set_xlim(max(0.0, min(x_values) - 2.5), max(x_values) + 3.0)
    ax.set_ylim(-1.8, max(y_values) + 3.2)
    ax.set_xlabel("Mean absolute timing error (steps)")
    ax.set_ylabel("Relocations per 100 deliveries")
    ax.set_title(
        "Development timing–relocation Pareto diagnostic",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.0,
        1.012,
        (
            f"{spec.instance_count} paired contention instances; "
            "strict completion is a prerequisite; lower values are better"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#4A4A4A",
    )
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.annotate(
        "Lower-left is better",
        xy=(0.035, 0.055),
        xytext=(0.20, 0.19),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=8,
        color="#4A4A4A",
        arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 1.0},
    )

    _draw_inset(ax, spec)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=12,
            markerfacecolor=SELECTED_COLOR,
            markeredgecolor="black",
            label="Selected VCG aggregate",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=FINAL_COLOR,
            markeredgewidth=1.6,
            label="Episode-500 aggregate (diagnostic)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle=(0, (3, 2)),
            color=SEED_SHIFT_COLOR,
            markersize=5,
            markerfacecolor=SELECTED_COLOR,
            markeredgecolor="white",
            label="Per-seed selected → final",
        ),
        Line2D(
            [],
            [],
            color=SHIFT_COLOR,
            linewidth=2.2,
            label="Aggregate selected → final",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.70, 0.115),
        frameon=True,
        framealpha=0.96,
        edgecolor="#CCCCCC",
    )

    excluded = "; ".join(_baseline_label(item) for item in spec.excluded_baselines)
    fig.text(
        0.105,
        0.095,
        (
            "Safety-excluded baselines (no numeric point): "
            f"{excluded}. Each achieved 29/30 strict completions."
        ),
        ha="left",
        va="center",
        fontsize=8.4,
        color="#7A1F1F",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF4F2",
            "edgecolor": "#D8A09A",
            "linewidth": 0.8,
        },
    )
    fig.text(
        0.105,
        0.038,
        (
            "Episode-500 weights are diagnostic only. The shaded +2.0 MAE band "
            "shows the predeclared point-estimate margin, not a confirmatory bound."
        ),
        ha="left",
        va="center",
        fontsize=7.8,
        color="#555555",
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Title": "VCG-Dense timing-relocation Pareto development diagnostic",
            "Description": "Derived from the frozen 30-instance Pareto report",
        },
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Title": "VCG-Dense timing-relocation Pareto development diagnostic",
            "Subject": "Development-only timing-relocation capacity diagnostic",
            "Keywords": "VCG Dense, Pareto, timing error, relocation",
        },
    )
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the frozen VCG-Dense Pareto development report"
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> tuple[Path, Path]:
    args = build_parser().parse_args(argv)
    report = load_report(args.report)
    spec = build_plot_spec(report)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.report.parent / DEFAULT_FIGURE_NAME
    paths = render_plot(spec, output_prefix, dpi=args.dpi)
    print(f"PNG: {paths[0]}")
    print(f"PDF: {paths[1]}")
    return paths


if __name__ == "__main__":
    main()
