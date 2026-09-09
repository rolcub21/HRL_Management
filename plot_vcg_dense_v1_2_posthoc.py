#!/usr/bin/env python3
"""Plot the VCG-Dense V1.2 posthoc development candidate.

Coordinates come only from the frozen posthoc report and its authenticated
source Pareto report.  The candidate and episode-500 policies are deliberately
styled as diagnostic artifacts, never as deployment checkpoints.
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

from plot_vcg_dense_pareto import (
    ENHANCED_GA_COLOR,
    FINAL_COLOR,
    NEAREST_COLOR,
    SELECTED_COLOR,
    ParetoPoint,
    build_plot_spec as build_source_plot_spec,
    load_report as load_source_report,
)


POSTHOC_PROTOCOL = "vcg_dense_v1_2_posthoc_relocation_selection_development_v1"
DEFAULT_POSTHOC_REPORT = Path(
    "results/vcg-dense-v1-2-posthoc-development-30seed/posthoc-report.json"
)
DEFAULT_SOURCE_REPORT = Path(
    "results/vcg-dense-v1-1-pareto-development-30seed/pareto-report.json"
)
DEFAULT_FIGURE_NAME = "vcg-dense-v1-2-posthoc-pareto"
ENHANCED_ID = "duration_aware_enhanced_complete_rolling_ga"
NEAREST_ID = "duration_aware_nearest_free"

CANDIDATE_COLOR = "#AA3377"
CANDIDATE_LIGHT = "#E8B5D2"
SHIFT_COLOR = "#7A285F"
SEED_SHIFT_COLOR = "#A36A91"

INTERVAL_KEYS = (
    "conditional_30_instance_bootstrap_95_ci",
    "crossed_seed_instance_bootstrap_sensitivity_95_ci",
    "training_seed_t95_ci",
)


@dataclass(frozen=True)
class CandidateShift:
    model_seed: int
    chosen_episode: int
    selected: ParetoPoint
    candidate: ParetoPoint


@dataclass(frozen=True)
class PosthocPlotSpec:
    selected: ParetoPoint
    candidate: ParetoPoint
    episode500: ParetoPoint
    enhanced_ga: ParetoPoint
    nearest_free: ParetoPoint
    seed_shifts: tuple[CandidateShift, ...]
    instance_count: int
    candidate_mae_cost_vs_enhanced: float
    candidate_relocation_saving_vs_enhanced: float
    mae_intervals_cross_zero: bool
    relocation_intervals_cross_zero: bool


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


def load_posthoc_report(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("posthoc report must contain a JSON object")
    return payload


def _summary_point(
    point_id: str,
    label: str,
    summary: Mapping,
    *,
    group: str,
    diagnostic: bool,
    seed: Optional[int] = None,
) -> ParetoPoint:
    if int(summary.get("episodes", 0)) != 30:
        raise ValueError(f"{point_id} must contain 30 episodes")
    if (
        summary.get("strict_method_success_rate") != 1
        or summary.get("completion_rate") != 1
    ):
        raise ValueError(f"{point_id} is not strictly complete")
    return ParetoPoint(
        point_id=point_id,
        label=label,
        mae=_finite(summary.get("mean_absolute_error"), f"{point_id}.MAE"),
        relocations_per_100=_finite(
            summary.get("relocations_per_100_deliveries"),
            f"{point_id}.relocations_per_100",
        ),
        policy_group=group,
        diagnostic_only=diagnostic,
        model_seed=seed,
    )


def _assert_same_point(left: ParetoPoint, right: ParetoPoint, name: str) -> None:
    if not (
        _close(left.mae, right.mae)
        and _close(left.relocations_per_100, right.relocations_per_100)
    ):
        raise ValueError(f"{name} disagrees with the source Pareto report")


def _all_intervals_cross_zero(metric: Mapping, name: str) -> bool:
    results = []
    for key in INTERVAL_KEYS:
        interval = metric.get(key)
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{name}.{key} must be a two-value interval")
        low = _finite(interval[0], f"{name}.{key}.low")
        high = _finite(interval[1], f"{name}.{key}.high")
        if low > high:
            raise ValueError(f"{name}.{key} is reversed")
        results.append(low <= 0.0 <= high)
    return all(results)


def build_posthoc_plot_spec(
    posthoc: Mapping,
    source: Mapping,
    *,
    source_report_path: Optional[Path] = None,
) -> PosthocPlotSpec:
    """Fail closed unless report provenance and point relations are coherent."""

    if posthoc.get("protocol") != POSTHOC_PROTOCOL:
        raise ValueError(f"unexpected posthoc protocol: {posthoc.get('protocol')!r}")
    if posthoc.get("scope") != "posthoc_development_diagnostic_only":
        raise ValueError("posthoc scope is not development-diagnostic only")
    if posthoc.get("performance_claim_authorized") is not False:
        raise ValueError("posthoc report cannot authorize a performance claim")
    if posthoc.get("deployment_checkpoint_eligible") is not False:
        raise ValueError("posthoc candidate cannot be deployment eligible")
    guardrails = _mapping(posthoc.get("guardrails"), "guardrails")
    required_guards = (
        "no_new_baseline_runs",
        "posthoc_not_deployment_eligible",
        "sealed_panels_opened",
        "source_enhanced_ga_rows_reused",
        "source_selected_best_rows_reused",
    )
    for name in required_guards:
        expected = False if name == "sealed_panels_opened" else True
        if guardrails.get(name) is not expected:
            raise ValueError(f"posthoc guardrail failed: {name}")
    if int(guardrails.get("analysis_rows", 0)) != 90:
        raise ValueError("posthoc analysis must contain 90 rows")

    if source_report_path is not None:
        provenance = _mapping(posthoc.get("source_provenance"), "source_provenance")
        source_dir = Path(str(provenance.get("source_dir"))).resolve()
        if source_dir != Path(source_report_path).resolve().parent:
            raise ValueError("source Pareto directory does not match provenance")

    source_spec = build_source_plot_spec(source)
    source_points = {point.point_id: point for point in source_spec.aggregate_points}
    selected = source_spec.selected_aggregate
    episode500 = source_spec.final_aggregate
    enhanced = source_points[ENHANCED_ID]
    nearest = source_points[NEAREST_ID]

    per_seed_source = {
        shift.model_seed: shift.selected for shift in source_spec.seed_shifts
    }
    rows = posthoc.get("per_training_seed")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("posthoc report must contain exactly three training seeds")
    artifacts = posthoc.get("artifact_manifest")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValueError("posthoc artifact manifest must contain three checkpoints")
    artifact_episode = {}
    for artifact in artifacts:
        artifact = _mapping(artifact, "artifact")
        seed = int(artifact.get("model_seed"))
        if (
            artifact.get("development_only") is not True
            or artifact.get("diagnostic_only") is not True
            or artifact.get("posthoc") is not True
            or artifact.get("deployment_checkpoint_eligible") is not False
        ):
            raise ValueError(f"posthoc artifact flags are invalid for seed {seed}")
        artifact_episode[seed] = int(artifact.get("checkpoint_episode"))
    if set(artifact_episode) != {0, 1, 2}:
        raise ValueError("artifact manifest seeds are incomplete")

    shifts = []
    candidate_points = []
    for row in sorted(rows, key=lambda item: int(item["model_seed"])):
        row = _mapping(row, "per_training_seed row")
        seed = int(row.get("model_seed"))
        chosen_episode = int(row.get("chosen_episode"))
        if chosen_episode != artifact_episode.get(seed):
            raise ValueError(f"chosen episode disagrees with artifact for seed {seed}")
        selected_row = _summary_point(
            f"selected_seed{seed}",
            f"Selected s{seed}",
            _mapping(row.get("selected_best_summary"), "selected_best_summary"),
            group="selected_best",
            diagnostic=False,
            seed=seed,
        )
        _assert_same_point(selected_row, per_seed_source[seed], f"selected seed {seed}")
        enhanced_row = _summary_point(
            f"enhanced_seed{seed}",
            "Enhanced rolling GA",
            _mapping(row.get("enhanced_ga_summary"), "enhanced_ga_summary"),
            group="baseline",
            diagnostic=False,
        )
        _assert_same_point(enhanced_row, enhanced, f"enhanced GA seed {seed}")
        candidate = _summary_point(
            f"posthoc_seed{seed}_ep{chosen_episode}",
            f"Candidate s{seed}",
            _mapping(row.get("posthoc_summary"), "posthoc_summary"),
            group="posthoc_v1_2_candidate",
            diagnostic=True,
            seed=seed,
        )
        candidate_points.append(candidate)
        shifts.append(CandidateShift(seed, chosen_episode, selected_row, candidate))

    across = _mapping(posthoc.get("across_training_seeds"), "across_training_seeds")
    candidate = ParetoPoint(
        point_id="posthoc_v1_2_candidate",
        label="Posthoc V1.2 candidate (3-seed mean)",
        mae=_finite(
            _mapping(across.get("mean_absolute_error"), "across.MAE").get("mean"),
            "candidate aggregate MAE",
        ),
        relocations_per_100=_finite(
            _mapping(
                across.get("relocations_per_100_deliveries"),
                "across.relocations",
            ).get("mean"),
            "candidate aggregate relocations",
        ),
        policy_group="posthoc_v1_2_candidate",
        diagnostic_only=True,
    )
    if not _close(candidate.mae, sum(point.mae for point in candidate_points) / 3):
        raise ValueError("candidate aggregate MAE is not equal-seed weighted")
    if not _close(
        candidate.relocations_per_100,
        sum(point.relocations_per_100 for point in candidate_points) / 3,
    ):
        raise ValueError("candidate aggregate relocations are not equal-seed weighted")

    contrasts = _mapping(posthoc.get("aggregate_contrasts"), "aggregate_contrasts")
    vs_enhanced = _mapping(
        contrasts.get("candidate_vs_enhanced_ga"), "candidate_vs_enhanced_ga"
    )
    metrics = _mapping(vs_enhanced.get("metrics"), "candidate_vs_enhanced.metrics")
    mae_metric = _mapping(metrics.get("mae_cost"), "candidate_vs_enhanced.mae_cost")
    relocation_metric = _mapping(
        metrics.get("relocations_saving_per_100_deliveries"),
        "candidate_vs_enhanced.relocations",
    )
    mae_cost = _finite(mae_metric.get("mean_across_training_seeds"), "MAE cost")
    relocation_saving = _finite(
        relocation_metric.get("mean_across_training_seeds"), "relocation saving"
    )
    if not _close(candidate.mae - enhanced.mae, mae_cost):
        raise ValueError("candidate/enhanced MAE contrast is inconsistent")
    if not _close(
        enhanced.relocations_per_100 - candidate.relocations_per_100,
        relocation_saving,
    ):
        raise ValueError("candidate/enhanced relocation contrast is inconsistent")
    if not (mae_cost < 0.0 and relocation_saving > 0.0):
        raise ValueError("candidate does not point-dominate enhanced GA")

    mae_crosses = _all_intervals_cross_zero(mae_metric, "mae_cost")
    relocation_crosses = _all_intervals_cross_zero(
        relocation_metric, "relocation_saving"
    )
    if not (mae_crosses and relocation_crosses):
        raise ValueError("uncertainty annotation is not supported by the report")

    vs_selected = _mapping(
        contrasts.get("candidate_vs_selected_best"), "candidate_vs_selected_best"
    )
    selected_metrics = _mapping(vs_selected.get("metrics"), "selected metrics")
    selected_mae_cost = _finite(
        _mapping(selected_metrics.get("mae_cost"), "selected.mae_cost").get(
            "mean_across_training_seeds"
        ),
        "selected MAE cost",
    )
    selected_relocation_saving = _finite(
        _mapping(
            selected_metrics.get("relocations_saving_per_100_deliveries"),
            "selected.relocations",
        ).get("mean_across_training_seeds"),
        "selected relocation saving",
    )
    if not _close(candidate.mae - selected.mae, selected_mae_cost):
        raise ValueError("candidate/selected MAE contrast is inconsistent")
    if not _close(
        selected.relocations_per_100 - candidate.relocations_per_100,
        selected_relocation_saving,
    ):
        raise ValueError("candidate/selected relocation contrast is inconsistent")

    return PosthocPlotSpec(
        selected=selected,
        candidate=candidate,
        episode500=episode500,
        enhanced_ga=enhanced,
        nearest_free=nearest,
        seed_shifts=tuple(shifts),
        instance_count=30,
        candidate_mae_cost_vs_enhanced=mae_cost,
        candidate_relocation_saving_vs_enhanced=relocation_saving,
        mae_intervals_cross_zero=mae_crosses,
        relocation_intervals_cross_zero=relocation_crosses,
    )


def _xy(point: ParetoPoint) -> tuple[float, float]:
    return point.mae, point.relocations_per_100


def _draw_seed_shift(ax, shift: CandidateShift, *, labels: bool) -> None:
    start, end = _xy(shift.selected), _xy(shift.candidate)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": SEED_SHIFT_COLOR,
            "linestyle": (0, (3, 2)),
            "linewidth": 1.25,
            "mutation_scale": 9,
        },
        zorder=2,
    )
    ax.scatter(
        *start,
        s=38,
        marker="o",
        facecolor=SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        *end,
        s=53,
        marker="P",
        facecolor=CANDIDATE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    if labels:
        offsets = {0: (-28, -16), 1: (-24, 8), 2: (-48, 10)}
        ax.annotate(
            f"s{shift.model_seed} → ep{shift.chosen_episode}",
            start,
            xytext=offsets[shift.model_seed],
            textcoords="offset points",
            fontsize=7.4,
            color=SELECTED_COLOR,
            fontweight="bold",
            zorder=8,
        )


def _draw_aggregate_shift(ax, spec: PosthocPlotSpec, linewidth: float) -> None:
    ax.annotate(
        "",
        xy=_xy(spec.candidate),
        xytext=_xy(spec.selected),
        arrowprops={
            "arrowstyle": "-|>",
            "color": SHIFT_COLOR,
            "linewidth": linewidth,
            "mutation_scale": 13,
            "shrinkA": 7,
            "shrinkB": 8,
        },
        zorder=6,
    )


def _draw_aggregate_points(ax, spec: PosthocPlotSpec, *, annotate: bool) -> None:
    styles = {
        "selected": (spec.selected, "*", SELECTED_COLOR, "black", 220, 0.8),
        "candidate": (spec.candidate, "P", CANDIDATE_COLOR, "black", 150, 0.8),
        "episode500": (spec.episode500, "D", "white", FINAL_COLOR, 118, 2.0),
        "enhanced": (spec.enhanced_ga, "s", ENHANCED_GA_COLOR, "black", 102, 0.7),
        "nearest": (spec.nearest_free, "X", NEAREST_COLOR, "black", 102, 0.7),
    }
    for _, (point, marker, face, edge, size, width) in styles.items():
        ax.scatter(
            *_xy(point),
            marker=marker,
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=width,
            zorder=9,
        )
    if not annotate:
        return
    labels = {
        "selected": ("Selected VCG", -5, 15, "left", "#202020"),
        "candidate": (
            "Posthoc V1.2 candidate\n(development only)",
            55,
            -5,
            "left",
            CANDIDATE_COLOR,
        ),
        "episode500": (
            "Episode-500 diagnostic",
            13,
            16,
            "left",
            FINAL_COLOR,
        ),
        "enhanced": ("Enhanced rolling GA", 11, 13, "left", "#202020"),
        "nearest": ("Nearest-free", -8, 15, "right", "#202020"),
    }
    for key, (text, dx, dy, align, color) in labels.items():
        point = styles[key][0]
        ax.annotate(
            text,
            _xy(point),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=8.4,
            color=color,
            fontweight="bold" if key in ("selected", "candidate", "episode500") else "normal",
            zorder=10,
        )


def _draw_inset(ax, spec: PosthocPlotSpec) -> None:
    inset = ax.inset_axes([0.51, 0.39, 0.46, 0.53])
    for shift in spec.seed_shifts:
        _draw_seed_shift(inset, shift, labels=True)
    _draw_aggregate_shift(inset, spec, 2.1)
    _draw_aggregate_points(inset, spec, annotate=False)
    inset.annotate(
        "",
        xy=_xy(spec.candidate),
        xytext=_xy(spec.enhanced_ga),
        arrowprops={
            "arrowstyle": "->",
            "color": ENHANCED_GA_COLOR,
            "linestyle": ":",
            "linewidth": 1.3,
        },
        zorder=3,
    )
    midpoint = (
        (spec.candidate.mae + spec.enhanced_ga.mae) / 2,
        (spec.candidate.relocations_per_100 + spec.enhanced_ga.relocations_per_100)
        / 2,
    )
    inset.annotate(
        "point-estimate\nlower-left",
        midpoint,
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color="#176B52",
    )
    detail = [spec.selected, spec.candidate, spec.episode500, spec.enhanced_ga]
    detail.extend(
        point
        for shift in spec.seed_shifts
        for point in (shift.selected, shift.candidate)
    )
    x_values = [point.mae for point in detail]
    y_values = [point.relocations_per_100 for point in detail]
    inset.set_xlim(min(x_values) - 0.45, max(x_values) + 0.55)
    inset.set_ylim(max(0, min(y_values) - 2.0), max(y_values) + 2.2)
    inset.set_title(
        "Selected → posthoc candidate by training seed",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=5,
    )
    inset.set_xlabel("MAE", fontsize=7.5)
    inset.set_ylabel("Relocations / 100", fontsize=7.5)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)


def render_posthoc_plot(
    spec: PosthocPlotSpec,
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
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.215)
    for shift in spec.seed_shifts:
        _draw_seed_shift(ax, shift, labels=False)
    _draw_aggregate_shift(ax, spec, 2.3)
    _draw_aggregate_points(ax, spec, annotate=True)

    points = [
        spec.selected,
        spec.candidate,
        spec.episode500,
        spec.enhanced_ga,
        spec.nearest_free,
    ]
    points.extend(
        point
        for shift in spec.seed_shifts
        for point in (shift.selected, shift.candidate)
    )
    x_values = [point.mae for point in points]
    y_values = [point.relocations_per_100 for point in points]
    ax.set_xlim(max(0, min(x_values) - 2.5), max(x_values) + 3.0)
    ax.set_ylim(-1.8, max(y_values) + 3.2)
    ax.set_xlabel("Mean absolute timing error (steps)")
    ax.set_ylabel("Relocations per 100 deliveries")
    ax.set_title(
        "Posthoc V1.2 candidate timing–relocation diagnostic",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.0,
        1.012,
        (
            f"{spec.instance_count} paired instances · equal-weight 3-seed aggregates"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#4A4A4A",
    )
    ax.text(
        0.995,
        1.012,
        "DEVELOPMENT ONLY · POSTHOC · NOT DEPLOYMENT ELIGIBLE",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#7A1F5C",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#FFF0F8",
            "edgecolor": CANDIDATE_LIGHT,
            "linewidth": 0.8,
        },
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

    handles = [
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
            marker="P",
            linestyle="none",
            markersize=8,
            markerfacecolor=CANDIDATE_COLOR,
            markeredgecolor="black",
            label="Posthoc V1.2 candidate (diagnostic)",
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
            color=SEED_SHIFT_COLOR,
            linestyle=(0, (3, 2)),
            marker="o",
            markersize=4,
            markerfacecolor=SELECTED_COLOR,
            label="Per-seed selected → candidate",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.70, 0.115),
        frameon=True,
        framealpha=0.96,
        edgecolor="#CCCCCC",
    )

    mae_gain = -spec.candidate_mae_cost_vs_enhanced
    relocation_gain = spec.candidate_relocation_saving_vs_enhanced
    fig.text(
        0.105,
        0.092,
        (
            "Point estimates: candidate is lower-left of enhanced GA "
            f"(MAE −{mae_gain:.2f}; relocations −{relocation_gain:.2f}/100). "
            "All reported 95% intervals for both contrasts cross zero."
        ),
        ha="left",
        va="center",
        fontsize=8.35,
        color="#6A2050",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF3F9",
            "edgecolor": CANDIDATE_LIGHT,
            "linewidth": 0.8,
        },
    )
    fig.text(
        0.105,
        0.035,
        (
            "The lower-left relation is descriptive development evidence, not "
            "a confirmatory dominance or deployment claim."
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
            "Title": "VCG-Dense V1.2 posthoc timing-relocation diagnostic",
            "Description": "Development-only posthoc diagnostic",
        },
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        metadata={
            "Title": "VCG-Dense V1.2 posthoc timing-relocation diagnostic",
            "Subject": "Development-only posthoc timing-relocation diagnostic",
            "Keywords": "VCG Dense, posthoc, Pareto, development diagnostic",
        },
    )
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the frozen VCG-Dense V1.2 posthoc report"
    )
    parser.add_argument(
        "--posthoc-report", type=Path, default=DEFAULT_POSTHOC_REPORT
    )
    parser.add_argument(
        "--source-pareto-report", type=Path, default=DEFAULT_SOURCE_REPORT
    )
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> tuple[Path, Path]:
    args = build_parser().parse_args(argv)
    posthoc = load_posthoc_report(args.posthoc_report)
    source = load_source_report(args.source_pareto_report)
    spec = build_posthoc_plot_spec(
        posthoc,
        source,
        source_report_path=args.source_pareto_report,
    )
    prefix = args.output_prefix
    if prefix is None:
        prefix = args.posthoc_report.parent / DEFAULT_FIGURE_NAME
    paths = render_posthoc_plot(spec, prefix, dpi=args.dpi)
    print(f"PNG: {paths[0]}")
    print(f"PDF: {paths[1]}")
    return paths


if __name__ == "__main__":
    main()
