#!/usr/bin/env python3
"""Render E1 reliability and success-conditioned quality without new rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plot_vcg_conditioned_final_comparison_90k as e1plot
import run_vcg_conditioned_final_comparison_90k as final90


DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-final-comparison-90k-cpu-v3"
PROTOCOL = "vcg_conditioned_e1_reliability_quality_addendum_v1"
REPORT_NAME = "e1-reliability-quality-report.json"
FIGURE_STEM = "e1-benchmark-reliability-quality"
TABLE_STEM = "e1-benchmark-table-inclusive"
REPRESENTATIVE_TABLE_NAME = "e1-benchmark-table-inclusive-representative.md"
REPORTING_NAME = "e1-reliability-quality-reporting.md"

METRICS = (
    "dense_return",
    "mean_absolute_error",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
    "within_target_window_rate",
)
TABLE_METRICS = (
    ("dense_return", "Return ↑"),
    ("mean_absolute_error", "MAE ↓"),
    ("steps", "Steps ↓"),
    ("physical_rehandles_per_100_required_deliveries", "$R_{100}$ ↓"),
    ("within_target_window_rate", "Within ±20 ↑"),
)
REPRESENTATIVE_LAMBDAS = frozenset((0.0, 0.05, 0.1, 0.2))
PAPER_DISPLAY_NAMES = {
    final90.KIM_METHOD: "A3C adaptation",
}


class ReliabilityQualityError(RuntimeError):
    """Raised when the closed E1 inputs or reporting invariants disagree."""


def _canonical_hash(value: Any, hash_field: str | None = None) -> str:
    payload = dict(value) if isinstance(value, Mapping) else value
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metric(row: Mapping[str, Any], name: str) -> float:
    key = "dense_objective_return" if name == "dense_return" and row.get(name) is None else name
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReliabilityQualityError(f"missing conditional metric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ReliabilityQualityError(f"nonfinite conditional metric {name}")
    return result


def _load_kim_rows(output: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    ledger_bindings = []
    expected = {
        (model_seed, instance_seed, rollout)
        for model_seed in final90.MODEL_SEEDS
        for instance_seed in final90.INSTANCE_SEEDS
        for rollout in range(5)
    }
    observed = set()
    for model_seed in final90.MODEL_SEEDS:
        directory = output / "run-ledger" / f"seed-{model_seed}"
        for path in sorted(directory.glob("instance-*-roll-*.json")):
            ledger = final90._load_json(path, label="frozen E1 Kim ledger")
            final90._verify_hash(ledger, "ledger_sha256", label="frozen E1 Kim ledger")
            row = dict(ledger["run"])
            final90._verify_hash(row, "row_sha256", label="frozen E1 Kim row")
            identity = (
                int(row["model_seed"]),
                int(row["instance_seed"]),
                int(row["rollout_index"]),
            )
            if identity in observed:
                raise ReliabilityQualityError(f"duplicate Kim identity: {identity}")
            if ledger.get("grid_row", {}).get("model_seed") != identity[0]:
                raise ReliabilityQualityError(f"Kim ledger binding changed: {path}")
            if ledger.get("grid_row", {}).get("instance_seed") != identity[1]:
                raise ReliabilityQualityError(f"Kim ledger binding changed: {path}")
            if ledger.get("grid_row", {}).get("rollout_index") != identity[2]:
                raise ReliabilityQualityError(f"Kim ledger binding changed: {path}")
            observed.add(identity)
            rows.append(row)
            ledger_bindings.append(
                {
                    "path": str(path.relative_to(output)),
                    "ledger_sha256": ledger["ledger_sha256"],
                }
            )
    if observed != expected or len(rows) != 450:
        raise ReliabilityQualityError("frozen E1 Kim grid is incomplete")
    return rows, _canonical_hash(ledger_bindings)


def _load_dynamic_rows(output: Path) -> tuple[list[dict[str, Any]], str]:
    rows = final90._load_baseline_rows(output, final90.DYNAMIC_METHOD)
    if len(rows) != 30:
        raise ReliabilityQualityError("frozen E1 Dynamic PSLAP grid is incomplete")
    bindings = []
    for instance_seed in final90.INSTANCE_SEEDS:
        path = final90.final86._ledger_path(
            output, final90.DYNAMIC_METHOD, instance_seed, None, None
        )
        ledger = final90._load_json(path, label="frozen E1 Dynamic PSLAP ledger")
        final90._verify_hash(
            ledger, "ledger_sha256", label="frozen E1 Dynamic PSLAP ledger"
        )
        bindings.append(
            {
                "path": str(path.relative_to(output)),
                "ledger_sha256": ledger["ledger_sha256"],
            }
        )
    return rows, _canonical_hash(bindings)


def _dynamic_conditional(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("strict_safe_complete") is True]
    if len(completed) != 28 or len({int(row["instance_seed"]) for row in completed}) != 28:
        raise ReliabilityQualityError("unexpected Dynamic PSLAP completion pattern")
    return {
        "strict_safe_complete_rows": len(completed),
        "expected_rows": len(rows),
        "completion_rate": len(completed) / len(rows),
        "completed_instance_count": 28,
        "conditional_hierarchy": "mean_over_28_strictly_completed_instances",
        "conditional_metrics": {
            metric: float(fmean(_metric(row, metric) for row in completed))
            for metric in METRICS
        },
    }


def _kim_conditional(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("strict_safe_complete") is True]
    if len(completed) != 448:
        raise ReliabilityQualityError("unexpected Kim2020 completion count")
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(int(row["model_seed"]), int(row["instance_seed"]))].append(row)
    expected_groups = {
        (model_seed, instance_seed)
        for model_seed in final90.MODEL_SEEDS
        for instance_seed in final90.INSTANCE_SEEDS
    }
    if set(grouped) != expected_groups or sorted(len(group) for group in grouped.values()) != [4, 4] + [5] * 88:
        raise ReliabilityQualityError("unexpected Kim2020 success-conditioned hierarchy")
    metrics = {}
    for metric in METRICS:
        instance_points = []
        for instance_seed in final90.INSTANCE_SEEDS:
            model_points = [
                fmean(_metric(row, metric) for row in grouped[(model_seed, instance_seed)])
                for model_seed in final90.MODEL_SEEDS
            ]
            instance_points.append(fmean(model_points))
        metrics[metric] = float(fmean(instance_points))
    return {
        "strict_safe_complete_rows": len(completed),
        "expected_rows": len(rows),
        "completion_rate": len(completed) / len(rows),
        "completed_instance_count": 30,
        "conditional_hierarchy": (
            "successful_rollouts_within_model_instance_then_equal_model_seed_mean_"
            "within_instance_then_mean_over_30_instances"
        ),
        "conditional_metrics": metrics,
    }


def build_report(project_root: Path, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report, conditioned_rows, provenance = e1plot._load_frozen_e1_artifacts(output)
    dynamic_rows, dynamic_set_hash = _load_dynamic_rows(output)
    kim_rows, kim_set_hash = _load_kim_rows(output)
    dynamic = _dynamic_conditional(dynamic_rows)
    kim = _kim_conditional(kim_rows)

    original = {record["method_key"]: record for record in report["methods"]}
    if original[final90.DYNAMIC_METHOD]["strict_safe_complete_rows"] != 28:
        raise ReliabilityQualityError("Dynamic count disagrees with frozen E1 report")
    if original[final90.KIM_METHOD]["strict_safe_complete_rows"] != 448:
        raise ReliabilityQualityError("Kim count disagrees with frozen E1 report")

    method_rows = []
    for record in report["methods"]:
        metrics = record["metrics"]
        basis = "all_declared_evaluations"
        if record["method"] == final90.DYNAMIC_METHOD:
            metrics = dynamic["conditional_metrics"]
            basis = "conditional_on_strict_completion"
        elif record["method"] == final90.KIM_METHOD:
            metrics = kim["conditional_metrics"]
            basis = "conditional_on_strict_completion"
        if metrics is None:
            raise ReliabilityQualityError(f"no reporting metrics for {record['display_name']}")
        method_rows.append(
            {
                "method": record["method"],
                "method_key": record["method_key"],
                "display_name": PAPER_DISPLAY_NAMES.get(
                    record["method"], record["display_name"]
                ),
                "preference_lambda": record["preference_lambda"],
                "strict_safe_complete_rows": int(record["strict_safe_complete_rows"]),
                "expected_rows": int(record["expected_rows"]),
                "completion_rate": (
                    int(record["strict_safe_complete_rows"]) / int(record["expected_rows"])
                ),
                "metric_basis": basis,
                "metrics": {metric: float(metrics[metric]) for metric in METRICS},
            }
        )

    addendum = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "analysis_type": "post_hoc_descriptive_reanalysis_of_closed_e1_rows",
        "training_runs": 0,
        "evaluation_runs": 0,
        "original_strict_gate_report_remains_immutable": True,
        "source": provenance,
        "ledger_set_sha256": {
            "dynamic_pslap": dynamic_set_hash,
            "kim2020": kim_set_hash,
        },
        "incomplete_method_summaries": {
            final90.DYNAMIC_METHOD: dynamic,
            final90.KIM_METHOD: kim,
        },
        "method_rows": method_rows,
        "reporting_rules": {
            "completion_is_primary": True,
            "incomplete_method_metrics_are_success_conditioned": True,
            "success_conditioned_metrics_are_descriptive": True,
            "success_conditioned_metrics_may_be_optimistic": True,
            "incomplete_methods_excluded_from_unconditional_pareto_and_paired_superiority": True,
            "no_failure_penalty_imputed": True,
        },
    }
    addendum["report_sha256"] = _canonical_hash(addendum, "report_sha256")
    return addendum, conditioned_rows


def _display_metric(value: float, metric: str) -> str:
    if metric == "within_target_window_rate":
        return f"{100.0 * value:.1f}%"
    return f"{value:.2f}"


def _table_rows(addendum: Mapping[str, Any], representative: bool) -> list[Mapping[str, Any]]:
    selected = []
    for row in addendum["method_rows"]:
        value = row["preference_lambda"]
        if representative and value is not None and float(value) not in REPRESENTATIVE_LAMBDAS:
            continue
        selected.append(row)
    baseline_order = {
        final90.V23_METHOD: 1,
        final90.GA_METHOD: 2,
        final90.DYNAMIC_METHOD: 3,
        final90.KIM_METHOD: 4,
    }
    return sorted(
        selected,
        key=lambda row: (
            0 if row["method"] == final90.CONDITIONED_METHOD else 1,
            float(row["preference_lambda"])
            if row["preference_lambda"] is not None
            else baseline_order.get(row["method"], 99),
        ),
    )


def render_tables(addendum: Mapping[str, Any], output: Path) -> list[str]:
    rows = _table_rows(addendum, representative=False)
    csv_path = output / f"{TABLE_STEM}.csv"
    fields = (
        "method",
        "completion",
        "completion_rate",
        "metric_basis",
        *(metric for metric, _ in TABLE_METRICS),
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["display_name"],
                    "completion": f"{row['strict_safe_complete_rows']}/{row['expected_rows']}",
                    "completion_rate": row["completion_rate"],
                    "metric_basis": row["metric_basis"],
                    **row["metrics"],
                }
            )

    def markdown(selected: Sequence[Mapping[str, Any]]) -> str:
        headers = ("Method", "Strict completion", *(label for _, label in TABLE_METRICS))
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        for row in selected:
            conditional = row["metric_basis"] == "conditional_on_strict_completion"
            suffix = "†" if conditional else ""
            values = [
                row["display_name"],
                (
                    f"{row['strict_safe_complete_rows']}/{row['expected_rows']} "
                    f"({100.0 * row['completion_rate']:.1f}%)"
                ),
            ]
            values.extend(
                _display_metric(row["metrics"][metric], metric) + suffix
                for metric, _ in TABLE_METRICS
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "† Conditional on strict completion. These descriptive summaries exclude "
                "unsuccessful evaluations and are not used for unconditional dominance, "
                "Pareto, or paired-superiority claims.",
            ]
        )
        return "\n".join(lines) + "\n"

    markdown_path = output / f"{TABLE_STEM}.md"
    markdown_path.write_text(markdown(rows), encoding="utf-8")
    representative_path = output / REPRESENTATIVE_TABLE_NAME
    representative_path.write_text(
        markdown(_table_rows(addendum, representative=True)), encoding="utf-8"
    )
    return [str(csv_path), str(markdown_path), str(representative_path)]


def render_figure(
    addendum: Mapping[str, Any], conditioned_rows: Sequence[Mapping[str, Any]], output: Path
) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = addendum["method_rows"]
    by_key = {row["method_key"]: row for row in rows}
    conditioned = sorted(
        (row for row in rows if row["method"] == final90.CONDITIONED_METHOD),
        key=lambda row: float(row["preference_lambda"]),
    )
    seed_paths = e1plot._seed_operating_paths(conditioned_rows, {
        "methods": [
            {
                "method": row["method"],
                "preference_lambda": row["preference_lambda"],
                "metrics": row["metrics"],
            }
            for row in rows
        ]
    })

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13.4, 5.7), gridspec_kw={"width_ratios": [0.82, 1.72]}
    )
    fig.subplots_adjust(left=0.095, right=0.92, bottom=0.19, top=0.82, wspace=0.34)

    family_specs = [
        ("VCG family (10 λ)", 900, 900, "#0072B2"),
        ("Historical VCG 2.3", 360, 360, e1plot.COLORS[final90.V23_METHOD]),
        ("Capacity-aware GA", 120, 120, e1plot.COLORS[final90.GA_METHOD]),
        ("Dynamic PSLAP", 28, 30, e1plot.COLORS[final90.DYNAMIC_METHOD]),
        ("A3C adaptation", 448, 450, e1plot.COLORS[final90.KIM_METHOD]),
    ]
    names = [item[0] for item in family_specs]
    rates = [100.0 * item[1] / item[2] for item in family_specs]
    colors = [item[3] for item in family_specs]
    ypos = list(range(len(names)))
    bars = ax0.barh(ypos, rates, color=colors, height=0.56, alpha=0.92)
    for bar, (_, completed, expected, _) in zip(bars, family_specs):
        rate = 100.0 * completed / expected
        ax0.text(
            min(rate - 1.0, 98.7),
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%  ({completed}/{expected})",
            ha="right",
            va="center",
            fontsize=9,
            color="white" if rate < 99.9 else "#1F2937",
            fontweight="bold",
        )
    ax0.set_yticks(ypos, names)
    ax0.invert_yaxis()
    ax0.set_xlim(0, 103)
    ax0.set_xlabel("Strict-completion rate (%)")
    ax0.set_title("(a) Reliability on the closed panel", loc="left")
    ax0.grid(axis="x", color="#D1D5DB", alpha=0.65, linewidth=0.8)
    ax0.set_axisbelow(True)

    lambdas = [float(row["preference_lambda"]) for row in conditioned]
    normalizer = Normalize(vmin=min(lambdas), vmax=max(lambdas))
    for model_seed in seed_paths["model_seeds"]:
        path = seed_paths["paths"][model_seed]
        color, linestyle = e1plot.SEED_STYLES[int(model_seed)]
        ax1.plot(
            [point["physical_rehandles_per_100_required_deliveries"] for point in path],
            [point["mean_absolute_error"] for point in path],
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            alpha=0.65,
            zorder=1,
        )
        ax1.scatter(
            [point["physical_rehandles_per_100_required_deliveries"] for point in path],
            [point["mean_absolute_error"] for point in path],
            c=lambdas,
            cmap="viridis",
            norm=normalizer,
            s=24,
            alpha=0.35,
            edgecolor="none",
            zorder=2,
        )
    x = [row["metrics"]["physical_rehandles_per_100_required_deliveries"] for row in conditioned]
    y = [row["metrics"]["mean_absolute_error"] for row in conditioned]
    ax1.plot(x, y, color="#374151", linewidth=2.6, zorder=3)
    vcg_scatter = ax1.scatter(
        x,
        y,
        c=lambdas,
        cmap="viridis",
        norm=normalizer,
        s=82,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    for row in conditioned:
        value = float(row["preference_lambda"])
        if value in e1plot.SELECTED_LAMBDA_LABELS:
            px = row["metrics"]["physical_rehandles_per_100_required_deliveries"]
            py = row["metrics"]["mean_absolute_error"]
            ax1.annotate(
                f"λ={value:g}",
                (px, py),
                xytext=(5, 8 if value in (0.0, 0.2) else -13),
                textcoords="offset points",
                fontsize=8.7,
                color="#374151",
            )

    baseline_specs = (
        (final90.V23_METHOD, e1plot.MARKERS[final90.V23_METHOD], True),
        (final90.GA_METHOD, e1plot.MARKERS[final90.GA_METHOD], True),
        (final90.DYNAMIC_METHOD, e1plot.MARKERS[final90.DYNAMIC_METHOD], False),
        (final90.KIM_METHOD, e1plot.MARKERS[final90.KIM_METHOD], False),
    )
    legend_handles = [Line2D([0], [0], color="#374151", marker="o", linewidth=2.5, label="VCG mean (3 model seeds)")]
    for method, marker, complete in baseline_specs:
        row = by_key[method]
        color = e1plot.COLORS[method]
        px = row["metrics"]["physical_rehandles_per_100_required_deliveries"]
        py = row["metrics"]["mean_absolute_error"]
        ax1.scatter(
            [px],
            [py],
            marker=marker,
            s=115,
            facecolor=color if complete else "white",
            edgecolor=color,
            linewidth=2.0 if not complete else 0.8,
            zorder=6,
        )
        label = row["display_name"] + ("†" if not complete else "")
        legend_handles.append(
            Line2D(
                [0], [0], marker=marker, linestyle="None", markersize=8,
                markerfacecolor=color if complete else "white",
                markeredgecolor=color, markeredgewidth=1.6, label=label,
            )
        )
        if not complete:
            rate = 100.0 * row["completion_rate"]
            if method == final90.DYNAMIC_METHOD:
                offset = (-8, 9)
                horizontal_alignment = "right"
            else:
                offset = (8, -15)
                horizontal_alignment = "left"
            ax1.annotate(
                f"{rate:.1f}% complete",
                (px, py),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal_alignment,
                fontsize=8.8,
                color=color,
                fontweight="bold",
            )
    for model_seed in seed_paths["model_seeds"]:
        color, linestyle = e1plot.SEED_STYLES[int(model_seed)]
        legend_handles.append(
            Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.4, label=f"VCG model seed {model_seed}")
        )
    ax1.set_xlim(-1.0, 35.0)
    ax1.set_ylim(11.4, 22.8)
    ax1.set_xlabel("Physical rehandles / 100 required deliveries ↓")
    ax1.set_ylabel("Mean absolute delivery error ↓")
    ax1.set_title("(b) Operating quality", loc="left")
    ax1.grid(color="#D1D5DB", alpha=0.6, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(handles=legend_handles, loc="upper right", frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.45)
    colorbar = fig.colorbar(vcg_scatter, ax=ax1, pad=0.018, fraction=0.045)
    colorbar.set_label("Handling preference λ")

    fig.suptitle("Benchmark reliability and preference-controlled operating quality", fontsize=15, y=0.96)
    fig.text(
        0.5,
        0.035,
        "† Hollow markers are strict-completion-conditioned summaries and may be optimistic; "
        "they are excluded from unconditional Pareto and paired-superiority claims. "
        "VCG lines connect λ values, not time.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4B5563",
    )
    paths = []
    for suffix in ("pdf", "png"):
        path = output / f"{FIGURE_STEM}.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def render_reporting(addendum: Mapping[str, Any], output: Path) -> str:
    dynamic = addendum["incomplete_method_summaries"][final90.DYNAMIC_METHOD]
    kim = addendum["incomplete_method_summaries"][final90.KIM_METHOD]
    dynamic_metrics = dynamic["conditional_metrics"]
    kim_metrics = kim["conditional_metrics"]
    text = f"""# E1 reliability--quality reporting addendum

The original authenticated strict-gate report is unchanged. This post-hoc
descriptive addendum launches no training or evaluation runs.

## Ready-to-paste manuscript passage

All 900 final VCG evaluations completed strictly, covering 90 model--instance
rows at each of ten deployment preferences. Historical VCG 2.3 completed
360/360 rows and the capacity-aware GA completed 120/120. Dynamic PSLAP
completed {dynamic['strict_safe_complete_rows']}/{dynamic['expected_rows']}
({100.0 * dynamic['completion_rate']:.1f}%), while the A3C adaptation
completed {kim['strict_safe_complete_rows']}/{kim['expected_rows']}
({100.0 * kim['completion_rate']:.1f}%). The denominators follow the methods'
predeclared evaluation grids and therefore differ. Figure 5 separates this
reliability result from operating quality. Dynamic PSLAP's strictly completed
runs had a success-conditioned MAE of
{dynamic_metrics['mean_absolute_error']:.2f} and
{dynamic_metrics['physical_rehandles_per_100_required_deliveries']:.2f}
physical rehandles per 100 required deliveries; the corresponding A3C
values were {kim_metrics['mean_absolute_error']:.2f} and
{kim_metrics['physical_rehandles_per_100_required_deliveries']:.2f}. These
values describe service quality when the implementations complete; they do not
erase the unsuccessful evaluations. Because conditioning omits failures, the
two hollow operating points may be optimistic and are excluded from
unconditional dominance, Pareto, and paired-superiority claims.

Among methods satisfying universal strict completion, VCG at $\\lambda=0$
provided the lowest displayed MAE (12.38), while increasing the handling
preference moved the same frozen controller toward lower-handling operating
points. At $\\lambda=.20$, VCG required 1.67 physical rehandles per 100
required deliveries and 152.09 primitive steps per episode, compared with
6.25 and 230.96 for historical VCG 2.3 and 13.75 and 167.77 for the
capacity-aware GA. The comparison therefore supports a preference-controlled
timing--handling family, while treating reliability as a separate outcome
rather than converting an incomplete baseline into a single penalized score.

## Figure caption

Benchmark reliability and preference-controlled operating quality on the closed
90k panel. Panel (a) reports strict-completion rates over every declared
evaluation. Panel (b) reports MAE against physical rehandles per 100 required
deliveries. Filled baseline markers have universal strict completion; hollow
markers are conditional on strict completion, with completion rates annotated.
Conditional points exclude unsuccessful evaluations and are not included in
unconditional dominance or Pareto claims. Light VCG paths are model-seed means;
the bold path is the three-seed mean. Lines follow preference order rather than
time. Evaluation denominators follow the predeclared grid for each method.

## Table note

† Conditional on strict completion. These summaries omit unsuccessful
evaluations and may therefore be optimistic. They are shown descriptively and
are not used for unconditional dominance, Pareto, or paired-superiority claims.
"""
    path = output / REPORTING_NAME
    path.write_text(text, encoding="utf-8")
    return str(path)


def render(project_root: Path, output: Path) -> dict[str, Any]:
    addendum, conditioned_rows = build_report(project_root, output)
    report_path = output / REPORT_NAME
    report_path.write_text(json.dumps(addendum, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures = render_figure(addendum, conditioned_rows, output)
    tables = render_tables(addendum, output)
    reporting = render_reporting(addendum, output)
    return {
        "status": "complete",
        "report": str(report_path),
        "figure_pdf": figures[0],
        "figure_png": figures[1],
        "tables": tables,
        "reporting": reporting,
        "training_runs": 0,
        "evaluation_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
