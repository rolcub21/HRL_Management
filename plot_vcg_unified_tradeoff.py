#!/usr/bin/env python3
"""Plot the VCG timing--handling trade-off without pooling evaluation panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vcg-matplotlib-cache"))

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SWEEP_85 = ROOT / "results/vcg-unified-frozen-lambda-sweep-seed14-85k-development/frozen-lambda-comparison.json"
BASELINES_85 = ROOT / "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k/expanded-report.json"
KIM_85 = ROOT / "results/vcg-v2-3-kim2020-supplement-evaluation-85k/extended-report.json"
BASELINES_86 = ROOT / "results/vcg-final86-baseline-correction/corrected-final-report.json"
KIM_86 = ROOT / "results/kim2020-final86-comparison/comparison-report.json"
CONFIRMATION_87 = ROOT / "results/vcg-unified-frozen-lambda-confirmation-87k/confirmation-report.json"

METHOD_STYLE = {
    "vcg_1_1": ("VCG 1.1 (legacy)", "s", "#6b7280"),
    "vcg_2_3": ("VCG 2.3 (precursor)", "D", "#2563a6"),
    "duration_aware_dynamic_pslap": ("Dynamic PSLAP", "^", "#dd8a19"),
    "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial": (
        "Capacity-aware GA",
        "P",
        "#2f855a",
    ),
}


class TradeoffPlotError(ValueError):
    pass


def _load(path: Path) -> Mapping:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TradeoffPlotError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(metrics: Mapping, *, final: bool) -> tuple[float, float]:
    x_key = (
        "physical_rehandles_per_100_required_deliveries"
        if final
        else "physical_rehandles_per_100_required_deliveries"
    )
    x = metrics.get(x_key)
    y = metrics.get("mean_absolute_error")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TradeoffPlotError("an eligible plotted method is missing MAE or rehandles")
    return float(x), float(y)


def _decorate_matched_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(10.0, 25.0)
    ax.set_xlabel("Physical rehandles / 100 deliveries  →")
    ax.grid(True, color="#d9dde2", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.04, "better  ↙", transform=ax.transAxes, color="#667085", fontsize=9)


def _plot_baseline(ax, method_id: str, x: float, y: float, *, offset=(6, 6)) -> None:
    label, marker, color = METHOD_STYLE[method_id]
    ax.scatter(x, y, s=88, marker=marker, color=color, edgecolor="white", linewidth=1.2, zorder=4)
    ax.annotate(
        label,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.5,
        color="#30343b",
    )


def build_figure(output_dir: Path) -> dict:
    sweep = _load(SWEEP_85)
    baselines85 = _load(BASELINES_85)
    kim85 = _load(KIM_85)
    baselines86 = _load(BASELINES_86)["corrected_final_report"]
    kim86 = _load(KIM_86)
    confirmation = _load(CONFIRMATION_87)

    if sweep.get("status") != "complete" or sweep.get("same_frozen_weights_for_all_rows") is not True:
        raise TradeoffPlotError("85k frozen-lambda sweep is not complete and common-weight")
    if confirmation.get("status") != "passed" or confirmation.get("all_720_rows_strict_safe_complete") is not True:
        raise TradeoffPlotError("87k confirmation is not complete and safe")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#30343b",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.2),
        gridspec_kw={"width_ratios": [1.25, 1.05, 0.9]},
    )
    figure.patch.set_facecolor("#fbfaf7")
    for ax in axes:
        ax.set_facecolor("#fffefd")

    # 85k: the only legitimate connected curve in the current artifacts.
    ax = axes[0]
    _decorate_matched_axis(ax, "A  ·  85k development — frozen-weight λ sensitivity")
    ax.set_ylabel("Mean absolute timing error (steps)  →")
    ordered = sorted(sweep["table"], key=lambda row: float(row["fixed_lambda"]))
    xs = [float(row["physical_rehandles_per_100"]) for row in ordered]
    ys = [float(row["mean_absolute_error"]) for row in ordered]
    ax.plot(xs, ys, color="#a13b79", linewidth=2.2, alpha=0.9, zorder=2)
    ax.scatter(xs, ys, s=66, color="#c53e7e", edgecolor="white", linewidth=1.2, zorder=3)
    lambda_offsets = {
        0.0: (8, -13),
        0.05: (8, -13),
        0.1: (15, -5),
        0.2: (8, 9),
        0.3: (9, 2),
    }
    for row, x, y in zip(ordered, xs, ys):
        value = float(row["fixed_lambda"])
        ax.annotate(
            f"λ={value:g}",
            (x, y),
            xytext=lambda_offsets[value],
            textcoords="offset points",
            fontsize=8,
            color="#8d2f69",
        )

    methods85 = {row["method_id"]: row for row in baselines85["method_summaries"]}
    offsets85 = {
        "vcg_1_1": (-90, 8),
        "vcg_2_3": (7, -13),
        "duration_aware_dynamic_pslap": (7, 6),
        "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial": (-102, -14),
    }
    methods85_to_style = {
        "vcg_dense_v1_1_selected_three_seed": "vcg_1_1",
        "vcg_constrained_v2_3_gamma1_episode160": "vcg_2_3",
        "duration_aware_dynamic_pslap": "duration_aware_dynamic_pslap",
        "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial": (
            "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
        ),
    }
    for method_id, style_id in methods85_to_style.items():
        metrics = methods85[method_id].get("metrics")
        if not isinstance(metrics, Mapping):
            raise TradeoffPlotError(f"85k method {method_id} is not numerically eligible")
        x, y = _point(metrics, final=False)
        _plot_baseline(ax, style_id, x, y, offset=offsets85[style_id])
    kim85_summary = kim85["kim_summary"]
    ax.text(
        0.02,
        -0.20,
        f"Kim not plotted: {kim85_summary['strict_safe_complete_row_count']}/{kim85_summary['expected_row_count']} safe; whole-method metrics suppressed.",
        transform=ax.transAxes,
        fontsize=8,
        color="#6b7280",
    )

    # 86k: baseline-only final panel. Unrelated methods are never connected.
    ax = axes[1]
    _decorate_matched_axis(ax, "B  ·  86k matched baselines — points only")
    methods86 = {row["method"]: row for row in baselines86["methods"]}
    offsets86 = {
        "vcg_1_1": (-89, 7),
        "vcg_2_3": (7, 7),
        "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial": (7, -14),
    }
    for method_id in (
        "vcg_1_1",
        "vcg_2_3",
        "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial",
    ):
        row = methods86[method_id]
        if row.get("whole_method_eligible") is not True:
            raise TradeoffPlotError(f"86k method {method_id} is not eligible")
        x, y = _point(row["metrics"], final=True)
        _plot_baseline(ax, method_id, x, y, offset=offsets86[method_id])
    dynamic = methods86["duration_aware_dynamic_pslap"]
    kim_method = next(row for row in kim86["methods"] if row["method"] == "kim2020_a3c_spatial_adapted__stochastic")
    ax.text(
        0.02,
        -0.20,
        "Not plotted (whole-method gate): "
        f"Dynamic PSLAP {dynamic['strict_safe_complete_rows']}/{dynamic['expected_rows']}; "
        f"Kim {kim_method['strict_safe_complete_rows']}/{kim_method['expected_rows']}.",
        transform=ax.transAxes,
        fontsize=8,
        color="#6b7280",
    )

    # 87k: confirmed selected operating shift. Two points are an arrow, not a curve.
    ax = axes[2]
    ax.set_title("C  ·  87k confirmation — selected shift (zoom)", loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(7.8, 9.5)
    ax.set_ylim(19.35, 20.12)
    ax.set_xlabel("Physical rehandles / 100 deliveries  →")
    ax.grid(True, color="#d9dde2", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    c0 = confirmation["aggregate"]["lambda0"]
    c5 = confirmation["aggregate"]["lambda005"]
    x0, y0 = float(c0["physical_rehandles_per_100"]), float(c0["mean_absolute_error"])
    x5, y5 = float(c5["physical_rehandles_per_100"]), float(c5["mean_absolute_error"])
    ax.annotate(
        "",
        xy=(x5, y5),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "color": "#9d376f", "lw": 2.5, "mutation_scale": 16},
        zorder=2,
    )
    ax.scatter(x0, y0, s=115, color="#2968b9", edgecolor="white", linewidth=1.4, zorder=3)
    ax.scatter(x5, y5, s=115, color="#c53e7e", edgecolor="white", linewidth=1.4, zorder=3)
    ax.annotate("λ = 0", (x0, y0), xytext=(8, -16), textcoords="offset points", color="#2968b9", fontweight="bold")
    ax.annotate("λ = 0.05", (x5, y5), xytext=(-8, 10), textcoords="offset points", ha="right", color="#a83269", fontweight="bold")
    delta = confirmation["aggregate"]["paired_difference_lambda005_minus_lambda0"]
    rh = delta["physical_rehandles_per_100"]
    mae = delta["mean_absolute_error"]
    ax.text(
        0.04,
        0.05,
        f"Δ rehandles = {rh['mean_difference']:+.2f}  [{rh['nominal_two_sided_95_ci'][0]:+.2f}, {rh['nominal_two_sided_95_ci'][1]:+.2f}]\n"
        f"Δ MAE = {mae['mean_difference']:+.2f}  [{mae['nominal_two_sided_95_ci'][0]:+.2f}, {mae['nominal_two_sided_95_ci'][1]:+.2f}]",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#3f4650",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f3f5", "edgecolor": "#ded4da"},
    )

    figure.suptitle(
        "VCG operational timing–handling trade-off",
        x=0.055,
        y=1.02,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#252a31",
    )
    figure.text(
        0.055,
        0.955,
        "Primary coordinates: MAE and total physical rehandles. Lower-left is better; panels are not pooled.",
        ha="left",
        fontsize=9.5,
        color="#626a73",
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.22, wspace=0.30)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "vcg-unified-tradeoff.png"
    pdf = output_dir / "vcg-unified-tradeoff.pdf"
    figure.savefig(png, dpi=220, facecolor=figure.get_facecolor(), bbox_inches="tight")
    figure.savefig(pdf, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)

    sources = [SWEEP_85, BASELINES_85, KIM_85, BASELINES_86, KIM_86, CONFIRMATION_87]
    manifest = {
        "protocol": "vcg_unified_tradeoff_plot_v1",
        "primary_coordinates": ["mean_absolute_error", "physical_rehandles_per_100_required_deliveries"],
        "cross_panel_pooling": False,
        "curve_scope": "85k_same_frozen_weights_lambda_order_descriptive_only",
        "outputs": {
            "png": {"path": str(png.resolve()), "sha256": _sha256(png)},
            "pdf": {"path": str(pdf.resolve()), "sha256": _sha256(pdf)},
        },
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
    }
    manifest_path = output_dir / "vcg-unified-tradeoff.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(png), "pdf": str(pdf), "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/vcg-unified-tradeoff-figure",
    )
    args = parser.parse_args()
    print(json.dumps(build_figure(args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
