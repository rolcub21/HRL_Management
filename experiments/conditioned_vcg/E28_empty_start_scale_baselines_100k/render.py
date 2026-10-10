"""Four-panel E28 system-level scale figure from the authenticated report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


METHODS = (
    ("vcg", "VCG", "#0072b2"),
    ("duration_aware_dynamic_pslap", "Dynamic PSLAP adaptation", "#009e73"),
    ("duration_aware_pslap_ga_2009_rolling_capacity_aware_partial",
     "Capacity-aware GA adaptation", "#d55e00"),
    ("kim2020_a3c_spatial_adapted__stochastic",
     "A3C adaptation", "#cc79a7"),
)
PANELS = (
    ("mean_instance_completion_rate", "Strict completion (%)", 100.0),
    ("mean_absolute_error", "Placement MAE (steps)", 1.0),
    ("physical_rehandles_per_100_required_deliveries",
     "Rehandles per 100 deliveries", 1.0),
    ("steps_per_delivery", "Steps per delivery", 1.0),
)
SCENARIOS = ("empty_5x5_n8", "empty_8x8_n36", "empty_10x10_n64")
LAMBDAS = (0.0, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2)


def render(report: dict, output: Path) -> dict:
    lookup = {
        (item["scenario_id"], item["method"], item.get("preference_lambda")): item
        for item in report["summaries"]
    }
    expected = {
        (scenario, method, value)
        for scenario in SCENARIOS for method, _, _ in METHODS
        for value in (LAMBDAS if method == "vcg" else (None,))
    }
    if set(lookup) != expected:
        raise ValueError("E28 figure received an incomplete or duplicate panel")
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.5), sharex=True)
    x = np.array([5, 8, 10], dtype=float)
    for ax, (metric, label, factor) in zip(axes.flat, PANELS):
        for method, title, color in METHODS:
            if method == "vcg":
                family = np.array([
                    [
                        np.nan if lookup[(scenario, method, value)][metric] is None
                        else factor * float(lookup[(scenario, method, value)][metric])
                        for value in LAMBDAS
                    ] for scenario in SCENARIOS
                ])
                lower = np.array([
                    np.nanmin(row) if np.isfinite(row).any() else np.nan
                    for row in family
                ])
                upper = np.array([
                    np.nanmax(row) if np.isfinite(row).any() else np.nan
                    for row in family
                ])
                ax.fill_between(x, lower, upper, color=color, alpha=0.13,
                                linewidth=0)
                y = family[:, LAMBDAS.index(0.1)]
                title = "VCG, λ=.10"
            else:
                values = [
                    lookup[(scenario, method, None)][metric]
                    for scenario in SCENARIOS
                ]
                y = np.array([
                    np.nan if value is None else factor * float(value)
                    for value in values
                ])
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.55,
                    color=color, label=title)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(x, ["5×5", "8×8", "10×10"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylim(-3, 103)
    axes[1, 0].set_xlabel("Yard size", fontsize=9)
    axes[1, 1].set_xlabel("Yard size", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.insert(1, Patch(facecolor="#0072b2", alpha=0.13, edgecolor="none"))
    labels.insert(1, "VCG preference range")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=1.35, h_pad=1.15)
    pdf = output / "e28-empty-start-scale-comparison.pdf"
    png = output / "e28-empty-start-scale-comparison.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}
