"""Reporting-only E28 presentation; leave the contract-hashed renderer intact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import render as frozen

plt = frozen.plt
np = frozen.np


def render(report: dict, output: Path) -> dict:
    """Highlight the highest declared preference, not a per-scenario optimum."""
    lookup = {
        (item["scenario_id"], item["method"], item.get("preference_lambda")): item
        for item in report["summaries"]
    }
    expected = {
        (scenario, method, value)
        for scenario in frozen.SCENARIOS for method, _, _ in frozen.METHODS
        for value in (frozen.LAMBDAS if method == "vcg" else (None,))
    }
    if set(lookup) != expected or len(lookup) != len(report["summaries"]):
        raise ValueError("E28 figure received an incomplete or duplicate panel")
    highlighted_lambda = max(frozen.LAMBDAS)
    highlighted_index = frozen.LAMBDAS.index(highlighted_lambda)
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.5), sharex=True)
    x = np.array([5, 8, 10], dtype=float)
    plotted = []
    for ax, (metric, label, factor) in zip(axes.flat, frozen.PANELS):
        for method, title, color in frozen.METHODS:
            record = {"metric": metric, "method": method, "display_factor": factor}
            if method == "vcg":
                values = [
                    [lookup[(scenario, method, value)][metric]
                     for value in frozen.LAMBDAS]
                    for scenario in frozen.SCENARIOS
                ]
                family = np.array([
                    [np.nan if value is None else factor * float(value)
                     for value in row] for row in values
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
                y = family[:, highlighted_index]
                title = f"VCG, λ={highlighted_lambda:.2f}".replace("λ=0.", "λ=.")
                record.update(
                    preference_lambda=highlighted_lambda,
                    family_values=values,
                    line_values=[row[highlighted_index] for row in values],
                )
            else:
                values = [lookup[(scenario, method, None)][metric]
                          for scenario in frozen.SCENARIOS]
                y = np.array([
                    np.nan if value is None else factor * float(value)
                    for value in values
                ])
                record.update(preference_lambda=None, line_values=values)
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.55,
                    color=color, label=title)
            plotted.append(record)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(x, ["5×5", "8×8", "10×10"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylim(-3, 103)
    axes[1, 0].set_xlabel("Yard size", fontsize=9)
    axes[1, 1].set_xlabel("Yard size", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.insert(1, frozen.Patch(facecolor="#0072b2", alpha=0.13, edgecolor="none"))
    labels.insert(1, "VCG range across all 10 preferences")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=1.35, h_pad=1.15)
    pdf = output / "e28-empty-start-scale-comparison.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    data = output / "e28-empty-start-scale-comparison-data.json"
    data.write_text(json.dumps({
        "schema_version": 1,
        "report_sha256": report["report_sha256"],
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "highlighted_lambda": highlighted_lambda,
        "highlight_rule": "highest declared lambda, fixed across scenarios",
        "band": "min/max across ten preferences, not an uncertainty interval",
        "scenarios": list(frozen.SCENARIOS),
        "preference_lambdas": list(frozen.LAMBDAS),
        "series": plotted,
        "rollouts_rerun": 0,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"pdf": str(pdf), "png": str(png), "data": str(data)}
