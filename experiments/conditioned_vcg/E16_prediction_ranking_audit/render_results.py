#!/usr/bin/env python3
"""Render the paper figure for E16 from authenticated result reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
E16B_ROOT = ROOT / "results/vcg-conditioned-e16-continuation-bank-96k"
E16B_REPORT = E16B_ROOT / "e16b-continuation-report.json"
E16C_REPORT = E16B_ROOT / "e16c-component-decomposition-report.json"
E16D_ROOT = ROOT / "results/vcg-conditioned-e16-component-rollout-ablation-92k"
E16D_CONTRACT = E16D_ROOT / "e16d-contract.json"
E16D_REPORT = E16D_ROOT / "e16d-report.json"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e16-mechanism-figure"
FIGURE_STEM = "e16-prediction-to-ranking-mechanism"
PAPER_FIGURE_STEM = "e16-component-effects"
DATA_NAME = "e16-figure-data.json"

ARMS = (
    "immediate_only",
    "action_type_only",
    "candidate_residual_only",
    "full_future",
)
LABELS = {
    "immediate_only": "Immediate cost only",
    "action_type_only": "Action-type component",
    "candidate_residual_only": "Candidate-specific residual",
    "full_future": "Full future signal",
}
COLORS = {
    "immediate_only": "#4B5563",
    "action_type_only": "#E69F00",
    "candidate_residual_only": "#0072B2",
    "full_future": "#009E73",
}
MARKERS = {
    "immediate_only": "o",
    "action_type_only": "s",
    "candidate_residual_only": "D",
    "full_future": "P",
}


class E16FigureError(RuntimeError):
    """Raised when a required result binding or invariant is invalid."""


def _digest(value: Any, hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_hashed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise E16FigureError(f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    hash_field = "report_sha256" if "report_sha256" in value else "contract_sha256"
    expected = value.get(hash_field)
    if expected is not None and expected != _digest(value, hash_field):
        raise E16FigureError(f"self-hash mismatch: {path}")
    return value


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    e16b = _load_hashed(E16B_REPORT)
    e16c = _load_hashed(E16C_REPORT)
    contract = _load_hashed(E16D_CONTRACT)
    e16d = _load_hashed(E16D_REPORT)

    for name, report in (("E16-B", e16b), ("E16-C", e16c), ("E16-D", e16d)):
        if report.get("status") != "complete" or report.get("paper_evidence") is not True:
            raise E16FigureError(f"{name} is not complete paper evidence")
    if e16c.get("e16b_report_sha256") != e16b.get("report_sha256"):
        raise E16FigureError("E16-C is not bound to the current E16-B report")
    if e16d.get("contract_sha256") != contract.get("contract_sha256"):
        raise E16FigureError("E16-D is not bound to the current contract")
    if e16d.get("scope") != "full_90_pair_confirmation":
        raise E16FigureError("E16-D is not the full confirmation")
    return e16b, e16c, e16d


def _figure_data(
    e16b: dict[str, Any], e16c: dict[str, Any], e16d: dict[str, Any]
) -> dict[str, Any]:
    c = e16c["aggregate"]
    expected_counts = {
        "cases": 10,
        "action_type_only_exact_choice_matches_full": 2,
        "candidate_residual_only_exact_choice_matches_full": 8,
        "full_choice_changed_within_action_type": 6,
    }
    for key, expected in expected_counts.items():
        if c.get(key) != expected:
            raise E16FigureError(f"unexpected E16-C {key}: {c.get(key)}")
    within = c["among_within_action_type_changes"]
    if within != {
        "action_type_only_exact_choice_matches_full": 0,
        "candidate_residual_only_exact_choice_matches_full": 6,
    }:
        raise E16FigureError("unexpected within-action-type reproduction counts")

    mechanisms = c["mechanism_labels"]
    expected_mechanisms = {
        "action_type_component_reproduces_full": 1,
        "both_components_individually_reproduce_full": 1,
        "both_components_or_selector_interaction_required": 1,
        "candidate_residual_reproduces_full": 7,
    }
    if mechanisms != expected_mechanisms:
        raise E16FigureError("unexpected mutually exclusive mechanism counts")

    if e16d.get("matched_coordinates") != 90 or e16d.get("current_rollout_rows") != 360:
        raise E16FigureError("unexpected E16-D sample size")
    for arm in ARMS:
        row = e16d["aggregate"].get(arm, {})
        if row.get("rows") != 90 or row.get("strict_safe_complete") != 90:
            raise E16FigureError(f"{arm} is not 90/90 strict-safe complete")
        if row.get("complete_case_metrics_suppressed") is not False:
            raise E16FigureError(f"{arm} metrics are suppressed")
    for control in ("immediate_only", "full_future"):
        parity = e16d["historical_control_parity"][control]
        if parity.get("behavior_digest_equal") != 90 or parity.get("all_outcome_fields_equal") != 90:
            raise E16FigureError(f"{control} failed historical parity")

    points: dict[str, dict[str, float]] = {}
    immediate = e16d["aggregate"]["immediate_only"]
    base_x = float(immediate["physical_rehandles_per_100_required_deliveries"])
    base_y = float(immediate["mean_absolute_error"])
    for arm in ARMS:
        row = e16d["aggregate"][arm]
        x = float(row["physical_rehandles_per_100_required_deliveries"])
        y = float(row["mean_absolute_error"])
        points[arm] = {
            "rehandles_per_100": x,
            "mean_absolute_error": y,
            "delta_rehandles_per_100": x - base_x,
            "delta_mean_absolute_error": y - base_y,
            "strict_safe_complete": int(row["strict_safe_complete"]),
            "rows": int(row["rows"]),
        }
        if arm != "immediate_only":
            paired = e16d["paired_arm_minus_immediate"][arm]
            rehandle_effect = paired[
                "physical_rehandles_per_100_required_deliveries_arm_minus_immediate"
            ]
            mae_effect = paired["mean_absolute_error_arm_minus_immediate"]
            paired_dx = float(rehandle_effect["mean"])
            paired_dy = float(mae_effect["mean"])
            if not math.isclose(points[arm]["delta_rehandles_per_100"], paired_dx, abs_tol=1e-10):
                raise E16FigureError(f"{arm} rehandle delta mismatch")
            if not math.isclose(points[arm]["delta_mean_absolute_error"], paired_dy, abs_tol=1e-10):
                raise E16FigureError(f"{arm} MAE delta mismatch")
            points[arm]["paired_effects"] = {
                "rehandles_per_100": {
                    "mean": paired_dx,
                    "ci95_normal": [float(value) for value in rehandle_effect["ci95_normal"]],
                    "n": int(rehandle_effect["n"]),
                },
                "mean_absolute_error": {
                    "mean": paired_dy,
                    "ci95_normal": [float(value) for value in mae_effect["ci95_normal"]],
                    "n": int(mae_effect["n"]),
                },
            }

    return {
        "protocol": "vcg_conditioned_e16_prediction_to_ranking_figure_v1",
        "status": "complete",
        "source_reports": {
            "e16b": str(E16B_REPORT.relative_to(ROOT)),
            "e16b_sha256": e16b["report_sha256"],
            "e16c": str(E16C_REPORT.relative_to(ROOT)),
            "e16c_sha256": e16c["report_sha256"],
            "e16d": str(E16D_REPORT.relative_to(ROOT)),
            "e16d_sha256": e16d["report_sha256"],
        },
        "claim_scopes": {
            "panel_a": "bounded_descriptive_evidence_at_10_predeclared_visited_crossings",
            "panel_b": "matched_episode_level_component_intervention_at_lambda_0.10",
        },
        "crossing_reproduction": {
            "cases": int(c["cases"]),
            "action_type_only_exact": int(c["action_type_only_exact_choice_matches_full"]),
            "candidate_residual_only_exact": int(
                c["candidate_residual_only_exact_choice_matches_full"]
            ),
            "within_action_type_cases": int(c["full_choice_changed_within_action_type"]),
            "within_action_type_action_type_only_exact": int(
                within["action_type_only_exact_choice_matches_full"]
            ),
            "within_action_type_candidate_residual_only_exact": int(
                within["candidate_residual_only_exact_choice_matches_full"]
            ),
            "mechanism_labels": mechanisms,
            "between_action_type_variation_fraction": float(
                c["pooled_between_action_type_variation_fraction"]
            ),
            "within_action_type_residual_variation_fraction": float(
                c["pooled_within_action_type_residual_variation_fraction"]
            ),
        },
        "rollout_ablation": {
            "matched_coordinates_per_arm": int(e16d["matched_coordinates"]),
            "total_rows": int(e16d["current_rollout_rows"]),
            "strict_safe_complete_rows": sum(
                int(e16d["aggregate"][arm]["strict_safe_complete"]) for arm in ARMS
            ),
            "points": points,
        },
    }


def _render_figure(data: dict[str, Any], output: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.7))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.82, wspace=0.38)

    crossing = data["crossing_reproduction"]
    names = ["Candidate-specific\nresidual", "Action-type\ncomponent"]
    values = [crossing["candidate_residual_only_exact"], crossing["action_type_only_exact"]]
    colors = [COLORS["candidate_residual_only"], COLORS["action_type_only"]]
    bars = ax0.barh([0, 1], values, color=colors, height=0.52, edgecolor="white", linewidth=1.2)
    for bar, value in zip(bars, values):
        x = value - 0.20 if value >= 2 else value + 0.18
        ax0.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value}/10",
            va="center",
            ha="right" if value >= 2 else "left",
            color="white" if value >= 2 else "#1F2937",
            fontweight="bold",
            fontsize=12,
        )
    ax0.set_xlim(0, 10)
    ax0.set_ylim(-0.92, 1.55)
    ax0.set_yticks([0, 1], names)
    ax0.set_xlabel("Exact reproduction of the full-signal choice")
    ax0.set_title("(a) Which component reproduces local crossings?", loc="left", fontweight="bold")
    ax0.grid(axis="x", color="#D1D5DB", alpha=0.65, linewidth=0.8)
    ax0.set_axisbelow(True)
    ax0.text(
        0.02,
        0.035,
        "Within-action-type switches: residual 6/6; action type 0/6\n"
        "Exclusive cases: residual 7 | action type 1 | either 1 | interaction 1",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        color="#4B5563",
        fontsize=9.5,
        linespacing=1.45,
    )

    points = data["rollout_ablation"]["points"]
    base = points["immediate_only"]
    bx, by = base["rehandles_per_100"], base["mean_absolute_error"]
    for arm in ARMS[1:]:
        point = points[arm]
        ax1.annotate(
            "",
            xy=(point["rehandles_per_100"], point["mean_absolute_error"]),
            xytext=(bx, by),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS[arm],
                "linewidth": 2.4,
                "mutation_scale": 15,
                "alpha": 0.82,
                "shrinkA": 10,
                "shrinkB": 10,
            },
            zorder=1,
        )
    for arm in ARMS:
        point = points[arm]
        ax1.scatter(
            point["rehandles_per_100"],
            point["mean_absolute_error"],
            s=115 if arm == "full_future" else 90,
            marker=MARKERS[arm],
            facecolor=COLORS[arm],
            edgecolor="white",
            linewidth=1.5,
            label=LABELS[arm],
            zorder=3,
        )
    offsets = {
        "immediate_only": (-8, 10, "right", "bottom"),
        "action_type_only": (0, 13, "center", "bottom"),
        "candidate_residual_only": (12, -12, "left", "top"),
        "full_future": (12, -11, "left", "top"),
    }
    for arm in ARMS:
        point = points[arm]
        dx, dy, ha, va = offsets[arm]
        text = LABELS[arm]
        if arm != "immediate_only":
            text += (
                f"\nΔR={point['delta_rehandles_per_100']:+.2f}; "
                f"ΔMAE={point['delta_mean_absolute_error']:+.2f}"
            )
        ax1.annotate(
            text,
            (point["rehandles_per_100"], point["mean_absolute_error"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=COLORS[arm],
            fontweight="bold" if arm == "full_future" else "normal",
            fontsize=9.1,
            linespacing=1.25,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax1.set_xlabel("Physical rehandles / 100 required deliveries  ↓")
    ax1.set_ylabel("Mean absolute delivery error  ↓")
    ax1.set_title("(b) What does each component change in deployment?", loc="left", fontweight="bold")
    ax1.grid(color="#D1D5DB", alpha=0.65, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_xlim(1.65, 9.45)
    ax1.set_ylim(13.20, 14.13)

    fig.suptitle(
        "From learned future-consequence components to deployed behavior",
        fontsize=15.5,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        "Panel (a): 10 predeclared visited crossings (bounded descriptive evidence).  "
        "Panel (b): λ=0.10, 90 matched model-seed × instance coordinates per arm; "
        "360/360 strict-safe complete.",
        ha="center",
        va="bottom",
        fontsize=9.1,
        color="#4B5563",
    )
    pdf = output / f"{FIGURE_STEM}.pdf"
    png = output / f"{FIGURE_STEM}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _render_paper_figure(data: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Render the conventional statistical-summary version for the paper."""
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.65),
        gridspec_kw={"width_ratios": [1.18, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.75, wspace=0.45)
    ax0, ax1, ax2 = axes

    crossing = data["crossing_reproduction"]
    groups = ["All crossings\n($n=10$)", "Within-type switches\n($n=6$)"]
    action_counts = [crossing["action_type_only_exact"], crossing["within_action_type_action_type_only_exact"]]
    residual_counts = [
        crossing["candidate_residual_only_exact"],
        crossing["within_action_type_candidate_residual_only_exact"],
    ]
    denominators = [crossing["cases"], crossing["within_action_type_cases"]]
    action_rates = [100.0 * count / denominator for count, denominator in zip(action_counts, denominators)]
    residual_rates = [100.0 * count / denominator for count, denominator in zip(residual_counts, denominators)]
    xpos = [0, 1]
    width = 0.34
    bars_action = ax0.bar(
        [x - width / 2 for x in xpos],
        action_rates,
        width,
        color=COLORS["action_type_only"],
        label="Action-type component",
    )
    bars_residual = ax0.bar(
        [x + width / 2 for x in xpos],
        residual_rates,
        width,
        color=COLORS["candidate_residual_only"],
        label="Candidate-specific residual",
    )
    for bars, counts in ((bars_action, action_counts), (bars_residual, residual_counts)):
        for index, (bar, count) in enumerate(zip(bars, counts)):
            denominator = denominators[index]
            y = bar.get_height()
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                max(y + 3.0, 3.0),
                f"{count}/{denominator}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
    ax0.set_xticks(xpos, groups)
    ax0.set_ylim(0, 112)
    ax0.set_ylabel("Exact-choice reproduction (%)")
    ax0.set_title("(a) Local ranking reproduction", loc="left")
    ax0.grid(axis="y", color="#D1D5DB", alpha=0.65, linewidth=0.8)
    ax0.set_axisbelow(True)
    ax0.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        frameon=False,
        ncol=1,
        handlelength=1.5,
    )

    forest_arms = ("action_type_only", "candidate_residual_only", "full_future")
    y_positions = [2, 1, 0]
    forest_labels = ["Action type", "Candidate residual", "Full signal"]

    def draw_forest(ax: Any, metric: str, xlabel: str, title: str, xlim: tuple[float, float]) -> None:
        ax.axvline(0.0, color="#6B7280", linestyle="--", linewidth=1.2, zorder=0)
        for y, arm in zip(y_positions, forest_arms):
            effect = data["rollout_ablation"]["points"][arm]["paired_effects"][metric]
            mean = effect["mean"]
            lo, hi = effect["ci95_normal"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - lo], [hi - mean]],
                fmt=MARKERS[arm],
                markersize=8,
                color=COLORS[arm],
                markeredgecolor="white",
                markeredgewidth=1.0,
                elinewidth=2.0,
                capsize=4,
                capthick=1.5,
                zorder=2,
            )
            ax.text(
                0.98,
                y,
                f"{mean:+.2f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color="#374151",
            )
        ax.set_yticks(y_positions, forest_labels)
        ax.set_ylim(-0.65, 2.65)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="#D1D5DB", alpha=0.65, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(0.98, 1.035, "Mean", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#6B7280")

    draw_forest(
        ax1,
        "rehandles_per_100",
        "Paired difference vs immediate only\n(rehandles / 100; ↓ better)",
        "(b) Handling effect",
        (-9.3, 1.4),
    )
    draw_forest(
        ax2,
        "mean_absolute_error",
        "Paired difference vs immediate only\n(MAE; ↓ better)",
        "(c) Timing effect",
        (-1.12, 0.82),
    )

    fig.suptitle("Prediction components: local ranking and deployment effects", fontsize=15, y=0.95)
    fig.text(
        0.5,
        0.045,
        "Panel (a): 10 predeclared visited crossings (descriptive).  "
        "Panels (b,c): paired means and normal-approximation 95% CIs over 90 matched "
        "model-seed × instance coordinates per arm at λ=0.10; all 360 rows strict-safe complete.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4B5563",
    )

    pdf = output / f"{PAPER_FIGURE_STEM}.pdf"
    png = output / f"{PAPER_FIGURE_STEM}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def render(output: Path) -> dict[str, Any]:
    e16b, e16c, e16d = _load_inputs()
    data = _figure_data(e16b, e16c, e16d)
    output.mkdir(parents=True, exist_ok=True)
    explanatory_pdf, explanatory_png = _render_figure(data, output)
    paper_pdf, paper_png = _render_paper_figure(data, output)
    data_path = output / DATA_NAME
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "complete",
        "paper_pdf": str(paper_pdf),
        "paper_png": str(paper_png),
        "explanatory_pdf": str(explanatory_pdf),
        "explanatory_png": str(explanatory_png),
        "data": str(data_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
