#!/usr/bin/env python3
"""Render the final E13/E14 Block 6 figure and companion tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    budget_panel as e14,
)
from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    threshold_refinement as e14_threshold,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-block6-scalability-95k"
FIGURE_STEM = "block6-scalability"
TABLE_NAME = "block6-companion-tables.md"
DATA_NAME = "block6-figure-data.json"

OCCUPANCIES = ("low", "medium", "high")
SIZES = (5, 8, 10)
COLORS = {"low": "#2c7fb8", "medium": "#10a37f", "high": "#e69f00"}
MARKERS = {5: "o", 8: "s", 10: "D"}


class Block6RenderError(RuntimeError):
    pass


def _load_report(path: Path, hash_field: str, label: str) -> dict:
    value = e13._self_hashed(path, hash_field, label)
    return value


def _inputs(allow_partial: bool) -> tuple[dict, dict]:
    e13_contract, _manifest, _parent = e13.authenticate_frozen(
        e13.DEFAULT_OUTPUT
    )
    e14_threshold_contract = e14_threshold.authenticate_frozen(
        e14_threshold.DEFAULT_OUTPUT
    )
    e13_report = _load_report(
        e13.DEFAULT_OUTPUT / e13.REPORT_NAME, "report_sha256", "E13 report"
    )
    e14_report = _load_report(
        e14_threshold.DEFAULT_OUTPUT / e14_threshold.REPORT_NAME,
        "report_sha256",
        "E14 threshold-refinement report",
    )
    if e13_report["contract_sha256"] != e13_contract["contract_sha256"]:
        raise Block6RenderError("E13 report contract mismatch")
    if (
        e14_report["contract_sha256"]
        != e14_threshold_contract["contract_sha256"]
    ):
        raise Block6RenderError("E14 threshold report contract mismatch")
    if not allow_partial and (
        not e13_report["all_coordinates_complete"]
        or e14_report["partial"]
    ):
        raise Block6RenderError("E13/E14 panels are incomplete")
    return e13_report, e14_report


def _main_summaries(report: Mapping) -> list[dict]:
    by_id = {
        item["scenario"]["scenario_id"]: item for item in report["summaries"]
    }
    return [by_id[item] for item in e13.MAIN_SCENARIOS]


def _companion_summaries(report: Mapping) -> list[dict]:
    by_id = {
        item["scenario"]["scenario_id"]: item for item in report["summaries"]
    }
    return [by_id[item] for item in e13.COMPANION_SCENARIOS]


def _all_summaries(report: Mapping) -> list[dict]:
    return _main_summaries(report) + _companion_summaries(report)


def _scenario_code(item: Mapping) -> str:
    scenario = item["scenario"]
    initial = 100.0 * item["occupancy"]["initial_storage_occupancy_ratio"]["mean"]
    return (
        f'{scenario["rows"]}×{scenario["cols"]} '
        f'{scenario["occupancy_level"][0].upper()} ({initial:.1f}%)'
    )


def _companion_code(item: Mapping) -> str:
    scenario = item["scenario"]
    comparison = scenario["comparison"]
    size = f'{scenario["rows"]}×{scenario["cols"]}'
    if comparison == "geometry_fixed_workload":
        return (
            f'{size}, fixed workload '
            f'K={scenario["initial_occupied_slots"]}, N={scenario["total_jobs"]}'
        )
    if comparison == "episode_length":
        return f'{size}, medium occupancy, N={scenario["total_jobs"]}'
    if comparison == "aspect_ratio":
        return f"{size}, matched medium occupancy"
    raise Block6RenderError(
        f"unknown E13 companion comparison: {comparison}"
    )


def _scenario_tick(item: Mapping) -> str:
    scenario = item["scenario"]
    initial = 100.0 * item["occupancy"]["initial_storage_occupancy_ratio"]["mean"]
    return (
        f'{scenario["rows"]}×{scenario["cols"]}\n'
        f'{scenario["occupancy_level"][0].upper()}={initial:.1f}%'
    )


def _number(value, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _budget_label(value: int) -> str:
    return "20k ref." if int(value) == 20_000 else str(int(value))


def _append_companion_table(lines: list[str], report: Mapping) -> None:
    lines.extend(
        [
            "",
            "## E13 — Companion isolation controls",
            "",
            "These six coordinates isolate geometry at fixed workload, episode length at fixed 8×8 medium occupancy, and aspect ratio at matched capacity and occupancy.",
            "",
            "| Control | Strict | Initial occ. | Mean occ. | Peak occ. | MAE | Rehandles/100 | Steps/del. | Within ±20 | Physical cand. | Certified cand. | Warm p95 (s) | >1 s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in _companion_summaries(report):
        metrics = item["all_required_rows_metrics"]
        occupancy = item["occupancy"]
        latency = item["latency"]
        lines.append(
            "| {code} | {strict}/{expected} | {initial} | {mean} | {peak} | "
            "{mae} | {rehandles} | {steps} | {window} | {physical} | "
            "{certified} | {warm_p95} | {over} |".format(
                code=_companion_code(item),
                strict=item["strict_safe_complete"],
                expected=item["expected_rows"],
                initial=_number(
                    occupancy["initial_storage_occupancy_ratio"]["mean"], 3
                ),
                mean=_number(
                    occupancy[
                        "time_weighted_mean_storage_occupancy_ratio"
                    ]["mean"],
                    3,
                ),
                peak=_number(
                    occupancy["peak_storage_occupancy_ratio"]["mean"], 3
                ),
                mae=_number(
                    None if metrics is None else metrics["mean_absolute_error"]
                ),
                rehandles=_number(
                    None
                    if metrics is None
                    else metrics[
                        "physical_rehandles_per_100_required_deliveries"
                    ]
                ),
                steps=_number(
                    None if metrics is None else metrics["steps_per_delivery"]
                ),
                window=_number(
                    None
                    if metrics is None
                    else 100.0 * metrics["within_target_window_rate"],
                    1,
                ),
                physical=_number(
                    item["frontier_by_action"]["physical_candidate_count"][
                        "mean"
                    ],
                    1,
                ),
                certified=_number(
                    item["frontier_by_action"]["certified_candidate_count"][
                        "mean"
                    ],
                    1,
                ),
                warm_p95=_number(
                    latency["warm_end_to_end_decision_seconds"]["p95"]
                ),
                over=_number(
                    None
                    if latency["warm_deadline_exceedance_rate"]["1.0"] is None
                    else 100.0
                    * latency["warm_deadline_exceedance_rate"]["1.0"],
                    1,
                ),
            )
        )


def _make_tables(e13_report: Mapping, e14_report: Mapping) -> str:
    lines = [
        "# Block 6 scalability companion tables",
        "",
        "## E13 — Scale by usable-storage occupancy",
        "",
        "| Coordinate | Strict | Initial occ. | Mean occ. | Peak occ. | MAE | Rehandles/100 | Steps/del. | Within ±20 | Physical cand. | Certified cand. | D12 proofs/misses | Native checks/decision | Native p95 (ms) | Warm p95 (s) | >1 s | Peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in _main_summaries(e13_report):
        metrics = item["all_required_rows_metrics"]
        occ = item["occupancy"]
        family = item["family"]
        native = item["native_searches"]
        latency = item["latency"]
        memory = item["memory"]
        lines.append(
            "| {code} | {strict}/{expected} | {initial_occ} | {mean_occ} | {peak_occ} | {mae} | "
            "{rehandles} | {steps} | {window} | {physical} | {certified} | "
            "{proofs}/{misses} | {checks} | {native_p95} | {warm_p95} | "
            "{over} | {rss} |".format(
                code=_scenario_code(item),
                strict=item["strict_safe_complete"],
                expected=item["expected_rows"],
                initial_occ=_number(
                    occ["initial_storage_occupancy_ratio"]["mean"], 3
                ),
                mean_occ=_number(
                    occ["time_weighted_mean_storage_occupancy_ratio"]["mean"], 3
                ),
                peak_occ=_number(occ["peak_storage_occupancy_ratio"]["mean"], 3),
                mae=_number(None if metrics is None else metrics["mean_absolute_error"]),
                rehandles=_number(
                    None
                    if metrics is None
                    else metrics["physical_rehandles_per_100_required_deliveries"]
                ),
                steps=_number(None if metrics is None else metrics["steps_per_delivery"]),
                window=_number(
                    None
                    if metrics is None
                    else 100.0 * metrics["within_target_window_rate"],
                    1,
                ),
                physical=_number(
                    item["frontier_by_action"]["physical_candidate_count"]["mean"],
                    1,
                ),
                certified=_number(
                    item["frontier_by_action"]["certified_candidate_count"]["mean"],
                    1,
                ),
                proofs=family["proofs"],
                misses=family["misses"],
                checks=_number(native["per_decision"], 1),
                native_p95=_number(
                    None
                    if native["seconds_per_search"]["p95"] is None
                    else 1000.0 * native["seconds_per_search"]["p95"],
                    1,
                ),
                warm_p95=_number(latency["warm_end_to_end_decision_seconds"]["p95"]),
                over=_number(
                    None
                    if latency["warm_deadline_exceedance_rate"]["1.0"] is None
                    else 100.0 * latency["warm_deadline_exceedance_rate"]["1.0"],
                    1,
                ),
                rss=_number(
                    None
                    if memory["peak_rss_kib"]["mean"] is None
                    else memory["peak_rss_kib"]["mean"] / 1024.0,
                    0,
                ),
            )
        )
    lines.extend(
        [
            "",
            "Quality fields are suppressed unless all three frozen instances strictly complete.",
        ]
    )
    _append_companion_table(lines, e13_report)
    lines.extend(
        [
            "",
            "## E14 — Native-search budget sensitivity",
            "",
            "| Max nodes | Strict | Initial anchor SAFE | Initial certified/physical | Initial UNKNOWN | Initial empty | D12 proof rate | Native checks | Native expansions p95 | Warm p95 (s) | >1 s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    matched_by_budget = {
        item["max_nodes"]: item
        for item in e14_report["matched_initial_frontiers"]
    }
    for item in e14_report["summaries"]:
        matched = matched_by_budget[item["max_nodes"]]
        family = item["family"]
        native = item["native_searches"]
        latency = item["latency"]
        lines.append(
            "| {budget} | {strict}/{expected} | {safe}/{matched_n} | {coverage} | {unknown} | {empty} | "
            "{family} | {checks} | {nodes} | {p95} | {over} |".format(
                budget=_budget_label(item["max_nodes"]),
                strict=item["strict_safe_complete"],
                expected=item["expected_rows"],
                safe=matched["current_state_safe"],
                matched_n=matched["matched_initial_states"],
                coverage=_number(matched["certified_to_physical_ratio"], 3),
                unknown=matched["unknown_candidates"],
                empty=matched["empty_certified_frontiers"],
                family=_number(family["proof_rate"], 3),
                checks=native["count"],
                nodes=_number(native["expanded_nodes"]["p95"], 1),
                p95=_number(latency["warm_end_to_end_decision_seconds"]["p95"]),
                over=_number(
                    None
                    if latency["warm_deadline_exceedance_rate"]["1.0"] is None
                    else 100.0 * latency["warm_deadline_exceedance_rate"]["1.0"],
                    1,
                ),
            )
        )
    lines.extend(
        [
            "",
            "Coverage, UNKNOWN, and empty-frontier counts use the same nine initial states at every budget; they do not mix trajectories of different lengths.",
            "",
            "Warm p95 latency is descriptive for the decisions each arm actually reached; failed arms stop early, so this latency series is not a matched-decision comparison.",
            "",
            "The 32- and 64-node arms are a post-hoc descriptive refinement of the predeclared E14 sweep. The 20,000-node reference consists of authenticated E13 rows; its observed native searches used at most 126 nodes (128 is a conservative sufficient cap only for these deterministic reference trajectories).",
        ]
    )
    return "\n".join(lines) + "\n"


def _style_axis(ax, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", fontsize=11, pad=10)
    ax.grid(True, color="#d9dee3", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def render(output: Path, *, allow_partial: bool = False) -> dict:
    e13_report, e14_report = _inputs(allow_partial)
    main = _main_summaries(e13_report)
    all_e13 = _all_summaries(e13_report)
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": "#fbfbfa",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))
    ax = axes[0, 0]
    for item in main:
        metrics = item["all_required_rows_metrics"]
        if metrics is None:
            continue
        scenario = item["scenario"]
        size = int(scenario["rows"])
        occupancy = scenario["occupancy_level"]
        ax.scatter(
            metrics["physical_rehandles_per_100_required_deliveries"],
            metrics["mean_absolute_error"],
            s=75,
            marker=MARKERS[size],
            color=COLORS[occupancy],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{size}×{size}",
            (
                metrics["physical_rehandles_per_100_required_deliveries"],
                metrics["mean_absolute_error"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Physical rehandles / 100 deliveries ↓")
    ax.set_ylabel("Delivery-time MAE (simulation steps) ↓")
    main_strict = sum(item["strict_safe_complete"] for item in main)
    main_expected = sum(item["expected_rows"] for item in main)
    all_strict = sum(item["strict_safe_complete"] for item in all_e13)
    all_expected = sum(item["expected_rows"] for item in all_e13)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLORS[level],
                markeredgecolor="white",
                markersize=8,
                label=(
                    f"{level.title()}: "
                    f"{min(100.0 * item['occupancy']['initial_storage_occupancy_ratio']['mean'] for item in main if item['scenario']['occupancy_level'] == level):.0f}–"
                    f"{max(100.0 * item['occupancy']['initial_storage_occupancy_ratio']['mean'] for item in main if item['scenario']['occupancy_level'] == level):.0f}% initial"
                ),
            )
            for level in OCCUPANCIES
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    _style_axis(
        ax,
        "(a)",
        (
            f"Operational quality (main {main_strict}/{main_expected}; "
            f"all E13 {all_strict}/{all_expected} strict)"
        ),
    )

    ax = axes[0, 1]
    labels = [_scenario_tick(item) for item in main]
    xpos = np.arange(len(main))
    physical = [
        item["frontier_by_action"]["physical_candidate_count"]["mean"]
        if item["observed_rows"]
        else np.nan
        for item in main
    ]
    certified = [
        item["frontier_by_action"]["certified_candidate_count"]["mean"]
        if item["observed_rows"]
        else np.nan
        for item in main
    ]
    ax.bar(xpos - 0.18, physical, width=0.36, color="#9aa5b1", label="Physical")
    ax.bar(xpos + 0.18, certified, width=0.36, color="#0f8f72", label="Certified")
    ax.set_xticks(xpos, labels, rotation=40, ha="right")
    ax.set_ylabel("Mean candidates per decision\n(pooled across 3 episodes)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(ax, "(b)", "Decision-space breadth")

    ax = axes[1, 0]
    family = []
    native = []
    for item in main:
        decisions = item["frontier_by_action"]["physical_candidate_count"]["n"]
        family.append(item["family"]["proofs"] / decisions if decisions else np.nan)
        native.append(
            item["native_searches"]["per_decision"]
            if item["native_searches"]["per_decision"] is not None
            else np.nan
        )
    ax.bar(xpos, family, color="#65c2a5", label="D12 constructive proof")
    ax.bar(
        xpos,
        native,
        bottom=family,
        color="#4c566a",
        label="Native exact search",
    )
    ax.set_xticks(xpos, labels, rotation=40, ha="right")
    ax.set_ylabel("Mean requests per decision\n(pooled across 3 episodes)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(ax, "(c)", "Certification requests by resolution method")

    ax = axes[1, 1]
    budgets = np.array([item["max_nodes"] for item in e14_report["summaries"]])
    budget_positions = np.arange(len(budgets))
    matched_by_budget = {
        item["max_nodes"]: item
        for item in e14_report["matched_initial_frontiers"]
    }
    coverage = np.array(
        [
            np.nan
            if matched_by_budget[item["max_nodes"]][
                "certified_to_physical_ratio"
            ]
            is None
            else 100.0
            * matched_by_budget[item["max_nodes"]][
                "certified_to_physical_ratio"
            ]
            for item in e14_report["summaries"]
        ]
    )
    completion = np.array(
        [
            np.nan
            if item["strict_completion_rate"] is None
            else 100.0 * item["strict_completion_rate"]
            for item in e14_report["summaries"]
        ]
    )
    latency = np.array(
        [
            np.nan
            if item["latency"]["warm_end_to_end_decision_seconds"]["p95"] is None
            else item["latency"]["warm_end_to_end_decision_seconds"]["p95"]
            for item in e14_report["summaries"]
        ]
    )
    ax.plot(
        budget_positions,
        coverage,
        "o-",
        color="#0f8f72",
        label="Initial certified/physical (9 states)",
    )
    ax.plot(
        budget_positions,
        completion,
        "s--",
        color="#2c7fb8",
        label="Strict completion (9 episodes)",
    )
    ax.set_xticks(
        budget_positions,
        [
            f"{_budget_label(item)}*" if int(item) in (32, 64)
            else _budget_label(item)
            for item in budgets
        ],
    )
    ax.set_xlabel(
        "Native search budget (max nodes; * post-hoc refinement)\n"
        "Latency p95 uses decisions reached by each arm"
    )
    ax.set_ylabel("Initial candidate coverage / strict completion (%)")
    ax.set_ylim(-3, 103)
    latency_ax = ax.twinx()
    latency_ax.plot(
        budget_positions,
        latency,
        "D-.",
        color="#e69f00",
        label="Observed warm p95 latency",
    )
    latency_ax.set_ylabel("Warm p95 decision latency (s)", color="#9a6700")
    latency_ax.spines["top"].set_visible(False)
    handles, legend_labels = ax.get_legend_handles_labels()
    h2, l2 = latency_ax.get_legend_handles_labels()
    ax.legend(
        handles + h2,
        legend_labels + l2,
        frameon=False,
        loc="upper left",
        fontsize=8,
    )
    _style_axis(ax, "(d)", "Budget–coverage–latency sensitivity")

    figure.suptitle(
        "Frozen VCG scalability with E14 cleanup and D12 amortization",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.2, w_pad=2.0)
    pdf = output / f"{FIGURE_STEM}.pdf"
    png = output / f"{FIGURE_STEM}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(figure)

    table = output / TABLE_NAME
    table.write_text(_make_tables(e13_report, e14_report), encoding="utf-8")
    data = {
        "schema_version": 1,
        "e13_report_sha256": e13_report["report_sha256"],
        "e13_main_scenarios": len(main),
        "e13_main_strict_safe_complete": main_strict,
        "e13_main_expected_rows": main_expected,
        "e13_companion_scenarios": len(_companion_summaries(e13_report)),
        "e13_all_strict_safe_complete": all_strict,
        "e13_all_expected_rows": all_expected,
        "e14_threshold_report_sha256": e14_report["report_sha256"],
        "e14_post_hoc_threshold_refinement": True,
        "partial": bool(e13_report["partial"] or e14_report["partial"]),
        "figure_pdf": str(pdf.relative_to(ROOT)),
        "figure_png": str(png.relative_to(ROOT)),
        "table": str(table.relative_to(ROOT)),
    }
    (output / DATA_NAME).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            render(args.output_dir.resolve(), allow_partial=args.allow_partial),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
