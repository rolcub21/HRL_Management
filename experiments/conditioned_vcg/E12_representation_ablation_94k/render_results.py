#!/usr/bin/env python3
"""Render the complete E12 representation-attribution result."""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import fmean
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    program,
)


DEFAULT_OUTPUT = program.DEFAULT_OUTPUT
REPORT_PATH = DEFAULT_OUTPUT / "analysis/e11_93k-report.json"
FIGURE_STEM = "e12-representation-shift-interactions"
TABLE_NAME = "e12-representation-summary.md"
DATA_NAME = "e12-figure-data.json"

ABLATIONS = (
    "nonrelational_successor",
    "relational_current_candidate",
)
ABLATION_LABELS = {
    "nonrelational_successor": "No relational edges",
    "relational_current_candidate": "No explicit successor",
}
COLORS = {
    "nonrelational_successor": "#2878b5",
    "relational_current_candidate": "#c44e84",
}
MARKERS = {
    "nonrelational_successor": "o",
    "relational_current_candidate": "s",
}
REGIME_LABELS = {
    "arrival_spread": "Spread\narrivals",
    "dwell_short": "Short\ndwell",
    "dwell_long": "Long\ndwell",
    "dwell_bimodal": "Bimodal\ndwell",
    "mirrored_entry": "Mirrored\nentry",
    "combined_shift": "Combined\nshift",
}
METRICS = (
    ("mean_absolute_error", "(a)  Delivery-time MAE", "Simulation steps"),
    (
        "physical_rehandles_per_100_required_deliveries",
        "(b)  Physical rehandles",
        "Rehandles / 100 deliveries",
    ),
    ("steps", "(c)  Episode length", "Simulation steps"),
)


class E12RenderError(RuntimeError):
    pass


def _load_complete_report(output: Path) -> tuple[dict, dict]:
    contract, _manifest = program.authenticate(output)
    report = program.load_json(
        output / "analysis/e11_93k-report.json",
        label="E12 confirmation report",
    )
    if report.get("report_sha256") != program.digest(
        report, hash_field="report_sha256"
    ):
        raise E12RenderError("E12 confirmation report self-hash mismatch")
    if report.get("contract_sha256") != contract["contract_sha256"]:
        raise E12RenderError("E12 report/contract mismatch")
    if (
        report.get("partial") is not False
        or report.get("observed_rows") != report.get("expected_rows")
    ):
        raise E12RenderError("paper rendering requires the complete E12 panel")
    return contract, report


def _regimes() -> tuple[str, ...]:
    return tuple(
        item.regime_id
        for item in program.e11.REGIMES
        if item.regime_id != "reference"
    )


def _interactions(report: Mapping) -> list[dict]:
    rows = [
        dict(item)
        for item in report["shift_degradation_interactions"]
        if item["n"] > 0
    ]
    expected = len(ABLATIONS) * 3 * 4 * len(_regimes()) * 3
    if len(rows) != expected:
        raise E12RenderError(
            f"expected {expected} populated interaction cells, found {len(rows)}"
        )
    return rows


def _failures(output: Path, contract: Mapping) -> list[dict]:
    failures = []
    evaluation = output / "evaluation/e11_93k"
    for path in sorted(evaluation.glob("**/instance-*.json")):
        ledger = program.load_json(path, label="E12 evaluation ledger")
        if ledger.get("ledger_sha256") != program.digest(
            ledger, hash_field="ledger_sha256"
        ):
            raise E12RenderError(f"ledger self-hash mismatch: {path}")
        if ledger.get("contract_sha256") != contract["contract_sha256"]:
            raise E12RenderError(f"ledger contract mismatch: {path}")
        row = ledger["row"]
        if not row["strict_safe_complete"]:
            failures.append(
                {
                    "representation_variant": row["representation_variant"],
                    "model_seed": row["model_seed"],
                    "preference_lambda": row["preference_lambda"],
                    "regime_id": row["regime_id"],
                    "instance_seed": row["instance_seed"],
                    "reason": row["method_failure_reason"],
                    "all_selected_candidates_exact_safe": row[
                        "all_selected_candidates_exact_safe"
                    ],
                }
            )
    return failures


def _completion(report: Mapping) -> dict[str, tuple[int, int]]:
    values = defaultdict(lambda: [0, 0])
    for item in report["aggregates"]:
        key = item["representation_variant"]
        values[key][0] += int(item["strict_safe_complete"])
        values[key][1] += int(item["rows"])
    return {key: tuple(value) for key, value in values.items()}


def _cell_groups(rows: Sequence[Mapping]) -> dict[tuple, list[float]]:
    groups = defaultdict(list)
    for item in rows:
        groups[
            (
                item["ablation"],
                item["metric"],
                item["regime_id"],
                int(item["model_seed"]),
            )
        ].append(float(item["mean"]))
    return groups


def _overall(rows: Sequence[Mapping], ablation: str, metric: str) -> dict:
    selected = [
        item
        for item in rows
        if item["ablation"] == ablation and item["metric"] == metric
    ]
    return {
        "cells": len(selected),
        "mean": fmean(float(item["mean"]) for item in selected),
        "positive": sum(float(item["mean"]) > 0 for item in selected),
        "negative": sum(float(item["mean"]) < 0 for item in selected),
        "zero": sum(float(item["mean"]) == 0 for item in selected),
        "minimum_n": min(int(item["n"]) for item in selected),
        "maximum_n": max(int(item["n"]) for item in selected),
    }


def _fmt(value: float) -> str:
    return f"{value:+.2f}"


def _table(
    report: Mapping,
    rows: Sequence[Mapping],
    failures: Sequence[Mapping],
) -> str:
    completion = _completion(report)
    groups = _cell_groups(rows)
    regimes = _regimes()
    lines = [
        "# E12 representation attribution",
        "",
        "The interaction is `(shift - reference)_ablation - (shift - reference)_full`. Positive lower-is-better values mean the ablated representation deteriorated more under shift.",
        "",
        "## Strict completion",
        "",
        "| Representation | Strict completion |",
        "|---|---:|",
    ]
    for variant, label in (
        ("full_relational_successor", "Full relational + successor"),
        ("nonrelational_successor", "No relational edges"),
        ("relational_current_candidate", "No explicit successor"),
    ):
        strict, total = completion[variant]
        lines.append(f"| {label} | {strict}/{total} |")

    lines.extend(
        [
            "",
            "## Overall shift-degradation interactions",
            "",
            "| Ablation | Metric | Mean interaction | Positive cells | Matched pairs/cell |",
            "|---|---|---:|---:|---:|",
        ]
    )
    metric_labels = {
        "mean_absolute_error": "Delivery-time MAE",
        "physical_rehandles_per_100_required_deliveries": "Rehandles/100",
        "steps": "Steps",
    }
    for ablation in ABLATIONS:
        for metric, _title, _ylabel in METRICS:
            value = _overall(rows, ablation, metric)
            pairs = (
                str(value["minimum_n"])
                if value["minimum_n"] == value["maximum_n"]
                else f'{value["minimum_n"]}–{value["maximum_n"]}'
            )
            lines.append(
                f"| {ABLATION_LABELS[ablation]} | {metric_labels[metric]} | "
                f"{_fmt(value['mean'])} | {value['positive']}/{value['cells']} | "
                f"{pairs} |"
            )

    lines.extend(
        [
            "",
            "## MAE interaction by shift",
            "",
            "Each mean below summarizes 3 model seeds × 4 deployment preferences. The sign counts refer to those 12 seed–preference cells.",
            "",
            "| Shift | No relational edges | Positive | No explicit successor | Positive |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for regime in regimes:
        values = []
        for ablation in ABLATIONS:
            selected = [
                item
                for item in rows
                if item["ablation"] == ablation
                and item["metric"] == "mean_absolute_error"
                and item["regime_id"] == regime
            ]
            values.extend(
                (
                    _fmt(fmean(float(item["mean"]) for item in selected)),
                    f"{sum(float(item['mean']) > 0 for item in selected)}/"
                    f"{len(selected)}",
                )
            )
        lines.append(
            f"| {REGIME_LABELS[regime].replace(chr(10), ' ')} | "
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} |"
        )

    lines.extend(
        [
            "",
            "## Incomplete rows",
            "",
            "| Representation | Seed | Lambda | Shift | Instance | Exact-SAFE selections | Reason |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for item in failures:
        lines.append(
            f"| {item['representation_variant']} | {item['model_seed']} | "
            f"{item['preference_lambda']:.2f} | {item['regime_id']} | "
            f"{item['instance_seed']} | "
            f"{'yes' if item['all_selected_candidates_exact_safe'] else 'no'} | "
            f"`{item['reason']}` |"
        )
    lines.extend(
        [
            "",
            "The three failures are strict-completion failures during delivery execution, not recorded violations of exact-SAFE candidate selection. Quality metrics for any affected 30-instance aggregate remain suppressed by the frozen E12 protocol.",
            "",
            "The interaction summaries are descriptive. Model-seed variation is displayed explicitly in the figure; no inferential interval is claimed from only three training seeds.",
        ]
    )
    return "\n".join(lines) + "\n"


def render(output: Path) -> dict:
    contract, report = _load_complete_report(output)
    rows = _interactions(report)
    failures = _failures(output, contract)
    if len(failures) != report["expected_rows"] - report["strict_safe_complete_rows"]:
        raise E12RenderError("failure ledger count does not match report")

    regimes = _regimes()
    groups = _cell_groups(rows)
    xpos = np.arange(len(regimes), dtype=float)
    offsets = {
        "nonrelational_successor": -0.16,
        "relational_current_candidate": 0.16,
    }
    seed_jitter = {0: -0.045, 1: 0.0, 2: 0.045}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": "#fbfbfa",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    for axis, (metric, title, ylabel) in zip(axes, METRICS):
        axis.axhline(
            0.0,
            color="#3f4852",
            linewidth=1.25,
            linestyle="--",
            zorder=1,
        )
        for ablation in ABLATIONS:
            seed_means = []
            for seed in program.MODEL_SEEDS:
                values = np.array(
                    [
                        fmean(groups[(ablation, metric, regime, seed)])
                        for regime in regimes
                    ]
                )
                seed_means.append(values)
                axis.scatter(
                    xpos + offsets[ablation] + seed_jitter[seed],
                    values,
                    s=25,
                    marker=MARKERS[ablation],
                    color=COLORS[ablation],
                    alpha=0.30,
                    edgecolor="none",
                    zorder=2,
                )
            means = np.mean(np.vstack(seed_means), axis=0)
            axis.plot(
                xpos + offsets[ablation],
                means,
                marker=MARKERS[ablation],
                markersize=7,
                linewidth=2.0,
                color=COLORS[ablation],
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )
        axis.set_xticks(xpos, [REGIME_LABELS[item] for item in regimes])
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=11, pad=9)
        axis.grid(True, color="#d9dee3", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    legend = [
        Line2D(
            [0],
            [0],
            color="#3f4852",
            linestyle="--",
            linewidth=1.25,
            label="Full VCG (interaction reference = 0)",
        )
    ] + [
        Line2D(
            [0],
            [0],
            color=COLORS[item],
            marker=MARKERS[item],
            linewidth=2,
            markersize=7,
            label=ABLATION_LABELS[item],
        )
        for item in ABLATIONS
    ]
    figure.legend(
        handles=legend,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
    )
    completion = _completion(report)
    figure.suptitle(
        "Representation contribution to performance under distribution shift",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.025,
        (
            "Interaction = ablation shift degradation − Full-VCG shift degradation; "
            "positive favors Full VCG. Light points: model-seed means "
            "over λ; bold lines: means over 3 seeds × 4 λ.  "
            f"Strict completion: full {completion['full_relational_successor'][0]}/"
            f"{completion['full_relational_successor'][1]}, no edges "
            f"{completion['nonrelational_successor'][0]}/"
            f"{completion['nonrelational_successor'][1]}, no successor "
            f"{completion['relational_current_candidate'][0]}/"
            f"{completion['relational_current_candidate'][1]}."
        ),
        ha="center",
        fontsize=8,
        color="#4c566a",
    )
    figure.tight_layout(rect=(0, 0.105, 1, 0.88), w_pad=2.0)

    pdf = output / f"{FIGURE_STEM}.pdf"
    png = output / f"{FIGURE_STEM}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(figure)

    table = output / TABLE_NAME
    table.write_text(_table(report, rows, failures), encoding="utf-8")
    result = {
        "schema_version": 1,
        "protocol": program.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "report_sha256": report["report_sha256"],
        "observed_rows": report["observed_rows"],
        "strict_safe_complete_rows": report["strict_safe_complete_rows"],
        "figure_pdf": str(pdf.relative_to(ROOT)),
        "figure_png": str(png.relative_to(ROOT)),
        "table": str(table.relative_to(ROOT)),
        "descriptive_seed_level_figure": True,
        "inferential_interval_claimed": False,
    }
    (output / DATA_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
