#!/usr/bin/env python3
"""Render the paired-outcome and rare-intervention E3 figures."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vcg-e03-mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

import run_vcg_conditioned_final_comparison_90k as final90
from experiments.conditioned_vcg.E03_certification_ablation_90k import run as e3


PROTOCOL = "vcg_conditioned_e03_standalone_summary_figures_v1"
BG = "#F7F6F2"
INK = "#26313A"
MUTED = "#67727C"
GRID = "#D7DDE1"
BOTH = "#9CBFD5"
CERT_WIN = "#159570"
PHYSICAL_WIN = "#EDA935"
NEITHER = "#C93642"
PHYSICAL = "#2968B9"
CERTIFIED = "#129170"
UNSAFE = "#E5A32D"
DANGER = "#C93642"


class SummaryFigureError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Mapping) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(project_root: Path, output_dir: Path) -> tuple[dict, list[dict]]:
    contract = e3.authenticate_contract(project_root, output_dir)
    report = e3._load_json(output_dir / e3.REPORT_NAME, label="E3 report")
    e3._verify_hash(report, "report_sha256", label="E3 report")
    if report.get("status") != "complete":
        raise SummaryFigureError("E3 report is not complete")
    manifest = final90.authenticate_manifest(project_root, final90.DEFAULT_OUTPUT)
    records = {int(row["seed"]): row for row in manifest["instances"]}
    rows = []
    for model_seed in e3.MODEL_SEEDS:
        for instance_seed in e3.INSTANCE_SEEDS:
            path = e3._ledger_path(output_dir, model_seed, instance_seed)
            ledger = e3._load_json(path, label="E3 physical ledger")
            e3._verify_hash(ledger, "ledger_sha256", label="E3 physical ledger")
            if ledger.get("contract_sha256") != contract["contract_sha256"]:
                raise SummaryFigureError("physical ledger contract binding changed")
            row = ledger["run"]
            e3._validate_physical_row(
                row,
                records[instance_seed],
                model_seed=model_seed,
            )
            rows.append(dict(row))
    if len(rows) != 90:
        raise SummaryFigureError("E3 physical grid is incomplete")
    physical_complete = sum(bool(row["strict_completion"]) for row in rows)
    certified_complete = int(
        report["summaries"][e3.CERTIFIED]["strict_completion"]["count"]
    )
    paired = report["paired_strict_completion"]
    if (
        physical_complete != 86
        or certified_complete != 90
        or int(paired["certified_success_physical_failure"]) != 4
        or int(paired["physical_success_certified_failure"]) != 0
    ):
        raise SummaryFigureError("E3 paired outcome counts changed")
    return report, rows


def _paired_category(row: Mapping) -> str:
    physical = bool(row["strict_completion"])
    certified = True
    if physical and certified:
        return "both"
    if certified:
        return "certified_only"
    if physical:
        return "physical_only"
    return "neither"


def _render_paired(rows: Sequence[Mapping], output_base: Path) -> dict:
    colors = {
        "both": BOTH,
        "certified_only": CERT_WIN,
        "physical_only": PHYSICAL_WIN,
        "neither": NEITHER,
    }
    counts = Counter(_paired_category(row) for row in rows)
    fig, ax = plt.subplots(figsize=(13.0, 3.8), facecolor=BG)
    ax.set_facecolor(BG)
    indexed = {(int(row["model_seed"]), int(row["instance_seed"])): row for row in rows}
    for seed_row, model_seed in enumerate(e3.MODEL_SEEDS):
        y = len(e3.MODEL_SEEDS) - seed_row - 1
        for column, instance_seed in enumerate(e3.INSTANCE_SEEDS):
            category = _paired_category(indexed[(model_seed, instance_seed)])
            ax.add_patch(
                Rectangle(
                    (column + 0.06, y + 0.06),
                    0.88,
                    0.88,
                    facecolor=colors[category],
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    tick_columns = (0, 4, 9, 14, 19, 24, 29)
    ax.set_xticks([value + 0.5 for value in tick_columns])
    ax.set_xticklabels([str(value + 1) for value in tick_columns], fontsize=10)
    ax.set_yticks((2.5, 1.5, 0.5))
    ax.set_yticklabels(("Model seed 0", "Model seed 1", "Model seed 2"), fontsize=11)
    ax.set_xlabel("Matched EpisodeInstance index", fontsize=11, color=INK, labelpad=7)
    ax.tick_params(axis="both", length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Paired strict-completion outcomes",
        x=0.12,
        y=0.96,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    legend = (
        Patch(facecolor=BOTH, label=f"Both complete  {counts['both']}/90"),
        Patch(
            facecolor=CERT_WIN,
            label=f"Certified only  {counts['certified_only']}/90",
        ),
        Patch(
            facecolor=PHYSICAL_WIN,
            label=f"Physical only  {counts['physical_only']}/90",
        ),
        Patch(facecolor=NEITHER, label=f"Neither  {counts['neither']}/90"),
    )
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.89),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.text(
        0.12,
        0.075,
        "Certified: 90/90 complete     Physical-only: 86/90 complete     "
        "All four certification gains occurred under model seed 0.",
        ha="left",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    fig.subplots_adjust(left=0.12, right=0.94, top=0.68, bottom=0.23)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=240, facecolor=BG)
    fig.savefig(pdf, facecolor=BG)
    plt.close(fig)
    return {
        "counts": dict(counts),
        "png": {"path": str(png.resolve()), "sha256": _sha256(png)},
        "pdf": {"path": str(pdf.resolve()), "sha256": _sha256(pdf)},
    }


def _box(
    ax,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    facecolor: str,
    textcolor: str = "white",
    fontsize: float = 12,
) -> None:
    x, y = center
    width, height = size
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=facecolor,
            edgecolor="white",
            linewidth=1.5,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=textcolor,
        linespacing=1.15,
    )


def _arrow(ax, start, end, *, color=MUTED, connectionstyle="arc3") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def _render_flow(rows: Sequence[Mapping], output_base: Path) -> dict:
    total = sum(int(row["shadow_audit"]["physical_candidates_exposed"]) for row in rows)
    unsafe = sum(int(row["shadow_audit"]["unsafe_candidates_exposed"]) for row in rows)
    selected = sum(int(row["shadow_audit"]["selection_divergences"]) for row in rows)
    self_blocking = sum(int(row["self_blocking"]) for row in rows)
    unsafe_admissions = sum(int(row["unsafe_admission"]) for row in rows)
    if (total, unsafe, selected, self_blocking, unsafe_admissions) != (
        27_298,
        4,
        4,
        4,
        4,
    ):
        raise SummaryFigureError("E3 candidate-flow counts changed")
    rare_rate = 100.0 * unsafe / total

    fig, ax = plt.subplots(figsize=(13.0, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "Rare verifier intervention, catastrophic consequence",
        x=0.055,
        y=0.96,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.055,
        0.88,
        "Same frozen critic and deterministic selection rule in both branches",
        ha="left",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    common_center = (0.14, 0.52)
    unsafe_center = (0.38, 0.52)
    _box(
        ax,
        common_center,
        (0.19, 0.22),
        f"{total:,}\nphysically feasible\ncandidates",
        facecolor=PHYSICAL,
    )
    _box(
        ax,
        unsafe_center,
        (0.16, 0.22),
        f"{unsafe}\nnot recoverability-safe\n({rare_rate:.4f}%)",
        facecolor=UNSAFE,
        textcolor=INK,
        fontsize=11.5,
    )
    _arrow(ax, (0.245, 0.52), (0.29, 0.52))

    selected_center = (0.62, 0.69)
    blocked_center = (0.86, 0.69)
    removed_center = (0.58, 0.29)
    zero_center = (0.76, 0.29)
    complete_center = (0.915, 0.29)
    _box(
        ax,
        selected_center,
        (0.19, 0.20),
        f"PHYSICAL ONLY\n{selected}/{unsafe} preferred by unchanged\n"
        "critic + selector",
        facecolor="#A74A72",
        fontsize=10.8,
    )
    _box(
        ax,
        blocked_center,
        (0.15, 0.20),
        f"{self_blocking}/{selected}\nself-blocked",
        facecolor=DANGER,
        fontsize=13,
    )
    _box(
        ax,
        removed_center,
        (0.17, 0.19),
        "CERTIFIED\nCertificate removes 4/4",
        facecolor=CERTIFIED,
        fontsize=11,
    )
    _box(
        ax,
        zero_center,
        (0.13, 0.19),
        "0 unsafe\nadmissions",
        facecolor="#4FA98C",
        fontsize=11.5,
    )
    _box(
        ax,
        complete_center,
        (0.11, 0.19),
        "90/90\ncomplete",
        facecolor="#247B60",
        fontsize=11.5,
    )
    _arrow(
        ax,
        (0.46, 0.56),
        (0.515, 0.65),
        color=PHYSICAL,
        connectionstyle="arc3,rad=-0.08",
    )
    _arrow(ax, (0.72, 0.69), (0.78, 0.69), color=DANGER)
    _arrow(
        ax,
        (0.46, 0.47),
        (0.505, 0.33),
        color=CERTIFIED,
        connectionstyle="arc3,rad=0.08",
    )
    _arrow(ax, (0.675, 0.29), (0.69, 0.29), color=CERTIFIED)
    _arrow(ax, (0.83, 0.29), (0.85, 0.29), color=CERTIFIED)
    ax.text(
        0.055,
        0.075,
        "The verifier removed only 4 of 27,298 exposed candidates—and prevented "
        "every observed catastrophic choice.",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.05)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=240, facecolor=BG)
    fig.savefig(pdf, facecolor=BG)
    plt.close(fig)
    return {
        "physically_feasible_candidates": total,
        "unsafe_candidates": unsafe,
        "unsafe_rate_percent": rare_rate,
        "unsafe_selected": selected,
        "self_blocking": self_blocking,
        "unsafe_admissions": unsafe_admissions,
        "png": {"path": str(png.resolve()), "sha256": _sha256(png)},
        "pdf": {"path": str(pdf.resolve()), "sha256": _sha256(pdf)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=e3.DEFAULT_OUTPUT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, rows = _load(project_root, output_dir)
    paired = _render_paired(rows, output_dir / "e03-a-paired-outcomes")
    flow = _render_flow(rows, output_dir / "e03-b-rare-candidate-flow")
    manifest = {
        "protocol": PROTOCOL,
        "source_report_sha256": report["report_sha256"],
        "figures": {"paired_outcomes": paired, "rare_candidate_flow": flow},
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path = output_dir / "e03-summary-figures.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "e03-a-paired-outcomes.pdf")
    print(output_dir / "e03-b-rare-candidate-flow.pdf")
    print(manifest_path)


if __name__ == "__main__":
    main()
