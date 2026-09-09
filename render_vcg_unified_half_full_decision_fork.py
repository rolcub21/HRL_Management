#!/usr/bin/env python3
"""Render a five-step, single-decision VCG mechanism fork.

This renderer consumes a probe produced by
``probe_vcg_unified_decision_mechanism.py``.  It never runs a model.  The
figure follows one reachable half-full state through the exact viability
gate, two critic heads, one shared stochastic draw, and one macro per arm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vcg-fork-mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import render_vcg_unified_decision_mechanism as common


BG = common.BG
CARD = common.CARD
INK = common.INK
MUTED = common.MUTED
SAFE = common.SAFE
BLUE = common.LAMBDA_ZERO
MAGENTA = common.LAMBDA_COST
DELIVER = common.ACTION_COLORS["deliver"]
HOLD = common.ACTION_COLORS["hold"]
RECONFIGURE = common.ACTION_COLORS["reconfigure"]
CELL_COLORS = {
    "wall": "#626D78",
    "storage": "#E7F0F8",
    "exit": "#CBEEDC",
    "pickup": "#FFE2A6",
    "waiting": "#DED2F2",
    "path": "#F6F6F3",
}


class ForkFigureError(ValueError):
    pass


def _candidate(probe: dict, key: str) -> dict:
    rows = [item for item in probe["selected_probe"]["candidates"] if item["key"] == key]
    if len(rows) != 1:
        raise ForkFigureError(f"candidate {key!r} is missing or duplicated")
    return rows[0]


def validate(probe: dict) -> dict:
    if probe.get("exactly_reproduces_persisted_confirmation_row") is not True:
        raise ForkFigureError("probe must reproduce its persisted confirmation row")
    if probe.get("probe_execution_device") != "cuda":
        raise ForkFigureError("paper fork must be extracted on the frozen CUDA device")
    selected = probe.get("selected_probe")
    if not isinstance(selected, dict) or selected.get("occupied_storage_cells") != 4:
        raise ForkFigureError("probe must contain a reachable 4/8 state")
    samples = selected.get("same_rng_counterfactual_samples", {})
    if set(samples) != {"lambda0", "lambda005"}:
        raise ForkFigureError("both same-RNG samples are required")
    if samples["lambda0"]["selected_key"] == samples["lambda005"]["selected_key"]:
        raise ForkFigureError("the selected state does not diverge under shared RNG")
    if samples["lambda0"].get("selected_action_type") != "deliver":
        raise ForkFigureError("lambda-zero illustration arm must Deliver")
    if samples["lambda005"].get("selected_action_type") != "defer":
        raise ForkFigureError("handling illustration arm must Hold")
    forks = selected.get("one_macro_counterfactual_forks")
    if not isinstance(forks, dict) or set(forks) != {"lambda0", "lambda005"}:
        raise ForkFigureError("one-macro successor evidence is missing")
    for label, arm in forks.items():
        if (
            arm.get("option_success") is not True
            or arm.get("failure_reason") is not None
            or arm.get("truncated") is not False
            or arm.get("illegal_drops") != 0
        ):
            raise ForkFigureError(f"{label} successor macro is not strict")
    deliver = _candidate(probe, samples["lambda0"]["selected_key"])
    hold = _candidate(probe, samples["lambda005"]["selected_key"])
    if not (
        float(deliver["q_operational"]) > float(hold["q_operational"])
        and float(deliver["q_predicted_physical_rehandles"])
        > float(hold["q_predicted_physical_rehandles"])
    ):
        raise ForkFigureError("selected candidates do not express the Qop/QN trade-off")
    return {
        "selected": selected,
        "samples": samples,
        "forks": forks,
        "deliver": deliver,
        "hold": hold,
    }


def _card(ax, title: str, number: int, subtitle: str = "") -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.014,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=CARD,
            edgecolor="#DFE3E7",
            linewidth=1.2,
            clip_on=False,
            zorder=-10,
        )
    )
    ax.text(
        0.045, 0.94, str(number), transform=ax.transAxes,
        ha="center", va="center", fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="circle,pad=0.34", facecolor=INK, edgecolor="none"),
    )
    ax.text(0.105, 0.945, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=12.2, fontweight="bold", color=INK)
    if subtitle:
        ax.text(0.105, 0.895, subtitle, transform=ax.transAxes, ha="left", va="center",
                fontsize=7.6, color=MUTED)


def _active_blocks(snapshot: dict) -> dict[tuple[int, int], list[str]]:
    result: dict[tuple[int, int], list[str]] = {}
    for block in snapshot["blocks"]:
        position = block.get("position")
        if position is None or block.get("delivered"):
            continue
        result.setdefault(tuple(position), []).append(str(block["label"]))
    return result


def _draw_yard(ax, snapshot: dict, *, x0: float, y0: float, size: float,
               highlight: str | None = None, caption: str = "") -> None:
    rooms = snapshot["rooms"]
    rows, cols = len(rooms), len(rooms[0])
    storage = {tuple(cell) for cell in snapshot["storage_positions"]}
    exits = {tuple(cell) for cell in snapshot["exit_cells"]}
    pickup = tuple(snapshot["pickup_cell"])
    waiting = tuple(snapshot["waiting_cell"])
    blocks = _active_blocks(snapshot)
    cell = size / max(rows, cols)
    for row in range(rows):
        for col in range(cols):
            position = (row, col)
            if rooms[row][col] == "#":
                kind = "wall"
            elif position in exits:
                kind = "exit"
            elif position == pickup:
                kind = "pickup"
            elif position == waiting:
                kind = "waiting"
            elif position in storage:
                kind = "storage"
            else:
                kind = "path"
            x = x0 + col * cell
            y = y0 + (rows - row - 1) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, transform=ax.transAxes,
                                   facecolor=CELL_COLORS[kind], edgecolor="#C9D1D9", lw=0.75))
            labels = blocks.get(position, [])
            if labels:
                label = ",".join(labels)
                color = "#9E4B21" if highlight and highlight in labels else INK
                ax.text(x + cell / 2, y + cell / 2, label, transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.4, fontweight="bold",
                        color=color)
            elif kind in {"pickup", "waiting", "exit"}:
                ax.text(x + cell / 2, y + cell / 2, kind.upper(), transform=ax.transAxes,
                        ha="center", va="center", fontsize=5.4, color=MUTED,
                        fontweight="bold")
    occupied = sum(1 for pos in blocks if pos in storage)
    ax.text(x0 + size / 2, y0 - 0.035, caption or f"storage {occupied}/8",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.3, color=MUTED)


def _placeholder(ax, number: int, label: str) -> None:
    _card(ax, label, number)
    ax.text(0.5, 0.48, "revealed next", transform=ax.transAxes,
            ha="center", va="center", color="#A0A6AC", fontsize=9)


def _draw_common_state(ax, data: dict) -> None:
    selected = data["selected"]
    _card(ax, "Common half-full state", 1, "Same yard, checkpoint, SAFE frontier, and RNG state")
    _draw_yard(ax, selected["physical_snapshot"], x0=0.13, y0=0.15, size=0.72,
               highlight="B4", caption="4 of 8 storage cells occupied · B4 due in 1 step")


def _draw_gate(ax, data: dict) -> None:
    selected = data["selected"]
    _card(ax, "Hard viability first", 2, "Learned values never admit an uncertified macro")
    ax.text(0.5, 0.70, "EXACT VERIFIER", transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.7", facecolor=INK, edgecolor="none"))
    ax.add_patch(FancyArrowPatch((0.5, 0.61), (0.5, 0.50), transform=ax.transAxes,
                                arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.4))
    ax.text(0.5, 0.42, f"{selected['safe_candidate_count']} exact-SAFE macros",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
            fontweight="bold", color=SAFE)
    ax.text(0.5, 0.31, "Both highlighted choices carry deterministic\nrecovery witnesses",
            transform=ax.transAxes, ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(0.5, 0.17, "Deliver B4  ✓ SAFE     Hold 2  ✓ SAFE",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=INK)


def _draw_values(ax, data: dict) -> None:
    _card(ax, "Two heads expose the trade-off", 3,
          "Qop: predicted operational value · QN: predicted future physical rehandles")
    deliver, hold = data["deliver"], data["hold"]
    rows = [
        ["Deliver B4", f"{deliver['q_operational']:.3f}",
         f"{deliver['q_predicted_physical_rehandles']:.3f}",
         f"{deliver['lambda0_merit']:.3f}", f"{deliver['lambda005_merit']:.3f}"],
        ["Hold 2", f"{hold['q_operational']:.3f}",
         f"{hold['q_predicted_physical_rehandles']:.3f}",
         f"{hold['lambda0_merit']:.3f}", f"{hold['lambda005_merit']:.3f}"],
    ]
    table = ax.table(cellText=rows,
                     colLabels=["SAFE macro", "Qop ↑", "QN ↓", "M(0)", "M(.05)"],
                     cellLoc="center", colLoc="center", bbox=[0.055, 0.37, 0.89, 0.40],
                     colWidths=[0.28, 0.15, 0.15, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.set_zorder(5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE2E7")
        cell.set_facecolor("#EEF1F4" if row == 0 else CARD)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
    ax.text(0.08, 0.23, "Operational pull", transform=ax.transAxes, ha="left", va="center",
            fontsize=8, color=MUTED)
    ax.text(0.36, 0.23, "Deliver B4", transform=ax.transAxes, ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=DELIVER)
    ax.text(0.62, 0.23, "Handling pull", transform=ax.transAxes, ha="left", va="center",
            fontsize=8, color=MUTED)
    ax.text(0.86, 0.23, "Hold 2", transform=ax.transAxes, ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=HOLD)


def _draw_probability_bar(ax, y: float, probabilities: dict, draw: float,
                          label: str, selected: str, color: str) -> None:
    order = ("deliver", "reconfigure", "hold")
    colors = {"deliver": DELIVER, "reconfigure": RECONFIGURE, "hold": HOLD}
    x0, width = 0.20, 0.70
    cursor = x0
    for mode in order:
        value = float(probabilities.get(mode, 0.0))
        if value <= 0:
            continue
        ax.add_patch(Rectangle((cursor, y), width * value, 0.13, transform=ax.transAxes,
                               facecolor=colors[mode], edgecolor="white", lw=0.7))
        if value > 0.07:
            ax.text(cursor + width * value / 2, y + 0.065,
                    f"{mode}\n{value:.1%}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.7, color="white",
                    fontweight="bold")
        cursor += width * value
    draw_x = x0 + width * draw
    ax.plot([draw_x, draw_x], [y - 0.025, y + 0.165], transform=ax.transAxes,
            color="#111111", lw=2.0, zorder=8)
    ax.text(draw_x, y + 0.18, f"u={draw:.3f}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7, fontweight="bold", color=INK)
    ax.text(0.04, y + 0.065, label, transform=ax.transAxes, ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=color)
    ax.text(0.90, y - 0.038, f"selected  →  {selected}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.6, fontweight="bold", color=color)


def _draw_draw(ax, data: dict) -> None:
    selected = data["selected"]
    _card(ax, "The same stochastic draw forks", 4,
          "Only lambda changes; the state, verifier, heads, and random draw stay fixed")
    draw = float(selected["shared_policy_rng_unit_draws"]["outer_mode_draw"])
    _draw_probability_bar(ax, 0.58,
                          selected["policies"]["lambda0"]["action_family_probabilities"],
                          draw, "λ = 0", "DELIVER", BLUE)
    _draw_probability_bar(ax, 0.29,
                          selected["policies"]["lambda005"]["action_family_probabilities"],
                          draw, "λ = .05", "HOLD", MAGENTA)
    ax.text(0.5, 0.105,
            "u falls inside Deliver under λ=0, but inside Hold after the handling penalty shifts mass",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=MUTED)


def _draw_outcomes(ax, data: dict) -> None:
    _card(ax, "One macro later", 5,
          "Immediate consequence of the controlled decision fork")
    left = data["forks"]["lambda0"]
    right = data["forks"]["lambda005"]
    _draw_yard(ax, left["post_action_snapshot"], x0=0.06, y0=0.20, size=0.36,
               caption="λ=0 · B4 delivered · storage 3/8")
    _draw_yard(ax, right["post_action_snapshot"], x0=0.58, y0=0.20, size=0.36,
               highlight="B4", caption="λ=.05 · Hold · storage 4/8")
    ax.text(0.24, 0.74, "DELIVER B4", transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=DELIVER)
    ax.text(0.76, 0.74, "HOLD 2 STEPS", transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=HOLD)
    ax.text(0.24, 0.11,
            f"duration {left['duration']} · immediate return {left['raw_environment_return']:+.2f}\n"
            f"delivery deviation {left['delivery_deviations'][0]:+.0f} · immediate rehandles {left['physical_rehandles']}",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.6, color=INK)
    ax.text(0.76, 0.11,
            f"duration {right['duration']} · immediate return {right['raw_environment_return']:+.2f}\n"
            f"no delivery · immediate rehandles {right['physical_rehandles']}",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.6, color=INK)


def render(probe: dict, output: Path, *, reveal: int = 5) -> None:
    data = validate(probe)
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    grid = fig.add_gridspec(2, 3, left=0.035, right=0.975, top=0.84, bottom=0.12,
                           width_ratios=[1.02, 1.12, 1.42], height_ratios=[1, 1],
                           wspace=0.075, hspace=0.10)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[:, 2])]
    drawers = (_draw_common_state, _draw_gate, _draw_values, _draw_draw, _draw_outcomes)
    labels = ("Common half-full state", "Exact viability", "Two critic heads",
              "Shared stochastic draw", "One-macro successors")
    for index, (ax, drawer, label) in enumerate(zip(axes, drawers, labels), start=1):
        if reveal >= index:
            drawer(ax, data)
        else:
            _placeholder(ax, index, label)
    fig.text(0.035, 0.95, "One state, one draw, two VCG decisions", ha="left", va="top",
             fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.905,
             "A controlled half-full decision fork isolates immediate operational value from predicted future handling cost.",
             ha="left", va="top", fontsize=11, color=MUTED)
    fig.text(0.035, 0.064,
             "Mechanism probe—not a performance estimate. QN predicts future physical rehandles; both displayed macros are exact-SAFE.",
             ha="left", va="center", fontsize=8.5, color=INK)
    fig.text(0.035, 0.035,
             "Performance evidence remains the prospective 720-row panel: λ=.05 changed rehandles by −0.94/100 and MAE by +0.26; all rows strict-safe.",
             ha="left", va="center", fontsize=7.7, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vcg-half-full-decision-fork")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--keep-stages", action="store_true")
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    validate(probe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{args.stem}.png"
    pdf = args.output_dir / f"{args.stem}.pdf"
    render(probe, png)
    render(probe, pdf)
    if not args.no_video:
        with tempfile.TemporaryDirectory(prefix="vcg-decision-fork-") as temp:
            stages = []
            for reveal in range(1, 6):
                path = Path(temp) / f"stage-{reveal}.png"
                render(probe, path, reveal=reveal)
                stages.append(path)
            common.write_video(
                stages, args.output_dir / f"{args.stem}.mp4",
                fps=24, hold=1.45, fade=0.28,
            )
            if args.keep_stages:
                import shutil
                for index, path in enumerate(stages, start=1):
                    shutil.copy2(path, args.output_dir / f"{args.stem}-stage-{index}.png")
    print(png)
    print(pdf)
    if not args.no_video:
        print(args.output_dir / f"{args.stem}.mp4")


if __name__ == "__main__":
    main()
