#!/usr/bin/env python3
"""Render a paper figure for the unified VCG decision mechanism.

The input is a *single-state counterfactual probe*: identical state, model
weights, and SAFE frontier, scored once with lambda=0 and once with
lambda=0.05.  This renderer does not run a model or reconstruct policy data.

Outputs:
  - vcg-mechanism.png: full-resolution raster figure
  - vcg-mechanism.pdf: vector paper figure
  - vcg-mechanism.mp4: short three-beat reveal animation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keep rendering self-contained on hosts where the default Matplotlib cache is
# read-only (for example, a sandboxed evaluation worker).
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vcg-mechanism-mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


BG = "#F4F3EE"
CARD = "#FFFFFF"
INK = "#17202A"
MUTED = "#5D6873"
GRID = "#D5D9DE"
SAFE = "#16845B"
UNKNOWN = "#D48A13"
UNSAFE = "#B84A4A"
LAMBDA_ZERO = "#276FBF"
LAMBDA_COST = "#C44E84"
ACTION_COLORS = {
    "accept": "#3B82C4",
    "deliver": "#138A72",
    "reconfigure": "#E07A26",
    "hold": "#7C5BB4",
}
YARD_KIND_COLORS = {
    "wall": "#59636D",
    "exit": "#CDEEDC",
    "pickup": "#FFE2A8",
    "waiting": "#DED2F2",
    "path": "#EEEDE8",
}


class ProbeError(ValueError):
    """Raised when a probe would support a misleading figure."""


@dataclass(frozen=True)
class Condition:
    lambda_value: float
    label: str
    modes: dict[str, float]
    marginal: dict[str, float]
    conditional: dict[str, float]


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ProbeError(f"{field} must be finite")
    return result


def _parse_condition(raw: Mapping[str, Any], index: int) -> Condition:
    prefix = f"policy.conditions[{index}]"
    modes: dict[str, float] = {}
    for item in raw.get("mode_probabilities", []):
        mode = str(item["mode"])
        if mode in modes:
            raise ProbeError(f"{prefix} duplicates mode {mode!r}")
        modes[mode] = _as_float(item["probability"], f"{prefix}.{mode}")

    marginal: dict[str, float] = {}
    conditional: dict[str, float] = {}
    for item in raw.get("candidate_probabilities", []):
        candidate_id = str(item["candidate_id"])
        if candidate_id in marginal:
            raise ProbeError(f"{prefix} duplicates candidate {candidate_id!r}")
        marginal[candidate_id] = _as_float(
            item["marginal_probability"], f"{prefix}.{candidate_id}.marginal"
        )
        conditional[candidate_id] = _as_float(
            item["conditional_probability"],
            f"{prefix}.{candidate_id}.conditional",
        )
    return Condition(
        lambda_value=_as_float(raw["lambda"], f"{prefix}.lambda"),
        label=str(raw.get("label", f"lambda={raw['lambda']}")),
        modes=modes,
        marginal=marginal,
        conditional=conditional,
    )


def validate_probe(probe: Mapping[str, Any]) -> tuple[Condition, Condition]:
    """Validate the scientific invariants required by this illustration."""
    if probe.get("schema_version") != "vcg-single-state-probe/v1":
        raise ProbeError("schema_version must be 'vcg-single-state-probe/v1'")

    policy = probe.get("policy")
    if not isinstance(policy, Mapping):
        raise ProbeError("policy must be an object")
    if policy.get("type") != "nested_stochastic_softmax":
        raise ProbeError("policy.type must be 'nested_stochastic_softmax'")
    if policy.get("same_safe_set") is not True:
        raise ProbeError("policy.same_safe_set must be true")

    candidates = policy.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProbeError("policy.candidates must be a non-empty list")
    candidate_ids = [str(item["id"]) for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ProbeError("policy.candidates contains duplicate ids")
    candidate_modes = {str(item["id"]): str(item["mode"]) for item in candidates}

    viability = probe.get("viability", {})
    if int(viability.get("safe_exposed_count", -1)) != len(candidates):
        raise ProbeError(
            "viability.safe_exposed_count must equal the full candidate-list length"
        )

    raw_conditions = policy.get("conditions")
    if not isinstance(raw_conditions, list) or len(raw_conditions) != 2:
        raise ProbeError("policy.conditions must contain exactly two conditions")
    parsed = sorted(
        (_parse_condition(item, i) for i, item in enumerate(raw_conditions)),
        key=lambda item: item.lambda_value,
    )
    zero, cost = parsed
    if not math.isclose(zero.lambda_value, 0.0, abs_tol=1e-10):
        raise ProbeError("one policy condition must have lambda=0")
    if not math.isclose(cost.lambda_value, 0.05, abs_tol=1e-8):
        raise ProbeError("one policy condition must have lambda=0.05")

    expected_ids = set(candidate_ids)
    expected_modes = set(candidate_modes.values())
    tolerance = 2e-4
    for condition in parsed:
        if set(condition.marginal) != expected_ids:
            raise ProbeError(
                f"{condition.label}: candidate probabilities do not cover the SAFE set"
            )
        if set(condition.modes) != expected_modes:
            raise ProbeError(
                f"{condition.label}: mode probabilities do not cover candidate modes"
            )
        for label, values in (
            ("mode", condition.modes), ("candidate", condition.marginal)
        ):
            if any(value < -tolerance or value > 1.0 + tolerance for value in values.values()):
                raise ProbeError(f"{condition.label}: invalid {label} probability")
            if not math.isclose(sum(values.values()), 1.0, abs_tol=tolerance):
                raise ProbeError(f"{condition.label}: {label} probabilities do not sum to 1")
        for candidate_id, mode in candidate_modes.items():
            factorized = condition.modes[mode] * condition.conditional[candidate_id]
            if not math.isclose(
                factorized, condition.marginal[candidate_id], abs_tol=tolerance
            ):
                raise ProbeError(
                    f"{condition.label}: P({candidate_id}) != P({mode}) "
                    "* P(candidate|mode)"
                )
        for mode in expected_modes:
            within_sum = sum(
                condition.conditional[candidate_id]
                for candidate_id, candidate_mode in candidate_modes.items()
                if candidate_mode == mode
            )
            if not math.isclose(within_sum, 1.0, abs_tol=tolerance):
                raise ProbeError(
                    f"{condition.label}: conditional probabilities in mode {mode!r} "
                    "do not sum to 1"
                )
    return zero, cost


def _card(ax: Axes) -> None:
    ax.set_facecolor(CARD)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    patch = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor=CARD,
        edgecolor="#E1E3E6",
        linewidth=1.2,
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(patch)


def _panel_title(ax: Axes, number: str, title: str, subtitle: str) -> None:
    ax.text(
        0.035,
        0.965,
        number,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.35", facecolor=INK, edgecolor="none"),
    )
    ax.text(
        0.12,
        0.975,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.12,
        0.93,
        subtitle,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        color=MUTED,
    )


def _block_label(block: Any) -> str:
    if isinstance(block, Mapping):
        return str(block.get("label", block.get("id", "?")))
    return str(block)


def _candidate_subset(
    candidates: Sequence[Mapping[str, Any]],
    zero: Condition,
    cost: Condition,
    limit: int,
) -> list[Mapping[str, Any]]:
    def rank(item: Mapping[str, Any]) -> tuple[float, float, str]:
        candidate_id = str(item["id"])
        priority = _as_float(item.get("figure_priority", 0), "figure_priority")
        shift = abs(cost.marginal[candidate_id] - zero.marginal[candidate_id])
        return (priority, shift, candidate_id)

    return sorted(candidates, key=rank, reverse=True)[:limit]


def _cell_center(rows: int, row: int, col: int) -> tuple[float, float]:
    return col + 0.5, rows - row - 0.5


def draw_yard(
    ax: Axes,
    probe: Mapping[str, Any],
    zero: Condition,
    cost: Condition,
) -> None:
    _card(ax)
    _panel_title(ax, "1", "One reachable yard state", "Environment supplies the state; it is held fixed.")
    state = probe["state"]
    yard = state["yard"]
    rows, cols = int(yard["rows"]), int(yard["cols"])
    capacity = int(yard.get("stack_capacity", 1))
    ax.set_xlim(-0.65, cols + 0.65)
    ax.set_ylim(-0.6, rows + 0.8)
    ax.set_aspect("equal", adjustable="box")

    cells = {(int(c["row"]), int(c["col"])): c for c in yard.get("cells", [])}
    for row in range(rows):
        for col in range(cols):
            x, y = col, rows - row - 1
            cell = cells.get((row, col), {})
            blocks = cell.get("blocks", [])
            occupancy = min(len(blocks) / max(capacity, 1), 1.0)
            kind = str(cell.get("kind", yard.get("default_kind", "storage"))).lower()
            fill = (
                plt.cm.Blues(0.07 + 0.24 * occupancy)
                if kind == "storage"
                else YARD_KIND_COLORS.get(kind, "#F5F5F2")
            )
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=fill, edgecolor=GRID, linewidth=1)
            )
            if kind == "storage" or blocks:
                ax.text(
                    x + 0.09,
                    y + 0.12,
                    f"{len(blocks)}/{capacity}",
                    ha="left",
                    va="bottom",
                    fontsize=6.6,
                    color="white" if kind == "wall" else MUTED,
                )
            if blocks:
                ax.text(
                    x + 0.5,
                    y + 0.58,
                    _block_label(blocks[-1]),
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    fontweight="bold",
                    color="white" if kind == "wall" else INK,
                )
            elif kind != "storage":
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    kind.upper(),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    fontweight="bold",
                    color="white" if kind == "wall" else MUTED,
                )

    candidates = probe["policy"]["candidates"]
    displayed = [item for item in candidates if item.get("show_arrow") is True]
    if not displayed:
        displayed = [
            item
            for item in _candidate_subset(candidates, zero, cost, 3)
            if item.get("geometry")
        ]
    for index, candidate in enumerate(displayed[:4], start=1):
        geometry = candidate.get("geometry") or {}
        kind = str(geometry.get("kind", "none")).lower()
        mode = str(candidate["mode"])
        color = ACTION_COLORS.get(mode.lower(), ACTION_COLORS.get(kind, "#555555"))
        start: tuple[float, float] | None = None
        end: tuple[float, float] | None = None
        if geometry.get("from") is not None:
            row, col = map(int, geometry["from"])
            start = _cell_center(rows, row, col)
        if geometry.get("to") is not None:
            row, col = map(int, geometry["to"])
            end = _cell_center(rows, row, col)
        if kind == "accept" and end is not None:
            start = (-0.45, end[1])
        elif kind == "hold":
            center = start or end
            if center is not None:
                ax.text(
                    center[0],
                    center[1],
                    f"C{index}\nHOLD",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.32", facecolor=color, edgecolor="white"),
                    zorder=8,
                )
            continue
        if start is None or end is None:
            continue
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2.3,
                color=color,
                connectionstyle=f"arc3,rad={0.06 * (index - 2)}",
                zorder=7,
            )
        )
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            midpoint[0],
            midpoint[1] + 0.12,
            f"C{index}",
            fontsize=7,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="none"),
            zorder=9,
        )

    inbound = yard.get("inbound_queue", [])
    if inbound:
        ax.text(
            -0.52,
            rows + 0.18,
            "Inbound: " + ", ".join(map(str, inbound[:3])),
            ha="left",
            va="center",
            fontsize=7.2,
            color=MUTED,
        )
    state_label = (
        f"instance {state.get('instance_id', '?')} · decision {state.get('decision_index', '?')}"
        f" · t={state.get('sim_time', '?')}"
    )
    ax.text(0.5, -0.045, state_label, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)


def _count_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    count: str,
    title: str,
    detail: str,
    color: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color + "18",
            edgecolor=color,
            linewidth=1.4,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.06, y + height * 0.62, count, transform=ax.transAxes, fontsize=18, fontweight="bold", color=color, va="center")
    ax.text(x + 0.22, y + height * 0.66, title, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=INK, va="center")
    ax.text(x + 0.22, y + height * 0.34, detail, transform=ax.transAxes, fontsize=7.2, color=MUTED, va="center")


def draw_gate(ax: Axes, probe: Mapping[str, Any]) -> None:
    _card(ax)
    _panel_title(ax, "2", "Exact viability gate", "A hard filter runs before learned preference.")
    viability = probe["viability"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(
        FancyBboxPatch(
            (0.18, 0.68),
            0.64,
            0.13,
            boxstyle="round,pad=0.025,rounding_size=0.04",
            transform=ax.transAxes,
            facecolor=INK,
            edgecolor="none",
        )
    )
    ax.text(0.5, 0.748, "EXACT VIABILITY VERIFIER", transform=ax.transAxes, color="white", fontsize=10.2, fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.702, "certificates determine what may be done", transform=ax.transAxes, color="#DDE3E8", fontsize=7.1, ha="center", va="center")

    ax.annotate("", xy=(0.5, 0.61), xytext=(0.5, 0.68), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    _count_box(
        ax,
        0.09,
        0.43,
        0.82,
        0.15,
        str(viability["safe_exposed_count"]),
        "SAFE candidates exposed",
        "identical set for both lambda conditions",
        SAFE,
    )
    unknown = int(viability.get("fail_closed_unknown_excluded_count", 0))
    unsafe = viability.get("proven_unsafe_excluded_count")
    unsafe = 0 if unsafe is None else int(unsafe)
    if unknown or unsafe:
        _count_box(
            ax,
            0.09,
            0.24,
            0.82,
            0.15,
            str(unknown + unsafe),
            "UNSAFE / UNKNOWN excluded",
            "UNKNOWN is excluded fail-closed, not proven unsafe",
            UNKNOWN,
        )
    else:
        ax.text(
            0.5,
            0.315,
            "This state: no candidate required rejection",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.1,
            color=MUTED,
        )
        ax.text(
            0.5,
            0.27,
            "Only certified SAFE candidates reach the learned policy",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color=SAFE,
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.14, 0.075),
            0.72,
            0.075,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            transform=ax.transAxes,
            facecolor="#E9F6F0",
            edgecolor=SAFE,
            linewidth=1.1,
        )
    )
    ax.text(0.5, 0.112, "lambda never bypasses this gate", transform=ax.transAxes, ha="center", va="center", fontsize=8.4, fontweight="bold", color=SAFE)


def _bar_pairs(
    ax: Axes,
    labels: Sequence[str],
    values_zero: Sequence[float],
    values_cost: Sequence[float],
    label_zero: str,
    label_cost: str,
) -> None:
    y = np.arange(len(labels))
    height = 0.32
    ax.barh(y + height / 2, values_zero, height=height, color=LAMBDA_ZERO, label=label_zero)
    ax.barh(y - height / 2, values_cost, height=height, color=LAMBDA_COST, label=label_cost)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max([*values_zero, *values_cost], default=1.0) * 1.12))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", color="#E5E7EA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", labelsize=7.2, colors=MUTED, length=0)


def draw_policy(
    container: Any,
    probe: Mapping[str, Any],
    zero: Condition,
    cost: Condition,
) -> None:
    sub = container.subgridspec(4, 1, height_ratios=[0.34, 1.05, 1.15, 1.25], hspace=0.48)
    title_ax = plt.gcf().add_subplot(sub[0])
    mode_ax = plt.gcf().add_subplot(sub[1])
    candidate_ax = plt.gcf().add_subplot(sub[2])
    table_ax = plt.gcf().add_subplot(sub[3])
    for ax in (title_ax, mode_ax, candidate_ax, table_ax):
        ax.set_facecolor(CARD)
    title_ax.axis("off")
    title_ax.text(0.0, 0.84, "3", va="top", ha="left", fontsize=10, fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.35", facecolor=INK, edgecolor="none"))
    title_ax.text(0.085, 0.92, "Soft handling preference", va="top", ha="left", fontsize=14, fontweight="bold", color=INK)
    title_ax.text(0.085, 0.43, "Nested stochastic policy on the same SAFE frontier", va="top", ha="left", fontsize=8.3, color=MUTED)

    mode_order = sorted(zero.modes, key=lambda mode: max(zero.modes[mode], cost.modes[mode]), reverse=True)
    _bar_pairs(
        mode_ax,
        mode_order,
        [zero.modes[mode] for mode in mode_order],
        [cost.modes[mode] for mode in mode_order],
        zero.label,
        cost.label,
    )
    mode_ax.set_title("Outer draw: P(mode)", loc="left", fontsize=8.7, fontweight="bold", color=INK, pad=4)
    mode_ax.legend(frameon=False, fontsize=7.0, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.01))

    candidates = probe["policy"]["candidates"]
    selected = _candidate_subset(candidates, zero, cost, 4)
    labels = [str(item.get("short_label", item["id"])) for item in selected]
    ids = [str(item["id"]) for item in selected]
    _bar_pairs(
        candidate_ax,
        labels,
        [zero.marginal[candidate_id] for candidate_id in ids],
        [cost.marginal[candidate_id] for candidate_id in ids],
        zero.label,
        cost.label,
    )
    candidate_ax.set_title("Marginal candidate draw: P(c) = P(mode) x P(c | mode)", loc="left", fontsize=8.7, fontweight="bold", color=INK, pad=4)

    table_ax.axis("off")
    op_weight = _as_float(probe["policy"].get("operational_weight", 1.0), "operational_weight")
    rows: list[list[str]] = []
    for item in selected:
        q_op = _as_float(item["q_op"], f"{item['id']}.q_op")
        q_n = _as_float(item["q_n"], f"{item['id']}.q_n")
        merit_zero = op_weight * q_op - zero.lambda_value * q_n
        merit_cost = op_weight * q_op - cost.lambda_value * q_n
        rows.append(
            [
                str(item.get("short_label", item["id"])),
                f"{q_op:.3f}",
                f"{q_n:.3f}",
                f"{merit_zero:.3f}",
                f"{merit_cost:.3f}",
            ]
        )
    table = table_ax.table(
        cellText=rows,
        colLabels=["SAFE c", "Qop", "QN", "M(0)", "M(.05)"],
        colWidths=[0.34, 0.14, 0.14, 0.19, 0.19],
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.06, 1, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#E1E4E8")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#EEF1F4" if row == 0 else CARD)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
    table_ax.set_title("Shared heads; merit M_lambda = w_op Qop - lambda QN", loc="left", fontsize=8.7, fontweight="bold", color=INK, pad=0)
    table_ax.text(0, -0.02, "QN estimates future physical rehandles; it is not an immediate-move label.", transform=table_ax.transAxes, fontsize=7.0, color=MUTED, ha="left", va="top")


def _placeholder(ax: Axes, number: str, label: str) -> None:
    _card(ax)
    ax.set_facecolor("#ECEDEA")
    ax.text(0.5, 0.54, number, transform=ax.transAxes, ha="center", va="center", fontsize=28, color="#B5BABF", fontweight="bold")
    ax.text(0.5, 0.43, label, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#8A9298")


def render_figure(
    probe: Mapping[str, Any],
    zero: Condition,
    cost: Condition,
    output: Path,
    reveal_stage: int = 3,
) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.18, 0.86, 1.48],
        left=0.035,
        right=0.975,
        top=0.84,
        bottom=0.115,
        wspace=0.075,
    )
    yard_ax = fig.add_subplot(outer[0])
    gate_ax = fig.add_subplot(outer[1])

    title = str(probe.get("title", "VCG: hard viability, soft handling preference"))
    fig.text(0.035, 0.95, title, ha="left", va="top", fontsize=23, fontweight="bold", color=INK)
    fig.text(
        0.035,
        0.905,
        "One state and one SAFE frontier; lambda changes sampling probabilities, not admissibility.",
        ha="left",
        va="top",
        fontsize=11,
        color=MUTED,
    )

    draw_yard(yard_ax, probe, zero, cost)
    if reveal_stage >= 2:
        draw_gate(gate_ax, probe)
    else:
        _placeholder(gate_ax, "2", "exact viability gate")

    if reveal_stage >= 3:
        draw_policy(outer[2], probe, zero, cost)
    else:
        placeholder = fig.add_subplot(outer[2])
        _placeholder(placeholder, "3", "nested stochastic preference")

    provenance = probe.get("provenance", {})
    auth_status = str(provenance.get("status", "unspecified"))
    footer = (
        "Interpretation: SAFE is certified; UNKNOWN is conservatively excluded but not thereby proven unsafe. "
        "Bars are probabilities, not deterministic argmax choices."
    )
    fig.text(0.035, 0.052, footer, ha="left", va="center", fontsize=8.5, color=INK)
    source = str(provenance.get("source", "source not recorded"))
    digest = str(probe.get("viability", {}).get("safe_set_digest", "digest not recorded"))
    fig.text(0.035, 0.026, f"Probe: {auth_status} · {source} · SAFE digest: {digest}", ha="left", va="center", fontsize=7.2, color=MUTED)

    if auth_status != "authenticated_replay":
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC LAYOUT TEST — NOT EXPERIMENTAL EVIDENCE",
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            color="#B84A4A",
            alpha=0.16,
            rotation=18,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG, bbox_inches=None)
    plt.close(fig)


def write_video(stage_paths: Sequence[Path], output: Path, fps: int, hold: float, fade: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP4 output; use --no-video to skip it")
    images = [np.asarray(Image.open(path).convert("RGB")) for path in stage_paths]
    if any(image.shape != images[0].shape for image in images[1:]):
        raise RuntimeError("reveal frames have inconsistent sizes")
    height, width, _ = images[0].shape
    if width % 2 or height % 2:
        raise RuntimeError("video dimensions must be even")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    hold_frames = max(1, round(hold * fps))
    fade_frames = max(1, round(fade * fps))
    try:
        for index, image in enumerate(images):
            for _ in range(hold_frames):
                process.stdin.write(image.tobytes())
            if index + 1 < len(images):
                following = images[index + 1]
                for step in range(1, fade_frames + 1):
                    alpha = step / fade_frames
                    blended = np.rint((1 - alpha) * image + alpha * following).astype(np.uint8)
                    process.stdin.write(blended.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True, help="single-state probe JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vcg-mechanism")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--keep-stages", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold-seconds", type=float, default=1.6)
    parser.add_argument("--fade-seconds", type=float, default=0.35)
    args = parser.parse_args(argv)

    with args.probe.open("r", encoding="utf-8") as handle:
        probe = json.load(handle)
    zero, cost = validate_probe(probe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    render_figure(probe, zero, cost, png_path, reveal_stage=3)
    render_figure(probe, zero, cost, pdf_path, reveal_stage=3)

    if not args.no_video:
        with tempfile.TemporaryDirectory(prefix="vcg-figure-") as temp_dir:
            temp_root = Path(temp_dir)
            stage_paths: list[Path] = []
            for stage in (1, 2, 3):
                stage_path = temp_root / f"stage-{stage}.png"
                render_figure(probe, zero, cost, stage_path, reveal_stage=stage)
                stage_paths.append(stage_path)
            write_video(
                stage_paths,
                args.output_dir / f"{args.stem}.mp4",
                fps=args.fps,
                hold=args.hold_seconds,
                fade=args.fade_seconds,
            )
            if args.keep_stages:
                for stage, stage_path in enumerate(stage_paths, start=1):
                    shutil.copy2(stage_path, args.output_dir / f"{args.stem}-stage-{stage}.png")

    print(png_path)
    print(pdf_path)
    if not args.no_video:
        print(args.output_dir / f"{args.stem}.mp4")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProbeError, KeyError, TypeError) as exc:
        print(f"probe error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
