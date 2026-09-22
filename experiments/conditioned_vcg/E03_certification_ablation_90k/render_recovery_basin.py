#!/usr/bin/env python3
"""Render the exact compact 5x5 closed-admission recovery basin."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "results/viability-kernel-5x5-all-goal.json"
DEFAULT_OUTPUT = ROOT / "results/vcg-compact-recovery-basin-5x5"
SOURCE_PROTOCOL = "exhaustive_compact_recovery_kernel_post_transfer_states_v1"
REPORT_PROTOCOL = "vcg_compact_recovery_basin_5x5_figure_v1"
FIGURE_STEM = "vcg-compact-recovery-basin-5x5"
AGGREGATE_FIGURE_STEM = "vcg-compact-recovery-basin-5x5-aggregate"

SAFE_COLOR = "#198754"
UNSAFE_COLOR = "#C73E4D"
WALL_COLOR = "#596873"
STORAGE_COLOR = "#E4EEF4"
BLOCK_COLOR = "#90AFC0"
BLOCK_EDGE = "#466575"
PICKUP_COLOR = "#F5CF70"
WAIT_COLOR = "#D9CBEF"
EXIT_COLOR = "#BFE7D2"


class RecoveryBasinFigureError(RuntimeError):
    """Raised when the frozen recovery-basin input is inconsistent."""


def _canonical_hash(value: Any, hash_field: str | None = None) -> str:
    payload = dict(value) if isinstance(value, Mapping) else value
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryBasinFigureError(f"cannot read recovery-basin source: {path}") from exc
    if payload.get("protocol") != SOURCE_PROTOCOL:
        raise RecoveryBasinFigureError("unexpected recovery-basin protocol")
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        raise RecoveryBasinFigureError("source has no geometry")
    if (geometry.get("grid_rows"), geometry.get("grid_cols")) != (5, 5):
        raise RecoveryBasinFigureError("figure requires the frozen 5x5 geometry")
    storage = tuple(tuple(cell) for cell in geometry.get("storage_cells", ()))
    if len(storage) != 8 or geometry.get("storage_cell_count") != 8:
        raise RecoveryBasinFigureError("unexpected usable-storage geometry")
    states = payload.get("states")
    if not isinstance(states, list) or len(states) != payload.get("state_count"):
        raise RecoveryBasinFigureError("source does not contain its declared state set")
    observed = set()
    status_counts: dict[str, int] = defaultdict(int)
    for row in states:
        key = (int(row["occupancy_mask"]), tuple(row["agent_position"]))
        if key in observed:
            raise RecoveryBasinFigureError(f"duplicate compact state: {key}")
        observed.add(key)
        mask = int(row["occupancy_mask"])
        if mask.bit_count() != int(row["occupancy_count"]):
            raise RecoveryBasinFigureError("occupancy mask/count mismatch")
        status = str(row["status"])
        if status not in {"SAFE", "UNSAFE"}:
            raise RecoveryBasinFigureError("frozen exact basin contains UNKNOWN state")
        status_counts[status] += 1
        if status == "SAFE":
            steps = row.get("witness_primitive_steps")
            if not isinstance(steps, int) or steps < 0:
                raise RecoveryBasinFigureError("SAFE state lacks a finite witness")
        elif row.get("witness_primitive_steps") is not None:
            raise RecoveryBasinFigureError("UNSAFE state unexpectedly has a witness")
    if dict(status_counts) != payload.get("status_counts"):
        raise RecoveryBasinFigureError("status counts disagree with frozen source")
    if len(states) != 1280 or status_counts != {"SAFE": 1261, "UNSAFE": 19}:
        raise RecoveryBasinFigureError("frozen 5x5 basin identity changed")
    return payload


def _position_label(position: tuple[int, int], exits: set[tuple[int, int]]) -> str:
    if position in exits:
        return "Exit"
    return f"({position[0]},{position[1]})"


def _static_exit_distances(geometry: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    storage = {tuple(cell) for cell in geometry["storage_cells"]}
    exits = {tuple(cell) for cell in geometry["exit_cells"]}
    usable = storage | exits | {
        tuple(geometry["pickup_cell"]),
        tuple(geometry["waiting_cell"]),
    }
    distances = {cell: 0 for cell in exits}
    queue = deque((cell, 0) for cell in sorted(exits))
    while queue:
        (row, col), distance = queue.popleft()
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + delta_row, col + delta_col)
            if neighbor in usable and neighbor not in distances:
                distances[neighbor] = distance + 1
                queue.append((neighbor, distance + 1))
    required = storage | exits
    if not required.issubset(distances):
        raise RecoveryBasinFigureError("usable-cell graph is disconnected from every exit")
    return {position: distances[position] for position in sorted(required)}


def build_summary(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    geometry = payload["geometry"]
    storage = tuple(tuple(cell) for cell in geometry["storage_cells"])
    exits = tuple(tuple(cell) for cell in geometry["exit_cells"])
    positions = storage + tuple(cell for cell in exits if cell not in storage)
    position_set = set(positions)
    cells: dict[tuple[int, tuple[int, int]], list[Mapping[str, Any]]] = defaultdict(list)
    occupancy_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload["states"]:
        position = tuple(row["agent_position"])
        if position not in position_set:
            raise RecoveryBasinFigureError(f"unexpected agent position: {position}")
        occupancy = int(row["occupancy_count"])
        cells[(occupancy, position)].append(row)
        occupancy_groups[occupancy].append(row)

    occupancy_rows = []
    for occupancy in range(9):
        group = occupancy_groups[occupancy]
        safe = sum(row["status"] == "SAFE" for row in group)
        unsafe = len(group) - safe
        occupancy_rows.append(
            {
                "occupancy": occupancy,
                "state_count": len(group),
                "safe_count": safe,
                "unsafe_count": unsafe,
                "safe_fraction": safe / len(group),
            }
        )

    landscape_rows = []
    for position in positions:
        for occupancy in range(9):
            group = cells.get((occupancy, position), [])
            if not group:
                continue
            safe_rows = [row for row in group if row["status"] == "SAFE"]
            landscape_rows.append(
                {
                    "occupancy": occupancy,
                    "agent_position": list(position),
                    "agent_position_label": _position_label(position, set(exits)),
                    "state_count": len(group),
                    "safe_count": len(safe_rows),
                    "unsafe_count": len(group) - len(safe_rows),
                    "safe_fraction": len(safe_rows) / len(group),
                    "mean_safe_witness_primitive_steps": (
                        float(fmean(row["witness_primitive_steps"] for row in safe_rows))
                        if safe_rows
                        else None
                    ),
                }
            )

    full_rows = [
        {
            "agent_position": list(row["agent_position"]),
            "status": row["status"],
            "witness_primitive_steps": row["witness_primitive_steps"],
        }
        for row in payload["states"]
        if int(row["occupancy_count"]) == 8
    ]
    full_rows.sort(key=lambda row: tuple(row["agent_position"]))
    if len(full_rows) != 9 or sum(row["status"] == "SAFE" for row in full_rows) != 4:
        raise RecoveryBasinFigureError("unexpected full-occupancy boundary slice")

    exit_distances = _static_exit_distances(geometry)
    distance_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload["states"]:
        key = (
            int(row["occupancy_count"]),
            int(exit_distances[tuple(row["agent_position"])]),
        )
        distance_groups[key].append(row)
    exit_distance_rows = []
    for occupancy in range(9):
        for distance in sorted(set(exit_distances.values())):
            group = distance_groups.get((occupancy, distance), [])
            if not group:
                continue
            safe = sum(row["status"] == "SAFE" for row in group)
            exit_distance_rows.append(
                {
                    "occupancy": occupancy,
                    "static_exit_distance": distance,
                    "state_count": len(group),
                    "safe_count": safe,
                    "unsafe_count": len(group) - safe,
                    "safe_fraction": safe / len(group),
                }
            )

    witness_step_summaries = []
    for occupancy in range(9):
        values = sorted(
            int(row["witness_primitive_steps"])
            for row in occupancy_groups[occupancy]
            if row["status"] == "SAFE"
        )
        if not values:
            continue
        quantiles = np.quantile(values, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
        witness_step_summaries.append(
            {
                "occupancy": occupancy,
                "safe_state_count": len(values),
                "minimum": float(quantiles[0]),
                "q10": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "median": float(quantiles[3]),
                "q75": float(quantiles[4]),
                "q90": float(quantiles[5]),
                "maximum": float(quantiles[6]),
            }
        )

    report = {
        "schema_version": 1,
        "protocol": REPORT_PROTOCOL,
        "status": "complete",
        "analysis_type": "exact_reanalysis_of_frozen_enumerated_states",
        "training_runs": 0,
        "evaluation_runs": 0,
        "source": {
            "path": str(source.resolve()),
            "sha256": _file_sha256(source),
            "protocol": payload["protocol"],
        },
        "claim_scope": (
            "exact_compact_5x5_closed_admission_post_transfer_recovery_basin_"
            "without_future_schedule"
        ),
        "not_claimed": [
            "full_dynamic_episode_viability_kernel",
            "future_arrival_robustness",
            "timing_aware_recoverability",
            "recursive_disturbance_robust_kernel",
        ],
        "state_count": len(payload["states"]),
        "status_counts": dict(payload["status_counts"]),
        "unknown_count": 0,
        "geometry": dict(geometry),
        "search": dict(payload["search"]),
        "occupancy_rows": occupancy_rows,
        "landscape_rows": landscape_rows,
        "static_exit_distance_definition": (
            "shortest_path_in_empty_usable_cell_graph_to_nearest_exit"
        ),
        "agent_position_static_exit_distances": [
            {
                "agent_position": list(position),
                "static_exit_distance": distance,
            }
            for position, distance in exit_distances.items()
        ],
        "exit_distance_rows": exit_distance_rows,
        "witness_step_summaries": witness_step_summaries,
        "full_occupancy_boundary": full_rows,
        "witness_step_semantics": (
            "validated_goal_directed_completion_witness_length_not_shortest_distance"
        ),
    }
    report["report_sha256"] = _canonical_hash(report, "report_sha256")
    return report


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _render_landscape(ax, report: Mapping[str, Any]) -> None:
    positions = [
        tuple(cell) for cell in report["geometry"]["storage_cells"]
    ] + [
        tuple(cell)
        for cell in report["geometry"]["exit_cells"]
        if tuple(cell) not in {tuple(value) for value in report["geometry"]["storage_cells"]}
    ]
    position_index = {position: index for index, position in enumerate(positions)}
    exits = {tuple(cell) for cell in report["geometry"]["exit_cells"]}
    cmap = plt.get_cmap("RdYlGn")
    normalizer = colors.Normalize(vmin=0.0, vmax=1.0)
    zero_points = []
    for row in report["landscape_rows"]:
        x = float(row["occupancy"])
        y = float(position_index[tuple(row["agent_position"])])
        rate = float(row["safe_fraction"])
        if rate == 0.0:
            zero_points.append((x, y))
            continue
        ax.bar3d(
            x - 0.34,
            y - 0.34,
            0.0,
            0.68,
            0.68,
            rate,
            color=cmap(normalizer(rate)),
            edgecolor=(1.0, 1.0, 1.0, 0.85),
            linewidth=0.35,
            shade=True,
            alpha=0.94,
        )
    if zero_points:
        ax.scatter(
            [item[0] for item in zero_points],
            [item[1] for item in zero_points],
            [0.025] * len(zero_points),
            marker="x",
            s=32,
            linewidth=1.8,
            color=UNSAFE_COLOR,
            depthshade=False,
        )
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(-0.55, len(positions) - 0.45)
    ax.set_zlim(0.0, 1.04)
    ax.set_xticks(range(9))
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels([_position_label(position, exits) for position in positions])
    ax.set_zticks((0.0, 0.5, 1.0), ("0", ".5", "1"))
    ax.set_xlabel("Occupied storage cells", labelpad=7)
    ax.set_ylabel("Transporter end position", labelpad=8)
    ax.set_zlabel("Recoverable fraction", labelpad=7)
    ax.set_title("(a) Exact recovery-basin landscape", loc="left", pad=10)
    ax.view_init(elev=27, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.78))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 0.78))
        axis.pane.set_edgecolor((0.86, 0.88, 0.90, 0.8))
    ax.grid(True, alpha=0.27)


def _render_contraction(ax, report: Mapping[str, Any]) -> None:
    rows = report["occupancy_rows"]
    occupancy = np.array([row["occupancy"] for row in rows], dtype=float)
    safe = np.array([row["safe_fraction"] for row in rows], dtype=float)
    unsafe = 1.0 - safe
    ax.bar(
        occupancy,
        safe,
        width=0.72,
        color=SAFE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Recoverable",
    )
    ax.bar(
        occupancy,
        unsafe,
        width=0.72,
        bottom=safe,
        color=UNSAFE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Unsafe",
    )
    for row in rows:
        x = row["occupancy"]
        ax.text(
            x,
            1.035,
            f"n={row['state_count']}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#5B6470",
            rotation=90 if row["state_count"] >= 100 else 0,
        )
        if row["unsafe_count"]:
            ax.text(
                x,
                max(0.11, min(0.86, row["safe_fraction"] - 0.08)),
                f"{100.0 * row['safe_fraction']:.1f}%",
                ha="center",
                va="center",
                fontsize=7.8,
                color="white",
                fontweight="bold",
                rotation=90,
            )
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(0.0, 1.16)
    ax.set_xticks(range(9))
    ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0), ("0", "25", "50", "75", "100"))
    ax.set_xlabel("Occupied storage cells")
    ax.set_ylabel("Enumerated states (%)")
    ax.set_title("(b) Basin contraction with occupancy", loc="left")
    ax.grid(axis="y", color="#CBD1D8", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False)


def _render_boundary(ax, report: Mapping[str, Any]) -> None:
    geometry = report["geometry"]
    rows = int(geometry["grid_rows"])
    cols = int(geometry["grid_cols"])
    storage = {tuple(cell) for cell in geometry["storage_cells"]}
    exits = {tuple(cell) for cell in geometry["exit_cells"]}
    pickup = tuple(geometry["pickup_cell"])
    waiting = tuple(geometry["waiting_cell"])
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            face = WALL_COLOR
            if cell in storage:
                face = STORAGE_COLOR
            elif cell in exits:
                face = EXIT_COLOR
            elif cell == pickup:
                face = PICKUP_COLOR
            elif cell == waiting:
                face = WAIT_COLOR
            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#CBD5DC" if cell in storage else "white",
                    linewidth=1.0,
                )
            )
            if cell in storage:
                ax.add_patch(
                    Rectangle(
                        (col + 0.12, row + 0.12),
                        0.76,
                        0.76,
                        facecolor=BLOCK_COLOR,
                        edgecolor=BLOCK_EDGE,
                        linewidth=1.2,
                    )
                )
            if cell == waiting:
                ax.text(col + 0.5, row + 0.5, "W", ha="center", va="center", color="#5F4B8B", fontweight="bold")
            elif cell == pickup:
                ax.text(col + 0.5, row + 0.5, "P", ha="center", va="center", color="#805D13", fontweight="bold")
            elif cell in exits:
                ax.text(col + 0.5, row + 0.5, "E", ha="center", va="center", color="#2A7452", fontweight="bold")

    for state in report["full_occupancy_boundary"]:
        row, col = state["agent_position"]
        x = col + 0.5
        y = row + 0.5
        if state["status"] == "SAFE":
            ax.scatter(
                [x],
                [y],
                s=235,
                marker="o",
                facecolor=SAFE_COLOR,
                edgecolor="white",
                linewidth=1.6,
                zorder=6,
            )
            ax.text(
                x,
                y,
                str(state["witness_primitive_steps"]),
                ha="center",
                va="center",
                fontsize=7.4,
                color="white",
                fontweight="bold",
                zorder=7,
            )
        else:
            ax.scatter(
                [x],
                [y],
                s=170,
                marker="x",
                color=UNSAFE_COLOR,
                linewidth=2.6,
                zorder=7,
            )
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols) + 0.5, range(cols))
    ax.set_yticks(np.arange(rows) + 0.5, range(rows))
    ax.tick_params(length=0)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(
        "(c) Full occupancy: 4/9 recoverable\n"
        "same placement, transporter position varies",
        loc="left",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=SAFE_COLOR,
            markeredgecolor="white",
            markersize=9,
            label="SAFE (number = witness steps)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color=UNSAFE_COLOR,
            markeredgewidth=2.2,
            markersize=8,
            label="UNSAFE",
        ),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.26), frameon=False)


def render_figure(report: Mapping[str, Any], output: Path) -> list[str]:
    _style()
    fig = plt.figure(figsize=(15.8, 5.75))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.48, 1.0, 0.9),
        left=0.045,
        right=0.975,
        bottom=0.18,
        top=0.82,
        wspace=0.28,
    )
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    _render_landscape(ax0, report)
    _render_contraction(ax1, report)
    _render_boundary(ax2, report)
    fig.suptitle(
        "Exact compact recovery basin of the 5×5 yard",
        fontsize=15.5,
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        "Closed admission; post-transfer states; no future-arrival schedule. "
        "1,280 states: 1,261 SAFE, 19 UNSAFE, 0 UNKNOWN. "
        "Witness steps are validated completion lengths, not shortest distances.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#525D69",
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output / f"{FIGURE_STEM}.{suffix}"
        fig.savefig(path, dpi=260, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _render_aggregate_surface(ax, report: Mapping[str, Any]) -> None:
    occupancies = np.arange(9, dtype=float)
    distances = np.array(
        sorted(
            {
                int(row["static_exit_distance"])
                for row in report["exit_distance_rows"]
            }
        ),
        dtype=float,
    )
    values = np.full((len(distances), len(occupancies)), np.nan, dtype=float)
    distance_index = {int(value): index for index, value in enumerate(distances)}
    for row in report["exit_distance_rows"]:
        values[
            distance_index[int(row["static_exit_distance"])],
            int(row["occupancy"]),
        ] = float(row["safe_fraction"])
    x_grid, y_grid = np.meshgrid(occupancies, distances)
    ax.plot_surface(
        x_grid,
        y_grid,
        values,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        edgecolor=(1.0, 1.0, 1.0, 0.72),
        linewidth=0.6,
        antialiased=True,
        alpha=0.96,
        shade=True,
    )
    ax.scatter([0], [0], [1], s=28, color=SAFE_COLOR, depthshade=False)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, max(distances))
    ax.set_zlim(0, 1.03)
    ax.set_xticks(range(9))
    ax.set_yticks(range(int(max(distances)) + 1))
    ax.set_zticks((0.0, 0.5, 1.0), ("0", ".5", "1"))
    ax.set_xlabel("Occupied storage cells", labelpad=7)
    ax.set_ylabel("Static graph distance to exit", labelpad=8)
    ax.set_zlabel("Recoverable fraction", labelpad=7)
    ax.set_title("(a) Aggregated recovery-basin landscape", loc="left", pad=10)
    ax.view_init(elev=28, azim=-57)
    ax.set_box_aspect((1.35, 1.05, 0.75))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 0.8))
        axis.pane.set_edgecolor((0.86, 0.88, 0.90, 0.8))
    ax.grid(True, alpha=0.27)


def _render_witness_effort(ax, report: Mapping[str, Any]) -> None:
    rows = report["witness_step_summaries"]
    x = np.array([row["occupancy"] for row in rows], dtype=float)
    q10 = np.array([row["q10"] for row in rows], dtype=float)
    q25 = np.array([row["q25"] for row in rows], dtype=float)
    median = np.array([row["median"] for row in rows], dtype=float)
    q75 = np.array([row["q75"] for row in rows], dtype=float)
    q90 = np.array([row["q90"] for row in rows], dtype=float)
    minimum = np.array([row["minimum"] for row in rows], dtype=float)
    maximum = np.array([row["maximum"] for row in rows], dtype=float)
    ax.fill_between(
        x,
        q10,
        q90,
        color="#8EC9D6",
        alpha=0.35,
        linewidth=0,
        label="10th–90th percentile",
    )
    ax.fill_between(
        x,
        q25,
        q75,
        color="#278DA6",
        alpha=0.42,
        linewidth=0,
        label="Interquartile range",
    )
    ax.plot(
        x,
        minimum,
        color="#7A8793",
        linestyle=":",
        linewidth=1.1,
        label="Observed range",
    )
    ax.plot(x, maximum, color="#7A8793", linestyle=":", linewidth=1.1)
    ax.plot(
        x,
        median,
        color="#0B5563",
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        label="Median",
    )
    ax.set_xlim(-0.25, 8.25)
    ax.set_ylim(0, max(maximum) + 5)
    ax.set_xticks(range(9))
    ax.set_xlabel("Occupied storage cells")
    ax.set_ylabel("Validated witness primitive steps")
    ax.set_title("(c) Recovery effort within the basin", loc="left")
    ax.grid(color="#CBD1D8", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)


def render_aggregate_figure(report: Mapping[str, Any], output: Path) -> list[str]:
    _style()
    fig = plt.figure(figsize=(15.8, 5.65))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.45, 0.98, 1.05),
        left=0.045,
        right=0.975,
        bottom=0.17,
        top=0.82,
        wspace=0.3,
    )
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    _render_aggregate_surface(ax0, report)
    _render_contraction(ax1, report)
    _render_witness_effort(ax2, report)
    fig.suptitle(
        "Aggregate structure of the exact compact 5×5 recovery basin",
        fontsize=15.5,
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        "Fractions aggregate distinct occupancy configurations; the surface joins discrete aggregate cells. "
        "Static exit distance uses the empty usable-cell graph. Witness lengths are validated upper bounds, not shortest distances.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#525D69",
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output / f"{AGGREGATE_FIGURE_STEM}.{suffix}"
        fig.savefig(path, dpi=260, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def render_caption(output: Path) -> str:
    text = """# Compact 5x5 recovery-basin figures

## Suggested main caption: aggregate figure

Aggregate structure of the exact compact closed-admission recovery basin for
the 5×5 yard. Panel (a) projects the enumerated state set onto occupancy and
the transporter's static graph distance to the exit; surface height and color
give the recoverable fraction within each aggregate cell, with the surface
joining those discrete cells only for visualization. Panel (b) shows the
corresponding contraction in total basin volume. Panel (c) summarizes the
primitive-step lengths of validated goal-directed completion witnesses within
the basin. The state set remains universally recoverable through four occupied
storage cells, while recoverability first contracts in distant transporter
states and reaches 44.4% at full occupancy. The enumeration contains 1,280
states (1,261 SAFE, 19 UNSAFE, no UNKNOWN). Distances are measured in the empty
usable-cell graph, and witness lengths are completion upper bounds rather than
exact shortest distances. The figure represents the compact post-transfer
recovery model with closed admission and no future-arrival schedule; it is not
the full dynamic-episode or disturbance-robust viability kernel.

## Suggested companion caption: spatial boundary slice

Exact compact closed-admission recovery basin for the 5×5 yard. Panel (a)
shows the fraction of enumerated occupancy configurations certified recoverable
for each occupancy level and valid post-transfer transporter endpoint. Panel
(b) aggregates the exact state labels by occupancy. Recoverability remains
universal through four occupied storage cells, then contracts to 44.4% at full
occupancy. Panel (c) holds the fully occupied placement fixed and varies only
the transporter position: four of nine post-transfer states admit a validated
completion witness. Numbers inside SAFE markers are primitive steps in the
goal-directed witness and are completion upper bounds, not exact shortest
distances. The enumeration contains 1,280 states (1,261 SAFE, 19 UNSAFE, no
UNKNOWN). It represents the compact post-transfer recovery model with closed
admission and no future-arrival schedule; it is not the full dynamic-episode or
disturbance-robust viability kernel.

## Recommended terminology

Use **exact compact 5×5 closed-admission recovery basin** in the paper. Avoid
calling this figure the full VCG viability kernel.
"""
    path = output / "vcg-compact-recovery-basin-5x5-caption.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def render(source: Path, output: Path) -> dict[str, Any]:
    payload = load_source(source)
    report = build_summary(payload, source)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "vcg-compact-recovery-basin-5x5-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure_pdf, figure_png = render_figure(report, output)
    aggregate_pdf, aggregate_png = render_aggregate_figure(report, output)
    caption = render_caption(output)
    return {
        "status": "complete",
        "source": str(source),
        "report": str(report_path),
        "figure_pdf": figure_pdf,
        "figure_png": figure_png,
        "preferred_aggregate_figure_pdf": aggregate_pdf,
        "preferred_aggregate_figure_png": aggregate_png,
        "caption": caption,
        "training_runs": 0,
        "evaluation_runs": 0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = render(args.source.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
