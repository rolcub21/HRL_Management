"""Visualize a completed nine-goal hidden-goal AntMaze panel.

This is read-only analysis.  The evaluation logs retain realized endpoints
after each 100-step command, not primitive Ant trajectories.  Lines therefore
connect command endpoints within each physical episode; the first segment from
the reset state is not reconstructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent / "results" / "antmaze"
DEFAULT_RESULTS = RESULTS_ROOT / "e549_5m_hidden_goal_r59" / "original9"
DEFAULT_SUPPLIED = RESULTS_ROOT / "e549_5m_supplied_goal_r58" / "original9"
PROTOCOL = HERE / "r48_rule_free_protocol.json"

SUCCESS = "#16837a"
SEARCH_FAILURE = "#d07a21"
STACK_FAILURE = "#b74343"
WALL = "#353d48"
OPEN = "#f7f8fa"
GRID = "#d9dde3"
EPISODE_COLORS = ["#2f6fbb", "#7655a5", "#17866b", "#dd7b22", "#c13b67"]

# Gymnasium-Robotics AntMaze_Medium_Diverse_GR-v4, used only as a plotting
# background.  This script never creates or steps an environment.
MAZE_MAP = np.asarray(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)
MAZE_SCALE_M = 4.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def maze_cell_center(row: int, column: int) -> np.ndarray:
    map_length, map_width = MAZE_MAP.shape
    return np.asarray(
        [
            (column + 0.5) * MAZE_SCALE_M - map_width * MAZE_SCALE_M / 2.0,
            map_length * MAZE_SCALE_M / 2.0 - (row + 0.5) * MAZE_SCALE_M,
        ]
    )


def draw_maze(ax: plt.Axes) -> None:
    for row in range(MAZE_MAP.shape[0]):
        for column in range(MAZE_MAP.shape[1]):
            center = maze_cell_center(row, column)
            wall = MAZE_MAP[row, column] == 1
            ax.add_patch(
                Rectangle(
                    center - MAZE_SCALE_M / 2.0,
                    MAZE_SCALE_M,
                    MAZE_SCALE_M,
                    facecolor=WALL if wall else OPEN,
                    edgecolor=WALL if wall else GRID,
                    linewidth=0.25,
                    zorder=0,
                )
            )
    extent = MAZE_MAP.shape[0] * MAZE_SCALE_M / 2.0
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def result_class(task: dict, supplied_success: bool) -> tuple[str, str]:
    if task["first_contact_episode"] is not None:
        return "ACQUIRED", SUCCESS
    if supplied_success:
        return "SEARCH FAIL", SEARCH_FAILURE
    return "STACK FAIL", STACK_FAILURE


def plot_task(
    ax: plt.Axes,
    task: dict,
    rows: list[dict],
    supplied_success: bool,
    contact_radius: float,
) -> None:
    draw_maze(ax)
    acquisition = [row for row in rows if row.get("phase") == "acquisition"]
    goal = np.asarray(task["goal"], dtype=float)
    label, title_color = result_class(task, supplied_success)

    ax.add_patch(
        Circle(
            goal,
            contact_radius,
            facecolor=STACK_FAILURE,
            edgecolor=STACK_FAILURE,
            linewidth=0.8,
            alpha=0.14,
            zorder=2,
        )
    )
    ax.scatter(
        *goal,
        marker="X",
        s=68,
        color=STACK_FAILURE,
        edgecolor="white",
        linewidth=0.6,
        zorder=9,
    )

    for episode in sorted({int(row["episode"]) for row in acquisition}):
        episode_rows = [
            row for row in acquisition if int(row["episode"]) == episode
        ]
        endpoints = np.asarray(
            [row["actual_endpoint"] for row in episode_rows], dtype=float
        )
        color = EPISODE_COLORS[episode]
        if len(endpoints) > 1:
            ax.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color=color,
                linewidth=1.35,
                alpha=0.88,
                zorder=3,
            )
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=13,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
        )
        ax.scatter(
            *endpoints[0],
            marker="o",
            s=34,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=6,
        )
        stalled = np.asarray(
            [float(row["moved_distance"]) < 0.2 for row in episode_rows],
            dtype=bool,
        )
        if stalled.any():
            ax.scatter(
                endpoints[stalled, 0],
                endpoints[stalled, 1],
                marker="s",
                s=34,
                facecolor="none",
                edgecolor="#151a21",
                linewidth=0.75,
                zorder=7,
            )
        contacts = np.asarray([bool(row["reward"]) for row in episode_rows])
        if contacts.any():
            first_contact = int(np.flatnonzero(contacts)[0])
            ax.scatter(
                *endpoints[first_contact],
                marker="*",
                s=120,
                color="#f1b82d",
                edgecolor="#604500",
                linewidth=0.6,
                zorder=10,
            )

    all_endpoints = np.asarray(
        [row["actual_endpoint"] for row in acquisition], dtype=float
    )
    endpoint_distances = np.linalg.norm(all_endpoints - goal, axis=1)
    nearest_distance = float(endpoint_distances.min())
    if task["first_contact_episode"] is None:
        nearest = int(endpoint_distances.argmin())
        ax.scatter(
            *all_endpoints[nearest],
            marker="D",
            s=42,
            facecolor="#f1b82d",
            edgecolor="#604500",
            linewidth=0.6,
            zorder=8,
        )
        detail = (
            f"closest endpoint {nearest_distance:.2f} m | "
            f"arrival {task['online_destination_arrival_rate']:.0%}"
        )
    else:
        contact = next(row for row in acquisition if bool(row["reward"]))
        detail = (
            f"contact E{int(contact['episode']) + 1}, "
            f"C{int(contact['command_index']) + 1} | "
            f"arrival {task['online_destination_arrival_rate']:.0%}"
        )

    ax.set_title(
        f"T{task['task_id']} goal=({goal[0]:.0f}, {goal[1]:.0f}) — {label}\n{detail}",
        fontsize=8.8,
        color=title_color,
        fontweight="semibold",
    )
    ax.set_xticks([-12, 0, 12])
    ax.set_yticks([-12, 0, 12])
    ax.tick_params(labelsize=7)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--supplied", type=Path, default=DEFAULT_SUPPLIED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-title", default=None)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    results = args.results.resolve()
    supplied = args.supplied.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results / "plots"
    )
    summary = load_json(results / "summary.json")
    tasks = sorted(summary["rows"], key=lambda row: int(row["task_id"]))
    if [int(row["task_id"]) for row in tasks] != list(range(9)):
        raise ValueError("original-nine atlas requires task ids 0..8")
    supplied_rows = {
        int(row["task_id"]): bool(row["success"])
        for row in load_json(supplied / "summary.json")["rows"]
    }
    contact_radius = float(
        load_json(PROTOCOL)["evaluator"]["success_radius_m_evaluator_only"]
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14.8, 14.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for ax, task in zip(axes.flat, tasks):
        task_id = int(task["task_id"])
        plot_task(
            ax,
            task,
            load_jsonl(results / f"task_{task_id:03d}.jsonl"),
            supplied_rows[task_id],
            contact_radius,
        )

    for row_axes in axes:
        row_axes[0].set_ylabel("y (m)")
    for ax in axes[-1]:
        ax.set_xlabel("x (m)")

    legend = [
        Line2D(
            [0], [0], marker="X", color="none", markerfacecolor=STACK_FAILURE,
            markeredgecolor="white", markersize=8, label="hidden goal"
        ),
        Line2D(
            [0], [0], marker="o", color=STACK_FAILURE, markerfacecolor="none",
            markeredgecolor=STACK_FAILURE, markersize=8,
            label="evaluator-only 0.8 m contact radius"
        ),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#f1b82d",
            markeredgecolor="#604500", markersize=11,
            label="endpoint of contact command"
        ),
        Line2D(
            [0], [0], marker="D", color="none", markerfacecolor="#f1b82d",
            markeredgecolor="#604500", markersize=6,
            label="closest failed endpoint"
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="none",
            markeredgecolor="#151a21", markersize=6, label="<0.2 m movement"
        ),
    ]
    legend.extend(
        Line2D([0], [0], color=color, lw=2, label=f"episode {episode + 1}")
        for episode, color in enumerate(EPISODE_COLORS)
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    acquired = sum(row["first_contact_episode"] is not None for row in tasks)
    title = args.figure_title or (
        f"{summary['method_revision']}: {acquired}/9 goals found"
    )
    fig.suptitle(
        f"{title}\n"
        "Realized 100-step command endpoints; reset segments and primitive paths were not logged",
        fontsize=14,
        fontweight="bold",
    )
    stem = output_dir / (
        args.stem or "r59_hidden_goal_original9_trajectories"
    )
    save_figure(fig, stem)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
