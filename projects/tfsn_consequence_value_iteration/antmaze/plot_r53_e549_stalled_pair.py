"""Visualize the completed E549/R53 hidden-goal stalled-pair audit.

The R53 logs retain one realized endpoint per 100-step command, not every
primitive Ant state.  Solid lines therefore connect command endpoints.  For
commands whose realized displacement is below the diagnostic 3 cm threshold,
the intended command is additionally shown as a dashed segment ending in an
``x``.  This makes physical non-movement distinguishable from a planner that
merely selected a short command.
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

from scripts.antmaze.current_tfsn.executor import full_state, make_env


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = (
    HERE.parent
    / "results"
    / "antmaze"
    / "e549_5m_destination_arrival_r53"
    / "stalled_pair"
)
R48_PROTOCOL = HERE / "r48_rule_free_protocol.json"
LOW_MOTION_M = 0.03


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def draw_maze(ax, unwrapped):
    maze = getattr(unwrapped, "maze", unwrapped)
    maze_map = np.asarray(maze.maze_map, dtype=object)
    unit = float(
        getattr(maze, "maze_size_scaling", getattr(unwrapped, "_maze_unit", 4.0))
    )
    to_xy = getattr(maze, "cell_rowcol_to_xy", None)
    if to_xy is None:
        to_xy = unwrapped.ij_to_xy
    centers = []
    for row in range(maze_map.shape[0]):
        for column in range(maze_map.shape[1]):
            center = np.asarray(to_xy((row, column)), dtype=float)
            centers.append(center)
            wall = str(maze_map[row, column]) == "1"
            ax.add_patch(
                Rectangle(
                    center - unit / 2,
                    unit,
                    unit,
                    facecolor="#303946" if wall else "#f6f7f8",
                    edgecolor="#303946" if wall else "#d9dde1",
                    linewidth=0.35,
                    zorder=0,
                )
            )
    centers = np.asarray(centers)
    ax.set_xlim(centers[:, 0].min() - unit / 2, centers[:, 0].max() + unit / 2)
    ax.set_ylim(centers[:, 1].min() - unit / 2, centers[:, 1].max() + unit / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)


def reset_xy(env, reset_seed: int):
    observation, _ = env.reset(seed=int(reset_seed))
    return full_state(observation)[:2].astype(float)


def plot_task(ax, env, task, rows, contact_radius, episode_colors):
    draw_maze(ax, env.unwrapped)
    goal = np.asarray(task["goal"], dtype=float)
    start = reset_xy(env, task["reset_seed"])
    acquisition = [row for row in rows if row["phase"] == "acquisition"]
    endpoints_all = np.asarray(
        [row["actual_endpoint"] for row in acquisition], dtype=float
    )
    goal_distances = np.linalg.norm(endpoints_all - goal[None], axis=1)
    closest_index = int(np.argmin(goal_distances))
    closest_distance = float(goal_distances[closest_index])
    low_motion_count = int(
        sum(float(row["moved_distance"]) < LOW_MOTION_M for row in acquisition)
    )

    ax.add_patch(
        Circle(
            goal,
            contact_radius,
            facecolor="#e85d5d",
            edgecolor="#a82727",
            linewidth=1.2,
            alpha=0.18,
            zorder=2,
        )
    )
    ax.scatter(
        *goal,
        marker="X",
        s=90,
        color="#b52b2b",
        edgecolor="white",
        linewidth=0.7,
        zorder=9,
    )

    for episode in sorted({int(row["episode"]) for row in acquisition}):
        episode_rows = [
            row for row in acquisition if int(row["episode"]) == episode
        ]
        endpoints = np.asarray(
            [row["actual_endpoint"] for row in episode_rows], dtype=float
        )
        path = np.vstack((start, endpoints))
        color = episode_colors[episode % len(episode_colors)]
        ax.plot(
            path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=0.9, zorder=4
        )
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=18,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=6,
        )

        for command_index, row in enumerate(episode_rows):
            if float(row["moved_distance"]) >= LOW_MOTION_M:
                continue
            source = start if command_index == 0 else endpoints[command_index - 1]
            command = np.asarray(row["command"], dtype=float)
            endpoint = endpoints[command_index]
            ax.plot(
                [source[0], command[0]],
                [source[1], command[1]],
                color="#111827",
                linewidth=0.75,
                linestyle=(0, (3, 2)),
                alpha=0.7,
                zorder=5,
            )
            ax.scatter(
                *command,
                marker="x",
                s=28,
                color="#111827",
                linewidth=0.9,
                zorder=7,
            )
            ax.scatter(
                *endpoint,
                s=52,
                facecolor="none",
                edgecolor="#111827",
                linewidth=1.0,
                zorder=8,
            )

    closest = endpoints_all[closest_index]
    ax.scatter(
        *closest,
        marker="D",
        s=46,
        facecolor="#f0b429",
        edgecolor="#6b4f00",
        linewidth=0.8,
        zorder=10,
    )
    ax.scatter(
        *start,
        marker="o",
        s=52,
        facecolor="white",
        edgecolor="#111827",
        linewidth=1.2,
        zorder=9,
    )
    ax.set_title(
        f"Task {task['task_id']}: goal ({goal[0]:.0f}, {goal[1]:.0f})\n"
        f"closest={closest_distance:.2f} m; new cells/command="
        f"{task['acquisition_new_1m_cells_per_command']:.2f}; "
        f"<3 cm={low_motion_count}/50",
        fontsize=11,
        fontweight="semibold",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = args.results.resolve()
    summary = load_json(results / "summary.json")
    protocol = load_json(R48_PROTOCOL)
    contact_radius = float(
        protocol["evaluator"]["success_radius_m_evaluator_only"]
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else results / "r53_e549_stalled_pair_trajectories.png"
    )
    episode_colors = ["#2776b8", "#6f4aa8", "#17866b", "#dd7b22", "#c13b67"]

    env = make_env(max_episode_steps=1000)
    try:
        fig, axes = plt.subplots(
            1, len(summary["rows"]), figsize=(13.2, 6.4), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        for ax, task in zip(axes, summary["rows"]):
            rows = load_jsonl(
                results / f"task_{int(task['task_id']):03d}.jsonl"
            )
            plot_task(ax, env, task, rows, contact_radius, episode_colors)

        legend = [
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white",
                markeredgecolor="#111827", label="physical reset"
            ),
            Line2D(
                [0], [0], marker="X", color="none", markerfacecolor="#b52b2b",
                markeredgecolor="white", markersize=9, label="hidden goal"
            ),
            Line2D(
                [0], [0], marker="D", color="none", markerfacecolor="#f0b429",
                markeredgecolor="#6b4f00", markersize=7, label="closest endpoint"
            ),
            Line2D(
                [0], [0], marker="o", color="#111827", linestyle=(0, (3, 2)),
                markerfacecolor="none", markeredgecolor="#111827", markersize=7,
                label="<3 cm: intended target and endpoint"
            ),
        ]
        legend.extend(
            Line2D([0], [0], color=color, lw=2, label=f"episode {index + 1}")
            for index, color in enumerate(episode_colors)
        )
        fig.legend(
            handles=legend,
            loc="outside lower center",
            ncol=5,
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            "AntMaze R53: E549 5M hidden-goal acquisition (command endpoints)",
            fontsize=15,
            fontweight="bold",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    finally:
        env.close()

    print(output)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
