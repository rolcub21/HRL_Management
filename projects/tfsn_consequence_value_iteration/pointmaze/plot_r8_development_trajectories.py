"""Plot positive and negative PointMaze R8 hidden-goal trajectories.

The evaluation artifacts retain the realized endpoint of every 100-step
command, not every primitive point-mass state.  Lines in this figure therefore
connect command endpoints within each carried episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = (
    HERE.parent
    / "results"
    / "pointmaze"
    / "density_movement_reach_development_r8"
)


def load_json(path: Path):
    return json.loads(path.read_text())


def draw_maze(ax, env):
    maze = env._env.unwrapped.maze
    maze_map = np.asarray(maze.maze_map, dtype=object)
    unit = float(maze.maze_size_scaling)
    centers = []
    for row in range(maze_map.shape[0]):
        for col in range(maze_map.shape[1]):
            center = np.asarray(maze.cell_rowcol_to_xy((row, col)), dtype=float)
            centers.append(center)
            wall = str(maze_map[row, col]) == "1"
            ax.add_patch(Rectangle(
                center - unit / 2.0,
                unit,
                unit,
                facecolor="#313b49" if wall else "#f7f8f9",
                edgecolor="#313b49" if wall else "#d6dbe0",
                linewidth=0.35,
                zorder=0,
            ))
    centers = np.asarray(centers)
    ax.set_xlim(centers[:, 0].min() - unit / 2, centers[:, 0].max() + unit / 2)
    ax.set_ylim(centers[:, 1].min() - unit / 2, centers[:, 1].max() + unit / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def acquisition_rows(task):
    return [row for row in task["decisions"] if row["phase"] == "acquisition"]


def plot_task(ax, env, task, success_radius, cmap, episode_count):
    draw_maze(ax, env)
    goal = np.asarray(task["goal_xy"], dtype=float)
    acquired = bool(task["metrics"]["acquisition_at_horizon"])
    rows = acquisition_rows(task)

    footprint = Circle(
        goal,
        success_radius,
        facecolor="#d94b4b",
        edgecolor="#9d2020",
        linewidth=1.0,
        alpha=0.17,
        zorder=2,
    )
    ax.add_patch(footprint)
    ax.scatter(
        *goal,
        marker="X",
        s=58,
        color="#b12828",
        edgecolor="white",
        linewidth=0.6,
        zorder=9,
    )

    closest_xy = None
    closest_distance = np.inf
    for episode in sorted({int(row["episode"]) for row in rows}):
        episode_rows = [row for row in rows if int(row["episode"]) == episode]
        if not episode_rows:
            continue
        start = np.asarray(episode_rows[0]["start_xy"], dtype=float)
        endpoints = np.asarray([row["endpoint_xy"] for row in episode_rows], dtype=float)
        path = np.vstack((start, endpoints))
        color = cmap(episode / max(1, episode_count - 1))
        ax.plot(
            path[:, 0], path[:, 1], color=color, linewidth=1.35,
            alpha=0.82, zorder=4,
        )
        ax.scatter(
            endpoints[:, 0], endpoints[:, 1], s=10, color=color,
            edgecolor="white", linewidth=0.2, zorder=5,
        )
        ax.scatter(
            *start, marker="o", s=24, facecolor="white", edgecolor=color,
            linewidth=0.9, zorder=7,
        )
        for row in episode_rows:
            distance = float(row["best_goal_distance_m"])
            if distance < closest_distance:
                closest_distance = distance
                closest_xy = np.asarray(row["endpoint_xy"], dtype=float)
            if row["contact_step"] is not None:
                ax.scatter(
                    *np.asarray(row["endpoint_xy"], dtype=float),
                    marker="*", s=92, color="#f1b82d", edgecolor="#654900",
                    linewidth=0.65, zorder=10,
                )

    if not acquired and closest_xy is not None:
        ax.scatter(
            *closest_xy, marker="D", s=38, facecolor="none",
            edgecolor="#1f2937", linewidth=1.1, zorder=10,
        )
        ax.annotate(
            f"closest {closest_distance:.3f} m",
            xy=closest_xy,
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
            color="#1f2937",
            zorder=11,
        )

    status = "ACQUIRED" if acquired else "MISSED"
    status_color = "#18794e" if acquired else "#b42318"
    first = int(task["metrics"]["first_contact_episode"])
    detail = (
        f"contact episode {first + 1}; E2E {task['metrics']['mean_e2e']:.3f}"
        if acquired
        else f"closest {closest_distance:.3f} m; E2E 0"
    )
    ax.set_title(
        f"{task['goal_name'].replace('spatial_diagnostic_native_goal_', '')}  "
        f"{status}\n{detail}",
        fontsize=9.3,
        color=status_color,
        fontweight="semibold",
    )
    for spine in ax.spines.values():
        spine.set_color(status_color)
        spine.set_linewidth(1.1 if not acquired else 0.75)
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = args.results.resolve()
    summary = load_json(results / "summary.json")
    output = (
        args.output.resolve()
        if args.output is not None
        else results / "pointmaze_r8_positive_negative_trajectories.png"
    )
    tasks = [load_json(path) for path in sorted((results / "tasks").glob("*.json"))]
    if len(tasks) != int(summary["tasks"]):
        raise ValueError("PointMaze R8 task artifacts are incomplete")

    episode_count = int(summary["episodes_per_task"])
    cmap = plt.get_cmap("viridis")
    env = PointMazeTimeStepEnv("PointMaze_Medium-v3", max_episode_steps=2000)
    try:
        fig, axes = plt.subplots(
            3, 3, figsize=(11.5, 11.3), constrained_layout=True,
            sharex=True, sharey=True,
        )
        for ax, task in zip(axes.ravel(), tasks):
            plot_task(
                ax,
                env,
                task,
                float(summary["success_radius_evaluator_only_m"]),
                cmap,
                episode_count,
            )
        for row in axes:
            row[0].set_ylabel("y (m)", fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("x (m)", fontsize=9)

        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#4b5563", label="episode reset"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#b12828",
                   markeredgecolor="white", markersize=8, label="hidden goal"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#f1b82d",
                   markeredgecolor="#654900", markersize=10, label="contact command"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="none",
                   markeredgecolor="#1f2937", markersize=6, label="closest miss"),
        ]
        fig.legend(
            handles=legend, loc="outside lower left", ncol=4,
            frameon=False, fontsize=8.5,
        )
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(1, episode_count), cmap=cmap),
            ax=axes.ravel().tolist(), location="bottom", fraction=0.025,
            pad=0.055, aspect=38,
        )
        colorbar.set_label("carried episode", fontsize=8.5)
        colorbar.set_ticks([1, episode_count])
        colorbar.ax.tick_params(labelsize=8)
        fig.suptitle(
            "PointMaze R8 hidden-goal development: 7 acquisitions, 2 boundary misses\n"
            "Lines connect realized 100-step command endpoints; circles show the 0.4 m contact footprint",
            fontsize=14,
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
