"""Plot carried hidden-goal trajectories for the frozen R74 PointMaze port.

Evaluation logs retain one endpoint per 100-step command, so plotted segments
connect command endpoints rather than primitive PointMaze states.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv
from projects.tfsn_consequence_value_iteration.pointmaze.plot_r8_development_trajectories import (
    draw_maze,
)


def _load(path: Path):
    return json.loads(path.read_text())


def _short_goal_name(name: str) -> str:
    for prefix in ("spatial_diagnostic_native_goal_", "native_goal_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _plot_task(ax, env, task, success_radius: float, cmap, episode_count: int):
    draw_maze(ax, env)
    goal = np.asarray(task["goal_xy"], dtype=float)
    rows = [row for row in task["decisions"] if row["phase"] == "acquisition"]

    ax.add_patch(Circle(
        goal,
        success_radius,
        facecolor="#d94b4b",
        edgecolor="#9d2020",
        linewidth=0.9,
        alpha=0.16,
        zorder=2,
    ))
    ax.scatter(
        *goal,
        marker="X",
        s=47,
        color="#b12828",
        edgecolor="white",
        linewidth=0.55,
        zorder=9,
    )

    for episode in sorted({int(row["episode"]) for row in rows}):
        episode_rows = [row for row in rows if int(row["episode"]) == episode]
        if not episode_rows:
            continue
        start = np.asarray(episode_rows[0]["start_xy"], dtype=float)
        endpoints = np.asarray([row["endpoint_xy"] for row in episode_rows], dtype=float)
        path = np.vstack((start, endpoints))
        color = cmap(episode / max(1, episode_count - 1))
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=1.15,
            alpha=0.82,
            zorder=4,
        )
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=7,
            color=color,
            edgecolor="white",
            linewidth=0.15,
            zorder=5,
        )
        ax.scatter(
            *start,
            marker="o",
            s=18,
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            zorder=7,
        )
        for row in episode_rows:
            if row["contact_step"] is not None:
                ax.scatter(
                    *np.asarray(row["endpoint_xy"], dtype=float),
                    marker="*",
                    s=72,
                    color="#f1b82d",
                    edgecolor="#654900",
                    linewidth=0.55,
                    zorder=10,
                )

    first = int(task["metrics"]["first_contact_episode"])
    ax.set_title(
        f"{_short_goal_name(task['goal_name'])}  contact E{first + 1}\n"
        f"E2E {task['metrics']['mean_e2e']:.3f}",
        fontsize=8.2,
        color="#18794e",
        fontweight="semibold",
    )
    for spine in ax.spines.values():
        spine.set_color("#18794e")
        spine.set_linewidth(0.65)
    ax.tick_params(labelsize=6.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    results = args.results.resolve()
    output = args.output.resolve()
    summary = _load(results / "summary.json")
    tasks = [_load(path) for path in sorted((results / "tasks").glob("*.json"))]
    if len(tasks) != int(summary["tasks"]):
        raise ValueError("PointMaze R15 task artifacts are incomplete")
    if not all(task["metrics"]["acquisition_at_horizon"] for task in tasks):
        raise ValueError("this frozen-port figure expects the passed 24-goal gate")

    columns = max(1, int(args.columns))
    rows = int(math.ceil(len(tasks) / columns))
    episode_count = int(summary["episodes_per_task"])
    cmap = plt.get_cmap("viridis")
    env = PointMazeTimeStepEnv("PointMaze_Medium-v3", max_episode_steps=2000)
    try:
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.25 * columns, 3.05 * rows + 0.7),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for ax, task in zip(axes.ravel(), tasks):
            _plot_task(
                ax,
                env,
                task,
                float(summary["success_radius_evaluator_only_m"]),
                cmap,
                episode_count,
            )
        for ax in axes.ravel()[len(tasks):]:
            ax.set_visible(False)
        for row_axes in axes:
            row_axes[0].set_ylabel("y (m)", fontsize=8)
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel("x (m)", fontsize=8)

        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#4b5563", label="episode reset"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#b12828",
                   markeredgecolor="white", markersize=7, label="hidden goal"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#f1b82d",
                   markeredgecolor="#654900", markersize=9, label="first contact"),
        ]
        fig.legend(
            handles=legend,
            loc="outside lower left",
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(1, episode_count), cmap=cmap),
            ax=axes.ravel().tolist(),
            location="bottom",
            fraction=0.018,
            pad=0.045,
            aspect=45,
        )
        colorbar.set_label("carried episode", fontsize=8)
        colorbar.set_ticks([1, episode_count])
        colorbar.ax.tick_params(labelsize=7.5)
        fig.suptitle(
            f"{args.title}\n"
            "Lines connect realized 100-step command endpoints; shaded circles are evaluator-only contact footprints",
            fontsize=12.5,
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
