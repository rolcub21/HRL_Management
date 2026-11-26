"""Plot the two continuous 50-command AntMaze acquisition diagnostics.

The logs contain one realized endpoint per 100-step command.  Colors therefore
show consecutive ten-command blocks inside the single physical episode, not
primitive Ant trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from plot_r39_gate2_trajectories import draw_maze, reset_xy
from scripts.antmaze.current_tfsn.executor import make_env


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = (
    HERE.parent
    / "results"
    / "antmaze"
    / "live_posterior_passage_r39"
    / "diagnostic_gate2_continuous50"
)
BLOCK_COLORS = ["#2776b8", "#6f4aa8", "#17866b", "#dd7b22", "#c13b67"]


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def plot_task(ax, env, task, rows, contact_radius):
    draw_maze(ax, env.unwrapped)
    goal = np.asarray(task["goal"], dtype=float)
    start = reset_xy(env, int(task["reset_seed"]))
    endpoints = np.asarray([row["actual_endpoint"] for row in rows], dtype=float)
    path = np.vstack((start, endpoints))
    moves = np.asarray([float(row["moved_distance"]) for row in rows])
    stalled = moves < 0.01

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

    for block in range(5):
        lo = block * 10
        hi = min((block + 1) * 10, len(rows))
        if lo >= hi:
            continue
        block_path = path[lo : hi + 1]
        color = BLOCK_COLORS[block]
        ax.plot(
            block_path[:, 0],
            block_path[:, 1],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(
            endpoints[lo:hi, 0],
            endpoints[lo:hi, 1],
            s=20,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=6,
        )

    # Mark the first endpoint after which the run remains trapped.  This is
    # more legible than drawing dozens of overlapping stall rings.
    sustained_stall = None
    for index in range(len(stalled)):
        if stalled[index] and np.mean(stalled[index:]) >= 0.9:
            sustained_stall = index
            break
    if sustained_stall is not None:
        point = endpoints[sustained_stall]
        ax.scatter(
            *point,
            marker="o",
            s=135,
            facecolor="none",
            edgecolor="#111827",
            linewidth=2.0,
            zorder=10,
        )
        ax.annotate(
            f"sustained stall begins\ncommand {sustained_stall + 1}",
            xy=point,
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="semibold",
            arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 0.8},
            zorder=11,
        )

    nearest = int(np.argmin(np.linalg.norm(endpoints - goal[None, :], axis=1)))
    nearest_distance = float(np.linalg.norm(endpoints[nearest] - goal))
    ax.scatter(
        *endpoints[nearest],
        marker="D",
        s=55,
        facecolor="#f0b429",
        edgecolor="#6b4f00",
        linewidth=0.8,
        zorder=10,
    )
    ax.scatter(
        *start,
        marker="o",
        s=55,
        facecolor="white",
        edgecolor="#111827",
        linewidth=1.2,
        zorder=9,
    )
    ax.scatter(
        *endpoints[-1],
        marker="s",
        s=48,
        facecolor="#111827",
        edgecolor="white",
        linewidth=0.5,
        zorder=10,
    )
    ax.set_title(
        f"Task {task['task_id']}: goal ({goal[0]:.0f}, {goal[1]:.0f})\n"
        f"not acquired; nearest={nearest_distance:.2f} m; "
        f"stalled={np.mean(stalled):.0%}",
        fontsize=11,
        fontweight="semibold",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = args.results.resolve()
    summary = load_json(results / "summary.json")
    baseline = load_json(results.parent / "evaluation_gate2" / "summary.json")
    contact_radius = float(baseline["acquisition_detection_radius_m"])
    output = (
        args.output.resolve()
        if args.output is not None
        else results / "r40_continuous50_acquisition_trajectories.png"
    )

    env = make_env(max_episode_steps=5000)
    try:
        fig, axes = plt.subplots(
            1, len(summary["rows"]), figsize=(13.2, 6.2), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        for ax, task in zip(axes, summary["rows"]):
            rows = load_jsonl(results / f"task_{int(task['task_id']):03d}.jsonl")
            plot_task(ax, env, task, rows, contact_radius)

        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#111827", label="physical reset"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#b52b2b",
                   markeredgecolor="white", markersize=9, label="hidden goal"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#f0b429",
                   markeredgecolor="#6b4f00", markersize=7, label="closest endpoint"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#111827", markersize=9, label="sustained stall onset"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#111827",
                   markeredgecolor="white", markersize=7, label="final endpoint"),
        ]
        legend.extend(
            Line2D([0], [0], color=color, lw=2, label=f"commands {10*i+1}–{10*i+10}")
            for i, color in enumerate(BLOCK_COLORS)
        )
        fig.legend(
            handles=legend,
            loc="outside lower center",
            ncol=5,
            frameon=False,
            fontsize=8.5,
        )
        fig.suptitle(
            "AntMaze R40: one continuous 50-command acquisition episode",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.055,
            "Lines connect realized 100-step command endpoints; colors are consecutive ten-command blocks.",
            ha="center",
            fontsize=9,
            color="#4b5563",
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
