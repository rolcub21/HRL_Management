"""Plot R48 full-panel AntMaze acquisition successes and failures.

This is a read-only analysis of completed evaluation logs.  The trajectory
atlas uses the realized endpoint recorded after each 100-step command; it does
not reset or step the environment and does not reconstruct primitive paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent / "results" / "antmaze"
DEFAULT_RESULTS = (
    RESULTS_ROOT
    / "r39_r46_rule_removal_r48_full15"
    / "evaluation_all15"
)
DEFAULT_BASELINE = RESULTS_ROOT / "live_posterior_passage_r39" / "evaluation_all15"

SUCCESS = "#16837a"
FAILURE = "#c75b39"
BASELINE = "#5072a7"
NEUTRAL = "#68717d"
GRID = "#d9dde3"
WALL = "#353d48"
OPEN = "#f7f8fa"
EPISODE_COLORS = ["#2f6fbb", "#7655a5", "#17866b", "#dd7b22", "#c13b67"]

# Gymnasium-Robotics AntMaze_Medium_Diverse_GR-v4 layout.  It is included only
# as a plotting background; no environment is created by this script.
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


def sorted_rows(summary: dict) -> list[dict]:
    rows = sorted(summary["rows"], key=lambda row: int(row["task_id"]))
    task_ids = [int(row["task_id"]) for row in rows]
    if task_ids != list(range(15)):
        raise ValueError(f"Expected task ids 0..14, got {task_ids}")
    return rows


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def outcome_matrix(rows: list[dict]) -> np.ndarray:
    matrix = np.asarray([row["successes"] for row in rows], dtype=np.int8)
    if matrix.shape != (15, 5):
        raise ValueError(f"Expected a 15 x 5 outcome matrix, got {matrix.shape}")
    return matrix


def draw_outcome_heatmap(
    ax: plt.Axes,
    rows: list[dict],
    title: str,
    show_ylabels: bool,
) -> None:
    matrix = outcome_matrix(rows)
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        cmap=ListedColormap(["#f5d8cf", "#b9dfd9"]),
    )
    for task_index in range(matrix.shape[0]):
        for episode_index in range(matrix.shape[1]):
            success = bool(matrix[task_index, episode_index])
            ax.text(
                episode_index,
                task_index,
                "S" if success else "–",
                ha="center",
                va="center",
                color="#0b534d" if success else "#7e321f",
                fontsize=9,
                fontweight="bold" if success else "normal",
            )
    ax.set_xticks(np.arange(5), labels=np.arange(1, 6))
    ax.set_xlabel("Episode")
    ax.set_yticks(np.arange(15))
    if show_ylabels:
        labels = [
            f"T{row['task_id']}  ({row['goal'][0]:.0f}, {row['goal'][1]:.0f})"
            for row in rows
        ]
        ax.set_yticklabels(labels)
        ax.set_ylabel("Task and hidden goal (m)")
    else:
        ax.set_yticklabels([])
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 15, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, fontsize=12, fontweight="semibold")


def plot_outcomes(
    current_rows: list[dict],
    baseline_rows: list[dict],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(13.8, 11.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[3.3, 1.25])
    ax_r39 = fig.add_subplot(grid[0, 0])
    ax_r48 = fig.add_subplot(grid[0, 1])
    ax_rates = fig.add_subplot(grid[1, :])

    draw_outcome_heatmap(ax_r39, baseline_rows, "R39: 6/15 tasks acquired", True)
    draw_outcome_heatmap(ax_r48, current_rows, "R48: 5/15 tasks acquired", False)

    task_ids = np.arange(15)
    baseline_rates = outcome_matrix(baseline_rows).mean(axis=1)
    current_rates = outcome_matrix(current_rows).mean(axis=1)
    width = 0.36
    ax_rates.bar(
        task_ids - width / 2,
        baseline_rates,
        width,
        color=BASELINE,
        label="R39",
    )
    ax_rates.bar(
        task_ids + width / 2,
        current_rates,
        width,
        color=SUCCESS,
        label="R48",
    )
    for task_id, old, new in zip(task_ids, baseline_rates, current_rates):
        if not np.isclose(old, new):
            delta = new - old
            ax_rates.text(
                task_id,
                max(old, new) + 0.055,
                f"{delta:+.1f}",
                ha="center",
                va="bottom",
                color=SUCCESS if delta > 0 else FAILURE,
                fontsize=8.5,
                fontweight="semibold",
            )
    ax_rates.set_xlim(-0.7, 14.7)
    ax_rates.set_ylim(0, 1.18)
    ax_rates.set_xticks(task_ids, labels=[f"T{i}" for i in task_ids])
    ax_rates.set_ylabel("Successful episodes / 5")
    ax_rates.set_xlabel("Task")
    ax_rates.set_title(
        "Per-task end-to-end success; labels show R48 − R39",
        fontsize=11,
        fontweight="semibold",
    )
    ax_rates.grid(axis="y", color=GRID, linewidth=0.7)
    ax_rates.spines[["top", "right"]].set_visible(False)
    ax_rates.legend(frameon=False, ncol=2, loc="upper left")

    fig.suptitle(
        "AntMaze hidden-goal acquisition outcomes",
        fontsize=16,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "r48_acquisition_outcomes")


def maze_cell_center(row: int, column: int) -> np.ndarray:
    map_length, map_width = MAZE_MAP.shape
    center_x = map_width * MAZE_SCALE_M / 2.0
    center_y = map_length * MAZE_SCALE_M / 2.0
    return np.asarray(
        [
            (column + 0.5) * MAZE_SCALE_M - center_x,
            center_y - (row + 0.5) * MAZE_SCALE_M,
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


def plot_task_trajectory(
    ax: plt.Axes,
    task: dict,
    log_rows: list[dict],
    evaluator_contact_radius: float,
) -> None:
    draw_maze(ax)
    acquisition = [row for row in log_rows if row.get("phase") == "acquisition"]
    goal = np.asarray(task["goal"], dtype=float)
    ax.add_patch(
        Circle(
            goal,
            evaluator_contact_radius,
            facecolor=FAILURE,
            edgecolor=FAILURE,
            linewidth=0.7,
            alpha=0.14,
            zorder=2,
        )
    )
    ax.scatter(
        *goal,
        marker="X",
        s=62,
        color=FAILURE,
        edgecolor="white",
        linewidth=0.6,
        zorder=8,
    )

    for episode in sorted({int(row["episode"]) for row in acquisition}):
        episode_rows = [row for row in acquisition if int(row["episode"]) == episode]
        endpoints = np.asarray([row["actual_endpoint"] for row in episode_rows], dtype=float)
        color = EPISODE_COLORS[episode]
        if len(endpoints) > 1:
            ax.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color=color,
                linewidth=1.25,
                alpha=0.88,
                zorder=3,
            )
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=11,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
        )
        ax.scatter(
            *endpoints[0],
            marker="o",
            s=31,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=6,
        )
        contacts = np.asarray([bool(row["reward"]) for row in episode_rows])
        if contacts.any():
            first_contact = int(np.flatnonzero(contacts)[0])
            ax.scatter(
                *endpoints[first_contact],
                marker="*",
                s=105,
                color="#f1b82d",
                edgecolor="#604500",
                linewidth=0.6,
                zorder=9,
            )

    all_endpoints = np.asarray(
        [row["actual_endpoint"] for row in acquisition], dtype=float
    )
    nearest_distance = float(np.min(np.linalg.norm(all_endpoints - goal, axis=1)))
    acquired = task["first_contact_episode"] is not None
    if acquired:
        contact_row = next(row for row in acquisition if bool(row["reward"]))
        subtitle = (
            f"ACQUIRED E{int(contact_row['episode']) + 1}, "
            f"C{int(contact_row['command_index']) + 1} • {len(acquisition)} search cmds"
        )
        title_color = SUCCESS
    else:
        nearest = int(np.argmin(np.linalg.norm(all_endpoints - goal, axis=1)))
        ax.scatter(
            *all_endpoints[nearest],
            marker="D",
            s=36,
            facecolor="#f1b82d",
            edgecolor="#604500",
            linewidth=0.55,
            zorder=8,
        )
        subtitle = f"FAILED • nearest endpoint {nearest_distance:.2f} m • 50 cmds"
        title_color = FAILURE
    ax.set_title(
        f"T{task['task_id']}  goal=({goal[0]:.0f}, {goal[1]:.0f})\n{subtitle}",
        fontsize=8.8,
        color=title_color,
        fontweight="semibold",
    )
    ax.set_xticks([-12, 0, 12])
    ax.set_yticks([-12, 0, 12])
    ax.tick_params(labelsize=7)


def plot_trajectory_atlas(
    results: Path,
    current_rows: list[dict],
    output_dir: Path,
) -> None:
    evaluator_contact_radius = float(
        current_rows[0]["success_radius_evaluator_only_m"]
    )
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(18.2, 12.2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for ax, task in zip(axes.flat, current_rows):
        task_id = int(task["task_id"])
        log_rows = load_jsonl(results / f"task_{task_id:03d}.jsonl")
        plot_task_trajectory(
            ax,
            task,
            log_rows,
            evaluator_contact_radius,
        )

    for row_index, row_axes in enumerate(axes):
        row_axes[0].set_ylabel("y (m)")
        if row_index == len(axes) - 1:
            for ax in row_axes:
                ax.set_xlabel("x (m)")

    legend = [
        Line2D(
            [0], [0], marker="X", color="none", markerfacecolor=FAILURE,
            markeredgecolor="white", markersize=8, label="hidden goal"
        ),
        Line2D(
            [0], [0], marker="o", color=FAILURE, markerfacecolor="none",
            markeredgecolor=FAILURE, markersize=8,
            label="evaluator-only contact radius"
        ),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#f1b82d",
            markeredgecolor="#604500", markersize=11,
            label="endpoint of contact command"
        ),
        Line2D(
            [0], [0], marker="D", color="none", markerfacecolor="#f1b82d",
            markeredgecolor="#604500", markersize=6, label="closest failed endpoint"
        ),
        Line2D(
            [0], [0], marker="o", color=NEUTRAL, markerfacecolor="white",
            markeredgecolor=NEUTRAL, markersize=6, label="first logged endpoint in episode"
        ),
    ]
    legend.extend(
        Line2D([0], [0], color=color, lw=2, label=f"episode {episode + 1}")
        for episode, color in enumerate(EPISODE_COLORS)
    )
    fig.legend(
        handles=legend,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "R48 full-panel acquisition paths: 5 successes and 10 failures\n"
        "Realized 100-step command endpoints; reset segments and primitive paths were not logged",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "r48_acquisition_trajectory_atlas")


def plot_diagnostics(current_rows: list[dict], output_dir: Path) -> None:
    acquired = np.asarray(
        [row["first_contact_episode"] is not None for row in current_rows]
    )
    new_cells = np.asarray(
        [row["acquisition_new_1m_cells_per_command"] for row in current_rows],
        dtype=float,
    )
    revisit = np.asarray(
        [row["acquisition_revisited_endpoint_fraction"] for row in current_rows],
        dtype=float,
    )
    zero_new = np.asarray(
        [row["acquisition_zero_new_1m_cell_fraction"] for row in current_rows],
        dtype=float,
    )
    task_ids = np.arange(15)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), constrained_layout=True)
    colors = np.where(acquired, SUCCESS, FAILURE)
    axes[0].bar(task_ids, new_cells, color=colors)
    axes[0].set_xticks(task_ids, labels=[f"T{i}" for i in task_ids], rotation=45)
    axes[0].set_ylabel("New 1 m cells / command")
    axes[0].set_xlabel("Task")
    axes[0].set_title("Coverage obtained during acquisition", fontweight="semibold")
    axes[0].grid(axis="y", color=GRID, linewidth=0.7)

    for ax, y, ylabel, title in [
        (
            axes[1],
            revisit,
            "Revisited-endpoint fraction",
            "Coverage versus endpoint revisiting",
        ),
        (
            axes[2],
            zero_new,
            "Zero-new-cell command fraction",
            "Coverage versus unproductive commands",
        ),
    ]:
        for task_id in task_ids:
            if acquired[task_id]:
                ax.scatter(
                    new_cells[task_id], y[task_id], s=65, color=SUCCESS,
                    edgecolor="white", linewidth=0.55, zorder=3
                )
            else:
                ax.scatter(
                    new_cells[task_id], y[task_id], s=65, facecolor="white",
                    edgecolor=FAILURE, linewidth=1.4, zorder=3
                )
            ax.annotate(
                f"T{task_id}",
                (new_cells[task_id], y[task_id]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.8,
            )
        ax.set_xlabel("New 1 m cells / command")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="semibold")
        ax.grid(color=GRID, linewidth=0.7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=SUCCESS,
            markeredgecolor="white", markersize=8, label="acquired"
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor=FAILURE, markersize=8, label="not acquired"
        ),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "R48 acquisition coverage and repetition by task",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "r48_acquisition_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results = args.results.resolve()
    baseline = args.baseline.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results / "plots"
    )
    current_rows = sorted_rows(load_json(results / "summary.json"))
    baseline_rows = sorted_rows(load_json(baseline / "summary.json"))

    plot_outcomes(current_rows, baseline_rows, output_dir)
    plot_trajectory_atlas(results, current_rows, output_dir)
    plot_diagnostics(current_rows, output_dir)
    for stem in (
        "r48_acquisition_outcomes",
        "r48_acquisition_trajectory_atlas",
        "r48_acquisition_diagnostics",
    ):
        print(output_dir / f"{stem}.png")
        print(output_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
