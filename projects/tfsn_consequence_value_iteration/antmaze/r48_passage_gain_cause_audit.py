"""Offline mechanism test for R48 passage-gain overprediction.

The audit asks whether commands selected for high expected passage gain realize
little new ground, especially when their endpoints revisit old cells, and
whether that failure grows with the selected command's empirical-passage match
error.  It consumes completed JSON/JSONL logs and frozen graph artifacts only;
it never constructs or steps an environment.

PointMaze R11 is included as a cross-domain check.  Its saved shards retain
exact newly visited maze cells per episode but not primitive trajectories or
new-cell counts per command, so that comparison is necessarily episode-level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DEFAULT_ANT_EVALUATION = (
    RESULTS
    / "antmaze"
    / "r39_r46_rule_removal_r48_full15"
    / "evaluation_all15"
)
DEFAULT_POINT_EVALUATION = (
    RESULTS
    / "pointmaze"
    / "continuous_reference_hidden_goal_r11_disjoint15_e120"
)

SUCCESS = "#16837a"
FAILURE = "#c75b39"
BLUE = "#5072a7"
PURPLE = "#7655a5"
GRID = "#d9dde3"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def fractional_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=float)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def add_within_task_rank_gap(rows: list[dict]) -> None:
    tasks = sorted({str(row["task_key"]) for row in rows})
    for task in tasks:
        indices = [i for i, row in enumerate(rows) if str(row["task_key"]) == task]
        predicted = np.asarray([rows[i]["predicted_gain"] for i in indices])
        realized = np.asarray([rows[i]["realized_new_ground"] for i in indices])
        predicted_rank = fractional_rank(predicted)
        realized_rank = fractional_rank(realized)
        for local, index in enumerate(indices):
            rows[index]["predicted_gain_rank_within_task"] = float(
                predicted_rank[local]
            )
            rows[index]["realized_gain_rank_within_task"] = float(
                realized_rank[local]
            )
            rows[index]["overprediction_rank_gap"] = float(
                predicted_rank[local] - realized_rank[local]
            )


def stratified_label_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    values = np.asarray(values, float)
    labels = np.asarray(labels, bool)
    strata = np.asarray(strata)
    if not labels.any() or labels.all():
        return {
            "difference_label_true_minus_false": None,
            "two_sided_permutation_p": None,
            "permutations": int(permutations),
        }
    observed = float(values[labels].mean() - values[~labels].mean())
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == key) for key in np.unique(strata)]
    exceed = 0
    permuted = labels.copy()
    for _ in range(int(permutations)):
        for indices in groups:
            permuted[indices] = rng.permutation(labels[indices])
        difference = float(
            values[permuted].mean() - values[~permuted].mean()
        )
        exceed += int(abs(difference) >= abs(observed))
    return {
        "difference_label_true_minus_false": observed,
        "two_sided_permutation_p": float((exceed + 1) / (permutations + 1)),
        "permutations": int(permutations),
    }


def stratified_rank_association(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    strata = np.asarray(strata)
    x_rank = np.zeros(len(x), float)
    y_rank = np.zeros(len(y), float)
    groups = [np.flatnonzero(strata == key) for key in np.unique(strata)]
    for indices in groups:
        x_rank[indices] = fractional_rank(x[indices])
        y_rank[indices] = fractional_rank(y[indices])
    observed = float(np.corrcoef(x_rank, y_rank)[0, 1])
    rng = np.random.default_rng(seed)
    permuted = y_rank.copy()
    exceed = 0
    for _ in range(int(permutations)):
        for indices in groups:
            permuted[indices] = rng.permutation(y_rank[indices])
        coefficient = float(np.corrcoef(x_rank, permuted)[0, 1])
        exceed += int(abs(coefficient) >= abs(observed))
    return {
        "within_task_rank_correlation": observed,
        "two_sided_permutation_p": float((exceed + 1) / (permutations + 1)),
        "permutations": int(permutations),
    }


def infer_ant_start(
    row: dict,
    previous_by_episode: dict[int, tuple[int, np.ndarray]],
) -> np.ndarray | None:
    episode = int(row["episode"])
    command_index = int(row["command_index"])
    previous = previous_by_episode.get(episode)
    if previous is None or previous[0] != command_index - 1:
        return None
    return previous[1].copy()


def ant_rows(evaluation: Path) -> tuple[list[dict], dict]:
    summary_path = evaluation / "summary.json"
    summary = load_json(summary_path)
    sigma = float(summary["rows"][0]["landing_sigma_m"])
    graph_path = evaluation.parent / "recorded_transition_graph.npz"
    with np.load(graph_path) as stored:
        nodes = stored["nodes"].astype(np.float32)
        source_index = stored["source_index"].astype(np.int32)
        endpoint_index = stored["endpoint_index"].astype(np.int32)
    recorded_start = nodes[source_index, :2]
    recorded_displacement = nodes[endpoint_index, :2] - recorded_start
    passage_tree = cKDTree(
        np.concatenate((recorded_start, recorded_displacement), axis=1) / sigma
    )

    rows: list[dict] = []
    log_paths = sorted(evaluation.glob("task_*.jsonl"))
    for path in log_paths:
        task_id = int(path.stem.split("_")[-1])
        previous_by_episode: dict[int, tuple[int, np.ndarray]] = {}
        for logged in load_jsonl(path):
            episode = int(logged["episode"])
            command_index = int(logged["command_index"])
            endpoint = np.asarray(logged["actual_endpoint"], dtype=float)
            start = infer_ant_start(logged, previous_by_episode)
            previous_by_episode[episode] = (command_index, endpoint)
            if logged.get("phase") != "acquisition":
                continue
            predicted = float(logged["selected_expected_posterior_mass_fraction"])
            residual_area = float(logged["residual_reachable_mass_m2"])
            realized_cells = int(logged["new_1m_cells"])
            row = {
                "domain": "antmaze",
                "task_key": str(task_id),
                "task_id": task_id,
                "episode": episode,
                "command_index": command_index,
                "selected_kind": str(logged["selected_kind"]),
                "predicted_gain": predicted,
                "realized_new_ground": realized_cells,
                "residual_reachable_mass_m2": residual_area,
                "realized_new_area_fraction": (
                    realized_cells / residual_area if residual_area > 0 else None
                ),
                "raw_fraction_overprediction": (
                    predicted - realized_cells / residual_area
                    if residual_area > 0 else None
                ),
                "revisited_endpoint": bool(
                    logged["revisited_endpoint_1m_cell"]
                ),
                "zero_new_ground": realized_cells == 0,
                "selected_connection_weight": float(
                    logged["selected_connection_weight"]
                ),
                "selected_command_distance_m": float(
                    logged["selected_command_distance"]
                ),
                "match_geometry_available": False,
                "joint_match_distance_sigma": None,
                "source_match_error_m": None,
                "displacement_match_error_m": None,
            }
            if start is not None and row["selected_kind"] == "model_bridge":
                command = np.asarray(logged["command"], dtype=float)
                displacement = command - start
                query = np.concatenate((start, displacement)) / sigma
                distance, edge = passage_tree.query(query, k=1)
                edge = int(edge)
                row.update({
                    "match_geometry_available": True,
                    "joint_match_distance_sigma": float(distance),
                    "source_match_error_m": float(
                        np.linalg.norm(start - recorded_start[edge])
                    ),
                    "displacement_match_error_m": float(
                        np.linalg.norm(
                            displacement - recorded_displacement[edge]
                        )
                    ),
                })
            rows.append(row)
    add_within_task_rank_gap(rows)
    provenance = {
        "evaluation_summary": str(summary_path.resolve()),
        "evaluation_summary_sha256": sha256(summary_path),
        "command_logs": len(log_paths),
        "command_logs_combined_sha256": directory_digest(log_paths),
        "recorded_transition_graph": str(graph_path.resolve()),
        "recorded_transition_graph_sha256": sha256(graph_path),
        "landing_sigma_m": sigma,
    }
    return rows, provenance


def summarize_ant(rows: list[dict], permutations: int) -> dict:
    predicted = np.asarray([row["predicted_gain"] for row in rows])
    realized = np.asarray([row["realized_new_ground"] for row in rows])
    rank_gap = np.asarray([row["overprediction_rank_gap"] for row in rows])
    revisit = np.asarray([row["revisited_endpoint"] for row in rows])
    task = np.asarray([row["task_id"] for row in rows])
    raw_gap = np.asarray([row["raw_fraction_overprediction"] for row in rows])
    match_rows = [row for row in rows if row["match_geometry_available"]]
    match_error = np.asarray(
        [row["joint_match_distance_sigma"] for row in match_rows]
    )
    match_rank_gap = np.asarray(
        [row["overprediction_rank_gap"] for row in match_rows]
    )
    match_task = np.asarray([row["task_id"] for row in match_rows])
    match_revisit = np.asarray(
        [row["revisited_endpoint"] for row in match_rows]
    )

    overall_spearman = spearmanr(predicted, realized)
    revisit_gap_test = stratified_label_permutation(
        rank_gap, revisit, task, permutations, seed=4811
    )
    raw_revisit_test = stratified_label_permutation(
        raw_gap, revisit, task, permutations, seed=4812
    )
    match_gap_test = stratified_rank_association(
        match_error, match_rank_gap, match_task, permutations, seed=4813
    )
    match_revisit_test = stratified_label_permutation(
        match_error, match_revisit, match_task, permutations, seed=4814
    )

    task_rows = []
    for task_id in sorted(set(task.tolist())):
        selected = task == task_id
        local_revisit = revisit[selected]
        local_rank_gap = rank_gap[selected]
        local_raw_gap = raw_gap[selected]
        local_match = [row for row in match_rows if row["task_id"] == task_id]
        revisit_rank_difference = (
            float(
                local_rank_gap[local_revisit].mean()
                - local_rank_gap[~local_revisit].mean()
            )
            if local_revisit.any() and (~local_revisit).any() else None
        )
        revisit_raw_difference = (
            float(
                local_raw_gap[local_revisit].mean()
                - local_raw_gap[~local_revisit].mean()
            )
            if local_revisit.any() and (~local_revisit).any() else None
        )
        local_match_correlation = (
            float(spearmanr(
                [row["joint_match_distance_sigma"] for row in local_match],
                [row["overprediction_rank_gap"] for row in local_match],
            ).statistic)
            if len(local_match) >= 3 else None
        )
        task_rows.append({
            "task_id": int(task_id),
            "commands": int(selected.sum()),
            "revisited_endpoint_fraction": float(local_revisit.mean()),
            "zero_new_ground_fraction": float((realized[selected] == 0).mean()),
            "mean_predicted_gain": float(predicted[selected].mean()),
            "mean_realized_new_1m_cells": float(realized[selected].mean()),
            "mean_overprediction_rank_gap_on_revisits": (
                float(local_rank_gap[local_revisit].mean())
                if local_revisit.any() else None
            ),
            "revisit_overprediction_rank_gap_difference": revisit_rank_difference,
            "revisit_raw_fraction_overprediction_difference": revisit_raw_difference,
            "selected_match_error_vs_overprediction_spearman": (
                local_match_correlation
            ),
            "median_joint_match_distance_sigma": (
                float(np.median([
                    row["joint_match_distance_sigma"] for row in local_match
                ])) if local_match else None
            ),
            "p90_joint_match_distance_sigma": (
                float(np.quantile([
                    row["joint_match_distance_sigma"] for row in local_match
                ], 0.9)) if local_match else None
            ),
        })

    supports_revisit_overprediction = bool(
        revisit_gap_test["difference_label_true_minus_false"] > 0
        and revisit_gap_test["two_sided_permutation_p"] < 0.05
    )
    supports_match_error_mechanism = bool(
        match_gap_test["within_task_rank_correlation"] > 0
        and match_gap_test["two_sided_permutation_p"] < 0.05
    )
    supports_match_error_enrichment_on_revisits = bool(
        match_revisit_test["difference_label_true_minus_false"] > 0
        and match_revisit_test["two_sided_permutation_p"] < 0.05
    )
    if (
        supports_revisit_overprediction
        and supports_match_error_mechanism
        and supports_match_error_enrichment_on_revisits
    ):
        verdict = "strong_support_for_matching_as_the_primary_revisit_mechanism"
    elif supports_revisit_overprediction and supports_match_error_mechanism:
        verdict = (
            "supports_matching_error_as_an_overprediction_contributor_but_"
            "not_a_complete_explanation_of_revisiting"
        )
    elif not supports_revisit_overprediction and not supports_match_error_mechanism:
        verdict = "does_not_support_passage_matching_as_the_primary_mechanism"
    else:
        verdict = "mixed_evidence_mechanism_not_settled"

    return {
        "commands": len(rows),
        "commands_with_selected_match_geometry": len(match_rows),
        "predicted_gain_definition": (
            "selected_expected_posterior_mass_fraction_from_the_R48_log"
        ),
        "realized_gain_definition": (
            "new_1m_cells_swept_by_the_actual_primitive_trajectory"
        ),
        "unit_free_primary_error": (
            "within_task_percentile_rank(predicted_gain)-"
            "within_task_percentile_rank(realized_new_1m_cells)"
        ),
        "raw_fraction_secondary_error": (
            "predicted_gain-new_1m_cells/residual_reachable_mass_m2;_"
            "secondary_because_predicted_gain_is_posterior_mass_while_"
            "new_cells_is_unweighted_area"
        ),
        "predicted_vs_realized_spearman": {
            "rho": float(overall_spearman.statistic),
            "p": float(overall_spearman.pvalue),
        },
        "revisit_overprediction_rank_gap_test": revisit_gap_test,
        "revisit_raw_fraction_overprediction_test": raw_revisit_test,
        "selected_match_error_vs_overprediction_test": match_gap_test,
        "selected_match_error_on_revisit_test": match_revisit_test,
        "task_diagnostics": task_rows,
        "predeclared_focus_tasks": [1, 4],
        "supports_revisit_overprediction": supports_revisit_overprediction,
        "supports_match_error_mechanism": supports_match_error_mechanism,
        "supports_match_error_enrichment_on_revisits": (
            supports_match_error_enrichment_on_revisits
        ),
        "verdict": verdict,
    }


def point_rows(evaluation: Path) -> tuple[list[dict], dict]:
    summary_path = evaluation / "summary.json"
    summary = load_json(summary_path)
    task_paths = sorted((evaluation / "tasks").glob("*.json"))
    rows: list[dict] = []
    for path in task_paths:
        task = load_json(path)
        episodes = {int(row["episode"]): row for row in task["episodes"]}
        decisions_by_episode: dict[int, list[dict]] = {}
        for decision in task["decisions"]:
            if decision.get("phase") != "acquisition":
                continue
            if "selected_first_passage_gain" not in decision:
                continue
            decisions_by_episode.setdefault(
                int(decision["episode"]), []
            ).append(decision)
        for episode, decisions in sorted(decisions_by_episode.items()):
            episode_row = episodes[episode]
            predicted = np.asarray(
                [row["selected_first_passage_gain"] for row in decisions],
                dtype=float,
            )
            rows.append({
                "domain": "pointmaze",
                "task_key": str(task["goal_name"]),
                "goal_name": str(task["goal_name"]),
                "episode": int(episode),
                "acquisition_commands": len(decisions),
                "predicted_gain": float(predicted.sum()),
                "mean_predicted_gain_per_command": float(predicted.mean()),
                "realized_new_ground": int(
                    episode_row["newly_visited_maze_cell_count"]
                ),
                "realized_new_ground_per_command": float(
                    episode_row["newly_visited_maze_cell_count"] / len(decisions)
                ),
                "zero_new_ground": (
                    int(episode_row["newly_visited_maze_cell_count"]) == 0
                ),
            })
    add_within_task_rank_gap(rows)
    provenance = {
        "evaluation_summary": str(summary_path.resolve()),
        "evaluation_summary_sha256": sha256(summary_path),
        "task_shards": len(task_paths),
        "task_shards_combined_sha256": directory_digest(task_paths),
        "method_revision": summary["method_revision"],
        "acquisition_at_5": summary["metrics"]["acquisition_at_5"],
    }
    return rows, provenance


def summarize_point(rows: list[dict]) -> dict:
    predicted = np.asarray([row["predicted_gain"] for row in rows])
    realized = np.asarray([row["realized_new_ground"] for row in rows])
    predicted_rate = np.asarray(
        [row["mean_predicted_gain_per_command"] for row in rows]
    )
    realized_rate = np.asarray(
        [row["realized_new_ground_per_command"] for row in rows]
    )
    episode_correlation = spearmanr(predicted, realized)
    rate_correlation = spearmanr(predicted_rate, realized_rate)
    threshold = float(np.quantile(predicted, 0.75))
    high = predicted >= threshold
    return {
        "acquisition_episodes": len(rows),
        "comparison_level": "episode",
        "predicted_gain_definition": (
            "sum_of_selected_first_passage_gain_over_acquisition_commands"
        ),
        "realized_gain_definition": (
            "exact_newly_visited_maze_cells_from_primitive_trajectory_"
            "aggregated_per_episode"
        ),
        "exact_command_level_cross_check_available": False,
        "limitation": (
            "R11_shards_do_not_retain_primitive_trajectory_or_new_ground_"
            "per_command;_do_not_interpret_this_as_command_level_calibration"
        ),
        "predicted_sum_vs_realized_cells_spearman": {
            "rho": float(episode_correlation.statistic),
            "p": float(episode_correlation.pvalue),
        },
        "predicted_mean_vs_realized_cells_per_command_spearman": {
            "rho": float(rate_correlation.statistic),
            "p": float(rate_correlation.pvalue),
        },
        "top_predicted_quartile_threshold": threshold,
        "top_predicted_quartile_zero_new_ground_fraction": float(
            np.mean(realized[high] == 0)
        ),
        "all_episode_zero_new_ground_fraction": float(np.mean(realized == 0)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_report(
    ant: list[dict],
    point: list[dict],
    ant_summary: dict,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.7), constrained_layout=True)

    predicted = np.asarray([row["predicted_gain"] for row in ant])
    realized_fraction = np.asarray(
        [row["realized_new_area_fraction"] for row in ant]
    )
    revisit = np.asarray([row["revisited_endpoint"] for row in ant])
    task = np.asarray([row["task_id"] for row in ant])
    for label, selected, color, marker in [
        ("fresh endpoint", ~revisit, SUCCESS, "o"),
        ("revisited endpoint", revisit, FAILURE, "x"),
    ]:
        axes[0, 0].scatter(
            predicted[selected],
            realized_fraction[selected],
            s=23,
            alpha=0.62,
            color=color,
            marker=marker,
            label=label,
        )
    bound = max(float(predicted.max()), float(realized_fraction.max()))
    axes[0, 0].plot([0, bound], [0, bound], color="#68717d", lw=1, ls="--")
    axes[0, 0].set_xlabel("Predicted posterior-mass fraction")
    axes[0, 0].set_ylabel("New 1 m cells / residual area")
    axes[0, 0].set_title("AntMaze: prediction versus new ground")
    axes[0, 0].legend(frameon=False)

    rank_gap = np.asarray([row["overprediction_rank_gap"] for row in ant])
    axes[0, 1].boxplot(
        [rank_gap[~revisit], rank_gap[revisit]],
        tick_labels=["fresh", "revisited"],
        patch_artist=True,
        boxprops={"facecolor": "#d7ebe7"},
        medianprops={"color": "#2f3741"},
    )
    axes[0, 1].axhline(0, color="#68717d", lw=1)
    effect = ant_summary["revisit_overprediction_rank_gap_test"]
    axes[0, 1].set_ylabel("Predicted rank − realized rank")
    axes[0, 1].set_title(
        "Overprediction on revisited endpoints\n"
        f"difference={effect['difference_label_true_minus_false']:.3f}, "
        f"p={effect['two_sided_permutation_p']:.3g}"
    )

    matched = [row for row in ant if row["match_geometry_available"]]
    match_error = np.asarray(
        [row["joint_match_distance_sigma"] for row in matched]
    )
    match_gap = np.asarray(
        [row["overprediction_rank_gap"] for row in matched]
    )
    match_task = np.asarray([row["task_id"] for row in matched])
    colors = np.where(np.isin(match_task, [1, 4]), FAILURE, BLUE)
    axes[0, 2].scatter(match_error, match_gap, s=20, alpha=0.55, color=colors)
    bins = np.quantile(match_error, np.linspace(0, 1, 6))
    centers, medians = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (match_error >= low) & (match_error <= high)
        centers.append(float(np.median(match_error[selected])))
        medians.append(float(np.median(match_gap[selected])))
    axes[0, 2].plot(centers, medians, color="#202832", marker="o", lw=2)
    association = ant_summary["selected_match_error_vs_overprediction_test"]
    axes[0, 2].axhline(0, color="#68717d", lw=1)
    axes[0, 2].set_xlabel("Selected passage match error (σ units)")
    axes[0, 2].set_ylabel("Predicted rank − realized rank")
    axes[0, 2].set_title(
        "Does mismatch increase overprediction?\n"
        f"within-task ρ={association['within_task_rank_correlation']:.3f}, "
        f"p={association['two_sided_permutation_p']:.3g}"
    )

    task_ids = np.arange(15)
    task_gap = []
    for task_id in task_ids:
        selected = task == task_id
        local_revisit = revisit[selected]
        local_gap = rank_gap[selected]
        task_gap.append(
            local_gap[local_revisit].mean() - local_gap[~local_revisit].mean()
            if local_revisit.any() and (~local_revisit).any() else 0.0
        )
    task_gap = np.asarray(task_gap)
    task_colors = [FAILURE if task_id in (1, 4) else BLUE for task_id in task_ids]
    axes[1, 0].bar(task_ids, task_gap, color=task_colors)
    axes[1, 0].axhline(0, color="#68717d", lw=1)
    axes[1, 0].set_xticks(task_ids, labels=[f"T{i}" for i in task_ids], rotation=45)
    axes[1, 0].set_ylabel("Revisit gap − fresh gap")
    axes[1, 0].set_title("AntMaze revisit-conditioned overprediction by task")

    point_predicted = np.asarray([row["predicted_gain"] for row in point])
    point_realized = np.asarray([row["realized_new_ground"] for row in point])
    point_zero = point_realized == 0
    axes[1, 1].scatter(
        point_predicted[~point_zero], point_realized[~point_zero],
        color=SUCCESS, s=38, alpha=0.75, label="new cells"
    )
    axes[1, 1].scatter(
        point_predicted[point_zero], point_realized[point_zero],
        facecolor="white", edgecolor=FAILURE, s=48, label="zero new cells"
    )
    axes[1, 1].set_xlabel("Sum of predicted first-passage gain")
    axes[1, 1].set_ylabel("New maze cells in episode")
    axes[1, 1].set_title("PointMaze R11 episode-level cross-check")
    axes[1, 1].legend(frameon=False)

    point_rate_predicted = np.asarray(
        [row["mean_predicted_gain_per_command"] for row in point]
    )
    point_rate_realized = np.asarray(
        [row["realized_new_ground_per_command"] for row in point]
    )
    axes[1, 2].scatter(
        point_rate_predicted,
        point_rate_realized,
        color=PURPLE,
        s=40,
        alpha=0.72,
    )
    axes[1, 2].set_xlabel("Mean predicted passage gain / command")
    axes[1, 2].set_ylabel("New maze cells / command")
    axes[1, 2].set_title("PointMaze rate-normalized cross-check")

    for ax in axes.flat:
        ax.grid(color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Passage-gain cause audit: does predicted gain become revisiting?",
        fontsize=16,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ant-evaluation", type=Path, default=DEFAULT_ANT_EVALUATION)
    parser.add_argument("--point-evaluation", type=Path, default=DEFAULT_POINT_EVALUATION)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=20_000)
    args = parser.parse_args()

    ant_evaluation = args.ant_evaluation.resolve()
    point_evaluation = args.point_evaluation.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else ant_evaluation.parent / "diagnostics" / "passage_gain_cause_audit.json"
    )
    ant, ant_provenance = ant_rows(ant_evaluation)
    point, point_provenance = point_rows(point_evaluation)
    ant_summary = summarize_ant(ant, int(args.permutations))
    point_summary = summarize_point(point)
    report = {
        "schema_version": 1,
        "experiment_id": "r48_passage_gain_cause_audit",
        "audit_type": "offline_executed_command_mechanism_test",
        "environment_steps": 0,
        "task_reward_used_for_fitting": False,
        "hidden_goal_coordinates_used": False,
        "causal_scope": (
            "mechanism_consistency_and_falsification_on_executed_commands;_"
            "not_randomized_causal_identification"
        ),
        "antmaze_provenance": ant_provenance,
        "pointmaze_provenance": point_provenance,
        "antmaze": ant_summary,
        "pointmaze_cross_check": point_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(output.with_name("passage_gain_cause_ant_commands.csv"), ant)
    write_csv(output.with_name("passage_gain_cause_point_episodes.csv"), point)
    plot_report(ant, point, ant_summary, output.with_suffix(""))
    print(json.dumps(report, indent=2))
    print(output)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
