"""Area-corrected offline exploration-hub audit on frozen R52 preparation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    plot_r59_hidden_goal_original9 as maze_plot,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r64_r52_offline_hub_ranking as r64,
)


MANAGEMENT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).with_name("R65_R52_AREA_WEIGHTED_HUB_REFERENCE.md")
PROTOCOL = Path(__file__).with_name("r65_r52_area_weighted_hub_protocol.json")
METHOD_REVISION = "antmaze_r52_area_weighted_hub_ranking_r65"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r52_area_weighted_hub_ranking_r65"
)
R64_SUMMARY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r52_offline_hub_ranking_r64/summary.json"
)


def protocol_value() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": (
            "reward_free_offline_area_weighted_hub_feasibility_audit_only"
        ),
        "oracle_inputs": [],
        "preparation_revision": r64.PROTOCOL_VALUE["preparation_revision"],
        "reported_prefix_length": 20,
        "deployed_hub_count": None,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_created_or_stepped": False,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R65 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R65 method reference changed")
    return value


PROTOCOL_VALUE = protocol_value()


def area_weights(
    goal_xy: np.ndarray, sigma: float
) -> tuple[np.ndarray, int, np.ndarray]:
    origin = goal_xy.min(axis=0).astype(np.float64)
    cell = np.floor((goal_xy - origin) / sigma).astype(np.int32)
    _, inverse, count = np.unique(
        cell, axis=0, return_inverse=True, return_counts=True
    )
    occupied = len(count)
    weight = 1.0 / (float(occupied) * count[inverse].astype(np.float64))
    if not np.isclose(weight.sum(), 1.0, atol=1e-12):
        raise RuntimeError("area-corrected prior does not sum to one")
    return weight, occupied, origin


def weighted_greedy_prefix(
    probability: np.ndarray, weight: np.ndarray, length: int
) -> list[dict]:
    covered = np.zeros(probability.shape[0], np.float32)
    selected = np.zeros(probability.shape[1], bool)
    result = []
    for rank in range(int(length)):
        gain = np.empty(probability.shape[1], np.float64)
        for start in range(0, probability.shape[1], 128):
            stop = min(start + 128, probability.shape[1])
            delta = np.maximum(
                probability[:, start:stop] - covered[:, None], 0.0
            )
            gain[start:stop] = np.sum(
                delta.astype(np.float64) * weight[:, None], axis=0
            )
        gain[selected] = -np.inf
        index = int(np.argmax(gain))
        if not np.isfinite(gain[index]):
            raise RuntimeError("weighted hub ranking exhausted")
        covered = np.maximum(covered, probability[:, index])
        selected[index] = True
        result.append({
            "rank": rank + 1,
            "candidate_index": index,
            "marginal_contact_probability": float(gain[index]),
            "cumulative_contact_probability": float(
                np.sum(covered.astype(np.float64) * weight)
            ),
        })
    return result


def pairwise_spacing(rows: list[dict]) -> dict:
    xy = np.asarray([row["xy"] for row in rows], np.float64)
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    distance[np.eye(len(xy), dtype=bool)] = np.inf
    nearest = distance.min(axis=1)
    return {
        "nearest_hub_distance_minimum_m": float(nearest.min()),
        "nearest_hub_distance_median_m": float(np.median(nearest)),
        "nearest_hub_distance_mean_m": float(nearest.mean()),
    }


def plot_result(output: Path, node_xy: np.ndarray, rows: list[dict]) -> None:
    old = json.loads(R64_SUMMARY.read_text())["hubs"]
    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, 7.2), constrained_layout=True
    )
    for ax, values, title in (
        (axes[0], old, "R64: each recorded coordinate equal"),
        (axes[1], rows, "R65: each occupied $\\sigma$-cell equal"),
    ):
        maze_plot.draw_maze(ax)
        ax.scatter(
            node_xy[:, 0], node_xy[:, 1], s=1.3, color="#9ca4ae",
            alpha=0.10, linewidths=0, zorder=1,
        )
        xy = np.asarray([row["xy"] for row in values], np.float64)
        reach = np.asarray([row["reachability_mean"] for row in values])
        marginal = np.asarray(
            [row["marginal_contact_probability"] for row in values]
        )
        size = 55.0 + 580.0 * marginal / max(float(marginal.max()), 1e-12)
        points = ax.scatter(
            xy[:, 0], xy[:, 1], c=reach, s=size, cmap="viridis",
            vmin=0.0, vmax=1.0, edgecolor="white", linewidth=0.8, zorder=5,
        )
        for row in values:
            ax.annotate(
                str(row["rank"]), row["xy"], ha="center", va="center",
                fontsize=7.2, fontweight="bold", color="black", zorder=6,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.colorbar(points, ax=axes, fraction=0.026, pad=0.02).set_label(
        "Mean 10-command reachability from R52 preparation resets"
    )
    fig.suptitle(
        "R65 area correction removes preparation-visitation mass bias\n"
        "Circle area = marginal expected contact; color = reachability",
        fontsize=14, fontweight="bold",
    )
    stem = output / "r65_r52_area_weighted_hub_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def audit(args) -> dict:
    base = r64.command_audit(args)
    report = {
        **base,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r64.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "area_weighting": PROTOCOL_VALUE["goal_prior"],
    }
    print(json.dumps(report, indent=2))
    return report


def run(args) -> dict:
    conformity = audit(args)
    if not conformity["ready"]:
        raise FileNotFoundError("R52 preparation is incomplete")
    root = Path(args.preparation).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = r64.preparation_paths(root)
    consequence = json.loads(paths["consequence_summary"].read_text())
    collection = json.loads(paths["collection_summary"].read_text())
    sigma = float(consequence["landing_sigma_m"])
    horizon = int(collection["commands_per_physical_episode"])
    with np.load(paths["graph"], allow_pickle=False) as graph:
        nodes = graph["nodes"].astype(np.float32)
        source = graph["source_index"].astype(np.int32)
        endpoint = graph["endpoint_index"].astype(np.int32)
        edge_weight = graph["edge_reliability"].astype(np.float32)
    with np.load(paths["bridges"], allow_pickle=False) as bridges:
        bridge_target = bridges["target_index"].astype(np.int32)
        bridge_weight = bridges["weight"].astype(np.float32)

    goal_xy = np.unique(nodes[:, :2].astype(np.float64), axis=0)
    goal_weight, occupied_cells, weight_origin = area_weights(goal_xy, sigma)
    rho_max = r64.exact_support_diameter(goal_xy)
    representative, _, alias_origin = r64.alias_hub_candidates(
        nodes[:, :2], sigma
    )
    hub_xy = nodes[representative, :2].astype(np.float64)
    probability = r64.contact_matrix(goal_xy, hub_xy, sigma, rho_max)
    ranked = weighted_greedy_prefix(
        probability, goal_weight, PROTOCOL_VALUE["reported_prefix_length"]
    )
    reset_nodes, reset_error = r64.reset_source_nodes(
        paths["base_collection"], source, nodes
    )
    for row in ranked:
        candidate = int(row.pop("candidate_index"))
        node_index = int(representative[candidate])
        row.update({
            "node_index": node_index,
            "xy": nodes[node_index, :2].astype(float).tolist(),
            **r64.hub_reachability(
                node_index, horizon, source, endpoint, edge_weight,
                bridge_target, bridge_weight, reset_nodes,
            ),
        })
        print(json.dumps(row), flush=True)

    xy = np.asarray([row["xy"] for row in ranked], np.float64)
    quadrants = {(int(x >= 0), int(y >= 0)) for x, y in xy}
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r64.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "preparation": str(root),
        "preparation_revision": PROTOCOL_VALUE["preparation_revision"],
        "goal_prior": PROTOCOL_VALUE["goal_prior"],
        "unique_goal_coordinates": int(len(goal_xy)),
        "occupied_goal_prior_cells": int(occupied_cells),
        "goal_prior_weight_sum": float(goal_weight.sum()),
        "goal_prior_weight_minimum": float(goal_weight.min()),
        "goal_prior_weight_maximum": float(goal_weight.max()),
        "goal_prior_grid_origin_data_derived_m": weight_origin.tolist(),
        "hub_candidate_source": PROTOCOL_VALUE["hub_candidate_source"],
        "hub_candidates": int(len(representative)),
        "landing_sigma_m": sigma,
        "rho_min_m": sigma,
        "rho_max_m": rho_max,
        "rho_prior": PROTOCOL_VALUE["rho_prior"],
        "ranking": PROTOCOL_VALUE["ranking"],
        "reported_prefix_length": len(ranked),
        "deployed_hub_count": None,
        "reachability_horizon_commands": horizon,
        "reachability_reset_states": int(len(reset_nodes)),
        "reset_source_reconstruction_max_error": reset_error,
        "alias_grid_origin_data_derived_m": alias_origin.tolist(),
        "descriptive_spread": {
            "x_span_m": float(np.ptp(xy[:, 0])),
            "y_span_m": float(np.ptp(xy[:, 1])),
            "occupied_xy_quadrants": int(len(quadrants)),
            **pairwise_spacing(ranked),
        },
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_created_or_stepped": False,
        "confirmation_panel_accessed": False,
        "hubs": ranked,
    }
    r64.dump(output / "summary.json", result)
    with (output / "ranked_hubs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "rank", "node_index", "x", "y",
                "marginal_contact_probability",
                "cumulative_contact_probability", "reachability_mean",
                "reachability_median", "reachability_p10",
                "reachability_minimum", "reachability_maximum",
            ],
        )
        writer.writeheader()
        for row in ranked:
            value = dict(row)
            value["x"], value["y"] = value.pop("xy")
            writer.writerow(value)
    if not R64_SUMMARY.is_file():
        raise FileNotFoundError("run R64 before the R65 comparison plot")
    plot_result(output, nodes[:, :2], ranked)
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "png": str(output / "r65_r52_area_weighted_hub_comparison.png"),
        "pdf": str(output / "r65_r52_area_weighted_hub_comparison.pdf"),
        "descriptive_spread": result["descriptive_spread"],
        "cumulative_contact_probability_at_20": (
            ranked[-1]["cumulative_contact_probability"]
        ),
        "top20_reachability_mean": float(np.mean([
            row["reachability_mean"] for row in ranked
        ])),
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preparation", type=Path, default=r64.DEFAULT_PREPARATION)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=audit)
    commands.add_parser("run").set_defaults(func=run)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
