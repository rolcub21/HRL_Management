#!/usr/bin/env python3
"""Rank latent-coverage hubs and audit R52 graph reachability offline."""

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

from projects.tfsn_consequence_value_iteration.antmaze import (
    plot_r59_hidden_goal_original9 as maze_plot,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    exact_support_diameter,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R64_R52_OFFLINE_HUB_RANKING_REFERENCE.md"
PROTOCOL = HERE / "r64_r52_offline_hub_ranking_protocol.json"
METHOD_REVISION = "antmaze_r52_offline_latent_coverage_hub_ranking_r64"
DEFAULT_PREPARATION = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_preparation_r52"
)
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r52_offline_hub_ranking_r64"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "oracle_inputs": [],
        "preparation_revision": "antmaze_e549_5m_preparation_r52",
        "goal_prior": "uniform_over_unique_recorded_roadmap_xy",
        "rho_prior": "log_uniform",
        "reported_prefix_length": 20,
        "deployed_hub_count": None,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_created_or_stepped": False,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R64 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R64 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def preparation_paths(root: Path) -> dict[str, Path]:
    runtime = root / "runtime"
    return {
        "collection_summary": root / "collection/summary.json",
        "base_collection": root / "collection/base_graph",
        "consequence_summary": root / "consequence/summary.json",
        "graph": runtime / "recorded_transition_graph.npz",
        "graph_metadata": runtime / "recorded_transition_graph.json",
        "bridges": runtime / "planner/bridges.npz",
        "bridges_metadata": runtime / "planner/bridges.json",
    }


def command_audit(args) -> dict:
    root = Path(args.preparation).resolve()
    paths = preparation_paths(root)
    required = list(paths.values())
    revisions = {}
    for key in ("collection_summary", "consequence_summary", "graph_metadata", "bridges_metadata"):
        if paths[key].is_file():
            revisions[key] = json.loads(paths[key].read_text()).get(
                "method_revision"
            )
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "preparation": str(root),
        "preparation_revisions": revisions,
        "required_artifacts": [
            {"path": str(path), "exists": path.exists()}
            for path in required
        ],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_created_or_stepped": False,
        "ready": bool(
            all(path.exists() for path in required)
            and all(
                item == PROTOCOL_VALUE["preparation_revision"]
                for item in revisions.values()
            )
        ),
    }
    print(json.dumps(report, indent=2))
    return report


def alias_hub_candidates(
    node_xy: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = node_xy.min(axis=0).astype(np.float64)
    cell = np.floor((node_xy.astype(np.float64) - origin) / sigma).astype(
        np.int32
    )
    _, first, inverse = np.unique(
        cell, axis=0, return_index=True, return_inverse=True
    )
    return first.astype(np.int32), inverse.astype(np.int32), origin


def contact_matrix(
    goal_xy: np.ndarray,
    hub_xy: np.ndarray,
    rho_min: float,
    rho_max: float,
) -> np.ndarray:
    width = float(np.log(rho_max / rho_min))
    output = np.empty((len(goal_xy), len(hub_xy)), np.float32)
    for start in range(0, len(hub_xy), 128):
        stop = min(start + 128, len(hub_xy))
        distance = np.linalg.norm(
            goal_xy[:, None, :] - hub_xy[None, start:stop, :], axis=2
        )
        value = np.zeros_like(distance, np.float64)
        certain = distance <= rho_min
        possible = (distance > rho_min) & (distance < rho_max)
        value[certain] = 1.0
        value[possible] = np.log(rho_max / distance[possible]) / width
        output[:, start:stop] = np.clip(value, 0.0, 1.0).astype(
            np.float32
        )
    return output


def greedy_prefix(probability: np.ndarray, length: int) -> list[dict]:
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
            gain[start:stop] = delta.mean(axis=0, dtype=np.float64)
        gain[selected] = -np.inf
        index = int(np.argmax(gain))
        if not np.isfinite(gain[index]):
            raise RuntimeError("hub ranking exhausted before requested prefix")
        covered = np.maximum(covered, probability[:, index])
        selected[index] = True
        result.append(
            {
                "rank": rank + 1,
                "candidate_index": index,
                "marginal_contact_probability": float(gain[index]),
                "cumulative_contact_probability": float(
                    covered.mean(dtype=np.float64)
                ),
            }
        )
    return result


def reset_source_nodes(
    base_collection: Path, source_index: np.ndarray, nodes: np.ndarray
) -> tuple[np.ndarray, float]:
    selected = []
    maximum_error = 0.0
    offset = 0
    for path in sorted(base_collection.glob("part_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            command_index = shard["command_index"]
            local = np.flatnonzero(command_index == 0)
            graph_rows = offset + local
            selected.append(source_index[graph_rows])
            error = np.linalg.norm(
                nodes[source_index[graph_rows]] - shard["states"][local, 0],
                axis=1,
            )
            maximum_error = max(maximum_error, float(error.max(initial=0.0)))
            offset += len(command_index)
    if offset != len(source_index):
        raise ValueError("collection rows no longer align with graph edges")
    return np.concatenate(selected).astype(np.int32), maximum_error


def hub_reachability(
    target_node: int,
    horizon: int,
    source_index: np.ndarray,
    endpoint_index: np.ndarray,
    edge_weight: np.ndarray,
    bridge_target: np.ndarray,
    bridge_weight: np.ndarray,
    reset_nodes: np.ndarray,
) -> dict:
    count = bridge_target.shape[0]
    value = np.zeros(count, np.float32)
    value[int(target_node)] = 1.0
    for _ in range(int(horizon)):
        bridge_q = (
            bridge_weight * value[bridge_target]
            + (1.0 - bridge_weight) * value[:, None]
        )
        best = bridge_q.max(axis=1)
        recorded_q = (
            edge_weight * value[endpoint_index]
            + (1.0 - edge_weight) * value[source_index]
        )
        np.maximum.at(best, source_index, recorded_q)
        best[int(target_node)] = 1.0
        value = best.astype(np.float32, copy=False)
    reset_value = value[reset_nodes].astype(np.float64)
    return {
        "reachability_mean": float(reset_value.mean()),
        "reachability_median": float(np.median(reset_value)),
        "reachability_p10": float(np.quantile(reset_value, 0.10)),
        "reachability_minimum": float(reset_value.min()),
        "reachability_maximum": float(reset_value.max()),
    }


def plot_result(
    output: Path,
    node_xy: np.ndarray,
    rows: list[dict],
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, 7.2), constrained_layout=True
    )
    ax = axes[0]
    maze_plot.draw_maze(ax)
    ax.scatter(
        node_xy[:, 0], node_xy[:, 1], s=1.5, color="#9ca4ae",
        alpha=0.12, linewidths=0, zorder=1,
    )
    xy = np.asarray([row["xy"] for row in rows], np.float64)
    reach = np.asarray([row["reachability_mean"] for row in rows])
    marginal = np.asarray(
        [row["marginal_contact_probability"] for row in rows]
    )
    size = 65.0 + 650.0 * marginal / max(float(marginal.max()), 1e-12)
    points = ax.scatter(
        xy[:, 0], xy[:, 1], c=reach, s=size, cmap="viridis",
        vmin=0.0, vmax=1.0, edgecolor="white", linewidth=0.8, zorder=5,
    )
    for row in rows:
        ax.annotate(
            str(row["rank"]), row["xy"], ha="center", va="center",
            fontsize=7.5, color="black", fontweight="bold", zorder=6,
        )
    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.02).set_label(
        "Mean 10-command reachability from R52 preparation resets"
    )
    ax.set_title("First 20 coverage-ranked hubs", fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    rank = np.arange(1, len(rows) + 1)
    cumulative = np.asarray(
        [row["cumulative_contact_probability"] for row in rows]
    )
    p10 = np.asarray([row["reachability_p10"] for row in rows])
    axes[1].plot(
        rank, cumulative, "o-", color="#2f6fbb", linewidth=2,
        label="Cumulative expected contact",
    )
    axes[1].plot(
        rank, reach, "s-", color="#17866b", linewidth=1.6,
        label="Mean reset reachability",
    )
    axes[1].plot(
        rank, p10, "^-", color="#dd7b22", linewidth=1.4,
        label="10th-percentile reset reachability",
    )
    axes[1].set_xlim(0.5, len(rows) + 0.5)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xticks(rank)
    axes[1].set_xlabel("Hub rank")
    axes[1].set_ylabel("Probability")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="best")
    axes[1].set_title(
        "Coverage gain versus graph reachability", fontweight="bold"
    )
    fig.suptitle(
        "R64 offline R52 exploration-hub feasibility audit\n"
        "Ranking uses latent coverage only; reachability is an independent audit",
        fontsize=14, fontweight="bold",
    )
    stem = output / "r64_r52_offline_hub_ranking"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def command_run(args) -> dict:
    audit = command_audit(args)
    if not audit["ready"]:
        raise FileNotFoundError("R52 preparation is incomplete")
    root = Path(args.preparation).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = preparation_paths(root)
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
    rho_max = exact_support_diameter(goal_xy)
    representative, _, alias_origin = alias_hub_candidates(
        nodes[:, :2], sigma
    )
    hub_xy = nodes[representative, :2].astype(np.float64)
    probability = contact_matrix(goal_xy, hub_xy, sigma, rho_max)
    ranked = greedy_prefix(
        probability, int(PROTOCOL_VALUE["reported_prefix_length"])
    )
    reset_nodes, reset_error = reset_source_nodes(
        paths["base_collection"], source, nodes
    )
    for row in ranked:
        candidate = int(row.pop("candidate_index"))
        node_index = int(representative[candidate])
        row.update(
            {
                "node_index": node_index,
                "xy": nodes[node_index, :2].astype(float).tolist(),
                **hub_reachability(
                    node_index,
                    horizon,
                    source,
                    endpoint,
                    edge_weight,
                    bridge_target,
                    bridge_weight,
                    reset_nodes,
                ),
            }
        )
        print(json.dumps(row), flush=True)

    quadrants = {
        (int(x >= 0.0), int(y >= 0.0))
        for x, y in (row["xy"] for row in ranked)
    }
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "preparation": str(root),
        "preparation_revision": PROTOCOL_VALUE["preparation_revision"],
        "artifact_hashes": {
            key: sha256(paths[key])
            for key in ("graph", "graph_metadata", "bridges", "bridges_metadata")
        },
        "goal_prior": PROTOCOL_VALUE["goal_prior"],
        "unique_goal_coordinates": int(len(goal_xy)),
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
            "x_span_m": float(np.ptp([row["xy"][0] for row in ranked])),
            "y_span_m": float(np.ptp([row["xy"][1] for row in ranked])),
            "occupied_xy_quadrants": int(len(quadrants)),
        },
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_created_or_stepped": False,
        "confirmation_panel_accessed": False,
        "hubs": ranked,
    }
    dump(output / "summary.json", result)
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
    plot_result(output, nodes[:, :2], ranked)
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "png": str(output / "r64_r52_offline_hub_ranking.png"),
        "pdf": str(output / "r64_r52_offline_hub_ranking.pdf"),
        "descriptive_spread": result["descriptive_spread"],
        "first_hub_reachability_mean": ranked[0]["reachability_mean"],
        "top20_reachability_mean": float(np.mean([
            row["reachability_mean"] for row in ranked
        ])),
        "top20_reachability_p10_mean": float(np.mean([
            row["reachability_p10"] for row in ranked
        ])),
        "cumulative_contact_probability_at_20": ranked[-1][
            "cumulative_contact_probability"
        ],
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command = value.add_subparsers(dest="command", required=True)
    command.add_parser("audit").set_defaults(func=command_audit)
    command.add_parser("run").set_defaults(func=command_run)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
