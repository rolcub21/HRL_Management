"""Reset-specific reach-weighted diagnostic over the R65 hub prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from projects.tfsn_consequence_value_iteration.antmaze import (
    plot_r59_hidden_goal_original9 as maze_plot,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r64_r52_offline_hub_ranking as r64,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r65_r52_area_weighted_hubs as r65,
)


MANAGEMENT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).with_name("R66_R52_RESET_SPECIFIC_HUB_REFERENCE.md")
PROTOCOL = Path(__file__).with_name("r66_r52_reset_specific_hub_protocol.json")
METHOD_REVISION = "antmaze_r52_reset_specific_reach_weighted_hubs_r66"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r52_reset_specific_reach_weighted_hubs_r66"
)
R59_TASK_ROOT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_hidden_goal_r59/original9"
)


def protocol_value() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": (
            "consumed_development_reset_specific_hub_diagnostic_only"
        ),
        "preparation_revision": r64.PROTOCOL_VALUE["preparation_revision"],
        "base_hub_revision": r65.METHOD_REVISION,
        "task_ids": [1, 2, 8],
        "reachability_cutoff": None,
        "hidden_goal_episodes_executed": False,
        "task_reward_used": False,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R66 protocol has another {key}")
    return value


PROTOCOL_VALUE = protocol_value()


def reachability_values(
    target_node: int,
    horizon: int,
    source_index: np.ndarray,
    endpoint_index: np.ndarray,
    edge_weight: np.ndarray,
    bridge_target: np.ndarray,
    bridge_weight: np.ndarray,
    reset_nodes: np.ndarray,
) -> np.ndarray:
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
    return value[reset_nodes].astype(np.float64)


def standardized_nearest_nodes(
    states: np.ndarray, nodes: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_nodes = nodes.astype(np.float64) / scale[None]
    result, standardized, xy_error = [], [], []
    for state in states.astype(np.float64):
        distance = np.square(normalized_nodes - state[None] / scale[None]).sum(
            axis=1
        )
        index = int(np.argmin(distance))
        result.append(index)
        standardized.append(float(np.sqrt(distance[index])))
        xy_error.append(float(np.linalg.norm(nodes[index, :2] - state[:2])))
    return (
        np.asarray(result, np.int32),
        np.asarray(standardized, np.float64),
        np.asarray(xy_error, np.float64),
    )


def reach_weighted_ranking(
    probability: np.ndarray,
    goal_weight: np.ndarray,
    reachability: np.ndarray,
    hub_rows: list[dict],
) -> list[dict]:
    covered = np.zeros(probability.shape[0], np.float32)
    selected = np.zeros(probability.shape[1], bool)
    output = []
    for rank in range(probability.shape[1]):
        delta = np.maximum(probability - covered[:, None], 0.0)
        marginal = np.sum(
            delta.astype(np.float64) * goal_weight[:, None], axis=0
        )
        score = marginal * reachability
        score[selected] = -np.inf
        index = int(np.argmax(score))
        covered = np.maximum(covered, probability[:, index])
        selected[index] = True
        output.append({
            "rank": rank + 1,
            "r65_rank": int(hub_rows[index]["rank"]),
            "node_index": int(hub_rows[index]["node_index"]),
            "xy": list(hub_rows[index]["xy"]),
            "reset_specific_reachability": float(reachability[index]),
            "marginal_contact_probability": float(marginal[index]),
            "combined_score": float(score[index]),
            "cumulative_contact_probability": float(
                np.sum(covered.astype(np.float64) * goal_weight)
            ),
        })
    return output


def contact_probability(distance: float, rho_min: float, rho_max: float) -> float:
    if distance <= rho_min:
        return 1.0
    if distance >= rho_max:
        return 0.0
    return float(
        np.log(rho_max / distance) / np.log(rho_max / rho_min)
    )


def plot_result(output: Path, nodes: np.ndarray, tasks: list[dict]) -> None:
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(17.0, 6.0), constrained_layout=True
    )
    for ax, task in zip(axes, tasks):
        maze_plot.draw_maze(ax)
        ax.scatter(
            nodes[:, 0], nodes[:, 1], s=1.1, color="#9ca4ae",
            alpha=0.09, linewidths=0, zorder=1,
        )
        rows = task["ranking"]
        xy = np.asarray([row["xy"] for row in rows], np.float64)
        reach = np.asarray(
            [row["reset_specific_reachability"] for row in rows]
        )
        points = ax.scatter(
            xy[:, 0], xy[:, 1], c=reach, s=75, cmap="viridis",
            vmin=0.0, vmax=1.0, edgecolor="white", linewidth=0.7, zorder=5,
        )
        for row in rows[:10]:
            ax.annotate(
                str(row["rank"]), row["xy"], ha="center", va="center",
                fontsize=7.2, fontweight="bold", color="black", zorder=6,
            )
        reset = np.asarray(task["reset_xy"])
        goal = np.asarray(task["goal_overlay_xy"])
        ax.scatter(
            reset[0], reset[1], marker="X", s=145, color="black",
            edgecolor="white", linewidth=0.8, label="reset", zorder=8,
        )
        ax.scatter(
            goal[0], goal[1], marker="*", s=230, color="#d62728",
            edgecolor="white", linewidth=0.8, label="goal (overlay)", zorder=8,
        )
        nearest = np.asarray(task["nearest_ranked_hub_xy"])
        ax.plot(
            [goal[0], nearest[0]], [goal[1], nearest[1]], "--",
            color="#d62728", linewidth=1.0, alpha=0.75, zorder=4,
        )
        ax.set_title(
            f"Task {task['task_id']}  |  nearest hub "
            f"{task['nearest_ranked_hub_distance_m']:.2f} m\n"
            f"max latent contact={task['maximum_goal_contact_probability']:.2f}",
            fontweight="bold",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.colorbar(points, ax=axes, fraction=0.022, pad=0.02).set_label(
        "10-command reachability from this task's reset"
    )
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "R66 reset-specific reach × area-coverage ranking\n"
        "Goals are post-ranking development overlays only; labels show top 10",
        fontsize=14, fontweight="bold",
    )
    stem = output / "r66_reset_specific_hubs_tasks_1_2_8"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args) -> dict:
    if not r65.DEFAULT_OUTPUT.joinpath("summary.json").is_file():
        raise FileNotFoundError("run R65 before R66")
    base = json.loads(r65.DEFAULT_OUTPUT.joinpath("summary.json").read_text())
    if base.get("method_revision") != r65.METHOD_REVISION:
        raise ValueError("R65 summary revision changed")
    task_metadata = []
    for task_id in PROTOCOL_VALUE["task_ids"]:
        path = R59_TASK_ROOT / f"task_{task_id:03d}.json"
        row = json.loads(path.read_text())
        task_metadata.append({
            "task_id": task_id,
            "reset_seed": int(row["reset_seed"]),
            "goal": np.asarray(row["goal"], np.float64),
        })

    preparation = Path(args.preparation).resolve()
    paths = r64.preparation_paths(preparation)
    collection = json.loads(paths["collection_summary"].read_text())
    consequence_summary = json.loads(paths["consequence_summary"].read_text())
    horizon = int(collection["commands_per_physical_episode"])
    sigma = float(consequence_summary["landing_sigma_m"])
    with np.load(paths["graph"], allow_pickle=False) as graph:
        nodes = graph["nodes"].astype(np.float32)
        source = graph["source_index"].astype(np.int32)
        endpoint = graph["endpoint_index"].astype(np.int32)
        edge_weight = graph["edge_reliability"].astype(np.float32)
    with np.load(paths["bridges"], allow_pickle=False) as bridges:
        bridge_target = bridges["target_index"].astype(np.int32)
        bridge_weight = bridges["weight"].astype(np.float32)

    environment = r48.make_env(max_episode_steps=10_000_000)
    reset_states = []
    try:
        for task in task_metadata:
            observation, _ = environment.reset(seed=task["reset_seed"])
            reset_states.append(r48.full_state(observation))
    finally:
        environment.close()
    reset_states = np.asarray(reset_states, np.float32)

    checkpoint = torch.load(
        consequence_summary["checkpoint"], map_location="cpu",
        weights_only=False,
    )
    state_scale = np.asarray(checkpoint["base_input_std"], np.float64)[:29]
    state_scale = np.maximum(state_scale, 1e-6)
    reset_nodes, standardized_error, reset_xy_error = standardized_nearest_nodes(
        reset_states, nodes, state_scale
    )

    hub_rows = base["hubs"]
    hub_nodes = np.asarray([row["node_index"] for row in hub_rows], np.int32)
    reachability = np.stack([
        reachability_values(
            int(target), horizon, source, endpoint, edge_weight,
            bridge_target, bridge_weight, reset_nodes,
        )
        for target in hub_nodes
    ], axis=1)

    goal_xy = np.unique(nodes[:, :2].astype(np.float64), axis=0)
    goal_weight, _, _ = r65.area_weights(goal_xy, sigma)
    rho_max = r64.exact_support_diameter(goal_xy)
    hub_xy = nodes[hub_nodes, :2].astype(np.float64)
    probability = r64.contact_matrix(goal_xy, hub_xy, sigma, rho_max)

    # Ranking is complete before any development goal coordinate is used.
    rankings = [
        reach_weighted_ranking(
            probability, goal_weight, reachability[index], hub_rows
        )
        for index in range(len(task_metadata))
    ]

    tasks = []
    for index, (metadata, ranking) in enumerate(zip(task_metadata, rankings)):
        ranked_xy = np.asarray([row["xy"] for row in ranking], np.float64)
        distance = np.linalg.norm(ranked_xy - metadata["goal"][None], axis=1)
        nearest = int(np.argmin(distance))
        contact = np.asarray([
            contact_probability(float(item), sigma, rho_max)
            for item in distance
        ])
        tasks.append({
            "task_id": int(metadata["task_id"]),
            "reset_seed": int(metadata["reset_seed"]),
            "reset_xy": reset_states[index, :2].astype(float).tolist(),
            "associated_graph_node": int(reset_nodes[index]),
            "associated_graph_node_xy": (
                nodes[reset_nodes[index], :2].astype(float).tolist()
            ),
            "reset_association_standardized_distance": float(
                standardized_error[index]
            ),
            "reset_association_xy_error_m": float(reset_xy_error[index]),
            "goal_overlay_xy": metadata["goal"].astype(float).tolist(),
            "nearest_ranked_hub_rank": int(ranking[nearest]["rank"]),
            "nearest_ranked_hub_xy": ranking[nearest]["xy"],
            "nearest_ranked_hub_distance_m": float(distance[nearest]),
            "maximum_goal_contact_probability": float(contact.max()),
            "goal_contact_probability_first_5": float(contact[:5].max()),
            "goal_contact_probability_first_10": float(contact[:10].max()),
            "ranking": ranking,
        })

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r64.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": PROTOCOL_VALUE["oracle_inputs"],
        "preparation_revision": PROTOCOL_VALUE["preparation_revision"],
        "base_hub_revision": PROTOCOL_VALUE["base_hub_revision"],
        "ranking": PROTOCOL_VALUE["ranking"],
        "reachability_cutoff": None,
        "candidate_scope": PROTOCOL_VALUE["candidate_scope"],
        "landing_sigma_m": sigma,
        "rho_max_m": rho_max,
        "environment_resets": len(tasks),
        "environment_steps": 0,
        "hidden_goal_episodes_executed": False,
        "task_reward_used": False,
        "confirmation_panel_accessed": False,
        "tasks": tasks,
    }
    r64.dump(output / "summary.json", result)
    plot_result(output, nodes, tasks)
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "png": str(output / "r66_reset_specific_hubs_tasks_1_2_8.png"),
        "pdf": str(output / "r66_reset_specific_hubs_tasks_1_2_8.pdf"),
        "tasks": [{
            key: task[key] for key in (
                "task_id", "reset_xy", "reset_association_xy_error_m",
                "nearest_ranked_hub_rank", "nearest_ranked_hub_distance_m",
                "maximum_goal_contact_probability",
                "goal_contact_probability_first_5",
                "goal_contact_probability_first_10",
            )
        } for task in tasks],
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preparation", type=Path, default=r64.DEFAULT_PREPARATION)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.set_defaults(func=run)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
