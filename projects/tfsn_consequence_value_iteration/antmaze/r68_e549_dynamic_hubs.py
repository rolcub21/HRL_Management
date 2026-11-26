#!/usr/bin/env python3
"""Dynamic full-bank hub refinement on consumed AntMaze tasks 1, 2, and 8."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r53_e549_r51_evaluation as r53,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r56_e549_supplied_goal as r56,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r64_r52_offline_hub_ranking as r64,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r65_r52_area_weighted_hubs as r65,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R68_E549_DYNAMIC_HUB_REFERENCE.md"
PROTOCOL = HERE / "r68_e549_dynamic_hub_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_dynamic_area_hub_development_r68"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_dynamic_area_hub_development_r68/tasks_1_2_8"
)
R61_SUMMARY = r67.R61_SUMMARY
REACHABILITY_FILE = "all_hub_reachability_layers.npy"
REACHABILITY_PARTIAL = "all_hub_reachability_layers.partial.npy"
REACHABILITY_METADATA = "all_hub_reachability_layers.json"
REACHABILITY_PROGRESS = "all_hub_reachability_progress.json"
CANDIDATE_FILE = "dynamic_hub_candidates.npz"


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_hub_revision": r65.METHOD_REVISION,
        "task_ids": [1, 2, 8],
        "episodes": 5,
        "commands_per_episode": 10,
        "fixed_command_duration": 100,
        "evaluation_seed": 607,
        "hub_candidate_count": "data_derived",
        "hub_scoring_chunk_computational_only": 64,
        "reachability_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R68 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R68 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def _install_r68_contract() -> None:
    """Reuse R67's evaluator while replacing its revision and hub scorer."""
    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE
    r67._score_hubs = _score_all_hubs


def _artifact_paths(output: Path) -> dict[str, Path]:
    output = Path(output).resolve()
    return {
        "output": output,
        "table": output / REACHABILITY_FILE,
        "partial": output / REACHABILITY_PARTIAL,
        "metadata": output / REACHABILITY_METADATA,
        "progress": output / REACHABILITY_PROGRESS,
        "candidates": output / CANDIDATE_FILE,
    }


def _candidate_contract(args):
    arrays = r67._graph_arrays(args.preparation)
    (
        paths, consequence, nodes, source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    ) = arrays
    sigma = float(consequence["landing_sigma_m"])
    representative, _cell, origin = r64.alias_hub_candidates(
        nodes[:, :2], sigma
    )
    hub_nodes = np.asarray(representative, np.int32)
    hub_xy = nodes[hub_nodes, :2].astype(np.float64)
    contract = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "preparation_revision": r52.METHOD_REVISION,
        "graph_sha256": r52.sha256(paths["graph"]),
        "bridges_sha256": r52.sha256(paths["bridges"]),
        "landing_sigma_m": sigma,
        "hub_candidate_source": PROTOCOL_VALUE["hub_candidate_source"],
        "hub_candidates": int(len(hub_nodes)),
        "hub_nodes_sha256": r48._hash_arrays(hub_nodes),
        "alias_grid_origin_data_derived_m": origin.astype(float).tolist(),
        "horizon": int(r48.HORIZON),
        "nodes": int(len(nodes)),
        "dtype": "float32",
        "shape": [len(hub_nodes), int(r48.HORIZON) + 1, len(nodes)],
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    return contract, hub_nodes, hub_xy, arrays


def _validate_metadata(path: Path, contract: dict) -> dict:
    metadata = json.loads(path.read_text())
    for key, value in contract.items():
        if metadata.get(key) != value:
            raise ValueError(f"R68 reachability artifact has another {key}")
    return metadata


def _ready(args) -> bool:
    artifact = _artifact_paths(args.output)
    if not artifact["table"].is_file() or not artifact["metadata"].is_file():
        return False
    contract, _hub_nodes, _hub_xy, _arrays = _candidate_contract(args)
    _validate_metadata(artifact["metadata"], contract)
    table = np.load(artifact["table"], mmap_mode="r")
    if list(table.shape) != contract["shape"] or table.dtype != np.float32:
        raise ValueError("R68 reachability array shape or dtype changed")
    return True


def command_audit(args) -> dict:
    required = r67._required(args)
    contract, hub_nodes, _hub_xy, _arrays = _candidate_contract(args)
    table_ready = _ready(args)
    artifact = _artifact_paths(args.output)
    required_bytes = int(np.prod(contract["shape"], dtype=np.int64) * 4)
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": [],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": PROTOCOL_VALUE["episodes"],
        "hub_candidates_data_derived": int(len(hub_nodes)),
        "reachability_cutoff": None,
        "reachability_table_bytes": required_bytes,
        "reachability_table_gib": required_bytes / 1024**3,
        "reachability_table_ready": table_ready,
        "reachability_table": str(artifact["table"]),
        "preparation_artifacts_ready": all(path.is_file() for path in required),
        "ready_for_evaluation": bool(
            table_ready and all(path.is_file() for path in required)
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    print(json.dumps(report, indent=2))
    return report


def command_prepare(args) -> dict:
    contract, hub_nodes, hub_xy, arrays = _candidate_contract(args)
    (
        _paths, _consequence, nodes, source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    ) = arrays
    artifact = _artifact_paths(args.output)
    artifact["output"].mkdir(parents=True, exist_ok=True)
    required_bytes = int(np.prod(contract["shape"], dtype=np.int64) * 4)
    if artifact["table"].is_file():
        _validate_metadata(artifact["metadata"], contract)
        print(json.dumps({
            "artifact": str(artifact["table"]),
            "artifact_sha256": r52.sha256(artifact["table"]),
            "hub_candidates": len(hub_nodes),
            "resumed": True,
            "complete": True,
        }, indent=2))
        return contract

    np.savez_compressed(
        artifact["candidates"],
        node_index=hub_nodes,
        xy=hub_xy.astype(np.float32),
    )
    if artifact["partial"].is_file():
        if not artifact["progress"].is_file():
            raise FileNotFoundError("partial R68 table has no progress contract")
        progress = _validate_metadata(artifact["progress"], contract)
        begin = int(progress["completed_hubs"])
        table = np.load(artifact["partial"], mmap_mode="r+")
    else:
        available = shutil.disk_usage(artifact["output"]).free
        if available < required_bytes * 2:
            raise OSError(
                "R68 requires free space at least twice its reachability table "
                f"size ({required_bytes / 1024**3:.2f} GiB)"
            )
        table = np.lib.format.open_memmap(
            artifact["partial"],
            mode="w+",
            dtype=np.float32,
            shape=tuple(contract["shape"]),
        )
        begin = 0
        r48.dump(artifact["progress"], {**contract, "completed_hubs": 0})

    started = time.perf_counter()
    for index in range(begin, len(hub_nodes)):
        table[index] = r67._one_target_layers(
            int(hub_nodes[index]),
            r48.HORIZON,
            len(nodes),
            source,
            endpoint,
            edge_weight,
            bridge_target,
            bridge_weight,
        )
        if (index + 1) % 8 == 0 or index + 1 == len(hub_nodes):
            table.flush()
            r48.dump(
                artifact["progress"],
                {**contract, "completed_hubs": index + 1},
            )
            elapsed = time.perf_counter() - started
            rate = (index + 1 - begin) / max(elapsed, 1e-9)
            remaining = (len(hub_nodes) - index - 1) / max(rate, 1e-9)
            print(
                f"R68 reachability hubs={index + 1}/{len(hub_nodes)} "
                f"rate={rate:.2f}/s eta_seconds={remaining:.1f}",
                flush=True,
            )
    table.flush()
    del table
    os.replace(artifact["partial"], artifact["table"])
    metadata = {
        **contract,
        "artifact": str(artifact["table"]),
        "artifact_sha256": r52.sha256(artifact["table"]),
        "candidate_artifact": str(artifact["candidates"]),
        "candidate_artifact_sha256": r52.sha256(artifact["candidates"]),
        "wall_clock_seconds_this_invocation": time.perf_counter() - started,
    }
    r48.dump(artifact["metadata"], metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def _all_hub_coverage(posterior, hub_xy: np.ndarray) -> np.ndarray:
    posterior.assert_not_exhausted("before R68 dynamic hub scoring")
    hubs = np.asarray(hub_xy, np.float64).reshape(-1, 2)
    chunk = int(PROTOCOL_VALUE["hub_scoring_chunk_computational_only"])
    output = np.empty(len(hubs), np.float64)
    log_width = np.zeros(len(posterior.goal_xy), np.float64)
    log_width[posterior.feasible] = np.log(
        posterior.upper[posterior.feasible]
        / posterior.lower[posterior.feasible]
    )
    for begin in range(0, len(hubs), chunk):
        stop = min(begin + chunk, len(hubs))
        distance = np.linalg.norm(
            posterior.goal_xy[:, None, :] - hubs[None, begin:stop, :],
            axis=2,
        )
        conditional = np.zeros(distance.shape, np.float64)
        certain = posterior.feasible[:, None] & (
            distance <= posterior.lower[:, None]
        )
        possible = (
            posterior.feasible[:, None]
            & (distance > posterior.lower[:, None])
            & (distance < posterior.upper[:, None])
        )
        conditional[certain] = 1.0
        row, column = np.nonzero(possible)
        conditional[row, column] = (
            np.log(posterior.upper[row] / distance[row, column])
            / log_width[row]
        )
        output[begin:stop] = posterior.goal_probability @ conditional
    return np.clip(output, 0.0, 1.0)


def _score_all_hubs(
    posterior, hub_xy, reachability, graph, state, remaining,
):
    started = time.perf_counter()
    coverage = _all_hub_coverage(posterior, hub_xy)
    current_node = int(graph.project(np.asarray(state)[None])[0][0])
    reach = np.asarray(
        reachability[:, int(remaining), current_node], np.float64
    )
    score = coverage * reach
    selected = int(np.argmax(score))
    return (
        selected,
        coverage,
        reach,
        score,
        current_node,
        time.perf_counter() - started,
    )


def _load_dynamic(args):
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError(
            "run prepare-r68-e549-5m-dynamic-hubs before R68 evaluation"
        )
    contract, hub_nodes, hub_xy, arrays = _candidate_contract(args)
    artifact = _artifact_paths(args.output)
    _validate_metadata(artifact["metadata"], contract)
    reachability = np.load(artifact["table"], mmap_mode="r")
    planner, executor, env = r53._load_runtime(args)
    return planner, executor, env, hub_nodes, hub_xy, arrays[2], reachability


def _normalize_task_record(
    task: dict, log_path: Path, hub_nodes: np.ndarray
) -> dict:
    changed = False
    if "selected_r65_ranks" in task:
        indices = [int(value) - 1 for value in task.pop("selected_r65_ranks")]
        task["selected_dynamic_hub_indices"] = indices
        task["selected_dynamic_hub_nodes"] = [
            int(hub_nodes[index]) for index in indices
        ]
        changed = True
    logs = [
        json.loads(line) for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    log_changed = False
    for row in logs:
        selection = row.get("hub_selection")
        if selection is not None and "r65_rank" in selection:
            selection["dynamic_candidate_index_one_based"] = selection.pop(
                "r65_rank"
            )
            log_changed = True
    if log_changed:
        log_path.write_text(
            "\n".join(json.dumps(row) for row in logs) + "\n"
        )
    if changed:
        r48.dump(log_path.with_suffix(".json"), task)
    return task


def _plot(output: Path, rows: list[dict]) -> dict:
    fig, axes = r67.plt.subplots(
        1, 3, figsize=(16.5, 5.8), constrained_layout=True
    )
    for ax, task in zip(axes, rows):
        r67.maze_plot.draw_maze(ax)
        logs = [
            json.loads(line)
            for line in (output / f"task_{task['task_id']:03d}.jsonl")
            .read_text().splitlines()
            if line.strip()
        ]
        goal = np.asarray(task["goal"], np.float64)
        ax.scatter(*goal, marker="X", s=85, color="#b74343", zorder=9)
        for episode in range(int(PROTOCOL_VALUE["episodes"])):
            episode_rows = [
                row for row in logs
                if row["phase"] == "acquisition"
                and int(row["episode"]) == episode
            ]
            if not episode_rows:
                continue
            start = np.asarray(episode_rows[0]["start_xy"], np.float64)
            endpoints = np.asarray(
                [row["actual_endpoint"] for row in episode_rows], np.float64
            )
            path = np.vstack((start, endpoints))
            color = r67.EPISODE_COLORS[episode]
            ax.plot(
                path[:, 0], path[:, 1], color=color, linewidth=1.5, zorder=3
            )
            ax.scatter(
                endpoints[:, 0], endpoints[:, 1], s=15, color=color, zorder=5
            )
            selected = [
                row for row in episode_rows
                if row["hub_selected_this_command"]
            ]
            if selected:
                target = np.asarray(
                    [row["active_hub_xy"] for row in selected], np.float64
                )
                ax.scatter(
                    target[:, 0], target[:, 1], marker="D", s=52,
                    facecolor="none", edgecolor=color, linewidth=1.1, zorder=7,
                )
        status = (
            f"contact E{task['first_contact_episode'] + 1}"
            if task["first_contact_episode"] is not None else "no contact"
        )
        ax.set_title(
            f"Task {task['task_id']}: {status}\n"
            f"{task['unique_hubs_selected']} unique hubs, "
            f"{task['acquisition_new_1m_cells_per_command']:.2f} new cells/cmd",
            fontsize=10,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle(
        "R68 E549-5M dynamic full-bank hubs — diamonds are selected hubs",
        fontsize=14,
        fontweight="bold",
    )
    stem = output / "r68_dynamic_hubs_tasks_1_2_8"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    r67.plt.close(fig)
    return {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }


def command_timing(args) -> dict:
    _install_r68_contract()
    (
        planner, _executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_dynamic(args)
    try:
        task_id = int(PROTOCOL_VALUE["task_ids"][0])
        task = r48.r31.r22._task(args.root, task_id)
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = r48.full_state(observation)
        posterior = r67._area_posterior(nodes, float(r48.SIGMA))
        memory_seconds = r67._negative_path_update(
            posterior, state[:2].reshape(1, 2)
        )
        planner.rng = np.random.default_rng(r67._task_seed(args.seed, task_id))
        planner.seed = r67._task_seed(args.seed, task_id)
        planner.goal_key = None
        planner.reset_online_correction_task()
        (
            selected, coverage, reach, score, current_node, hub_seconds,
        ) = _score_all_hubs(
            posterior, hub_xy, reachability, planner.graph, state, r48.HORIZON
        )
        field = r56.R56SuppliedGoalField(float(r48.SIGMA), hub_xy[selected])
        started = time.perf_counter()
        command, diagnostic = planner.choose(
            state, 0, field, [], [[state.copy()]]
        )
        routing_seconds = time.perf_counter() - started
        result = {
            "schema_version": 1,
            "method_revision": METHOD_REVISION,
            "environment_steps": 0,
            "hub_candidates_scored": int(len(hub_nodes)),
            "timing_seconds": {
                "memory_update": memory_seconds,
                "all_hub_scoring": hub_seconds,
                "routing": routing_seconds,
                "total": memory_seconds + hub_seconds + routing_seconds,
            },
            "selected_hub": {
                "candidate_index": selected,
                "node_index": int(hub_nodes[selected]),
                "xy": hub_xy[selected].astype(float).tolist(),
                "current_graph_node": current_node,
                "posterior_contact_probability": float(coverage[selected]),
                "reachability": float(reach[selected]),
                "score": float(score[selected]),
            },
            "issued_command_without_execution": np.asarray(command).tolist(),
            "router_selected_kind": diagnostic.get("selected_kind"),
        }
        r48.dump(Path(args.output) / "one_decision_timing.json", result)
        print(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def command_evaluate(args) -> dict:
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes"]):
        raise ValueError("R68 requires five episodes per development task")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R68 requires evaluation seed 607")
    _install_r68_contract()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R68 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved
    (
        planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_dynamic(args)
    sigma = float(r48.SIGMA)
    try:
        rows = []
        for task_id in PROTOCOL_VALUE["task_ids"]:
            row = r67._evaluate_task(
                args,
                planner,
                executor,
                env,
                int(task_id),
                output,
                hub_nodes,
                hub_xy,
                reachability,
                nodes,
                sigma,
            )
            rows.append(_normalize_task_record(
                row, output / f"task_{int(task_id):03d}.jsonl", hub_nodes
            ))
    finally:
        env.close()
    contacts = [row["first_contact_episode"] for row in rows]
    returns = [
        success
        for row in rows
        if row["first_contact_episode"] is not None
        for success in row["successes"][row["first_contact_episode"] + 1 :]
    ]
    baseline_summary = json.loads(R61_SUMMARY.read_text())
    baseline = {
        int(row["task_id"]): row for row in baseline_summary["rows"]
        if int(row["task_id"]) in PROTOCOL_VALUE["task_ids"]
    }
    comparison = [{
        "task_id": int(row["task_id"]),
        "r61_first_contact_episode": baseline[int(row["task_id"])][
            "first_contact_episode"
        ],
        "r68_first_contact_episode": row["first_contact_episode"],
        "r61_new_1m_cells_per_command": baseline[int(row["task_id"])][
            "acquisition_new_1m_cells_per_command"
        ],
        "r68_new_1m_cells_per_command": row[
            "acquisition_new_1m_cells_per_command"
        ],
    } for row in rows]
    figures = _plot(output, rows)
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": [],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": int(args.episodes),
        "dynamic_hub_candidates": int(len(hub_nodes)),
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([value is not None for value in contacts])),
        "pooled_conditional_return": float(np.mean(returns)) if returns else None,
        "mean_e2e": float(np.mean([
            success for row in rows for success in row["successes"]
        ])),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "hub_selections": int(sum(row["hub_selections"] for row in rows)),
        "hub_arrivals": int(sum(row["hub_arrivals"] for row in rows)),
        "timing_seconds": {
            "memory_update_total": float(sum(
                row["timing_seconds"]["memory_update_total"] for row in rows
            )),
            "hub_scoring_total": float(sum(
                row["timing_seconds"]["hub_scoring_total"] for row in rows
            )),
            "routing_total": float(sum(
                row["timing_seconds"]["routing_total"] for row in rows
            )),
        },
        "r61_same_task_comparison": comparison,
        "figures": figures,
        "rows": rows,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(summary_path, result)
    print(json.dumps({
        key: result[key] for key in (
            "tasks", "task_ids", "dynamic_hub_candidates", "acquired_tasks",
            "acquisition_at_1", "acquisition_at_5",
            "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command", "hub_selections",
            "hub_arrivals", "timing_seconds", "r61_same_task_comparison",
            "figures",
        )
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=5)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("prepare").set_defaults(func=command_prepare)
    commands.add_parser("timing").set_defaults(func=command_timing)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
