"""Evaluate frozen R10 and its references on six unused AntMaze goals.

The canonical goal permutation has 26 free cells.  Indices 0--1 were consumed,
2--10 form the original nine-goal development panel, 11--19 remain reserved
for confirmation, and 20--25 are the only unused cells that can extend
development without opening confirmation.  This module materializes only
indices 20--25 and never reads or creates ``confirmation.json``.

Each task is an independent, resumable process target.  The shell runner can
therefore execute the six tasks concurrently while keeping the frozen R10
planner and both references bit-for-bit separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.collect import (
    evaluate_supplied_goal_task,
    execute_command,
)
from scripts.antmaze.current_tfsn.constants import (
    COMMANDS_PER_EPISODE,
    COMMAND_STEPS,
    CONTACT_RADIUS,
    EPISODE_STEPS,
    LANDING_SIGMA,
)
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import (
    _oracle_cell_route,
    dump,
    free_cells,
    paths,
)

from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    CLAIM_LEVEL,
    METHOD_REVISION,
    MobilityModel,
    MobilityStitchedPlanner,
    _artifact_paths,
    _load_base,
    _prepare_landmarks,
    _verify_bridges,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    sha256_file,
)


PANEL_REVISION = "antmaze_distinct_development_extension_15_r1"
GOAL_PARTITION_SEED = 37700
PANEL_SEED = 37701
ORIGINAL_DEVELOPMENT_INDICES = tuple(range(2, 11))
RESERVED_CONFIRMATION_INDICES = tuple(range(11, 20))
EXTENSION_INDICES = tuple(range(20, 26))
EXTENSION_TASK_IDS = tuple(range(9, 15))


def output_paths(root):
    directory = paths(root)["eval"] / "development_extension_unused6"
    return {
        "directory": directory,
        "panel": paths(root)["panels"] / "development_extension_unused6.json",
        "r10": directory / "mobility_stitched_value_iteration_r10",
        "oracle": directory / "oracle_fixed_command_contract",
        "direct": directory / "goal_given_executor_direct",
        "combined": directory / "combined_summary.json",
    }


def build_extension_panel(original, unused_goals):
    if original.get("name") != "development":
        raise ValueError("the base panel is not development")
    if original.get("goal_partition_seed") != GOAL_PARTITION_SEED:
        raise ValueError("the base panel uses another goal permutation")
    if original.get("goal_offset") != ORIGINAL_DEVELOPMENT_INDICES[0]:
        raise ValueError("the base panel has another goal offset")
    tasks = original.get("tasks", [])
    if len(tasks) != len(ORIGINAL_DEVELOPMENT_INDICES):
        raise ValueError("the base development panel is not the frozen nine")
    if [int(row["task_id"]) for row in tasks] != list(range(9)):
        raise ValueError("the base development task ids changed")
    if len(unused_goals) != len(EXTENSION_INDICES):
        raise ValueError("the extension must contain exactly six goals")
    added = [
        {
            "task_id": int(task_id),
            "goal_index": int(goal_index),
            "goal": list(goal),
            "reset_seed": int(PANEL_SEED * 10_000 + task_id),
        }
        for task_id, goal_index, goal in zip(
            EXTENSION_TASK_IDS, EXTENSION_INDICES, unused_goals
        )
    ]
    return {
        "name": "development_extension_unused6",
        "method_revision": PANEL_REVISION,
        "goal_partition_seed": GOAL_PARTITION_SEED,
        "panel_seed": PANEL_SEED,
        "original_development_goal_indices": list(
            ORIGINAL_DEVELOPMENT_INDICES
        ),
        "reserved_confirmation_goal_indices": list(
            RESERVED_CONFIRMATION_INDICES
        ),
        "extension_goal_indices": list(EXTENSION_INDICES),
        "confirmation_panel_accessed": False,
        "confirmation_panel_created": False,
        "distinct_goals_combined": 15,
        "tasks": added,
    }


def prepare_panel(root):
    output = output_paths(root)
    if output["panel"].exists():
        saved = json.loads(output["panel"].read_text())
        if saved.get("method_revision") != PANEL_REVISION:
            raise ValueError("the saved extension panel has another revision")
        if saved.get("extension_goal_indices") != list(EXTENSION_INDICES):
            raise ValueError("the saved extension goal indices changed")
        if saved.get("reserved_confirmation_goal_indices") != list(
            RESERVED_CONFIRMATION_INDICES
        ):
            raise ValueError("the reserved confirmation indices changed")
        if saved.get("confirmation_panel_accessed") is not False:
            raise ValueError("the extension claims confirmation access")
        return saved
    original_path = paths(root)["panels"] / "development.json"
    if not original_path.exists():
        raise FileNotFoundError("the frozen nine-goal development panel is missing")
    original = json.loads(original_path.read_text())
    unused_goals = free_cells(
        GOAL_PARTITION_SEED,
        len(EXTENSION_INDICES),
        EXTENSION_INDICES[0],
    )
    expected = build_extension_panel(original, unused_goals)
    dump(output["panel"], expected)
    return expected


def selected_task(args):
    selected = prepare_panel(args.root)
    task_id = int(args.task_id)
    if task_id not in EXTENSION_TASK_IDS:
        raise ValueError(f"task id must be one of {EXTENSION_TASK_IDS}")
    return next(row for row in selected["tasks"] if row["task_id"] == task_id)


def _existing_task_result(path, task, arm):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("task_id") != task["task_id"]:
        raise ValueError(f"wrong task in existing {path}")
    if value.get("goal") != task["goal"]:
        raise ValueError(f"wrong goal in existing {path}")
    if value.get("reset_seed") != task["reset_seed"]:
        raise ValueError(f"wrong reset in existing {path}")
    if value.get("evaluation_arm") != arm:
        raise ValueError(f"wrong arm in existing {path}")
    return value


def load_frozen_r10(args):
    graph, consequence, consequence_path = _load_base(args)
    artifact, landmark_index, targets, _distance, _metadata = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    if not artifact["bridges"].exists():
        raise FileNotFoundError("the frozen R10 bridge artifact is missing")
    with np.load(artifact["bridges"]) as data:
        saved_landmarks = data["landmark_index"]
        saved_targets = data["target_index"]
        weight = data["weight"]
    if not np.array_equal(landmark_index, saved_landmarks):
        raise ValueError("the frozen R10 landmark set changed")
    if not np.array_equal(targets, saved_targets):
        raise ValueError("the frozen R10 bridge targets changed")
    _verify_bridges(
        args,
        graph,
        consequence_path,
        artifact,
        landmark_index,
        targets,
        weight,
    )
    mobility = MobilityModel(artifact["mobility"], args.device)
    planner = MobilityStitchedPlanner(
        consequence,
        mobility,
        graph,
        landmark_index,
        targets,
        weight,
        np.random.default_rng(args.seed),
        args.seed,
    )
    hashes = {
        "graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(artifact["mobility"]),
        "bridges_sha256": sha256_file(artifact["bridges"]),
    }
    return planner, hashes


def command_prepare_panel(args):
    value = prepare_panel(args.root)
    print(json.dumps(value, indent=2))


def command_r10_task(args):
    task = selected_task(args)
    output = output_paths(args.root)["r10"]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task['task_id']:03d}.json"
    existing = _existing_task_result(result_path, task, "frozen_r10")
    if existing is not None:
        print(json.dumps(existing))
        return
    planner, hashes = load_frozen_r10(args)
    task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    row = evaluate_supplied_goal_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        LANDING_SIGMA,
        max_steps=EPISODE_STEPS,
        log_path=output / f"task_{task['task_id']:03d}.jsonl",
    )
    env.close()
    row.update(task)
    row.update({
        "evaluation_arm": "frozen_r10",
        "method_revision": METHOD_REVISION,
        "task_gradient_updates": 0,
        "value_coverage": planner.coverage,
        **hashes,
    })
    dump(result_path, row)
    print(json.dumps(row))


def command_direct_task(args):
    task = selected_task(args)
    output = output_paths(args.root)["direct"]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task['task_id']:03d}.json"
    existing = _existing_task_result(result_path, task, "direct_executor")
    if existing is not None:
        print(json.dumps(existing))
        return
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    goal = np.asarray(task["goal"], np.float32)
    observation, _ = env.reset(seed=int(task["reset_seed"]))
    state = full_state(observation)
    initial_distance = float(np.linalg.norm(state[:2] - goal))
    minimum_distance = initial_distance
    success = initial_distance <= CONTACT_RADIUS
    steps = 0
    while not success and steps < EPISODE_STEPS:
        observation, _reward, terminated, truncated, _info = env.step(
            executor.action(state, goal)
        )
        state = full_state(observation)
        steps += 1
        distance = float(np.linalg.norm(state[:2] - goal))
        minimum_distance = min(minimum_distance, distance)
        success = distance <= CONTACT_RADIUS
        if terminated or truncated:
            break
    env.close()
    row = {
        **task,
        "evaluation_arm": "direct_executor",
        "reference": "goal_given_executor_direct",
        "initial_distance": initial_distance,
        "success": bool(success),
        "steps_to_goal": steps if success else None,
        "steps_executed": steps,
        "minimum_distance": minimum_distance,
        "contact_radius": CONTACT_RADIUS,
        "task_gradient_updates": 0,
    }
    dump(result_path, row)
    print(json.dumps(row))


def command_oracle_task(args):
    task = selected_task(args)
    output = output_paths(args.root)["oracle"]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task['task_id']:03d}.json"
    existing = _existing_task_result(
        result_path, task, "oracle_fixed_command_contract"
    )
    if existing is not None:
        print(json.dumps(existing))
        return
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    goal = np.asarray(task["goal"], np.float32)
    observation, _ = env.reset(seed=int(task["reset_seed"]))
    state = full_state(observation)
    initial_distance = float(np.linalg.norm(state[:2] - goal))
    cells, waypoints = _oracle_cell_route(
        env.unwrapped.maze, state[:2], goal
    )
    waypoint_index = 0
    minimum_distance = initial_distance
    success = initial_distance <= CONTACT_RADIUS
    steps_to_goal = 0 if success else None
    decisions = []
    for command_index in range(COMMANDS_PER_EPISODE):
        if success:
            break
        while (
            waypoint_index < len(waypoints)
            and np.linalg.norm(
                state[:2] - waypoints[waypoint_index]
            ) <= CONTACT_RADIUS
        ):
            waypoint_index += 1
        if waypoint_index >= len(waypoints):
            success = np.linalg.norm(state[:2] - goal) <= CONTACT_RADIUS
            break
        command = np.asarray(waypoints[waypoint_index], np.float32)
        start = state.copy()
        trajectory, rewards = execute_command(
            env, executor, state, command, hidden_goal=goal
        )
        goal_distances = np.linalg.norm(
            trajectory[1:, :2] - goal[None], axis=1
        )
        minimum_distance = min(minimum_distance, float(goal_distances.min()))
        if steps_to_goal is None and rewards.any():
            steps_to_goal = (
                command_index * COMMAND_STEPS
                + int(np.flatnonzero(rewards)[0])
                + 1
            )
            success = True
        decisions.append({
            "task_id": int(task["task_id"]),
            "command_index": int(command_index),
            "waypoint_index": int(waypoint_index),
            "command": command.tolist(),
            "start_xy": start[:2].tolist(),
            "actual_endpoint_xy": trajectory[-1, :2].tolist(),
            "moved_distance_m": float(
                np.linalg.norm(trajectory[-1, :2] - start[:2])
            ),
            "observed_stall": bool(
                np.linalg.norm(trajectory[-1, :2] - start[:2]) < 0.2
            ),
        })
        state = trajectory[-1]
    env.close()
    log_path = output / f"task_{task['task_id']:03d}.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(value) for value in decisions) + "\n"
    )
    row = {
        **task,
        "evaluation_arm": "oracle_fixed_command_contract",
        "method_revision": "antmaze_oracle_fixed_command_contract_r7",
        "initial_distance": initial_distance,
        "route_cells": [list(cell) for cell in cells],
        "route_edges": len(cells) - 1,
        "route_waypoints": [waypoint.tolist() for waypoint in waypoints],
        "waypoints_advanced_at_boundaries": int(waypoint_index),
        "success": bool(success),
        "steps_to_goal": steps_to_goal,
        "steps_executed": int(len(decisions) * COMMAND_STEPS),
        "commands_executed": int(len(decisions)),
        "minimum_distance": minimum_distance,
        "contact_radius": CONTACT_RADIUS,
        "command_duration_steps": COMMAND_STEPS,
        "primitive_step_waypoint_switching": False,
        "task_reward_used_by_controller": False,
        "oracle_inputs": ["maze_occupancy", "supplied_goal"],
        "task_gradient_updates": 0,
    }
    dump(result_path, row)
    print(json.dumps(row))


def _summary(rows, arm):
    successes = int(sum(row["success"] for row in rows))
    return {
        "panel_revision": PANEL_REVISION,
        "claim_level": "unused_distinct_goal_development_extension",
        "evaluation_arm": arm,
        "goal_indices": list(EXTENSION_INDICES),
        "tasks": len(rows),
        "successes": successes,
        "success_rate": float(successes / len(rows)),
        "mean_steps_to_goal_success": (
            float(np.mean([
                row["steps_to_goal"] for row in rows if row["success"]
            ]))
            if successes else None
        ),
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }


def command_summarize_arm(args):
    prepare_panel(args.root)
    arm_path = output_paths(args.root)[args.arm]
    rows = []
    for task_id in EXTENSION_TASK_IDS:
        task_path = arm_path / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(f"missing task result {task_path}")
        rows.append(json.loads(task_path.read_text()))
    result = _summary(rows, args.arm)
    dump(arm_path / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in ("evaluation_arm", "successes", "tasks", "success_rate")
    }, indent=2))


def _original_arm(root, arm):
    evaluation = paths(root)["eval"] / "development"
    if arm == "r10":
        source = evaluation / "mobility_stitched_value_iteration_r10" / "summary.json"
    elif arm == "oracle":
        source = evaluation / "oracle_fixed_command_contract_r7" / "summary.json"
    elif arm == "direct":
        source = evaluation / "goal_given_reference.json"
    else:
        raise ValueError(arm)
    if not source.exists():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text())
    rows = value["rows"]
    return {
        "tasks": len(rows),
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
    }


def command_summarize_combined(args):
    prepare_panel(args.root)
    result = {
        "panel_revision": PANEL_REVISION,
        "claim_level": "expanded_consumed_development_comparison",
        "distinct_goals": 15,
        "original_goal_indices": list(ORIGINAL_DEVELOPMENT_INDICES),
        "new_goal_indices": list(EXTENSION_INDICES),
        "reserved_confirmation_goal_indices": list(
            RESERVED_CONFIRMATION_INDICES
        ),
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "arms": {},
    }
    for arm in ("r10", "oracle", "direct"):
        original = _original_arm(args.root, arm)
        extension_path = output_paths(args.root)[arm] / "summary.json"
        if not extension_path.exists():
            raise FileNotFoundError(extension_path)
        extension_value = json.loads(extension_path.read_text())
        extension = {
            "tasks": int(extension_value["tasks"]),
            "successes": int(extension_value["successes"]),
            "success_rate": float(extension_value["success_rate"]),
            "source": str(extension_path.resolve()),
            "source_sha256": sha256_file(extension_path),
        }
        combined_tasks = original["tasks"] + extension["tasks"]
        combined_successes = original["successes"] + extension["successes"]
        result["arms"][arm] = {
            "original_9": original,
            "new_6": extension,
            "combined_15": {
                "tasks": combined_tasks,
                "successes": combined_successes,
                "success_rate": float(combined_successes / combined_tasks),
            },
        }
    oracle_original = result["arms"]["oracle"]["original_9"]["successes"]
    oracle_new = result["arms"]["oracle"]["new_6"]["successes"]
    r10_original = result["arms"]["r10"]["original_9"]["successes"]
    r10_new = result["arms"]["r10"]["new_6"]["successes"]
    result["r10_fraction_of_reachable_ceiling"] = {
        "original_9": float(r10_original / oracle_original),
        "new_6": float(r10_new / oracle_new) if oracle_new else None,
        "combined_15": float(
            (r10_original + r10_new) / (oracle_original + oracle_new)
        ),
    }
    dump(output_paths(args.root)["combined"], result)
    print(json.dumps(result["arms"], indent=2))
    print(json.dumps({
        "r10_fraction_of_reachable_ceiling": result[
            "r10_fraction_of_reachable_ceiling"
        ]
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-panel").set_defaults(func=command_prepare_panel)
    for name, function in (
        ("r10-task", command_r10_task),
        ("oracle-task", command_oracle_task),
        ("direct-task", command_direct_task),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--task-id", type=int, required=True)
        command.set_defaults(func=function)
    summarize = subparsers.add_parser("summarize-arm")
    summarize.add_argument("--arm", choices=("r10", "oracle", "direct"), required=True)
    summarize.set_defaults(func=command_summarize_arm)
    subparsers.add_parser("summarize-combined").set_defaults(
        func=command_summarize_combined
    )
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = (
            args.root
            / "model_planning"
            / "round2_recorded_transition_graph_r6.npz"
        )
    args.func(args)


if __name__ == "__main__":
    main()
