"""Hidden-goal development evaluation for the completed aligned PointMaze planner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.pointmaze.gc_executor.runtime import (
    GCSACExecutorRuntime,
    achieved_xy,
    observation_vector,
)
from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv

from .core import atomic_json, load_json, protocol, resolve_repository, sha256
from .fields import TaskFields
from .panel import load_development_panel, load_frozen_hidden_goal_panel
from .planner import load_planner
from .frozen_r74_planner import (
    METHOD_REVISION as FROZEN_R74_PORT_METHOD_REVISION,
    PROTOCOL as FROZEN_R74_PORT_PROTOCOL,
    load_protocol as load_frozen_r74_port_protocol,
    load_frozen_r74_planner,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    COMMAND_STEPS,
    CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH,
    CONSUMPTIVE_LOOKAHEAD_DEPTH,
)


METHOD_REVISION = "pointmaze_current_aligned_roadmap_hidden_goal_r5"
MOVEMENT_REACH_METHOD_REVISION = (
    "pointmaze_density_movement_reach_advance_hidden_goal_r8"
)
DETECTION_FOOTPRINT_METHOD_REVISION = (
    "pointmaze_exact_detection_footprint_hidden_goal_r9"
)
CONSUMPTIVE_LOOKAHEAD_METHOD_REVISION = (
    "pointmaze_exact_consumptive_lookahead_hidden_goal_r10"
)
REFERENCE_METHOD_REVISION = "pointmaze_continuous_reference_hidden_goal_r11"


def free_maze_cells(env):
    maze = env._env.unwrapped.maze
    raw = np.asarray(maze.maze_map, dtype=object)
    labels, centers = [], []
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            if str(raw[row, col]) == "1":
                continue
            labels.append(f"r{row:02d}_c{col:02d}")
            centers.append(
                np.asarray(maze.cell_rowcol_to_xy((row, col)), np.float32)
            )
    return labels, np.asarray(centers, np.float32)


def occupied_maze_cell(env, xy):
    maze = env._env.unwrapped.maze
    row, col = np.asarray(
        maze.cell_xy_to_rowcol(np.asarray(xy, np.float32)), np.int64
    ).reshape(2)
    return f"r{int(row):02d}_c{int(col):02d}"


def summarize_task(episodes):
    success = np.asarray([row["success"] for row in episodes], dtype=np.int64)
    steps = np.asarray([row["steps"] for row in episodes], dtype=np.int64)
    hits = np.flatnonzero(success)
    first = int(hits[0]) if len(hits) else -1
    post = success[first + 1 :] if first >= 0 else np.asarray([], dtype=np.int64)
    return {
        "mean_e2e": float(success.mean()),
        "first_contact_episode": first,
        "acquisition_at_1": float(first == 0),
        "acquisition_at_2": float(0 <= first < 2),
        "acquisition_at_5": float(0 <= first < 5),
        "acquisition_at_horizon": float(first >= 0),
        "conditional_return": float(post.mean()) if len(post) else None,
        "return_successes": int(post.sum()),
        "return_episodes": int(len(post)),
        "capped_contact_latency_steps": int(
            steps[: first + 1].sum() if first >= 0 else steps.sum()
        ),
    }


def evaluate_task(
    *, frame, planner, actor, env, tau, episode_steps, radius,
):
    frame = frame.sort_values("episode")
    goal_name = str(frame.iloc[0].goal_name)
    goal = frame.iloc[0][["goal_x", "goal_y"]].to_numpy(dtype=np.float32)
    fields = TaskFields(planner.sigma)
    planner.reset_task_memory()
    episodes, decisions = [], []
    first_contact_episode = -1
    cell_labels, cell_centers = free_maze_cells(env)
    previously_visited_cells = set()

    for _, panel_row in frame.iterrows():
        episode = int(panel_row.episode)
        start = panel_row[["start_x", "start_y"]].to_numpy(dtype=np.float32)
        started_post_contact = fields.contacted
        planner.begin_episode()
        unsearched_before = fields.search(cell_centers) > 0.5
        env.reset(seed=int(panel_row.seed) * 1000 + episode, start_xy=start)
        xy = achieved_xy(env.raw_obs).astype(np.float32)
        distance = float(np.linalg.norm(xy - goal))
        reward = float(distance <= radius)
        fields.extend(xy[None], np.asarray([reward], np.float32))
        success = bool(reward)
        first_contact_step = 0 if success else None
        best_distance = distance
        commands = 0
        episode_visited_cells = {occupied_maze_cell(env, xy)}

        while commands * tau < episode_steps and not success:
            state = observation_vector(env.raw_obs)[:4].astype(np.float32)
            command, diagnostic = planner.choose(state, commands, fields)
            command_start = xy.copy()
            command_best = distance
            contact_in_command = None
            primitive_xy, primitive_reward = [], []
            for primitive in range(tau):
                action = actor.act(env.raw_obs, command)
                time_step = env.step(action)
                xy = achieved_xy(env.raw_obs).astype(np.float32)
                distance = float(np.linalg.norm(xy - goal))
                reward = float(distance <= radius)
                primitive_xy.append(xy.copy())
                episode_visited_cells.add(occupied_maze_cell(env, xy))
                primitive_reward.append(reward)
                best_distance = min(best_distance, distance)
                command_best = min(command_best, distance)
                if reward and first_contact_step is None:
                    first_contact_step = commands * tau + primitive + 1
                    contact_in_command = first_contact_step
                    success = True
                if time_step.last() and commands * tau + primitive + 1 < episode_steps:
                    raise RuntimeError("PointMaze ended before the fixed episode budget")
            fields.extend(
                np.asarray(primitive_xy, np.float32),
                np.asarray(primitive_reward, np.float32),
            )
            if hasattr(planner, "observe_passage"):
                planner.observe_passage(primitive_xy, primitive_reward)
            endpoint_state = observation_vector(env.raw_obs)[:4].astype(
                np.float32
            )
            planner.observe_transition(state, command, endpoint_state)
            decisions.append({
                "goal_name": goal_name,
                "episode": episode,
                "command_index": commands,
                "start_xy": command_start.tolist(),
                "command_xy": np.asarray(command, np.float32).tolist(),
                "endpoint_xy": xy.tolist(),
                "travel_m": float(np.linalg.norm(xy - command_start)),
                "best_goal_distance_m": float(command_best),
                "contact_step": contact_in_command,
                **diagnostic,
            })
            commands += 1

        if success and first_contact_episode < 0:
            first_contact_episode = episode
        unsearched_after = fields.search(cell_centers) > 0.5
        newly_ruled_out = [
            label
            for label, before, after in zip(
                cell_labels, unsearched_before, unsearched_after
            )
            if before and not after
        ]
        newly_visited = sorted(episode_visited_cells - previously_visited_cells)
        previously_visited_cells.update(episode_visited_cells)
        episodes.append({
            "goal_name": goal_name,
            "episode": episode,
            "start_xy": start.tolist(),
            "started_post_contact": bool(started_post_contact),
            "success": int(success),
            "steps": int(first_contact_step if first_contact_step is not None else episode_steps),
            "commands": commands,
            "best_goal_distance_m": float(best_distance),
            "first_contact_episode": first_contact_episode,
            "visited_maze_cells": sorted(episode_visited_cells),
            "newly_visited_maze_cells": newly_visited,
            "newly_visited_maze_cell_count": len(newly_visited),
            "cumulative_visited_maze_cell_count": len(previously_visited_cells),
            "newly_ruled_out_maze_cells": newly_ruled_out,
            "newly_ruled_out_maze_cell_count": len(newly_ruled_out),
            "remaining_unsearched_maze_cell_centers": int(
                unsearched_after.sum()
            ),
        })
        if hasattr(planner, "end_episode"):
            planner.end_episode(fields.contacted, xy)
        print(
            f"point_current {goal_name} ep={episode + 1}/{len(frame)} "
            f"start_post={int(started_post_contact)} success={int(success)} "
            f"steps={episodes[-1]['steps']} commands={commands}",
            flush=True,
        )

    result = {
        "goal_name": goal_name,
        "goal_xy": goal.tolist(),
        "episodes": episodes,
        "decisions": decisions,
        "metrics": summarize_task(episodes),
    }
    if hasattr(planner, "diagnostics"):
        result["planner_diagnostics"] = planner.diagnostics()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-episodes-per-task", type=int)
    parser.add_argument("--advance-weighting", action="store_true")
    parser.add_argument("--reference-method", action="store_true")
    parser.add_argument("--goal-name", action="append", default=[])
    parser.add_argument("--consumptive-lookahead", action="store_true")
    parser.add_argument("--panel-protocol", type=Path)
    parser.add_argument("--frozen-r74-port", action="store_true")
    args = parser.parse_args()
    started = time.time()

    if args.reference_method:
        args.advance_weighting = True
        args.consumptive_lookahead = True
    if args.frozen_r74_port:
        args.reference_method = True
        args.advance_weighting = True
        args.consumptive_lookahead = True

    contract = protocol()
    evaluator = contract["evaluator"]
    executor = contract["executor"]["seeds"][str(contract["executor"]["development_seed"])]
    if sha256(executor["checkpoint"]) != executor["checkpoint_sha256"]:
        raise ValueError("frozen development executor changed")

    panel_contract = None
    panel_protocol_path = None
    if args.panel_protocol is not None:
        if not args.reference_method:
            raise ValueError("an external hidden-goal panel requires the reference method")
        panel_protocol_path = args.panel_protocol.resolve()
        panel_contract = load_json(panel_protocol_path)
        if resolve_repository(panel_contract["method_reference"]) != resolve_repository(
            contract["method_reference"]
        ):
            raise ValueError("hidden-goal panel method reference changed")
        freeze = panel_contract["r11_freeze"]
        if sha256(resolve_repository(freeze["path"])) != freeze["sha256"]:
            raise ValueError("hidden-goal panel R11 freeze changed")
        panel = load_frozen_hidden_goal_panel(panel_protocol_path)
    else:
        panel = load_development_panel()
    full_panel_rows = len(panel)
    full_panel_tasks = int(panel.goal_name.nunique())
    if args.goal_name:
        missing = sorted(set(args.goal_name) - set(panel.goal_name.unique()))
        if missing:
            raise ValueError(f"unknown exposed development goals: {missing}")
        panel = panel[panel.goal_name.isin(args.goal_name)]
    names = panel.goal_name.drop_duplicates().tolist()
    if args.max_tasks is not None:
        names = names[: int(args.max_tasks)]
    panel = panel[panel.goal_name.isin(names)]
    if args.max_episodes_per_task is not None:
        panel = panel.groupby("goal_name", sort=False).head(
            int(args.max_episodes_per_task)
        )
    panel = panel.sort_values(["goal_name", "episode"]).reset_index(drop=True)

    model_meta = load_json(args.models / "factorized_consequence_metrics.json")
    road_meta = load_json(args.roadmap / "roadmap.json")
    bridge_meta = load_json(args.planner / "bridges.json")
    artifact_hashes = {
        "factorized_consequence_sha256": model_meta["artifact_sha256"],
        "recorded_graph_sha256": road_meta["graph_sha256"],
        "bridges_sha256": bridge_meta["artifact_sha256"],
    }
    if panel_contract is not None:
        expected_artifacts = panel_contract["r11_artifacts"]
        for key, value in artifact_hashes.items():
            if value != expected_artifacts[key]:
                raise ValueError(f"hidden-goal R11 artifact changed: {key}")
    if args.frozen_r74_port:
        roadmap_planner = load_frozen_r74_planner(
            models=args.models,
            roadmap=args.roadmap,
            planner=args.planner,
            device=args.device,
            seed=4101,
        )
    else:
        roadmap_planner = load_planner(
            args.models,
            args.roadmap,
            args.planner,
            args.device,
            seed=4101,
            advance_weighting=args.advance_weighting,
            consumptive_lookahead=args.consumptive_lookahead,
        )
    if args.frozen_r74_port:
        method_revision = FROZEN_R74_PORT_METHOD_REVISION
    elif panel_contract is not None:
        method_revision = panel_contract["method_revision"]
    elif args.reference_method:
        method_revision = REFERENCE_METHOD_REVISION
    elif args.consumptive_lookahead:
        method_revision = CONSUMPTIVE_LOOKAHEAD_METHOD_REVISION
    else:
        method_revision = (
            MOVEMENT_REACH_METHOD_REVISION
            if args.advance_weighting else METHOD_REVISION
        )
    actor = GCSACExecutorRuntime(Path(executor["checkpoint"]), args.device)
    episode_steps = int(evaluator["episode_primitive_steps"])
    env = PointMazeTimeStepEnv(contract["environment"]["id"], episode_steps)
    output = args.output.resolve()
    task_dir = output / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        for goal_name, frame in panel.groupby("goal_name", sort=True):
            shard = task_dir / f"{goal_name}.json"
            if shard.exists():
                existing = load_json(shard)
                if existing.get("artifact_hashes") != artifact_hashes:
                    raise ValueError(f"stale development shard: {shard}")
                expected_episodes = frame.episode.astype(int).tolist()
                saved_episodes = [
                    int(row["episode"]) for row in existing.get("episodes", [])
                ]
                if saved_episodes != expected_episodes:
                    raise ValueError(
                        f"development shard has another episode contract: {shard}"
                    )
                continue
            result = evaluate_task(
                frame=frame,
                planner=roadmap_planner,
                actor=actor,
                env=env,
                tau=COMMAND_STEPS,
                episode_steps=episode_steps,
                radius=float(evaluator["success_radius_m_evaluator_only"]),
            )
            result.update({
                "method_revision": method_revision,
                "artifact_hashes": artifact_hashes,
            })
            atomic_json(shard, result)
    finally:
        env.close()

    tasks = [load_json(path) for path in sorted(task_dir.glob("*.json"))]
    if len(tasks) != len(names):
        raise ValueError("development evaluation is incomplete")
    metric_names = (
        "mean_e2e",
        "acquisition_at_1",
        "acquisition_at_2",
        "acquisition_at_5",
        "acquisition_at_horizon",
        "conditional_return",
        "capped_contact_latency_steps",
    )
    summary_metrics = {}
    for name in metric_names:
        values = [task["metrics"][name] for task in tasks if task["metrics"][name] is not None]
        summary_metrics[name] = float(np.mean(values)) if values else None
    used_panel_subset = len(names) < full_panel_tasks or len(panel) < full_panel_rows
    if args.frozen_r74_port:
        port_contract = load_frozen_r74_port_protocol()
        claim_level = port_contract["claim_level"]
        declared_deviations = list(port_contract["declared_deviations"])
        if used_panel_subset:
            claim_level = f"{claim_level}_subset"
            declared_deviations.append(
                "only_a_subset_of_the_declared_development_gate_is_used"
            )
        oracle_inputs = list(port_contract["oracle_inputs"])
        experiment_id = port_contract["experiment_id"]
    elif panel_contract is not None:
        claim_level = panel_contract["claim_level"]
        declared_deviations = list(panel_contract["declared_deviations"])
        if used_panel_subset:
            claim_level = f"{claim_level}_subset"
            declared_deviations.append(
                "only_a_subset_of_the_declared_hidden_goal_panel_is_used"
            )
        oracle_inputs = list(panel_contract["oracle_inputs"])
        experiment_id = panel_contract["experiment_id"]
    else:
        claim_level = (
            "consumed_development_reference_gate_only"
            if args.reference_method
            and len(names) < int(contract["development_panel"]["tasks"])
            else (
                "consumed_development_reference_panel_only"
                if args.reference_method
                else (
                    "consumed_development_gate_only"
                    if len(names) < int(contract["development_panel"]["tasks"])
                    else "consumed_development_panel_only"
                )
            )
        )
        declared_deviations = (
            ["only_a_subset_of_exposed_development_tasks_or_episodes_is_used"]
            if used_panel_subset else []
        )
        oracle_inputs = []
        experiment_id = method_revision

    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "method_reference": (
            load_frozen_r74_port_protocol()["method_reference"]
            if args.frozen_r74_port else contract["method_reference"]
        ),
        "method_protocol_sha256": (
            sha256(FROZEN_R74_PORT_PROTOCOL)
            if args.frozen_r74_port else None
        ),
        "method_revision": method_revision,
        "claim_level": claim_level,
        "declared_deviations": declared_deviations,
        "oracle_inputs": oracle_inputs,
        "panel_protocol_sha256": (
            sha256(panel_protocol_path) if panel_protocol_path is not None else None
        ),
        "tasks": len(tasks),
        "episodes_per_task": int(panel.groupby("goal_name").size().iloc[0]),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_memory_persisted_across_physical_resets": True,
        "task_reward_used_for_learning": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": bool(
            panel_contract.get("confirmation_panel_accessed", False)
            if panel_contract is not None else False
        ),
        "fixed_command_duration": COMMAND_STEPS,
        "continuous_advance_weighting": bool(args.advance_weighting),
        "episode_primitive_steps": episode_steps,
        "success_radius_evaluator_only_m": float(evaluator["success_radius_m_evaluator_only"]),
        "success_radius_read_by_acquisition_planner": False,
        "acquisition_search_posterior": (
            "area_weighted_latent_radius_exact_posterior"
            if args.frozen_r74_port
            else "gaussian_complement_at_landing_precision"
        ),
        "acquisition_detection_radius_m": None,
        "return_field_sigma_m": float(roadmap_planner.sigma),
        "consumptive_lookahead": bool(args.consumptive_lookahead),
        "reward_free_online_consequence_correction": bool(
            args.reference_method
        ),
        "frozen_r74_cross_domain_port": bool(args.frozen_r74_port),
        "consumptive_lookahead_depth_global": (
            int(CONSUMPTIVE_LOOKAHEAD_DEPTH)
            if args.consumptive_lookahead else None
        ),
        "consumptive_lookahead_beam_width_global": (
            int(CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH)
            if args.consumptive_lookahead else None
        ),
        "artifact_hashes": artifact_hashes,
        "metrics": summary_metrics,
        "task_metrics": [
            {"goal_name": task["goal_name"], **task["metrics"]} for task in tasks
        ],
        "task_planner_diagnostics": [
            {
                "goal_name": task["goal_name"],
                **task.get("planner_diagnostics", {}),
            }
            for task in tasks
        ],
        "maze_cell_coverage_by_episode": [
            {
                "goal_name": task["goal_name"],
                "episodes": [
                    {
                        "episode": int(episode["episode"]),
                        "started_post_contact": bool(
                            episode["started_post_contact"]
                        ),
                        "newly_visited_maze_cells": episode[
                            "newly_visited_maze_cells"
                        ],
                        "newly_ruled_out_maze_cells": episode[
                            "newly_ruled_out_maze_cells"
                        ],
                        "remaining_unsearched_maze_cell_centers": int(
                            episode["remaining_unsearched_maze_cell_centers"]
                        ),
                    }
                    for episode in task["episodes"]
                ],
            }
            for task in tasks
        ],
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    atomic_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
