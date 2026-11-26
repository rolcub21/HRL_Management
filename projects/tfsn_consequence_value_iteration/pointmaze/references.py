"""Direct-executor and geometry-oracle references on the exposed panel.

Both references obey the current method's fixed 100-primitive-step command
contract.  The goal and maze geometry are explicit oracle inputs to these
references only; neither result is a TF-SN result.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from scripts.pointmaze.gc_executor.runtime import (
    GCSACExecutorRuntime,
    achieved_xy,
    observation_vector,
)
from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv

from .core import atomic_json, load_json, protocol, resolve_repository, sha256
from .panel import load_development_panel
from projects.tfsn_consequence_value_iteration.shared.settings import COMMAND_STEPS


class GeometryOracle:
    """Shortest-path next-cell commands from native PointMaze geometry."""

    def __init__(self, env: PointMazeTimeStepEnv):
        maze = env._env.unwrapped.maze
        raw = np.asarray(maze.maze_map, dtype=object)
        self.maze = maze
        self.free = {
            (int(row), int(col))
            for row in range(raw.shape[0])
            for col in range(raw.shape[1])
            if raw[row, col] != 1
        }

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        value = np.asarray(self.maze.cell_xy_to_rowcol(xy), dtype=np.int64).reshape(2)
        cell = (int(value[0]), int(value[1]))
        if cell in self.free:
            return cell
        # This should not be needed, but makes numerical boundary behavior explicit.
        return min(
            self.free,
            key=lambda item: float(
                np.linalg.norm(
                    np.asarray(self.maze.cell_rowcol_to_xy(np.asarray(item))) - xy
                )
            ),
        )

    def next_command(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, int]:
        start = self.cell(current_xy)
        target = self.cell(goal_xy)
        if start == target:
            return np.asarray(goal_xy, dtype=np.float32).reshape(2), 0

        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        queue = deque([start])
        while queue and target not in parent:
            row, col = queue.popleft()
            for candidate in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if candidate in self.free and candidate not in parent:
                    parent[candidate] = (row, col)
                    queue.append(candidate)
        if target not in parent:
            raise RuntimeError(f"native maze has no route from {start} to {target}")

        reverse = [target]
        while reverse[-1] != start:
            reverse.append(parent[reverse[-1]])
        route = list(reversed(reverse))
        next_cell = route[1]
        command = np.asarray(
            self.maze.cell_rowcol_to_xy(np.asarray(next_cell)), dtype=np.float32
        ).reshape(2)
        return command, len(route) - 1


def evaluate_episode(*, system, row, env, actor, oracle, tau, episode_steps, radius):
    goal = row[["goal_x", "goal_y"]].to_numpy(dtype=np.float32)
    start = row[["start_x", "start_y"]].to_numpy(dtype=np.float32)
    seed = int(row.seed) * 1000 + int(row.episode)
    env.reset(seed=seed, start_xy=start)
    xy = achieved_xy(env.raw_obs)
    best = float(np.linalg.norm(xy - goal))
    first_contact = 0 if best <= radius else None
    simulated = 0
    commands = []

    while simulated < episode_steps and first_contact is None:
        if system == "direct_executor":
            command = goal.copy()
            route_cells = None
        elif system == "fixed_tau_geometry_oracle":
            command, route_cells = oracle.next_command(xy, goal)
        else:
            raise ValueError(system)
        start_xy = xy.copy()
        start_state = observation_vector(env.raw_obs)[:4].copy()
        contact_in_command = None
        for _ in range(tau):
            action = actor.act(env.raw_obs, command)
            time_step = env.step(action)
            simulated += 1
            if time_step.last() and simulated < episode_steps:
                raise RuntimeError("environment ended before the declared episode budget")
            xy = achieved_xy(env.raw_obs)
            distance = float(np.linalg.norm(xy - goal))
            best = min(best, distance)
            if distance <= radius and first_contact is None:
                first_contact = simulated
                contact_in_command = simulated
        endpoint_state = observation_vector(env.raw_obs)[:4].copy()
        commands.append({
            "command_index": len(commands),
            "start_xy": start_xy.tolist(),
            "start_state": start_state.tolist(),
            "command_xy": command.tolist(),
            "endpoint_xy": xy.tolist(),
            "endpoint_state": endpoint_state.tolist(),
            "travel_m": float(np.linalg.norm(xy - start_xy)),
            "route_cells_before": route_cells,
            "first_contact_step": contact_in_command,
        })

    return {
        "system": system,
        "goal_name": str(row.goal_name),
        "seed": int(row.seed),
        "episode": int(row.episode),
        "goal_xy": goal.tolist(),
        "start_xy": start.tolist(),
        "success": int(first_contact is not None),
        "first_contact_step": first_contact,
        "simulated_steps_fixed_command_boundary": int(simulated),
        "commands": len(commands),
        "best_goal_distance_m": best,
        "command_trace": commands,
    }


def summarize(rows):
    success = np.asarray([row["success"] for row in rows], dtype=np.float64)
    by_system = {}
    for system in sorted({row["system"] for row in rows}):
        chosen = [row for row in rows if row["system"] == system]
        values = np.asarray([row["success"] for row in chosen], dtype=np.float64)
        by_system[system] = {
            "episodes": len(chosen),
            "successes": int(values.sum()),
            "success_rate": float(values.mean()),
            "mean_best_goal_distance_m": float(
                np.mean([row["best_goal_distance_m"] for row in chosen])
            ),
        }
    return {"episodes": len(rows), "success_rate": float(success.mean()), "systems": by_system}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-episodes-per-task", type=int)
    args = parser.parse_args()
    started = time.time()
    contract = protocol()
    executor = contract["executor"]["seeds"][str(contract["executor"]["development_seed"])]
    checkpoint = Path(executor["checkpoint"])
    if sha256(checkpoint) != executor["checkpoint_sha256"]:
        raise ValueError("frozen development executor changed")

    panel = load_development_panel()
    task_names = panel.goal_name.drop_duplicates().tolist()
    if args.max_tasks is not None:
        task_names = task_names[: int(args.max_tasks)]
    panel = panel[panel.goal_name.isin(task_names)]
    if args.max_episodes_per_task is not None:
        panel = panel.groupby("goal_name", sort=False).head(int(args.max_episodes_per_task))
    panel = panel.sort_values(["goal_name", "episode"]).reset_index(drop=True)

    evaluator = contract["evaluator"]
    tau = COMMAND_STEPS
    episode_steps = int(evaluator["episode_primitive_steps"])
    radius = float(evaluator["success_radius_m_evaluator_only"])
    env = PointMazeTimeStepEnv(contract["environment"]["id"], episode_steps)
    actor = GCSACExecutorRuntime(checkpoint, args.device)
    oracle = GeometryOracle(env)
    rows = []
    shard_directory = args.output.resolve().parent / "episode_shards"
    try:
        for system in ("direct_executor", "fixed_tau_geometry_oracle"):
            for _, row in panel.iterrows():
                shard = shard_directory / (
                    f"{system}__{row.goal_name}__ep{int(row.episode):02d}.json"
                )
                if shard.exists():
                    result = load_json(shard)
                    expected = {
                        "system": system,
                        "goal_name": str(row.goal_name),
                        "seed": int(row.seed),
                        "episode": int(row.episode),
                        "goal_xy": row[["goal_x", "goal_y"]].to_numpy(
                            dtype=np.float32
                        ).tolist(),
                        "start_xy": row[["start_x", "start_y"]].to_numpy(
                            dtype=np.float32
                        ).tolist(),
                    }
                    for key, value in expected.items():
                        if result.get(key) != value:
                            raise ValueError(f"reference shard has another {key}: {shard}")
                    if any(
                        "start_state" not in trace or "endpoint_state" not in trace
                        for trace in result["command_trace"]
                    ):
                        raise ValueError(f"reference shard predates full-state logs: {shard}")
                else:
                    result = evaluate_episode(
                        system=system,
                        row=row,
                        env=env,
                        actor=actor,
                        oracle=oracle,
                        tau=tau,
                        episode_steps=episode_steps,
                        radius=radius,
                    )
                    atomic_json(shard, result)
                rows.append(result)
                print(
                    f"{system} {result['goal_name']} ep={result['episode'] + 1} "
                    f"success={result['success']} best={result['best_goal_distance_m']:.3f}",
                    flush=True,
                )
    finally:
        env.close()

    payload = {
        "schema_version": 1,
        "experiment_id": "pointmaze_r28_roadmap_references_r1",
        "method_reference": str(resolve_repository(contract["method_reference"])),
        "method_revision": contract["method_revision"],
        "claim_level": "development_reference_only",
        "declared_deviations": [
            "goal_is_supplied_to_both_reference_systems",
            "native_maze_geometry_is_supplied_only_to_the_oracle_reference",
        ],
        "oracle_inputs": [
            "supplied_goal_coordinate",
            "native_maze_geometry_for_fixed_tau_geometry_oracle",
        ],
        "command_primitive_steps": tau,
        "episode_primitive_steps": episode_steps,
        "success_radius_m_evaluator_only": radius,
        "contact_semantics": evaluator["contact_semantics"],
        "development_panel_tasks": len(task_names),
        "episodes_per_system": len(panel),
        "executor_checkpoint_sha256": executor["checkpoint_sha256"],
        "summary": summarize(rows),
        "episodes": rows,
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

