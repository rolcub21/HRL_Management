"""Current R11 planner on the exact E373 supplied-goal panel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pointmaze.gc_executor.runtime import (
    GCSACExecutorRuntime,
    achieved_xy,
    observation_vector,
)
from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv

from .core import atomic_json, load_json, protocol, resolve_repository, sha256
from .fields import SuppliedGoalField
from .planner import load_planner
from projects.tfsn_consequence_value_iteration.shared.settings import COMMAND_STEPS


PROTOCOL_PATH = Path(__file__).with_name("supplied_goal_protocol.json")


def verify_file(item, key):
    path = resolve_repository(item[key])
    hash_key = "sha256" if key == "path" else f"{key}_sha256"
    if sha256(path) != item[hash_key]:
        raise ValueError(f"supplied-goal frozen input changed: {key}")
    return path


def load_panel(contract, environment_seed):
    panel_contract = contract["panel"]
    if int(environment_seed) not in panel_contract["environment_seeds"]:
        raise ValueError("environment seed is outside the E373 panel")
    goals = pd.read_csv(verify_file(panel_contract, "goals"))
    starts = pd.read_csv(verify_file(panel_contract, "starts"))
    starts = starts[starts.seed.astype(int).eq(int(environment_seed))]
    panel = starts.merge(goals, on="goal_name", how="left", validate="many_to_one")
    source_episodes = int(
        panel_contract.get(
            "source_episodes_per_goal", panel_contract["episodes_per_goal_seed"]
        )
    )
    expected = int(panel_contract["goals_count"]) * source_episodes
    if len(panel) != expected or panel.goal_name.nunique() != int(
        panel_contract["goals_count"]
    ):
        raise ValueError("supplied-goal seed block is incomplete")
    return panel.sort_values(["goal_name", "episode"]).reset_index(drop=True)


def artifact_hashes(contract):
    artifacts = contract["r11_artifacts"]
    models = resolve_repository(artifacts["models"])
    roadmap = resolve_repository(artifacts["roadmap"])
    planner = resolve_repository(artifacts["planner"])
    model_meta = load_json(models / "factorized_consequence_metrics.json")
    road_meta = load_json(roadmap / "roadmap.json")
    bridge_meta = load_json(planner / "bridges.json")
    observed = {
        "factorized_consequence_sha256": model_meta["artifact_sha256"],
        "recorded_graph_sha256": road_meta["graph_sha256"],
        "bridges_sha256": bridge_meta["artifact_sha256"],
    }
    for key, value in observed.items():
        if value != artifacts[key]:
            raise ValueError(f"R11 supplied-goal artifact changed: {key}")
    return models, roadmap, planner, observed


def evaluate_episode(
    *, env, actor, roadmap_planner, goal, start, seed, episode,
    episode_steps, radius,
):
    env.reset(seed=int(seed) * 1000 + int(episode), start_xy=start)
    xy = achieved_xy(env.raw_obs).astype(np.float32)
    initial_distance = float(np.linalg.norm(xy - goal))
    best_distance = initial_distance
    success = bool(initial_distance <= radius)
    contact_step = 0 if success else None
    commands = 0
    decisions = []
    fields = SuppliedGoalField(roadmap_planner.sigma, goal)
    roadmap_planner.reset_task_memory()
    roadmap_planner.begin_episode()

    while commands * COMMAND_STEPS < episode_steps and not success:
        state = observation_vector(env.raw_obs)[:4].astype(np.float32)
        command, diagnostic = roadmap_planner.choose(state, commands, fields)
        option_start = xy.copy()
        command_best = float(np.linalg.norm(xy - goal))
        for primitive in range(COMMAND_STEPS):
            action = actor.act(env.raw_obs, command)
            time_step = env.step(action)
            xy = achieved_xy(env.raw_obs).astype(np.float32)
            distance = float(np.linalg.norm(xy - goal))
            best_distance = min(best_distance, distance)
            command_best = min(command_best, distance)
            if distance <= radius and contact_step is None:
                contact_step = commands * COMMAND_STEPS + primitive + 1
                success = True
            if time_step.last() and (
                commands * COMMAND_STEPS + primitive + 1 < episode_steps
            ):
                raise RuntimeError("PointMaze ended before the fixed budget")
        endpoint = observation_vector(env.raw_obs)[:4].astype(np.float32)
        roadmap_planner.observe_transition(state, command, endpoint)
        decisions.append({
            "command_index": int(commands),
            "start_xy": option_start.tolist(),
            "command_xy": np.asarray(command, np.float32).tolist(),
            "endpoint_xy": xy.tolist(),
            "travel_m": float(np.linalg.norm(xy - option_start)),
            "best_goal_distance_m": float(command_best),
            **diagnostic,
        })
        commands += 1

    return {
        "seed": int(seed),
        "episode": int(episode),
        "start_x": float(start[0]),
        "start_y": float(start[1]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "initial_goal_distance_m": initial_distance,
        "success": int(success),
        "steps": int(contact_step if contact_step is not None else episode_steps),
        "commands": int(commands),
        "best_goal_distance_m": float(best_distance),
        "task_gradient_updates": 0,
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-goals", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--development-subset", action="store_true")
    args = parser.parse_args()
    started = time.time()

    protocol_path = args.protocol.resolve()
    contract = load_json(protocol_path)
    verify_file(contract["r11_freeze"], "path")
    panel = load_panel(contract, args.environment_seed)
    protocol_episode_limit = contract["panel"].get("evaluation_episodes_per_goal")
    if protocol_episode_limit is not None:
        panel = panel[
            panel.episode.astype(int).lt(int(protocol_episode_limit))
        ]
        expected = int(contract["panel"]["goals_count"]) * int(
            protocol_episode_limit
        )
        if len(panel) != expected:
            raise ValueError("supplied-goal protocol episode subset is incomplete")
    if args.development_subset:
        subset = contract["development_subset"]
        if int(args.environment_seed) != int(subset["environment_seed"]):
            raise ValueError("development subset requires its frozen seed block")
        panel = panel[panel.goal_name.isin(subset["goal_names"])]
        panel = panel[panel.episode.astype(int) < int(subset["episodes_per_goal"])]
        if len(panel) != int(subset["rows"]):
            raise ValueError("supplied-goal development subset is incomplete")
    elif args.max_goals is not None:
        keep = panel.goal_name.drop_duplicates().iloc[: int(args.max_goals)]
        panel = panel[panel.goal_name.isin(keep)]
    if args.max_episodes is not None:
        panel = panel.groupby("goal_name", sort=False).head(int(args.max_episodes))
    panel = panel.reset_index(drop=True)

    models, roadmap, planner_dir, hashes = artifact_hashes(contract)
    roadmap_planner = load_planner(
        models, roadmap, planner_dir, args.device, seed=4101,
        advance_weighting=True, consumptive_lookahead=True,
    )
    base_contract = protocol()
    executor = base_contract["executor"]["seeds"][
        str(base_contract["executor"]["development_seed"])
    ]
    if sha256(executor["checkpoint"]) != executor["checkpoint_sha256"]:
        raise ValueError("frozen R11 executor changed")
    actor = GCSACExecutorRuntime(Path(executor["checkpoint"]), args.device)

    panel_contract = contract["panel"]
    episode_steps = int(panel_contract["episode_primitive_steps"])
    radius = float(panel_contract["success_radius_m_evaluator_only"])
    env = PointMazeTimeStepEnv(panel_contract["environment"], episode_steps)
    output = args.output.resolve()
    task_dir = output / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha256(protocol_path)
    try:
        for goal_name, frame in panel.groupby("goal_name", sort=True):
            shard = task_dir / f"{goal_name}__seed{args.environment_seed}.json"
            expected_episodes = frame.episode.astype(int).tolist()
            if shard.exists():
                saved = load_json(shard)
                if saved.get("protocol_sha256") != protocol_sha:
                    raise ValueError(f"stale supplied-goal shard: {shard}")
                if [row["episode"] for row in saved["episodes"]] != expected_episodes:
                    raise ValueError(f"episode contract changed: {shard}")
                continue
            goal = frame.iloc[0][["goal_x", "goal_y"]].to_numpy(np.float32)
            episodes = []
            for _, row in frame.iterrows():
                result = evaluate_episode(
                    env=env,
                    actor=actor,
                    roadmap_planner=roadmap_planner,
                    goal=goal,
                    start=row[["start_x", "start_y"]].to_numpy(np.float32),
                    seed=int(row.seed),
                    episode=int(row.episode),
                    episode_steps=episode_steps,
                    radius=radius,
                )
                result["goal_name"] = str(goal_name)
                episodes.append(result)
                print(
                    f"r11_supplied {goal_name} seed={args.environment_seed} "
                    f"ep={int(row.episode) + 1}/{len(frame)} "
                    f"success={result['success']} steps={result['steps']} "
                    f"commands={result['commands']}",
                    flush=True,
                )
            atomic_json(shard, {
                "schema_version": 1,
                "protocol_sha256": protocol_sha,
                "artifact_hashes": hashes,
                "goal_name": str(goal_name),
                "environment_seed": int(args.environment_seed),
                "episodes": episodes,
            })
    finally:
        env.close()

    shards = [load_json(path) for path in sorted(task_dir.glob("*.json"))]
    expected_goals = panel.goal_name.nunique()
    matching = [
        row for row in shards
        if int(row["environment_seed"]) == int(args.environment_seed)
    ]
    if len(matching) != expected_goals:
        raise ValueError("supplied-goal seed block is incomplete")
    episodes = [episode for task in matching for episode in task["episodes"]]
    flat = [{k: v for k, v in row.items() if k != "decisions"} for row in episodes]
    episode_frame = pd.DataFrame(flat).sort_values(["goal_name", "episode"])
    episode_frame.to_csv(
        output / "episodes.csv", index=False
    )
    success = np.asarray([row["success"] for row in episodes], np.float32)
    steps = np.asarray([row["steps"] for row in episodes], np.float32)
    comparator = None
    if "mega_comparator" in contract:
        mega_contract = contract["mega_comparator"]
        mega_path = verify_file(mega_contract, "episodes")
        mega = pd.read_csv(mega_path)
        paired = episode_frame.merge(
            mega[["seed", "goal_name", "episode", "success", "steps"]],
            on=["seed", "goal_name", "episode"],
            suffixes=("_r11", "_mega"),
            validate="one_to_one",
        )
        if len(paired) != len(episode_frame):
            raise ValueError("R11 supplied-goal episodes do not pair with MEGA")
        comparator = {
            "episodes": int(len(paired)),
            "training_seed": int(mega_contract["training_seed"]),
            "success_rate": float(paired.success_mega.mean()),
            "mean_capped_steps": float(paired.steps_mega.mean()),
            "r11_minus_mega_success_rate": float(
                paired.success_r11.mean() - paired.success_mega.mean()
            ),
            "paired_outcomes": {
                "both_succeed": int(
                    ((paired.success_r11 == 1) & (paired.success_mega == 1)).sum()
                ),
                "r11_only": int(
                    ((paired.success_r11 == 1) & (paired.success_mega == 0)).sum()
                ),
                "mega_only": int(
                    ((paired.success_r11 == 0) & (paired.success_mega == 1)).sum()
                ),
                "neither": int(
                    ((paired.success_r11 == 0) & (paired.success_mega == 0)).sum()
                ),
            },
        }
    subset_claim = None
    subset_deviations = []
    if args.development_subset:
        subset = contract["development_subset"]
        subset_claim = subset.get(
            "claim_level", "exposed_nine_goal_supplied_goal_development_comparison"
        )
        subset_deviations = subset.get(
            "declared_deviations",
            ["only_the_frozen_72_episode_development_subset_is_used"],
        )
    summary = {
        "schema_version": 1,
        "experiment_id": contract["experiment_id"],
        "method_reference": contract["method_reference"],
        "method_revision": contract["method_revision"],
        "base_method_revision": contract["base_method_revision"],
        "claim_level": subset_claim if args.development_subset else contract["claim_level"],
        "declared_deviations": contract["declared_deviations"] + subset_deviations,
        "oracle_inputs": contract["oracle_inputs"],
        "protocol_sha256": protocol_sha,
        "artifact_hashes": hashes,
        "environment_seed": int(args.environment_seed),
        "goals": int(expected_goals),
        "episodes": int(len(episodes)),
        "success_rate": float(success.mean()),
        "mean_capped_steps": float(steps.mean()),
        "goal_visible_at_time_zero": True,
        "search_phase_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": bool(
            contract.get("confirmation_panel_accessed", False)
        ),
        "reward_free_budget": contract["reward_free_budget"],
        "mega_comparator": comparator,
        "equal_budget_comparison": False,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
