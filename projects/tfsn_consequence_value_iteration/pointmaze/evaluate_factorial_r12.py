"""Evaluate the isolated PointMaze latent-memory x passage 2x2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.pointmaze.gc_executor.runtime import (
    GCSACExecutorRuntime,
    achieved_xy,
    observation_vector,
)
from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv

from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    PosteriorFalsifiedError,
)
from projects.tfsn_consequence_value_iteration.shared.settings import COMMAND_STEPS

from .core import atomic_json, load_json, protocol, sha256
from .factorial_planner_r12 import MANIFEST, load_factorial_planner
from .fields import TaskFields
from .panel import load_development_panel


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "factorial_protocol_r12.json"
REFERENCE = HERE / "R12_FACTORIAL_REFERENCE.md"
METHOD_REVISION = "pointmaze_latent_detection_passage_factorial_r12"
CLAIM_LEVEL = "consumed_development_factorial_only"
REQUIRED_ARM = None
ALLOWED_ARMS = None
REQUIRED_COMMAND_CAP = None
PLANNER_LOADER = load_factorial_planner
METHOD_MANIFEST = MANIFEST
GOAL_NAMES = (
    "spatial_diagnostic_native_goal_r01_c06",
    "spatial_diagnostic_native_goal_r06_c06",
)


def task_metrics(episodes):
    success = np.asarray([row["success"] for row in episodes], np.int64)
    hits = np.flatnonzero(success)
    first = int(hits[0]) if len(hits) else None
    post = success[first + 1:] if first is not None else np.asarray([], np.int64)
    return {
        "mean_e2e": float(success.mean()),
        "first_contact_episode": first,
        "first_contact_episode_ordinal": first + 1 if first is not None else None,
        "acquisition_at_horizon": float(first is not None),
        "conditional_return": float(post.mean()) if len(post) else None,
        "contact_latency": int(sum(row["steps"] for row in episodes[: first + 1]))
        if first is not None else int(sum(row["steps"] for row in episodes)),
    }


def evaluate_task(
    *, frame, planner, actor, env, radius, episode_steps, output,
    command_cap, benchmark_horizon,
):
    frame = frame.sort_values("episode")
    goal_name = str(frame.iloc[0].goal_name)
    goal = frame.iloc[0][["goal_x", "goal_y"]].to_numpy(dtype=np.float32)
    arm = f"{planner.observation_level}__{planner.passage_level}"
    fields = TaskFields(planner.sigma)
    planner.reset_task_memory()
    episodes, decisions = [], []
    print(
        f"[point-r12][evaluate] arm={arm} goal={goal_name} "
        f"episodes={len(frame)} start",
        flush=True,
    )
    try:
        for local_episode, (_, panel_row) in enumerate(frame.iterrows()):
            episode = int(panel_row.episode)
            start = panel_row[["start_x", "start_y"]].to_numpy(dtype=np.float32)
            planner.begin_episode()
            env.reset(seed=int(panel_row.seed) * 1000 + episode, start_xy=start)
            xy = achieved_xy(env.raw_obs).astype(np.float32)
            closest_distance = float(np.linalg.norm(xy - goal))
            reward = float(np.linalg.norm(xy - goal) <= radius)
            fields.extend(xy[None], [reward])
            planner.observe_detection_consequence(xy[None], [reward])
            success = bool(reward)
            first_contact_step = 0 if success else None
            commands = 0
            print(
                f"[point-r12][evaluate] arm={arm} goal={goal_name} "
                f"episode={local_episode + 1}/{len(frame)} reset",
                flush=True,
            )
            while commands * COMMAND_STEPS < episode_steps and not success:
                state = observation_vector(env.raw_obs)[:4].astype(np.float32)
                command, diagnostic = planner.choose(state, commands, fields)
                primitive_xy, primitive_reward = [], []
                contact_in_command = None
                for primitive in range(COMMAND_STEPS):
                    time_step = env.step(actor.act(env.raw_obs, command))
                    xy = achieved_xy(env.raw_obs).astype(np.float32)
                    closest_distance = min(
                        closest_distance, float(np.linalg.norm(xy - goal))
                    )
                    reward = float(np.linalg.norm(xy - goal) <= radius)
                    primitive_xy.append(xy.copy())
                    primitive_reward.append(reward)
                    if reward and first_contact_step is None:
                        first_contact_step = commands * COMMAND_STEPS + primitive + 1
                        contact_in_command = first_contact_step
                        success = True
                    if time_step.last() and (
                        commands * COMMAND_STEPS + primitive + 1 < episode_steps
                    ):
                        raise RuntimeError("PointMaze ended before fixed episode budget")
                primitive_xy = np.asarray(primitive_xy, np.float32)
                primitive_reward = np.asarray(primitive_reward, np.float32)
                fields.extend(primitive_xy, primitive_reward)
                planner.observe_detection_consequence(
                    primitive_xy, primitive_reward
                )
                endpoint_state = observation_vector(env.raw_obs)[:4].astype(np.float32)
                planner.observe_transition(state, command, endpoint_state)
                decisions.append({
                    "goal_name": goal_name,
                    "episode": episode,
                    "command_index": commands,
                    "command_xy": np.asarray(command, np.float32).tolist(),
                    "endpoint_xy": xy.tolist(),
                    "contact_step": contact_in_command,
                    **diagnostic,
                })
                commands += 1
                region_progress = ""
                if "selected_region_index" in diagnostic:
                    region_progress = (
                        f" target_region={diagnostic['selected_region_index']}"
                        f" region_mass="
                        f"{diagnostic['selected_region_posterior_mass']:.6f}"
                        f" node_local_mass="
                        f"{diagnostic['selected_node_local_contact_mass']:.6f}"
                        f" command_xy=({float(command[0]):.3f},"
                        f"{float(command[1]):.3f})"
                        f" endpoint_xy=({float(xy[0]):.3f},{float(xy[1]):.3f})"
                    )
                print(
                    f"[point-r12][evaluate] arm={arm} goal={goal_name} "
                    f"episode={local_episode + 1}/{len(frame)} "
                    f"command={commands}/{episode_steps // COMMAND_STEPS} "
                    f"phase={diagnostic['phase']} reward={int(success)} "
                    f"closest_evaluator_m={closest_distance:.3f}"
                    f"{region_progress}",
                    flush=True,
                )
            posterior = planner.factorial.posterior
            corner_distance = np.linalg.norm(
                posterior.goal_xy - goal[None], axis=1
            )
            nearest_corner_candidate = int(np.argmin(corner_distance))
            corner_probability = float(
                posterior.goal_probability[nearest_corner_candidate]
            )
            episodes.append({
                "episode": episode,
                "start_xy": start.tolist(),
                "success": int(success),
                "steps": int(
                    first_contact_step
                    if first_contact_step is not None else episode_steps
                ),
                "commands": commands,
                # Evaluator-only, post-hoc diagnostics. These quantities are
                # never supplied to the planner or executor.
                "closest_evaluator_distance_m": closest_distance,
                "posthoc_nearest_corner_candidate_distance_m": float(
                    corner_distance[nearest_corner_candidate]
                ),
                "posthoc_nearest_corner_candidate_probability": (
                    corner_probability
                ),
                "posthoc_nearest_corner_candidate_probability_rank": int(
                    1 + np.count_nonzero(
                        posterior.goal_probability > corner_probability
                    )
                ),
                "posthoc_evaluator_contact_region_probability_mass": float(
                    posterior.goal_probability[corner_distance <= radius].sum()
                ),
            })
            print(
                f"[point-r12][evaluate] arm={arm} goal={goal_name} "
                f"episode={local_episode + 1}/{len(frame)} "
                f"success={int(success)} "
                f"closest_evaluator_m={closest_distance:.3f}",
                flush=True,
            )
    except PosteriorFalsifiedError as error:
        failure = {
            "method_revision": METHOD_REVISION,
            "arm": arm,
            "goal_name": goal_name,
            "status": "model_falsified",
            "error": str(error),
            **planner.factorial.diagnostic(),
        }
        atomic_json(output / f"{goal_name}_MODEL_FALSIFIED.json", failure)
        raise
    result = {
        "method_revision": METHOD_REVISION,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_manifest_sha256": sha256(METHOD_MANIFEST),
        "protocol_sha256": sha256(PROTOCOL),
        "claim_level": CLAIM_LEVEL,
        "oracle_inputs": [],
        "goal_name": goal_name,
        "arm": arm,
        "episodes": episodes,
        "decisions": decisions,
        "metrics": task_metrics(episodes),
        "success_radius_read_by_planner": False,
        "evaluation_command_cap": int(command_cap),
        "benchmark_horizon_commands": int(benchmark_horizon),
        "truncated_horizon": bool(command_cap < benchmark_horizon),
        **planner.factorial.diagnostic(),
    }
    atomic_json(output / f"{goal_name}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observation-level", required=True)
    parser.add_argument("--passage-level", required=True)
    parser.add_argument("--passage-artifact", type=Path)
    parser.add_argument(
        "--command-cap", type=int,
        help="Exploratory command horizon; omitted means the benchmark horizon.",
    )
    args = parser.parse_args()
    if args.passage_level == "shared_learned" and args.passage_artifact is None:
        raise ValueError("shared_learned requires --passage-artifact")
    arm = f"{args.observation_level}__{args.passage_level}"
    if REQUIRED_ARM is not None and arm != REQUIRED_ARM:
        raise ValueError(f"this evaluator requires arm {REQUIRED_ARM}")
    if ALLOWED_ARMS is not None and arm not in set(ALLOWED_ARMS):
        raise ValueError(
            f"this evaluator permits only {sorted(ALLOWED_ARMS)}, got {arm}"
        )
    contract = protocol()
    evaluator = contract["evaluator"]
    executor = contract["executor"]["seeds"][
        str(contract["executor"]["development_seed"])
    ]
    if sha256(executor["checkpoint"]) != executor["checkpoint_sha256"]:
        raise ValueError("frozen PointMaze executor changed")
    screen = load_json(PROTOCOL)
    episodes_per_goal = int(screen["episodes_per_goal"])
    if episodes_per_goal < 1:
        raise ValueError("episodes_per_goal must be positive")
    panel = load_development_panel()
    panel = panel[panel.goal_name.isin(GOAL_NAMES)]
    panel = panel.groupby("goal_name", sort=False).head(episodes_per_goal)
    if set(panel.goal_name) != set(GOAL_NAMES) or not panel.groupby(
        "goal_name"
    ).size().eq(episodes_per_goal).all():
        raise ValueError(
            "PointMaze evaluation panel is not exactly "
            f"{len(GOAL_NAMES)} goals x {episodes_per_goal} episodes"
        )
    output = args.output / arm
    output.mkdir(parents=True, exist_ok=True)
    benchmark_steps = int(evaluator["episode_primitive_steps"])
    if benchmark_steps % COMMAND_STEPS:
        raise ValueError("PointMaze episode budget is not divisible by command steps")
    benchmark_horizon = benchmark_steps // COMMAND_STEPS
    command_cap = (
        benchmark_horizon if args.command_cap is None else int(args.command_cap)
    )
    if not 1 <= command_cap <= benchmark_horizon:
        raise ValueError("command cap must lie within the benchmark horizon")
    if REQUIRED_COMMAND_CAP is not None and command_cap != REQUIRED_COMMAND_CAP:
        raise ValueError(
            f"this evaluator requires command cap {REQUIRED_COMMAND_CAP}"
        )
    evaluation_steps = command_cap * COMMAND_STEPS
    point_planner = PLANNER_LOADER(
        models=args.models,
        roadmap=args.roadmap,
        planner=args.planner,
        device=args.device,
        seed=12049,
        observation_level=args.observation_level,
        passage_level=args.passage_level,
        passage_artifact=args.passage_artifact,
    )
    point_planner.horizon = command_cap
    point_planner.base.horizon = command_cap
    actor = GCSACExecutorRuntime(Path(executor["checkpoint"]), args.device)
    env = PointMazeTimeStepEnv(
        contract["environment"]["id"], evaluation_steps
    )
    try:
        rows = []
        for goal_name, frame in panel.groupby("goal_name", sort=True):
            shard = output / f"{goal_name}.json"
            if shard.exists():
                existing = load_json(shard)
                expected = {
                    "method_revision": METHOD_REVISION,
                    "method_reference_sha256": sha256(REFERENCE),
                    "method_manifest_sha256": sha256(METHOD_MANIFEST),
                    "protocol_sha256": sha256(PROTOCOL),
                    "arm": arm,
                    "goal_name": str(goal_name),
                }
                for name, value in expected.items():
                    if existing.get(name) != value:
                        raise ValueError(f"stale PointMaze factorial shard: {shard}")
                if len(existing.get("episodes", [])) != episodes_per_goal:
                    raise ValueError(f"incomplete PointMaze factorial shard: {shard}")
                print(
                    f"[point-r12][evaluate] arm={arm} goal={goal_name} "
                    "already complete; reusing shard",
                    flush=True,
                )
                rows.append(existing)
                continue
            rows.append(evaluate_task(
                frame=frame,
                planner=point_planner,
                actor=actor,
                env=env,
                radius=float(evaluator["success_radius_m_evaluator_only"]),
                episode_steps=evaluation_steps,
                output=output,
                command_cap=command_cap,
                benchmark_horizon=benchmark_horizon,
            ))
    finally:
        env.close()
    summary = {
        "method_revision": METHOD_REVISION,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_manifest_sha256": sha256(METHOD_MANIFEST),
        "protocol_sha256": sha256(PROTOCOL),
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": load_json(PROTOCOL)["declared_deviations"],
        "oracle_inputs": [],
        "arm": arm,
        "goals": len(rows),
        "episodes_per_goal": episodes_per_goal,
        "evaluation_command_cap": command_cap,
        "benchmark_horizon_commands": benchmark_horizon,
        "truncated_horizon": bool(command_cap < benchmark_horizon),
        "acquired_goals": int(sum(
            row["metrics"]["first_contact_episode"] is not None for row in rows
        )),
        "mean_e2e": float(np.mean([row["metrics"]["mean_e2e"] for row in rows])),
        "rows": rows,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
