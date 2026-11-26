"""Resumable reward-free acquisition and synthetic-return alignment."""

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

from .core import atomic_json, atomic_npz, load_json, protocol, sha256
from .fields import TaskFields
from .planner import load_planner
from projects.tfsn_consequence_value_iteration.shared.settings import (
    ALIGNMENT_COMMANDS_PER_TASK,
    ALIGNMENT_TASKS_PER_PHASE,
    COMMAND_STEPS,
    finite_horizon,
)


METHOD_REVISION = "pointmaze_current_planner_alignment_r2"
ROUND2_METHOD_REVISION = "pointmaze_current_planner_alignment_round2_r4"


def _anchor_support(planner):
    xy = np.concatenate(
        (planner.graph.nodes[:, :2], planner.graph.primitive_xy.reshape(-1, 2)),
        axis=0,
    )
    cells = np.floor(xy / planner.sigma).astype(np.int32)
    _unique, index = np.unique(cells, axis=0, return_index=True)
    return xy[np.sort(index)].astype(np.float32)


def _episode_path(output, phase, task_id, episode):
    return output / "shards" / phase / (
        f"task_{task_id:03d}_episode_{episode:02d}.npz"
    )


def _load_episode(path, expected_commands, tau):
    with np.load(path, allow_pickle=False) as stored:
        value = {name: stored[name] for name in stored.files}
    required = {
        "start_observation": (expected_commands, 4),
        "command_xy": (expected_commands, 2),
        "endpoint_observation": (expected_commands, 4),
        "primitive_observation": (expected_commands, tau, 4),
        "primitive_reward": (expected_commands, tau),
        "reset_observation": (4,),
        "reset_reward": (),
    }
    for name, shape in required.items():
        if name not in value or value[name].shape != shape:
            raise ValueError(f"invalid alignment shard {path}: {name}")
    return value


def _restore_memory(output, phase, task_id, episode_lengths, tau):
    fields_xy, fields_reward = [], []
    for episode, expected_commands in enumerate(episode_lengths):
        path = _episode_path(output, phase, task_id, episode)
        if not path.exists():
            break
        value = _load_episode(path, expected_commands, tau)
        fields_xy.append(value["reset_observation"][None, :2])
        fields_reward.append(
            np.asarray([value["reset_reward"]], np.float32)
        )
        fields_xy.append(value["primitive_observation"][:, :, :2].reshape(-1, 2))
        fields_reward.append(value["primitive_reward"].reshape(-1))
    return fields_xy, fields_reward


def _reward(xy, phase, anchor, radius):
    if phase != "return":
        return 0.0
    return float(np.linalg.norm(np.asarray(xy) - anchor) <= radius)


def collect_task(
    *, phase, task_id, commands_total, planner, actor, env, output,
    anchor, radius, tau, commands_per_episode, seed,
):
    episode_lengths = []
    remaining = int(commands_total)
    while remaining:
        count = min(commands_per_episode, remaining)
        episode_lengths.append(count)
        remaining -= count
    restored_xy, restored_reward = _restore_memory(
        output, phase, task_id, episode_lengths, tau
    )
    fields = TaskFields(planner.sigma)
    for xy, reward in zip(restored_xy, restored_reward):
        fields.extend(xy, reward)
    planner.reset_task_memory()
    for episode, expected_commands in enumerate(episode_lengths):
        path = _episode_path(output, phase, task_id, episode)
        if not path.exists():
            break
        value = _load_episode(path, expected_commands, tau)
        planner.begin_episode()
        for start, command, endpoint in zip(
            value["start_observation"],
            value["command_xy"],
            value["endpoint_observation"],
        ):
            planner.observe_transition(start, command, endpoint)

    rows = []
    for episode, command_count in enumerate(episode_lengths):
        path = _episode_path(output, phase, task_id, episode)
        if path.exists():
            value = _load_episode(path, command_count, tau)
            rows.append(value)
            continue
        reset_seed = int(seed + 10_000_019 * task_id + 1009 * episode)
        planner.begin_episode()
        env.reset(seed=reset_seed)
        reset = observation_vector(env.raw_obs)[:4].astype(np.float32)
        reset_reward = _reward(reset[:2], phase, anchor, radius)
        fields.extend(reset[None, :2], np.asarray([reset_reward], np.float32))
        starts, commands, endpoints = [], [], []
        primitive, rewards, diagnostics = [], [], []
        for command_index in range(command_count):
            state = observation_vector(env.raw_obs)[:4].astype(np.float32)
            command, diagnostic = planner.choose(
                state, command_index, fields
            )
            command_states, command_rewards = [], []
            for _ in range(tau):
                action = actor.act(env.raw_obs, command)
                env.step(action)
                observation = observation_vector(env.raw_obs)[:4].astype(np.float32)
                reward = _reward(observation[:2], phase, anchor, radius)
                command_states.append(observation)
                command_rewards.append(reward)
            endpoint = observation_vector(env.raw_obs)[:4].astype(np.float32)
            command_states = np.asarray(command_states, np.float32)
            command_rewards = np.asarray(command_rewards, np.float32)
            fields.extend(command_states[:, :2], command_rewards)
            planner.observe_transition(state, command, endpoint)
            starts.append(state)
            commands.append(command)
            endpoints.append(endpoint)
            primitive.append(command_states)
            rewards.append(command_rewards)
            diagnostics.append(diagnostic)
            print(
                f"point_alignment {phase} task={task_id + 1} "
                f"episode={episode + 1}/{len(episode_lengths)} "
                f"command={command_index + 1}/{command_count} "
                f"contacted={int(fields.contacted)} "
                f"move={np.linalg.norm(endpoint[:2] - state[:2]):.3f}",
                flush=True,
            )
        atomic_npz(
            path,
            start_observation=np.asarray(starts, np.float32),
            command_xy=np.asarray(commands, np.float32),
            endpoint_observation=np.asarray(endpoints, np.float32),
            primitive_observation=np.asarray(primitive, np.float32),
            primitive_reward=np.asarray(rewards, np.float32),
            reset_observation=reset,
            reset_reward=np.asarray(reset_reward, np.float32),
            reset_seed=np.asarray(reset_seed, np.int64),
            task_id=np.asarray(task_id, np.int32),
            physical_episode=np.asarray(episode, np.int32),
        )
        atomic_json(path.with_suffix(".json"), {
            "phase": phase,
            "task_id": task_id,
            "physical_episode": episode,
            "commands": command_count,
            "diagnostics": diagnostics,
        })
        rows.append(_load_episode(path, command_count, tau))
    return rows


def summarize(
    output, phase, tasks, commands_per_task, commands_per_episode, tau
):
    move, command_distance, contact = [], [], []
    rows = 0
    for task_id in range(tasks):
        remaining, episode = commands_per_task, 0
        while remaining:
            count = min(commands_per_episode, remaining)
            value = _load_episode(
                _episode_path(output, phase, task_id, episode), count, tau
            )
            move.extend(np.linalg.norm(
                value["endpoint_observation"][:, :2]
                - value["start_observation"][:, :2], axis=1
            ).tolist())
            command_distance.extend(np.linalg.norm(
                value["command_xy"] - value["start_observation"][:, :2], axis=1
            ).tolist())
            contact.append(bool(value["primitive_reward"].max() > 0.0))
            rows += count
            remaining -= count
            episode += 1
    return {
        "rows": rows,
        "realized_displacement_mean_m": float(np.mean(move)),
        "realized_displacement_median_m": float(np.median(move)),
        "command_distance_mean_m": float(np.mean(command_distance)),
        "physical_episodes_with_synthetic_contact": int(sum(contact)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("acquisition", "return", "both"), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--alignment-round", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    started = time.time()
    contract = protocol()
    tasks = 1 if args.profile == "smoke" else ALIGNMENT_TASKS_PER_PHASE
    commands_per_task = 2 if args.profile == "smoke" else int(
        ALIGNMENT_COMMANDS_PER_TASK
    )
    tau = COMMAND_STEPS
    commands_per_episode = finite_horizon(
        contract["evaluator"]["episode_primitive_steps"]
    )
    executor = contract["executor"]["seeds"][str(contract["executor"]["development_seed"])]
    if sha256(executor["checkpoint"]) != executor["checkpoint_sha256"]:
        raise ValueError("frozen PointMaze executor changed")
    planner = load_planner(
        args.models, args.roadmap, args.planner, args.device, args.seed
    )
    actor = GCSACExecutorRuntime(Path(executor["checkpoint"]), args.device)
    env = PointMazeTimeStepEnv(
        contract["environment"]["id"],
        int(contract["evaluator"]["episode_primitive_steps"]),
    )
    output = args.output.resolve()
    anchors = _anchor_support(planner)
    phases = ("acquisition", "return") if args.phase == "both" else (args.phase,)
    try:
        for phase_index, phase in enumerate(phases):
            for task_id in range(tasks):
                rng = np.random.default_rng(
                    args.seed + 100_003 * phase_index + 1_000_003 * task_id
                )
                anchor = (
                    anchors[int(rng.integers(len(anchors)))]
                    if phase == "return" else None
                )
                collect_task(
                    phase=phase,
                    task_id=task_id,
                    commands_total=commands_per_task,
                    planner=planner,
                    actor=actor,
                    env=env,
                    output=output,
                    anchor=anchor,
                    radius=float(contract["evaluator"]["success_radius_m_evaluator_only"]),
                    tau=tau,
                    commands_per_episode=commands_per_episode,
                    seed=args.seed + 100_003 * phase_index,
                )
    finally:
        env.close()
    phase_summary = {
        phase: summarize(
            output, phase, tasks, commands_per_task, commands_per_episode, tau
        )
        for phase in phases
    }
    method_revision = (
        METHOD_REVISION if args.alignment_round == 1
        else ROUND2_METHOD_REVISION
    )
    model_metrics = load_json(args.models / "factorized_consequence_metrics.json")
    roadmeta = load_json(args.roadmap / "roadmap.json")
    bridge_meta = load_json(args.planner / "bridges.json")
    report = {
        "schema_version": 1,
        "experiment_id": method_revision,
        "method_reference": contract["method_reference"],
        "method_revision": method_revision,
        "claim_level": (
            "smoke_only" if args.profile == "smoke"
            else "reward_free_current_planner_alignment_data_only"
        ),
        "declared_deviations": [],
        "oracle_inputs": [],
        "profile": args.profile,
        "alignment_round": int(args.alignment_round),
        "phases": list(phases),
        "tasks_per_phase": tasks,
        "commands_per_task": commands_per_task,
        "fixed_command_duration": tau,
        "physical_episode_commands": commands_per_episode,
        "task_memory_persisted_across_physical_resets": True,
        "synthetic_return_anchor_rule": (
            "uniform_over_landing_precision_cells_of_reward_free_support"
        ),
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "planner_artifacts": {
            "factorized_consequence_sha256": model_metrics["artifact_sha256"],
            "recorded_graph_sha256": roadmeta["graph_sha256"],
            "bridges_sha256": bridge_meta["artifact_sha256"],
        },
        "phase_summary": phase_summary,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    atomic_json(output / f"summary_{args.phase}_{args.profile}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
