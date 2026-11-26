"""Collect a resumable reward-free alignment round with frozen R28."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_bounded_consumption_r28 as r28,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from scripts.antmaze.current_tfsn.collect import execute_command, task_fields
from scripts.antmaze.current_tfsn.constants import (
    COMMANDS_PER_EPISODE,
    COMMAND_STEPS,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.executor import full_state
from scripts.antmaze.current_tfsn.main import dump


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "r29_alignment_protocol.json"
METHOD_REFERENCE = HERE / "R29_ALIGNMENT_REFERENCE.md"
METHOD_REVISION = "antmaze_r28_reward_free_consequence_alignment_r29"
CLAIM_LEVEL = "reward_free_R28_behavior_alignment_data_only"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dimensions(profile, protocol):
    if profile == "full":
        return (
            int(protocol["preparation_tasks"]),
            int(protocol["commands_per_preparation_task"]),
        )
    # Cross one physical reset boundary in the smoke test.
    return 1, COMMANDS_PER_EPISODE + 2


def validate_protocol(protocol):
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "oracle_inputs": [],
        "fixed_command_duration": COMMAND_STEPS,
        "commands_per_physical_episode": COMMANDS_PER_EPISODE,
        "confirmation_panel_accessed": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R29 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = Path.cwd() / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R29 method reference changed")
    return reference


def load_shard(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def validate_shard(path, task_id, commands):
    shard = load_shard(path)
    expected = {
        "states": (commands, COMMAND_STEPS + 1, STATE_DIM),
        "commands": (commands, 2),
        "task_id": (commands,),
        "episode_id": (commands,),
        "command_index": (commands,),
        "reset_seed": (commands,),
        "task_type": (commands,),
        "selected_connection_weight": (commands,),
        "selected_score": (commands,),
        "selected_command_distance": (commands,),
        "selected_kind": (commands,),
        "actual_connection_within_sigma": (commands,),
    }
    if set(shard) != set(expected):
        raise ValueError(f"invalid R29 shard keys: {path}")
    for name, shape in expected.items():
        if shard[name].shape != shape:
            raise ValueError(
                f"invalid {path}: {name} has {shard[name].shape}, expected {shape}"
            )
    numeric = [name for name in expected if name != "selected_kind"]
    if any(not np.isfinite(shard[name]).all() for name in numeric):
        raise ValueError(f"non-finite value in {path}")
    if not np.all(shard["task_id"] == int(task_id)):
        raise ValueError(f"task id changed in {path}")
    index = np.arange(commands)
    if not np.array_equal(
        shard["episode_id"], index // COMMANDS_PER_EPISODE
    ):
        raise ValueError(f"physical episode ids changed in {path}")
    if not np.array_equal(
        shard["command_index"], index % COMMANDS_PER_EPISODE
    ):
        raise ValueError(f"command indices changed in {path}")
    if not np.all(shard["task_type"] == 0):
        raise ValueError(f"R29 shard is not acquisition-only: {path}")
    for row in range(1, commands):
        if row % COMMANDS_PER_EPISODE == 0:
            continue
        if not np.array_equal(shard["states"][row, 0], shard["states"][row - 1, -1]):
            raise ValueError(f"command chain is broken in {path}")
    observed = (
        np.linalg.norm(shard["states"][:, -1, :2] - shard["commands"], axis=1)
        <= float(r22.SIGMA)
    ).astype(np.uint8)
    if not np.array_equal(observed, shard["actual_connection_within_sigma"]):
        raise ValueError(f"actual connection labels changed in {path}")
    return shard


def collect_task(planner, executor, env, task_id, seed, commands):
    rng = np.random.default_rng(seed)
    planner.rng = rng
    planner.seed = int(seed)
    planner.goal_key = None
    fields = task_fields(planner, float(r22.SIGMA))
    online = []
    completed_trajectories = []
    current_trajectory = None
    state = None
    rows = []
    command_values = []
    connection = []
    scores = []
    distances = []
    kinds = []
    reset_seed = int(seed)

    for row in range(commands):
        command_index = row % COMMANDS_PER_EPISODE
        if command_index == 0:
            observation, _ = env.reset(seed=reset_seed)
            state = full_state(observation)
            current_trajectory = [state.copy()]
            fields.add(state[:2], 0.0)
        if state is None or current_trajectory is None:
            raise RuntimeError("R29 physical state was not initialized")
        fields.rebuild()
        command, diagnostic = planner.choose(
            state,
            command_index,
            fields,
            online,
            completed_trajectories + [current_trajectory],
        )
        trajectory, rewards = execute_command(
            env, executor, state, command, hidden_goal=None
        )
        if rewards.any():
            raise RuntimeError("reward appeared in reward-free R29 collection")
        for next_state in trajectory[1:]:
            fields.add(next_state[:2], 0.0)
            online.append(next_state.copy())
            current_trajectory.append(next_state.copy())
        rows.append(trajectory.astype(np.float32))
        command_values.append(np.asarray(command, np.float32))
        connection.append(float(diagnostic["selected_connection_weight"]))
        scores.append(float(diagnostic["selected_score"]))
        distances.append(float(np.linalg.norm(command - state[:2])))
        kinds.append(str(diagnostic["selected_kind"]))
        state = trajectory[-1]
        if command_index + 1 == COMMANDS_PER_EPISODE:
            completed_trajectories.append(np.asarray(current_trajectory, np.float32))
            current_trajectory = None

    states = np.asarray(rows, np.float32)
    command_values = np.asarray(command_values, np.float32)
    actual = (
        np.linalg.norm(states[:, -1, :2] - command_values, axis=1)
        <= float(r22.SIGMA)
    ).astype(np.uint8)
    index = np.arange(commands, dtype=np.int16)
    return {
        "states": states,
        "commands": command_values,
        "task_id": np.full(commands, task_id, np.int16),
        "episode_id": index // COMMANDS_PER_EPISODE,
        "command_index": index % COMMANDS_PER_EPISODE,
        "reset_seed": np.full(commands, reset_seed, np.int64),
        "task_type": np.zeros(commands, np.int8),
        "selected_connection_weight": np.asarray(connection, np.float32),
        "selected_score": np.asarray(scores, np.float32),
        "selected_command_distance": np.asarray(distances, np.float32),
        "selected_kind": np.asarray(kinds, dtype="<U64"),
        "actual_connection_within_sigma": actual,
    }


def command_collect(args):
    protocol = json.loads(PROTOCOL.read_text())
    reference = validate_protocol(protocol)
    tasks, commands = dimensions(args.profile, protocol)
    output = Path(args.output).resolve()
    shards = output / "shards"
    completed = 0
    while (shards / f"task_{completed:04d}.npz").is_file():
        validate_shard(
            shards / f"task_{completed:04d}.npz", completed, commands
        )
        completed += 1
    if completed > tasks:
        raise RuntimeError("R29 output contains too many tasks")
    if shards.is_dir() and len(list(shards.glob("task_*.npz"))) != completed:
        raise RuntimeError("R29 shards are not a contiguous prefix")

    planner, executor, env = r28._load_runtime(args)
    artifact_hashes = dict(planner.artifact_hashes)
    started = time.time()
    try:
        seed_base = int(protocol["seed_base"])
        for task_id in range(completed, tasks):
            shard = collect_task(
                planner,
                executor,
                env,
                task_id,
                seed_base + task_id,
                commands,
            )
            path = shards / f"task_{task_id:04d}.npz"
            atomic_savez(path, **shard)
            validate_shard(path, task_id, commands)
            long_mask = shard["selected_command_distance"] >= 6.0
            print(
                f"r29_alignment task={task_id + 1}/{tasks} "
                f"rows={(task_id + 1) * commands} "
                f"long_commands={int(long_mask.sum())} "
                f"connection_rate={shard['actual_connection_within_sigma'].mean():.3f}",
                flush=True,
            )
    finally:
        env.close()

    all_shards = [
        validate_shard(shards / f"task_{task_id:04d}.npz", task_id, commands)
        for task_id in range(tasks)
    ]
    distance = np.concatenate([item["selected_command_distance"] for item in all_shards])
    predicted = np.concatenate([item["selected_connection_weight"] for item in all_shards])
    observed = np.concatenate([item["actual_connection_within_sigma"] for item in all_shards])
    long_mask = distance >= 6.0
    summary = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "method_reference": str(reference),
        "method_reference_sha256": sha256(reference),
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "declared_deviations": protocol["declared_deviations"],
        "oracle_inputs": [],
        "profile": args.profile,
        "preparation_tasks": tasks,
        "commands_per_preparation_task": commands,
        "rows": int(tasks * commands),
        "fixed_command_duration": COMMAND_STEPS,
        "task_memory_reset_per_preparation_task": True,
        "task_memory_persisted_across_physical_resets": True,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "landing_sigma_m": float(r22.SIGMA),
        "connection_rate": float(observed.mean()),
        "predicted_connection_mean": float(predicted.mean()),
        "long_command_definition_m_diagnostic_only": 6.0,
        "long_commands": int(long_mask.sum()),
        "long_command_fraction": float(long_mask.mean()),
        "long_connection_rate": (
            float(observed[long_mask].mean()) if long_mask.any() else None
        ),
        "long_predicted_connection_mean": (
            float(predicted[long_mask].mean()) if long_mask.any() else None
        ),
        "planner_artifacts": artifact_hashes,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    value.add_argument("--profile", choices=("smoke", "full"), required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r22.graph_path(args.root)
    command_collect(args)


if __name__ == "__main__":
    main()
