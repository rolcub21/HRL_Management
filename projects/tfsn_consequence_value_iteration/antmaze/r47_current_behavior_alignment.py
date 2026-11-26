"""Collect reward-free alignment data from the current R39/R46 behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r46_online_consequence_correction as r46,
)
from scripts.antmaze.current_tfsn.collect import execute_command
from scripts.antmaze.current_tfsn.constants import (
    COMMANDS_PER_EPISODE,
    COMMAND_STEPS,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.executor import full_state
from scripts.antmaze.current_tfsn.main import dump


r31 = r46.r31
HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r47_current_behavior_alignment_protocol.json"
METHOD_REFERENCE = HERE / "R47_CURRENT_BEHAVIOR_ALIGNMENT_REFERENCE.md"
METHOD_REVISION = "antmaze_r39_r46_current_behavior_alignment_r47"
CLAIM_LEVEL = "reward_free_current_behavior_alignment_data_only"
RESULT_DIRECTORY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "current_behavior_alignment_r47"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r46.METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "oracle_inputs": [],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "fixed_command_duration": COMMAND_STEPS,
        "commands_per_physical_episode": COMMANDS_PER_EPISODE,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R47 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R47 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dimensions(profile):
    if profile == "full":
        return (
            int(PROTOCOL_VALUE["preparation_tasks"]),
            int(PROTOCOL_VALUE["commands_per_preparation_task"]),
        )
    return 1, COMMANDS_PER_EPISODE + 2


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
        "selected_connection_weight": (commands,),
        "selected_offline_connection_weight": (commands,),
        "selected_score": (commands,),
        "selected_command_distance": (commands,),
        "selected_kind": (commands,),
        "actual_landing_kernel": (commands,),
        "continuation_mobility_target": (commands,),
        "continuation_mobility_valid": (commands,),
    }
    if set(shard) != set(expected):
        raise ValueError(f"invalid R47 shard keys: {path}")
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
    episode = index // COMMANDS_PER_EPISODE
    command_index = index % COMMANDS_PER_EPISODE
    if not np.array_equal(shard["episode_id"], episode):
        raise ValueError(f"physical episode ids changed in {path}")
    if not np.array_equal(shard["command_index"], command_index):
        raise ValueError(f"command indices changed in {path}")
    for row in range(1, commands):
        if command_index[row] == 0:
            continue
        if not np.array_equal(shard["states"][row, 0], shard["states"][row - 1, -1]):
            raise ValueError(f"command chain is broken in {path}")
    valid = (
        (command_index + 1 < COMMANDS_PER_EPISODE)
        & (np.arange(commands) + 1 < commands)
    )
    if not np.array_equal(shard["continuation_mobility_valid"], valid.astype(np.uint8)):
        raise ValueError(f"continuation validity changed in {path}")
    expected_landing = np.exp(
        -np.square(shard["states"][:, -1, :2] - shard["commands"]).sum(axis=1)
        / (2.0 * float(r31.SIGMA) ** 2)
    )
    if not np.allclose(expected_landing, shard["actual_landing_kernel"], atol=1e-6):
        raise ValueError(f"landing outcomes changed in {path}")
    if valid.any():
        next_movement = np.linalg.norm(
            shard["states"][1:, -1, :2] - shard["states"][1:, 0, :2], axis=1
        )
        expected_mobility = 1.0 - np.exp(
            -np.square(next_movement) / (2.0 * float(r31.SIGMA) ** 2)
        )
        rows = np.flatnonzero(valid)
        if not np.allclose(
            shard["continuation_mobility_target"][rows],
            expected_mobility[rows],
            atol=1e-6,
        ):
            raise ValueError(f"continuation outcomes changed in {path}")
    return shard


def _continuation_targets(states, command_index):
    count = len(states)
    target = np.zeros(count, np.float32)
    valid = (
        (command_index + 1 < COMMANDS_PER_EPISODE)
        & (np.arange(count) + 1 < count)
    )
    rows = np.flatnonzero(valid)
    if len(rows):
        movement = np.linalg.norm(
            states[rows + 1, -1, :2] - states[rows + 1, 0, :2], axis=1
        )
        target[rows] = 1.0 - np.exp(
            -np.square(movement) / (2.0 * float(r31.SIGMA) ** 2)
        )
    return target, valid.astype(np.uint8)


def collect_task(planner, executor, env, task_id, seed, commands):
    planner.rng = np.random.default_rng(seed)
    planner.seed = int(seed)
    planner.goal_key = None
    planner.reset_online_correction_task()
    fields = r46.r39.r37.exact_task_fields(planner, float(r31.SIGMA))
    online = []
    completed_trajectories = []
    current_trajectory = None
    state = None
    trajectories = []
    command_values = []
    connection = []
    offline_connection = []
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
            raise RuntimeError("reward appeared in reward-free R47 collection")
        for next_state in trajectory[1:]:
            fields.add(next_state[:2], 0.0)
            online.append(next_state.copy())
            current_trajectory.append(next_state.copy())
        trajectories.append(trajectory.astype(np.float32))
        command_values.append(np.asarray(command, np.float32))
        connection.append(float(diagnostic["selected_connection_weight"]))
        offline_connection.append(float(diagnostic.get(
            "selected_offline_connection_weight",
            diagnostic["selected_connection_weight"],
        )))
        scores.append(float(diagnostic["selected_score"]))
        distances.append(float(np.linalg.norm(command - state[:2])))
        kinds.append(str(diagnostic["selected_kind"]))
        state = trajectory[-1]
        if command_index + 1 == COMMANDS_PER_EPISODE:
            completed_trajectories.append(np.asarray(current_trajectory, np.float32))
            current_trajectory = None

    states = np.asarray(trajectories, np.float32)
    command_values = np.asarray(command_values, np.float32)
    index = np.arange(commands, dtype=np.int16)
    command_index = index % COMMANDS_PER_EPISODE
    landing = np.exp(
        -np.square(states[:, -1, :2] - command_values).sum(axis=1)
        / (2.0 * float(r31.SIGMA) ** 2)
    ).astype(np.float32)
    continuation, continuation_valid = _continuation_targets(states, command_index)
    return {
        "states": states,
        "commands": command_values,
        "task_id": np.full(commands, task_id, np.int16),
        "episode_id": index // COMMANDS_PER_EPISODE,
        "command_index": command_index,
        "reset_seed": np.full(commands, reset_seed, np.int64),
        "selected_connection_weight": np.asarray(connection, np.float32),
        "selected_offline_connection_weight": np.asarray(offline_connection, np.float32),
        "selected_score": np.asarray(scores, np.float32),
        "selected_command_distance": np.asarray(distances, np.float32),
        "selected_kind": np.asarray(kinds, dtype="<U64"),
        "actual_landing_kernel": landing,
        "continuation_mobility_target": continuation,
        "continuation_mobility_valid": continuation_valid,
    }


def command_collect(args):
    tasks, commands = dimensions(args.profile)
    output = Path(args.output).resolve()
    shards = output / "shards"
    completed = 0
    while (shards / f"task_{completed:04d}.npz").is_file():
        validate_shard(shards / f"task_{completed:04d}.npz", completed, commands)
        completed += 1
    if completed > tasks:
        raise RuntimeError("R47 output contains too many tasks")
    if shards.is_dir() and len(list(shards.glob("task_*.npz"))) != completed:
        raise RuntimeError("R47 shards are not a contiguous prefix")

    planner, executor, env = r46._load_runtime(args)
    artifact_hashes = dict(planner.artifact_hashes)
    started = time.time()
    try:
        for task_id in range(completed, tasks):
            shard = collect_task(
                planner,
                executor,
                env,
                task_id,
                int(PROTOCOL_VALUE["seed_base"]) + task_id,
                commands,
            )
            path = shards / f"task_{task_id:04d}.npz"
            atomic_savez(path, **shard)
            validate_shard(path, task_id, commands)
            valid = shard["continuation_mobility_valid"].astype(bool)
            print(
                f"r47_alignment task={task_id + 1}/{tasks} "
                f"rows={(task_id + 1) * commands} "
                f"landing={shard['actual_landing_kernel'].mean():.3f} "
                f"continuation={shard['continuation_mobility_target'][valid].mean():.3f}",
                flush=True,
            )
    finally:
        env.close()

    all_shards = [
        validate_shard(shards / f"task_{task_id:04d}.npz", task_id, commands)
        for task_id in range(tasks)
    ]
    landing = np.concatenate([row["actual_landing_kernel"] for row in all_shards])
    predicted = np.concatenate([row["selected_connection_weight"] for row in all_shards])
    offline = np.concatenate([
        row["selected_offline_connection_weight"] for row in all_shards
    ])
    mobility = np.concatenate([
        row["continuation_mobility_target"] for row in all_shards
    ])
    mobility_valid = np.concatenate([
        row["continuation_mobility_valid"] for row in all_shards
    ]).astype(bool)
    summary = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r46.METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if args.profile == "full" else "smoke_only",
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "profile": args.profile,
        "preparation_tasks": tasks,
        "commands_per_preparation_task": commands,
        "rows": int(tasks * commands),
        "fixed_command_duration": COMMAND_STEPS,
        "physical_episode_commands": COMMANDS_PER_EPISODE,
        "task_memory_reset_per_preparation_task": True,
        "task_memory_persisted_across_physical_resets": True,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "landing_sigma_m": float(r31.SIGMA),
        "actual_landing_kernel_mean": float(landing.mean()),
        "selected_corrected_connection_mean": float(predicted.mean()),
        "selected_offline_connection_mean": float(offline.mean()),
        "continuation_targets": int(mobility_valid.sum()),
        "continuation_mobility_mean": float(mobility[mobility_valid].mean()),
        "planner_artifacts": artifact_hashes,
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def command_audit(_args):
    print(json.dumps({
        "method_revision": METHOD_REVISION,
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": sha256(METHOD_REFERENCE),
        "base_method_revision": r46.METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "preparation_tasks": PROTOCOL_VALUE["preparation_tasks"],
        "commands_per_preparation_task": PROTOCOL_VALUE[
            "commands_per_preparation_task"
        ],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=647)
    value.add_argument("--profile", choices=("smoke", "full"), default="full")
    value.add_argument("--output", type=Path, default=RESULT_DIRECTORY)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("collect").set_defaults(func=command_collect)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r31.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
