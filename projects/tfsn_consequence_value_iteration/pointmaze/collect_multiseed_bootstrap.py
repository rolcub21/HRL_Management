#!/usr/bin/env python3
"""Collect the search-memory-independent PointMaze bootstrap for one executor seed."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from projects.tfsn_current_definition.pointmaze.core import ExperienceCommandPool

from .core import HERE, REPOSITORY, atomic_json, atomic_npz, load_json, sha256


PROTOCOL_PATH = HERE / "multiseed_bootstrap_protocol.json"
DEFAULT_RESULTS = (
    HERE.parent
    / "results"
    / "pointmaze"
    / "independent_multiseed_bootstrap_r1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_protocol() -> dict:
    value = load_json(PROTOCOL_PATH)
    reference = REPOSITORY / value["method_reference"]
    if reference.resolve() != (HERE / "METHOD_REFERENCE.md").resolve():
        raise ValueError("multiseed bootstrap method reference changed")
    if value["declared_deviations"]:
        raise ValueError("multiseed bootstrap declares a method deviation")
    if value["oracle_inputs"]:
        raise ValueError("reward-free bootstrap cannot use oracle inputs")
    collection = value["collection"]
    if int(collection["fixed_command_duration"]) != 100:
        raise ValueError("shared fixed command duration changed")
    for preparation_seed, seed_contract in value["preparation_seeds"].items():
        expected = 560000 + 100000 * int(preparation_seed)
        if int(seed_contract["reset_seed_base"]) != expected:
            raise ValueError("reset seed-base rule changed")
    return value


def validate_actor(seed_contract: dict) -> tuple[Path, Path]:
    checkpoint = Path(seed_contract["executor_checkpoint"]).resolve()
    manifest = Path(seed_contract["executor_manifest"]).resolve()
    for path, expected in (
        (checkpoint, seed_contract["executor_checkpoint_sha256"]),
        (manifest, seed_contract["executor_manifest_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen executor artifact changed: {path}")
    return checkpoint, manifest


def validate_reset_bootstrap(path: Path, chains: int, seed_base: int) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        value = {name: stored[name] for name in stored.files}
    expected_seeds = np.arange(seed_base, seed_base + chains, dtype=np.int64)
    if not np.array_equal(value.get("seed"), expected_seeds):
        raise ValueError("reset bootstrap seeds changed")
    observations = value.get("reset_observation")
    if observations is None or observations.shape != (chains, 4):
        raise ValueError("reset bootstrap observations have the wrong shape")
    if not np.isfinite(observations).all():
        raise ValueError("reset bootstrap contains non-finite observations")
    return value


def validate_shard(path: Path, chain: int, rows: int, tau: int) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        value = {name: stored[name] for name in stored.files}
    required = {
        "reset_observation": (4,),
        "start_observation": (rows, 4),
        "command_xy": (rows, 2),
        "endpoint_observation": (rows, 4),
        "primitive_observation": (rows, tau, 4),
        "primitive_action": (rows, tau, 2),
        "duration": (rows,),
        "command_pool_size_before": (rows,),
        "sampled_pool_index": (rows,),
        "chain_id": (rows,),
    }
    for name, shape in required.items():
        array = value.get(name)
        if array is None or array.shape != shape:
            actual = None if array is None else array.shape
            raise ValueError(f"invalid {path}: {name} {actual}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite {name} in {path}")
    if not np.all(value["duration"] == tau):
        raise ValueError(f"command duration changed in {path}")
    if not np.all(value["chain_id"] == chain):
        raise ValueError(f"chain identifier changed in {path}")
    if not np.all(value["sampled_pool_index"] < value["command_pool_size_before"]):
        raise ValueError(f"command left the exact experience pool in {path}")
    return value


def seed_pool(pool: ExperienceCommandPool, bootstrap: dict) -> None:
    for observation in bootstrap["reset_observation"]:
        pool.add(observation[:2])


def replay_shard(pool: ExperienceCommandPool, shard: dict) -> None:
    for row, endpoint in enumerate(shard["endpoint_observation"]):
        expected_size = int(shard["command_pool_size_before"][row])
        if len(pool) != expected_size:
            raise ValueError("stored command-pool size differs from exact replay")
        index = int(shard["sampled_pool_index"][row])
        if not np.array_equal(pool.array()[index], shard["command_xy"][row]):
            raise ValueError("stored command differs from exact pool selection")
        pool.add(endpoint[:2])


def collect_chain(
    *,
    chain: int,
    seed_base: int,
    commands_per_chain: int,
    tau: int,
    expected_reset: np.ndarray,
    pool: ExperienceCommandPool,
    actor,
    env,
    observation_vector,
    achieved_xy,
) -> dict:
    rng = np.random.default_rng(seed_base + chain)
    env.reset(seed=seed_base + chain)
    reset = observation_vector(env.raw_obs).astype(np.float32)
    if not np.array_equal(reset, expected_reset.astype(np.float32)):
        raise RuntimeError("reset bootstrap replay changed")

    starts, commands, endpoints = [], [], []
    primitive_observations, primitive_actions = [], []
    pool_sizes, sampled_indices = [], []
    for command_index in range(commands_per_chain):
        start = observation_vector(env.raw_obs).copy()
        pool_size = len(pool)
        sampled_index, command = pool.sample_uniform(rng)
        observations, actions = [], []
        for primitive_step in range(tau):
            action = actor.act(env.raw_obs, command)
            time_step = env.step(action)
            actions.append(np.asarray(action, dtype=np.float32).copy())
            observations.append(observation_vector(env.raw_obs).copy())
            if time_step.last():
                raise RuntimeError(
                    "environment ended before the fixed command duration: "
                    f"chain={chain}, command={command_index}, "
                    f"step={primitive_step + 1}/{tau}"
                )
        endpoint = observation_vector(env.raw_obs).copy()
        starts.append(start)
        commands.append(command)
        endpoints.append(endpoint)
        primitive_observations.append(observations)
        primitive_actions.append(actions)
        pool_sizes.append(pool_size)
        sampled_indices.append(sampled_index)
        pool.add(achieved_xy(env.raw_obs))

    return {
        "reset_observation": reset,
        "start_observation": np.asarray(starts, dtype=np.float32),
        "command_xy": np.asarray(commands, dtype=np.float32),
        "endpoint_observation": np.asarray(endpoints, dtype=np.float32),
        "primitive_observation": np.asarray(primitive_observations, dtype=np.float32),
        "primitive_action": np.asarray(primitive_actions, dtype=np.float32),
        "duration": np.full(commands_per_chain, tau, dtype=np.int16),
        "command_pool_size_before": np.asarray(pool_sizes, dtype=np.int32),
        "sampled_pool_index": np.asarray(sampled_indices, dtype=np.int32),
        "chain_id": np.full(commands_per_chain, chain, dtype=np.int16),
    }


def main() -> None:
    args = parse_args()
    contract = load_protocol()
    seed_contract = contract["preparation_seeds"][str(args.preparation_seed)]
    checkpoint, manifest = validate_actor(seed_contract)
    collection = contract["collection"]
    if args.profile == "full":
        chains = int(collection["chains"])
        commands_per_chain = int(collection["commands_per_chain"])
    else:
        chains = 2
        commands_per_chain = 3
    tau = int(collection["fixed_command_duration"])
    seed_base = int(seed_contract["reset_seed_base"])
    output = (
        args.output
        or DEFAULT_RESULTS / f"seed{args.preparation_seed}" / args.profile
    ).resolve()
    shards = output / "shards"

    from scripts.pointmaze.gc_executor.runtime import (  # noqa: PLC0415
        GCSACExecutorRuntime,
        achieved_xy,
        observation_vector,
    )
    from scripts.pointmaze.rnd_fb.env import PointMazeTimeStepEnv  # noqa: PLC0415

    actor = GCSACExecutorRuntime(checkpoint, args.device)
    env = PointMazeTimeStepEnv(
        contract["environment"]["env_id"],
        max_episode_steps=commands_per_chain * tau + 1,
    )
    started = time.time()
    try:
        bootstrap_path = output / "reset_bootstrap.npz"
        if not bootstrap_path.is_file():
            if shards.is_dir() and any(shards.glob("chain_*.npz")):
                raise RuntimeError("shards exist without the immutable reset bootstrap")
            reset_observations = []
            for chain in range(chains):
                env.reset(seed=seed_base + chain)
                reset_observations.append(observation_vector(env.raw_obs).copy())
            atomic_npz(
                bootstrap_path,
                seed=np.arange(seed_base, seed_base + chains, dtype=np.int64),
                reset_observation=np.asarray(reset_observations, dtype=np.float32),
            )
        bootstrap = validate_reset_bootstrap(bootstrap_path, chains, seed_base)
        pool = ExperienceCommandPool()
        seed_pool(pool, bootstrap)

        completed = 0
        while completed < chains:
            path = shards / f"chain_{completed:04d}.npz"
            if not path.is_file():
                break
            shard = validate_shard(path, completed, commands_per_chain, tau)
            if not np.array_equal(
                shard["reset_observation"], bootstrap["reset_observation"][completed]
            ):
                raise ValueError("stored chain reset differs from bootstrap")
            replay_shard(pool, shard)
            completed += 1
        existing = sorted(shards.glob("chain_*.npz")) if shards.is_dir() else []
        if len(existing) != completed:
            raise RuntimeError("bootstrap shards are not a contiguous resume prefix")

        for chain in range(completed, chains):
            value = collect_chain(
                chain=chain,
                seed_base=seed_base,
                commands_per_chain=commands_per_chain,
                tau=tau,
                expected_reset=bootstrap["reset_observation"][chain],
                pool=pool,
                actor=actor,
                env=env,
                observation_vector=observation_vector,
                achieved_xy=achieved_xy,
            )
            path = shards / f"chain_{chain:04d}.npz"
            atomic_npz(path, **value)
            validate_shard(path, chain, commands_per_chain, tau)
            print(
                f"point_bootstrap seed={args.preparation_seed} "
                f"chain={chain + 1}/{chains} "
                f"rows={(chain + 1) * commands_per_chain} pool={len(pool)}",
                flush=True,
            )
    finally:
        env.close()

    if sha256(checkpoint) != seed_contract["executor_checkpoint_sha256"]:
        raise RuntimeError("frozen executor changed during bootstrap collection")
    summary = {
        "schema_version": 1,
        "experiment_id": contract["experiment_id"],
        "method_reference": str((HERE / "METHOD_REFERENCE.md").resolve()),
        "method_revision": contract["method_revision"],
        "claim_level": contract["claim_level"] if args.profile == "full" else "smoke_only",
        "declared_deviations": contract["declared_deviations"],
        "oracle_inputs": contract["oracle_inputs"],
        "preparation_seed": args.preparation_seed,
        "profile": args.profile,
        "environment": contract["environment"],
        "executor_checkpoint": str(checkpoint),
        "executor_checkpoint_sha256": sha256(checkpoint),
        "executor_manifest": str(manifest),
        "executor_manifest_sha256": sha256(manifest),
        "behavior_distribution": collection["behavior_distribution"],
        "behavior_support_semantics": "exact_float32_set_without_distance_tolerance",
        "reset_seed_base": seed_base,
        "reset_seed_base_rule": collection["reset_seed_base_rule"],
        "reset_bootstrap_sha256": sha256(bootstrap_path),
        "chains": chains,
        "options_per_chain": commands_per_chain,
        "rows": chains * commands_per_chain,
        "primitive_transitions": chains * commands_per_chain * tau,
        "duration_values": [tau],
        "command_pool_final_size": len(pool),
        "search_memory_used": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "hidden_goals_used": False,
        "downstream_status": contract["downstream_status"],
        "gymnasium_robotics_version": package_version("gymnasium-robotics"),
        "gymnasium_version": package_version("gymnasium"),
        "wall_clock_seconds_this_invocation": time.time() - started,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
