#!/usr/bin/env python3
"""Continue the frozen OGBench CRL executor on chained command data.

The learner and loss are the vendored OGBench CRL implementation. Only the
data stream changes: commands last exactly 100 primitive steps, ten commands
share one physical episode, and failed commands never trigger an early reset.
Training targets and reset cells come exclusively from the reward-free OGBench
training split. The native evaluation task panel is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .audit_executor import (
    audited_datasets,
    create_agent,
    fixed_contract_env_kwargs,
    sha256,
    torso_measurements,
    trajectory_ranges,
    verify_audited_dataset,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)
    temporary.replace(path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def empirical_command_pairs(
    terminals: np.ndarray, command_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    sources = []
    targets = []
    for start, end in trajectory_ranges(terminals):
        last_source = end - command_steps
        if last_source < start:
            continue
        source = np.arange(start, last_source + 1, dtype=np.int64)
        sources.append(source)
        targets.append(source + command_steps)
    if not sources:
        raise RuntimeError("training data has no complete command-duration pairs")
    return np.concatenate(sources), np.concatenate(targets)


class EmpiricalCommandSampler:
    """Sample exact data-derived reach offsets, then snap to recorded support."""

    def __init__(self, observations: np.ndarray, terminals: np.ndarray, command_steps: int):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.ranges = trajectory_ranges(terminals)
        self.reset_indices = np.asarray([start for start, _end in self.ranges], np.int64)
        self.sources, self.targets = empirical_command_pairs(terminals, command_steps)
        self.offsets = (
            self.observations[self.targets, :2]
            - self.observations[self.sources, :2]
        ).astype(np.float32)
        self.tree = cKDTree(self.observations[:, :2])

    def sample_target(
        self, live_xy: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        pair = int(rng.integers(len(self.offsets)))
        proposed = np.asarray(live_xy, np.float32) + self.offsets[pair]
        snap_distance, support_index = self.tree.query(proposed, k=1)
        target = self.observations[int(support_index), :2].copy()
        return target, {
            "pair_index": pair,
            "source_index": int(self.sources[pair]),
            "future_index": int(self.targets[pair]),
            "empirical_offset_distance_m": float(np.linalg.norm(self.offsets[pair])),
            "live_target_distance_m": float(
                np.linalg.norm(target - np.asarray(live_xy, np.float32))
            ),
            "support_snap_distance_m": float(snap_distance),
            "support_index": int(support_index),
        }

    def sample_reset_xy(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        index = int(rng.choice(self.reset_indices))
        return self.observations[index, :2].copy(), index


class OnlineCommandReplay:
    """Preallocated replay with declared CRL future-goal boundaries."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        goal_boundary: str = "command",
    ):
        if goal_boundary not in {"command", "physical_episode"}:
            raise ValueError("online replay has an unsupported goal boundary")
        self.goal_boundary = str(goal_boundary)
        self.observations = np.zeros((capacity, observation_dim), np.float32)
        self.actions = np.zeros((capacity, action_dim), np.float32)
        self.ends = np.zeros(capacity, np.int64)
        self.valid_rows = np.zeros(capacity, np.int64)
        self.size = 0
        self.valid_size = 0
        self.commands = 0
        self.current_episode_start = None

    def start_episode(self) -> None:
        self.current_episode_start = self.size

    def add_command(self, states: np.ndarray, actions: np.ndarray) -> None:
        states = np.asarray(states, np.float32)
        actions = np.asarray(actions, np.float32)
        if len(states) != len(actions) + 1 or len(actions) == 0:
            raise RuntimeError("online replay requires a complete nonempty command")
        start = self.size
        end = start + len(states) - 1
        stop = end + 1
        if stop > len(self.observations):
            raise RuntimeError("online replay capacity exhausted")
        self.observations[start:stop] = states
        self.actions[start:end] = actions
        self.actions[end] = 0.0
        if self.goal_boundary == "physical_episode":
            if self.current_episode_start is None:
                raise RuntimeError("physical-episode replay was not started")
            self.ends[self.current_episode_start : stop] = end
        else:
            self.ends[start:stop] = end
        self.valid_rows[self.valid_size : self.valid_size + len(actions)] = np.arange(
            start, end, dtype=np.int64
        )
        self.valid_size += len(actions)
        self.size = stop
        self.commands += 1

    def sample(self, batch_size: int, config, rng: np.random.Generator) -> dict:
        if self.valid_size == 0:
            raise RuntimeError("online replay is empty")
        if not (
            float(config["value_p_curgoal"]) == 0.0
            and float(config["value_p_trajgoal"]) == 1.0
            and float(config["value_p_randomgoal"]) == 0.0
            and bool(config["value_geom_sample"])
            and float(config["actor_p_curgoal"]) == 0.0
            and float(config["actor_p_trajgoal"]) == 1.0
            and float(config["actor_p_randomgoal"]) == 0.0
            and not bool(config["actor_geom_sample"])
        ):
            raise RuntimeError("vendored CRL goal-relabeling contract changed")
        idxs = self.valid_rows[
            rng.integers(self.valid_size, size=int(batch_size))
        ]
        final = self.ends[idxs]
        offsets = rng.geometric(
            p=1.0 - float(config["discount"]), size=int(batch_size)
        )
        value_goal_idxs = np.minimum(idxs + offsets, final)
        distances = rng.random(int(batch_size))
        actor_goal_idxs = np.round(
            np.minimum(idxs + 1, final) * distances + final * (1.0 - distances)
        ).astype(np.int64)
        return {
            "observations": self.observations[idxs],
            "actions": self.actions[idxs],
            "next_observations": self.observations[idxs + 1],
            "value_goals": self.observations[value_goal_idxs],
            "actor_goals": self.observations[actor_goal_idxs],
            "masks": np.ones(int(batch_size), np.float32),
            "rewards": np.zeros(int(batch_size), np.float32),
        }

    def save(self, path: Path) -> None:
        atomic_npz(
            path,
            observations=self.observations[: self.size],
            actions=self.actions[: self.size],
            ends=self.ends[: self.size],
            valid_rows=self.valid_rows[: self.valid_size],
            size=np.asarray(self.size, np.int64),
            valid_size=np.asarray(self.valid_size, np.int64),
            commands=np.asarray(self.commands, np.int64),
            goal_boundary=np.asarray(self.goal_boundary),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        goal_boundary: str = "command",
    ) -> "OnlineCommandReplay":
        replay = cls(
            capacity, observation_dim, action_dim,
            goal_boundary=goal_boundary,
        )
        with np.load(path, allow_pickle=False) as stored:
            saved_boundary = (
                str(stored["goal_boundary"].item())
                if "goal_boundary" in stored.files else "command"
            )
            if saved_boundary != goal_boundary:
                raise RuntimeError("online replay goal boundary changed")
            size = int(stored["size"])
            valid_size = int(stored["valid_size"])
            replay.observations[:size] = stored["observations"]
            replay.actions[:size] = stored["actions"]
            replay.ends[:size] = stored["ends"]
            replay.valid_rows[:valid_size] = stored["valid_rows"]
            replay.size = size
            replay.valid_size = valid_size
            replay.commands = int(stored["commands"])
        return replay


def mix_update_batches(offline: dict, online: dict) -> dict:
    """Join official offline and chained-online CRL batches exactly once."""

    required = {
        "observations", "actions", "next_observations", "value_goals",
        "actor_goals", "masks", "rewards",
    }
    if not required.issubset(offline) or not required.issubset(online):
        raise RuntimeError("CRL update batch is missing a required field")
    return {
        key: np.concatenate((np.asarray(offline[key]), np.asarray(online[key])))
        for key in sorted(required)
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def quantiles(values: list[float]) -> dict:
    points = (0.0, 0.1, 0.5, 0.9, 1.0)
    result = np.quantile(values, points)
    return {f"q{int(point * 100):02d}": float(value) for point, value in zip(points, result)}


def save_command_diagnostics(path: Path, commands: list[dict]) -> None:
    if not commands:
        raise RuntimeError("fine-tuning produced no command diagnostics")
    keys = sorted(commands[0])
    if any(sorted(row) != keys for row in commands):
        raise RuntimeError("fine-tuning command diagnostics changed schema")
    atomic_npz(
        path,
        **{key: np.asarray([row[key] for row in commands]) for key in keys},
    )


def load_command_diagnostics(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as stored:
        keys = sorted(stored.files)
        count = len(stored[keys[0]])
        return [
            {
                key: stored[key][index].item()
                for key in keys
            }
            for index in range(count)
        ]


def run(args: argparse.Namespace) -> dict:
    import flax.serialization
    import jax
    import ogbench

    protocol = load_json(args.protocol)
    development = load_json(args.development_protocol)
    source_record = load_json(args.executor)
    datasets = audited_datasets(args.dataset_audit)
    verify_audited_dataset(args.train_dataset, datasets["train"], "training")
    if sha256(args.development_protocol) != protocol["parent_development_protocol_sha256"]:
        raise RuntimeError("parent development protocol changed")
    if source_record.get("checkpoint_sha256") != protocol["source_executor"]["checkpoint_sha256"]:
        raise RuntimeError("source executor differs from frozen fine-tuning protocol")
    if args.target_steps not in protocol["checkpoints"]:
        raise RuntimeError("target is not a frozen fine-tuning checkpoint")
    command_steps = int(protocol["continuation"]["command_duration_primitive_steps"])
    episode_steps = int(protocol["continuation"]["physical_episode_primitive_steps"])
    commands_per_episode = int(protocol["continuation"]["commands_per_physical_episode"])
    offline_fraction = float(
        protocol["continuation"].get(
            "offline_dataset_fraction_per_update_batch", 0.0
        )
    )
    online_boundary = str(
        protocol["continuation"].get(
            "online_goal_relabel_boundary", "command"
        )
    )
    if not 0.0 <= offline_fraction < 1.0:
        raise RuntimeError("offline update fraction must be in [0, 1)")
    if online_boundary not in {"command", "physical_episode"}:
        raise RuntimeError("online goal relabel boundary is unsupported")
    if episode_steps != command_steps * commands_per_episode:
        raise RuntimeError("fine-tuning command and physical horizons disagree")
    if args.target_steps % episode_steps:
        raise RuntimeError("checkpoint must lie on a physical episode boundary")

    output = args.output_root / f"steps_{args.target_steps:07d}"
    record_path = output / "executor.json"
    if record_path.is_file():
        existing = load_json(record_path)
        if (
            existing.get("passed")
            and Path(existing["checkpoint"]).is_file()
            and sha256(Path(existing["checkpoint"])) == existing["checkpoint_sha256"]
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return existing

    flags_path = Path(source_record["flags"])
    if sha256(flags_path) != source_record["flags_sha256"]:
        raise RuntimeError("source executor flags changed")
    flags = load_json(flags_path)
    agent = create_agent(flags, args.train_dataset)
    observation_dim = int(development["benchmark"]["observation_dimension"])
    action_dim = int(development["benchmark"]["action_dimension"])

    maximum_steps = max(int(value) for value in protocol["checkpoints"])
    capacity = maximum_steps + maximum_steps // command_steps + 2
    rng = np.random.default_rng(int(protocol["source_executor"]["training_seed"]))
    action_key = jax.random.PRNGKey(int(protocol["source_executor"]["training_seed"]))
    total_steps = 0
    total_updates = 0
    commands: list[dict] = []

    if args.resume_record is None:
        checkpoint = Path(source_record["checkpoint"])
        if sha256(checkpoint) != source_record["checkpoint_sha256"]:
            raise RuntimeError("source executor checkpoint changed")
        with checkpoint.open("rb") as handle:
            agent = flax.serialization.from_state_dict(agent, pickle.load(handle)["agent"])
        replay = OnlineCommandReplay(
            capacity, observation_dim, action_dim,
            goal_boundary=online_boundary,
        )
        input_checkpoint = checkpoint
        input_checkpoint_sha256 = source_record["checkpoint_sha256"]
    else:
        resume = load_json(args.resume_record)
        if not resume.get("passed"):
            raise RuntimeError("resume fine-tuning record did not pass")
        if int(resume["finetuning_environment_steps"]) >= args.target_steps:
            raise RuntimeError("resume record is not earlier than target")
        checkpoint = Path(resume["checkpoint"])
        replay_path = Path(resume["online_replay"])
        diagnostics_path = Path(resume["command_diagnostics"])
        if sha256(checkpoint) != resume["checkpoint_sha256"]:
            raise RuntimeError("resume checkpoint changed")
        if sha256(replay_path) != resume["online_replay_sha256"]:
            raise RuntimeError("resume replay changed")
        if sha256(diagnostics_path) != resume["command_diagnostics_sha256"]:
            raise RuntimeError("resume command diagnostics changed")
        with checkpoint.open("rb") as handle:
            agent = flax.serialization.from_state_dict(agent, pickle.load(handle)["agent"])
        replay = OnlineCommandReplay.load(
            replay_path, capacity, observation_dim, action_dim,
            goal_boundary=online_boundary,
        )
        total_steps = int(resume["finetuning_environment_steps"])
        total_updates = int(resume["gradient_updates"])
        commands = load_command_diagnostics(diagnostics_path)
        rng.bit_generator.state = resume["collector_rng_state"]
        action_key = np.asarray(resume["action_key"], dtype=np.uint32)
        input_checkpoint = checkpoint
        input_checkpoint_sha256 = resume["checkpoint_sha256"]
    if total_steps % episode_steps:
        raise RuntimeError("resume point is not a physical episode boundary")

    with np.load(args.train_dataset, allow_pickle=False) as raw:
        training_arrays = {
            key: np.asarray(raw[key]).copy() for key in raw.files
        }
    training_observations = np.asarray(
        training_arrays["observations"], np.float32
    )
    training_terminals = np.asarray(training_arrays["terminals"], bool)
    sampler = EmpiricalCommandSampler(
        training_observations, training_terminals, command_steps
    )
    if training_observations.shape[1] != observation_dim:
        raise RuntimeError("OGBench Ant observation dimension changed")
    offline_dataset = None
    if offline_fraction > 0.0:
        from utils.datasets import Dataset, GCDataset

        offline_dataset = GCDataset(
            Dataset.create(**training_arrays), agent.config
        )
    update_batch_size = int(protocol["continuation"]["batch_size"])
    offline_batch_size = int(round(update_batch_size * offline_fraction))
    online_batch_size = update_batch_size - offline_batch_size
    if online_batch_size < 1:
        raise RuntimeError("every mixed update needs online examples")

    action_fn = jax.jit(
        lambda current_agent, observation, goal, key: current_agent.sample_actions(
            observation, goal, seed=key, temperature=float(
                protocol["continuation"]["actor_sampling_temperature"]
            )
        )
    )
    env_kwargs = fixed_contract_env_kwargs(command_steps, commands_per_episode)
    env = ogbench.make_env_and_datasets(
        development["benchmark"]["dataset"], env_only=True, **env_kwargs
    )
    native = env.unwrapped if hasattr(env, "unwrapped") else env
    try:
        while total_steps < args.target_steps:
            reset_xy, reset_index = sampler.sample_reset_xy(rng)
            reset_seed = int(rng.integers(0, 2**31 - 1))
            reset_ij = tuple(int(value) for value in native.xy_to_ij(reset_xy))
            task_info = {
                "task_name": f"reward_free_training_reset_{reset_index}",
                "init_ij": reset_ij,
                "goal_ij": reset_ij,
            }
            np.random.seed(reset_seed)
            env.action_space.seed(reset_seed)
            env.reset(seed=reset_seed, options={"task_info": task_info})
            native.set_xy(reset_xy)
            native.set_goal(goal_xy=reset_xy)
            observation = np.asarray(native.get_ob(), np.float32)
            episode_inverted = False
            replay.start_episode()

            for command_index in range(commands_per_episode):
                target_xy, target_info = sampler.sample_target(observation[:2], rng)
                command = observation.copy()
                command[:2] = target_xy
                native.set_goal(goal_xy=target_xy)
                states = [observation.copy()]
                actions = []
                initial_xy = observation[:2].copy()
                initial_error = float(np.linalg.norm(initial_xy - target_xy))
                best_error = initial_error
                minimum_uprightness = torso_measurements(native)[0]
                terminated = False
                truncated = False
                for primitive_index in range(command_steps):
                    action_key, step_key = jax.random.split(action_key)
                    action = np.asarray(
                        action_fn(agent, observation[None], command[None], step_key)
                    )[0]
                    observation, _unused_reward, terminated, truncated, _info = env.step(action)
                    observation = np.asarray(observation, np.float32)
                    states.append(observation.copy())
                    actions.append(np.asarray(action, np.float32))
                    total_steps += 1
                    best_error = min(
                        best_error,
                        float(np.linalg.norm(observation[:2] - target_xy)),
                    )
                    minimum_uprightness = min(
                        minimum_uprightness, torso_measurements(native)[0]
                    )
                    final_primitive = (
                        command_index == commands_per_episode - 1
                        and primitive_index == command_steps - 1
                    )
                    if (terminated or truncated) and not final_primitive:
                        raise RuntimeError(
                            "fine-tuning environment ended before the fixed chain horizon"
                        )
                replay.add_command(np.asarray(states), np.asarray(actions))
                latest_info = None
                for _update in range(command_steps):
                    online_batch = replay.sample(
                        online_batch_size, agent.config, rng
                    )
                    if offline_dataset is None:
                        batch = online_batch
                    else:
                        offline_indices = rng.integers(
                            offline_dataset.size,
                            size=offline_batch_size,
                        )
                        # GCDataset uses NumPy's RNG for future-goal sampling.
                        np.random.seed(int(rng.integers(0, 2**31 - 1)))
                        offline_batch = offline_dataset.sample(
                            offline_batch_size, idxs=offline_indices
                        )
                        batch = mix_update_batches(
                            offline_batch, online_batch
                        )
                    agent, latest_info = agent.update(batch)
                    total_updates += 1
                endpoint_uprightness = torso_measurements(native)[0]
                command_inverted = minimum_uprightness <= 0.0
                episode_inverted = episode_inverted or command_inverted
                commands.append(
                    {
                        "physical_episode": (total_steps - 1) // episode_steps,
                        "command_index": command_index,
                        "reset_training_index": reset_index,
                        **target_info,
                        "initial_error_m": initial_error,
                        "best_error_m": best_error,
                        "endpoint_error_m": float(
                            np.linalg.norm(observation[:2] - target_xy)
                        ),
                        "displacement_m": float(
                            np.linalg.norm(observation[:2] - initial_xy)
                        ),
                        "inverted_during_command": bool(command_inverted),
                        "inverted_at_endpoint": bool(endpoint_uprightness <= 0.0),
                    }
                )
                if len(commands) % 100 == 0:
                    loss = None
                    if latest_info is not None:
                        loss = float(latest_info["actor/actor_loss"])
                    print(
                        f"chained CRL fine-tune steps={total_steps}/{args.target_steps} "
                        f"commands={len(commands)} updates={total_updates} "
                        f"episode_inverted={int(episode_inverted)} actor_loss={loss}",
                        flush=True,
                    )
    finally:
        env.close()

    if total_steps != args.target_steps or total_updates != args.target_steps:
        raise RuntimeError("fine-tuning budget accounting failed")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"params_chained_{args.target_steps}.pkl"
    replay_path = output / f"online_replay_{args.target_steps}.npz"
    diagnostics_path = output / f"command_diagnostics_{args.target_steps}.npz"
    atomic_pickle(
        checkpoint_path, {"agent": flax.serialization.to_state_dict(agent)}
    )
    replay.save(replay_path)
    save_command_diagnostics(diagnostics_path, commands)
    live_distances = [row["live_target_distance_m"] for row in commands]
    offset_distances = [row["empirical_offset_distance_m"] for row in commands]
    snap_distances = [row["support_snap_distance_m"] for row in commands]
    payload = {
        "schema_version": 1,
        "protocol_id": development["protocol_id"],
        "finetuning_protocol_id": protocol["protocol_id"],
        "claim_level": protocol["claim_level"],
        "executor": "official_OGBench_CRL_continued_on_chained_commands",
        "environment": development["benchmark"]["dataset"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "flags": str(flags_path.resolve()),
        "flags_sha256": source_record["flags_sha256"],
        "input_checkpoint": str(input_checkpoint.resolve()),
        "input_checkpoint_sha256": input_checkpoint_sha256,
        "source_offline_training_steps": int(
            protocol["source_executor"]["offline_training_steps"]
        ),
        "finetuning_environment_steps": total_steps,
        "reported_total_executor_steps": int(
            protocol["source_executor"]["offline_training_steps"]
        ) + total_steps,
        "gradient_updates": total_updates,
        "training_seed": int(protocol["source_executor"]["training_seed"]),
        "online_replay": str(replay_path.resolve()),
        "online_replay_sha256": sha256(replay_path),
        "command_diagnostics": str(diagnostics_path.resolve()),
        "command_diagnostics_sha256": sha256(diagnostics_path),
        "commands": len(commands),
        "collector_rng_state": rng.bit_generator.state,
        "action_key": np.asarray(action_key, np.uint32).tolist(),
        "command_distance_quantiles_m": quantiles(live_distances),
        "empirical_offset_distance_quantiles_m": quantiles(offset_distances),
        "support_snap_distance_quantiles_m": quantiles(snap_distances),
        "offline_dataset_fraction_per_update_batch": offline_fraction,
        "offline_rows_per_update_batch": offline_batch_size,
        "online_rows_per_update_batch": online_batch_size,
        "online_goal_relabel_boundary": online_boundary,
        "command_duration_primitive_steps": command_steps,
        "physical_episode_primitive_steps": episode_steps,
        "reset_after_command_failure": False,
        "reset_after_command_success": False,
        "task_reward_used": False,
        "native_task_rows_accessed": False,
        "native_evaluation_panel_accessed": False,
        "passed": True,
    }
    atomic_json(record_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--development-protocol", type=Path, required=True)
    result.add_argument("--dataset-audit", type=Path, required=True)
    result.add_argument("--executor", type=Path, required=True)
    result.add_argument("--train-dataset", type=Path, required=True)
    result.add_argument("--target-steps", type=int, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--resume-record", type=Path)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
