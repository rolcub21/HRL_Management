#!/usr/bin/env python3
"""Freeze and execute the non-task chained CRL waypoint audit.

The waypoint bank is the complete OGBench validation set, partitioned at its
recorded terminals.  Every trajectory contributes one fresh-reset chain and
the ten command targets at 100-step intervals.  The evaluator never reads a
native benchmark task row or task goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def trajectory_ranges(terminals: np.ndarray) -> list[tuple[int, int]]:
    terminals = np.asarray(terminals, dtype=bool)
    ends = np.flatnonzero(terminals)
    if len(ends) == 0 or int(ends[-1]) != len(terminals) - 1:
        raise RuntimeError("validation data must contain complete trajectories")
    ranges: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        ranges.append((start, int(end)))
        start = int(end) + 1
    return ranges


def derive_bank(
    observations: np.ndarray,
    terminals: np.ndarray,
    command_steps: int,
    episode_commands: int,
) -> dict[str, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError("validation observations must be a state matrix")
    if command_steps <= 0 or episode_commands <= 0:
        raise ValueError("command contract must be positive")
    expected_transitions = command_steps * episode_commands
    starts, targets = [], []
    for start, end in trajectory_ranges(terminals):
        if end - start != expected_transitions:
            raise RuntimeError(
                "every validation trajectory must exactly match the derived "
                f"episode contract ({expected_transitions} transitions)"
            )
        starts.append(start)
        targets.append(
            [start + command_steps * offset for offset in range(1, episode_commands + 1)]
        )
    start_index = np.asarray(starts, dtype=np.int64)
    target_index = np.asarray(targets, dtype=np.int64)
    return {
        "start_index": start_index,
        "target_index": target_index,
        "source_xy": observations[start_index, :2].astype(np.float32),
        "target_xy": observations[target_index, :2].astype(np.float32),
    }


def fixed_contract_env_kwargs(command_steps: int, episode_commands: int) -> dict:
    """Return the evaluator adapter needed to execute every command in full.

    OGBench's task evaluator normally terminates as soon as its current goal is
    reached. A chained executor audit instead has a fixed primitive-step
    contract, so native goal contact remains observable but cannot end a
    command or chain.
    """
    if command_steps <= 0 or episode_commands <= 0:
        raise ValueError("command contract must be positive")
    return {
        "terminate_at_goal": False,
        "success_timing": "post",
        "max_episode_steps": command_steps * episode_commands,
    }


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("native_evaluation_panel_accessed"):
        raise RuntimeError("executor audit refuses a protocol that opened the native panel")
    return protocol


def audited_datasets(path: Path) -> dict:
    report = json.loads(path.read_text())
    if not report.get("passed") or report.get("mode") != "dataset":
        raise RuntimeError("a passing dataset preflight is required")
    return report["details"]["dataset"]


def verify_audited_dataset(path: Path, record: dict, label: str) -> None:
    if Path(record["path"]).resolve() != path.resolve():
        raise RuntimeError(f"{label} dataset path differs from its preflight")
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"{label} dataset changed after its preflight")


def freeze_bank(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol)
    datasets = audited_datasets(args.dataset_audit)
    verify_audited_dataset(args.validation, datasets["validation"], "validation")
    command_steps = int(protocol["frozen_method_instantiation"]["command_duration_primitive_steps"])
    episode_commands = int(protocol["frozen_method_instantiation"]["episode_commands"])
    with np.load(args.validation, allow_pickle=False) as data:
        bank = derive_bank(
            data["observations"], data["terminals"], command_steps, episode_commands
        )
    if args.bank.is_file():
        with np.load(args.bank, allow_pickle=False) as stored:
            for key, expected in bank.items():
                if key not in stored or not np.array_equal(stored[key], expected):
                    raise RuntimeError("frozen executor audit bank changed")
    else:
        atomic_npz(args.bank, **bank)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "claim_level": protocol["claim_level"],
        "selection_rule": (
            "all complete held-out validation trajectories; one target every "
            "command_duration steps through the terminal row"
        ),
        "validation_dataset": str(args.validation.resolve()),
        "validation_dataset_sha256": sha256(args.validation),
        "dataset_audit": str(args.dataset_audit.resolve()),
        "dataset_audit_sha256": sha256(args.dataset_audit),
        "bank": str(args.bank.resolve()),
        "bank_sha256": sha256(args.bank),
        "chains": int(len(bank["start_index"])),
        "commands_per_chain": episode_commands,
        "command_duration_primitive_steps": command_steps,
        "commands": int(bank["target_index"].size),
        "native_task_rows_accessed": False,
        "native_task_goals_accessed": False,
        "task_reward_used": False,
        "passed": True,
    }
    atomic_json(args.bank_meta, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def create_agent(flags: dict, train_dataset: Path):
    from agents import agents
    from agents.crl import get_config

    config = get_config()
    config.update(flags["agent"])
    with np.load(train_dataset, allow_pickle=False) as data:
        observations = np.asarray(data["observations"][:1], dtype=np.float32)
        actions = np.asarray(data["actions"][:1], dtype=np.float32)
    return agents[config["agent_name"]].create(
        int(flags["seed"]), observations, actions, config
    )


def torso_measurements(native) -> tuple[float, float]:
    body = native.model.body("torso").id
    upright = float(native.data.xmat[body].reshape(3, 3)[2, 2])
    height = float(native.data.xpos[body, 2])
    return upright, height


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    return {
        "commands": len(rows),
        "contact_rate_at_evaluator_radius": mean("contact"),
        "positive_progress_rate": mean("positive_progress"),
        "initial_distance_mean_m": mean("initial_distance_m"),
        "endpoint_error_mean_m": mean("endpoint_error_m"),
        "endpoint_error_median_m": float(
            np.median([row["endpoint_error_m"] for row in rows])
        ),
        "best_error_mean_m": mean("best_error_m"),
        "displacement_mean_m": mean("displacement_m"),
        "best_progress_fraction_mean": mean("best_progress_fraction"),
        "endpoint_inversion_rate": mean("inverted_at_endpoint"),
        "any_inversion_rate": mean("inverted_during_command"),
        "endpoint_uprightness_mean": mean("endpoint_uprightness"),
    }


def chain_inversion_summary(rows: list[dict]) -> dict:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["chain"]), []).append(row)
    inverted = 0
    recovered = 0
    first_events = []
    for chain, chain_rows in sorted(grouped.items()):
        chain_rows.sort(key=lambda row: int(row["command_index"]))
        first = next(
            (row for row in chain_rows if bool(row["inverted_during_command"])),
            None,
        )
        if first is None:
            continue
        inverted += 1
        first_events.append(
            {
                "chain": chain,
                "command_index": int(first["command_index"]),
                "initial_distance_m": float(first["initial_distance_m"]),
            }
        )
        if not bool(chain_rows[-1]["inverted_at_endpoint"]):
            recovered += 1
    count = len(grouped)
    return {
        "chains": count,
        "chains_with_inversion": inverted,
        "chain_inversion_rate": inverted / count if count else None,
        "inverted_chains_recovered_by_chain_end": recovered,
        "first_inversion_events": first_events,
    }


def evaluate(args: argparse.Namespace) -> dict:
    import flax.serialization
    import jax
    import ogbench

    protocol = load_protocol(args.protocol)
    executor = json.loads(args.executor.read_text())
    bank_meta = json.loads(args.bank_meta.read_text())
    datasets = audited_datasets(args.dataset_audit)
    verify_audited_dataset(args.train_dataset, datasets["train"], "training")
    verify_audited_dataset(args.validation, datasets["validation"], "validation")
    if not executor.get("passed") or executor["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("executor record does not conform to this protocol")
    checkpoint = Path(executor["checkpoint"])
    flags_path = Path(executor["flags"])
    if sha256(checkpoint) != executor["checkpoint_sha256"]:
        raise RuntimeError("executor checkpoint changed")
    if sha256(flags_path) != executor["flags_sha256"]:
        raise RuntimeError("executor flags changed")
    if sha256(args.bank) != bank_meta["bank_sha256"]:
        raise RuntimeError("executor audit bank changed")
    if sha256(args.validation) != bank_meta["validation_dataset_sha256"]:
        raise RuntimeError("validation dataset changed")

    flags = json.loads(flags_path.read_text())
    agent = create_agent(flags, args.train_dataset)
    with checkpoint.open("rb") as handle:
        agent = flax.serialization.from_state_dict(agent, pickle.load(handle)["agent"])
    action_fn = jax.jit(
        lambda observation, goal, key: agent.sample_actions(
            observation, goal, seed=key, temperature=0.0
        )
    )
    action_key = jax.random.PRNGKey(int(protocol["executor"]["training_seed"]))

    with np.load(args.bank, allow_pickle=False) as stored:
        source_xy = np.asarray(stored["source_xy"], dtype=np.float32)
        target_xy = np.asarray(stored["target_xy"], dtype=np.float32)
        start_index = np.asarray(stored["start_index"], dtype=np.int64)
        target_index = np.asarray(stored["target_index"], dtype=np.int64)

    command_steps = int(protocol["frozen_method_instantiation"]["command_duration_primitive_steps"])
    episode_commands = int(protocol["frozen_method_instantiation"]["episode_commands"])
    evaluator_radius = float(protocol["benchmark"]["success_radius_m"])
    environment_adapter = fixed_contract_env_kwargs(command_steps, episode_commands)
    env = ogbench.make_env_and_datasets(
        protocol["benchmark"]["dataset"],
        env_only=True,
        **environment_adapter,
    )
    native = env.unwrapped if hasattr(env, "unwrapped") else env
    if bool(native._terminate_at_goal):
        raise RuntimeError("fixed-horizon audit environment still terminates at goal")
    if native._success_timing != "post":
        raise RuntimeError("fixed-horizon audit environment changed success timing")
    rows: list[dict] = []
    executed_steps = 0
    try:
        for chain in range(len(source_xy)):
            reset_seed = int(protocol["executor"]["training_seed"] + chain)
            source_ij = tuple(int(value) for value in native.xy_to_ij(source_xy[chain]))
            task_info = {
                "task_name": f"validation_chain_{chain:03d}",
                "init_ij": source_ij,
                "goal_ij": source_ij,
            }
            np.random.seed(reset_seed)
            env.action_space.seed(reset_seed)
            env.reset(seed=reset_seed, options={"task_info": task_info})
            native.set_xy(source_xy[chain])
            native.set_goal(goal_xy=source_xy[chain])
            observation = np.asarray(native.get_ob(), dtype=np.float32)

            for command_index in range(episode_commands):
                command_start = observation.copy()
                command = command_start.copy()
                command[:2] = target_xy[chain, command_index]
                native.set_goal(goal_xy=command[:2])
                initial_xy = command_start[:2].copy()
                initial_distance = float(np.linalg.norm(initial_xy - command[:2]))
                best_distance = initial_distance
                minimum_uprightness = torso_measurements(native)[0]
                endpoint_height = torso_measurements(native)[1]
                for step in range(command_steps):
                    action_key, step_key = jax.random.split(action_key)
                    action = np.asarray(
                        action_fn(observation[None], command[None], step_key)
                    )[0]
                    observation, _reward, terminated, truncated, _info = env.step(action)
                    observation = np.asarray(observation, dtype=np.float32)
                    executed_steps += 1
                    distance = float(np.linalg.norm(observation[:2] - command[:2]))
                    best_distance = min(best_distance, distance)
                    uprightness, endpoint_height = torso_measurements(native)
                    minimum_uprightness = min(minimum_uprightness, uprightness)
                    if (terminated or truncated) and not (
                        command_index == episode_commands - 1
                        and step == command_steps - 1
                    ):
                        raise RuntimeError("audit environment ended before its fixed command contract")
                endpoint_distance = float(np.linalg.norm(observation[:2] - command[:2]))
                endpoint_uprightness, endpoint_height = torso_measurements(native)
                denominator = max(initial_distance, np.finfo(np.float32).eps)
                rows.append(
                    {
                        "chain": chain,
                        "command_index": command_index,
                        "source_validation_index": int(start_index[chain]),
                        "target_validation_index": int(target_index[chain, command_index]),
                        "initial_distance_m": initial_distance,
                        "endpoint_error_m": endpoint_distance,
                        "best_error_m": best_distance,
                        "displacement_m": float(np.linalg.norm(observation[:2] - initial_xy)),
                        "best_progress_fraction": float((initial_distance - best_distance) / denominator),
                        "positive_progress": bool(best_distance < initial_distance),
                        "contact": bool(best_distance <= evaluator_radius),
                        "minimum_uprightness": minimum_uprightness,
                        "endpoint_uprightness": endpoint_uprightness,
                        "endpoint_torso_height": endpoint_height,
                        "inverted_during_command": bool(minimum_uprightness <= 0.0),
                        "inverted_at_endpoint": bool(endpoint_uprightness <= 0.0),
                    }
                )
            print(
                f"ogbench large chained executor chain={chain + 1}/{len(source_xy)}",
                flush=True,
            )
    finally:
        env.close()

    by_position = []
    for command_index in range(episode_commands):
        by_position.append(
            {
                "command_index": command_index,
                **aggregate([row for row in rows if row["command_index"] == command_index]),
            }
        )
    split = episode_commands // 2
    early = [row for row in rows if row["command_index"] < split]
    late = [row for row in rows if row["command_index"] >= split]
    numeric_finite = all(
        np.isfinite(value)
        for row in rows
        for key, value in row.items()
        if key not in {
            "chain", "command_index", "source_validation_index", "target_validation_index",
            "positive_progress", "contact", "inverted_during_command", "inverted_at_endpoint",
        }
    )
    checks = {
        "checkpoint_matches_executor_record": True,
        "bank_matches_frozen_validation_rule": True,
        "all_validation_trajectories_evaluated": len(source_xy) == bank_meta["chains"],
        "fixed_command_duration_respected": executed_steps
        == bank_meta["commands"] * command_steps,
        "native_goal_contact_does_not_end_fixed_horizon": not bool(
            native._terminate_at_goal
        ),
        "native_success_is_measured_after_each_step": native._success_timing == "post",
        "all_metrics_finite": numeric_finite,
        "native_task_rows_not_accessed": True,
        "native_task_goals_not_accessed": True,
        "task_reward_not_used": True,
        "no_task_gradient_updates": True,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "claim_level": protocol["claim_level"],
        "scope": (
            "complete held-out-validation chained 100-step executor audit; "
            "no TF-SN planner or native OGBench task verdict"
        ),
        "executor_checkpoint": str(checkpoint.resolve()),
        "executor_checkpoint_sha256": executor["checkpoint_sha256"],
        "bank": str(args.bank.resolve()),
        "bank_sha256": bank_meta["bank_sha256"],
        "dataset_audit": str(args.dataset_audit.resolve()),
        "dataset_audit_sha256": sha256(args.dataset_audit),
        "chains": len(source_xy),
        "commands_per_chain": episode_commands,
        "command_duration_primitive_steps": command_steps,
        "primitive_steps": len(rows) * command_steps,
        "command_goal_representation": (
            "copy command-start native observation and replace only native XY"
        ),
        "reset_contract": (
            "fresh simulator reset per validation trajectory; exact source XY only; "
            "no qpos/qvel restoration"
        ),
        "environment_execution_adapter": {
            **environment_adapter,
            "role": (
                "evaluator_only: suppress native goal-contact termination so the "
                "frozen TF-SN command and chain horizons execute in full"
            ),
            "dynamics_changed": False,
            "policy_inputs_changed": False,
        },
        "success_metric_role": (
            "benchmark radius used by auditor only and not as a performance gate or planner input"
        ),
        "benchmark_evaluator_radius_m": evaluator_radius,
        "overall": aggregate(rows),
        "early_half": aggregate(early),
        "late_half": aggregate(late),
        "chain_inversion": chain_inversion_summary(rows),
        "by_command_position": by_position,
        "rows": rows,
        "checks": checks,
        "performance_threshold_applied": False,
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "passed": all(checks.values()),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"chained executor audit failed conformance: {checks}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--dataset-audit", type=Path, required=True)
    freeze.add_argument("--validation", type=Path, required=True)
    freeze.add_argument("--bank", type=Path, required=True)
    freeze.add_argument("--bank-meta", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--dataset-audit", type=Path, required=True)
    run.add_argument("--executor", type=Path, required=True)
    run.add_argument("--train-dataset", type=Path, required=True)
    run.add_argument("--validation", type=Path, required=True)
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument("--bank-meta", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "freeze":
        freeze_bank(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
