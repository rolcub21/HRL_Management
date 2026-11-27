#!/usr/bin/env python3
"""Audit the official CRL executor on adaptive empirical-reach commands.

Every command is an exact 100-step source-to-endpoint transition from the
complete held-out validation dataset.  At every handoff the candidate whose
source best matches the Ant's actual current observation is selected.  This
keeps command reach tied to the OGBench data and mirrors TF-SN's replanning
from the realized state without importing a metric distance band.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from . import audit_executor as base


def load_audit_protocol(path: Path, development_protocol: dict) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("native_evaluation_panel_accessed"):
        raise RuntimeError("adaptive reach audit protocol opened the native panel")
    if protocol["parent_development_protocol_sha256"] != base.sha256(
        Path(protocol["parent_development_protocol"])
    ):
        raise RuntimeError("adaptive reach audit parent protocol changed")
    if protocol["source_executor"]["checkpoint_sha256"] == "":
        raise RuntimeError("adaptive reach audit must freeze the executor hash")
    contract = protocol["audit_contract"]
    frozen = development_protocol["frozen_method_instantiation"]
    if int(contract["command_duration_primitive_steps"]) != int(
        frozen["command_duration_primitive_steps"]
    ):
        raise RuntimeError("adaptive audit changed the frozen command duration")
    if int(contract["commands_per_chain"]) != int(frozen["episode_commands"]):
        raise RuntimeError("adaptive audit changed the frozen command horizon")
    if contract["distance_cutoffs_m"] or contract["fixed_metric_distance_band"] is not None:
        raise RuntimeError("adaptive audit may not use a metric distance band")
    return protocol


def derive_transition_bank(
    observations: np.ndarray,
    terminals: np.ndarray,
    command_steps: int,
) -> dict[str, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError("validation observations must be a state matrix")
    if command_steps <= 0:
        raise ValueError("command duration must be positive")
    source_indices: list[int] = []
    target_indices: list[int] = []
    reset_indices: list[int] = []
    for start, end in base.trajectory_ranges(terminals):
        if end - start < command_steps:
            raise RuntimeError("validation trajectory is shorter than one command")
        reset_indices.append(start)
        for source in range(start, end - command_steps + 1):
            source_indices.append(source)
            target_indices.append(source + command_steps)
    source_index = np.asarray(source_indices, dtype=np.int64)
    target_index = np.asarray(target_indices, dtype=np.int64)
    reset_index = np.asarray(reset_indices, dtype=np.int64)
    source_observation = observations[source_index].astype(np.float32)
    target_xy = observations[target_index, :2].astype(np.float32)
    coordinate_std = source_observation.astype(np.float64).std(axis=0)
    active_coordinate = coordinate_std > 0.0
    if not np.any(active_coordinate):
        raise RuntimeError("validation source observations have no varying coordinates")
    return {
        "source_index": source_index,
        "target_index": target_index,
        "reset_index": reset_index,
        "source_observation": source_observation,
        "target_xy": target_xy,
        "source_mean": source_observation.astype(np.float64).mean(axis=0).astype(np.float32),
        "source_std": coordinate_std.astype(np.float32),
        "active_coordinate": active_coordinate.astype(bool),
    }


def standardized_sources(bank: dict[str, np.ndarray]) -> np.ndarray:
    active = np.asarray(bank["active_coordinate"], dtype=bool)
    mean = np.asarray(bank["source_mean"], dtype=np.float32)[active]
    scale = np.asarray(bank["source_std"], dtype=np.float32)[active]
    if np.any(scale <= 0.0):
        raise RuntimeError("active source coordinates must have positive scale")
    return (
        (np.asarray(bank["source_observation"], dtype=np.float32)[:, active] - mean)
        / scale
    ).astype(np.float32)


def standardize_query(observation: np.ndarray, bank: dict[str, np.ndarray]) -> np.ndarray:
    active = np.asarray(bank["active_coordinate"], dtype=bool)
    mean = np.asarray(bank["source_mean"], dtype=np.float32)[active]
    scale = np.asarray(bank["source_std"], dtype=np.float32)[active]
    return ((np.asarray(observation, dtype=np.float32)[active] - mean) / scale).astype(
        np.float32
    )


def nearest_transition(
    standardized_source: np.ndarray,
    standardized_observation: np.ndarray,
) -> tuple[int, float]:
    difference = np.asarray(standardized_source, dtype=np.float32) - np.asarray(
        standardized_observation, dtype=np.float32
    )[None]
    squared = np.einsum("ij,ij->i", difference, difference, optimize=True)
    index = int(np.argmin(squared))
    return index, float(np.sqrt(squared[index]))


def freeze_bank(args: argparse.Namespace) -> dict:
    development = base.load_protocol(args.development_protocol)
    protocol = load_audit_protocol(args.protocol, development)
    datasets = base.audited_datasets(args.dataset_audit)
    base.verify_audited_dataset(args.validation, datasets["validation"], "validation")
    command_steps = int(protocol["audit_contract"]["command_duration_primitive_steps"])
    with np.load(args.validation, allow_pickle=False) as data:
        bank = derive_transition_bank(
            data["observations"], data["terminals"], command_steps
        )
    if args.bank.is_file():
        with np.load(args.bank, allow_pickle=False) as stored:
            for key, expected in bank.items():
                if key not in stored or not np.array_equal(stored[key], expected):
                    raise RuntimeError("frozen adaptive reach bank changed")
    else:
        base.atomic_npz(args.bank, **bank)
    recorded_distance = np.linalg.norm(
        bank["target_xy"] - bank["source_observation"][:, :2], axis=1
    )
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "claim_level": protocol["claim_level"],
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": base.sha256(args.protocol),
        "selection_rule": protocol["audit_contract"]["selection_at_each_handoff"],
        "candidate_rule": protocol["audit_contract"]["candidate_commands"],
        "source_metric": protocol["audit_contract"]["source_metric"],
        "distance_cutoffs_m": [],
        "validation_dataset": str(args.validation.resolve()),
        "validation_dataset_sha256": base.sha256(args.validation),
        "dataset_audit": str(args.dataset_audit.resolve()),
        "dataset_audit_sha256": base.sha256(args.dataset_audit),
        "bank": str(args.bank.resolve()),
        "bank_sha256": base.sha256(args.bank),
        "chains": int(len(bank["reset_index"])),
        "candidate_transitions": int(len(bank["source_index"])),
        "active_source_coordinates": int(np.sum(bank["active_coordinate"])),
        "command_duration_primitive_steps": command_steps,
        "recorded_command_distance_m": {
            "mean": float(recorded_distance.mean()),
            "rms": float(np.sqrt(np.mean(np.square(recorded_distance)))),
            "median": float(np.median(recorded_distance)),
            "p90": float(np.quantile(recorded_distance, 0.9)),
        },
        "native_task_rows_accessed": False,
        "native_task_goals_accessed": False,
        "task_reward_used": False,
        "passed": True,
    }
    base.atomic_json(args.bank_meta, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def evaluate(args: argparse.Namespace) -> dict:
    import flax.serialization
    import jax
    import ogbench
    from scipy.spatial import cKDTree

    development = base.load_protocol(args.development_protocol)
    protocol = load_audit_protocol(args.protocol, development)
    executor = json.loads(args.executor.read_text())
    bank_meta = json.loads(args.bank_meta.read_text())
    datasets = base.audited_datasets(args.dataset_audit)
    base.verify_audited_dataset(args.train_dataset, datasets["train"], "training")
    base.verify_audited_dataset(args.validation, datasets["validation"], "validation")
    checkpoint = Path(executor["checkpoint"])
    flags_path = Path(executor["flags"])
    if executor["checkpoint_sha256"] != protocol["source_executor"]["checkpoint_sha256"]:
        raise RuntimeError("executor differs from the audit's frozen official checkpoint")
    if base.sha256(checkpoint) != executor["checkpoint_sha256"]:
        raise RuntimeError("executor checkpoint changed")
    if base.sha256(flags_path) != executor["flags_sha256"]:
        raise RuntimeError("executor flags changed")
    if base.sha256(args.bank) != bank_meta["bank_sha256"]:
        raise RuntimeError("adaptive reach bank changed")
    if base.sha256(args.protocol) != bank_meta["protocol_sha256"]:
        raise RuntimeError("adaptive reach audit protocol changed")

    flags = json.loads(flags_path.read_text())
    agent = base.create_agent(flags, args.train_dataset)
    with checkpoint.open("rb") as handle:
        agent = flax.serialization.from_state_dict(agent, pickle.load(handle)["agent"])
    action_fn = jax.jit(
        lambda observation, goal, key: agent.sample_actions(
            observation, goal, seed=key, temperature=0.0
        )
    )
    action_key = jax.random.PRNGKey(int(development["executor"]["training_seed"]))

    with np.load(args.bank, allow_pickle=False) as stored:
        bank = {key: np.asarray(stored[key]) for key in stored.files}
    standardized_source = standardized_sources(bank)
    source_tree = cKDTree(standardized_source)
    command_steps = int(protocol["audit_contract"]["command_duration_primitive_steps"])
    episode_commands = int(protocol["audit_contract"]["commands_per_chain"])
    evaluator_radius = float(development["benchmark"]["success_radius_m"])
    environment_adapter = base.fixed_contract_env_kwargs(
        command_steps, episode_commands
    )
    env = ogbench.make_env_and_datasets(
        development["benchmark"]["dataset"], env_only=True, **environment_adapter
    )
    native = env.unwrapped if hasattr(env, "unwrapped") else env
    if bool(native._terminate_at_goal) or native._success_timing != "post":
        raise RuntimeError("adaptive audit environment violates fixed contract")

    with np.load(args.validation, allow_pickle=False) as data:
        validation_observation = np.asarray(data["observations"], dtype=np.float32)
    rows: list[dict] = []
    executed_steps = 0
    try:
        for chain, reset_index in enumerate(bank["reset_index"]):
            reset_xy = validation_observation[int(reset_index), :2]
            reset_seed = int(development["executor"]["training_seed"] + chain)
            reset_ij = tuple(int(value) for value in native.xy_to_ij(reset_xy))
            task_info = {
                "task_name": f"adaptive_reach_chain_{chain:03d}",
                "init_ij": reset_ij,
                "goal_ij": reset_ij,
            }
            np.random.seed(reset_seed)
            env.action_space.seed(reset_seed)
            env.reset(seed=reset_seed, options={"task_info": task_info})
            native.set_xy(reset_xy)
            native.set_goal(goal_xy=reset_xy)
            observation = np.asarray(native.get_ob(), dtype=np.float32)

            for command_index in range(episode_commands):
                match_distance, selected = source_tree.query(
                    standardize_query(observation, bank), k=1, eps=0.0
                )
                selected = int(selected)
                match_distance = float(match_distance)
                command_start = observation.copy()
                command = command_start.copy()
                command[:2] = bank["target_xy"][selected]
                native.set_goal(goal_xy=command[:2])
                initial_xy = command_start[:2].copy()
                initial_distance = float(np.linalg.norm(initial_xy - command[:2]))
                recorded_distance = float(
                    np.linalg.norm(
                        bank["source_observation"][selected, :2]
                        - bank["target_xy"][selected]
                    )
                )
                best_distance = initial_distance
                minimum_uprightness, endpoint_height = base.torso_measurements(native)
                for step in range(command_steps):
                    action_key, step_key = jax.random.split(action_key)
                    action = np.asarray(
                        action_fn(observation[None], command[None], step_key)
                    )[0]
                    observation, _reward, terminated, truncated, _info = env.step(action)
                    observation = np.asarray(observation, dtype=np.float32)
                    executed_steps += 1
                    best_distance = min(
                        best_distance,
                        float(np.linalg.norm(observation[:2] - command[:2])),
                    )
                    uprightness, endpoint_height = base.torso_measurements(native)
                    minimum_uprightness = min(minimum_uprightness, uprightness)
                    if (terminated or truncated) and not (
                        command_index == episode_commands - 1
                        and step == command_steps - 1
                    ):
                        raise RuntimeError(
                            "adaptive audit environment ended before fixed contract"
                        )
                endpoint_distance = float(
                    np.linalg.norm(observation[:2] - command[:2])
                )
                endpoint_uprightness, endpoint_height = base.torso_measurements(native)
                denominator = max(initial_distance, np.finfo(np.float32).eps)
                rows.append(
                    {
                        "chain": chain,
                        "command_index": command_index,
                        "selected_source_validation_index": int(
                            bank["source_index"][selected]
                        ),
                        "selected_target_validation_index": int(
                            bank["target_index"][selected]
                        ),
                        "source_match_standardized_distance": match_distance,
                        "recorded_command_distance_m": recorded_distance,
                        "initial_distance_m": initial_distance,
                        "endpoint_error_m": endpoint_distance,
                        "best_error_m": best_distance,
                        "displacement_m": float(
                            np.linalg.norm(observation[:2] - initial_xy)
                        ),
                        "best_progress_fraction": float(
                            (initial_distance - best_distance) / denominator
                        ),
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
                f"ogbench adaptive reach executor chain={chain + 1}/{len(bank['reset_index'])}",
                flush=True,
            )
    finally:
        env.close()

    inversion = base.chain_inversion_summary(rows)
    maximum_inversion = float(protocol["decision"]["maximum_chain_inversion_rate"])
    finite_metrics = all(
        np.isfinite(row[key])
        for row in rows
        for key in (
            "source_match_standardized_distance",
            "recorded_command_distance_m",
            "initial_distance_m",
            "endpoint_error_m",
            "best_error_m",
            "displacement_m",
            "best_progress_fraction",
            "minimum_uprightness",
            "endpoint_uprightness",
            "endpoint_torso_height",
        )
    )
    checks = {
        "checkpoint_matches_frozen_official_executor": True,
        "bank_matches_frozen_protocol": True,
        "all_held_out_reset_chains_evaluated": len(bank["reset_index"])
        == bank_meta["chains"],
        "fixed_command_duration_respected": executed_steps
        == len(rows) * command_steps,
        "no_metric_distance_band_used": not protocol["audit_contract"][
            "distance_cutoffs_m"
        ],
        "all_metrics_finite": finite_metrics,
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
            "official OGBench CRL executor-only adaptive empirical-reach audit; "
            "no TF-SN planner or native OGBench task verdict"
        ),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": base.sha256(args.protocol),
        "executor_checkpoint": str(checkpoint.resolve()),
        "executor_checkpoint_sha256": executor["checkpoint_sha256"],
        "bank": str(args.bank.resolve()),
        "bank_sha256": bank_meta["bank_sha256"],
        "chains": int(len(bank["reset_index"])),
        "commands_per_chain": episode_commands,
        "command_duration_primitive_steps": command_steps,
        "primitive_steps": executed_steps,
        "command_selection": protocol["audit_contract"],
        "overall": base.aggregate(rows),
        "chain_inversion": inversion,
        "source_match_standardized_distance": {
            "mean": float(
                np.mean([row["source_match_standardized_distance"] for row in rows])
            ),
            "median": float(
                np.median([row["source_match_standardized_distance"] for row in rows])
            ),
            "p90": float(
                np.quantile(
                    [row["source_match_standardized_distance"] for row in rows], 0.9
                )
            ),
        },
        "recorded_command_distance_m": {
            "mean": float(np.mean([row["recorded_command_distance_m"] for row in rows])),
            "rms": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            [row["recorded_command_distance_m"] for row in rows]
                        )
                    )
                )
            ),
        },
        "issued_command_distance_m": {
            "mean": float(np.mean([row["initial_distance_m"] for row in rows])),
            "rms": float(
                np.sqrt(np.mean(np.square([row["initial_distance_m"] for row in rows])))
            ),
        },
        "decision": {
            "maximum_chain_inversion_rate": maximum_inversion,
            "observed_chain_inversion_rate": inversion["chain_inversion_rate"],
            "official_executor_accepted": bool(
                inversion["chain_inversion_rate"] <= maximum_inversion
            ),
            "pass_action": protocol["decision"]["pass"],
            "fail_action": protocol["decision"]["fail"],
        },
        "rows": rows,
        "checks": checks,
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "passed": all(checks.values()),
    }
    base.atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"adaptive reach audit failed conformance: {checks}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("freeze", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--development-protocol", type=Path, required=True)
        command.add_argument("--dataset-audit", type=Path, required=True)
        command.add_argument("--validation", type=Path, required=True)
        command.add_argument("--bank", type=Path, required=True)
        command.add_argument("--bank-meta", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--executor", type=Path, required=True)
            command.add_argument("--train-dataset", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "freeze":
        freeze_bank(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
