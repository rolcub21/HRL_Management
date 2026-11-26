"""Separate the Ant oracle command contract from consequence-model fit.

Step 1 replays the existing oracle cell routes while holding every selected
waypoint for the complete TF-SN 100-step command.  Step 2 is offline: it scores
the frozen round-2 consequence model on exactly those executed commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.antmaze.current_tfsn.collect import execute_command
from scripts.antmaze.current_tfsn.constants import (
    COMMANDS_PER_EPISODE,
    COMMAND_STEPS,
    CONTACT_RADIUS,
    LANDING_SIGMA,
    STATE_DIM,
    VALIDATION_FRACTION,
)
from scripts.antmaze.current_tfsn.data import consequence_arrays
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import (
    _oracle_cell_route,
    dump,
    panel,
    paths,
)
from scripts.antmaze.current_tfsn.mdn import MDNEnsemble, mixture_nll

from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


METHOD_REVISION = "antmaze_oracle_fixed_command_contract_r7"
MODEL_DIAGNOSTIC_REVISION = "antmaze_oracle_command_model_diagnostic_r7"
ORIGINAL_ORACLE_SUCCESSES = (0, 2, 3, 4, 5, 7, 8)
STALL_RADIUS_M = 0.2
STALL_SAMPLES_PER_MEMBER = 2048


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_directory(root):
    return (
        paths(root)["eval"]
        / "development"
        / "oracle_fixed_command_contract_r7"
    )


def command_oracle_contract(args):
    """Run the fixed oracle routes with exactly one decision per 100 steps."""
    selected = panel(args.root, "development", 9, 37701)
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    stored_states = []
    stored_commands = []
    stored_task_ids = []
    stored_command_indices = []
    stored_waypoint_indices = []

    for task in selected["tasks"]:
        task_id = int(task["task_id"])
        goal = np.asarray(task["goal"], np.float32)
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = full_state(observation)
        initial_distance = float(np.linalg.norm(state[:2] - goal))
        cells, waypoints = _oracle_cell_route(
            env.unwrapped.maze, state[:2], goal
        )
        waypoint_index = 0
        minimum_distance = initial_distance
        success = initial_distance <= CONTACT_RADIUS
        steps_to_goal = 0 if success else None
        decisions = []

        for command_index in range(COMMANDS_PER_EPISODE):
            if success:
                break
            # This is the only waypoint advancement point.  Unlike the old
            # oracle, no primitive-step proximity event changes the command.
            while (
                waypoint_index < len(waypoints)
                and np.linalg.norm(
                    state[:2] - waypoints[waypoint_index]
                ) <= CONTACT_RADIUS
            ):
                waypoint_index += 1
            if waypoint_index >= len(waypoints):
                success = np.linalg.norm(state[:2] - goal) <= CONTACT_RADIUS
                break

            command = np.asarray(waypoints[waypoint_index], np.float32)
            start = state.copy()
            trajectory, rewards = execute_command(
                env, executor, state, command, hidden_goal=goal
            )
            goal_distances = np.linalg.norm(
                trajectory[1:, :2] - goal[None], axis=1
            )
            waypoint_distances = np.linalg.norm(
                trajectory[1:, :2] - command[None], axis=1
            )
            minimum_distance = min(
                minimum_distance, float(goal_distances.min())
            )
            if steps_to_goal is None and rewards.any():
                steps_to_goal = (
                    command_index * COMMAND_STEPS
                    + int(np.flatnonzero(rewards)[0])
                    + 1
                )
                success = True

            movement = float(
                np.linalg.norm(trajectory[-1, :2] - start[:2])
            )
            decisions.append({
                "task_id": task_id,
                "command_index": int(command_index),
                "waypoint_index": int(waypoint_index),
                "command": command.tolist(),
                "start_xy": start[:2].tolist(),
                "actual_endpoint_xy": trajectory[-1, :2].tolist(),
                "moved_distance_m": movement,
                "observed_stall": bool(movement < STALL_RADIUS_M),
                "minimum_waypoint_distance_in_command_m": float(
                    waypoint_distances.min()
                ),
                "endpoint_waypoint_distance_m": float(
                    waypoint_distances[-1]
                ),
                "waypoint_contact_inside_command": bool(
                    np.any(waypoint_distances <= CONTACT_RADIUS)
                ),
                "goal_contact_inside_command": bool(rewards.any()),
            })
            stored_states.append(trajectory.astype(np.float32))
            stored_commands.append(command.copy())
            stored_task_ids.append(task_id)
            stored_command_indices.append(command_index)
            stored_waypoint_indices.append(waypoint_index)
            state = trajectory[-1]

        task_log = output / f"task_{task_id:03d}.jsonl"
        task_log.write_text(
            "\n".join(json.dumps(row) for row in decisions) + "\n"
        )
        row = {
            "task_id": task_id,
            "goal": task["goal"],
            "reset_seed": int(task["reset_seed"]),
            "initial_distance": initial_distance,
            "route_cells": [list(cell) for cell in cells],
            "route_edges": len(cells) - 1,
            "route_waypoints": [waypoint.tolist() for waypoint in waypoints],
            "waypoints_advanced_at_boundaries": int(waypoint_index),
            "success": bool(success),
            "steps_to_goal": steps_to_goal,
            "steps_executed": int(len(decisions) * COMMAND_STEPS),
            "commands_executed": int(len(decisions)),
            "minimum_distance": minimum_distance,
            "contact_radius": CONTACT_RADIUS,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    env.close()

    trajectories = np.stack(stored_states).astype(np.float32)
    command_array = np.asarray(stored_commands, np.float32)
    dataset = output / "executed_commands.npz"
    np.savez_compressed(
        dataset,
        states=trajectories,
        commands=command_array,
        task_id=np.asarray(stored_task_ids, np.int16),
        command_index=np.asarray(stored_command_indices, np.int16),
        waypoint_index=np.asarray(stored_waypoint_indices, np.int16),
    )
    result = {
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_oracle_contract_diagnostic",
        "panel": "development",
        "tasks": len(rows),
        "command_duration_steps": COMMAND_STEPS,
        "primitive_step_waypoint_switching": False,
        "waypoint_advancement": "command_boundaries_only",
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "successes": int(sum(row["success"] for row in rows)),
        "mean_steps_to_goal_success": (
            float(np.mean([
                row["steps_to_goal"] for row in rows if row["success"]
            ]))
            if any(row["success"] for row in rows) else None
        ),
        "old_primitive_switch_oracle_successes": 7,
        "old_primitive_switch_oracle_tasks": 9,
        "executed_commands": int(len(command_array)),
        "executed_command_dataset": str(dataset.resolve()),
        "executed_command_dataset_sha256": sha256_file(dataset),
        "task_reward_used_by_controller": False,
        "oracle_inputs": ["maze_occupancy", "supplied_goal"],
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "successes", "tasks", "success_rate",
            "mean_steps_to_goal_success", "executed_commands",
        )
    }, indent=2))


def member_nll_rows(model, inputs, targets):
    inputs = np.asarray(inputs, np.float32).reshape(-1, model.config.input_dim)
    targets = np.asarray(targets, np.float32).reshape(
        -1, model.config.output_dim
    )
    normalized_input = (inputs - model.input_mean) / model.input_std
    normalized_target = (targets - model.target_mean) / model.target_std
    x = torch.as_tensor(
        normalized_input, dtype=torch.float32, device=model.device
    )
    y = torch.as_tensor(
        normalized_target, dtype=torch.float32, device=model.device
    )
    rows = []
    for member in model.members:
        with torch.no_grad():
            rows.append(mixture_nll(*member(x), y).cpu().numpy())
    return np.stack(rows, axis=1).astype(np.float32)


def nearest_mode_xy_distance(model, inputs, targets):
    start_xy = np.asarray(inputs, np.float32)[:, :2]
    target_xy = np.asarray(targets, np.float32)[:, :2]
    values = []
    for member in range(len(model.members)):
        _weight, means, _stds = model.parameters(inputs, member)
        predicted_endpoint = start_xy[:, None] + means[..., :2]
        actual_endpoint = start_xy + target_xy
        values.append(
            np.linalg.norm(
                predicted_endpoint - actual_endpoint[:, None], axis=2
            ).min(axis=1)
        )
    return np.stack(values, axis=1).min(axis=1).astype(np.float32)


def predicted_stall_probability(model, inputs, seed):
    rng = np.random.default_rng(seed)
    values = []
    for member in range(len(model.members)):
        samples = model.sample_member(
            inputs, member, STALL_SAMPLES_PER_MEMBER, rng
        )
        values.append(
            np.linalg.norm(samples[..., :2], axis=2) < STALL_RADIUS_M
        )
    return np.concatenate(values, axis=1).mean(axis=1).astype(np.float32)


def calibration_bins(probability, observed):
    probability = np.asarray(probability, np.float64)
    observed = np.asarray(observed, bool)
    result = []
    for index in range(10):
        low = index / 10.0
        high = (index + 1) / 10.0
        mask = (probability >= low) & (
            probability <= high if index == 9 else probability < high
        )
        result.append({
            "low": low,
            "high": high,
            "count": int(mask.sum()),
            "predicted": float(probability[mask].mean()) if mask.any() else None,
            "observed": float(observed[mask].mean()) if mask.any() else None,
        })
    return result


def summarize_subset(name, mask, nll, percentile, mode_distance, predicted_stall,
                     observed_stall):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {"name": name, "commands": 0}
    return {
        "name": name,
        "commands": int(mask.sum()),
        "normalized_nll_mean": float(nll[mask].mean()),
        "normalized_nll_median": float(np.median(nll[mask])),
        "validation_nll_percentile_median": float(
            np.median(percentile[mask])
        ),
        "fraction_worse_than_validation_p90": float(
            np.mean(percentile[mask] > 0.90)
        ),
        "fraction_worse_than_validation_p95": float(
            np.mean(percentile[mask] > 0.95)
        ),
        "nearest_mode_xy_distance_mean_m": float(
            mode_distance[mask].mean()
        ),
        "nearest_mode_xy_distance_median_m": float(
            np.median(mode_distance[mask])
        ),
        "predicted_stall_rate": float(predicted_stall[mask].mean()),
        "observed_stall_rate": float(observed_stall[mask].mean()),
        "stall_brier": float(np.mean(
            np.square(predicted_stall[mask] - observed_stall[mask])
        )),
        "stall_calibration_bins": calibration_bins(
            predicted_stall[mask], observed_stall[mask]
        ),
    }


def command_model_diagnostic(args):
    output = output_directory(args.root)
    summary_path = output / "summary.json"
    dataset_path = output / "executed_commands.npz"
    if not summary_path.exists() or not dataset_path.exists():
        raise FileNotFoundError(
            "run oracle-contract before the model diagnostic"
        )
    oracle_summary = json.loads(summary_path.read_text())
    expected_hash = oracle_summary["executed_command_dataset_sha256"]
    if sha256_file(dataset_path) != expected_hash:
        raise ValueError("executed command dataset hash changed")
    with np.load(dataset_path) as data:
        trajectories = data["states"].astype(np.float32)
        commands = data["commands"].astype(np.float32)
        task_id = data["task_id"].astype(np.int64)
        command_index = data["command_index"].astype(np.int64)
        waypoint_index = data["waypoint_index"].astype(np.int64)
    starts = trajectories[:, 0]
    endpoints = trajectories[:, -1]
    targets = endpoints - starts
    inputs = consequence_inputs(starts, commands)

    checkpoint = paths(args.root, args.round)["consequence"]
    model = MDNEnsemble(checkpoint, args.device)
    oracle_member_nll = member_nll_rows(model, inputs, targets)
    oracle_nll = oracle_member_nll.mean(axis=1)
    mode_distance = nearest_mode_xy_distance(model, inputs, targets)
    predicted_stall = predicted_stall_probability(model, inputs, args.seed)
    observed_stall = (
        np.linalg.norm(targets[:, :2], axis=1) < STALL_RADIUS_M
    )

    round_dirs = [
        paths(args.root, index)["round"] for index in range(args.round + 1)
    ]
    reference_inputs, reference_targets = consequence_arrays(round_dirs)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    split_seed = int(payload["seed"])
    permutation = np.random.default_rng(split_seed).permutation(
        len(reference_inputs)
    )
    validation_count = max(
        1, int(round(VALIDATION_FRACTION * len(reference_inputs)))
    )
    validation_index = permutation[:validation_count]
    validation_member_nll = member_nll_rows(
        model,
        reference_inputs[validation_index],
        reference_targets[validation_index],
    )
    validation_nll = validation_member_nll.mean(axis=1)
    sorted_validation = np.sort(validation_nll)
    percentile = np.searchsorted(
        sorted_validation, oracle_nll, side="right"
    ) / len(sorted_validation)

    fixed_success_tasks = {
        int(row["task_id"])
        for row in oracle_summary["rows"] if row["success"]
    }
    original_success_mask = np.isin(task_id, ORIGINAL_ORACLE_SUCCESSES)
    fixed_success_mask = np.isin(task_id, sorted(fixed_success_tasks))
    subsets = [
        summarize_subset(
            "all_fixed_contract_commands", np.ones(len(task_id), bool),
            oracle_nll, percentile, mode_distance, predicted_stall,
            observed_stall,
        ),
        summarize_subset(
            "commands_on_original_oracle_success_tasks",
            original_success_mask, oracle_nll, percentile, mode_distance,
            predicted_stall, observed_stall,
        ),
        summarize_subset(
            "commands_on_fixed_contract_success_tasks",
            fixed_success_mask, oracle_nll, percentile, mode_distance,
            predicted_stall, observed_stall,
        ),
    ]
    command_rows = []
    for index in range(len(task_id)):
        command_rows.append({
            "task_id": int(task_id[index]),
            "command_index": int(command_index[index]),
            "waypoint_index": int(waypoint_index[index]),
            "command": commands[index].tolist(),
            "actual_endpoint_xy": endpoints[index, :2].tolist(),
            "actual_movement_m": float(
                np.linalg.norm(targets[index, :2])
            ),
            "normalized_nll": float(oracle_nll[index]),
            "validation_nll_percentile": float(percentile[index]),
            "nearest_mode_xy_distance_m": float(mode_distance[index]),
            "predicted_stall_probability": float(predicted_stall[index]),
            "observed_stall": bool(observed_stall[index]),
            "original_oracle_success_task": bool(
                original_success_mask[index]
            ),
            "fixed_contract_success_task": bool(fixed_success_mask[index]),
        })
    (output / "model_command_diagnostic.jsonl").write_text(
        "\n".join(json.dumps(row) for row in command_rows) + "\n"
    )
    result = {
        "method_revision": MODEL_DIAGNOSTIC_REVISION,
        "claim_level": "consumed_development_oracle_command_model_diagnostic",
        "source_contract_revision": oracle_summary["method_revision"],
        "source_dataset": str(dataset_path.resolve()),
        "source_dataset_sha256": expected_hash,
        "consequence_checkpoint": str(checkpoint.resolve()),
        "consequence_sha256": sha256_file(checkpoint),
        "commands": int(len(task_id)),
        "validation_reference_commands": int(len(validation_nll)),
        "validation_split": (
            "reconstructed_frozen_training_time_validation_indices"
        ),
        "validation_normalized_nll": {
            "median": float(np.median(validation_nll)),
            "p90": float(np.quantile(validation_nll, 0.9)),
            "p95": float(np.quantile(validation_nll, 0.95)),
        },
        "stall_definition": f"endpoint_xy_displacement_below_{STALL_RADIUS_M}m",
        "stall_probability_samples_per_ensemble_member": (
            STALL_SAMPLES_PER_MEMBER
        ),
        "subsets": subsets,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "interpretation_policy": (
            "descriptive_component_localization_without_a_post_hoc_pass_threshold"
        ),
    }
    dump(output / "model_diagnostic.json", result)
    print(json.dumps(result, indent=2))


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=707)
    subparsers = parser.add_subparsers(dest="command", required=True)
    oracle = subparsers.add_parser("oracle-contract")
    oracle.set_defaults(func=command_oracle_contract)
    diagnose = subparsers.add_parser("model-diagnostic")
    diagnose.add_argument("--round", type=int, default=2)
    diagnose.set_defaults(func=command_model_diagnostic)
    return parser


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
