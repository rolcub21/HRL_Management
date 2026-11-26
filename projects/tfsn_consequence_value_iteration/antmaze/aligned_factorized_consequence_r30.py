"""Refit the unchanged factorized consequence model on R29 behavior data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.antmaze.current_tfsn.constants import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.main import paths

from projects.tfsn_consequence_value_iteration.antmaze import r29_alignment
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    _train_head,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
    sha256_file,
)
from projects.tfsn_consequence_value_iteration.shared.factorized import (
    diagonal_gaussian_circle_probability,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r30_aligned_consequence_protocol.json"
METHOD_REFERENCE = HERE / "R30_ALIGNED_CONSEQUENCE_REFERENCE.md"
METHOD_REVISION = "antmaze_r29_aligned_factorized_consequence_r30"


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_protocol():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "oracle_inputs": [],
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R30 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R30 method reference changed")
    return value


def alignment_root(protocol):
    root = Path(protocol["alignment_source"])
    if not root.is_absolute():
        root = MANAGEMENT_ROOT / root
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("method_revision") != protocol["alignment_source_revision"]:
        raise ValueError("R29 alignment source revision changed")
    if summary.get("profile") != "full" or int(summary.get("rows", -1)) != 6000:
        raise ValueError("R29 alignment source is incomplete")
    if summary.get("hidden_goals_accessed") is not False:
        raise ValueError("R29 alignment source accessed hidden goals")
    return root, summary


def load_alignment(protocol, profile):
    root, summary = alignment_root(protocol)
    tasks = int(protocol["alignment_tasks"])
    commands_per_task = int(summary["commands_per_preparation_task"])
    if profile == "smoke":
        tasks = 4
    shards = []
    for task_id in range(tasks):
        path = root / "shards" / f"task_{task_id:04d}.npz"
        shards.append(r29_alignment.validate_shard(
            path, task_id, commands_per_task
        ))
    states = np.concatenate([row["states"][:, 0] for row in shards])
    commands = np.concatenate([row["commands"] for row in shards])
    endpoints = np.concatenate([row["states"][:, -1] for row in shards])
    task_id = np.concatenate([row["task_id"] for row in shards])
    distance = np.concatenate([row["selected_command_distance"] for row in shards])
    return states, commands, endpoints, task_id, distance, root


def split_tasks(protocol, available_tasks, profile):
    rng = np.random.default_rng(int(protocol["split_seed"]))
    order = rng.permutation(int(available_tasks))
    held_count = (
        1 if profile == "smoke"
        else int(protocol["held_out_alignment_tasks"])
    )
    held = np.sort(order[:held_count]).astype(np.int16)
    trained = np.sort(order[held_count:]).astype(np.int16)
    if len(trained) == 0 or len(held) == 0:
        raise ValueError("R30 split must contain training and held-out tasks")
    return trained, held


def connection_probability(model, states, commands, centers, radius, batch=4096):
    member_rows = [[] for _ in model.members]
    for begin in range(0, len(states), batch):
        stop = min(begin + batch, len(states))
        inputs = consequence_inputs(states[begin:stop], commands[begin:stop])
        for member in model.members:
            weight, residual, scale = model.spatial_parameters(inputs, member)
            means = commands[begin:stop, None, :] + residual
            mass = diagonal_gaussian_circle_probability(
                means, scale, centers[begin:stop], radius
            )
            member_rows[member].append((weight * mass).sum(axis=1))
    return np.stack(
        [np.concatenate(rows) for rows in member_rows], axis=1
    ).astype(np.float32)


def landing_precision(model, states, commands, batch=4096):
    values = []
    for begin in range(0, len(states), batch):
        stop = min(begin + batch, len(states))
        inputs = consequence_inputs(states[begin:stop], commands[begin:stop])
        member_values = []
        for member in model.members:
            weight, _residual, scale = model.spatial_parameters(inputs, member)
            radial = np.sqrt(0.5 * np.square(scale).sum(axis=2))
            member_values.append((weight * radial).sum(axis=1))
        values.append(np.stack(member_values, axis=1))
    values = np.concatenate(values)
    return float(values.mean()), values


def calibration_rows(predicted, observed, bins=10):
    rows = []
    for low in np.linspace(0.0, 1.0, bins, endpoint=False):
        high = low + 1.0 / bins
        mask = (predicted >= low) & (
            predicted <= high if high >= 1.0 else predicted < high
        )
        if mask.any():
            rows.append({
                "low": float(low),
                "high": float(high),
                "count": int(mask.sum()),
                "predicted": float(predicted[mask].mean()),
                "observed": float(observed[mask].mean()),
            })
    return rows


def subset(name, mask, probabilities, observed):
    mask = np.asarray(mask, bool)
    mean_prediction = probabilities.mean(axis=1)
    pessimistic = probabilities.min(axis=1)
    return {
        "name": name,
        "commands": int(mask.sum()),
        "observed_connection_rate": float(observed[mask].mean()),
        "ensemble_mean_prediction": float(mean_prediction[mask].mean()),
        "ensemble_mean_bias": float(
            (mean_prediction[mask] - observed[mask]).mean()
        ),
        "ensemble_mean_brier": float(
            np.square(mean_prediction[mask] - observed[mask]).mean()
        ),
        "pessimistic_prediction": float(pessimistic[mask].mean()),
        "pessimistic_bias": float(
            (pessimistic[mask] - observed[mask]).mean()
        ),
        "pessimistic_brier": float(
            np.square(pessimistic[mask] - observed[mask]).mean()
        ),
        "member_predictions": probabilities[mask].mean(axis=0).tolist(),
        "ensemble_mean_calibration": calibration_rows(
            mean_prediction[mask], observed[mask]
        ),
    }


def evaluate_model(model, states, commands, endpoints, distance, protocol):
    radius = float(protocol["frozen_comparison_sigma_m"])
    observed = (
        np.linalg.norm(endpoints[:, :2] - commands, axis=1) <= radius
    ).astype(np.float32)
    probability = connection_probability(
        model, states, commands, commands, radius
    )
    long_mask = distance >= float(
        protocol["long_command_definition_m_diagnostic_only"]
    )
    sigma, sigma_rows = landing_precision(model, states, commands)
    return {
        "matched_connection_radius_m": radius,
        "landing_precision_m_current_specification": sigma,
        "landing_precision_rows": int(len(sigma_rows)),
        "subsets": [
            subset("all_held_out_R29_commands", np.ones(len(states), bool), probability, observed),
            subset("held_out_R29_long_commands_diagnostic_only", long_mask, probability, observed),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    states, commands, endpoints, task_ids, distance, source = load_alignment(
        protocol, args.profile
    )
    available_tasks = len(np.unique(task_ids))
    trained_tasks, held_tasks = split_tasks(protocol, available_tasks, args.profile)
    alignment_train = np.isin(task_ids, trained_tasks)
    alignment_held = np.isin(task_ids, held_tasks)

    round_dirs = [paths(args.root, index)["round"] for index in range(3)]
    old_starts, old_commands, old_endpoints = command_records(round_dirs)
    if args.profile == "smoke":
        old_starts = old_starts[:1000]
        old_commands = old_commands[:1000]
        old_endpoints = old_endpoints[:1000]
    train_states = np.concatenate((old_starts, states[alignment_train]))
    train_commands = np.concatenate((old_commands, commands[alignment_train]))
    train_endpoints = np.concatenate((old_endpoints, endpoints[alignment_train]))
    validation_states = states[alignment_held]
    validation_commands = commands[alignment_held]
    validation_endpoints = endpoints[alignment_held]
    validation_distance = distance[alignment_held]

    combined_states = np.concatenate((train_states, validation_states))
    combined_commands = np.concatenate((train_commands, validation_commands))
    combined_endpoints = np.concatenate((train_endpoints, validation_endpoints))
    training_index = np.arange(len(train_states), dtype=np.int64)
    validation_index = np.arange(
        len(train_states), len(combined_states), dtype=np.int64
    )
    inputs = consequence_inputs(combined_states, combined_commands)
    spatial_target = combined_endpoints[:, :2] - combined_commands
    posture_input = np.concatenate((inputs, combined_endpoints[:, :2]), axis=1)
    posture_target = combined_endpoints[:, 2:]
    schedule = protocol["training_schedule"]
    epochs = 2 if args.profile == "smoke" else int(schedule["epochs"])
    spatial = _train_head(
        inputs, spatial_target, training_index, validation_index,
        int(protocol["split_seed"]) + 801, args.device, epochs,
        int(schedule["batch_size"]), float(schedule["learning_rate"]), "R30 spatial",
    )
    posture = _train_head(
        posture_input, posture_target, training_index, validation_index,
        int(protocol["split_seed"]) + 802, args.device, epochs,
        int(schedule["batch_size"]), float(schedule["learning_rate"]), "R30 posture",
    )

    output = args.output.resolve()
    checkpoint = output / "consequence.pt"
    payload = {
        "artifact_type": "factorized_ant_consequence",
        "method_revision": METHOD_REVISION,
        "claim_level": (
            protocol["claim_level"] if args.profile == "full" else "smoke_only"
        ),
        "spatial_parameterization": "endpoint_xy_minus_command",
        "posture_parameterization": "absolute_endpoint_posture",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_posture|state,command,endpoint_xy)"
        ),
        "head_optimization": "unchanged_R8_independent_losses_and_model_selection",
        "training_source": [str(path.resolve()) for path in round_dirs] + [str(source)],
        "training_commands": int(len(training_index)),
        "validation_commands": int(len(validation_index)),
        "split_seed": int(protocol["split_seed"]),
        "split": "whole_R29_preparation_tasks_held_out",
        "held_out_alignment_tasks": held_tasks.tolist(),
        "oracle_command_dataset_used_for_training_or_selection": False,
        "base_input_mean": spatial["input_mean"],
        "base_input_std": spatial["input_std"],
        "spatial_target_mean": spatial["target_mean"],
        "spatial_target_std": spatial["target_std"],
        "posture_input_mean": posture["input_mean"],
        "posture_input_std": posture["input_std"],
        "posture_target_mean": posture["target_mean"],
        "posture_target_std": posture["target_std"],
        "spatial_config": spatial["config"],
        "posture_config": posture["config"],
        "spatial_member_states": spatial["member_states"],
        "posture_member_states": posture["member_states"],
        "epochs": epochs,
        "batch_size": int(schedule["batch_size"]),
        "learning_rate": float(schedule["learning_rate"]),
    }
    atomic_torch(checkpoint, payload)
    aligned = FactorizedConsequenceEnsemble(checkpoint, args.device)
    old_checkpoint = (
        paths(args.root)["models"] / "factorized_consequence_r8" / "consequence.pt"
    )
    old = FactorizedConsequenceEnsemble(old_checkpoint, args.device)
    old_metrics = evaluate_model(
        old, validation_states, validation_commands, validation_endpoints,
        validation_distance, protocol,
    )
    aligned_metrics = evaluate_model(
        aligned, validation_states, validation_commands, validation_endpoints,
        validation_distance, protocol,
    )
    report = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": sha256_file(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": payload["claim_level"],
        "declared_deviations": protocol["declared_deviations"],
        "oracle_inputs": [],
        "profile": args.profile,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "old_checkpoint_sha256": sha256_file(old_checkpoint),
        "training_commands": int(len(training_index)),
        "validation_commands": int(len(validation_index)),
        "trained_alignment_tasks": trained_tasks.tolist(),
        "held_out_alignment_tasks": held_tasks.tolist(),
        "old_model_on_same_holdout": old_metrics,
        "aligned_model_on_same_holdout": aligned_metrics,
        "spatial_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"] for row in spatial["histories"]
        ],
        "posture_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"] for row in posture["histories"]
        ],
    }
    atomic_json(output / "summary.json", report)
    atomic_json(output / "training_history.json", {
        "spatial": spatial["histories"], "posture": posture["histories"]
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

