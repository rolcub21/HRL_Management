"""Train the initial PointMaze factorized consequence ensemble."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
    consequence_features,
    train_head,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VALIDATION_FRACTION,
)

from .core import atomic_json, atomic_torch, protocol, sha256
from .dataset import grouped_split, load_initial_dataset


METHOD_REVISION = "pointmaze_initial_factorized_consequence_r1"
CLAIM_LEVEL = "reward_free_initial_model_preparation_only"
SPLIT_SEED = 731
TRAINING_SEED = 881


def endpoint_metrics(model, dataset, index):
    states = dataset.starts[index]
    commands = dataset.commands[index]
    actual = dataset.endpoints[index, :2]
    features = consequence_features(states, commands)
    member_endpoint = []
    member_sigma = []
    for member in range(len(model.spatial_members)):
        weight, residual, std = model.spatial_parameters(features, member)
        member_endpoint.append(
            commands + (weight[..., None] * residual).sum(axis=1)
        )
        radial = np.sqrt(0.5 * np.square(std).sum(axis=2))
        member_sigma.append((weight * radial).sum(axis=1))
    predicted = np.stack(member_endpoint).mean(axis=0)
    error = np.linalg.norm(predicted - actual, axis=1)
    return {
        "endpoint_error_mean_m": float(error.mean()),
        "endpoint_error_median_m": float(np.median(error)),
        "endpoint_error_p90_m": float(np.quantile(error, 0.9)),
        "landing_precision_m": float(np.stack(member_sigma).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schedule", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    contract = protocol()
    schedule = args.schedule
    epochs = EPOCHS if schedule == "full" else 2
    max_groups = None if schedule == "full" else 3
    dataset = load_initial_dataset(max_groups_per_source=max_groups)
    train_index, validation_index = grouped_split(
        dataset, SPLIT_SEED, VALIDATION_FRACTION
    )
    features = consequence_features(dataset.starts, dataset.commands)
    spatial_target = dataset.endpoints[:, :2] - dataset.commands
    posture_features = np.concatenate(
        (features, dataset.endpoints[:, :2]), axis=1
    ).astype(np.float32)
    posture_target = dataset.endpoints[:, 2:]
    started = time.time()
    spatial = train_head(
        features,
        spatial_target,
        train_index,
        validation_index,
        TRAINING_SEED + 1,
        args.device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        label="pointmaze_spatial",
    )
    posture = train_head(
        posture_features,
        posture_target,
        train_index,
        validation_index,
        TRAINING_SEED + 2,
        args.device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        label="pointmaze_velocity",
    )
    output = args.output.resolve()
    checkpoint = output / "factorized_consequence.pt"
    payload = {
        "artifact_type": "factorized_consequence",
        "schema_version": 1,
        "domain": "PointMaze_Medium-v3",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL if schedule == "full" else "smoke_only",
        "method_reference": str(
            Path(contract["method_reference"]).name
        ),
        "state_dim": 4,
        "position_dim": 2,
        "posture_semantics": "endpoint_velocity_xy",
        "spatial_parameterization": "endpoint_xy_minus_command_xy",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_velocity|state,command,endpoint_xy)"
        ),
        "ensemble_bootstrap": True,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "split": "whole_reward_free_trajectory_groups_per_source",
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "source_names": list(dataset.source_names),
        "source_paths": [str(path) for path in dataset.source_paths],
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
        "spatial_histories": spatial["histories"],
        "posture_histories": posture["histories"],
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    atomic_torch(checkpoint, payload)
    model = FactorizedConsequenceEnsemble(checkpoint, args.device)
    validation = endpoint_metrics(model, dataset, validation_index)
    per_source = {}
    validation_mask = np.zeros(dataset.rows, bool)
    validation_mask[validation_index] = True
    for source_id, name in enumerate(dataset.source_names):
        index = np.flatnonzero(
            validation_mask & (dataset.source_id == source_id)
        )
        per_source[name] = endpoint_metrics(model, dataset, index)
    report = {
        "schema_version": 1,
        "artifact": str(checkpoint),
        "artifact_sha256": sha256(checkpoint),
        "method_revision": METHOD_REVISION,
        "claim_level": payload["claim_level"],
        "schedule": schedule,
        "rows": dataset.rows,
        "training_rows": len(train_index),
        "validation_rows": len(validation_index),
        "source_names": list(dataset.source_names),
        "validation": validation,
        "validation_by_source": per_source,
        "kernel_sigma_m": validation["landing_precision_m"],
        "kernel_sigma_estimator": (
            "held_out_ensemble_mean_mixture_weighted_within_mode_spatial_spread"
        ),
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(output / "factorized_consequence_metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
