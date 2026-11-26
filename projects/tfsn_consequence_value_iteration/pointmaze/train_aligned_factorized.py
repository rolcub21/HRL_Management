"""Refit PointMaze consequence models on current-planner alignment data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
    consequence_features,
    train_head,
)
from projects.tfsn_consequence_value_iteration.shared.density import (
    situation_density_weights,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VALIDATION_FRACTION,
    SITUATION_DENSITY_TRUNCATION_SIGMAS,
)

from .core import atomic_json, atomic_torch, load_json, protocol, sha256
from .dataset import grouped_split, load_current_aligned_dataset
from .train_factorized import endpoint_metrics


HERE = Path(__file__).resolve().parent
ALIGNED_PROTOCOL = HERE / "aligned_factorized_protocol.json"
METHOD_REVISION = "pointmaze_current_planner_aligned_factorized_r3"
DENSITY_METHOD_REVISION = "pointmaze_situation_density_aligned_factorized_r6"
CLAIM_LEVEL = "reward_free_current_planner_aligned_model_preparation_only"
SPLIT_SEED = 731
TRAINING_SEED = 881


def load_contract(alignment_root, base_models):
    value = load_json(ALIGNED_PROTOCOL)
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": "pointmaze_initial_factorized_consequence_r1",
        "alignment_method_revision": "pointmaze_current_planner_alignment_r2",
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [],
        "oracle_inputs": [],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "confirmation_panel_accessed": False,
        "alignment_round": 1,
        "maximum_alignment_rounds": 4,
        "alignment_relative_tolerance": 0.1,
        "alignment_rows": 12000,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"aligned factorized protocol has another {name}")
    summaries = {
        "alignment_acquisition_summary_sha256": (
            Path(alignment_root) / "summary_acquisition_full.json"
        ),
        "alignment_return_summary_sha256": (
            Path(alignment_root) / "summary_return_full.json"
        ),
        "base_checkpoint_sha256": (
            Path(base_models) / "factorized_consequence.pt"
        ),
    }
    for name, path in summaries.items():
        if sha256(path) != value[name]:
            raise ValueError(f"aligned factorized source changed: {path}")
    return value


def _train(
    dataset,
    train_index,
    validation_index,
    device,
    epochs,
    sample_weight=None,
):
    features = consequence_features(dataset.starts, dataset.commands)
    spatial_target = dataset.endpoints[:, :2] - dataset.commands
    posture_features = np.concatenate(
        (features, dataset.endpoints[:, :2]), axis=1
    ).astype(np.float32)
    posture_target = dataset.endpoints[:, 2:]
    spatial = train_head(
        features,
        spatial_target,
        train_index,
        validation_index,
        TRAINING_SEED + 1,
        device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        label="pointmaze_aligned_spatial",
        sample_weight=sample_weight,
    )
    posture = train_head(
        posture_features,
        posture_target,
        train_index,
        validation_index,
        TRAINING_SEED + 2,
        device,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        label="pointmaze_aligned_velocity",
        sample_weight=sample_weight,
    )
    return spatial, posture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--base-models", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schedule", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--situation-density-weighting",
        action="store_true",
        help=(
            "Correct repeated near-identical state/command situations using "
            "a reward-free inverse-density bootstrap."
        ),
    )
    args = parser.parse_args()
    started = time.time()
    method = protocol()
    aligned_contract = load_contract(args.alignment, args.base_models)
    epochs = EPOCHS if args.schedule == "full" else 2
    max_groups = None if args.schedule == "full" else 3
    dataset = load_current_aligned_dataset(
        args.alignment, max_groups_per_source=max_groups
    )
    train_index, validation_index = grouped_split(
        dataset, SPLIT_SEED, VALIDATION_FRACTION
    )
    density_report = None
    sample_weight = None
    if args.situation_density_weighting:
        base_metrics = load_json(
            Path(args.base_models) / "factorized_consequence_metrics.json"
        )
        sample_weight, density_report = situation_density_weights(
            dataset.starts,
            dataset.commands,
            train_index,
            float(base_metrics["kernel_sigma_m"]),
            SITUATION_DENSITY_TRUNCATION_SIGMAS,
        )
    spatial, posture = _train(
        dataset,
        train_index,
        validation_index,
        args.device,
        epochs,
        sample_weight=sample_weight,
    )
    output = args.output.resolve()
    checkpoint = output / "factorized_consequence.pt"
    payload = {
        "artifact_type": "factorized_consequence",
        "schema_version": 1,
        "domain": "PointMaze_Medium-v3",
        "method_revision": (
            DENSITY_METHOD_REVISION
            if args.situation_density_weighting else METHOD_REVISION
        ),
        "claim_level": CLAIM_LEVEL if args.schedule == "full" else "smoke_only",
        "method_reference": method["method_reference"],
        "alignment_protocol": str(ALIGNED_PROTOCOL),
        "alignment_protocol_sha256": sha256(ALIGNED_PROTOCOL),
        "alignment_round": aligned_contract["alignment_round"],
        "state_dim": 4,
        "position_dim": 2,
        "posture_semantics": "endpoint_velocity_xy",
        "spatial_parameterization": "endpoint_xy_minus_command_xy",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_velocity|state,command,endpoint_xy)"
        ),
        "ensemble_bootstrap": True,
        "ensemble_bootstrap_distribution": (
            "inverse_local_state_command_density"
            if args.situation_density_weighting else "uniform_rows"
        ),
        "situation_density_correction": density_report,
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
    validation_mask = np.zeros(dataset.rows, bool)
    validation_mask[validation_index] = True
    per_source = {}
    current_validation = []
    current_alignment_rows = 0
    for source_id, name in enumerate(dataset.source_names):
        index = np.flatnonzero(
            validation_mask & (dataset.source_id == source_id)
        )
        per_source[name] = endpoint_metrics(model, dataset, index)
        if name.startswith("current_"):
            current_validation.extend(index.tolist())
            current_alignment_rows += int(np.sum(dataset.source_id == source_id))
    current_validation = np.asarray(current_validation, np.int64)
    if not len(current_validation):
        raise ValueError("current alignment holdout is empty")
    base_checkpoint = Path(args.base_models) / "factorized_consequence.pt"
    base_model = FactorizedConsequenceEnsemble(base_checkpoint, args.device)
    base_on_current = endpoint_metrics(
        base_model, dataset, current_validation
    )
    aligned_on_current = endpoint_metrics(
        model, dataset, current_validation
    )
    base_error = float(base_on_current["endpoint_error_mean_m"])
    aligned_error = float(aligned_on_current["endpoint_error_mean_m"])
    relative_change = abs(aligned_error - base_error) / max(base_error, 1e-12)
    tolerance = float(aligned_contract["alignment_relative_tolerance"])
    stop = relative_change < tolerance
    report = {
        "schema_version": 1,
        "artifact": str(checkpoint),
        "artifact_sha256": sha256(checkpoint),
        "method_revision": payload["method_revision"],
        "claim_level": payload["claim_level"],
        "schedule": args.schedule,
        "alignment_round": aligned_contract["alignment_round"],
        "rows": dataset.rows,
        "training_rows": len(train_index),
        "validation_rows": len(validation_index),
        "current_alignment_rows": current_alignment_rows,
        "current_alignment_validation_rows": int(len(current_validation)),
        "source_names": list(dataset.source_names),
        "situation_density_correction": density_report,
        "validation": validation,
        "validation_by_source": per_source,
        "base_model_on_current_alignment_holdout": base_on_current,
        "aligned_model_on_current_alignment_holdout": aligned_on_current,
        "newest_round_endpoint_error_relative_change": relative_change,
        "alignment_relative_tolerance": tolerance,
        "stop_alignment": stop,
        "next_action": (
            "run_frozen_oracle_movement_calibration"
            if args.situation_density_weighting
            else (
                "freeze_aligned_consequence_and_rebuild_roadmap"
                if stop else "collect_next_reward_free_alignment_round"
            )
        ),
        "kernel_sigma_m": validation["landing_precision_m"],
        "kernel_sigma_estimator": (
            "held_out_ensemble_mean_mixture_weighted_within_mode_spatial_spread"
        ),
        "alignment_acquisition_summary_sha256": aligned_contract[
            "alignment_acquisition_summary_sha256"
        ],
        "alignment_return_summary_sha256": aligned_contract[
            "alignment_return_summary_sha256"
        ],
        "base_checkpoint_sha256": aligned_contract["base_checkpoint_sha256"],
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
        "confirmation_panel_accessed": False,
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(output / "factorized_consequence_metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
