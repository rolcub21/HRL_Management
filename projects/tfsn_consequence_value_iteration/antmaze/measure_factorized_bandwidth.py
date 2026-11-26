"""Measure the deployed factorized R8 model's data-derived field bandwidth.

The estimator matches the R17 bandwidth definition: on the frozen held-out
split, compute the posterior responsibility of every spatial mixture mode,
then take the median across commands of the posterior-expected distance from
the realized endpoint to its mode mean.  R8 deploys all three members, so the
posterior uses a uniform member prior rather than selecting one member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.antmaze.current_tfsn.constants import LANDING_SIGMA
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.main import paths

from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    _spatial_distribution,
    consequence_inputs,
    factorized_checkpoint,
)


METHOD_REVISION = "antmaze_factorized_r8_bandwidth_measurement_r1"
CLAIM_LEVEL = "held_out_deployed_model_bandwidth_measurement"
ESTIMATOR = "held_out_median_posterior_expected_within_mode_xy_error"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def posterior_expected_error(weights, means, scales, targets):
    """Return per-row posterior-expected distance to spatial mode means."""
    weights = np.asarray(weights, np.float64)
    means = np.asarray(means, np.float64)
    scales = np.asarray(scales, np.float64)
    targets = np.asarray(targets, np.float64)
    if means.shape != scales.shape or means.shape[:2] != weights.shape:
        raise ValueError("spatial mixture arrays do not align")
    if targets.shape != (len(weights), 2) or means.shape[2] != 2:
        raise ValueError("spatial targets must be two-dimensional")
    if np.any(weights < 0) or np.any(scales <= 0):
        raise ValueError("mixture weights and scales must be valid")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("joint ensemble-mixture weights must sum to one")

    difference = (targets[:, None] - means) / scales
    joint = (
        np.log(np.maximum(weights, 1e-300))
        - np.log(scales).sum(axis=2)
        - math.log(2.0 * math.pi)
        - 0.5 * np.square(difference).sum(axis=2)
    )
    joint -= joint.max(axis=1, keepdims=True)
    responsibility = np.exp(joint)
    responsibility /= responsibility.sum(axis=1, keepdims=True)
    error = np.linalg.norm(means - targets[:, None], axis=2)
    return (responsibility * error).sum(axis=1), responsibility, error


def measure(root, device):
    root = Path(root)
    checkpoint = factorized_checkpoint(root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    round_directories = [paths(root, index)["round"] for index in range(3)]
    starts, commands, endpoints = command_records(round_directories)
    inputs = consequence_inputs(starts, commands)

    split_seed = int(payload["split_seed"])
    permutation = np.random.default_rng(split_seed).permutation(len(inputs))
    validation_count = max(1, int(round(0.10 * len(inputs))))
    validation_index = permutation[:validation_count]

    model = FactorizedConsequenceEnsemble(checkpoint, device)
    weights, means, scales = _spatial_distribution(
        model, inputs[validation_index]
    )
    target = endpoints[validation_index, :2]
    expected_error, responsibility, error = posterior_expected_error(
        weights, means, scales, target
    )

    member_count = len(model.members)
    components = int(weights.shape[1] // member_count)
    member_metrics = []
    for member in range(member_count):
        begin = member * components
        stop = begin + components
        member_weights = weights[:, begin:stop]
        member_weights = member_weights / member_weights.sum(
            axis=1, keepdims=True
        )
        member_error, _member_responsibility, _member_distance = (
            posterior_expected_error(
                member_weights,
                means[:, begin:stop],
                scales[:, begin:stop],
                target,
            )
        )
        member_metrics.append({
            "member": member,
            "within_mode_xy_error_mean_m": float(member_error.mean()),
            "within_mode_xy_error_median_m": float(
                np.median(member_error)
            ),
        })

    sigma = float(np.median(expected_error))
    return {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [],
        "oracle_inputs": [],
        "deployed_model_revision": payload["method_revision"],
        "deployed_checkpoint": str(checkpoint.resolve()),
        "deployed_checkpoint_sha256": sha256(checkpoint),
        "training_sources": [
            str(directory.resolve()) for directory in round_directories
        ],
        "training_commands": int(len(inputs) - validation_count),
        "validation_commands": int(validation_count),
        "split_seed": split_seed,
        "split": payload["split"],
        "ensemble_members": member_count,
        "spatial_components_per_member": components,
        "ensemble_prior": "uniform",
        "endpoint_dispersion_estimator": ESTIMATOR,
        "endpoint_dispersion_m": sigma,
        "within_mode_xy_error_mean_m": float(expected_error.mean()),
        "within_mode_xy_error_rms_m": float(np.sqrt(
            (responsibility * np.square(error)).sum(axis=1).mean()
        )),
        "closest_mode_xy_error_mean_m": float(error.min(axis=1).mean()),
        "closest_mode_xy_error_median_m": float(
            np.median(error.min(axis=1))
        ),
        "member_metrics": member_metrics,
        "current_runtime_sigma_m": float(LANDING_SIGMA),
        "deployed_to_current_sigma_ratio": float(sigma / LANDING_SIGMA),
        "current_to_deployed_sigma_ratio": float(LANDING_SIGMA / sigma),
        "task_reward_used": False,
        "task_gradient_updates": 0,
    }


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cpu")
    value.add_argument(
        "--output",
        type=Path,
        default=Path(
            "projects/tfsn_consequence_value_iteration/reports/"
            "antmaze_factorized_r8_bandwidth.json"
        ),
    )
    return value


def main():
    args = parser().parse_args()
    result = measure(args.root, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
