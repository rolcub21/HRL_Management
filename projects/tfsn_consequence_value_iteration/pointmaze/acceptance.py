"""Part C.2 consequence acceptance on fixed-tau oracle commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
    consequence_features,
    diagonal_gaussian_circle_probability,
)
from .core import atomic_json, load_json, protocol, sha256


def calibration_bins(predicted, observed, count=10, sample_weight=None):
    predicted = np.asarray(predicted, np.float64)
    observed = np.asarray(observed, np.float64)
    if sample_weight is None:
        sample_weight = np.ones(len(predicted), np.float64)
    sample_weight = np.asarray(sample_weight, np.float64)
    rows = []
    for low in np.linspace(0.0, 1.0, count, endpoint=False):
        high = low + 1.0 / count
        mask = (predicted >= low) & (
            predicted <= high if high >= 1.0 else predicted < high
        )
        if mask.any():
            local = sample_weight[mask]
            if float(local.sum()) <= 0.0:
                continue
            rows.append({
                "low": float(low),
                "high": float(high),
                "count": int(mask.sum()),
                "effective_weight": float(local.sum()),
                "predicted": float(np.average(predicted[mask], weights=local)),
                "observed_kernel_outcome": float(
                    np.average(observed[mask], weights=local)
                ),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    contract = protocol()
    references = load_json(args.references)
    if references.get("method_revision") != contract["method_revision"]:
        raise ValueError("oracle references use another method contract")
    traces = []
    for episode in references["episodes"]:
        if episode["system"] == "fixed_tau_geometry_oracle":
            traces.extend(episode["command_trace"])
    if not traces:
        raise ValueError("oracle reference contains no executed commands")
    if any("start_state" not in row or "endpoint_state" not in row for row in traces):
        raise ValueError("rerun references: oracle traces predate full-state logging")

    states = np.asarray([row["start_state"] for row in traces], np.float32)
    commands = np.asarray([row["command_xy"] for row in traces], np.float32)
    actual = np.asarray([row["endpoint_state"] for row in traces], np.float32)
    metrics = load_json(args.models / "factorized_consequence_metrics.json")
    checkpoint = Path(metrics["artifact"])
    if sha256(checkpoint) != metrics["artifact_sha256"]:
        raise ValueError("factorized consequence checkpoint changed")
    if metrics.get("schedule") != "full":
        raise ValueError("acceptance requires the full consequence schedule")
    sigma = float(metrics["kernel_sigma_m"])
    radius = float(contract["evaluator"]["success_radius_m_evaluator_only"])
    model = FactorizedConsequenceEnsemble(checkpoint, args.device)
    features = consequence_features(states, commands)
    mass, stall, nearest_mode, expected_error = [], [], [], []
    for member in range(len(model.spatial_members)):
        weight, residual, std = model.spatial_parameters(features, member)
        means = commands[:, None, :] + residual
        component_mass = diagonal_gaussian_circle_probability(
            means, std, actual[:, :2], radius
        )
        mass.append((weight * component_mass).sum(axis=1))
        stall.append(model.stall_probability(states, commands, sigma, member))
        nearest_mode.append(
            np.linalg.norm(means - actual[:, None, :2], axis=2).min(axis=1)
        )
        expected = (weight[..., None] * means).sum(axis=1)
        expected_error.append(np.linalg.norm(expected - actual[:, :2], axis=1))
    mass = np.stack(mass, axis=1)
    stall = np.stack(stall, axis=1)
    nearest_mode = np.stack(nearest_mode, axis=1)
    expected_error = np.stack(expected_error, axis=1)
    predicted_stall = stall.mean(axis=1)
    observed_stall = np.exp(
        -np.square(actual[:, :2] - states[:, :2]).sum(axis=1)
        / (2.0 * sigma**2)
    )
    averaged_mass = mass.mean(axis=1)
    command_distance = np.linalg.norm(commands - states[:, :2], axis=1)
    request_weight = 1.0 - np.exp(
        -np.square(command_distance) / (2.0 * sigma**2)
    )
    if float(request_weight.sum()) <= 0.0:
        raise ValueError("oracle acceptance has zero requested movement mass")

    def weighted_mean(value):
        return float(np.average(np.asarray(value), weights=request_weight))

    report = {
        "schema_version": 1,
        "experiment_id": (
            f"{metrics['method_revision']}_frozen_oracle_acceptance"
        ),
        "method_reference": contract["method_reference"],
        "method_revision": contract["method_revision"],
        "claim_level": "development_component_acceptance_only",
        "declared_deviations": [],
        "oracle_inputs": [
            "supplied_goal_coordinate",
            "native_maze_geometry_for_oracle_commands",
            "recorded_oracle_command_outcomes_for_component_evaluation",
        ],
        "commands": len(traces),
        "factorized_consequence_sha256": metrics["artifact_sha256"],
        "factorized_consequence_method_revision": metrics["method_revision"],
        "landing_sigma_m": sigma,
        "endpoint_mass_radius_m_evaluator_only": radius,
        "endpoint_mass_near_actual_mean": float(averaged_mass.mean()),
        "endpoint_mass_near_actual_median": float(np.median(averaged_mass)),
        "endpoint_mass_near_actual_p10": float(np.quantile(averaged_mass, 0.1)),
        "nearest_mode_error_median_m": float(np.median(nearest_mode.mean(axis=1))),
        "ensemble_expected_endpoint_error_mean_m": float(
            expected_error.mean(axis=1).mean()
        ),
        "predicted_stall_kernel_mean": float(predicted_stall.mean()),
        "observed_stall_kernel_mean": float(observed_stall.mean()),
        "stall_kernel_bias": float((predicted_stall - observed_stall).mean()),
        "stall_kernel_mae": float(np.abs(predicted_stall - observed_stall).mean()),
        "stall_kernel_brier": float(np.square(predicted_stall - observed_stall).mean()),
        "stall_calibration_bins": calibration_bins(predicted_stall, observed_stall),
        "movement_request_definition": "continuous_one_minus_landing_kernel",
        "movement_request_threshold": None,
        "movement_request_effective_weight": float(request_weight.sum()),
        "movement_predicted_stall_kernel_mean": float(
            weighted_mean(predicted_stall)
        ),
        "movement_observed_stall_kernel_mean": float(
            weighted_mean(observed_stall)
        ),
        "movement_stall_kernel_bias": float(
            weighted_mean(predicted_stall - observed_stall)
        ),
        "movement_stall_kernel_mae": float(
            weighted_mean(np.abs(predicted_stall - observed_stall))
        ),
        "movement_stall_kernel_brier": float(
            weighted_mean(np.square(predicted_stall - observed_stall))
        ),
        "movement_stall_calibration_bins": calibration_bins(
            predicted_stall, observed_stall, sample_weight=request_weight
        ),
        "endpoint_majority_mass_check": bool(averaged_mass.mean() > 0.5),
        "automatic_stall_gate": None,
        "automatic_stall_gate_reason": (
            "the_method_spec_requires_calibration_but_declares_no_numeric_"
            "tolerance; report_bias_MAE_Brier_without_inventing_one"
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
