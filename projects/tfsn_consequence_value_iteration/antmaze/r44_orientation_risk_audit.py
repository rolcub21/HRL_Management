"""Test R43 on the exact deterministic command that flipped the Ant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r42_posture_risk_audit as r42,
)
from projects.tfsn_consequence_value_iteration.antmaze.orientation_factorized_consequence_r43 import (
    OUTPUT as R43_OUTPUT,
    OrientationFactorizedConsequenceEnsemble,
    rotation6d_to_matrix,
    upright_score,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    MobilityModel,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R44_ORIENTATION_RISK_AUDIT_REFERENCE.md"
PROTOCOL = HERE / "r44_orientation_risk_protocol.json"
METHOD_REVISION = "antmaze_r43_orientation_risk_audit_r44"
R43_CHECKPOINT = R43_OUTPUT / "consequence.pt"
R43_SUMMARY = R43_OUTPUT / "summary.json"
R42_SUMMARY = r42.OUTPUT / "summary.json"
OUTPUT = r42.R40 / "task3_orientation_risk_audit_r44"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def protocol():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_orientation_model_diagnostic_only",
        "task_reward_used": False,
        "hidden_goal_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R44 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE:
        raise ValueError("R44 method reference changed")
    summary = json.loads(R43_SUMMARY.read_text())
    if summary.get("profile") != "full":
        raise ValueError("R44 requires the full R43 checkpoint")
    if summary.get("checkpoint_sha256") != sha256(R43_CHECKPOINT):
        raise ValueError("R43 checkpoint no longer matches its report")
    return value, summary


def command_audit(_args):
    value, summary = protocol()
    print(json.dumps({
        **value,
        "method_reference_sha256": sha256(REFERENCE),
        "R43_checkpoint": str(R43_CHECKPOINT),
        "R43_checkpoint_sha256": sha256(R43_CHECKPOINT),
        "R43_held_out_flip_calibration": summary["held_out_flip_calibration"],
        "output": str(OUTPUT),
    }, indent=2))


def command_run(args):
    value, r43_summary = protocol()
    reconstruction = r42.reconstruct_command_two(args)
    model = OrientationFactorizedConsequenceEnsemble(
        R43_CHECKPOINT, args.device
    )
    mobility_path = r31.artifact_paths()["mobility"]
    mobility = MobilityModel(mobility_path, args.device)
    start = reconstruction["start"][None]
    command = reconstruction["command"][None]
    actual = reconstruction["endpoint"][None]
    inputs = consequence_inputs(start, command)
    samples = int(value["command_samples_per_member"])
    members = []
    for member in model.members:
        row = {"member": int(member)}
        for label, xy in (
            ("at_actual_endpoint_xy", actual[:, :2]),
            ("at_commanded_endpoint_xy", command),
        ):
            weights, means, stds = model._orientation_parameters(
                inputs, xy[:, None], member
            )
            component_matrix = rotation6d_to_matrix(means[0, 0])
            component_upright = component_matrix[:, 2, 2]
            rng = np.random.default_rng(44_000 + 1_000_003 * member)
            draw = model._draw(
                weights[:, 0], means[:, 0], stds[:, 0], samples, rng
            )[0]
            draw_upright = rotation6d_to_matrix(draw)[:, 2, 2]
            row[label] = {
                "mixture_weights": weights[0, 0].tolist(),
                "component_mean_torso_upright": component_upright.tolist(),
                "sampled_flip_probability": float((draw_upright < 0.0).mean()),
            }
        rng = np.random.default_rng(88_000 + 1_000_003 * member)
        delta = model.sample_member(inputs, member, samples, rng)
        endpoint = start[:, None] + delta
        flip = upright_score(endpoint[0]) < 0.0
        predicted_mobility = mobility.predict(endpoint[0])
        spatial = np.exp(
            -np.square(endpoint[0, :, :2] - command).sum(axis=1)
            / (2.0 * float(r31.SIGMA) ** 2)
        )
        row["joint_prediction"] = {
            "flip_probability": float(flip.mean()),
            "mean_endpoint_mobility": float(predicted_mobility.mean()),
            "mean_endpoint_mobility_if_flipped": (
                float(predicted_mobility[flip].mean()) if flip.any() else None
            ),
            "mean_endpoint_mobility_if_not_flipped": (
                float(predicted_mobility[~flip].mean()) if (~flip).any() else None
            ),
            "deployed_connection_integrand_mean_with_frozen_R31_mobility": float(
                (spatial * predicted_mobility).mean()
            ),
            "spatial_kernel_mean_without_mobility": float(spatial.mean()),
        }
        members.append(row)

    previous = json.loads(R42_SUMMARY.read_text())
    previous_flip = [
        item["joint_prediction"]["flip_probability"]
        for item in previous["command_2"]["high_sample_member_predictions"]
    ]
    report = {
        "schema_version": 1,
        "experiment_id": value["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": value["claim_level"],
        "declared_deviations": value["declared_deviations"],
        "oracle_inputs": value["oracle_inputs"],
        "R43_checkpoint": str(R43_CHECKPOINT.resolve()),
        "R43_checkpoint_sha256": sha256(R43_CHECKPOINT),
        "R43_held_out_flip_calibration": r43_summary[
            "held_out_flip_calibration"
        ],
        "command_2": {
            "command": reconstruction["command"].tolist(),
            "start_xy": reconstruction["start"][:2].tolist(),
            "actual_endpoint_xy": reconstruction["endpoint"][:2].tolist(),
            "actual_endpoint_torso_upright": float(
                upright_score(reconstruction["endpoint"])
            ),
            "actual_endpoint_inverted": bool(
                upright_score(reconstruction["endpoint"]) < 0.0
            ),
            "replay_endpoint_error_m": reconstruction["replay_endpoint_error_m"],
            "selected_connection_weight_R40": reconstruction[
                "selected_connection_weight_R40"
            ],
            "R42_posture_model_member_flip_probabilities": previous_flip,
            "R43_orientation_model_member_predictions": members,
        },
        "frozen_R31_mobility_sha256": sha256(mobility_path),
        "task_reward_used": False,
        "hidden_goal_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    output = Path(args.output).resolve()
    dump(output / "summary.json", report)
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--output", type=Path, default=OUTPUT)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("run").set_defaults(func=command_run)
    return value


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

