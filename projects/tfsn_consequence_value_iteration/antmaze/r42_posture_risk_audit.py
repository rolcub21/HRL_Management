"""Audit posture multimodality and flip-risk calibration in frozen R30."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.antmaze.current_tfsn.constants import ENDPOINT_SAMPLES, STATE_DIM
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import paths

from projects.tfsn_consequence_value_iteration.antmaze import (
    aligned_factorized_consequence_r30 as r30,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    MobilityModel,
)
from projects.tfsn_consequence_value_iteration.antmaze.r41_stall_posture_replay import (
    R40,
    source as r41_source,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R42_POSTURE_RISK_AUDIT_REFERENCE.md"
PROTOCOL = HERE / "r42_posture_risk_protocol.json"
METHOD_REVISION = "antmaze_r30_posture_risk_audit_r42"
OUTPUT = R40 / "task3_posture_risk_audit_r42"


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
        "claim_level": "consumed_development_consequence_model_diagnostic_only",
        "task_reward_used": False,
        "hidden_goal_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R42 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE:
        raise ValueError("R42 method reference changed")
    return value


def load_frozen_r29_alignment(r30_protocol):
    """Validate R29 with the scale that was frozen when it was collected.

    Importing R31 deliberately rebinds ``r22.SIGMA`` to the later R30 scale.
    R29's shard validator reads that shared module global, so restore the R29
    collection scale only around validation and then put R31 back exactly as
    it was.
    """
    _root, summary = r30.alignment_root(r30_protocol)
    frozen_sigma = float(summary["landing_sigma_m"])
    deployed_sigma = float(r30.r29_alignment.r22.SIGMA)
    try:
        r30.r29_alignment.r22.SIGMA = frozen_sigma
        return r30.load_alignment(r30_protocol, "full")
    finally:
        r30.r29_alignment.r22.SIGMA = deployed_sigma


def upright_score(states):
    states = np.asarray(states, np.float32)
    # Full state is [xy, height, quaternion(wxyz), ...].
    return 1.0 - 2.0 * (
        np.square(states[..., 4]) + np.square(states[..., 5])
    )


def posture_upright(posture):
    posture = np.asarray(posture, np.float32)
    # Posture begins at height, so quaternion x/y are indices 2/3.
    return 1.0 - 2.0 * (
        np.square(posture[..., 2]) + np.square(posture[..., 3])
    )


def flip_counts(name, starts, endpoints):
    start_flip = upright_score(starts) < 0.0
    endpoint_flip = upright_score(endpoints) < 0.0
    return {
        "name": name,
        "commands": int(len(starts)),
        "start_inverted": int(start_flip.sum()),
        "endpoint_inverted": int(endpoint_flip.sum()),
        "endpoint_inverted_rate": float(endpoint_flip.mean()),
        "upright_to_inverted": int(((~start_flip) & endpoint_flip).sum()),
        "inverted_to_upright": int((start_flip & (~endpoint_flip)).sum()),
        "inverted_to_inverted": int((start_flip & endpoint_flip).sum()),
    }


def calibration(predicted, observed):
    predicted = np.asarray(predicted, np.float64)
    observed = np.asarray(observed, np.float64)
    rows = []
    for low in np.linspace(0.0, 1.0, 10, endpoint=False):
        high = low + 0.1
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
    return {
        "rows": int(len(predicted)),
        "observed_rate": float(observed.mean()),
        "predicted_rate": float(predicted.mean()),
        "bias": float((predicted - observed).mean()),
        "brier": float(np.square(predicted - observed).mean()),
        "bins": rows,
    }


def _draw_posture(model, inputs, endpoint_xy, member, samples, rng):
    weights, means, stds = model._posture_parameters(
        inputs, endpoint_xy[:, None], member
    )
    return model._draw(
        weights[:, 0], means[:, 0], stds[:, 0], samples, rng
    )


def posture_probabilities(
    model, states, commands, endpoint_xy, samples, seed, batch=64
):
    result = []
    for member in model.members:
        chunks = []
        rng = np.random.default_rng(seed + 1_000_003 * member)
        for begin in range(0, len(states), batch):
            stop = min(begin + batch, len(states))
            inputs = consequence_inputs(states[begin:stop], commands[begin:stop])
            posture = _draw_posture(
                model, inputs, endpoint_xy[begin:stop], member, samples, rng
            )
            chunks.append((posture_upright(posture) < 0.0).mean(axis=1))
        result.append(np.concatenate(chunks))
    return np.stack(result, axis=1).astype(np.float32)


def joint_probabilities(model, states, commands, samples, seed, batch=64):
    result = []
    for member in model.members:
        chunks = []
        rng = np.random.default_rng(seed + 2_000_003 * member)
        for begin in range(0, len(states), batch):
            stop = min(begin + batch, len(states))
            inputs = consequence_inputs(states[begin:stop], commands[begin:stop])
            delta = model.sample_member(inputs, member, samples, rng)
            endpoint = states[begin:stop, None] + delta
            chunks.append((upright_score(endpoint) < 0.0).mean(axis=1))
        result.append(np.concatenate(chunks))
    return np.stack(result, axis=1).astype(np.float32)


def reconstruct_command_two(args):
    task, rows = r41_source()
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=1000)
    try:
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = full_state(observation)
        for command_index in range(2):
            start = state.copy()
            command = np.asarray(rows[command_index]["command"], np.float32)
            for _step in range(100):
                action = executor.action(state, command)
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("environment ended while reconstructing command 2")
                state = full_state(observation)
            expected = np.asarray(rows[command_index]["actual_endpoint"], np.float32)
            error = float(np.linalg.norm(state[:2] - expected))
            if error >= 0.05:
                raise ValueError(f"R42 replay diverged by {error:.6f} m")
            if command_index == 1:
                return {
                    "reset_seed": int(task["reset_seed"]),
                    "start": start,
                    "command": command,
                    "endpoint": state.copy(),
                    "expected_endpoint_xy": expected,
                    "replay_endpoint_error_m": error,
                    "selected_connection_weight_R40": float(
                        rows[command_index]["selected_connection_weight"]
                    ),
                }
    finally:
        env.close()
    raise AssertionError("command 2 was not reconstructed")


def command_audit(_args):
    value = protocol()
    payload = torch.load(r31.R30_CHECKPOINT, map_location="cpu", weights_only=False)
    print(json.dumps({
        **value,
        "method_reference_sha256": sha256(REFERENCE),
        "checkpoint_sha256": sha256(r31.R30_CHECKPOINT),
        "spatial_config": payload["spatial_config"],
        "posture_config": payload["posture_config"],
        "deployed_endpoint_samples_total": int(ENDPOINT_SAMPLES),
    }, indent=2))


def command_run(args):
    value = protocol()
    checkpoint = r31.R30_CHECKPOINT
    mobility_path = r31.artifact_paths()["mobility"]
    model = FactorizedConsequenceEnsemble(checkpoint, args.device)
    mobility = MobilityModel(mobility_path, args.device)
    payload = model.payload
    reconstruction = reconstruct_command_two(args)
    start = reconstruction["start"][None]
    command = reconstruction["command"][None]
    actual = reconstruction["endpoint"][None]
    inputs = consequence_inputs(start, command)
    command_samples = int(value["command_samples_per_member"])

    command_members = []
    for member in model.members:
        member_row = {"member": int(member)}
        for label, xy in (
            ("at_actual_endpoint_xy", actual[:, :2]),
            ("at_commanded_endpoint_xy", command),
        ):
            weights, means, stds = model._posture_parameters(
                inputs, xy[:, None], member
            )
            rng = np.random.default_rng(42_000 + 1_000_003 * member)
            samples = model._draw(
                weights[:, 0], means[:, 0], stds[:, 0],
                command_samples, rng,
            )[0]
            member_row[label] = {
                "mixture_weights": weights[0, 0].tolist(),
                "component_mean_torso_height": means[0, 0, :, 0].tolist(),
                "component_mean_torso_upright": posture_upright(
                    means[0, 0]
                ).tolist(),
                "sampled_flip_probability": float(
                    (posture_upright(samples) < 0.0).mean()
                ),
            }

        rng = np.random.default_rng(84_000 + 1_000_003 * member)
        delta = model.sample_member(inputs, member, command_samples, rng)
        endpoint = start[:, None] + delta
        flip = upright_score(endpoint[0]) < 0.0
        predicted_mobility = mobility.predict(endpoint[0])
        spatial = np.exp(
            -np.square(endpoint[0, :, :2] - command).sum(axis=1)
            / (2.0 * float(r31.SIGMA) ** 2)
        )
        member_row["joint_prediction"] = {
            "flip_probability": float(flip.mean()),
            "mean_endpoint_mobility": float(predicted_mobility.mean()),
            "mean_endpoint_mobility_if_flipped": (
                float(predicted_mobility[flip].mean()) if flip.any() else None
            ),
            "mean_endpoint_mobility_if_not_flipped": (
                float(predicted_mobility[~flip].mean()) if (~flip).any() else None
            ),
            "deployed_connection_integrand_mean": float(
                (spatial * predicted_mobility).mean()
            ),
            "spatial_kernel_mean_without_mobility": float(spatial.mean()),
        }
        command_members.append(member_row)

    r30_protocol = r30.load_protocol()
    states, commands, endpoints, task_ids, _distance, _source = (
        load_frozen_r29_alignment(r30_protocol)
    )
    _trained, held_tasks = r30.split_tasks(
        r30_protocol, len(np.unique(task_ids)), "full"
    )
    held = np.isin(task_ids, held_tasks)
    held_states = states[held]
    held_commands = commands[held]
    held_endpoints = endpoints[held]
    observed = (upright_score(held_endpoints) < 0.0).astype(np.float32)
    held_samples = int(value["held_out_samples_per_member"])
    conditional = posture_probabilities(
        model, held_states, held_commands, held_endpoints[:, :2],
        held_samples, 123_000,
    )
    joint = joint_probabilities(
        model, held_states, held_commands, held_samples, 456_000,
    )

    round_dirs = [paths(args.root, index)["round"] for index in range(3)]
    old_starts, _old_commands, old_endpoints = command_records(round_dirs)
    trained_tasks, _held = r30.split_tasks(
        r30_protocol, len(np.unique(task_ids)), "full"
    )
    report = {
        "schema_version": 1,
        "experiment_id": value["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": value["claim_level"],
        "declared_deviations": value["declared_deviations"],
        "oracle_inputs": value["oracle_inputs"],
        "architecture": {
            "spatial_components": int(payload["spatial_config"]["components"]),
            "posture_components": int(payload["posture_config"]["components"]),
            "ensemble_members": int(len(model.members)),
            "posture_head_is_mixture": bool(
                int(payload["posture_config"]["components"]) > 1
            ),
            "deployed_endpoint_samples_total": int(ENDPOINT_SAMPLES),
            "deployed_endpoint_samples_per_member": int(
                ENDPOINT_SAMPLES // len(model.members)
            ),
        },
        "command_2": {
            "reset_seed": reconstruction["reset_seed"],
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
            "high_sample_member_predictions": command_members,
        },
        "preparation_flip_frequency": [
            flip_counts("historical_R0_R2", old_starts, old_endpoints),
            flip_counts("R29_all", states, endpoints),
            flip_counts(
                "R29_train", states[np.isin(task_ids, trained_tasks)],
                endpoints[np.isin(task_ids, trained_tasks)],
            ),
            flip_counts("R29_held_out", held_states, held_endpoints),
        ],
        "held_out_flip_calibration": {
            "held_out_tasks": held_tasks.tolist(),
            "samples_per_member": held_samples,
            "conditional_posture_given_actual_xy": calibration(
                conditional.mean(axis=1), observed
            ),
            "full_joint_spatial_then_posture": calibration(
                joint.mean(axis=1), observed
            ),
            "conditional_member_predicted_rates": conditional.mean(axis=0).tolist(),
            "joint_member_predicted_rates": joint.mean(axis=0).tolist(),
        },
        "artifacts": {
            "consequence": str(checkpoint.resolve()),
            "consequence_sha256": sha256(checkpoint),
            "mobility": str(mobility_path.resolve()),
            "mobility_sha256": sha256(mobility_path),
            "source_R40_log_sha256": sha256(R40 / "task_003.jsonl"),
        },
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
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
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

