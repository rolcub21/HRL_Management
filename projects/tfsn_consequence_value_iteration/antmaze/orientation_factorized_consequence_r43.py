"""Orientation-factorized Ant consequence model with physical rotations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.main import paths
from scripts.antmaze.current_tfsn.mdn import MDNConfig, MixtureDensityNetwork

from projects.tfsn_consequence_value_iteration.antmaze import (
    aligned_factorized_consequence_r30 as r30,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    _normalize,
    _raw_parameters,
    _train_head,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R43_ORIENTATION_FACTORIZED_REFERENCE.md"
PROTOCOL = HERE / "r43_orientation_factorized_protocol.json"
METHOD_REVISION = "antmaze_orientation_factorized_consequence_r43"
R30_CHECKPOINT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r29_aligned_consequence_r30/consequence.pt"
)
OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "orientation_factorized_consequence_r43"
)
STATE_DIM = 29
ORIENTATION_DIM = 6
REST_DIM = 23


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_protocol():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": "reward_free_orientation_factorized_consequence_component_only",
        "oracle_inputs": [],
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R43 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE:
        raise ValueError("R43 method reference changed")
    return value


def quaternion_to_matrix(quaternion):
    quaternion = np.asarray(quaternion, np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, 1e-8)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty((*quaternion.shape[:-1], 3, 3), np.float32)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def matrix_to_rotation6d(matrix):
    matrix = np.asarray(matrix, np.float32)
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def quaternion_to_rotation6d(quaternion):
    return matrix_to_rotation6d(quaternion_to_matrix(quaternion))


def rotation6d_to_matrix(rotation):
    rotation = np.asarray(rotation, np.float32)
    a1, a2 = rotation[..., :3], rotation[..., 3:]
    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    fallback1 = np.zeros_like(a1)
    fallback1[..., 0] = 1.0
    b1 = np.where(norm1 > 1e-7, a1 / np.maximum(norm1, 1e-7), fallback1)
    residual = a2 - (a2 * b1).sum(axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(residual, axis=-1, keepdims=True)
    basis = np.eye(3, dtype=np.float32)[np.argmin(np.abs(b1), axis=-1)]
    fallback2 = basis - (basis * b1).sum(axis=-1, keepdims=True) * b1
    fallback2 /= np.maximum(
        np.linalg.norm(fallback2, axis=-1, keepdims=True), 1e-7
    )
    b2 = np.where(
        norm2 > 1e-7, residual / np.maximum(norm2, 1e-7), fallback2
    )
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1).astype(np.float32)


def rotation6d_to_quaternion(rotation, reference):
    matrix = rotation6d_to_matrix(rotation)
    shape = matrix.shape[:-2]
    xyzw = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
    wxyz = np.concatenate((xyzw[:, 3:4], xyzw[:, :3]), axis=1).reshape(
        *shape, 4
    ).astype(np.float32)
    reference = np.broadcast_to(np.asarray(reference, np.float32), wxyz.shape)
    sign = np.where((wxyz * reference).sum(axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return (wxyz * sign).astype(np.float32)


def endpoint_orientation(states):
    states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
    return quaternion_to_rotation6d(states[:, 3:7])


def endpoint_rest(states):
    states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
    return np.concatenate((states[:, 2:3], states[:, 7:]), axis=1)


def upright_score(states):
    states = np.asarray(states, np.float32)
    return 1.0 - 2.0 * (
        np.square(states[..., 4]) + np.square(states[..., 5])
    )


def calibration(predicted, observed):
    predicted = np.asarray(predicted, np.float64)
    observed = np.asarray(observed, np.float64)
    return {
        "rows": int(len(predicted)),
        "observed_rate": float(observed.mean()),
        "predicted_rate": float(predicted.mean()),
        "bias": float((predicted - observed).mean()),
        "brier": float(np.square(predicted - observed).mean()),
        "bins": r30.calibration_rows(predicted, observed),
    }


class OrientationFactorizedConsequenceEnsemble(FactorizedConsequenceEnsemble):
    """Spatial, physical-orientation, then remaining-posture distribution."""

    def __init__(self, checkpoint, device):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != "orientation_factorized_ant_consequence":
            raise ValueError("not an orientation-factorized consequence checkpoint")
        self.payload = payload
        self.device = torch.device(device)
        self.input_mean = np.asarray(payload["base_input_mean"], np.float32)
        self.input_std = np.asarray(payload["base_input_std"], np.float32)
        self.spatial_target_mean = np.asarray(payload["spatial_target_mean"], np.float32)
        self.spatial_target_std = np.asarray(payload["spatial_target_std"], np.float32)
        self.orientation_input_mean = np.asarray(payload["orientation_input_mean"], np.float32)
        self.orientation_input_std = np.asarray(payload["orientation_input_std"], np.float32)
        self.orientation_target_mean = np.asarray(payload["orientation_target_mean"], np.float32)
        self.orientation_target_std = np.asarray(payload["orientation_target_std"], np.float32)
        self.rest_input_mean = np.asarray(payload["rest_input_mean"], np.float32)
        self.rest_input_std = np.asarray(payload["rest_input_std"], np.float32)
        self.rest_target_mean = np.asarray(payload["rest_target_mean"], np.float32)
        self.rest_target_std = np.asarray(payload["rest_target_std"], np.float32)
        spatial_config = MDNConfig(**payload["spatial_config"])
        orientation_config = MDNConfig(**payload["orientation_config"])
        rest_config = MDNConfig(**payload["rest_config"])
        self.spatial_members = []
        self.orientation_members = []
        self.rest_members = []
        for state in payload["spatial_member_states"]:
            model = MixtureDensityNetwork(spatial_config).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.spatial_members.append(model)
        for state in payload["orientation_member_states"]:
            model = MixtureDensityNetwork(orientation_config).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.orientation_members.append(model)
        for state in payload["rest_member_states"]:
            model = MixtureDensityNetwork(rest_config).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.rest_members.append(model)
        count = {len(self.spatial_members), len(self.orientation_members), len(self.rest_members)}
        if len(count) != 1:
            raise ValueError("R43 ensemble heads do not align")
        self.members = list(range(len(self.spatial_members)))

    def _head_parameters(
        self, network, values, input_mean, input_std, target_mean, target_std
    ):
        normalized = _normalize(values, input_mean, input_std)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            weights, means, stds = _raw_parameters(
                network,
                tensor,
                torch.as_tensor(target_mean, device=self.device),
                torch.as_tensor(target_std, device=self.device),
            )
        return (
            weights.cpu().numpy().astype(np.float32),
            means.cpu().numpy().astype(np.float32),
            stds.cpu().numpy().astype(np.float32),
        )

    def _orientation_parameters(self, inputs, endpoint_xy, member):
        inputs = np.asarray(inputs, np.float32)
        endpoint_xy = np.asarray(endpoint_xy, np.float32)
        batch, samples, _ = endpoint_xy.shape
        repeated = np.repeat(inputs[:, None], samples, axis=1)
        values = np.concatenate((repeated, endpoint_xy), axis=2).reshape(
            batch * samples, -1
        )
        weights, means, stds = self._head_parameters(
            self.orientation_members[member], values,
            self.orientation_input_mean, self.orientation_input_std,
            self.orientation_target_mean, self.orientation_target_std,
        )
        components = weights.shape[-1]
        return (
            weights.reshape(batch, samples, components),
            means.reshape(batch, samples, components, ORIENTATION_DIM),
            stds.reshape(batch, samples, components, ORIENTATION_DIM),
        )

    def _rest_parameters(self, inputs, endpoint_xy, orientation, member):
        inputs = np.asarray(inputs, np.float32)
        batch, samples, _ = endpoint_xy.shape
        repeated = np.repeat(inputs[:, None], samples, axis=1)
        values = np.concatenate((repeated, endpoint_xy, orientation), axis=2).reshape(
            batch * samples, -1
        )
        weights, means, stds = self._head_parameters(
            self.rest_members[member], values,
            self.rest_input_mean, self.rest_input_std,
            self.rest_target_mean, self.rest_target_std,
        )
        components = weights.shape[-1]
        return (
            weights.reshape(batch, samples, components),
            means.reshape(batch, samples, components, REST_DIM),
            stds.reshape(batch, samples, components, REST_DIM),
        )

    @staticmethod
    def _draw_per_sample(weights, means, stds, rng):
        batch, samples, _components = weights.shape
        cumulative = np.cumsum(weights, axis=2)
        cumulative[:, :, -1] = 1.0
        component = (rng.random((batch, samples, 1)) > cumulative).sum(axis=2)
        rows = np.arange(batch)[:, None]
        draws = np.arange(samples)[None]
        selected_mean = means[rows, draws, component]
        selected_std = stds[rows, draws, component]
        return selected_mean + selected_std * rng.standard_normal(
            selected_mean.shape
        ).astype(np.float32)

    def sample_member(self, inputs, member, samples, rng):
        inputs = np.asarray(inputs, np.float32)
        endpoint_xy = self.sample_spatial_member(inputs, member, samples, rng)
        weights, means, stds = self._orientation_parameters(
            inputs, endpoint_xy, member
        )
        raw_orientation = self._draw_per_sample(weights, means, stds, rng)
        matrix = rotation6d_to_matrix(raw_orientation)
        orientation = matrix_to_rotation6d(matrix)
        start_quaternion = inputs[:, None, 3:7]
        quaternion = rotation6d_to_quaternion(orientation, start_quaternion)
        rest_weight, rest_mean, rest_std = self._rest_parameters(
            inputs, endpoint_xy, orientation, member
        )
        rest = self._draw_per_sample(rest_weight, rest_mean, rest_std, rng)
        endpoint = np.empty((len(inputs), samples, STATE_DIM), np.float32)
        endpoint[..., :2] = endpoint_xy
        endpoint[..., 2] = rest[..., 0]
        endpoint[..., 3:7] = quaternion
        endpoint[..., 7:] = rest[..., 1:]
        start = inputs[:, :STATE_DIM]
        return (endpoint - start[:, None]).astype(np.float32)


def held_out_flip_probability(model, states, commands, samples, seed, batch=64):
    members = []
    for member in model.members:
        rng = np.random.default_rng(seed + 1_000_003 * member)
        chunks = []
        for begin in range(0, len(states), batch):
            stop = min(begin + batch, len(states))
            inputs = consequence_inputs(states[begin:stop], commands[begin:stop])
            delta = model.sample_member(inputs, member, samples, rng)
            endpoint = states[begin:stop, None] + delta
            chunks.append((upright_score(endpoint) < 0.0).mean(axis=1))
        members.append(np.concatenate(chunks))
    return np.stack(members, axis=1).astype(np.float32)


def training_data(args, protocol):
    r30_protocol = r30.load_protocol()
    states, commands, endpoints, task_ids, _distance, source = r30.load_alignment(
        r30_protocol, args.profile
    )
    available = len(np.unique(task_ids))
    trained_tasks, held_tasks = r30.split_tasks(
        r30_protocol, available, args.profile
    )
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
    combined_states = np.concatenate((train_states, validation_states))
    combined_commands = np.concatenate((train_commands, validation_commands))
    combined_endpoints = np.concatenate((train_endpoints, validation_endpoints))
    training_index = np.arange(len(train_states), dtype=np.int64)
    validation_index = np.arange(len(train_states), len(combined_states), dtype=np.int64)
    return {
        "states": combined_states,
        "commands": combined_commands,
        "endpoints": combined_endpoints,
        "training_index": training_index,
        "validation_index": validation_index,
        "validation_states": validation_states,
        "validation_commands": validation_commands,
        "validation_endpoints": validation_endpoints,
        "trained_tasks": trained_tasks,
        "held_tasks": held_tasks,
        "round_dirs": round_dirs,
        "alignment_source": source,
    }


def command_audit(_args):
    value = load_protocol()
    print(json.dumps({
        **value,
        "method_reference_sha256": sha256(REFERENCE),
        "frozen_R30_spatial_checkpoint": str(R30_CHECKPOINT),
        "frozen_R30_spatial_checkpoint_sha256": sha256(R30_CHECKPOINT),
        "output": str(OUTPUT),
    }, indent=2))


def command_train(args):
    protocol = load_protocol()
    data = training_data(args, protocol)
    inputs = consequence_inputs(data["states"], data["commands"])
    orientation_target = endpoint_orientation(data["endpoints"])
    orientation_input = np.concatenate((inputs, data["endpoints"][:, :2]), axis=1)
    rest_target = endpoint_rest(data["endpoints"])
    rest_input = np.concatenate(
        (inputs, data["endpoints"][:, :2], orientation_target), axis=1
    )
    schedule = protocol["training_schedule"]
    epochs = 2 if args.profile == "smoke" else int(schedule["epochs"])
    orientation = _train_head(
        orientation_input, orientation_target,
        data["training_index"], data["validation_index"],
        int(protocol["split_seed"]) + 901, args.device, epochs,
        int(schedule["batch_size"]), float(schedule["learning_rate"]),
        "R43 orientation",
    )
    rest = _train_head(
        rest_input, rest_target,
        data["training_index"], data["validation_index"],
        int(protocol["split_seed"]) + 902, args.device, epochs,
        int(schedule["batch_size"]), float(schedule["learning_rate"]),
        "R43 remaining posture",
    )
    frozen = torch.load(R30_CHECKPOINT, map_location="cpu", weights_only=False)
    output = Path(args.output).resolve()
    checkpoint = output / "consequence.pt"
    payload = {
        "artifact_type": "orientation_factorized_ant_consequence",
        "method_revision": METHOD_REVISION,
        "claim_level": protocol["claim_level"] if args.profile == "full" else "smoke_only",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(rotation6d|state,command,endpoint_xy) times "
            "P(remaining_posture|state,command,endpoint_xy,rotation6d)"
        ),
        "rotation_representation": "projected_continuous_6D_rotation",
        "quaternion_sign_rule": "closest_to_start_quaternion",
        "spatial_head_source": str(R30_CHECKPOINT.resolve()),
        "spatial_head_source_sha256": sha256(R30_CHECKPOINT),
        "training_source": [str(path.resolve()) for path in data["round_dirs"]] + [
            str(data["alignment_source"])
        ],
        "training_commands": int(len(data["training_index"])),
        "validation_commands": int(len(data["validation_index"])),
        "split_seed": int(protocol["split_seed"]),
        "split": "unchanged_R30_whole_R29_preparation_tasks_held_out",
        "held_out_alignment_tasks": data["held_tasks"].tolist(),
        "oracle_command_dataset_used_for_training_or_selection": False,
        "base_input_mean": frozen["base_input_mean"],
        "base_input_std": frozen["base_input_std"],
        "spatial_target_mean": frozen["spatial_target_mean"],
        "spatial_target_std": frozen["spatial_target_std"],
        "spatial_config": frozen["spatial_config"],
        "spatial_member_states": copy.deepcopy(frozen["spatial_member_states"]),
        "orientation_input_mean": orientation["input_mean"],
        "orientation_input_std": orientation["input_std"],
        "orientation_target_mean": orientation["target_mean"],
        "orientation_target_std": orientation["target_std"],
        "orientation_config": orientation["config"],
        "orientation_member_states": orientation["member_states"],
        "rest_input_mean": rest["input_mean"],
        "rest_input_std": rest["input_std"],
        "rest_target_mean": rest["target_mean"],
        "rest_target_std": rest["target_std"],
        "rest_config": rest["config"],
        "rest_member_states": rest["member_states"],
        "epochs": epochs,
        "batch_size": int(schedule["batch_size"]),
        "learning_rate": float(schedule["learning_rate"]),
    }
    atomic_torch(checkpoint, payload)
    model = OrientationFactorizedConsequenceEnsemble(checkpoint, args.device)
    samples = int(protocol["validation_samples_per_member"])
    probability = held_out_flip_probability(
        model, data["validation_states"], data["validation_commands"],
        samples, int(protocol["split_seed"]) + 903,
    )
    observed = (
        upright_score(data["validation_endpoints"]) < 0.0
    ).astype(np.float32)
    report = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": payload["claim_level"],
        "declared_deviations": protocol["declared_deviations"],
        "oracle_inputs": [],
        "profile": args.profile,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "frozen_spatial_checkpoint_sha256": sha256(R30_CHECKPOINT),
        "training_commands": int(len(data["training_index"])),
        "validation_commands": int(len(data["validation_index"])),
        "trained_alignment_tasks": data["trained_tasks"].tolist(),
        "held_out_alignment_tasks": data["held_tasks"].tolist(),
        "orientation_components": int(orientation["config"]["components"]),
        "remaining_posture_components": int(rest["config"]["components"]),
        "orientation_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"] for row in orientation["histories"]
        ],
        "remaining_posture_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"] for row in rest["histories"]
        ],
        "held_out_flip_calibration": calibration(probability.mean(axis=1), observed),
        "held_out_member_flip_rates": probability.mean(axis=0).tolist(),
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    atomic_json(output / "summary.json", report)
    atomic_json(output / "training_history.json", {
        "orientation": orientation["histories"],
        "remaining_posture": rest["histories"],
    })
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--device", default="cuda")
    value.add_argument("--profile", choices=("smoke", "full"), default="full")
    value.add_argument("--output", type=Path, default=OUTPUT)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("train").set_defaults(func=command_train)
    return value


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

