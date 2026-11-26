"""Factorized Ant consequence model and untouched oracle-command diagnostic.

The spatial model has its own likelihood over ``endpoint_xy - command``.
The posture model is trained separately on endpoint posture conditioned on the
realized endpoint coordinate. At inference the spatial outcome is sampled
first and the posture distribution is then conditioned on that sample.

Only reward-free preparation rounds 0--2 are training data. The 51 commands
from the fixed-contract oracle diagnostic are read only after training and are
never included in either fit or model selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.antmaze.current_tfsn.constants import (
    BATCH_SIZE,
    EPOCHS,
    LANDING_SIGMA,
    LEARNING_RATE,
    MDN_COMPONENTS,
    STATE_DIM,
    VALIDATION_FRACTION,
)
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.main import dump, paths
from scripts.antmaze.current_tfsn.mdn import (
    MDNConfig,
    MixtureDensityNetwork,
    mixture_nll,
)

from projects.tfsn_consequence_value_iteration.antmaze.oracle_contract_diagnostic import (
    ORIGINAL_ORACLE_SUCCESSES,
    STALL_RADIUS_M,
    calibration_bins,
    output_directory,
    sha256_file,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


METHOD_REVISION = "antmaze_factorized_consequence_r8"
CLAIM_LEVEL = "consumed_development_factorized_consequence_diagnostic"
SPATIAL_DIM = 2
POSTURE_DIM = STATE_DIM - SPATIAL_DIM
DIAGNOSTIC_SAMPLES = 4096
BASELINE_SUCCESS_MASS_WITHIN_0P8 = 0.0127489
BASELINE_SUCCESS_STALL_BRIER = 1.9501865608617663e-05


def _safe_std(values):
    return np.maximum(
        np.asarray(values, np.float32).std(axis=0), 1e-6
    ).astype(np.float32)


def factorized_checkpoint(root):
    return (
        paths(root)["models"]
        / "factorized_consequence_r8"
        / "consequence.pt"
    )


def _normalize(values, mean, std):
    return (np.asarray(values, np.float32) - mean) / std


def _raw_parameters(network, normalized_inputs, target_mean, target_std):
    logits, means, stds = network(normalized_inputs)
    weights = torch.softmax(logits, dim=-1)
    means = means * target_std[None, None] + target_mean[None, None]
    stds = stds * target_std[None, None]
    return weights, means, stds


def _train_head(
    inputs,
    targets,
    training_index,
    validation_index,
    seed,
    device,
    epochs,
    batch_size,
    learning_rate,
    label,
):
    """Train an independent mixture head and select it on its own NLL."""
    input_mean = inputs[training_index].mean(axis=0).astype(np.float32)
    input_std = _safe_std(inputs[training_index])
    target_mean = targets[training_index].mean(axis=0).astype(np.float32)
    target_std = _safe_std(targets[training_index])
    x = torch.as_tensor(
        _normalize(inputs, input_mean, input_std), dtype=torch.float32
    )
    y = torch.as_tensor(
        _normalize(targets, target_mean, target_std), dtype=torch.float32
    )
    config = MDNConfig(
        input_dim=inputs.shape[1],
        output_dim=targets.shape[1],
        components=MDN_COMPONENTS,
    )
    states = []
    histories = []
    for member in range(config.ensemble_size):
        member_seed = int(seed + 10_000 * (member + 1))
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
        network = MixtureDensityNetwork(config).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        member_rng = np.random.default_rng(seed + 100_000 * (member + 1))
        best_loss = math.inf
        best_state = None
        history = []
        for epoch in range(epochs):
            network.train()
            order = member_rng.permutation(training_index)
            train_losses = []
            for begin in range(0, len(order), batch_size):
                index = order[begin:begin + batch_size]
                xb = x[index].to(device)
                yb = y[index].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = mixture_nll(*network(xb), yb).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            network.eval()
            validation_losses = []
            with torch.no_grad():
                for begin in range(0, len(validation_index), 4096):
                    index = validation_index[begin:begin + 4096]
                    validation_losses.append(
                        mixture_nll(
                            *network(x[index].to(device)),
                            y[index].to(device),
                        ).cpu().numpy()
                    )
            validation_loss = float(np.concatenate(validation_losses).mean())
            row = {
                "epoch": epoch + 1,
                "train_normalized_nll": float(np.mean(train_losses)),
                "validation_normalized_nll": validation_loss,
            }
            history.append(row)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(network.state_dict())
            if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
                print(
                    f"{label} member={member + 1}/{config.ensemble_size} "
                    f"epoch={epoch + 1}/{epochs} "
                    f"val_nll={validation_loss:.6f}",
                    flush=True,
                )
        if best_state is None:
            raise RuntimeError(f"{label} did not produce a checkpoint")
        states.append(best_state)
        histories.append({
            "member": member,
            "best_validation_normalized_nll": best_loss,
            "epochs": history,
        })
    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "config": config.__dict__,
        "member_states": states,
        "histories": histories,
    }


class FactorizedConsequenceEnsemble:
    """Sequential spatial-then-posture endpoint distribution."""

    def __init__(self, checkpoint, device):
        payload = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=False
        )
        if payload.get("artifact_type") != "factorized_ant_consequence":
            raise ValueError("not a factorized Ant consequence checkpoint")
        self.payload = payload
        self.device = torch.device(device)
        self.input_mean = np.asarray(payload["base_input_mean"], np.float32)
        self.input_std = np.asarray(payload["base_input_std"], np.float32)
        self.spatial_target_mean = np.asarray(
            payload["spatial_target_mean"], np.float32
        )
        self.spatial_target_std = np.asarray(
            payload["spatial_target_std"], np.float32
        )
        self.posture_input_mean = np.asarray(
            payload["posture_input_mean"], np.float32
        )
        self.posture_input_std = np.asarray(
            payload["posture_input_std"], np.float32
        )
        self.posture_target_mean = np.asarray(
            payload["posture_target_mean"], np.float32
        )
        self.posture_target_std = np.asarray(
            payload["posture_target_std"], np.float32
        )
        spatial_config = MDNConfig(**payload["spatial_config"])
        posture_config = MDNConfig(**payload["posture_config"])
        self.spatial_members = []
        self.posture_members = []
        for state in payload["spatial_member_states"]:
            model = MixtureDensityNetwork(spatial_config).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.spatial_members.append(model)
        for state in payload["posture_member_states"]:
            model = MixtureDensityNetwork(posture_config).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.posture_members.append(model)
        if len(self.spatial_members) != len(self.posture_members):
            raise ValueError("spatial and posture ensembles do not align")
        # Existing planners use this only to enumerate ensemble members.
        self.members = list(range(len(self.spatial_members)))

    def spatial_parameters(self, inputs, member):
        inputs = np.asarray(inputs, np.float32)
        normalized = _normalize(inputs, self.input_mean, self.input_std)
        tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            weights, means, stds = _raw_parameters(
                self.spatial_members[member],
                tensor,
                torch.as_tensor(
                    self.spatial_target_mean, device=self.device
                ),
                torch.as_tensor(
                    self.spatial_target_std, device=self.device
                ),
            )
        return (
            weights.cpu().numpy().astype(np.float32),
            means.cpu().numpy().astype(np.float32),
            stds.cpu().numpy().astype(np.float32),
        )

    @staticmethod
    def _draw(weights, means, stds, samples, rng):
        cumulative = np.cumsum(weights, axis=1)
        cumulative[:, -1] = 1.0
        uniforms = rng.random((len(weights), samples, 1))
        components = (uniforms > cumulative[:, None]).sum(axis=-1)
        rows = np.arange(len(weights))[:, None]
        selected_mean = means[rows, components]
        selected_std = stds[rows, components]
        noise = rng.standard_normal(selected_mean.shape).astype(np.float32)
        return (selected_mean + selected_std * noise).astype(np.float32)

    def sample_spatial_member(self, inputs, member, samples, rng):
        weights, residual, scale = self.spatial_parameters(inputs, member)
        residual = self._draw(weights, residual, scale, samples, rng)
        command = np.asarray(inputs, np.float32)[:, STATE_DIM:STATE_DIM + 2]
        return command[:, None] + residual

    def spatial_component_mean_endpoints(self, inputs, member):
        """Return deterministic full-state means for every spatial mode.

        The spatial mixture is preserved exactly.  Posture is represented by
        its conditional mixture mean at each spatial component mean.  This is
        the deterministic quadrature needed by planners whose fields depend
        on position but whose next consequence prediction still requires a
        complete Ant state.
        """
        inputs = np.asarray(inputs, np.float32)
        weights, residual, _scale = self.spatial_parameters(inputs, member)
        command = inputs[:, STATE_DIM:STATE_DIM + 2]
        endpoint_xy = command[:, None] + residual
        posture_weight, posture_mean, _posture_scale = (
            self._posture_parameters(inputs, endpoint_xy, member)
        )
        expected_posture = (
            posture_weight[..., None] * posture_mean
        ).sum(axis=2)
        endpoints = np.concatenate((endpoint_xy, expected_posture), axis=2)
        return weights.astype(np.float32), endpoints.astype(np.float32)

    def _posture_parameters(self, inputs, endpoint_xy, member):
        inputs = np.asarray(inputs, np.float32)
        endpoint_xy = np.asarray(endpoint_xy, np.float32)
        batch, samples, _ = endpoint_xy.shape
        repeated = np.repeat(inputs[:, None], samples, axis=1)
        conditioned = np.concatenate((repeated, endpoint_xy), axis=2).reshape(
            batch * samples, -1
        )
        normalized = _normalize(
            conditioned, self.posture_input_mean, self.posture_input_std
        )
        tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            weights, means, stds = _raw_parameters(
                self.posture_members[member],
                tensor,
                torch.as_tensor(
                    self.posture_target_mean, device=self.device
                ),
                torch.as_tensor(
                    self.posture_target_std, device=self.device
                ),
            )
        components = weights.shape[-1]
        return (
            weights.reshape(batch, samples, components)
            .cpu().numpy().astype(np.float32),
            means.reshape(batch, samples, components, POSTURE_DIM)
            .cpu().numpy().astype(np.float32),
            stds.reshape(batch, samples, components, POSTURE_DIM)
            .cpu().numpy().astype(np.float32),
        )

    def sample_member(self, inputs, member, samples, rng):
        """Return full-state deltas for compatibility with existing planners."""
        inputs = np.asarray(inputs, np.float32)
        endpoint_xy = self.sample_spatial_member(
            inputs, member, samples, rng
        )
        weights, means, stds = self._posture_parameters(
            inputs, endpoint_xy, member
        )
        batch = len(inputs)
        cumulative = np.cumsum(weights, axis=2)
        cumulative[:, :, -1] = 1.0
        uniforms = rng.random((batch, samples, 1))
        components = (uniforms > cumulative).sum(axis=2)
        rows = np.arange(batch)[:, None]
        draws = np.arange(samples)[None]
        selected_mean = means[rows, draws, components]
        selected_std = stds[rows, draws, components]
        posture = selected_mean + selected_std * rng.standard_normal(
            selected_mean.shape
        ).astype(np.float32)
        start = inputs[:, :STATE_DIM]
        delta = np.empty((batch, samples, STATE_DIM), np.float32)
        delta[..., :2] = endpoint_xy - start[:, None, :2]
        delta[..., 2:] = posture - start[:, None, 2:]
        return delta


def command_train(args):
    round_dirs = [
        paths(args.root, index)["round"] for index in range(args.round + 1)
    ]
    starts, commands, endpoints = command_records(round_dirs)
    inputs = consequence_inputs(starts, commands)
    spatial_target = endpoints[:, :2] - commands
    posture_input = np.concatenate((inputs, endpoints[:, :2]), axis=1)
    posture_target = endpoints[:, 2:]

    frozen = torch.load(
        paths(args.root, args.round)["consequence"],
        map_location="cpu",
        weights_only=False,
    )
    split_seed = int(frozen["seed"])
    permutation = np.random.default_rng(split_seed).permutation(len(inputs))
    validation_count = max(
        1, int(round(VALIDATION_FRACTION * len(inputs)))
    )
    validation_index = permutation[:validation_count]
    training_index = permutation[validation_count:]

    spatial = _train_head(
        inputs,
        spatial_target,
        training_index,
        validation_index,
        split_seed + 801,
        args.device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        "spatial",
    )
    posture = _train_head(
        posture_input,
        posture_target,
        training_index,
        validation_index,
        split_seed + 802,
        args.device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        "posture",
    )

    checkpoint = factorized_checkpoint(args.root)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "factorized_ant_consequence",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "spatial_parameterization": "endpoint_xy_minus_command",
        "posture_parameterization": "absolute_endpoint_posture",
        "factorization": (
            "P(endpoint_xy-command|state,command) times "
            "P(endpoint_posture|state,command,endpoint_xy)"
        ),
        "head_optimization": (
            "independent_losses_and_independent_model_selection"
        ),
        "training_source": [str(path.resolve()) for path in round_dirs],
        "training_commands": int(len(training_index)),
        "validation_commands": int(len(validation_index)),
        "split_seed": split_seed,
        "split": "same_indices_as_frozen_round2_joint_consequence",
        "oracle_command_dataset_used_for_training_or_selection": False,
        "oracle_command_dataset": str(
            (output_directory(args.root) / "executed_commands.npz").resolve()
        ),
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
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
    }
    torch.save(payload, checkpoint)
    report = {
        key: payload[key]
        for key in (
            "artifact_type",
            "method_revision",
            "claim_level",
            "spatial_parameterization",
            "posture_parameterization",
            "factorization",
            "head_optimization",
            "training_source",
            "training_commands",
            "validation_commands",
            "split_seed",
            "split",
            "oracle_command_dataset_used_for_training_or_selection",
            "epochs",
            "batch_size",
            "learning_rate",
        )
    }
    report.update({
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "spatial_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"]
            for row in spatial["histories"]
        ],
        "posture_best_validation_normalized_nll": [
            row["best_validation_normalized_nll"]
            for row in posture["histories"]
        ],
    })
    dump(checkpoint.with_suffix(".json"), report)
    history = {
        "spatial": spatial["histories"],
        "posture": posture["histories"],
    }
    dump(checkpoint.parent / "training_history.json", history)
    print(json.dumps(report, indent=2))


def _spatial_distribution(model, inputs):
    weights, means, stds = [], [], []
    command = np.asarray(inputs, np.float32)[:, STATE_DIM:STATE_DIM + 2]
    for member in range(len(model.members)):
        weight, residual, scale = model.spatial_parameters(inputs, member)
        weights.append(weight / len(model.members))
        means.append(command[:, None] + residual)
        stds.append(scale)
    return (
        np.concatenate(weights, axis=1),
        np.concatenate(means, axis=1),
        np.concatenate(stds, axis=1),
    )


def _spatial_nll(weights, means, stds, endpoints):
    difference = (endpoints[:, None, :2] - means) / stds
    component = (
        np.log(np.maximum(weights, 1e-30))
        - np.log(stds).sum(axis=2)
        - math.log(2.0 * math.pi)
        - 0.5 * np.square(difference).sum(axis=2)
    )
    maximum = component.max(axis=1)
    return -(
        maximum
        + np.log(np.exp(component - maximum[:, None]).sum(axis=1))
    )


def _sample_spatial_distribution(weights, means, stds, samples, seed):
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(weights, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random((len(weights), samples, 1))
    components = (uniforms > cumulative[:, None]).sum(axis=-1)
    rows = np.arange(len(weights))[:, None]
    selected_mean = means[rows, components]
    selected_std = stds[rows, components]
    noise = rng.standard_normal(selected_mean.shape).astype(np.float32)
    return (selected_mean + selected_std * noise).astype(np.float32)


def _summarize(
    name,
    mask,
    nll,
    percentile,
    nearest_mode,
    mixture_mean_error,
    endpoint_mass_0p8,
    endpoint_mass_sigma,
    predicted_stall,
    observed_stall,
):
    mask = np.asarray(mask, bool)
    return {
        "name": name,
        "commands": int(mask.sum()),
        "spatial_nll_mean": float(nll[mask].mean()),
        "spatial_nll_median": float(np.median(nll[mask])),
        "validation_spatial_nll_percentile_median": float(
            np.median(percentile[mask])
        ),
        "nearest_spatial_mode_error_mean_m": float(
            nearest_mode[mask].mean()
        ),
        "nearest_spatial_mode_error_median_m": float(
            np.median(nearest_mode[mask])
        ),
        "spatial_mixture_mean_error_mean_m": float(
            mixture_mean_error[mask].mean()
        ),
        "spatial_mixture_mean_error_median_m": float(
            np.median(mixture_mean_error[mask])
        ),
        "predicted_endpoint_mass_within_0p8_mean": float(
            endpoint_mass_0p8[mask].mean()
        ),
        "predicted_endpoint_mass_within_0p8_median": float(
            np.median(endpoint_mass_0p8[mask])
        ),
        "predicted_endpoint_mass_within_landing_sigma_mean": float(
            endpoint_mass_sigma[mask].mean()
        ),
        "predicted_endpoint_mass_within_landing_sigma_median": float(
            np.median(endpoint_mass_sigma[mask])
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


def command_diagnose(args):
    checkpoint = factorized_checkpoint(args.root)
    if not checkpoint.exists():
        raise FileNotFoundError("train-factorized-consequence first")
    output = output_directory(args.root)
    dataset = output / "executed_commands.npz"
    oracle_summary = json.loads((output / "summary.json").read_text())
    if sha256_file(dataset) != oracle_summary["executed_command_dataset_sha256"]:
        raise ValueError("oracle command dataset changed")
    with np.load(dataset) as data:
        trajectories = data["states"].astype(np.float32)
        commands = data["commands"].astype(np.float32)
        task_id = data["task_id"].astype(np.int64)
        command_index = data["command_index"].astype(np.int64)
    starts = trajectories[:, 0]
    endpoints = trajectories[:, -1]
    inputs = consequence_inputs(starts, commands)
    observed_stall = (
        np.linalg.norm(endpoints[:, :2] - starts[:, :2], axis=1)
        < STALL_RADIUS_M
    )

    model = FactorizedConsequenceEnsemble(checkpoint, args.device)
    weights, means, stds = _spatial_distribution(model, inputs)
    nll = _spatial_nll(weights, means, stds, endpoints)
    nearest_mode = np.linalg.norm(
        means - endpoints[:, None, :2], axis=2
    ).min(axis=1)
    mixture_mean = (weights[..., None] * means).sum(axis=1)
    mixture_mean_error = np.linalg.norm(
        mixture_mean - endpoints[:, :2], axis=1
    )
    samples = _sample_spatial_distribution(
        weights, means, stds, DIAGNOSTIC_SAMPLES, args.seed
    )
    endpoint_distance = np.linalg.norm(
        samples - endpoints[:, None, :2], axis=2
    )
    endpoint_mass_0p8 = (endpoint_distance <= 0.8).mean(axis=1)
    endpoint_mass_sigma = (endpoint_distance <= LANDING_SIGMA).mean(axis=1)
    predicted_stall = (
        np.linalg.norm(samples - starts[:, None, :2], axis=2)
        < STALL_RADIUS_M
    ).mean(axis=1)

    round_dirs = [
        paths(args.root, index)["round"] for index in range(args.round + 1)
    ]
    reference_starts, reference_commands, reference_endpoints = command_records(
        round_dirs
    )
    reference_inputs = consequence_inputs(reference_starts, reference_commands)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    permutation = np.random.default_rng(int(payload["split_seed"])).permutation(
        len(reference_inputs)
    )
    validation_count = max(
        1, int(round(VALIDATION_FRACTION * len(reference_inputs)))
    )
    validation_index = permutation[:validation_count]
    validation_weights, validation_means, validation_stds = (
        _spatial_distribution(model, reference_inputs[validation_index])
    )
    validation_nll = _spatial_nll(
        validation_weights,
        validation_means,
        validation_stds,
        reference_endpoints[validation_index],
    )
    sorted_validation = np.sort(validation_nll)
    percentile = np.searchsorted(
        sorted_validation, nll, side="right"
    ) / len(sorted_validation)

    fixed_success_tasks = {
        int(row["task_id"])
        for row in oracle_summary["rows"] if row["success"]
    }
    fixed_success = np.isin(task_id, sorted(fixed_success_tasks))
    original_success = np.isin(task_id, ORIGINAL_ORACLE_SUCCESSES)
    subsets = [
        _summarize(
            "all_fixed_contract_commands",
            np.ones(len(task_id), bool),
            nll,
            percentile,
            nearest_mode,
            mixture_mean_error,
            endpoint_mass_0p8,
            endpoint_mass_sigma,
            predicted_stall,
            observed_stall,
        ),
        _summarize(
            "commands_on_original_oracle_success_tasks",
            original_success,
            nll,
            percentile,
            nearest_mode,
            mixture_mean_error,
            endpoint_mass_0p8,
            endpoint_mass_sigma,
            predicted_stall,
            observed_stall,
        ),
        _summarize(
            "commands_on_fixed_contract_success_tasks",
            fixed_success,
            nll,
            percentile,
            nearest_mode,
            mixture_mean_error,
            endpoint_mass_0p8,
            endpoint_mass_sigma,
            predicted_stall,
            observed_stall,
        ),
    ]
    successful = subsets[-1]
    report = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_oracle_commands": str(dataset.resolve()),
        "source_oracle_commands_sha256": sha256_file(dataset),
        "oracle_commands_used_for_training_or_selection": False,
        "spatial_parameterization": "endpoint_xy_minus_command",
        "posture_conditioning": "state_command_and_sampled_endpoint_xy",
        "commands": int(len(task_id)),
        "diagnostic_samples_per_command": DIAGNOSTIC_SAMPLES,
        "validation_reference_commands": int(len(validation_index)),
        "validation_spatial_nll": {
            "median": float(np.median(validation_nll)),
            "p90": float(np.quantile(validation_nll, 0.9)),
            "p95": float(np.quantile(validation_nll, 0.95)),
        },
        "subsets": subsets,
        "comparison_to_frozen_joint_model": {
            "joint_model_success_endpoint_mass_within_0p8_mean": (
                BASELINE_SUCCESS_MASS_WITHIN_0P8
            ),
            "factorized_success_endpoint_mass_within_0p8_mean": successful[
                "predicted_endpoint_mass_within_0p8_mean"
            ],
            "endpoint_mass_ratio": float(
                successful["predicted_endpoint_mass_within_0p8_mean"]
                / BASELINE_SUCCESS_MASS_WITHIN_0P8
            ),
            "joint_model_success_stall_brier": BASELINE_SUCCESS_STALL_BRIER,
            "factorized_success_stall_brier": successful["stall_brier"],
        },
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "interpretation_policy": (
            "spatial_endpoint_mass_and_stall_calibration_are_reported_"
            "separately_without_a_post_hoc_pass_threshold"
        ),
    }
    rows = []
    for index in range(len(task_id)):
        rows.append({
            "task_id": int(task_id[index]),
            "command_index": int(command_index[index]),
            "observed_stall": bool(observed_stall[index]),
            "predicted_stall_probability": float(predicted_stall[index]),
            "spatial_nll": float(nll[index]),
            "validation_spatial_nll_percentile": float(percentile[index]),
            "nearest_spatial_mode_error_m": float(nearest_mode[index]),
            "spatial_mixture_mean_error_m": float(
                mixture_mean_error[index]
            ),
            "predicted_endpoint_mass_within_0p8": float(
                endpoint_mass_0p8[index]
            ),
            "predicted_endpoint_mass_within_landing_sigma": float(
                endpoint_mass_sigma[index]
            ),
            "fixed_contract_success_task": bool(fixed_success[index]),
        })
    (output / "factorized_model_command_diagnostic.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    dump(output / "factorized_model_diagnostic.json", report)
    print(json.dumps(report, indent=2))


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=808)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--round", type=int, default=2)
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    train.set_defaults(func=command_train)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--round", type=int, default=2)
    diagnose.set_defaults(func=command_diagnose)
    return parser


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
