"""Factorized mixture consequence model shared across continuous domains."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import ndtr

from .settings import (
    BATCH_SIZE,
    ENSEMBLE_SIZE,
    EPOCHS,
    HIDDEN_DIMENSION,
    HIDDEN_LAYERS,
    GAUSSIAN_CIRCLE_QUADRATURE_POINTS,
    LEARNING_RATE,
    MIXTURE_COMPONENTS,
    STD_FLOOR,
)


_CIRCLE_X, _CIRCLE_WEIGHT = np.polynomial.legendre.leggauss(
    GAUSSIAN_CIRCLE_QUADRATURE_POINTS
)


@dataclass(frozen=True)
class MDNConfig:
    input_dim: int
    output_dim: int
    components: int = MIXTURE_COMPONENTS
    hidden_dim: int = HIDDEN_DIMENSION
    hidden_layers: int = HIDDEN_LAYERS
    ensemble_size: int = ENSEMBLE_SIZE
    std_floor: float = STD_FLOOR


class MixtureDensityNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        width = int(config.input_dim)
        for _ in range(int(config.hidden_layers)):
            layers.extend((nn.Linear(width, config.hidden_dim), nn.SiLU()))
            width = int(config.hidden_dim)
        self.trunk = nn.Sequential(*layers)
        self.logits = nn.Linear(width, config.components)
        self.means = nn.Linear(
            width, config.components * config.output_dim
        )
        self.raw_stds = nn.Linear(
            width, config.components * config.output_dim
        )
        self.config = config

    def forward(self, inputs):
        hidden = self.trunk(inputs)
        shape = (
            *inputs.shape[:-1],
            self.config.components,
            self.config.output_dim,
        )
        return (
            self.logits(hidden),
            self.means(hidden).reshape(shape),
            torch.nn.functional.softplus(
                self.raw_stds(hidden).reshape(shape)
            ) + self.config.std_floor,
        )


def mixture_nll(logits, means, stds, target):
    target = target.unsqueeze(-2)
    component = -0.5 * (((target - means) / stds) ** 2)
    component = component - torch.log(stds) - 0.5 * math.log(2.0 * math.pi)
    component = component.sum(dim=-1)
    return -torch.logsumexp(
        torch.log_softmax(logits, dim=-1) + component, dim=-1
    )


def diagonal_gaussian_circle_probability(means, stds, centers, radius):
    """Deterministic mass of diagonal 2-D Gaussians inside circles."""

    means = np.asarray(means, np.float64)
    stds = np.asarray(stds, np.float64)
    centers = np.asarray(centers, np.float64).reshape(-1, 1, 2)
    radius = float(radius)
    if means.shape != stds.shape or means.ndim != 3 or means.shape[2] != 2:
        raise ValueError("circle probability expects [batch, component, xy]")
    if len(centers) != len(means) or radius <= 0.0:
        raise ValueError("circle centers and positive radius are required")
    quadrature_x = _CIRCLE_X[None, None]
    absolute_x = centers[..., 0, None] + radius * quadrature_x
    half_height = radius * np.sqrt(1.0 - np.square(quadrature_x))
    standardized_x = (absolute_x - means[..., 0, None]) / stds[..., 0, None]
    density_x = (
        np.exp(-0.5 * np.square(standardized_x))
        / (np.sqrt(2.0 * np.pi) * stds[..., 0, None])
    )
    upper = (
        centers[..., 1, None] + half_height - means[..., 1, None]
    ) / stds[..., 1, None]
    lower = (
        centers[..., 1, None] - half_height - means[..., 1, None]
    ) / stds[..., 1, None]
    probability = radius * np.sum(
        _CIRCLE_WEIGHT[None, None]
        * density_x
        * (ndtr(upper) - ndtr(lower)),
        axis=2,
    )
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def diagonal_gaussian_kernel_expectation(means, stds, centers, sigma):
    """Exact E[exp(-||X-center||^2/(2 sigma^2))] per component."""
    means = np.asarray(means, np.float32)
    stds = np.asarray(stds, np.float32)
    centers = np.asarray(centers, np.float32)
    if means.shape != stds.shape or means.shape[-1] != 2:
        raise ValueError("kernel expectation requires aligned 2-D Gaussians")
    if centers.shape == means.shape[:-2] + (2,):
        centers = centers[..., None, :]
    if centers.shape != means.shape[:-1] + (2,):
        centers = np.broadcast_to(centers, means.shape)
    kernel_variance = float(sigma) ** 2
    combined = kernel_variance + np.square(stds)
    scale = np.sqrt(kernel_variance / combined).prod(axis=-1)
    exponent = -0.5 * (np.square(means - centers) / combined).sum(axis=-1)
    return (scale * np.exp(exponent)).astype(np.float32)


def consequence_features(states, commands):
    states = np.asarray(states, np.float32)
    commands = np.asarray(commands, np.float32).reshape(-1, 2)
    if states.ndim != 2 or len(states) != len(commands):
        raise ValueError("states and commands must be aligned matrices")
    delta = commands - states[:, :2]
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.concatenate((states, commands, delta, distance), axis=1).astype(
        np.float32
    )


def _safe_std(values):
    return np.maximum(np.asarray(values, np.float32).std(0), 1e-6).astype(
        np.float32
    )


def _normalize(values, mean, std):
    return (np.asarray(values, np.float32) - mean) / std


def _raw_parameters(network, inputs, target_mean, target_std):
    logits, means, stds = network(inputs)
    return (
        torch.softmax(logits, dim=-1),
        means * target_std[None, None] + target_mean[None, None],
        stds * target_std[None, None],
    )


def train_head(
    inputs,
    targets,
    training_index,
    validation_index,
    seed,
    device,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    label="head",
    sample_weight=None,
):
    inputs = np.asarray(inputs, np.float32)
    targets = np.asarray(targets, np.float32)
    training_index = np.asarray(training_index, np.int64)
    validation_index = np.asarray(validation_index, np.int64)
    if sample_weight is None:
        sample_weight = np.ones(len(inputs), np.float32)
    sample_weight = np.asarray(sample_weight, np.float32).reshape(-1)
    if len(sample_weight) != len(inputs) or np.any(sample_weight <= 0.0):
        raise ValueError("sample weights must be positive and aligned")
    training_weight = sample_weight[training_index].astype(np.float64)
    training_weight /= training_weight.sum()
    input_mean = np.average(
        inputs[training_index], axis=0, weights=training_weight
    ).astype(np.float32)
    input_variance = np.average(
        np.square(inputs[training_index] - input_mean),
        axis=0,
        weights=training_weight,
    )
    input_std = np.maximum(np.sqrt(input_variance), 1e-6).astype(np.float32)
    target_mean = np.average(
        targets[training_index], axis=0, weights=training_weight
    ).astype(np.float32)
    target_variance = np.average(
        np.square(targets[training_index] - target_mean),
        axis=0,
        weights=training_weight,
    )
    target_std = np.maximum(np.sqrt(target_variance), 1e-6).astype(np.float32)
    x = torch.as_tensor(
        _normalize(inputs, input_mean, input_std), dtype=torch.float32
    )
    y = torch.as_tensor(
        _normalize(targets, target_mean, target_std), dtype=torch.float32
    )
    config = MDNConfig(inputs.shape[1], targets.shape[1])
    states, histories = [], []
    for member in range(config.ensemble_size):
        member_seed = int(seed + 10_000 * (member + 1))
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
        model = MixtureDensityNetwork(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed + 100_000 * (member + 1))
        # The ensemble differs through an explicit bootstrap, not only its
        # initialization.  This is the uncertainty source required by B.3.
        bootstrap = rng.choice(
            training_index,
            len(training_index),
            replace=True,
            p=training_weight,
        )
        best_loss, best_state, history = math.inf, None, []
        for epoch in range(int(epochs)):
            order = rng.permutation(bootstrap)
            model.train()
            losses = []
            for begin in range(0, len(order), int(batch_size)):
                index = order[begin:begin + int(batch_size)]
                optimizer.zero_grad(set_to_none=True)
                loss = mixture_nll(
                    *model(x[index].to(device)), y[index].to(device)
                ).mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            validation_losses = []
            with torch.no_grad():
                for begin in range(0, len(validation_index), 4096):
                    index = validation_index[begin:begin + 4096]
                    validation_losses.append(mixture_nll(
                        *model(x[index].to(device)),
                        y[index].to(device),
                    ).cpu().numpy())
            validation_values = np.concatenate(validation_losses)
            validation_loss = float(np.average(
                validation_values,
                weights=sample_weight[validation_index],
            ))
            history.append({
                "epoch": epoch + 1,
                "train_normalized_nll": float(np.mean(losses)),
                "validation_normalized_nll": validation_loss,
            })
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 25 == 0 or epoch + 1 == int(epochs):
                print(
                    f"{label} member={member + 1}/{config.ensemble_size} "
                    f"epoch={epoch + 1}/{epochs} val_nll={validation_loss:.6f}",
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
        "config": asdict(config),
        "member_states": states,
        "histories": histories,
    }


class FactorizedConsequenceEnsemble:
    def __init__(self, checkpoint, device):
        payload = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=False
        )
        if payload.get("artifact_type") != "factorized_consequence":
            raise ValueError("not a shared factorized consequence checkpoint")
        self.payload = payload
        self.device = torch.device(device)
        self.state_dim = int(payload["state_dim"])
        self.posture_dim = self.state_dim - 2
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
        self.spatial_members, self.posture_members = [], []
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
            raise ValueError("factorized ensemble heads do not align")

    def spatial_parameters(self, features, member):
        normalized = _normalize(features, self.input_mean, self.input_std)
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
        return tuple(
            value.cpu().numpy().astype(np.float32)
            for value in (weights, means, stds)
        )

    def posture_parameters(self, features, endpoint_xy, member):
        features = np.asarray(features, np.float32)
        endpoint_xy = np.asarray(endpoint_xy, np.float32)
        batch, outcomes, _ = endpoint_xy.shape
        repeated = np.repeat(features[:, None], outcomes, axis=1)
        conditioned = np.concatenate((repeated, endpoint_xy), axis=2).reshape(
            batch * outcomes, -1
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
            weights.reshape(batch, outcomes, components).cpu().numpy(),
            means.reshape(
                batch, outcomes, components, self.posture_dim
            ).cpu().numpy(),
            stds.reshape(
                batch, outcomes, components, self.posture_dim
            ).cpu().numpy(),
        )

    def component_mean_endpoints(self, states, commands, member):
        features = consequence_features(states, commands)
        weights, residual, spatial_std = self.spatial_parameters(
            features, member
        )
        endpoint_xy = commands[:, None] + residual
        posture_weight, posture_mean, _ = self.posture_parameters(
            features, endpoint_xy, member
        )
        posture = (posture_weight[..., None] * posture_mean).sum(axis=2)
        return (
            weights,
            np.concatenate((endpoint_xy, posture), axis=2).astype(np.float32),
            spatial_std,
        )

    def component_mean_endpoint_outcomes(self, states, commands, member):
        """Joint component means without averaging spatial or posture modes."""

        states = np.asarray(states, np.float32)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        features = consequence_features(states, commands)
        spatial_weight, residual, _spatial_std = self.spatial_parameters(
            features, member
        )
        endpoint_xy = commands[:, None] + residual
        posture_weight, posture_mean, _posture_std = self.posture_parameters(
            features, endpoint_xy, member
        )
        posture_components = posture_weight.shape[2]
        xy = np.repeat(endpoint_xy[:, :, None], posture_components, axis=2)
        joint_weight = spatial_weight[:, :, None] * posture_weight
        endpoint = np.concatenate((xy, posture_mean), axis=3)
        return (
            joint_weight.reshape(len(states), -1).astype(np.float32),
            endpoint.reshape(len(states), -1, self.state_dim).astype(np.float32),
        )

    def stall_probability(self, states, commands, sigma, member):
        """Exact Gaussian-kernel expectation under the spatial mixture."""
        features = consequence_features(states, commands)
        weights, residual, spatial_std = self.spatial_parameters(
            features, member
        )
        means = commands[:, None] + residual
        component = diagonal_gaussian_kernel_expectation(
            means, spatial_std, np.asarray(states, np.float32)[:, :2], sigma
        )
        return (weights * component).sum(axis=1)


def landing_precision(payload, validation_features, device):
    """B.3 within-mode spatial spread of the deployed ensemble."""
    temporary = Path(payload["_temporary_checkpoint"])
    torch.save({k: v for k, v in payload.items() if k != "_temporary_checkpoint"}, temporary)
    try:
        model = FactorizedConsequenceEnsemble(temporary, device)
        member_values = []
        for member in range(len(model.spatial_members)):
            weights, _means, stds = model.spatial_parameters(
                validation_features, member
            )
            radial = np.sqrt(0.5 * np.square(stds).sum(axis=2))
            member_values.append((weights * radial).sum(axis=1))
        return float(np.stack(member_values).mean())
    finally:
        temporary.unlink(missing_ok=True)

