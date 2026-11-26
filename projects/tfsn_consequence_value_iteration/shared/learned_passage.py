"""One cross-domain model class and training rule for primitive passages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import torch
from torch import nn


MODEL_CLASS = "shared.learned_passage.PassageMLP"
LOSS_NAME = "mean_squared_error_on_sigma_normalized_relative_passage_bins"
FEATURE_RULE = "concat(adapter_state,command_xy,command_xy-phi_xy(adapter_state))"
TARGET_RULE = "primitive_time_groups_mean_xy_minus_start_xy_divided_by_landing_sigma"
TRAINING_RULE = (
    "grouped_holdout_independent_initializations_adam_manifest_cap_"
    "validation_early_stopping"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passage_features(states: np.ndarray, commands: np.ndarray) -> np.ndarray:
    states = np.asarray(states, np.float32)
    commands = np.asarray(commands, np.float32).reshape(-1, 2)
    if states.ndim != 2 or len(states) != len(commands) or states.shape[1] < 2:
        raise ValueError("adapter states and xy commands must be aligned")
    return np.concatenate(
        (states, commands, commands - states[:, :2]), axis=1
    ).astype(np.float32)


def passage_targets(
    states: np.ndarray,
    primitive_xy: np.ndarray,
    sigma: float,
    passage_bins: int,
) -> np.ndarray:
    states = np.asarray(states, np.float32)
    primitive = np.asarray(primitive_xy, np.float32)
    if primitive.ndim != 3 or primitive.shape[2] != 2:
        raise ValueError("primitive passages must have shape [N,steps,2]")
    if len(states) != len(primitive) or states.shape[1] < 2:
        raise ValueError("adapter states and primitive passages must align")
    if int(passage_bins) <= 0 or int(passage_bins) > primitive.shape[1]:
        raise ValueError("passage bin count must lie within primitive length")
    if not np.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError("landing sigma must be finite and positive")
    groups = np.array_split(np.arange(primitive.shape[1]), int(passage_bins))
    centers = np.stack([primitive[:, group].mean(axis=1) for group in groups], 1)
    return ((centers - states[:, None, :2]) / float(sigma)).astype(np.float32)


def grouped_split(
    group_id: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    group = np.asarray(group_id).reshape(-1)
    unique = np.unique(group)
    if len(unique) < 2:
        raise ValueError("passage training needs at least two trajectory groups")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation fraction must lie strictly between zero and one")
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    count = min(len(unique) - 1, max(1, int(round(len(unique) * validation_fraction))))
    validation_groups = shuffled[:count]
    validation = np.isin(group, validation_groups)
    training = ~validation
    if not validation.any() or not training.any():
        raise ValueError("grouped passage split produced an empty partition")
    return np.flatnonzero(training), np.flatnonzero(validation)


class PassageMLP(nn.Module):
    """The only deployed passage-network class in either domain adapter."""

    def __init__(
        self,
        input_dim: int,
        passage_bins: int,
        hidden_dimension: int,
        hidden_layers: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        width = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend((nn.Linear(width, int(hidden_dimension)), nn.SiLU()))
            width = int(hidden_dimension)
        layers.append(nn.Linear(width, int(passage_bins) * 2))
        self.network = nn.Sequential(*layers)
        self.passage_bins = int(passage_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).reshape(-1, self.passage_bins, 2)


def passage_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The single shared loss used by PointMaze and AntMaze."""

    return torch.mean(torch.square(prediction - target))


@dataclass(frozen=True)
class PassageTrainingConfig:
    passage_bins: int
    ensemble_size: int
    hidden_dimension: int
    hidden_layers: int
    batch_size: int
    epochs: int
    learning_rate: float
    validation_fraction: float
    std_floor: float
    seed: int
    early_stopping_patience: int = 10
    early_stopping_relative_improvement: float = 1e-4
    progress_every_epochs: int = 1


def _normalization(value: np.ndarray, index: np.ndarray, floor: float):
    mean = value[index].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = value[index].std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, float(floor)).astype(np.float32)


def train_passage_ensemble(
    *,
    states: np.ndarray,
    commands: np.ndarray,
    primitive_xy: np.ndarray,
    group_id: np.ndarray,
    sigma: float,
    config: PassageTrainingConfig,
    artifact_path: Path,
    metadata_path: Path,
    provenance: dict,
    device: str,
    manifest_path: Optional[Path] = None,
    progress_label: str = "r49",
) -> dict:
    """Train and save the same passage estimator under either adapter."""

    feature = passage_features(states, commands)
    target = passage_targets(states, primitive_xy, sigma, config.passage_bins)
    training, validation = grouped_split(
        group_id, config.validation_fraction, config.seed
    )
    feature_mean, feature_std = _normalization(
        feature, training, config.std_floor
    )
    normalized = (feature - feature_mean) / feature_std
    device_value = torch.device(device)
    x = torch.as_tensor(normalized, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.float32)
    members, validation_rows = [], []
    progress = f"[{progress_label}][passage-train]"
    print(
        f"{progress} "
        f"device={device_value} rows={len(states)} train={len(training)} "
        f"validation={len(validation)} members={config.ensemble_size} "
        f"max_epochs={config.epochs} patience={config.early_stopping_patience}",
        flush=True,
    )
    training_started = time.monotonic()
    for member in range(int(config.ensemble_size)):
        member_seed = int(config.seed) + member
        torch.manual_seed(member_seed)
        if device_value.type == "cuda":
            torch.cuda.manual_seed_all(member_seed)
        model = PassageMLP(
            feature.shape[1], config.passage_bins,
            config.hidden_dimension, config.hidden_layers,
        ).to(device_value)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        rng = np.random.default_rng(member_seed)
        best_loss, best_state = np.inf, None
        significant_best = np.inf
        stale_epochs = 0
        member_started = time.monotonic()
        print(
            f"{progress} member={member + 1}/{config.ensemble_size} start",
            flush=True,
        )
        for epoch in range(int(config.epochs)):
            model.train()
            order = rng.permutation(training)
            for begin in range(0, len(order), int(config.batch_size)):
                row = order[begin:begin + int(config.batch_size)]
                prediction = model(x[row].to(device_value))
                loss = passage_loss(prediction, y[row].to(device_value))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            losses = []
            with torch.no_grad():
                for begin in range(0, len(validation), int(config.batch_size)):
                    row = validation[begin:begin + int(config.batch_size)]
                    losses.append(float(passage_loss(
                        model(x[row].to(device_value)), y[row].to(device_value)
                    ).cpu()))
            value = float(np.mean(losses))
            if value < best_loss:
                best_loss = value
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            required = (
                significant_best
                * (1.0 - float(config.early_stopping_relative_improvement))
            )
            if not np.isfinite(significant_best) or value < required:
                significant_best = value
                stale_epochs = 0
            else:
                stale_epochs += 1
            epoch_number = epoch + 1
            should_report = (
                epoch_number == 1
                or epoch_number % int(config.progress_every_epochs) == 0
                or stale_epochs >= int(config.early_stopping_patience)
                or epoch_number == int(config.epochs)
            )
            if should_report:
                print(
                    f"{progress} "
                    f"member={member + 1}/{config.ensemble_size} "
                    f"epoch={epoch_number}/{config.epochs} "
                    f"validation_mse={value:.8g} best={best_loss:.8g} "
                    f"stale={stale_epochs}/{config.early_stopping_patience} "
                    f"elapsed_s={time.monotonic() - member_started:.1f}",
                    flush=True,
                )
            if stale_epochs >= int(config.early_stopping_patience):
                print(
                    f"{progress} "
                    f"member={member + 1}/{config.ensemble_size} "
                    f"early_stop_epoch={epoch_number}",
                    flush=True,
                )
                break
        if best_state is None:
            raise RuntimeError("passage training did not produce a checkpoint")
        members.append(best_state)
        validation_rows.append({
            "member": member,
            "best_validation_mse": best_loss,
            "epochs_completed": epoch_number,
            "stopped_early": epoch_number < int(config.epochs),
        })
        print(
            f"{progress} "
            f"member={member + 1}/{config.ensemble_size} done "
            f"epochs={epoch_number} best_validation_mse={best_loss:.8g}",
            flush=True,
        )
    artifact_path, metadata_path = Path(artifact_path), Path(metadata_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_class": MODEL_CLASS,
        "input_dim": feature.shape[1],
        "state_dim": np.asarray(states).shape[1],
        "passage_bins": int(config.passage_bins),
        "hidden_dimension": int(config.hidden_dimension),
        "hidden_layers": int(config.hidden_layers),
        "ensemble_size": int(config.ensemble_size),
        "landing_sigma": float(sigma),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "members": members,
    }
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for member_state in members for tensor in member_state.values()
    )
    required_free = max(8 * 1024 * 1024, 2 * tensor_bytes)
    available_free = shutil.disk_usage(artifact_path.parent).free
    print(
        f"{progress} "
        f"checkpoint_capacity available_bytes={available_free} "
        f"required_bytes={required_free}",
        flush=True,
    )
    if available_free < required_free:
        raise OSError(
            "insufficient free space for atomic passage checkpoint: "
            f"available={available_free} required={required_free}"
        )
    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, artifact_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest_path = (
        Path(__file__).with_name("method_manifest_r12.json")
        if manifest_path is None else Path(manifest_path)
    )
    manifest = json.loads(manifest_path.read_text())
    bound_provenance = {
        **dict(provenance),
        "shared_method_manifest": str(manifest_path),
        "shared_method_manifest_sha256": sha256_file(manifest_path),
        "shared_method_revision": manifest["method_revision"],
    }
    metadata = {
        "schema_version": 1,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "model_class": MODEL_CLASS,
        "loss": LOSS_NAME,
        "feature_rule": FEATURE_RULE,
        "target_rule": TARGET_RULE,
        "training_rule": TRAINING_RULE,
        "rows": int(len(states)),
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "landing_sigma_data_derived": float(sigma),
        "config": config.__dict__,
        "validation": validation_rows,
        "reward_used": False,
        "hidden_goal_used": False,
        "oracle_inputs": [],
        "provenance": bound_provenance,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"{progress} "
        f"saved={artifact_path} total_elapsed_s={time.monotonic() - training_started:.1f}",
        flush=True,
    )
    return metadata


class LearnedPassageEnsemble:
    """Read-only runtime for artifacts made by ``train_passage_ensemble``."""

    def __init__(
        self,
        artifact_path: Path,
        device: str,
        manifest_path: Optional[Path] = None,
    ):
        artifact_path = Path(artifact_path).resolve()
        metadata_path = artifact_path.with_name("passage.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"learned passage metadata is required beside checkpoint: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "model_class": MODEL_CLASS,
            "loss": LOSS_NAME,
            "training_rule": TRAINING_RULE,
            "reward_used": False,
            "hidden_goal_used": False,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"learned passage metadata has another {key}")
        if metadata.get("artifact_sha256") != sha256_file(artifact_path):
            raise ValueError("learned passage checkpoint hash changed")
        manifest = (
            Path(__file__).with_name("method_manifest_r12.json")
            if manifest_path is None else Path(manifest_path)
        )
        provenance = metadata.get("provenance", {})
        if provenance.get("shared_method_manifest_sha256") != sha256_file(manifest):
            raise ValueError("learned passage checkpoint has a stale manifest hash")
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if payload.get("model_class") != MODEL_CLASS:
            raise ValueError("passage artifact uses another model class")
        self.state_dim = int(payload["state_dim"])
        self.passage_bins = int(payload["passage_bins"])
        self.sigma = float(payload["landing_sigma"])
        self.device = torch.device(device)
        self.feature_mean = np.asarray(payload["feature_mean"], np.float32)
        self.feature_std = np.asarray(payload["feature_std"], np.float32)
        self.members = []
        for state in payload["members"]:
            model = PassageMLP(
                payload["input_dim"], self.passage_bins,
                payload["hidden_dimension"], payload["hidden_layers"],
            ).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.members.append(model)

    def predict(self, states: np.ndarray, commands: np.ndarray) -> np.ndarray:
        states = np.asarray(states, np.float32).reshape(-1, self.state_dim)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        feature = (passage_features(states, commands) - self.feature_mean) / self.feature_std
        value = torch.as_tensor(feature, dtype=torch.float32, device=self.device)
        rows = []
        with torch.no_grad():
            for model in self.members:
                relative = model(value).cpu().numpy() * self.sigma
                path = states[:, None, :2] + relative
                rows.append(np.concatenate((states[:, None, :2], path), axis=1))
        # [ensemble,candidate,passage_point,xy]
        return np.stack(rows).astype(np.float32)

