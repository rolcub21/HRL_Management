"""Generate, train, and calibrate a dynamics-only viability-critic ensemble.

This is a development/calibration pipeline, not a sealed task-policy holdout.
It deliberately has no import or query path to REG, PSLAP, GA, nearest-free,
or another behavior teacher.  The learned ensemble may prioritize exact
checks; it never replaces the exact recoverability certificate as the hard
safety authority.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from PSLAP.viability import (
    CLOSED_ADMISSION_CONTRACT,
    STRICT_MACRO_ACTION_MODEL,
    ViabilityStatus,
)
from PSLAP.viability_critic import (
    CounterfactualViabilityCritic,
    DYNAMICS_LABEL_CONTRACT,
    ViabilityCriticEnsemble,
    compute_viability_critic_loss,
)
from PSLAP.viability_dataset import (
    DEFAULT_SPLIT_RATIOS,
    VIABILITY_DATASET_PROTOCOL,
    ViabilityExample,
    canonical_sha256,
    dataset_manifest,
    generate_examples,
    generate_layouts,
    read_dataset,
    split_examples_by_layout,
    stratified_bootstrap_indices,
    write_dataset,
)
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    YardGraph,
    encode_recovery_state,
    pad_yard_graphs,
)


TRAINER_VERSION = "dynamics_viability_critic_trainer_v1"
CHECKPOINT_SCHEMA_VERSION = "viability_critic_checkpoint_v1"
MODEL_SELECTION_CONTRACT = (
    "validation_lcb_at_0.5_min_false_safe_then_max_safe_coverage_v1"
)
CALIBRATION_CONTRACT = (
    "calibration_lcb_threshold_then_independent_test_binomial_upper_v1"
)


def parse_grid_size(value: str) -> tuple[int, int]:
    try:
        rows, cols = value.lower().split("x", 1)
        rows, cols = int(rows), int(cols)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "grid size must use ROWSxCOLS, for example 5x6"
        ) from error
    if rows < 4 or cols < 4:
        raise argparse.ArgumentTypeError("grid dimensions must be at least 4")
    return rows, cols


def optional_positive(value: str) -> Optional[int]:
    number = int(value)
    if number == 0:
        return None
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or positive")
    return number


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _binomial_log_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 0.0 if k >= n else -math.inf
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * math.log(probability)
        + (n - index) * math.log1p(-probability)
        for index in range(k + 1)
    ]
    maximum = max(logs)
    return maximum + math.log(sum(math.exp(value - maximum) for value in logs))


def one_sided_binomial_upper_bound(
    failures: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Clopper--Pearson upper confidence bound."""

    if (
        isinstance(failures, bool)
        or isinstance(trials, bool)
        or not isinstance(failures, int)
        or not isinstance(trials, int)
        or failures < 0
        or trials < 0
        or failures > trials
    ):
        raise ValueError("failures/trials must satisfy 0 <= failures <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if trials == 0 or failures == trials:
        return 1.0
    alpha = 1.0 - confidence
    if failures == 0:
        return 1.0 - alpha ** (1.0 / trials)
    target = math.log(alpha)
    low = failures / trials
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_log_cdf(failures, trials, midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return high


def _binary_auroc(targets: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(targets), dtype=np.float64)
    start = 0
    while start < len(targets):
        end = start + 1
        while (
            end < len(targets)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[targets == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(targets: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(targets.sum())
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def _expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> Optional[float]:
    if not len(targets):
        return None
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (
            (probabilities >= lower)
            & (probabilities <= upper if index == bins - 1 else probabilities < upper)
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(targets[mask].mean())
            )
    return result


@dataclass(frozen=True)
class EnsemblePredictions:
    safety_mean: np.ndarray
    safety_std: np.ndarray
    safety_lcb: np.ndarray
    recovery_rank_mean: np.ndarray
    primitive_steps_mean: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.safety_mean)
        if any(
            len(value) != length
            for value in (
                self.safety_std,
                self.safety_lcb,
                self.recovery_rank_mean,
                self.primitive_steps_mean,
            )
        ):
            raise ValueError("prediction arrays must have equal lengths")


def predict_ensemble(
    ensemble: ViabilityCriticEnsemble,
    examples: Sequence[ViabilityExample],
    *,
    batch_size: int,
    timing_scale: float,
    lcb_scale: float,
    graphs: Optional[Sequence[YardGraph]] = None,
) -> EnsemblePredictions:
    examples = tuple(examples)
    if not examples:
        empty = np.asarray([], dtype=np.float64)
        return EnsemblePredictions(empty, empty, empty, empty, empty)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if graphs is not None and len(graphs) != len(examples):
        raise ValueError("preencoded graph count differs from examples")
    ensemble.eval()
    values = {
        "safety_mean": [],
        "safety_std": [],
        "safety_lcb": [],
        "recovery_rank_mean": [],
        "primitive_steps_mean": [],
    }
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = pad_yard_graphs(
                (
                    tuple(graphs[start : start + batch_size])
                    if graphs is not None
                    else tuple(
                        encode_recovery_state(
                            example.state,
                            timing_scale=timing_scale,
                        )
                        for example in chunk
                    )
                )
            )
            estimate = ensemble.estimate(batch, lcb_scale=lcb_scale)
            values["safety_mean"].extend(
                estimate.safety_probability_mean.detach().cpu().tolist()
            )
            values["safety_std"].extend(
                estimate.safety_probability_std.detach().cpu().tolist()
            )
            values["safety_lcb"].extend(
                estimate.safety_probability_lcb.detach().cpu().tolist()
            )
            values["recovery_rank_mean"].extend(
                estimate.recovery_rank_mean.detach().cpu().tolist()
            )
            values["primitive_steps_mean"].extend(
                estimate.primitive_steps_mean.detach().cpu().tolist()
            )
    return EnsemblePredictions(
        **{
            name: np.asarray(items, dtype=np.float64)
            for name, items in values.items()
        }
    )


def evaluate_predictions(
    examples: Sequence[ViabilityExample],
    predictions: EnsemblePredictions,
    *,
    threshold: float,
    confidence: float,
) -> dict:
    examples = tuple(examples)
    if len(examples) != len(predictions.safety_lcb):
        raise ValueError("example and prediction counts differ")
    known_indices = [
        index for index, example in enumerate(examples) if example.label.has_safety_target
    ]
    targets = np.asarray(
        [int(examples[index].label.status is ViabilityStatus.SAFE) for index in known_indices],
        dtype=np.int64,
    )
    probabilities = predictions.safety_lcb[known_indices]
    accepted = probabilities >= float(threshold)
    safe_mask = targets == 1
    unsafe_mask = targets == 0
    safe_count = int(safe_mask.sum())
    unsafe_count = int(unsafe_mask.sum())
    safe_accepted = int((accepted & safe_mask).sum())
    false_safe = int((accepted & unsafe_mask).sum())
    accepted_count = int(accepted.sum())
    clipped_mean = np.clip(predictions.safety_mean[known_indices], 1e-7, 1.0 - 1e-7)
    rank_errors = [
        abs(
            float(predictions.recovery_rank_mean[index])
            - float(example.label.recovery_rank)
        )
        for index, example in enumerate(examples)
        if example.label.recovery_rank is not None
    ]
    step_errors = [
        abs(
            float(predictions.primitive_steps_mean[index])
            - float(example.label.primitive_steps)
        )
        for index, example in enumerate(examples)
        if example.label.primitive_steps is not None
    ]
    return {
        "example_count": len(examples),
        "known_count": len(known_indices),
        "unknown_count": len(examples) - len(known_indices),
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "threshold": float(threshold),
        "accepted_safe_count": safe_accepted,
        "false_safe_count": false_safe,
        "accepted_count": accepted_count,
        "safe_coverage": safe_accepted / max(1, safe_count),
        "unsafe_false_safe_rate": false_safe / max(1, unsafe_count),
        "unsafe_false_safe_upper_bound": one_sided_binomial_upper_bound(
            false_safe,
            unsafe_count,
            confidence=confidence,
        ),
        "accepted_precision": safe_accepted / max(1, accepted_count),
        "unsafe_recall": (unsafe_count - false_safe) / max(1, unsafe_count),
        "auroc": _binary_auroc(targets, probabilities),
        "average_precision": _average_precision(targets, probabilities),
        "brier_score": (
            float(np.mean((clipped_mean - targets) ** 2)) if len(targets) else None
        ),
        "negative_log_likelihood": (
            float(
                -np.mean(
                    targets * np.log(clipped_mean)
                    + (1 - targets) * np.log(1.0 - clipped_mean)
                )
            )
            if len(targets)
            else None
        ),
        "expected_calibration_error": _expected_calibration_error(
            targets,
            clipped_mean,
        ),
        "mean_lcb": float(probabilities.mean()) if len(probabilities) else None,
        "mean_ensemble_std": (
            float(predictions.safety_std[known_indices].mean())
            if known_indices
            else None
        ),
        "recovery_rank_target_count": len(rank_errors),
        "recovery_rank_mae": float(np.mean(rank_errors)) if rank_errors else None,
        "primitive_steps_target_count": len(step_errors),
        "primitive_steps_mae": float(np.mean(step_errors)) if step_errors else None,
    }


def choose_lcb_threshold(
    examples: Sequence[ViabilityExample],
    predictions: EnsemblePredictions,
    *,
    confidence: float,
    max_false_safe_upper_bound: float,
    min_safe_accepted: int,
) -> dict:
    """Select maximum safe coverage under a false-safe evidence budget."""

    examples = tuple(examples)
    if not 0.0 < max_false_safe_upper_bound < 1.0:
        raise ValueError("max_false_safe_upper_bound must be in (0, 1)")
    if min_safe_accepted <= 0:
        raise ValueError("min_safe_accepted must be positive")
    known = [
        index for index, example in enumerate(examples) if example.label.has_safety_target
    ]
    safe = [
        index for index in known if examples[index].label.status is ViabilityStatus.SAFE
    ]
    unsafe = [
        index for index in known if examples[index].label.status is ViabilityStatus.UNSAFE
    ]
    fail_closed = {
        "ready": False,
        "threshold": 1.000001,
        "reason": "",
        "calibration_contract": CALIBRATION_CONTRACT,
        "critic_certificate_authority": False,
        "exact_verifier_authoritative": True,
    }
    if not unsafe:
        return {**fail_closed, "reason": "calibration split contains no exact UNSAFE evidence"}
    if len(safe) < min_safe_accepted:
        return {**fail_closed, "reason": "calibration split has insufficient SAFE examples"}

    candidates = sorted(
        set(float(predictions.safety_lcb[index]) for index in known),
        reverse=True,
    )
    feasible = []
    for threshold in candidates:
        false_safe = sum(
            predictions.safety_lcb[index] >= threshold for index in unsafe
        )
        safe_accepted = sum(
            predictions.safety_lcb[index] >= threshold for index in safe
        )
        upper = one_sided_binomial_upper_bound(
            int(false_safe), len(unsafe), confidence=confidence
        )
        if upper <= max_false_safe_upper_bound and safe_accepted >= min_safe_accepted:
            feasible.append(
                (int(safe_accepted), -int(false_safe), float(threshold), upper)
            )
    if not feasible:
        zero_error_upper = one_sided_binomial_upper_bound(
            0, len(unsafe), confidence=confidence
        )
        return {
            **fail_closed,
            "reason": "no threshold satisfies the false-safe evidence budget",
            "unsafe_count": len(unsafe),
            "best_possible_zero_error_upper_bound": zero_error_upper,
        }
    safe_accepted, negative_false_safe, threshold, upper = max(feasible)
    return {
        **fail_closed,
        "ready": True,
        "threshold": threshold,
        "reason": "calibration threshold selected",
        "unsafe_count": len(unsafe),
        "safe_count": len(safe),
        "safe_accepted": safe_accepted,
        "false_safe_count": -negative_false_safe,
        "false_safe_upper_bound": upper,
        "confidence": confidence,
        "maximum_false_safe_upper_bound": max_false_safe_upper_bound,
        "minimum_safe_accepted": min_safe_accepted,
    }


def _model_config(args: argparse.Namespace) -> dict:
    return {
        "input_dim": NODE_FEATURE_DIM,
        "graph_hidden_dim": args.graph_hidden_dim,
        "graph_embedding_dim": args.graph_embedding_dim,
        "message_passing_steps": args.message_passing_steps,
        "head_hidden_dim": args.head_hidden_dim,
    }


def _training_config(args: argparse.Namespace) -> dict:
    """Resume-critical optimization settings (the epoch ceiling may grow)."""

    return {
        "steps_per_epoch": int(args.steps_per_epoch),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_every": int(args.eval_every),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "false_safe_weight": float(args.false_safe_weight),
        "safety_weight": float(args.safety_weight),
        "recovery_rank_weight": float(args.recovery_rank_weight),
        "primitive_steps_weight": float(args.primitive_steps_weight),
        "huber_beta": float(args.huber_beta),
        "unsafe_sampling_fraction": float(args.unsafe_sampling_fraction),
        "lcb_scale": float(args.lcb_scale),
        "calibration_confidence": float(args.calibration_confidence),
        "max_false_safe_upper_bound": float(args.max_false_safe_upper_bound),
        "min_calibration_safe_accepted": int(
            args.min_calibration_safe_accepted
        ),
    }


def _make_members(
    *,
    ensemble_size: int,
    model_config: Mapping,
    seed: int,
    device: torch.device,
) -> tuple[tuple[CounterfactualViabilityCritic, ...], tuple[int, ...]]:
    members = []
    seeds = []
    for index in range(ensemble_size):
        member_seed = int(seed) + 100_003 * (index + 1)
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
        members.append(
            CounterfactualViabilityCritic(**dict(model_config)).to(device)
        )
        seeds.append(member_seed)
    return tuple(members), tuple(seeds)


def _bootstrap_members(
    examples: Sequence[ViabilityExample],
    *,
    ensemble_size: int,
    seed: int,
) -> tuple[tuple[dict[str, tuple[int, ...]], ...], tuple[int, ...], tuple[str, ...]]:
    pools = []
    seeds = []
    hashes = []
    for index in range(ensemble_size):
        bootstrap_seed = int(seed) + 700_001 + index * 100_019
        pool = stratified_bootstrap_indices(examples, seed=bootstrap_seed)
        payload = {
            "safe": list(pool["safe"]),
            "unsafe": list(pool["unsafe"]),
        }
        digest = canonical_sha256(payload)
        if digest in hashes:
            raise RuntimeError(
                "ensemble bootstrap samples are duplicated; enlarge training data"
            )
        pools.append(pool)
        seeds.append(bootstrap_seed)
        hashes.append(digest)
    return tuple(pools), tuple(seeds), tuple(hashes)


def _sample_batch(
    pool: Mapping[str, Sequence[int]],
    *,
    batch_size: int,
    unsafe_fraction: float,
    rng: random.Random,
) -> tuple[int, ...]:
    unsafe_count = min(batch_size - 1, max(1, round(batch_size * unsafe_fraction)))
    safe_count = batch_size - unsafe_count
    indices = [rng.choice(pool["unsafe"]) for _ in range(unsafe_count)]
    indices.extend(rng.choice(pool["safe"]) for _ in range(safe_count))
    rng.shuffle(indices)
    return tuple(indices)


def _train_epoch(
    *,
    epoch: int,
    members: Sequence[CounterfactualViabilityCritic],
    optimizers: Sequence[torch.optim.Optimizer],
    bootstraps: Sequence[Mapping[str, Sequence[int]]],
    bootstrap_seeds: Sequence[int],
    examples: Sequence[ViabilityExample],
    graphs: Sequence[YardGraph],
    args: argparse.Namespace,
) -> dict:
    totals = {
        "total": 0.0,
        "safety": 0.0,
        "recovery_rank": 0.0,
        "primitive_steps": 0.0,
        "updates": 0,
    }
    for member_index, (member, optimizer, pool) in enumerate(
        zip(members, optimizers, bootstraps)
    ):
        member.train()
        rng = random.Random(
            int(bootstrap_seeds[member_index]) + epoch * 1_000_003
        )
        for _ in range(args.steps_per_epoch):
            indices = _sample_batch(
                pool,
                batch_size=args.batch_size,
                unsafe_fraction=args.unsafe_sampling_fraction,
                rng=rng,
            )
            chunk = tuple(examples[index] for index in indices)
            batch = pad_yard_graphs(
                tuple(graphs[index] for index in indices)
            )
            output = member(batch)
            loss = compute_viability_critic_loss(
                output,
                tuple(example.label for example in chunk),
                false_safe_weight=args.false_safe_weight,
                safety_weight=args.safety_weight,
                recovery_rank_weight=args.recovery_rank_weight,
                primitive_steps_weight=args.primitive_steps_weight,
                huber_beta=args.huber_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                member.parameters(), args.grad_clip
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise RuntimeError("non-finite viability-critic gradient")
            optimizer.step()
            totals["total"] += float(loss.total.detach())
            totals["safety"] += float(loss.safety.detach())
            totals["recovery_rank"] += float(loss.recovery_rank.detach())
            totals["primitive_steps"] += float(loss.primitive_steps.detach())
            totals["updates"] += 1
    updates = max(1, totals["updates"])
    return {
        "total_loss": totals["total"] / updates,
        "safety_loss": totals["safety"] / updates,
        "recovery_rank_loss": totals["recovery_rank"] / updates,
        "primitive_steps_loss": totals["primitive_steps"] / updates,
        "updates": totals["updates"],
    }


def _state_dicts_cpu(members: Sequence[CounterfactualViabilityCritic]) -> list[dict]:
    return [
        {
            key: value.detach().cpu().clone()
            for key, value in member.state_dict().items()
        }
        for member in members
    ]


def _load_state_dicts(
    members: Sequence[CounterfactualViabilityCritic],
    state_dicts: Sequence[Mapping],
) -> None:
    if len(members) != len(state_dicts):
        raise ValueError("checkpoint ensemble size differs from configured model")
    for member, state_dict in zip(members, state_dicts):
        member.load_state_dict(state_dict)


def _split_from_manifest(
    examples: Sequence[ViabilityExample], manifest: Mapping
) -> dict[str, tuple[ViabilityExample, ...]]:
    split_manifest = manifest.get("split_manifest", {})
    raw_splits = split_manifest.get("splits", {})
    if set(raw_splits) != set(DEFAULT_SPLIT_RATIOS):
        raise ValueError("dataset manifest has unsupported split names")
    layout_to_split = {}
    for split_name, payload in raw_splits.items():
        for layout_id in payload.get("layout_ids", []):
            if layout_id in layout_to_split:
                raise ValueError("layout appears in more than one dataset split")
            layout_to_split[layout_id] = split_name
    splits = {name: [] for name in raw_splits}
    for example in examples:
        try:
            split_name = layout_to_split[example.layout_id]
        except KeyError as error:
            raise ValueError(
                f"example layout {example.layout_id} is absent from split manifest"
            ) from error
        splits[split_name].append(example)
    result = {
        name: tuple(sorted(values, key=lambda item: item.example_id))
        for name, values in splits.items()
    }
    for name, values in result.items():
        if len(values) != int(raw_splits[name]["example_count"]):
            raise ValueError(f"split {name} count does not match manifest")
    return result


def _validate_manifest(manifest: Mapping, dataset_path: Path) -> None:
    if manifest.get("protocol") != VIABILITY_DATASET_PROTOCOL:
        raise ValueError("unsupported dataset manifest protocol")
    if manifest.get("baseline_viability_teacher") is not False:
        raise ValueError("manifest does not declare baseline-free labels")
    if manifest.get("exact_verifier_authoritative") is not True:
        raise ValueError("manifest does not preserve exact-verifier authority")
    if manifest.get("critic_certificate_authority") is not False:
        raise ValueError("manifest improperly grants critic certificate authority")
    if manifest.get("label_contract") != DYNAMICS_LABEL_CONTRACT:
        raise ValueError("dataset label contract mismatch")
    if manifest.get("certificate_contract") != CLOSED_ADMISSION_CONTRACT:
        raise ValueError("dataset certificate contract mismatch")
    if manifest.get("action_model") != STRICT_MACRO_ACTION_MODEL:
        raise ValueError("dataset action-model mismatch")
    if manifest.get("max_primitive_steps", "missing") is not None:
        raise ValueError("horizon-conditioned viability data are unsupported")
    if manifest.get("node_feature_names") != list(NODE_FEATURE_NAMES):
        raise ValueError("dataset node feature schema mismatch")
    if manifest.get("node_feature_dim") != NODE_FEATURE_DIM:
        raise ValueError("dataset node feature dimension mismatch")
    expected = dict(manifest)
    supplied_digest = expected.pop("manifest_sha256", None)
    if supplied_digest != canonical_sha256(expected):
        raise ValueError("dataset manifest digest mismatch")
    if manifest.get("records_sha256") != _file_sha256(dataset_path):
        raise ValueError("dataset record digest mismatch")


def _prepare_data(args: argparse.Namespace) -> tuple[tuple[ViabilityExample, ...], dict]:
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "viability-dataset.jsonl"
    manifest_path = data_dir / "dataset-manifest.json"
    if args.resume is not None or args.reuse_data:
        if not dataset_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                "resume/reuse requires the original dataset artifacts"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest, dataset_path)
        if float(manifest.get("timing_scale")) != float(args.timing_scale):
            raise ValueError("--timing-scale differs from the stored dataset")
        examples = read_dataset(dataset_path)
        return examples, manifest
    if args.dataset is not None:
        if args.dataset_manifest is None:
            raise ValueError("--dataset requires --dataset-manifest")
        examples = read_dataset(args.dataset)
        manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        _validate_manifest(manifest, args.dataset)
        if float(manifest.get("timing_scale")) != float(args.timing_scale):
            raise ValueError("--timing-scale differs from the supplied dataset")
        record_digest = write_dataset(dataset_path, examples)
        copied = dict(manifest)
        copied.pop("manifest_sha256", None)
        copied["records_sha256"] = record_digest
        copied["manifest_sha256"] = canonical_sha256(copied)
        _atomic_json(manifest_path, copied)
        return examples, copied

    layouts = generate_layouts(
        grid_sizes=args.grid_sizes,
        layouts_per_size=args.layouts_per_size,
        seed=args.data_seed,
        max_wall_fraction=args.max_wall_fraction,
    )

    def progress(index, total, layout, statuses):
        print(
            f"Data {index:3d}/{total:3d} | {layout.layout_id} | "
            f"states={sum(statuses.values()):4d} | "
            f"SAFE={statuses.get('SAFE', 0):4d} | "
            f"UNSAFE={statuses.get('UNSAFE', 0):4d} | "
            f"UNKNOWN={statuses.get('UNKNOWN', 0):4d}",
            flush=True,
        )

    examples = generate_examples(
        layouts,
        states_per_layout=args.states_per_layout,
        seed=args.data_seed,
        search_max_nodes=args.search_max_nodes,
        rank_label_fraction=args.rank_label_fraction,
        rank_search_max_nodes=args.rank_search_max_nodes,
        reservation_probability=args.reservation_probability,
        max_reservations=args.max_reservations,
        progress=progress,
    )
    _, split_manifest = split_examples_by_layout(
        examples,
        ratios=DEFAULT_SPLIT_RATIOS,
        seed=args.split_seed,
    )
    generation_config = {
        "grid_sizes": [list(value) for value in args.grid_sizes],
        "requested_layouts_per_size": args.layouts_per_size,
        "states_per_layout": args.states_per_layout,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "max_wall_fraction": args.max_wall_fraction,
        "search_order": "goal_directed_with_selective_breadth_first_rank",
        "search_max_nodes": args.search_max_nodes,
        "rank_label_fraction": args.rank_label_fraction,
        "rank_search_max_nodes": args.rank_search_max_nodes,
        "reservation_probability": args.reservation_probability,
        "max_reservations": args.max_reservations,
    }
    manifest = dataset_manifest(
        layouts=layouts,
        examples=examples,
        split_manifest=split_manifest,
        generation_config=generation_config,
        timing_scale=args.timing_scale,
    )
    record_digest = write_dataset(dataset_path, examples)
    manifest.pop("manifest_sha256", None)
    manifest["records_sha256"] = record_digest
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    return examples, manifest


def _checkpoint_common(
    *,
    args: argparse.Namespace,
    manifest: Mapping,
    model_config: Mapping,
    member_seeds: Sequence[int],
    bootstrap_seeds: Sequence[int],
    bootstrap_hashes: Sequence[str],
) -> dict:
    return {
        "viability_critic_checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "trainer_version": TRAINER_VERSION,
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
        "calibration_contract": CALIBRATION_CONTRACT,
        "label_contract": DYNAMICS_LABEL_CONTRACT,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "node_feature_dim": NODE_FEATURE_DIM,
        "model_config": dict(model_config),
        "training_config": _training_config(args),
        "ensemble_size": len(member_seeds),
        "member_seeds": list(member_seeds),
        "bootstrap_seeds": list(bootstrap_seeds),
        "bootstrap_hashes": list(bootstrap_hashes),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "split_manifest_sha256": manifest["split_manifest"]["split_sha256"],
        "timing_scale": float(args.timing_scale),
        "lcb_scale": float(args.lcb_scale),
        "baseline_viability_teacher": False,
        "critic_certificate_authority": False,
        "exact_verifier_authoritative": True,
        "unknown_supervision": "masked",
        "training_seed": int(args.seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "layouts_per_size",
        "states_per_layout",
        "ensemble_size",
        "epochs",
        "steps_per_epoch",
        "batch_size",
        "eval_batch_size",
        "eval_every",
        "graph_hidden_dim",
        "graph_embedding_dim",
        "head_hidden_dim",
        "grad_clip",
        "false_safe_weight",
        "huber_beta",
        "timing_scale",
        "min_calibration_safe_accepted",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.ensemble_size < 2:
        raise ValueError("--ensemble-size must be at least 2")
    if args.message_passing_steps < 0:
        raise ValueError("--message-passing-steps must be non-negative")
    if not 0.0 < args.unsafe_sampling_fraction < 1.0:
        raise ValueError("--unsafe-sampling-fraction must be in (0, 1)")
    if not 0.0 < args.calibration_confidence < 1.0:
        raise ValueError("--calibration-confidence must be in (0, 1)")
    if not 0.0 < args.max_false_safe_upper_bound < 1.0:
        raise ValueError("--max-false-safe-upper-bound must be in (0, 1)")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    for name in (
        "safety_weight",
        "recovery_rank_weight",
        "primitive_steps_weight",
        "lcb_scale",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.dataset is None and args.dataset_manifest is not None:
        raise ValueError("--dataset-manifest requires --dataset")
    if args.resume is not None and args.dataset is not None:
        raise ValueError("--resume cannot be combined with --dataset")
    if args.reuse_data and (args.resume is not None or args.dataset is not None):
        raise ValueError(
            "--reuse-data cannot be combined with --resume or --dataset"
        )
    existing_entries = (
        tuple(args.output_dir.iterdir()) if args.output_dir.exists() else ()
    )
    materially_nonempty = any(
        entry.is_file() or (entry.is_dir() and any(entry.iterdir()))
        for entry in existing_entries
    )
    if args.reuse_data:
        unexpected = {
            entry.name
            for entry in existing_entries
            if entry.name not in {"data", "generation-summary.json"}
        }
        if unexpected:
            raise FileExistsError(
                "--reuse-data refuses existing training artifacts: "
                f"{tuple(sorted(unexpected))}"
            )
    if materially_nonempty and args.resume is None and not args.reuse_data:
        raise FileExistsError(
            "output directory is not empty; choose a new directory, use "
            "--reuse-data for completed data artifacts, or use --resume"
        )
    if args.resume is not None:
        if args.resume.resolve() != (args.output_dir / "latest.pth").resolve():
            raise ValueError("--resume must name OUTPUT_DIR/latest.pth")
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a baseline-free graph viability ensemble and calibrate its "
            "LCB for prioritizing exact recoverability checks"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=81000)
    parser.add_argument("--split-seed", type=int, default=82000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reuse-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reuse and authenticate OUTPUT_DIR/data before a fresh fit",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--generate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--grid-sizes",
        type=parse_grid_size,
        nargs="+",
        default=[
            (4, 4),
            (4, 5),
            (5, 4),
            (5, 5),
            (5, 6),
            (6, 5),
        ],
    )
    parser.add_argument("--layouts-per-size", type=int, default=6)
    parser.add_argument("--states-per-layout", type=int, default=512)
    parser.add_argument("--max-wall-fraction", type=float, default=0.20)
    parser.add_argument("--search-max-nodes", type=optional_positive, default=100_000)
    parser.add_argument("--rank-label-fraction", type=float, default=0.10)
    parser.add_argument(
        "--rank-search-max-nodes", type=optional_positive, default=100_000
    )
    parser.add_argument("--reservation-probability", type=float, default=0.20)
    parser.add_argument("--max-reservations", type=int, default=2)
    parser.add_argument("--timing-scale", type=float, default=100.0)

    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--graph-embedding-dim", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=4)
    parser.add_argument("--head-hidden-dim", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--false-safe-weight", type=float, default=10.0)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--recovery-rank-weight", type=float, default=0.25)
    parser.add_argument("--primitive-steps-weight", type=float, default=0.05)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--unsafe-sampling-fraction", type=float, default=0.50)

    parser.add_argument("--lcb-scale", type=float, default=2.0)
    parser.add_argument("--calibration-confidence", type=float, default=0.95)
    parser.add_argument("--max-false-safe-upper-bound", type=float, default=0.10)
    parser.add_argument("--min-calibration-safe-accepted", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    started = time.perf_counter()
    print(
        "Viability critic | "
        f"device={device} | ensemble={args.ensemble_size} | "
        f"baseline-teacher=false | exact-authority=true",
        flush=True,
    )

    examples, manifest = _prepare_data(args)
    splits = _split_from_manifest(examples, manifest)
    print(
        "Data ready | "
        + " | ".join(
            f"{name}={len(values)}" for name, values in splits.items()
        ),
        flush=True,
    )
    if args.generate_only:
        summary = {
            "trainer_version": TRAINER_VERSION,
            "stage": "generate_only",
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "status_counts": manifest["status_counts"],
            "splits": manifest["split_manifest"]["splits"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(args.output_dir / "generation-summary.json", summary)
        print(f"Dataset: {args.output_dir / 'data' / 'viability-dataset.jsonl'}")
        print(f"Manifest: {args.output_dir / 'data' / 'dataset-manifest.json'}")
        return summary

    split_graphs = {
        name: tuple(
            encode_recovery_state(
                example.state,
                timing_scale=args.timing_scale,
            )
            for example in values
        )
        for name, values in splits.items()
    }
    print(
        f"Graphs encoded once | total={sum(len(items) for items in split_graphs.values())}",
        flush=True,
    )

    train_examples = splits["train"]
    if not any(item.label.status is ViabilityStatus.SAFE for item in train_examples):
        raise RuntimeError("training split contains no SAFE labels")
    if not any(item.label.status is ViabilityStatus.UNSAFE for item in train_examples):
        raise RuntimeError("training split contains no UNSAFE labels")
    model_config = _model_config(args)
    members, member_seeds = _make_members(
        ensemble_size=args.ensemble_size,
        model_config=model_config,
        seed=args.seed,
        device=device,
    )
    ensemble = ViabilityCriticEnsemble(members).to(device)
    optimizers = tuple(
        torch.optim.AdamW(
            member.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for member in members
    )
    bootstraps, bootstrap_seeds, bootstrap_hashes = _bootstrap_members(
        train_examples,
        ensemble_size=args.ensemble_size,
        seed=args.seed,
    )
    common = _checkpoint_common(
        args=args,
        manifest=manifest,
        model_config=model_config,
        member_seeds=member_seeds,
        bootstrap_seeds=bootstrap_seeds,
        bootstrap_hashes=bootstrap_hashes,
    )
    history = []
    best_score = None
    best_state_dicts = None
    best_epoch = 0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("checkpoint_kind") != "training" or not checkpoint.get("resumable"):
            raise ValueError("resume checkpoint is not a resumable training artifact")
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "model_config",
            "training_config",
            "ensemble_size",
            "bootstrap_hashes",
        ):
            if checkpoint.get(key) != common.get(key):
                raise ValueError(f"resume checkpoint mismatch: {key}")
        _load_state_dicts(members, checkpoint["member_state_dicts"])
        for optimizer, state in zip(optimizers, checkpoint["optimizer_state_dicts"]):
            optimizer.load_state_dict(state)
        history = list(checkpoint.get("history", []))
        best_score = tuple(checkpoint["best_score"]) if checkpoint.get("best_score") is not None else None
        best_state_dicts = checkpoint.get("best_member_state_dicts")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resume | epoch={start_epoch - 1} | best_epoch={best_epoch}", flush=True)

    latest_path = args.output_dir / "latest.pth"
    best_path = args.output_dir / "best.pth"
    if args.resume is None:
        initial_checkpoint = {
            **common,
            "checkpoint_kind": "training",
            "resumable": True,
            "epoch": 0,
            "member_state_dicts": _state_dicts_cpu(members),
            "optimizer_state_dicts": [
                optimizer.state_dict() for optimizer in optimizers
            ],
            "best_score": None,
            "best_epoch": 0,
            "best_member_state_dicts": None,
            "history": [],
        }
        _atomic_torch_save(latest_path, initial_checkpoint)
    for epoch in range(start_epoch, args.epochs + 1):
        losses = _train_epoch(
            epoch=epoch,
            members=members,
            optimizers=optimizers,
            bootstraps=bootstraps,
            bootstrap_seeds=bootstrap_seeds,
            examples=train_examples,
            graphs=split_graphs["train"],
            args=args,
        )
        evaluate_now = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        validation_metrics = None
        if evaluate_now:
            validation_predictions = predict_ensemble(
                ensemble,
                splits["validation"],
                batch_size=args.eval_batch_size,
                timing_scale=args.timing_scale,
                lcb_scale=args.lcb_scale,
                graphs=split_graphs["validation"],
            )
            validation_metrics = evaluate_predictions(
                splits["validation"],
                validation_predictions,
                threshold=0.5,
                confidence=args.calibration_confidence,
            )
            selection_score = (
                int(validation_metrics["false_safe_count"]),
                -int(validation_metrics["accepted_safe_count"]),
                float(validation_metrics["brier_score"]),
            )
            if best_score is None or selection_score < best_score:
                best_score = selection_score
                best_epoch = epoch
                best_state_dicts = _state_dicts_cpu(members)
        entry = {
            "epoch": epoch,
            **losses,
            "validation": validation_metrics,
        }
        history.append(entry)
        validation_text = ""
        if validation_metrics is not None:
            validation_text = (
                f" | ValFS {validation_metrics['false_safe_count']:3d}/"
                f"{validation_metrics['unsafe_count']:3d}"
                f" | ValCov {validation_metrics['safe_coverage']:.3f}"
                f" | ValAUC {validation_metrics['auroc'] if validation_metrics['auroc'] is not None else float('nan'):.3f}"
            )
        print(
            f"Epoch {epoch:3d}/{args.epochs:3d} | Loss {losses['total_loss']:.5f}"
            f" | SafeL {losses['safety_loss']:.5f}"
            f" | RankL {losses['recovery_rank_loss']:.5f}"
            f" | StepL {losses['primitive_steps_loss']:.5f}"
            f"{validation_text}",
            flush=True,
        )
        training_checkpoint = {
            **common,
            "checkpoint_kind": "training",
            "resumable": True,
            "epoch": epoch,
            "member_state_dicts": _state_dicts_cpu(members),
            "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
            "best_score": list(best_score) if best_score is not None else None,
            "best_epoch": best_epoch,
            "best_member_state_dicts": best_state_dicts,
            "history": history,
        }
        _atomic_torch_save(latest_path, training_checkpoint)
        _atomic_json(args.output_dir / "training-history.json", history)

    if best_state_dicts is None:
        raise RuntimeError("training produced no model-selection checkpoint")
    _load_state_dicts(members, best_state_dicts)
    calibration_predictions = predict_ensemble(
        ensemble,
        splits["calibration"],
        batch_size=args.eval_batch_size,
        timing_scale=args.timing_scale,
        lcb_scale=args.lcb_scale,
        graphs=split_graphs["calibration"],
    )
    calibration = choose_lcb_threshold(
        splits["calibration"],
        calibration_predictions,
        confidence=args.calibration_confidence,
        max_false_safe_upper_bound=args.max_false_safe_upper_bound,
        min_safe_accepted=args.min_calibration_safe_accepted,
    )
    threshold = float(calibration["threshold"])
    calibration_metrics = evaluate_predictions(
        splits["calibration"],
        calibration_predictions,
        threshold=threshold,
        confidence=args.calibration_confidence,
    )
    test_predictions = predict_ensemble(
        ensemble,
        splits["test"],
        batch_size=args.eval_batch_size,
        timing_scale=args.timing_scale,
        lcb_scale=args.lcb_scale,
        graphs=split_graphs["test"],
    )
    test_metrics = evaluate_predictions(
        splits["test"],
        test_predictions,
        threshold=threshold,
        confidence=args.calibration_confidence,
    )
    independent_confirmation = bool(
        calibration["ready"]
        and test_metrics["unsafe_count"] > 0
        and test_metrics["accepted_safe_count"] > 0
        and test_metrics["unsafe_false_safe_upper_bound"]
        <= args.max_false_safe_upper_bound
    )
    calibration_report = {
        "selection": calibration,
        "calibration_metrics": calibration_metrics,
        "independent_test_metrics": test_metrics,
        "independent_test_confirmation": independent_confirmation,
        "interpretation": (
            "empirical prioritization evidence only; every proposed SAFE "
            "candidate still requires exact dynamics certification"
        ),
        "critic_certificate_authority": False,
        "exact_verifier_authoritative": True,
    }
    deployment_checkpoint = {
        **common,
        "checkpoint_kind": "deployment",
        "resumable": False,
        "selected_epoch": best_epoch,
        "member_state_dicts": best_state_dicts,
        "calibration": calibration_report,
    }
    _atomic_torch_save(best_path, deployment_checkpoint)
    _atomic_json(args.output_dir / "calibration.json", calibration_report)
    summary = {
        "trainer_version": TRAINER_VERSION,
        "device": str(device),
        "epochs": args.epochs,
        "selected_epoch": best_epoch,
        "best_score": list(best_score),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "split_manifest_sha256": manifest["split_manifest"]["split_sha256"],
        "status_counts": manifest["status_counts"],
        "calibration": calibration_report,
        "best_checkpoint": str(best_path.resolve()),
        "latest_checkpoint": str(latest_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "training-summary.json", summary)
    print(
        "Calibration | "
        f"ready={int(calibration['ready'])} | "
        f"threshold={threshold:.6f} | "
        f"test-confirm={int(independent_confirmation)} | "
        f"test-FS={test_metrics['false_safe_count']}/{test_metrics['unsafe_count']} | "
        f"test-safe-coverage={test_metrics['safe_coverage']:.3f}",
        flush=True,
    )
    print(f"Best: {best_path}")
    print(f"Latest: {latest_path}")
    print(f"Summary: {args.output_dir / 'training-summary.json'}")
    return summary


if __name__ == "__main__":
    main()
