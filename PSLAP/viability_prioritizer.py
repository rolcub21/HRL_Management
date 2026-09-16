"""Authenticated critic ordering for exact recoverability checks.

The learned ensemble in this module is a proposal-ordering device only.  Its
``priority_pass`` flag and scores are empirical predictions; none is a safety
label.  Callers must still obtain an exact :class:`RecoverabilityCertificate`
for every state admitted to an executable frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Protocol, Sequence

import torch

from PSLAP.viability import (
    CLOSED_ADMISSION_CONTRACT,
    STRICT_MACRO_ACTION_MODEL,
    RecoveryState,
)
from PSLAP.viability_critic import (
    CounterfactualViabilityCritic,
    DYNAMICS_LABEL_CONTRACT,
    ViabilityCriticEnsemble,
)
from PSLAP.viability_dataset import (
    VIABILITY_DATASET_PROTOCOL,
    VIABILITY_SPLIT_PROTOCOL,
    canonical_sha256,
)
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    encode_recovery_state,
    pad_yard_graphs,
)


VIABILITY_PRIORITY_PROTOCOL = "calibrated_lcb_full_permutation_v1"
EXPECTED_CHECKPOINT_SCHEMA = "viability_critic_checkpoint_v1"
EXPECTED_TRAINER_VERSION = "dynamics_viability_critic_trainer_v1"
EXPECTED_MODEL_SELECTION_CONTRACT = (
    "validation_lcb_at_0.5_min_false_safe_then_max_safe_coverage_v1"
)
EXPECTED_CALIBRATION_CONTRACT = (
    "calibration_lcb_threshold_then_independent_test_binomial_upper_v1"
)


@dataclass(frozen=True)
class StatePriorityEntry:
    """One empirical score in original input coordinates."""

    original_index: int
    key: str
    state: RecoveryState
    safety_probability_mean: float
    safety_probability_std: float
    safety_probability_lcb: float
    recovery_rank_mean: float
    recovery_rank_std: float
    primitive_steps_mean: float
    primitive_steps_std: float
    priority_pass: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.original_index, bool)
            or not isinstance(self.original_index, int)
            or self.original_index < 0
        ):
            raise ValueError("priority original_index must be non-negative")
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("priority entry key must be nonempty")
        if not isinstance(self.state, RecoveryState):
            raise TypeError("priority entry state must be a RecoveryState")
        probabilities = (
            self.safety_probability_mean,
            self.safety_probability_lcb,
        )
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in probabilities
        ):
            raise ValueError("priority probabilities must be finite in [0, 1]")
        nonnegative = (
            self.safety_probability_std,
            self.recovery_rank_mean,
            self.recovery_rank_std,
            self.primitive_steps_mean,
            self.primitive_steps_std,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in nonnegative
        ):
            raise ValueError("priority estimates must be finite and non-negative")
        if not isinstance(self.priority_pass, bool):
            raise TypeError("priority_pass must be Boolean")


@dataclass(frozen=True)
class StatePriorityBatch:
    """A validated permutation plus non-authoritative critic evidence."""

    entries: tuple[StatePriorityEntry, ...]
    ordered_indices: tuple[int, ...]
    inference_seconds: float
    threshold: float
    checkpoint_sha256: str
    protocol: str = VIABILITY_PRIORITY_PROTOCOL
    critic_certificate_authority: bool = False
    exact_verifier_authoritative: bool = True

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.entries)))
        if tuple(sorted(self.ordered_indices)) != expected:
            raise ValueError("priority output must be a complete permutation")
        if tuple(entry.original_index for entry in self.entries) != expected:
            raise ValueError("priority entries must retain original coordinates")
        if self.critic_certificate_authority:
            raise ValueError("critic cannot be certificate-authoritative")
        if not self.exact_verifier_authoritative:
            raise ValueError("exact verifier must remain authoritative")
        if (
            not math.isfinite(float(self.inference_seconds))
            or float(self.inference_seconds) < 0.0
        ):
            raise ValueError("priority inference time must be finite and non-negative")
        if (
            not math.isfinite(float(self.threshold))
            or not 0.0 <= float(self.threshold) <= 1.0
        ):
            raise ValueError("priority threshold must be finite in [0, 1]")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_sha256.lower()
            )
        ):
            raise ValueError("priority checkpoint identity must be SHA-256")
        if not isinstance(self.protocol, str) or not self.protocol:
            raise ValueError("priority protocol must be nonempty")

    @property
    def ordered_entries(self) -> tuple[StatePriorityEntry, ...]:
        return tuple(self.entries[index] for index in self.ordered_indices)

    @property
    def priority_pass_count(self) -> int:
        return sum(entry.priority_pass for entry in self.entries)


class RecoveryStatePrioritizer(Protocol):
    """Minimal interface accepted by exact candidate enumeration."""

    protocol: str
    checkpoint_sha256: str
    threshold: float

    def prioritize(
        self,
        keyed_states: Sequence[tuple[str, RecoveryState]],
    ) -> StatePriorityBatch:
        ...


def validate_priority_batch(
    batch: StatePriorityBatch,
    keyed_states: Sequence[tuple[str, RecoveryState]],
    *,
    prioritizer: RecoveryStatePrioritizer,
) -> StatePriorityBatch:
    """Bind a returned permutation to the exact request and provider audit."""

    if not isinstance(batch, StatePriorityBatch):
        raise TypeError("prioritizer must return a StatePriorityBatch")
    keyed_states = tuple((str(key), state) for key, state in keyed_states)
    if len(batch.entries) != len(keyed_states):
        raise ValueError("priority batch size does not match its request")
    if batch.protocol != str(prioritizer.protocol):
        raise ValueError("priority batch/provider protocol mismatch")
    if batch.checkpoint_sha256 != str(prioritizer.checkpoint_sha256):
        raise ValueError("priority batch/provider checkpoint mismatch")
    if float(batch.threshold) != float(prioritizer.threshold):
        raise ValueError("priority batch/provider threshold mismatch")
    for entry, (key, state) in zip(batch.entries, keyed_states):
        if entry.key != key or entry.state != state:
            raise ValueError("priority entries do not match requested states")
    return batch


def _finite_number(value, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return result


def _positive_integer(value, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in logs)
    )


def _one_sided_binomial_upper_bound(
    failures: int,
    trials: int,
    *,
    confidence: float,
) -> float:
    if failures < 0 or trials < 0 or failures > trials:
        raise ValueError("false-safe counts must satisfy 0 <= failures <= trials")
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


def _validate_dataset_manifest(
    path: Path,
    checkpoint: Mapping,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied = payload.get("manifest_sha256")
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    computed = canonical_sha256(canonical)
    if supplied != computed:
        raise ValueError("viability dataset manifest digest is invalid")
    if supplied != checkpoint.get("dataset_manifest_sha256"):
        raise ValueError("checkpoint/dataset manifest digest mismatch")
    split_digest = payload.get("split_manifest", {}).get("split_sha256")
    if split_digest != checkpoint.get("split_manifest_sha256"):
        raise ValueError("checkpoint/dataset split digest mismatch")
    expected = {
        "protocol": VIABILITY_DATASET_PROTOCOL,
        "baseline_viability_teacher": False,
        "exact_verifier_authoritative": True,
        "critic_certificate_authority": False,
        "unknown_supervision": "masked",
        "certificate_contract": CLOSED_ADMISSION_CONTRACT,
        "action_model": STRICT_MACRO_ACTION_MODEL,
        "label_contract": DYNAMICS_LABEL_CONTRACT,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "node_feature_dim": NODE_FEATURE_DIM,
        "max_primitive_steps": None,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible viability dataset manifest: {mismatches}")
    if payload.get("split_manifest", {}).get("protocol") != VIABILITY_SPLIT_PROTOCOL:
        raise ValueError("incompatible viability dataset split protocol")
    if float(payload.get("timing_scale")) != float(checkpoint.get("timing_scale")):
        raise ValueError("checkpoint/dataset timing scale mismatch")
    return payload


class ViabilityCriticPrioritizer:
    """Frozen calibrated ensemble used only to order exact checks."""

    protocol = VIABILITY_PRIORITY_PROTOCOL

    def __init__(
        self,
        *,
        ensemble: ViabilityCriticEnsemble,
        threshold: float,
        timing_scale: float,
        lcb_scale: float,
        checkpoint_sha256: str,
        device: str | torch.device,
        dataset_manifest_sha256: str,
        split_manifest_sha256: str,
    ) -> None:
        if not isinstance(ensemble, ViabilityCriticEnsemble):
            raise TypeError("ensemble must be a ViabilityCriticEnsemble")
        self.device = torch.device(device)
        self.ensemble = ensemble.to(self.device)
        self.ensemble.requires_grad_(False)
        self.ensemble.eval()
        self.threshold = _finite_number(threshold, name="threshold")
        if self.threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.timing_scale = _finite_number(
            timing_scale, name="timing_scale", minimum=0.0
        )
        if self.timing_scale == 0.0:
            raise ValueError("timing_scale must be positive")
        self.lcb_scale = _finite_number(lcb_scale, name="lcb_scale")
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.dataset_manifest_sha256 = str(dataset_manifest_sha256)
        self.split_manifest_sha256 = str(split_manifest_sha256)
        if any(
            len(value) != 64
            for value in (
                self.checkpoint_sha256,
                self.dataset_manifest_sha256,
                self.split_manifest_sha256,
            )
        ):
            raise ValueError("checkpoint and manifest identities must be SHA-256")

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dataset_manifest: Optional[str | Path] = None,
        require_test_confirmation: bool = True,
        expected_checkpoint_sha256: Optional[str] = None,
    ) -> "ViabilityCriticPrioritizer":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        checkpoint_sha256 = _checkpoint_sha256(path)
        if expected_checkpoint_sha256 is not None:
            expected_checkpoint_sha256 = str(expected_checkpoint_sha256).lower()
            if (
                len(expected_checkpoint_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_checkpoint_sha256
                )
            ):
                raise ValueError("expected checkpoint identity must be SHA-256")
            if checkpoint_sha256 != expected_checkpoint_sha256:
                raise ValueError("viability critic checkpoint SHA-256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "viability_critic_checkpoint_schema_version": EXPECTED_CHECKPOINT_SCHEMA,
            "trainer_version": EXPECTED_TRAINER_VERSION,
            "model_selection_contract": EXPECTED_MODEL_SELECTION_CONTRACT,
            "calibration_contract": EXPECTED_CALIBRATION_CONTRACT,
            "label_contract": DYNAMICS_LABEL_CONTRACT,
            "checkpoint_kind": "deployment",
            "resumable": False,
            "baseline_viability_teacher": False,
            "critic_certificate_authority": False,
            "exact_verifier_authoritative": True,
            "unknown_supervision": "masked",
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "node_feature_dim": NODE_FEATURE_DIM,
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"incompatible viability critic checkpoint: {mismatches}")

        model_config = checkpoint.get("model_config")
        model_keys = {
            "input_dim",
            "graph_hidden_dim",
            "graph_embedding_dim",
            "message_passing_steps",
            "head_hidden_dim",
        }
        if not isinstance(model_config, Mapping) or set(model_config) != model_keys:
            raise ValueError("checkpoint model_config is noncanonical")
        if model_config.get("input_dim") != NODE_FEATURE_DIM:
            raise ValueError("checkpoint critic input dimension is incompatible")
        for key in model_keys - {"message_passing_steps"}:
            _positive_integer(model_config[key], name=f"model_config.{key}")
        _positive_integer(
            model_config["message_passing_steps"],
            name="model_config.message_passing_steps",
            allow_zero=True,
        )

        ensemble_size = _positive_integer(
            checkpoint.get("ensemble_size"), name="ensemble_size"
        )
        _positive_integer(checkpoint.get("selected_epoch"), name="selected_epoch")
        if ensemble_size < 2:
            raise ValueError("calibrated ensemble requires at least two members")
        state_dicts = checkpoint.get("member_state_dicts")
        if not isinstance(state_dicts, list) or len(state_dicts) != ensemble_size:
            raise ValueError("checkpoint member-state count mismatch")
        for name in ("member_seeds", "bootstrap_seeds", "bootstrap_hashes"):
            values = checkpoint.get(name)
            if not isinstance(values, list) or len(values) != ensemble_size:
                raise ValueError(f"checkpoint {name} count mismatch")
            if len(values) != len(set(values)):
                raise ValueError(f"checkpoint {name} must be unique")

        calibration = checkpoint.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError("checkpoint calibration record is missing")
        selection = calibration.get("selection")
        if not isinstance(selection, Mapping) or selection.get("ready") is not True:
            raise ValueError("critic calibration is not ready")
        if require_test_confirmation and calibration.get(
            "independent_test_confirmation"
        ) is not True:
            raise ValueError("critic lacks independent test confirmation")
        for source in (calibration, selection):
            if source.get("critic_certificate_authority") is not False:
                raise ValueError("calibration improperly grants critic authority")
            if source.get("exact_verifier_authoritative") is not True:
                raise ValueError("calibration does not preserve exact authority")
        if selection.get("calibration_contract") != EXPECTED_CALIBRATION_CONTRACT:
            raise ValueError("selection calibration contract mismatch")
        threshold = _finite_number(selection.get("threshold"), name="threshold")
        if threshold > 1.0:
            raise ValueError("calibrated threshold must be in [0, 1]")
        timing_scale = _finite_number(
            checkpoint.get("timing_scale"), name="timing_scale"
        )
        if timing_scale == 0.0:
            raise ValueError("checkpoint timing_scale must be positive")
        lcb_scale = _finite_number(checkpoint.get("lcb_scale"), name="lcb_scale")
        training_config = checkpoint.get("training_config")
        if (
            not isinstance(training_config, Mapping)
            or float(training_config.get("lcb_scale")) != lcb_scale
        ):
            raise ValueError("training/checkpoint LCB scale mismatch")
        maximum_upper = _finite_number(
            selection.get("maximum_false_safe_upper_bound"),
            name="maximum_false_safe_upper_bound",
        )
        if maximum_upper <= 0.0 or maximum_upper >= 1.0:
            raise ValueError("false-safe upper-bound limit must be in (0, 1)")
        calibration_confidence = _finite_number(
            training_config.get("calibration_confidence"),
            name="training_config.calibration_confidence",
        )
        if not 0.0 < calibration_confidence < 1.0:
            raise ValueError("calibration confidence must be in (0, 1)")
        training_maximum = _finite_number(
            training_config.get("max_false_safe_upper_bound"),
            name="training_config.max_false_safe_upper_bound",
        )
        minimum_accepted = _positive_integer(
            training_config.get("min_calibration_safe_accepted"),
            name="training_config.min_calibration_safe_accepted",
        )
        if float(selection.get("confidence")) != calibration_confidence:
            raise ValueError("selection/training confidence mismatch")
        if maximum_upper != training_maximum:
            raise ValueError("selection/training false-safe limit mismatch")
        if int(selection.get("minimum_safe_accepted")) != minimum_accepted:
            raise ValueError("selection/training accepted-safe minimum mismatch")
        for name in ("calibration_metrics", "independent_test_metrics"):
            metrics = calibration.get(name)
            if not isinstance(metrics, Mapping):
                raise ValueError(f"checkpoint {name} is missing")
            if float(metrics.get("threshold")) != threshold:
                raise ValueError(f"checkpoint {name} threshold mismatch")
            count_names = (
                "example_count",
                "known_count",
                "unknown_count",
                "safe_count",
                "unsafe_count",
                "accepted_safe_count",
                "false_safe_count",
                "accepted_count",
            )
            counts = {}
            for count_name in count_names:
                value = metrics.get(count_name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"checkpoint {name}.{count_name} must be non-negative"
                    )
                counts[count_name] = value
            if counts["unsafe_count"] <= 0:
                raise ValueError(f"checkpoint {name} lacks UNSAFE evidence")
            if counts["accepted_safe_count"] <= 0:
                raise ValueError(f"checkpoint {name} lacks accepted SAFE evidence")
            if (
                counts["known_count"]
                != counts["safe_count"] + counts["unsafe_count"]
                or counts["example_count"]
                != counts["known_count"] + counts["unknown_count"]
                or counts["accepted_count"]
                != counts["accepted_safe_count"] + counts["false_safe_count"]
                or counts["accepted_safe_count"] > counts["safe_count"]
                or counts["false_safe_count"] > counts["unsafe_count"]
            ):
                raise ValueError(f"checkpoint {name} has inconsistent counts")
            reported_upper = _finite_number(
                metrics.get("unsafe_false_safe_upper_bound"),
                name=f"checkpoint {name}.unsafe_false_safe_upper_bound",
            )
            recomputed_upper = _one_sided_binomial_upper_bound(
                counts["false_safe_count"],
                counts["unsafe_count"],
                confidence=calibration_confidence,
            )
            if not math.isclose(
                reported_upper,
                recomputed_upper,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"checkpoint {name} false-safe bound is stale")
            if reported_upper > maximum_upper:
                raise ValueError(f"checkpoint {name} exceeds false-safe limit")
        calibration_metrics = calibration["calibration_metrics"]
        selection_matches = {
            "unsafe_count": calibration_metrics.get("unsafe_count"),
            "safe_count": calibration_metrics.get("safe_count"),
            "safe_accepted": calibration_metrics.get("accepted_safe_count"),
            "false_safe_count": calibration_metrics.get("false_safe_count"),
            "false_safe_upper_bound": calibration_metrics.get(
                "unsafe_false_safe_upper_bound"
            ),
        }
        if any(selection.get(key) != value for key, value in selection_matches.items()):
            raise ValueError("selection/calibration metric mismatch")
        if int(selection["safe_accepted"]) < minimum_accepted:
            raise ValueError("calibration accepted-safe evidence is below minimum")

        manifest_path = (
            Path(dataset_manifest)
            if dataset_manifest is not None
            else path.parent / "data" / "dataset-manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"authenticated dataset manifest is required: {manifest_path}"
            )
        manifest = _validate_dataset_manifest(manifest_path, checkpoint)

        members = []
        for state_dict in state_dicts:
            if not isinstance(state_dict, Mapping) or not state_dict:
                raise ValueError("critic member state_dict is invalid")
            if any(
                not torch.is_tensor(value)
                or (value.is_floating_point() and not torch.isfinite(value).all())
                for value in state_dict.values()
            ):
                raise ValueError("critic member weights must be finite tensors")
            member = CounterfactualViabilityCritic(**dict(model_config))
            member.load_state_dict(state_dict, strict=True)
            member.requires_grad_(False)
            member.eval()
            members.append(member)
        ensemble = ViabilityCriticEnsemble(tuple(members))
        return cls(
            ensemble=ensemble,
            threshold=threshold,
            timing_scale=timing_scale,
            lcb_scale=lcb_scale,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
            dataset_manifest_sha256=manifest["manifest_sha256"],
            split_manifest_sha256=manifest["split_manifest"]["split_sha256"],
        )

    def prioritize(
        self,
        keyed_states: Sequence[tuple[str, RecoveryState]],
    ) -> StatePriorityBatch:
        keyed_states = tuple(keyed_states)
        keys = tuple(str(key) for key, _ in keyed_states)
        states = tuple(state for _, state in keyed_states)
        if len(keys) != len(set(keys)):
            raise ValueError("priority keys must be unique")
        if not all(keys):
            raise ValueError("priority keys must be nonempty")
        if not all(isinstance(state, RecoveryState) for state in states):
            raise TypeError("prioritizer accepts only RecoveryState values")
        if not states:
            return StatePriorityBatch(
                entries=(),
                ordered_indices=(),
                inference_seconds=0.0,
                threshold=self.threshold,
                checkpoint_sha256=self.checkpoint_sha256,
            )
        graphs = tuple(
            encode_recovery_state(state, timing_scale=self.timing_scale)
            for state in states
        )
        float32_max = torch.finfo(torch.float32).max
        if any(
            not math.isfinite(feature) or abs(feature) > float32_max
            for graph in graphs
            for row in graph.node_features
            for feature in row
        ):
            raise ValueError("critic input contains a non-finite float32 feature")
        batch = pad_yard_graphs(graphs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode():
            estimate = self.ensemble.estimate(batch, lcb_scale=self.lcb_scale)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - started
        estimate_tensors = (
            estimate.safety_probability_mean,
            estimate.safety_probability_std,
            estimate.safety_probability_lcb,
            estimate.recovery_rank_mean,
            estimate.recovery_rank_std,
            estimate.primitive_steps_mean,
            estimate.primitive_steps_std,
        )
        if any(not torch.isfinite(value).all() for value in estimate_tensors):
            raise RuntimeError("critic inference produced non-finite priority scores")
        entries = tuple(
            StatePriorityEntry(
                original_index=index,
                key=keys[index],
                state=states[index],
                safety_probability_mean=float(
                    estimate.safety_probability_mean[index].item()
                ),
                safety_probability_std=float(
                    estimate.safety_probability_std[index].item()
                ),
                safety_probability_lcb=float(
                    estimate.safety_probability_lcb[index].item()
                ),
                recovery_rank_mean=float(
                    estimate.recovery_rank_mean[index].item()
                ),
                recovery_rank_std=float(
                    estimate.recovery_rank_std[index].item()
                ),
                primitive_steps_mean=float(
                    estimate.primitive_steps_mean[index].item()
                ),
                primitive_steps_std=float(
                    estimate.primitive_steps_std[index].item()
                ),
                priority_pass=bool(
                    estimate.safety_probability_lcb[index].item()
                    >= self.threshold
                ),
            )
            for index in range(len(states))
        )
        ordered = tuple(
            sorted(
                range(len(entries)),
                key=lambda index: (
                    not entries[index].priority_pass,
                    -entries[index].safety_probability_lcb,
                    entries[index].key,
                ),
            )
        )
        return StatePriorityBatch(
            entries=entries,
            ordered_indices=ordered,
            inference_seconds=elapsed,
            threshold=self.threshold,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def audit_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "ensemble_size": len(self.ensemble.members),
            "threshold": self.threshold,
            "timing_scale": self.timing_scale,
            "lcb_scale": self.lcb_scale,
            "device": str(self.device),
            "critic_certificate_authority": False,
            "exact_verifier_authoritative": True,
            "certification_scope": "complete_frontier_exactly_verified",
        }


__all__ = [
    "EXPECTED_CALIBRATION_CONTRACT",
    "EXPECTED_CHECKPOINT_SCHEMA",
    "EXPECTED_MODEL_SELECTION_CONTRACT",
    "EXPECTED_TRAINER_VERSION",
    "RecoveryStatePrioritizer",
    "StatePriorityBatch",
    "StatePriorityEntry",
    "VIABILITY_PRIORITY_PROTOCOL",
    "ViabilityCriticPrioritizer",
    "validate_priority_batch",
]
