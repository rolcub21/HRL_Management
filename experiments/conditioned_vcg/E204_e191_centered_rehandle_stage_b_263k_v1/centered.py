"""Centered rehandle-weight adapter for frozen E191 controllers.

E191 already optimizes an objective whose physical-rehandle coefficient is
10.  This adapter therefore changes only the *difference* from that trained
coefficient::

    merit_lambda = Q_10 - reward_scale * (lambda - 10) * (I + N_hat)

At lambda=10 action selection delegates to E191 exactly.  The delegated
decision is merely wrapped afterwards so that its future-rehandle outcome can
still train the Stage-B consequence head.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from PSLAP.viability_candidates import ViabilityMode
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
    ConditionedHandlingDecision,
    FutureHandlingSample,
)
from viability_graph_hierarchy import (
    ID_TO_MODE,
    NoCertifiedViableAction,
    ViabilityGraphDecision,
)
from viability_graph_preference_conditioned import select_hierarchical_index


CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = "e204_e191_centered_rehandle_weight_adapter_v1"
NETWORK_ARCHITECTURE = "frozen_e191_features_lambda_conditioned_future_rehandles_v1"
TARGET_CONTRACT = "post_current_macro_undiscounted_physical_rehandles_mc_v1"


def detached_e191_features(base_network, records) -> Tensor:
    """Return E191's exact full pre-Q-head features with identity reuse.

    Unlike the historical VCG 1.1 Stage-B extractor, E191 also encodes its
    eight-field public queue/progress context.  Omitting that 32-dimensional
    embedding would silently define another architecture.
    """

    records = tuple(records)
    if not records:
        raise NoCertifiedViableAction("E204 received an empty record set")
    states = (
        tuple(record.current_state for record in records)
        + tuple(record.successor_state for record in records)
    )
    unique = []
    identity_to_index = {}
    inverse = []
    for state in states:
        identity = id(state)
        index = identity_to_index.get(identity)
        if index is None:
            index = len(unique)
            identity_to_index[identity] = index
            unique.append(state)
        elif unique[index] is not state:
            raise RuntimeError("Python object identity collision")
        inverse.append(index)
    parameter = next(base_network.parameters())
    with torch.no_grad():
        embeddings = fast_inputs._embed(base_network, unique).index_select(
            0,
            torch.as_tensor(inverse, dtype=torch.long, device=parameter.device),
        )
        count = len(records)
        action = torch.as_tensor(
            tuple(record.action_features for record in records),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        context = base_network.context_tensor(
            tuple(record.current_state for record in records)
        )
        features = torch.cat(
            (
                embeddings[:count],
                embeddings[count:],
                embeddings[count:] - embeddings[:count],
                base_network.action_encoder(action),
                base_network.context_encoder(context),
            ),
            dim=-1,
        )
    expected = int(base_network.q_head[0].in_features)
    if features.shape != (count, expected):
        raise RuntimeError("E204 full E191 feature width differs from Q head")
    return features.detach()


@dataclass(frozen=True)
class CenteredHandlingConfig:
    feature_dim: int
    hidden_dim: int = 128
    lambda_min: float = 0.0
    lambda_center: float = 10.0
    lambda_max: float = 40.0
    gamma_op: float = 1.0
    reward_scale: float = 0.01

    def __post_init__(self) -> None:
        for name in ("feature_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        values = (self.lambda_min, self.lambda_center, self.lambda_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("lambda bounds must be finite")
        if not 0.0 <= self.lambda_min < self.lambda_center < self.lambda_max:
            raise ValueError("lambda bounds must satisfy 0 <= min < center < max")
        if not math.isfinite(self.gamma_op) or not 0.0 <= self.gamma_op <= 1.0:
            raise ValueError("gamma_op must be finite and in [0,1]")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and positive")

    @property
    def gamma(self) -> float:
        return self.gamma_op

    def normalized_lambda(
        self,
        value,
        *,
        count: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim == 0:
            if count is None:
                tensor = tensor.reshape(1)
            else:
                tensor = tensor.expand(int(count))
        else:
            tensor = tensor.reshape(-1)
            if count is not None and tensor.numel() != int(count):
                raise ValueError("lambda count differs from feature count")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("lambda must be finite")
        if bool((tensor < self.lambda_min).any()) or bool(
            (tensor > self.lambda_max).any()
        ):
            raise ValueError("lambda is outside the declared Stage-B range")
        return (tensor - self.lambda_min) / (self.lambda_max - self.lambda_min)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> "CenteredHandlingConfig":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


class CenteredFutureHandlingNetwork(ConditionedFutureHandlingNetwork):
    """The historical head with E204's gamma-1-compatible configuration."""

    def initialize_near_zero(self, value: float = 0.01) -> None:
        """Use a deterministic low-information downstream-count prior."""

        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("initial prediction must be finite and positive")
        final = self.future_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("unexpected future-head output layer")
        bias = math.log(math.expm1(value))
        with torch.no_grad():
            final.weight.zero_()
            final.bias.fill_(bias)


def centered_calibration(
    network: CenteredFutureHandlingNetwork,
    samples: Sequence[FutureHandlingSample],
    *,
    device: torch.device,
) -> dict:
    samples = tuple(samples)
    if not samples:
        return {"sample_count": 0, "mae": None, "bias": None, "rmse": None}
    predictions: list[float] = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(samples), 1024):
            batch = samples[start : start + 1024]
            features = torch.stack([item.feature for item in batch]).to(device)
            lambdas = torch.as_tensor(
                [item.preference_lambda for item in batch],
                dtype=features.dtype,
                device=device,
            )
            predictions.extend(network(features, lambdas).cpu().tolist())
    targets = [float(item.future_rehandles) for item in samples]
    errors = [prediction - target for prediction, target in zip(predictions, targets)]

    def group(indices) -> dict:
        values = [errors[index] for index in indices]
        if not values:
            return {"sample_count": 0, "mae": None, "bias": None, "rmse": None}
        return {
            "sample_count": len(values),
            "mae": float(sum(abs(value) for value in values) / len(values)),
            "bias": float(sum(values) / len(values)),
            "rmse": float(math.sqrt(sum(value * value for value in values) / len(values))),
        }

    lambda_values = sorted({float(item.preference_lambda) for item in samples})
    actions = sorted({str(item.action_type) for item in samples})
    return {
        **group(range(len(samples))),
        "target_mean": float(sum(targets) / len(targets)),
        "prediction_mean": float(sum(predictions) / len(predictions)),
        "by_lambda": {
            f"{value:g}": group(
                i for i, item in enumerate(samples)
                if float(item.preference_lambda) == value
            )
            for value in lambda_values
        },
        "by_action_type": {
            action: group(
                i for i, item in enumerate(samples)
                if str(item.action_type) == action
            )
            for action in actions
        },
    }


def fit_centered_future_handling(
    network: CenteredFutureHandlingNetwork,
    training: Sequence[FutureHandlingSample],
    validation: Sequence[FutureHandlingSample],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    training = tuple(training)
    validation = tuple(validation)
    if not training or not validation:
        raise ValueError("centered handling fit requires train and validation samples")
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    rng = random.Random(int(seed))
    initial = centered_calibration(network, validation, device=device)
    records = []
    network.requires_grad_(True).train()
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(training)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(batch_size)):
            batch = [training[i] for i in order[start : start + int(batch_size)]]
            features = torch.stack([item.feature for item in batch]).to(device)
            lambdas = torch.as_tensor(
                [item.preference_lambda for item in batch],
                dtype=features.dtype,
                device=device,
            )
            target = torch.as_tensor(
                [item.future_rehandles for item in batch],
                dtype=features.dtype,
                device=device,
            )
            prediction = network(features, lambdas)
            loss = F.smooth_l1_loss(prediction, target, beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        validation_record = centered_calibration(network, validation, device=device)
        records.append({
            "epoch": epoch,
            "training_loss_mean": float(sum(losses) / len(losses)),
            "validation": validation_record,
        })
    network.requires_grad_(False).eval()
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "initial_validation": initial,
        "epoch_records": records,
        "final_validation": records[-1]["validation"],
    }


class CenteredHandlingAgent(ConditionedHandlingAgent):
    """Frozen E191 policy plus a centered future-rehandle correction."""

    config: CenteredHandlingConfig
    handling_network: CenteredFutureHandlingNetwork

    def __init__(
        self,
        base_agent,
        handling_network: CenteredFutureHandlingNetwork,
        *,
        config: CenteredHandlingConfig,
        seed: int,
        epsilon: float = 0.0,
    ) -> None:
        self.base_agent = base_agent
        self.handling_network = handling_network.to(base_agent.device)
        self.config = config
        self.seed = int(seed)
        self.device = base_agent.device
        expected = int(base_agent.Q_local.q_head[0].in_features)
        if config.feature_dim != expected:
            raise ValueError("centered feature width does not match E191 Q")
        if config.gamma_op != base_agent.config.gamma:
            raise ValueError("centered gamma must match E191")
        if config.reward_scale != base_agent.config.reward_scale:
            raise ValueError("centered reward scale must match E191")
        for network in (base_agent.Q_local, base_agent.Q_target):
            network.requires_grad_(False).eval()
        self.handling_network.requires_grad_(False).eval()
        self.rng = random.Random(self.seed)
        self.epsilon = self._validate_epsilon(epsilon)
        self.decision_count = 0
        self.transition_count = 0
        self.preference_decisions: Counter[str] = Counter()
        self.pending_observations = []
        self.completed_samples = None

    def _validate_lambda(self, value: float) -> float:
        value = float(value)
        self.config.normalized_lambda(value)
        return value

    def _wrap_delegated_center(self, decision, value: float):
        records = tuple(decision.prepared_snapshot.records)
        selected_index = next(
            index for index, record in enumerate(records)
            if record.key == decision.record.key
        )
        # Use the same complete-frontier batch as E191.  Besides keeping the
        # stored feature consistent with off-center collection, this avoids
        # batch-shape-dependent floating-point differences in graph pooling.
        features = detached_e191_features(self.base_agent.Q_local, records)
        with torch.no_grad():
            operational = self.base_agent.Q_local.q_head(features).squeeze(-1)
            immediate = int(decision.candidate.action_type.value == "reconfigure")
            handling = torch.as_tensor(
                [1.0 if record.action_type == "reconfigure" else 0.0 for record in records],
                dtype=features.dtype,
                device=features.device,
            ) + self.handling_network(features, value)
        self.decision_count += 1
        self.preference_decisions[str(value)] += 1
        return ConditionedHandlingDecision(
            base_decision=decision,
            preference_lambda=value,
            operational_values=tuple(float(item) for item in operational.cpu()),
            handling_values=tuple(float(item) for item in handling.cpu()),
            chosen_feature=features[selected_index].detach().cpu().clone(),
            chosen_immediate=immediate,
        )

    def select(
        self,
        snapshot,
        *,
        preference_lambda: float,
        training: bool = False,
        epsilon: Optional[float] = None,
    ):
        value = self._validate_lambda(preference_lambda)
        epsilon_value = self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        if value == self.config.lambda_center:
            if epsilon_value != 0.0:
                raise ValueError("lambda-center collection must preserve frozen E191")
            delegated = self.base_agent.select(snapshot, training=False, epsilon=0.0)
            return self._wrap_delegated_center(delegated, value)

        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE centered candidate")
        features = detached_e191_features(self.base_agent.Q_local, prepared.records)
        with torch.no_grad():
            operational = self.base_agent.Q_local.q_head(features).squeeze(-1)
            immediate = torch.as_tensor(
                [
                    1.0 if item.action_type == "reconfigure" else 0.0
                    for item in prepared.records
                ],
                dtype=features.dtype,
                device=features.device,
            )
            handling = immediate + self.handling_network(features, value)
            correction = self.config.reward_scale * (
                value - self.config.lambda_center
            )
            merit = operational - correction * handling
        hierarchy = select_hierarchical_index(
            merit,
            prepared.mode_ids,
            self.base_agent.within_temperatures,
            candidate_keys=tuple(item.key for item in prepared.records),
        )
        explored = bool(
            training and not liveness_forced and self.rng.random() < epsilon_value
        )
        if liveness_forced:
            selected_index = 0
            source = "exact_recovery_witness_guard"
        elif explored:
            live_modes = tuple(mode for mode, _ in hierarchy.mode_values)
            selected_mode = self.rng.choice(live_modes)
            indices = [
                i for i, mode in enumerate(prepared.mode_ids) if mode == selected_mode
            ]
            selected_index = self.rng.choice(indices)
            source = "safe_mode_balanced_epsilon"
        else:
            selected_index = hierarchy.selected_index
            source = (
                "singleton_safe"
                if len(prepared.records) == 1
                else "regularized_mode_map_candidate_map_centered_rehandle"
            )
        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        mode_pairs = tuple(
            (ID_TO_MODE[mode], score) for mode, score in hierarchy.mode_values
        )
        base_decision = ViabilityGraphDecision(
            candidate=candidate,
            record=prepared.records[selected_index],
            prepared_snapshot=prepared,
            q_values=tuple(float(item) for item in merit.detach().cpu()),
            mode_values=mode_pairs,
            explored=explored,
            selection_source=source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[source] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        if prepared.liveness_restricted:
            base.recovery_witness_guard.record_selection(
                action_type=candidate.action_type.value,
                forced=liveness_forced,
            )
        self.decision_count += 1
        self.preference_decisions[str(value)] += 1
        return ConditionedHandlingDecision(
            base_decision=base_decision,
            preference_lambda=value,
            operational_values=tuple(float(item) for item in operational.detach().cpu()),
            handling_values=tuple(float(item) for item in handling.detach().cpu()),
            chosen_feature=features[selected_index].detach().cpu().clone(),
            chosen_immediate=int(immediate[selected_index].item()),
        )

    def checkpoint(
        self,
        *,
        base_checkpoint_sha256: str,
        base_policy_digest: str,
    ) -> dict:
        if self.pending_observations or self.completed_samples is not None:
            raise ValueError("centered checkpoint requires a clean episode boundary")
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "target_contract": TARGET_CONTRACT,
            "config": self.config.to_dict(),
            "base_checkpoint_sha256": str(base_checkpoint_sha256),
            "base_policy_digest": str(base_policy_digest),
            "lambda_center": self.config.lambda_center,
            "lambda_center_exact_e191_delegation": True,
            "operational_critic_frozen": True,
            "exact_immediate_rehandle_term": True,
            "centered_reward_scale_applied_once": True,
            "network_state": self.handling_network.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping,
        *,
        base_agent,
        expected_base_checkpoint_sha256: str,
        expected_base_policy_digest: str,
        seed: int,
    ) -> "CenteredHandlingAgent":
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "target_contract": TARGET_CONTRACT,
            "base_checkpoint_sha256": expected_base_checkpoint_sha256,
            "base_policy_digest": expected_base_policy_digest,
            "lambda_center_exact_e191_delegation": True,
            "operational_critic_frozen": True,
            "exact_immediate_rehandle_term": True,
            "centered_reward_scale_applied_once": True,
        }
        mismatch = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatch:
            raise ValueError(f"incompatible centered checkpoint: {mismatch}")
        config = CenteredHandlingConfig.from_dict(payload["config"])
        if float(payload.get("lambda_center", math.nan)) != config.lambda_center:
            raise ValueError("centered checkpoint lambda identity differs")
        network = CenteredFutureHandlingNetwork(config, seed=seed).to(base_agent.device)
        network.load_state_dict(payload["network_state"])
        network.requires_grad_(False).eval()
        return cls(
            base_agent,
            network,
            config=config,
            seed=seed,
            epsilon=0.0,
        )

    def audit(self, *, include_decisions: bool = False) -> dict:
        del include_decisions
        return {
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "operational_critic_frozen": True,
            "lambda_center": self.config.lambda_center,
            "lambda_center_exact_e191_delegation": True,
            "centered_reward_scale": self.config.reward_scale,
            "exact_immediate_rehandle_term": True,
            "future_handling_nonnegative": True,
            "target_contract": TARGET_CONTRACT,
            "decision_count": self.decision_count,
            "transition_count": self.transition_count,
            "preference_decisions": dict(self.preference_decisions),
            "base_audit": self.base_agent.audit(include_decisions=False),
        }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONTROLLER_ARCHITECTURE",
    "CenteredFutureHandlingNetwork",
    "CenteredHandlingAgent",
    "CenteredHandlingConfig",
    "NETWORK_ARCHITECTURE",
    "TARGET_CONTRACT",
    "centered_calibration",
    "detached_e191_features",
    "fit_centered_future_handling",
]
