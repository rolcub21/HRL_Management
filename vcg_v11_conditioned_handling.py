"""VCG 1.1 with a policy-conditioned future physical-handling model.

The operational critic and exact viability machinery are immutable.  The only
learned adaptation is the downstream physical-handling consequence:

    Q_N(s,c,lambda) = 1[c is Reconfigure] + softplus(f_N(h(s,c), lambda)).

The head is trained from complete-episode Monte Carlo labels that exclude the
known immediate term.  At lambda zero selection delegates directly to VCG 1.1.
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
from vcg_v11_nested_handling import (
    DetachedHandlingCostNetwork,
    detached_v11_features,
)
from viability_graph_hierarchy import (
    ID_TO_MODE,
    NoCertifiedViableAction,
    ViabilityGraphDecision,
    ViabilityGraphHierarchyAgent,
)
from viability_graph_preference_conditioned import (
    normalize_preference_lambda,
    select_hierarchical_index,
)


CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = "vcg_v1_1_conditioned_future_handling_v1"
NETWORK_ARCHITECTURE = "frozen_v1_features_lambda_conditioned_future_cost_v1"
TARGET_CONTRACT = "post_current_macro_undiscounted_physical_rehandles_mc_v1"


@dataclass(frozen=True)
class ConditionedHandlingConfig:
    feature_dim: int
    hidden_dim: int = 128
    lambda_max: float = 0.2
    gamma_op: float = 0.99
    reward_scale: float = 0.01

    def __post_init__(self) -> None:
        for name in ("feature_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.lambda_max) or self.lambda_max <= 0:
            raise ValueError("lambda_max must be finite and positive")
        if not math.isfinite(self.gamma_op) or not 0 <= self.gamma_op < 1:
            raise ValueError("gamma_op must be finite and in [0,1)")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")

    @property
    def gamma(self) -> float:
        return self.gamma_op

    def normalized_lambda(self, value, **kwargs) -> Tensor:
        return normalize_preference_lambda(value, self.lambda_max, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> "ConditionedHandlingConfig":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


class ConditionedFutureHandlingNetwork(nn.Module):
    """Nonnegative downstream handling predictor over frozen V1 features."""

    def __init__(self, config: ConditionedHandlingConfig, *, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.future_head = nn.Sequential(
                nn.Linear(config.feature_dim + 1, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )

    def initialize_from_nested_head(
        self, source: DetachedHandlingCostNetwork
    ) -> None:
        """Copy the old future head and initially ignore lambda exactly."""

        source_layers = source.cost_head
        target_layers = self.future_head
        with torch.no_grad():
            if target_layers[0].weight.shape[1] != source_layers[0].weight.shape[1] + 1:
                raise ValueError("conditioned handling warm-start width mismatch")
            target_layers[0].weight.zero_()
            target_layers[0].weight[:, :-1].copy_(source_layers[0].weight)
            target_layers[0].bias.copy_(source_layers[0].bias)
            for index in (2, 4):
                target_layers[index].weight.copy_(source_layers[index].weight)
                target_layers[index].bias.copy_(source_layers[index].bias)

    def forward(self, features: Tensor, preference_lambda) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("conditioned handling features have the wrong shape")
        alpha = self.config.normalized_lambda(
            preference_lambda,
            count=features.shape[0],
            dtype=features.dtype,
            device=features.device,
        )
        joined = torch.cat((features, alpha.unsqueeze(-1)), dim=-1)
        return F.softplus(self.future_head(joined).squeeze(-1))


@dataclass(frozen=True)
class ConditionedHandlingDecision:
    base_decision: ViabilityGraphDecision
    preference_lambda: float
    operational_values: tuple[float, ...]
    handling_values: tuple[float, ...]
    chosen_feature: Tensor
    chosen_immediate: int

    @property
    def candidate(self):
        return self.base_decision.candidate

    @property
    def record(self):
        return self.base_decision.record

    @property
    def prepared_snapshot(self):
        return self.base_decision.prepared_snapshot

    @property
    def option(self):
        return self.base_decision.option

    @property
    def explored(self):
        return self.base_decision.explored

    @property
    def selection_source(self):
        return self.base_decision.selection_source

    @property
    def liveness_forced(self):
        return self.base_decision.liveness_forced

    @property
    def exact_rank_progress(self):
        return self.base_decision.exact_rank_progress


@dataclass(frozen=True)
class PendingHandlingObservation:
    feature: Tensor
    preference_lambda: float
    immediate_rehandles: int
    realized_rehandles: int
    action_type: str
    mode_id: int
    candidate_key: str


@dataclass(frozen=True)
class FutureHandlingSample:
    feature: Tensor
    preference_lambda: float
    future_rehandles: int
    immediate_rehandles: int
    action_type: str
    mode_id: int
    candidate_key: str

    def numeric_copy(self) -> "FutureHandlingSample":
        return FutureHandlingSample(
            feature=self.feature.detach().cpu().clone(),
            preference_lambda=float(self.preference_lambda),
            future_rehandles=int(self.future_rehandles),
            immediate_rehandles=int(self.immediate_rehandles),
            action_type=str(self.action_type),
            mode_id=int(self.mode_id),
            candidate_key=str(self.candidate_key),
        )


def future_samples_from_episode(
    observations: Sequence[PendingHandlingObservation],
) -> tuple[FutureHandlingSample, ...]:
    """Label future cost after the current macro, avoiding double counting."""

    observations = tuple(observations)
    future = 0
    reversed_samples = []
    for item in reversed(observations):
        if item.realized_rehandles != item.immediate_rehandles:
            raise ValueError(
                "realized physical handling disagrees with the action-semantic term"
            )
        reversed_samples.append(
            FutureHandlingSample(
                feature=item.feature,
                preference_lambda=item.preference_lambda,
                future_rehandles=future,
                immediate_rehandles=item.immediate_rehandles,
                action_type=item.action_type,
                mode_id=item.mode_id,
                candidate_key=item.candidate_key,
            )
        )
        future += item.realized_rehandles
    return tuple(sample.numeric_copy() for sample in reversed(reversed_samples))


def handling_calibration(
    network: ConditionedFutureHandlingNetwork,
    samples: Sequence[FutureHandlingSample],
    *,
    device: torch.device,
) -> dict:
    samples = tuple(samples)
    if not samples:
        return {"sample_count": 0, "mae": None, "bias": None, "rmse": None}
    network.eval()
    predictions = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features = torch.stack([item.feature for item in batch]).to(device)
            preferences = torch.as_tensor(
                [item.preference_lambda for item in batch],
                dtype=features.dtype,
                device=device,
            )
            predictions.extend(network(features, preferences).cpu().tolist())
    targets = [float(item.future_rehandles) for item in samples]
    errors = [prediction - target for prediction, target in zip(predictions, targets)]

    def group(indices):
        values = [errors[index] for index in indices]
        if not values:
            return {"sample_count": 0, "mae": None, "bias": None, "rmse": None}
        return {
            "sample_count": len(values),
            "mae": float(sum(abs(value) for value in values) / len(values)),
            "bias": float(sum(values) / len(values)),
            "rmse": float(math.sqrt(sum(value * value for value in values) / len(values))),
        }

    bins = {
        "low_[0,0.05)": [
            i for i, item in enumerate(samples) if item.preference_lambda < 0.05
        ],
        "middle_[0.05,0.15)": [
            i for i, item in enumerate(samples) if 0.05 <= item.preference_lambda < 0.15
        ],
        "high_[0.15,0.2]": [
            i for i, item in enumerate(samples) if item.preference_lambda >= 0.15
        ],
    }
    actions = {
        action: [i for i, item in enumerate(samples) if item.action_type == action]
        for action in sorted({item.action_type for item in samples})
    }
    return {
        **group(range(len(samples))),
        "target_mean": float(sum(targets) / len(targets)),
        "prediction_mean": float(sum(predictions) / len(predictions)),
        "by_lambda_bin": {name: group(indices) for name, indices in bins.items()},
        "by_action_type": {name: group(indices) for name, indices in actions.items()},
    }


def fit_conditioned_future_handling(
    network: ConditionedFutureHandlingNetwork,
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
        raise ValueError("conditioned handling fit requires train and validation samples")
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    rng = random.Random(int(seed))
    initial = handling_calibration(network, validation, device=device)
    epoch_records = []
    network.requires_grad_(True)
    network.train()
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(training)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(batch_size)):
            selected = [training[i] for i in order[start : start + int(batch_size)]]
            features = torch.stack([item.feature for item in selected]).to(device)
            preferences = torch.as_tensor(
                [item.preference_lambda for item in selected],
                dtype=features.dtype,
                device=device,
            )
            target = torch.as_tensor(
                [item.future_rehandles for item in selected],
                dtype=features.dtype,
                device=device,
            )
            prediction = network(features, preferences)
            loss = F.smooth_l1_loss(prediction, target, beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        validation_record = handling_calibration(network, validation, device=device)
        epoch_records.append(
            {
                "epoch": epoch,
                "training_loss_mean": float(sum(losses) / len(losses)),
                "validation": validation_record,
            }
        )
    network.requires_grad_(False).eval()
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "initial_validation": initial,
        "epoch_records": epoch_records,
        "final_validation": epoch_records[-1]["validation"],
    }


class ConditionedHandlingAgent:
    """Executor adapter and frozen-policy Monte Carlo trajectory collector."""

    def __init__(
        self,
        base_agent: ViabilityGraphHierarchyAgent,
        handling_network: ConditionedFutureHandlingNetwork,
        *,
        config: ConditionedHandlingConfig,
        seed: int,
        epsilon: float = 0.0,
    ) -> None:
        self.base_agent = base_agent
        self.handling_network = handling_network.to(base_agent.device)
        self.config = config
        self.seed = int(seed)
        self.device = base_agent.device
        expected = (
            3 * base_agent.config.graph_embedding_dim
            + base_agent.config.action_embedding_dim
        )
        if config.feature_dim != expected:
            raise ValueError("conditioned handling feature width does not match VCG 1.1")
        if config.gamma_op != base_agent.config.gamma:
            raise ValueError("conditioned handling gamma must match VCG 1.1")
        if config.reward_scale != base_agent.config.reward_scale:
            raise ValueError("conditioned handling reward scale must match VCG 1.1")
        for network in (base_agent.Q_local, base_agent.Q_target):
            network.requires_grad_(False).eval()
        self.handling_network.requires_grad_(False).eval()
        self.rng = random.Random(self.seed)
        self.epsilon = self._validate_epsilon(epsilon)
        self.decision_count = 0
        self.transition_count = 0
        self.preference_decisions: Counter[str] = Counter()
        self.pending_observations: list[PendingHandlingObservation] = []
        self.completed_samples: Optional[tuple[FutureHandlingSample, ...]] = None

    @staticmethod
    def _validate_epsilon(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("epsilon must lie in [0,1]")
        return value

    def _validate_lambda(self, value: float) -> float:
        value = float(value)
        self.config.normalized_lambda(value)
        return value

    def set_epsilon(self, value: float) -> None:
        self.epsilon = self._validate_epsilon(value)
        self.base_agent.set_epsilon(self.epsilon)

    def reset_episode_state(self) -> None:
        if self.pending_observations or self.completed_samples is not None:
            raise RuntimeError("previous handling trajectory was not consumed")
        self.base_agent.reset_episode_state()

    on_episode_reset = reset_episode_state

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
        if value == 0.0:
            if training and epsilon_value != 0.0:
                raise ValueError("lambda-zero collection must preserve VCG 1.1")
            return self.base_agent.select(
                snapshot, training=False, epsilon=0.0
            )
        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE conditioned candidate")
        features = detached_v11_features(self.base_agent.Q_local, prepared.records)
        with torch.no_grad():
            operational = self.base_agent.Q_local.q_head(features).squeeze(-1)
            immediate = torch.as_tensor(
                [1.0 if item.action_type == "reconfigure" else 0.0 for item in prepared.records],
                dtype=features.dtype,
                device=features.device,
            )
            handling = immediate + self.handling_network(features, value)
            merit = operational - value * handling
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
                else "regularized_mode_map_candidate_map_handling_augmented"
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

    def observe_outcome(self, decision, *, next_snapshot, done):
        base_decision = (
            decision.base_decision
            if isinstance(decision, ConditionedHandlingDecision)
            else decision
        )
        return self.base_agent.observe_outcome(
            base_decision, next_snapshot=next_snapshot, done=done
        )

    def remember(
        self,
        decision,
        *,
        operational_return: float,
        discounted_rehandles: float,
        raw_rehandles: int,
        duration: int,
        next_snapshot,
        done: bool,
        outcome_already_observed: bool = False,
    ):
        del operational_return, discounted_rehandles
        if not isinstance(decision, ConditionedHandlingDecision):
            raise TypeError("handling collection requires a positive-lambda decision")
        if not outcome_already_observed:
            self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=done
            )
        if int(duration) < 1:
            raise ValueError("handling observation requires positive macro duration")
        observation = PendingHandlingObservation(
            feature=decision.chosen_feature,
            preference_lambda=decision.preference_lambda,
            immediate_rehandles=decision.chosen_immediate,
            realized_rehandles=int(raw_rehandles),
            action_type=decision.candidate.action_type.value,
            mode_id=int(decision.record.mode_id),
            candidate_key=decision.candidate.key,
        )
        self.pending_observations.append(observation)
        self.transition_count += 1
        if done:
            self.completed_samples = future_samples_from_episode(
                self.pending_observations
            )
            self.pending_observations.clear()
        return observation

    def learn(self):
        return None

    def pop_completed_samples(self) -> tuple[FutureHandlingSample, ...]:
        if self.completed_samples is None:
            raise RuntimeError("episode did not produce complete handling samples")
        result = self.completed_samples
        self.completed_samples = None
        return result

    def checkpoint(
        self,
        *,
        base_checkpoint_sha256: str,
        base_policy_digest: str,
        source_cost_sha256: str,
    ) -> dict:
        if self.pending_observations or self.completed_samples is not None:
            raise ValueError("handling checkpoint requires a clean episode boundary")
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "target_contract": TARGET_CONTRACT,
            "config": self.config.to_dict(),
            "base_checkpoint_sha256": str(base_checkpoint_sha256),
            "base_policy_digest": str(base_policy_digest),
            "source_cost_sha256": str(source_cost_sha256),
            "lambda_zero_direct_vcg_v1_1_delegation": True,
            "operational_critic_frozen": True,
            "exact_immediate_rehandle_term": True,
            "network_state": self.handling_network.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping,
        *,
        base_agent: ViabilityGraphHierarchyAgent,
        expected_base_checkpoint_sha256: str,
        expected_base_policy_digest: str,
        expected_source_cost_sha256: str,
        seed: int,
    ) -> "ConditionedHandlingAgent":
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "target_contract": TARGET_CONTRACT,
            "base_checkpoint_sha256": expected_base_checkpoint_sha256,
            "base_policy_digest": expected_base_policy_digest,
            "source_cost_sha256": expected_source_cost_sha256,
            "lambda_zero_direct_vcg_v1_1_delegation": True,
            "operational_critic_frozen": True,
            "exact_immediate_rehandle_term": True,
        }
        mismatch = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatch:
            raise ValueError(f"incompatible conditioned handling checkpoint: {mismatch}")
        config = ConditionedHandlingConfig.from_dict(payload["config"])
        network = ConditionedFutureHandlingNetwork(config, seed=seed).to(
            base_agent.device
        )
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
            "lambda_zero_direct_vcg_v1_1_delegation": True,
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
    "ConditionedFutureHandlingNetwork",
    "ConditionedHandlingAgent",
    "ConditionedHandlingConfig",
    "FutureHandlingSample",
    "NETWORK_ARCHITECTURE",
    "TARGET_CONTRACT",
    "fit_conditioned_future_handling",
    "future_samples_from_episode",
    "handling_calibration",
]
