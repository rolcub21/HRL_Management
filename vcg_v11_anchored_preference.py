"""VCG 1.1 anchored preference-conditioned residual controller.

The frozen VCG 1.1 operational controller remains the anchor.  At lambda zero
selection delegates directly to that controller.  At positive lambda the
learned consequences are

    Q_op = Q_op^V1.1 + (lambda / lambda_max) * Delta Q_op
    Q_N  = I[Reconfigure] + softplus(f_N(h, lambda)).

The residual and handling towers consume detached V1.1 candidate features.
They therefore cannot alter the exact certificate, the liveness guard, or the
frozen V1.1 encoder/Q parameters.  Handling uses an undiscounted finite-episode
return because the reported operational metric is the physical rehandle count;
the operational component retains V1.1's primitive-time discount.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
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
    PreparedViabilityGraphSnapshot,
    ViabilityGraphCandidateRecord,
    ViabilityGraphDecision,
    ViabilityGraphHierarchyAgent,
)
from viability_graph_preference_conditioned import (
    normalize_preference_lambda,
    scalarized_merit,
    select_hierarchical_index,
)


CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = "vcg_v1_1_anchored_preference_residual_v1"
NETWORK_ARCHITECTURE = (
    "frozen_v1_features_lambda_conditioned_separate_delta_op_and_raw_cost_v1"
)
BACKUP_VERSION = (
    "cached_feature_double_q_op_gamma0p99_raw_handling_gamma1_shared_action_v1"
)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class AnchoredPreferenceConfig:
    feature_dim: int
    hidden_dim: int = 128
    lambda_max: float = 0.2
    gamma_op: float = 0.99
    reward_scale: float = 0.01
    learning_rate: float = 5.0e-5
    batch_size: int = 64
    replay_capacity: int = 20_000
    update_every: int = 1
    target_update_every: int = 200
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    operational_loss_weight: float = 1.0
    handling_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "feature_dim",
            "hidden_dim",
            "batch_size",
            "replay_capacity",
            "update_every",
            "target_update_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "lambda_max",
            "reward_scale",
            "learning_rate",
            "grad_clip",
            "huber_delta",
            "operational_loss_weight",
            "handling_loss_weight",
        ):
            _positive(getattr(self, name), name)
        if not math.isfinite(float(self.gamma_op)) or not 0.0 <= self.gamma_op < 1.0:
            raise ValueError("gamma_op must be finite and in [0, 1)")

    def normalized_lambda(self, value, **kwargs) -> Tensor:
        return normalize_preference_lambda(value, self.lambda_max, **kwargs)

    @property
    def gamma(self) -> float:
        """Executor-compatible name for the operational SMDP discount."""

        return self.gamma_op

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> "AnchoredPreferenceConfig":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


class AnchoredResidualNetwork(nn.Module):
    """Separate preference-conditioned towers over frozen V1.1 features."""

    def __init__(self, config: AnchoredPreferenceConfig, *, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = int(seed)
        width = config.feature_dim + 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.operational_residual = nn.Sequential(
                nn.Linear(width, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.handling_future = nn.Sequential(
                nn.Linear(width, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )
        # The initial operational policy is exactly the frozen anchor.
        nn.init.zeros_(self.operational_residual[-1].weight)
        nn.init.zeros_(self.operational_residual[-1].bias)

    def initialize_handling_from(
        self, source: DetachedHandlingCostNetwork
    ) -> None:
        """Embed the old lambda-independent MC head with a zero lambda column."""

        source_layers = source.cost_head
        target_layers = self.handling_future
        if not (
            isinstance(source_layers[0], nn.Linear)
            and isinstance(source_layers[2], nn.Linear)
            and isinstance(source_layers[4], nn.Linear)
        ):
            raise ValueError("unexpected detached handling-head architecture")
        with torch.no_grad():
            first_source = source_layers[0]
            first_target = target_layers[0]
            if first_target.weight.shape[1] != first_source.weight.shape[1] + 1:
                raise ValueError("handling warm-start feature width mismatch")
            first_target.weight.zero_()
            first_target.weight[:, :-1].copy_(first_source.weight)
            first_target.bias.copy_(first_source.bias)
            for source_index, target_index in ((2, 2), (4, 4)):
                target_layers[target_index].weight.copy_(
                    source_layers[source_index].weight
                )
                target_layers[target_index].bias.copy_(
                    source_layers[source_index].bias
                )

    def forward(
        self,
        features: Tensor,
        preference_lambda: Tensor | Sequence[float] | float,
        immediate_reconfigure: Tensor,
    ) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("anchored features have the wrong matrix shape")
        preference = self.config.normalized_lambda(
            preference_lambda,
            count=features.shape[0],
            dtype=features.dtype,
            device=features.device,
        )
        immediate = torch.as_tensor(
            immediate_reconfigure,
            dtype=features.dtype,
            device=features.device,
        )
        if immediate.shape != (features.shape[0],):
            raise ValueError("immediate rehandle indicator must align with features")
        joined = torch.cat((features, preference.unsqueeze(-1)), dim=-1)
        delta_op = self.operational_residual(joined).squeeze(-1)
        q_n = immediate + F.softplus(self.handling_future(joined).squeeze(-1))
        return torch.stack((delta_op, q_n), dim=-1)


@dataclass(frozen=True)
class AnchoredDecision:
    base_decision: ViabilityGraphDecision
    preference_lambda: float
    q_vectors: tuple[tuple[float, float], ...]
    cached_chosen_feature: Optional[Tensor] = None
    cached_chosen_base_q: Optional[float] = None
    cached_chosen_immediate: Optional[float] = None

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
class CachedAnchoredTransition:
    chosen_feature: Tensor
    chosen_base_q: float
    chosen_immediate: float
    chosen_mode_id: int
    operational_return: float
    raw_rehandles: int
    duration: int
    next_features: Tensor
    next_base_q: Tensor
    next_immediate: Tensor
    next_mode_ids: tuple[int, ...]
    next_keys: tuple[str, ...]
    done: bool
    behavior_lambda: float

    def __post_init__(self) -> None:
        if self.chosen_feature.ndim != 1:
            raise ValueError("chosen feature must be one-dimensional")
        if self.next_features.ndim != 2:
            raise ValueError("next features must be a matrix")
        count = self.next_features.shape[0]
        if self.next_base_q.shape != (count,) or self.next_immediate.shape != (count,):
            raise ValueError("cached next tensors do not align")
        if len(self.next_mode_ids) != count or len(self.next_keys) != count:
            raise ValueError("cached next metadata does not align")
        if self.done and count:
            raise ValueError("terminal transition retained a next frontier")
        if not self.done and not count:
            raise ValueError("nonterminal transition needs a next frontier")
        if self.duration < 1 or self.raw_rehandles < 0:
            raise ValueError("invalid macro outcome")

    def numeric_copy(self) -> "CachedAnchoredTransition":
        return CachedAnchoredTransition(
            chosen_feature=self.chosen_feature.detach().cpu().clone(),
            chosen_base_q=float(self.chosen_base_q),
            chosen_immediate=float(self.chosen_immediate),
            chosen_mode_id=int(self.chosen_mode_id),
            operational_return=float(self.operational_return),
            raw_rehandles=int(self.raw_rehandles),
            duration=int(self.duration),
            next_features=self.next_features.detach().cpu().clone(),
            next_base_q=self.next_base_q.detach().cpu().clone(),
            next_immediate=self.next_immediate.detach().cpu().clone(),
            next_mode_ids=tuple(int(value) for value in self.next_mode_ids),
            next_keys=tuple(self.next_keys),
            done=bool(self.done),
            behavior_lambda=float(self.behavior_lambda),
        )


class AnchoredReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.memory = deque(maxlen=self.capacity)

    def add(self, transition: CachedAnchoredTransition) -> None:
        if not isinstance(transition, CachedAnchoredTransition):
            raise TypeError("anchored replay accepts CachedAnchoredTransition only")
        self.memory.append(transition.numeric_copy())

    def sample_mode_balanced(self, batch_size: int, rng: random.Random):
        memory = list(self.memory)
        if batch_size <= 0 or batch_size > len(memory):
            raise ValueError("invalid replay batch size")
        buckets: dict[int, list[int]] = {}
        for index, item in enumerate(memory):
            buckets.setdefault(item.chosen_mode_id, []).append(index)
        live = sorted(buckets)
        base, remainder = divmod(batch_size, len(live))
        selected: list[int] = []
        selected_set: set[int] = set()
        for position, mode_id in enumerate(live):
            requested = base + int(position < remainder)
            chosen = rng.sample(
                buckets[mode_id], min(requested, len(buckets[mode_id]))
            )
            selected.extend(chosen)
            selected_set.update(chosen)
        if len(selected) < batch_size:
            remaining = [
                index for index in range(len(memory)) if index not in selected_set
            ]
            selected.extend(rng.sample(remaining, batch_size - len(selected)))
        rng.shuffle(selected)
        return [memory[index] for index in selected]

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "memory": list(self.memory)}

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch")
        self.memory = deque(
            (item.numeric_copy() for item in state["memory"]),
            maxlen=self.capacity,
        )

    def __len__(self) -> int:
        return len(self.memory)


class AnchoredPreferenceAgent:
    """Frozen VCG 1.1 plus trainable preference-conditioned residual heads."""

    def __init__(
        self,
        base_agent: ViabilityGraphHierarchyAgent,
        *,
        config: AnchoredPreferenceConfig,
        seed: int,
        warm_start_cost: Optional[DetachedHandlingCostNetwork] = None,
        epsilon: float = 0.25,
    ) -> None:
        self.base_agent = base_agent
        self.config = config
        self.seed = int(seed)
        self.device = base_agent.device
        expected_width = (
            3 * base_agent.config.graph_embedding_dim
            + base_agent.config.action_embedding_dim
        )
        if config.feature_dim != expected_width:
            raise ValueError("anchored feature dimension does not match VCG 1.1")
        if config.gamma_op != base_agent.config.gamma:
            raise ValueError("anchored operational gamma must match VCG 1.1")
        if config.reward_scale != base_agent.config.reward_scale:
            raise ValueError("anchored reward scale must match VCG 1.1")
        for network in (base_agent.Q_local, base_agent.Q_target):
            network.requires_grad_(False).eval()
        self.Q_local = AnchoredResidualNetwork(config, seed=seed).to(self.device)
        if warm_start_cost is not None:
            self.Q_local.initialize_handling_from(warm_start_cost)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=config.learning_rate
        )
        self.replay = AnchoredReplayBuffer(config.replay_capacity)
        self.rng = random.Random(self.seed)
        self.epsilon = self._validate_epsilon(epsilon)
        self.transition_count = 0
        self.decision_count = 0
        self.gradient_steps = 0
        self.target_updates = 0
        self.loss_history: list[float] = []
        self.operational_loss_history: list[float] = []
        self.handling_loss_history: list[float] = []
        self.preference_decisions: Counter[str] = Counter()

    @staticmethod
    def _validate_epsilon(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        return value

    def set_epsilon(self, value: float) -> None:
        self.epsilon = self._validate_epsilon(value)
        self.base_agent.set_epsilon(self.epsilon)

    def _validate_lambda(self, value: float) -> float:
        value = float(value)
        self.config.normalized_lambda(value)
        return value

    @staticmethod
    def _immediate(records: Sequence[ViabilityGraphCandidateRecord]) -> Tensor:
        return torch.as_tensor(
            [1.0 if item.action_type == "reconfigure" else 0.0 for item in records],
            dtype=torch.float32,
        )

    def _cache_records(self, records: Sequence[ViabilityGraphCandidateRecord]):
        records = tuple(records)
        if not records:
            width = self.config.feature_dim
            return (
                torch.empty((0, width), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
        features = detached_v11_features(
            self.base_agent.Q_local, records
        ).detach()
        with torch.no_grad():
            base_q = self.base_agent.Q_local.q_head(features).squeeze(-1)
        immediate = self._immediate(records).to(features.device, features.dtype)
        return features.cpu(), base_q.cpu(), immediate.cpu()

    def _absolute_vectors(
        self,
        network: AnchoredResidualNetwork,
        features: Tensor,
        base_q: Tensor,
        immediate: Tensor,
        preference_lambda: Tensor | Sequence[float] | float,
    ) -> Tensor:
        features = features.to(self.device)
        base_q = base_q.to(self.device, features.dtype)
        immediate = immediate.to(self.device, features.dtype)
        raw = network(features, preference_lambda, immediate)
        preference = torch.as_tensor(
            preference_lambda, dtype=features.dtype, device=self.device
        )
        if preference.ndim == 0:
            preference = preference.expand(features.shape[0])
        alpha = preference / self.config.lambda_max
        q_op = base_q + alpha * raw[:, 0]
        return torch.stack((q_op, raw[:, 1]), dim=-1)

    def reset_episode_state(self) -> None:
        self.base_agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    def select(
        self,
        snapshot,
        *,
        preference_lambda: float,
        training: bool = True,
        epsilon: Optional[float] = None,
    ) -> AnchoredDecision:
        value = self._validate_lambda(preference_lambda)
        epsilon_value = self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        if value == 0.0:
            base_decision = self.base_agent.select(
                snapshot, training=training, epsilon=epsilon_value
            )
            decision = AnchoredDecision(
                base_decision=base_decision,
                preference_lambda=0.0,
                q_vectors=tuple((float(q), 0.0) for q in base_decision.q_values),
            )
            self.decision_count += 1
            self.preference_decisions["0.0"] += 1
            return decision

        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE candidate to score")
        features, base_q, immediate = self._cache_records(prepared.records)
        vectors = self._absolute_vectors(
            self.Q_local, features, base_q, immediate, value
        )
        merit = scalarized_merit(vectors, value)
        hierarchy = select_hierarchical_index(
            merit,
            prepared.mode_ids,
            self.base_agent.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        explored = bool(
            training and not liveness_forced and self.rng.random() < epsilon_value
        )
        if liveness_forced:
            selected_index = 0
            source = "exact_recovery_witness_guard"
        elif explored:
            live_modes = tuple(mode_id for mode_id, _ in hierarchy.mode_values)
            selected_mode = self.rng.choice(live_modes)
            indices = [
                index
                for index, mode_id in enumerate(prepared.mode_ids)
                if mode_id == selected_mode
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
            (ID_TO_MODE[mode_id], mode_value)
            for mode_id, mode_value in hierarchy.mode_values
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
        base.decision_log.append(
            {
                "decision_index": base.decision_count - 1,
                "episode_instance_id": snapshot.episode_instance_id,
                "decision_epoch": int(snapshot.decision_epoch),
                "exact_safe_candidate_count": prepared.exact_safe_candidate_count,
                "admissible_candidate_count": len(prepared.records),
                "liveness_restricted": prepared.liveness_restricted,
                "interface_rejection_count": prepared.interface_rejection_count,
                "verifier_fail_closed_rejection_count": int(
                    snapshot.audit.fail_closed_rejection_count
                ),
                "live_modes": tuple(name for name, _ in mode_pairs),
                "selected_key": candidate.key,
                "selected_mode": candidate.mode.value,
                "selected_action_type": candidate.action_type.value,
                "selection_source": source,
                "explored": explored,
                "handling_lambda": value,
                "selected_operational_q": float(vectors[selected_index, 0]),
                "selected_handling_q": float(vectors[selected_index, 1]),
                "selected_merit": float(merit[selected_index]),
                "liveness_forced": liveness_forced,
                "exact_rank_progress": exact_rank_progress,
            }
        )
        self.decision_count += 1
        self.preference_decisions[str(value)] += 1
        return AnchoredDecision(
            base_decision=base_decision,
            preference_lambda=value,
            q_vectors=tuple(
                (float(row[0]), float(row[1]))
                for row in vectors.detach().cpu().tolist()
            ),
            cached_chosen_feature=features[selected_index].clone(),
            cached_chosen_base_q=float(base_q[selected_index]),
            cached_chosen_immediate=float(immediate[selected_index]),
        )

    def observe_outcome(self, decision: AnchoredDecision, *, next_snapshot, done):
        if not isinstance(decision, AnchoredDecision):
            raise TypeError("anchored outcome requires AnchoredDecision")
        return self.base_agent.observe_outcome(
            decision.base_decision, next_snapshot=next_snapshot, done=done
        )

    def remember(
        self,
        decision: AnchoredDecision,
        *,
        operational_return: float,
        discounted_rehandles: float,
        raw_rehandles: int,
        duration: int,
        next_snapshot,
        done: bool,
        outcome_already_observed: bool = False,
    ) -> CachedAnchoredTransition:
        del discounted_rehandles  # raw physical handling is the declared Q_N target
        if outcome_already_observed:
            if done:
                next_prepared = None
            else:
                if next_snapshot is None:
                    raise ValueError("nonterminal anchored transition needs next snapshot")
                next_prepared, _ = self.base_agent._admissible_prepared(next_snapshot)
                if not next_prepared.records:
                    raise NoCertifiedViableAction("empty nonterminal anchored frontier")
        else:
            next_prepared = self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=done
            )
        if decision.cached_chosen_feature is None:
            chosen_features, chosen_base_q, chosen_immediate = self._cache_records(
                (decision.record,)
            )
            chosen_feature = chosen_features[0]
            chosen_q = float(chosen_base_q[0])
            chosen_cost = float(chosen_immediate[0])
        else:
            chosen_feature = decision.cached_chosen_feature
            chosen_q = float(decision.cached_chosen_base_q)
            chosen_cost = float(decision.cached_chosen_immediate)
        next_records = () if next_prepared is None else next_prepared.records
        next_features, next_base_q, next_immediate = self._cache_records(next_records)
        transition = CachedAnchoredTransition(
            chosen_feature=chosen_feature,
            chosen_base_q=chosen_q,
            chosen_immediate=chosen_cost,
            chosen_mode_id=decision.record.mode_id,
            operational_return=float(operational_return),
            raw_rehandles=int(raw_rehandles),
            duration=int(duration),
            next_features=next_features,
            next_base_q=next_base_q,
            next_immediate=next_immediate,
            next_mode_ids=tuple(record.mode_id for record in next_records),
            next_keys=tuple(record.key for record in next_records),
            done=bool(done),
            behavior_lambda=float(decision.preference_lambda),
        )
        self.replay.add(transition)
        self.transition_count += 1
        return transition

    def _td_batch(self, transitions: Sequence[CachedAnchoredTransition]):
        transitions = tuple(transitions)
        features = torch.stack([item.chosen_feature for item in transitions])
        base_q = torch.as_tensor([item.chosen_base_q for item in transitions])
        immediate = torch.as_tensor([item.chosen_immediate for item in transitions])
        preferences = torch.as_tensor([item.behavior_lambda for item in transitions])
        prediction = self._absolute_vectors(
            self.Q_local, features, base_q, immediate, preferences
        )
        next_values = torch.zeros_like(prediction)
        live = [item for item in transitions if not item.done]
        if live:
            flat_features = torch.cat([item.next_features for item in live], dim=0)
            flat_base_q = torch.cat([item.next_base_q for item in live], dim=0)
            flat_immediate = torch.cat([item.next_immediate for item in live], dim=0)
            flat_preferences = torch.cat(
                [
                    torch.full((item.next_features.shape[0],), item.behavior_lambda)
                    for item in live
                ]
            )
            with torch.no_grad():
                online = self._absolute_vectors(
                    self.Q_local,
                    flat_features,
                    flat_base_q,
                    flat_immediate,
                    flat_preferences,
                )
                target = self._absolute_vectors(
                    self.Q_target,
                    flat_features,
                    flat_base_q,
                    flat_immediate,
                    flat_preferences,
                )
            offset = 0
            live_index = 0
            for batch_index, item in enumerate(transitions):
                if item.done:
                    continue
                count = item.next_features.shape[0]
                preference = item.behavior_lambda
                selection = select_hierarchical_index(
                    scalarized_merit(online[offset : offset + count], preference),
                    item.next_mode_ids,
                    self.base_agent.within_temperatures,
                    candidate_keys=item.next_keys,
                )
                next_values[batch_index] = target[
                    offset + selection.selected_index
                ]
                offset += count
                live_index += 1
            if live_index != len(live):
                raise RuntimeError("anchored next-frontier batching drifted")
        operational = torch.as_tensor(
            [item.operational_return for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        raw_cost = torch.as_tensor(
            [item.raw_rehandles for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        duration = torch.as_tensor(
            [item.duration for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        done = torch.as_tensor(
            [item.done for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        continuation = (~done).to(prediction.dtype)
        targets = torch.stack(
            (
                self.config.reward_scale * operational
                + torch.pow(
                    prediction.new_tensor(self.config.gamma_op), duration
                )
                * continuation
                * next_values[:, 0],
                raw_cost + continuation * next_values[:, 1],
            ),
            dim=-1,
        )
        return prediction, targets.detach()

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        batch = self.replay.sample_mode_balanced(self.config.batch_size, self.rng)
        prediction, target = self._td_batch(batch)
        op_loss = F.smooth_l1_loss(
            prediction[:, 0], target[:, 0], beta=self.config.huber_delta
        )
        handling_loss = F.smooth_l1_loss(
            prediction[:, 1], target[:, 1], beta=self.config.huber_delta
        )
        loss = (
            self.config.operational_loss_weight * op_loss
            + self.config.handling_loss_weight * handling_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())
            self.Q_target.requires_grad_(False).eval()
            self.target_updates += 1
        value = float(loss.detach().item())
        self.loss_history.append(value)
        self.operational_loss_history.append(float(op_loss.detach().item()))
        self.handling_loss_history.append(float(handling_loss.detach().item()))
        return value

    def checkpoint(
        self,
        *,
        base_checkpoint_sha256: str,
        base_policy_digest: str,
        source_cost_sha256: str,
        include_replay: bool,
        **metadata,
    ) -> dict:
        state = {
            "Q_local": self.Q_local.state_dict(),
            "Q_target": self.Q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "rng_state": self.rng.getstate(),
            "transition_count": self.transition_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "loss_history": tuple(self.loss_history),
            "operational_loss_history": tuple(self.operational_loss_history),
            "handling_loss_history": tuple(self.handling_loss_history),
            "preference_decisions": dict(self.preference_decisions),
        }
        if include_replay:
            state["replay"] = self.replay.state_dict()
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "config": self.config.to_dict(),
            "base_checkpoint_sha256": str(base_checkpoint_sha256),
            "base_policy_digest": str(base_policy_digest),
            "source_cost_sha256": str(source_cost_sha256),
            "lambda_zero_direct_vcg_v1_1_delegation": True,
            "positive_lambda_initial_policy_equals_nested_vcg": True,
            "operational_anchor_frozen": True,
            "handling_objective": "undiscounted_physical_rehandles",
            "agent_state": state,
        }
        overlap = set(payload).intersection(metadata)
        if overlap:
            raise ValueError(f"reserved anchored checkpoint metadata: {sorted(overlap)}")
        payload.update(metadata)
        return payload

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
        resumable: bool,
    ) -> "AnchoredPreferenceAgent":
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "base_checkpoint_sha256": expected_base_checkpoint_sha256,
            "base_policy_digest": expected_base_policy_digest,
            "source_cost_sha256": expected_source_cost_sha256,
            "lambda_zero_direct_vcg_v1_1_delegation": True,
            "operational_anchor_frozen": True,
        }
        mismatch = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatch:
            raise ValueError(f"incompatible anchored checkpoint: {mismatch}")
        config = AnchoredPreferenceConfig.from_dict(payload["config"])
        agent = cls(
            base_agent,
            config=config,
            seed=seed,
            warm_start_cost=None,
            epsilon=float(payload["agent_state"].get("epsilon", 0.0)),
        )
        state = payload["agent_state"]
        agent.Q_local.load_state_dict(state["Q_local"])
        agent.Q_target.load_state_dict(state["Q_target"])
        agent.Q_target.requires_grad_(False).eval()
        agent.transition_count = int(state.get("transition_count", 0))
        agent.decision_count = int(state.get("decision_count", 0))
        agent.gradient_steps = int(state.get("gradient_steps", 0))
        agent.target_updates = int(state.get("target_updates", 0))
        agent.loss_history = list(state.get("loss_history", ()))
        agent.operational_loss_history = list(
            state.get("operational_loss_history", ())
        )
        agent.handling_loss_history = list(state.get("handling_loss_history", ()))
        agent.preference_decisions = Counter(state.get("preference_decisions", {}))
        if resumable:
            for name in ("optimizer", "rng_state", "replay"):
                if name not in state:
                    raise ValueError(f"resumable anchored checkpoint lacks {name}")
            agent.optimizer.load_state_dict(state["optimizer"])
            agent.rng.setstate(state["rng_state"])
            agent.replay.load_state_dict(state["replay"])
        return agent

    def audit(self, *, include_decisions: bool = False) -> dict:
        del include_decisions
        return {
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "lambda_zero_direct_vcg_v1_1_delegation": True,
            "operational_anchor_frozen": True,
            "handling_objective": "undiscounted_physical_rehandles",
            "decision_count": self.decision_count,
            "transition_count": self.transition_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "replay_size": len(self.replay),
            "preference_decisions": dict(self.preference_decisions),
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "last_operational_loss": (
                self.operational_loss_history[-1]
                if self.operational_loss_history
                else None
            ),
            "last_handling_loss": (
                self.handling_loss_history[-1]
                if self.handling_loss_history
                else None
            ),
            "base_audit": self.base_agent.audit(include_decisions=False),
        }


__all__ = [
    "AnchoredDecision",
    "AnchoredPreferenceAgent",
    "AnchoredPreferenceConfig",
    "AnchoredResidualNetwork",
    "BACKUP_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CONTROLLER_ARCHITECTURE",
    "CachedAnchoredTransition",
    "NETWORK_ARCHITECTURE",
]
