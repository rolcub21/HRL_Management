"""Preference-conditioned vector SMDP critic behind exact VCG certification.

The exact verifier and the recovery-witness liveness guard remain authoritative.
This module only ranks the candidates that survive those two layers.  Each
candidate receives two consequences under one deployment preference:

``Q_op``
    Expected discounted operational return, in ``reward_scale`` units.

``Q_N``
    Expected discounted physical storage-to-storage rehandles.

The scalar merit is ``Q_op - preference_lambda * Q_N``.  Consequently,
``preference_lambda`` is measured in *scaled operational-value units per
discounted rehandle*.  Raw, undiscounted rehandles are retained in replay for
reporting but do not enter the Bellman target.

The yard/candidate representation is preference independent.  Lambda is
appended only after the current-state, counterfactual-successor, delta, and
action embeddings have been formed.  Setting ``condition_on_preference=False``
therefore gives the intended unconditioned-vector diagnostic without changing
the exact frontier, representation, selector, or training data contract.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from PSLAP.viability import RecoveryState
from PSLAP.viability_candidates import (
    EXACT_VERIFIER_AUTHORITY,
    FAIL_CLOSED_CERTIFICATION_CONTRACT,
    VIABILITY_CANDIDATE_INTERFACE,
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityMode,
)
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    YardGraphEncoder,
    encode_recovery_state,
    pad_yard_graphs,
)
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ACTION_FEATURE_NAMES,
    ID_TO_MODE,
    MODE_NAMES,
    NoCertifiedViableAction,
    PreparedViabilityGraphSnapshot,
    RecoveryWitnessGuard,
    RecoveryWitnessMismatch,
    ViabilityGraphCandidateRecord,
    prepare_viability_snapshot,
    regularized_mode_values,
)


PREFERENCE_CONDITIONED_CHECKPOINT_SCHEMA_VERSION = 1
PREFERENCE_CONDITIONED_CONTROLLER_ARCHITECTURE = (
    "exact_viability_preference_conditioned_vector_smdp_q_v1"
)
PREFERENCE_CONDITIONED_NETWORK_ARCHITECTURE = (
    "preference_independent_counterfactual_graph_lambda_vector_q_v1"
)
PREFERENCE_CONDITIONED_REPLAY_VERSION = (
    "safe_counterfactual_vector_smdp_preference_relabel_replay_v1"
)
PREFERENCE_CONDITIONED_BACKUP_VERSION = (
    "deterministic_hierarchical_double_q_shared_vector_continuation_v1"
)
PREFERENCE_CONDITIONED_POLICY_VERSION = (
    "lambda_scalarized_mode_logmeanexp_map_candidate_map_v1"
)


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def normalize_preference_lambda(
    preference_lambda: Tensor | Sequence[float] | float,
    lambda_max: float,
    *,
    count: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> Tensor:
    """Validate a deployment preference and map ``[0, lambda_max]`` to [0, 1]."""

    lambda_max = _positive_finite(lambda_max, "lambda_max")
    values = torch.as_tensor(
        preference_lambda,
        dtype=dtype or torch.float32,
        device=device,
    )
    if values.ndim > 1:
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        else:
            raise ValueError("preference_lambda must be scalar or one-dimensional")
    if count is not None:
        if isinstance(count, bool) or int(count) <= 0:
            raise ValueError("count must be a positive integer")
        count = int(count)
        if values.ndim == 0:
            values = values.expand(count)
        elif values.shape != (count,):
            raise ValueError(f"preference_lambda must contain {count} values")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("preference_lambda must be finite")
    tolerance = 1.0e-7 * max(lambda_max, 1.0)
    if bool((values < -tolerance).any()) or bool(
        (values > lambda_max + tolerance).any()
    ):
        raise ValueError(f"preference_lambda must lie in [0, {lambda_max}]")
    return values.clamp(0.0, lambda_max) / lambda_max


def scalarized_merit(
    q_vectors: Tensor | Sequence[Sequence[float]],
    preference_lambda: Tensor | Sequence[float] | float,
) -> Tensor:
    """Return ``Q_op - lambda Q_N`` without changing either vector head."""

    vectors = torch.as_tensor(q_vectors)
    if not vectors.is_floating_point():
        vectors = vectors.to(dtype=torch.float32)
    if vectors.ndim < 1 or vectors.shape[-1] != 2:
        raise ValueError("q_vectors must have final dimension two")
    preferences = torch.as_tensor(
        preference_lambda, dtype=vectors.dtype, device=vectors.device
    )
    target_shape = vectors.shape[:-1]
    if preferences.ndim == 0:
        preferences = preferences.expand(target_shape)
    elif preferences.shape != target_shape:
        raise ValueError("preference_lambda must be scalar or align with q_vectors")
    if not bool(torch.isfinite(preferences).all()) or bool((preferences < 0.0).any()):
        raise ValueError("preference_lambda must be finite and non-negative")
    return vectors[..., 0] - preferences * vectors[..., 1]


@dataclass(frozen=True)
class HierarchicalSelection:
    """Pure deterministic mode/candidate selection result."""

    selected_index: int
    selected_mode_id: int
    mode_values: tuple[tuple[int, float], ...]


def select_hierarchical_index(
    merits: Tensor | Sequence[float],
    mode_ids: Tensor | Sequence[int],
    within_temperatures: Tensor | Sequence[float],
    *,
    candidate_keys: Optional[Sequence[str]] = None,
) -> HierarchicalSelection:
    """Apply deployment's normalized-mode then candidate MAP rule.

    Mode ties use the smallest stable mode ID.  Candidate ties use the
    lexicographically smallest supplied candidate key, or the smallest index
    when keys are omitted.
    """

    merit_tensor = torch.as_tensor(merits)
    if not merit_tensor.is_floating_point():
        merit_tensor = merit_tensor.to(dtype=torch.float32)
    if merit_tensor.ndim != 1 or merit_tensor.numel() == 0:
        raise ValueError("merits must be a nonempty one-dimensional tensor")
    if not bool(torch.isfinite(merit_tensor).all()):
        raise ValueError("merits must be finite")
    modes = torch.as_tensor(mode_ids, dtype=torch.long, device=merit_tensor.device)
    if modes.shape != merit_tensor.shape:
        raise ValueError("mode_ids must align with merits")
    if candidate_keys is not None:
        candidate_keys = tuple(str(key) for key in candidate_keys)
        if len(candidate_keys) != merit_tensor.numel():
            raise ValueError("candidate_keys must align with merits")

    values, live_modes = regularized_mode_values(
        merit_tensor, modes, within_temperatures
    )
    detached_values = values.detach().cpu().tolist()
    maximum_mode_value = max(float(value) for value in detached_values)
    tied_mode_positions = [
        index
        for index, value in enumerate(detached_values)
        if float(value) == maximum_mode_value
    ]
    mode_position = min(
        tied_mode_positions,
        key=lambda index: int(live_modes[index].item()),
    )
    selected_mode_id = int(live_modes[mode_position].item())
    mode_candidate_indices = [
        index
        for index, mode_id in enumerate(modes.detach().cpu().tolist())
        if int(mode_id) == selected_mode_id
    ]
    merit_values = merit_tensor.detach().cpu().tolist()
    maximum_candidate_merit = max(
        float(merit_values[index]) for index in mode_candidate_indices
    )
    tied_candidate_indices = [
        index
        for index in mode_candidate_indices
        if float(merit_values[index]) == maximum_candidate_merit
    ]
    if candidate_keys is None:
        selected_index = min(tied_candidate_indices)
    else:
        selected_index = min(
            tied_candidate_indices,
            key=lambda index: (candidate_keys[index], index),
        )
    return HierarchicalSelection(
        selected_index=int(selected_index),
        selected_mode_id=selected_mode_id,
        mode_values=tuple(
            (int(mode.item()), float(value.item()))
            for value, mode in zip(values, live_modes)
        ),
    )


# Descriptive alias retained for callers that prefer the full policy name.
deterministic_mode_candidate_selection = select_hierarchical_index


def vector_smdp_target(
    operational_returns: Tensor | Sequence[float],
    discounted_rehandles: Tensor | Sequence[float],
    durations: Tensor | Sequence[int],
    dones: Tensor | Sequence[bool],
    next_vectors: Tensor | Sequence[Sequence[float]],
    *,
    gamma: float,
    reward_scale: float = 1.0,
) -> Tensor:
    """Build common-discount vector targets for a batch of complete macros.

    Both intra-macro inputs must already use primitive-time ``gamma``.  The
    first component is converted to scaled operational-value units; handling
    stays in discounted physical-rehandle units.  ``gamma=1`` is valid for a
    finite-horizon undiscounted experiment.
    """

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    reward_scale = _positive_finite(reward_scale, "reward_scale")
    next_tensor = torch.as_tensor(next_vectors)
    if not next_tensor.is_floating_point():
        next_tensor = next_tensor.to(dtype=torch.float32)
    if next_tensor.ndim != 2 or next_tensor.shape[-1] != 2:
        raise ValueError("next_vectors must have shape [batch, 2]")
    dtype = next_tensor.dtype
    device = next_tensor.device
    operational = torch.as_tensor(operational_returns, dtype=dtype, device=device)
    handling = torch.as_tensor(discounted_rehandles, dtype=dtype, device=device)
    duration_tensor = torch.as_tensor(durations, device=device)
    done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device)
    batch_shape = (next_tensor.shape[0],)
    if not (
        operational.shape
        == handling.shape
        == duration_tensor.shape
        == done_tensor.shape
        == batch_shape
    ):
        raise ValueError("all vector SMDP target inputs must share batch shape")
    if not bool(torch.isfinite(operational).all()):
        raise ValueError("operational_returns must be finite")
    if not bool(torch.isfinite(handling).all()) or bool((handling < 0.0).any()):
        raise ValueError("discounted_rehandles must be finite and non-negative")
    if bool((duration_tensor < 1).any()):
        raise ValueError("every macro duration must be at least one transition")
    discount = torch.pow(next_tensor.new_tensor(gamma), duration_tensor.to(dtype=dtype))
    continuation = (discount * (~done_tensor).to(dtype)).unsqueeze(-1) * next_tensor
    immediate = torch.stack((reward_scale * operational, handling), dim=-1)
    return immediate + continuation


# Plural alias for discoverability alongside batched target terminology.
preference_conditioned_vector_smdp_targets = vector_smdp_target


@dataclass(frozen=True)
class PreferenceConditionedVectorConfig:
    graph_hidden_dim: int = 64
    graph_embedding_dim: int = 64
    message_passing_steps: int = 3
    action_embedding_dim: int = 32
    candidate_hidden_dim: int = 128
    consequence_hidden_dim: int = 128
    tau_accept: float = 0.1
    tau_recover: float = 0.1
    tau_defer: float = 0.1
    timing_scale: float = 100.0
    gamma: float = 0.99
    reward_scale: float = 0.01
    lambda_max: float = 0.2
    condition_on_preference: bool = True
    preference_relabels: int = 4
    learning_rate: float = 5.0e-5
    batch_size: int = 128
    replay_capacity: int = 20_000
    update_every: int = 1
    target_update_every: int = 200
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    handling_loss_weight: float = 1.0
    max_nonprogress_recovery_decisions: int = 2
    force_recovery_witness_when_due: bool = False

    def __post_init__(self) -> None:
        for name in (
            "graph_hidden_dim",
            "graph_embedding_dim",
            "action_embedding_dim",
            "candidate_hidden_dim",
            "consequence_hidden_dim",
            "preference_relabels",
            "batch_size",
            "replay_capacity",
            "update_every",
            "target_update_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.message_passing_steps, bool)
            or not isinstance(self.message_passing_steps, int)
            or self.message_passing_steps < 0
        ):
            raise ValueError("message_passing_steps must be non-negative")
        if (
            isinstance(self.max_nonprogress_recovery_decisions, bool)
            or not isinstance(self.max_nonprogress_recovery_decisions, int)
            or self.max_nonprogress_recovery_decisions < 0
        ):
            raise ValueError("max_nonprogress_recovery_decisions must be non-negative")
        if not isinstance(self.condition_on_preference, bool):
            raise ValueError("condition_on_preference must be boolean")
        if not isinstance(self.force_recovery_witness_when_due, bool):
            raise ValueError("force_recovery_witness_when_due must be boolean")
        for name in (
            "tau_accept",
            "tau_recover",
            "tau_defer",
            "timing_scale",
            "reward_scale",
            "lambda_max",
            "learning_rate",
            "grad_clip",
            "huber_delta",
            "handling_loss_weight",
        ):
            _positive_finite(getattr(self, name), name)
        if not math.isfinite(float(self.gamma)) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")

    @property
    def within_temperatures(self) -> tuple[float, float, float]:
        return (self.tau_accept, self.tau_recover, self.tau_defer)

    def normalize_preference_lambda(
        self,
        preference_lambda: Tensor | Sequence[float] | float,
        *,
        count: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device | str] = None,
    ) -> Tensor:
        return normalize_preference_lambda(
            preference_lambda,
            self.lambda_max,
            count=count,
            dtype=dtype,
            device=device,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "PreferenceConditionedVectorConfig":
        allowed = cls.__dataclass_fields__
        return cls(**{name: values[name] for name in allowed if name in values})


class PreferenceConditionedVectorQNetwork(nn.Module):
    """Shared graph representation with two lambda-conditioned consequences."""

    def __init__(self, config: PreferenceConditionedVectorConfig) -> None:
        super().__init__()
        self.config = config
        self.graph_encoder = YardGraphEncoder(
            input_dim=NODE_FEATURE_DIM,
            hidden_dim=config.graph_hidden_dim,
            output_dim=config.graph_embedding_dim,
            message_passing_steps=config.message_passing_steps,
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, config.action_embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(config.action_embedding_dim),
        )
        candidate_input_dim = (
            3 * config.graph_embedding_dim + config.action_embedding_dim
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input_dim, config.candidate_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.candidate_hidden_dim),
        )
        self.conditioned_trunk = nn.Sequential(
            nn.Linear(config.candidate_hidden_dim + 1, config.consequence_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.consequence_hidden_dim, config.consequence_hidden_dim),
            nn.SiLU(),
        )
        self.operational_head = nn.Linear(config.consequence_hidden_dim, 1)
        self.handling_head = nn.Linear(config.consequence_hidden_dim, 1)

    def encode_candidates(
        self,
        current_states: Sequence[RecoveryState],
        successor_states: Sequence[RecoveryState],
        action_features: Tensor | Sequence[Sequence[float]],
    ) -> Tensor:
        """Create preference-independent ``h(s,c)`` representations."""

        current_states = tuple(current_states)
        successor_states = tuple(successor_states)
        if not current_states:
            raise ValueError("vector Q network requires at least one candidate")
        if len(current_states) != len(successor_states):
            raise ValueError("current and successor state counts must match")
        if not all(
            isinstance(state, RecoveryState)
            for state in current_states + successor_states
        ):
            raise TypeError("every candidate state must be a RecoveryState")
        parameter = next(self.parameters())
        feature_tensor = torch.as_tensor(
            action_features, dtype=parameter.dtype, device=parameter.device
        )
        if feature_tensor.ndim == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        if feature_tensor.shape != (len(current_states), ACTION_FEATURE_DIM):
            raise ValueError(
                "action features must have shape "
                f"[{len(current_states)}, {ACTION_FEATURE_DIM}]"
            )
        graphs = tuple(
            encode_recovery_state(state, timing_scale=self.config.timing_scale)
            for state in current_states + successor_states
        )
        embeddings = self.graph_encoder(pad_yard_graphs(graphs))
        count = len(current_states)
        current = embeddings[:count]
        successor = embeddings[count:]
        action = self.action_encoder(feature_tensor)
        return self.candidate_encoder(
            torch.cat((current, successor, successor - current, action), dim=-1)
        )

    def forward(
        self,
        current_states: Sequence[RecoveryState],
        successor_states: Sequence[RecoveryState],
        action_features: Tensor | Sequence[Sequence[float]],
        preference_lambda: Tensor | Sequence[float] | float,
    ) -> Tensor:
        representation = self.encode_candidates(
            current_states, successor_states, action_features
        )
        normalized = self.config.normalize_preference_lambda(
            preference_lambda,
            count=representation.shape[0],
            dtype=representation.dtype,
            device=representation.device,
        )
        if not self.config.condition_on_preference:
            normalized = torch.zeros_like(normalized)
        conditioned = self.conditioned_trunk(
            torch.cat((representation, normalized.unsqueeze(-1)), dim=-1)
        )
        # A physical-rehandle consequence cannot be negative.  Enforcing that
        # support prevents an immature critic from rewarding actions merely
        # because its cost head crossed below zero.
        return torch.cat(
            (
                self.operational_head(conditioned),
                F.softplus(self.handling_head(conditioned)),
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class PreferenceConditionedVectorDecision:
    candidate: ViabilityActionCandidate
    record: ViabilityGraphCandidateRecord
    prepared_snapshot: PreparedViabilityGraphSnapshot
    preference_lambda: float
    q_vectors: tuple[tuple[float, float], ...]
    merits: tuple[float, ...]
    mode_values: tuple[tuple[str, float], ...]
    explored: bool
    selection_source: str
    liveness_forced: bool = False
    exact_rank_progress: bool = False

    @property
    def option(self):
        return self.candidate.option


@dataclass(frozen=True)
class PreferenceConditionedVectorTransition:
    chosen: ViabilityGraphCandidateRecord
    operational_return: float
    discounted_rehandles: float
    raw_rehandles: int
    duration: int
    next_candidates: tuple[ViabilityGraphCandidateRecord, ...]
    done: bool
    behavior_preference_lambda: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.operational_return)):
            raise ValueError("operational_return must be finite")
        if (
            not math.isfinite(float(self.discounted_rehandles))
            or float(self.discounted_rehandles) < 0.0
        ):
            raise ValueError("discounted_rehandles must be finite and non-negative")
        if (
            isinstance(self.raw_rehandles, bool)
            or not isinstance(self.raw_rehandles, int)
            or self.raw_rehandles < 0
        ):
            raise ValueError("raw_rehandles must be a non-negative integer")
        if (
            isinstance(self.duration, bool)
            or not isinstance(self.duration, int)
            or self.duration < 1
        ):
            raise ValueError("duration must be a positive integer")
        if self.done and self.next_candidates:
            raise ValueError("terminal transition cannot retain next candidates")
        if not self.done and not self.next_candidates:
            raise ValueError("nonterminal transition needs a SAFE next frontier")
        if (
            not math.isfinite(float(self.behavior_preference_lambda))
            or float(self.behavior_preference_lambda) < 0.0
        ):
            raise ValueError(
                "behavior_preference_lambda must be finite and non-negative"
            )

    def numeric_copy(self) -> "PreferenceConditionedVectorTransition":
        return PreferenceConditionedVectorTransition(
            chosen=self.chosen,
            operational_return=float(self.operational_return),
            discounted_rehandles=float(self.discounted_rehandles),
            raw_rehandles=int(self.raw_rehandles),
            duration=int(self.duration),
            next_candidates=tuple(self.next_candidates),
            done=bool(self.done),
            behavior_preference_lambda=float(self.behavior_preference_lambda),
        )


class PreferenceConditionedVectorReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.memory = deque(maxlen=self.capacity)

    def add(self, transition: PreferenceConditionedVectorTransition) -> None:
        if not isinstance(transition, PreferenceConditionedVectorTransition):
            raise TypeError(
                "replay accepts PreferenceConditionedVectorTransition instances"
            )
        self.memory.append(transition.numeric_copy())

    def sample_mode_balanced(self, batch_size: int, rng: random.Random):
        batch_size = int(batch_size)
        memory = list(self.memory)
        if batch_size <= 0 or batch_size > len(memory):
            raise ValueError("invalid replay batch size")
        buckets: dict[int, list[int]] = {}
        for index, transition in enumerate(memory):
            buckets.setdefault(transition.chosen.mode_id, []).append(index)
        live_modes = sorted(buckets)
        base, remainder = divmod(batch_size, len(live_modes))
        selected: list[int] = []
        selected_set: set[int] = set()
        for position, mode_id in enumerate(live_modes):
            requested = base + int(position < remainder)
            chosen = rng.sample(buckets[mode_id], min(requested, len(buckets[mode_id])))
            selected.extend(chosen)
            selected_set.update(chosen)
        missing = batch_size - len(selected)
        if missing:
            available = [
                index for index in range(len(memory)) if index not in selected_set
            ]
            selected.extend(rng.sample(available, missing))
        rng.shuffle(selected)
        return [memory[index] for index in selected]

    def mode_counts(self) -> dict[int, int]:
        return dict(Counter(item.chosen.mode_id for item in self.memory))

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "memory": list(self.memory)}

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch")
        self.memory = deque(
            (item.numeric_copy() for item in state["memory"]),
            maxlen=self.capacity,
        )

    def __len__(self) -> int:
        return len(self.memory)


class PreferenceConditionedVectorAgent:
    """Exact-safe vector Double-DQN supporting a deployment lambda family."""

    def __init__(
        self,
        *,
        config: Optional[PreferenceConditionedVectorConfig] = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
        epsilon: float = 0.9,
    ) -> None:
        self.config = config or PreferenceConditionedVectorConfig()
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.epsilon = self._validate_epsilon(epsilon)
        self.rng = random.Random(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = PreferenceConditionedVectorQNetwork(self.config).to(
                self.device
            )
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=self.config.learning_rate
        )
        self.replay = PreferenceConditionedVectorReplayBuffer(
            self.config.replay_capacity
        )
        self.recovery_witness_guard = RecoveryWitnessGuard(
            max_nonprogress_recovery_decisions=(
                self.config.max_nonprogress_recovery_decisions
            ),
            force_when_due=self.config.force_recovery_witness_when_due,
        )
        self.transition_count = 0
        self.decision_count = 0
        self.gradient_steps = 0
        self.target_updates = 0
        self.mode_decisions: Counter[str] = Counter()
        self.action_decisions: Counter[str] = Counter()
        self.selection_sources: Counter[str] = Counter()
        self.exact_rejections_seen = 0
        self.interface_rejections = 0
        self.safe_candidates_scored = 0
        self.loss_history: list[float] = []
        self.operational_loss_history: list[float] = []
        self.handling_loss_history: list[float] = []
        self.decision_log: list[dict] = []

    @staticmethod
    def _validate_epsilon(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("epsilon must be finite and in [0, 1]")
        return value

    def _validate_preference(self, preference_lambda: float) -> float:
        value = float(preference_lambda)
        self.config.normalize_preference_lambda(value)
        return value

    @property
    def within_temperatures(self) -> Tensor:
        return torch.as_tensor(
            self.config.within_temperatures,
            dtype=torch.float32,
            device=self.device,
        )

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = self._validate_epsilon(epsilon)

    @staticmethod
    def prepare_snapshot(
        snapshot: ViabilityCandidateSnapshot,
    ) -> PreparedViabilityGraphSnapshot:
        return prepare_viability_snapshot(snapshot)

    def _admissible_prepared(
        self, snapshot: ViabilityCandidateSnapshot
    ) -> tuple[PreparedViabilityGraphSnapshot, bool]:
        initial = prepare_viability_snapshot(
            snapshot,
            guard_context=self.recovery_witness_guard.feature_context(
                snapshot, forced_frontier=False
            ),
        )
        if not initial.records:
            return initial, False
        forced_index = self.recovery_witness_guard.forced_prepared_index(
            snapshot, initial
        )
        forced = forced_index is not None
        prepared = prepare_viability_snapshot(
            snapshot,
            guard_context=self.recovery_witness_guard.feature_context(
                snapshot, forced_frontier=forced
            ),
        )
        if forced_index is None:
            return prepared, False
        source_index = initial.source_indices[forced_index]
        try:
            restricted_index = prepared.source_indices.index(source_index)
        except ValueError as error:
            raise RecoveryWitnessMismatch(
                "forced witness candidate vanished while preparing frontier"
            ) from error
        return (
            PreparedViabilityGraphSnapshot(
                episode_instance_id=prepared.episode_instance_id,
                decision_epoch=prepared.decision_epoch,
                records=(prepared.records[restricted_index],),
                source_indices=(source_index,),
                interface_rejection_count=prepared.interface_rejection_count,
                exact_safe_candidate_count=prepared.exact_safe_candidate_count,
                liveness_restricted=True,
            ),
            True,
        )

    @staticmethod
    def _record_tensors(records: Sequence[ViabilityGraphCandidateRecord]):
        records = tuple(records)
        return (
            tuple(record.current_state for record in records),
            tuple(record.successor_state for record in records),
            tuple(record.action_features for record in records),
        )

    def _score_records(
        self,
        records: Sequence[ViabilityGraphCandidateRecord],
        preference_lambda: Tensor | Sequence[float] | float,
        *,
        network: Optional[PreferenceConditionedVectorQNetwork] = None,
        grad: bool = False,
    ) -> Tensor:
        records = tuple(records)
        if not records:
            raise NoCertifiedViableAction("no exact SAFE candidate to score")
        network = network or self.Q_local
        current, successor, features = self._record_tensors(records)
        if grad:
            return network(current, successor, features, preference_lambda)
        with torch.no_grad():
            return network(current, successor, features, preference_lambda)

    def score_snapshot(
        self,
        snapshot: ViabilityCandidateSnapshot,
        *,
        preference_lambda: float,
    ) -> Tensor:
        value = self._validate_preference(preference_lambda)
        prepared = prepare_viability_snapshot(snapshot)
        return self._score_records(prepared.records, value).detach().cpu()

    def select(
        self,
        snapshot: ViabilityCandidateSnapshot,
        *,
        preference_lambda: float,
        training: bool = True,
        epsilon: Optional[float] = None,
    ) -> PreferenceConditionedVectorDecision:
        preference_lambda = self._validate_preference(preference_lambda)
        prepared, liveness_forced = self._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "exact verifier exposed no SAFE executable macro"
            )
        vectors = self._score_records(prepared.records, preference_lambda)
        merits = scalarized_merit(vectors, preference_lambda)
        modes = torch.as_tensor(
            prepared.mode_ids, dtype=torch.long, device=merits.device
        )
        hierarchy = select_hierarchical_index(
            merits,
            modes,
            self.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        epsilon_value = (
            self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        )
        explored = bool(
            not liveness_forced and training and self.rng.random() < epsilon_value
        )
        if liveness_forced:
            selected_index = 0
            selection_source = "exact_recovery_witness_guard"
        elif explored:
            live_modes = tuple(mode_id for mode_id, _ in hierarchy.mode_values)
            selected_mode = self.rng.choice(live_modes)
            indices = [
                index
                for index, mode_id in enumerate(prepared.mode_ids)
                if mode_id == selected_mode
            ]
            selected_index = self.rng.choice(indices)
            selection_source = "safe_mode_balanced_epsilon"
        else:
            selected_index = hierarchy.selected_index
            selection_source = (
                "singleton_safe"
                if len(prepared.records) == 1
                else "lambda_scalarized_mode_map_candidate_map"
            )

        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        record = prepared.records[selected_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        mode_value_pairs = tuple(
            (ID_TO_MODE[mode_id], value) for mode_id, value in hierarchy.mode_values
        )
        decision = PreferenceConditionedVectorDecision(
            candidate=candidate,
            record=record,
            prepared_snapshot=prepared,
            preference_lambda=preference_lambda,
            q_vectors=tuple(
                (float(vector[0]), float(vector[1]))
                for vector in vectors.detach().cpu().tolist()
            ),
            merits=tuple(float(value) for value in merits.detach().cpu().tolist()),
            mode_values=mode_value_pairs,
            explored=explored,
            selection_source=selection_source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        self.decision_count += 1
        self.mode_decisions[candidate.mode.value] += 1
        self.action_decisions[candidate.action_type.value] += 1
        self.selection_sources[selection_source] += 1
        self.safe_candidates_scored += len(prepared.records)
        self.interface_rejections += prepared.interface_rejection_count
        self.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        self.decision_log.append(
            {
                "decision_index": self.decision_count - 1,
                "episode_instance_id": snapshot.episode_instance_id,
                "decision_epoch": int(snapshot.decision_epoch),
                "preference_lambda": preference_lambda,
                "exact_safe_candidate_count": prepared.exact_safe_candidate_count,
                "admissible_candidate_count": len(prepared.records),
                "liveness_restricted": prepared.liveness_restricted,
                "interface_rejection_count": prepared.interface_rejection_count,
                "verifier_fail_closed_rejection_count": int(
                    snapshot.audit.fail_closed_rejection_count
                ),
                "live_modes": tuple(name for name, _ in mode_value_pairs),
                "selected_key": candidate.key,
                "selected_mode": candidate.mode.value,
                "selected_action_type": candidate.action_type.value,
                "selection_source": selection_source,
                "explored": explored,
                "liveness_forced": liveness_forced,
                "exact_rank_progress": exact_rank_progress,
            }
        )
        return decision

    def select_option(self, snapshot: ViabilityCandidateSnapshot, **kwargs):
        return self.select(snapshot, **kwargs).option

    def reset_episode_state(self) -> None:
        self.recovery_witness_guard.reset()

    on_episode_reset = reset_episode_state

    def observe_outcome(
        self,
        decision: PreferenceConditionedVectorDecision,
        *,
        next_snapshot: Optional[ViabilityCandidateSnapshot],
        done: bool,
    ) -> Optional[PreparedViabilityGraphSnapshot]:
        if not isinstance(decision, PreferenceConditionedVectorDecision):
            raise TypeError(
                "observe_outcome requires a PreferenceConditionedVectorDecision"
            )
        done = bool(done)
        if not done and next_snapshot is None:
            raise ValueError("nonterminal outcome requires next_snapshot")
        self.recovery_witness_guard.observe(
            decision, next_snapshot=next_snapshot, done=done
        )
        if done:
            return None
        prepared, _ = self._admissible_prepared(next_snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "nonterminal next state has no liveness-admissible SAFE macro"
            )
        return prepared

    def remember(
        self,
        decision: PreferenceConditionedVectorDecision,
        *,
        operational_return: float,
        discounted_rehandles: float,
        raw_rehandles: int,
        duration: int,
        next_snapshot: Optional[ViabilityCandidateSnapshot],
        done: bool,
        outcome_already_observed: bool = False,
    ) -> PreferenceConditionedVectorTransition:
        if not isinstance(decision, PreferenceConditionedVectorDecision):
            raise TypeError("remember requires a PreferenceConditionedVectorDecision")
        preference = self._validate_preference(decision.preference_lambda)
        done = bool(done)
        if outcome_already_observed:
            if done:
                next_prepared = None
            else:
                if next_snapshot is None:
                    raise ValueError("nonterminal transition requires next_snapshot")
                next_prepared, _ = self._admissible_prepared(next_snapshot)
                if not next_prepared.records:
                    raise NoCertifiedViableAction(
                        "nonterminal next state has no liveness-admissible SAFE macro"
                    )
        else:
            next_prepared = self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=done
            )
        transition = PreferenceConditionedVectorTransition(
            chosen=decision.record,
            operational_return=float(operational_return),
            discounted_rehandles=float(discounted_rehandles),
            raw_rehandles=raw_rehandles,
            duration=int(duration),
            next_candidates=(
                () if next_prepared is None else tuple(next_prepared.records)
            ),
            done=done,
            behavior_preference_lambda=preference,
        )
        self.replay.add(transition)
        self.transition_count += 1
        return transition

    def _sample_relabel_preferences(
        self,
        transitions: Sequence[PreferenceConditionedVectorTransition],
    ) -> tuple[float, ...]:
        values = []
        for transition in transitions:
            # Retaining the behavior preference anchors every sample to an
            # actually generated continuation; the remaining values relabel it.
            values.append(float(transition.behavior_preference_lambda))
            for _ in range(1, self.config.preference_relabels):
                values.append(self.rng.random() * self.config.lambda_max)
        return tuple(values)

    def _td_batch(
        self,
        transitions: Sequence[PreferenceConditionedVectorTransition],
        *,
        preference_lambdas: Optional[Sequence[float]] = None,
    ) -> tuple[Tensor, Tensor]:
        transitions = tuple(transitions)
        if not transitions:
            raise ValueError("TD batch cannot be empty")
        repeats = self.config.preference_relabels
        expanded_transitions = tuple(
            transition for transition in transitions for _ in range(repeats)
        )
        if preference_lambdas is None:
            preferences = self._sample_relabel_preferences(transitions)
        else:
            preferences = tuple(float(value) for value in preference_lambdas)
            if len(preferences) != len(expanded_transitions):
                raise ValueError(
                    "preference_lambdas must contain batch * relabel values"
                )
            for value in preferences:
                self._validate_preference(value)
        chosen = tuple(item.chosen for item in expanded_transitions)
        prediction = self._score_records(
            chosen,
            preferences,
            network=self.Q_local,
            grad=True,
        )
        next_vectors = torch.zeros_like(prediction)
        with torch.no_grad():
            for index, (transition, preference) in enumerate(
                zip(expanded_transitions, preferences)
            ):
                if transition.done:
                    continue
                online = self._score_records(
                    transition.next_candidates,
                    preference,
                    network=self.Q_local,
                )
                target = self._score_records(
                    transition.next_candidates,
                    preference,
                    network=self.Q_target,
                )
                merits = scalarized_merit(online, preference)
                selection = select_hierarchical_index(
                    merits,
                    [record.mode_id for record in transition.next_candidates],
                    self.within_temperatures,
                    candidate_keys=tuple(
                        record.key for record in transition.next_candidates
                    ),
                )
                # One online-selected candidate supplies both target heads.
                next_vectors[index] = target[selection.selected_index]
        operational = torch.as_tensor(
            [item.operational_return for item in expanded_transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        handling = torch.as_tensor(
            [item.discounted_rehandles for item in expanded_transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        durations = torch.as_tensor(
            [item.duration for item in expanded_transitions],
            dtype=torch.long,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in expanded_transitions],
            dtype=torch.bool,
            device=self.device,
        )
        targets = vector_smdp_target(
            operational,
            handling,
            durations,
            dones,
            next_vectors,
            gamma=self.config.gamma,
            reward_scale=self.config.reward_scale,
        )
        return prediction, targets.detach()

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        transitions = self.replay.sample_mode_balanced(self.config.batch_size, self.rng)
        prediction, target = self._td_batch(transitions)
        operational_loss = F.smooth_l1_loss(
            prediction[:, 0], target[:, 0], beta=self.config.huber_delta
        )
        handling_loss = F.smooth_l1_loss(
            prediction[:, 1], target[:, 1], beta=self.config.huber_delta
        )
        loss = operational_loss + self.config.handling_loss_weight * handling_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())
            self.Q_target.requires_grad_(False)
            self.Q_target.eval()
            self.target_updates += 1
        total_value = float(loss.detach().item())
        self.loss_history.append(total_value)
        self.operational_loss_history.append(float(operational_loss.detach().item()))
        self.handling_loss_history.append(float(handling_loss.detach().item()))
        return total_value

    def checkpoint_metadata(self) -> dict:
        return {
            "controller_architecture": (PREFERENCE_CONDITIONED_CONTROLLER_ARCHITECTURE),
            "candidate_interface": VIABILITY_CANDIDATE_INTERFACE,
            "network_architecture": PREFERENCE_CONDITIONED_NETWORK_ARCHITECTURE,
            "replay_version": PREFERENCE_CONDITIONED_REPLAY_VERSION,
            "backup_version": PREFERENCE_CONDITIONED_BACKUP_VERSION,
            "policy_version": PREFERENCE_CONDITIONED_POLICY_VERSION,
            "modes": MODE_NAMES,
            "action_types": tuple(item.value for item in ViabilityActionType),
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "graph_node_feature_dim": NODE_FEATURE_DIM,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "exact_safe_mask_authoritative": True,
            "unsafe_unknown_fail_closed": True,
            "exact_verifier_authority": EXACT_VERIFIER_AUTHORITY,
            "certification_contract": FAIL_CLOSED_CERTIFICATION_CONTRACT,
            "counterfactual_successor_graph": True,
            "preference_independent_candidate_representation": True,
            "preference_conditioned": self.config.condition_on_preference,
            "lambda_max": self.config.lambda_max,
            "lambda_normalization": "lambda_divided_by_lambda_max_v1",
            "preference_relabels": self.config.preference_relabels,
            "vector_heads": ("scaled_operational_return", "discounted_rehandles"),
            "handling_head_nonnegative": True,
            "scalar_merit": "Q_op_minus_lambda_times_Q_N",
            "lambda_units": (
                "scaled_operational_value_per_discounted_physical_rehandle"
            ),
            "common_primitive_time_gamma": self.config.gamma,
            "reward_scale": self.config.reward_scale,
            "raw_rehandles_reporting_only": True,
            "double_q": True,
            "shared_selected_next_candidate_across_heads": True,
            "deployment_policy": ("mode_logmeanexp_argmax_then_candidate_merit_argmax"),
            "training_exploration": (
                "uniform_live_mode_then_uniform_certified_candidate"
            ),
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "behavior_cloning": False,
            "recovery_liveness_guard": (
                "bounded_nonprogress_then_retained_exact_dynamics_witness_v1"
            ),
            "max_nonprogress_recovery_decisions": (
                self.config.max_nonprogress_recovery_decisions
            ),
            "force_recovery_witness_when_due": (
                self.config.force_recovery_witness_when_due
            ),
            "bellman_frontier_matches_deployment_shield": True,
        }

    def checkpoint_state(self, *, include_replay: bool = True) -> dict:
        state = {
            "Q_local": self.Q_local.state_dict(),
            "Q_target": self.Q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "transition_count": self.transition_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "mode_decisions": dict(self.mode_decisions),
            "action_decisions": dict(self.action_decisions),
            "selection_sources": dict(self.selection_sources),
            "exact_rejections_seen": self.exact_rejections_seen,
            "interface_rejections": self.interface_rejections,
            "safe_candidates_scored": self.safe_candidates_scored,
            "loss_history": tuple(self.loss_history),
            "operational_loss_history": tuple(self.operational_loss_history),
            "handling_loss_history": tuple(self.handling_loss_history),
            "rng_state": self.rng.getstate(),
            "recovery_witness_guard": self.recovery_witness_guard.state_dict(),
        }
        if include_replay:
            state["replay"] = self.replay.state_dict()
        return state

    def load_checkpoint_state(self, state: dict, *, resumable: bool = False) -> None:
        self.Q_local.load_state_dict(state["Q_local"])
        self.Q_target.load_state_dict(state.get("Q_target", state["Q_local"]))
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.epsilon = self._validate_epsilon(state.get("epsilon", self.epsilon))
        self.transition_count = int(state.get("transition_count", 0))
        self.decision_count = int(state.get("decision_count", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.target_updates = int(state.get("target_updates", 0))
        self.mode_decisions = Counter(state.get("mode_decisions", {}))
        self.action_decisions = Counter(state.get("action_decisions", {}))
        self.selection_sources = Counter(state.get("selection_sources", {}))
        self.exact_rejections_seen = int(state.get("exact_rejections_seen", 0))
        self.interface_rejections = int(state.get("interface_rejections", 0))
        self.safe_candidates_scored = int(state.get("safe_candidates_scored", 0))
        self.loss_history = list(state.get("loss_history", ()))
        self.operational_loss_history = list(state.get("operational_loss_history", ()))
        self.handling_loss_history = list(state.get("handling_loss_history", ()))
        guard_state = state.get("recovery_witness_guard")
        if resumable and guard_state is not None:
            self.recovery_witness_guard.load_state_dict(guard_state)
        else:
            self.recovery_witness_guard.reset()
        if resumable:
            required = ("optimizer", "replay", "rng_state")
            missing = [name for name in required if name not in state]
            if missing:
                raise ValueError(f"resumable checkpoint is missing {tuple(missing)!r}")
            self.optimizer.load_state_dict(state["optimizer"])
            self.replay.load_state_dict(state["replay"])
            self.rng.setstate(state["rng_state"])

    def checkpoint(self, *, include_replay: bool = True, **metadata) -> dict:
        payload = {
            "checkpoint_schema_version": (
                PREFERENCE_CONDITIONED_CHECKPOINT_SCHEMA_VERSION
            ),
            **self.checkpoint_metadata(),
            "config": self.config.to_dict(),
            "agent_state": self.checkpoint_state(include_replay=include_replay),
        }
        collisions = sorted(set(payload).intersection(metadata))
        if collisions:
            raise ValueError(
                "checkpoint metadata cannot override reserved keys: " f"{collisions!r}"
            )
        payload.update(metadata)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "PreferenceConditionedVectorAgent":
        expected = {
            "checkpoint_schema_version": (
                PREFERENCE_CONDITIONED_CHECKPOINT_SCHEMA_VERSION
            ),
            "controller_architecture": (PREFERENCE_CONDITIONED_CONTROLLER_ARCHITECTURE),
            "candidate_interface": VIABILITY_CANDIDATE_INTERFACE,
            "network_architecture": PREFERENCE_CONDITIONED_NETWORK_ARCHITECTURE,
            "backup_version": PREFERENCE_CONDITIONED_BACKUP_VERSION,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "exact_safe_mask_authoritative": True,
            "baseline_teacher": False,
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatches:
            raise ValueError(
                "incompatible preference-conditioned checkpoint: " f"{mismatches!r}"
            )
        config = PreferenceConditionedVectorConfig.from_dict(payload["config"])
        if payload.get("preference_conditioned") != config.condition_on_preference:
            raise ValueError("checkpoint preference-conditioning flag mismatch")
        agent = cls(
            config=config,
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent

    def audit(self, *, include_decisions: bool = False) -> dict:
        audit = {
            **self.checkpoint_metadata(),
            "epsilon": self.epsilon,
            "transition_count": self.transition_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "replay_size": len(self.replay),
            "replay_mode_counts": {
                ID_TO_MODE[mode_id]: count
                for mode_id, count in self.replay.mode_counts().items()
            },
            "replay_raw_rehandles": sum(
                item.raw_rehandles for item in self.replay.memory
            ),
            "replay_discounted_rehandles": sum(
                item.discounted_rehandles for item in self.replay.memory
            ),
            "mode_decisions": dict(self.mode_decisions),
            "action_decisions": dict(self.action_decisions),
            "selection_sources": dict(self.selection_sources),
            "exact_verifier_rejections_seen": self.exact_rejections_seen,
            "interface_non_safe_rejections": self.interface_rejections,
            "safe_candidates_scored": self.safe_candidates_scored,
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "last_operational_loss": (
                self.operational_loss_history[-1]
                if self.operational_loss_history
                else None
            ),
            "last_handling_loss": (
                self.handling_loss_history[-1] if self.handling_loss_history else None
            ),
            "recovery_witness_guard_audit": (self.recovery_witness_guard.audit_dict()),
        }
        if include_decisions:
            audit["decisions"] = tuple(self.decision_log)
        return audit


__all__ = [
    "HierarchicalSelection",
    "PREFERENCE_CONDITIONED_BACKUP_VERSION",
    "PREFERENCE_CONDITIONED_CHECKPOINT_SCHEMA_VERSION",
    "PREFERENCE_CONDITIONED_CONTROLLER_ARCHITECTURE",
    "PREFERENCE_CONDITIONED_NETWORK_ARCHITECTURE",
    "PREFERENCE_CONDITIONED_POLICY_VERSION",
    "PREFERENCE_CONDITIONED_REPLAY_VERSION",
    "PreferenceConditionedVectorAgent",
    "PreferenceConditionedVectorConfig",
    "PreferenceConditionedVectorDecision",
    "PreferenceConditionedVectorQNetwork",
    "PreferenceConditionedVectorReplayBuffer",
    "PreferenceConditionedVectorTransition",
    "deterministic_mode_candidate_selection",
    "normalize_preference_lambda",
    "preference_conditioned_vector_smdp_targets",
    "scalarized_merit",
    "select_hierarchical_index",
    "vector_smdp_target",
]
