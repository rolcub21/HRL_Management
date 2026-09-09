"""Teacher-free viability-constrained graph SMDP controller.

The exact verifier owns the feasible action set.  This module never proposes
an action outside the ``SAFE`` frontier provided by
``PSLAP.viability_candidates`` and never replaces an exact certificate with a
learned feasibility score.  Learning starts only *after* certification: one
shared task-Q network ranks the certified Accept, Recover, and Defer macros by
encoding each macro's counterfactual :class:`RecoveryState` as a yard graph.

The continuation is common to every temporal mode.  It uses a uniform
reference distribution within every live mode and over the live modes, so
duplicating a mode's complete candidate representation does not change its
value.  Double-Q evaluation is the regularized analogue of Double-DQN: the
online network induces the nested regularized policy and the target network
evaluates that policy (including its KL regularizers).  If the networks are
equal, it reduces exactly to the nested cardinality-normalized log-mean-exp
operator.

There is deliberately no behavior cloning, baseline action query, heuristic
assignment score, or baseline teacher in this module.
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

from PSLAP.viability import RecoveryAction, RecoveryState, ViabilityStatus
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


VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION = 1
VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE = (
    "exact_viability_constrained_graph_smdp_q_v1"
)
VIABILITY_GRAPH_NETWORK_ARCHITECTURE = (
    "counterfactual_yard_graph_shared_action_conditional_q_v1"
)
VIABILITY_GRAPH_REPLAY_VERSION = (
    "safe_counterfactual_variable_candidate_smdp_replay_v1"
)
VIABILITY_GRAPH_BACKUP_VERSION = (
    "nested_cardinality_normalized_regularized_double_q_common_smdp_v1"
)
VIABILITY_GRAPH_POLICY_VERSION = (
    "regularized_mode_map_candidate_map_safe_mode_balanced_epsilon_v1"
)

MODE_NAMES = tuple(mode.value for mode in ViabilityMode)
MODE_TO_ID = {name: index for index, name in enumerate(MODE_NAMES)}
ID_TO_MODE = {index: name for name, index in MODE_TO_ID.items()}

ACTION_FEATURE_NAMES = (
    "mode_accept",
    "mode_recover",
    "mode_defer",
    "type_accept",
    "type_deliver",
    "type_reconfigure",
    "type_defer",
    "target_present",
    "source_present",
    "source_row_fraction",
    "source_column_fraction",
    "destination_present",
    "destination_row_fraction",
    "destination_column_fraction",
    "macro_steps_known",
    "macro_steps_per_yard_cell",
    "defer_horizon_known",
    "defer_horizon_per_yard_cell",
    "rank_before_known",
    "rank_before_per_storage_cell",
    "rank_after_known",
    "rank_after_per_storage_cell",
    "rank_delta_known",
    "rank_delta_per_storage_cell",
    "guard_active",
    "guard_nonprogress_fraction",
    "guard_remaining_witness_per_storage_cell",
    "guard_forced_frontier",
)
ACTION_FEATURE_DIM = len(ACTION_FEATURE_NAMES)


class NoCertifiedViableAction(RuntimeError):
    """The exact frontier contains no SAFE action that the agent may select."""


class RecoveryWitnessMismatch(RuntimeError):
    """A retained exact recovery witness no longer matches the live frontier."""


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def normalized_logmeanexp(values: Tensor, temperature: float | Tensor) -> Tensor:
    """Uniform-reference log-sum-exp over one nonempty candidate set."""

    if not torch.is_tensor(values) or values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a nonempty one-dimensional tensor")
    tau = torch.as_tensor(
        temperature, dtype=values.dtype, device=values.device
    )
    if tau.numel() != 1 or not bool(torch.isfinite(tau)) or float(tau) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    count = values.new_tensor(float(values.numel()))
    return tau * (torch.logsumexp(values / tau, dim=0) - torch.log(count))


def _validate_mode_inputs(
    q_values: Tensor,
    mode_ids: Tensor,
    within_temperatures: Tensor | Sequence[float],
) -> tuple[Tensor, Tensor]:
    if not torch.is_tensor(q_values) or q_values.ndim != 1 or not q_values.numel():
        raise ValueError("q_values must be a nonempty one-dimensional tensor")
    modes = torch.as_tensor(
        mode_ids, dtype=torch.long, device=q_values.device
    )
    if modes.shape != q_values.shape:
        raise ValueError("mode_ids must align with q_values")
    temperatures = torch.as_tensor(
        within_temperatures, dtype=q_values.dtype, device=q_values.device
    ).flatten()
    if temperatures.numel() != len(MODE_NAMES):
        raise ValueError(
            f"within_temperatures must have {len(MODE_NAMES)} entries"
        )
    if not bool(torch.isfinite(temperatures).all()) or bool(
        (temperatures <= 0.0).any()
    ):
        raise ValueError("within-mode temperatures must be finite and positive")
    if bool((modes < 0).any()) or bool((modes >= temperatures.numel()).any()):
        raise ValueError("mode_ids contain an unknown temporal mode")
    return modes, temperatures


def regularized_mode_values(
    q_values: Tensor,
    mode_ids: Tensor,
    within_temperatures: Tensor | Sequence[float],
) -> tuple[Tensor, Tensor]:
    """Return normalized values and IDs for only the live temporal modes."""

    modes, temperatures = _validate_mode_inputs(
        q_values, mode_ids, within_temperatures
    )
    values = []
    live_modes = []
    for mode_id in range(len(MODE_NAMES)):
        mask = modes == mode_id
        if not bool(mask.any()):
            continue
        values.append(
            normalized_logmeanexp(q_values[mask], temperatures[mode_id])
        )
        live_modes.append(mode_id)
    if not values:  # guarded by nonempty q, retained as a fail-closed check
        raise NoCertifiedViableAction("no live certified temporal mode")
    return (
        torch.stack(values),
        torch.as_tensor(live_modes, dtype=torch.long, device=q_values.device),
    )


def cardinality_normalized_continuation(
    q_values: Tensor,
    mode_ids: Tensor,
    within_temperatures: Tensor | Sequence[float],
    mode_temperature: float | Tensor,
) -> Tensor:
    """Nested log-mean-exp continuation over certified candidates only."""

    mode_values, _ = regularized_mode_values(
        q_values, mode_ids, within_temperatures
    )
    return normalized_logmeanexp(mode_values, mode_temperature)


def double_dqn_regularized_continuation(
    online_q: Tensor,
    target_q: Tensor,
    mode_ids: Tensor,
    within_temperatures: Tensor | Sequence[float],
    mode_temperature: float | Tensor,
) -> Tensor:
    """Evaluate the online-induced regularized policy with target-network Q.

    This is the soft/regularized Double-DQN construction.  The online values
    determine both the within-mode Boltzmann policies and the mode policy; the
    target values are used only in policy evaluation.  Uniform reference
    measures are explicitly renormalized over each live candidate set.
    """

    if target_q.shape != online_q.shape:
        raise ValueError("online_q and target_q must have the same shape")
    if target_q.device != online_q.device:
        target_q = target_q.to(online_q.device)
    target_q = target_q.to(dtype=online_q.dtype)
    modes, temperatures = _validate_mode_inputs(
        online_q, mode_ids, within_temperatures
    )
    tau_mode = torch.as_tensor(
        mode_temperature, dtype=online_q.dtype, device=online_q.device
    )
    if (
        tau_mode.numel() != 1
        or not bool(torch.isfinite(tau_mode))
        or float(tau_mode) <= 0.0
    ):
        raise ValueError("mode_temperature must be finite and positive")

    online_mode_values = []
    evaluated_mode_values = []
    for mode_id in range(len(MODE_NAMES)):
        mask = modes == mode_id
        if not bool(mask.any()):
            continue
        q_online_mode = online_q[mask]
        q_target_mode = target_q[mask]
        tau = temperatures[mode_id]
        log_policy = F.log_softmax(q_online_mode / tau, dim=0)
        policy = log_policy.exp()
        log_reference = -torch.log(
            q_online_mode.new_tensor(float(q_online_mode.numel()))
        )
        online_mode_values.append(
            normalized_logmeanexp(q_online_mode, tau)
        )
        evaluated_mode_values.append(
            torch.sum(
                policy
                * (
                    q_target_mode
                    - tau * (log_policy - log_reference)
                )
            )
        )

    online_modes = torch.stack(online_mode_values)
    evaluated_modes = torch.stack(evaluated_mode_values)
    live_mode_count = online_modes.numel()
    mode_log_policy = F.log_softmax(online_modes / tau_mode, dim=0)
    mode_policy = mode_log_policy.exp()
    log_mode_reference = -torch.log(
        online_modes.new_tensor(float(live_mode_count))
    )
    return torch.sum(
        mode_policy
        * (
            evaluated_modes
            - tau_mode * (mode_log_policy - log_mode_reference)
        )
    )


def common_smdp_target(
    rewards: Tensor,
    durations: Tensor,
    dones: Tensor,
    next_values: Tensor,
    *,
    gamma: float,
    reward_scale: float = 1.0,
) -> Tensor:
    """One variable-duration Bellman target shared by all three modes.

    ``rewards`` must already be the discounted intra-macro return
    :math:`R^{(k)}`.  No mode-specific target path exists here.
    """

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be finite and in [0, 1)")
    reward_scale = float(reward_scale)
    if not math.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("reward_scale must be finite and positive")
    rewards = torch.as_tensor(rewards)
    durations = torch.as_tensor(durations, device=rewards.device)
    dones = torch.as_tensor(dones, dtype=torch.bool, device=rewards.device)
    next_values = torch.as_tensor(
        next_values, dtype=rewards.dtype, device=rewards.device
    )
    if not (
        rewards.shape == durations.shape == dones.shape == next_values.shape
    ):
        raise ValueError("all SMDP target tensors must have the same shape")
    if bool((durations < 1).any()):
        raise ValueError("every macro duration must be at least one transition")
    discount = torch.pow(
        rewards.new_tensor(gamma), durations.to(dtype=rewards.dtype)
    )
    return (
        reward_scale * rewards
        + discount * (~dones).to(rewards.dtype) * next_values
    )


@dataclass(frozen=True)
class ViabilityGraphConfig:
    graph_hidden_dim: int = 64
    graph_embedding_dim: int = 64
    message_passing_steps: int = 3
    action_embedding_dim: int = 32
    head_hidden_dim: int = 128
    tau_accept: float = 0.1
    tau_recover: float = 0.1
    tau_defer: float = 0.1
    tau_mode: float = 1.0
    timing_scale: float = 100.0
    gamma: float = 0.99
    reward_scale: float = 0.01
    learning_rate: float = 5.0e-5
    batch_size: int = 128
    replay_capacity: int = 20_000
    update_every: int = 1
    target_update_every: int = 200
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    max_nonprogress_recovery_decisions: int = 2
    force_recovery_witness_when_due: bool = False

    def __post_init__(self) -> None:
        for name in (
            "graph_hidden_dim",
            "graph_embedding_dim",
            "action_embedding_dim",
            "head_hidden_dim",
            "batch_size",
            "replay_capacity",
            "update_every",
            "target_update_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_nonprogress_recovery_decisions, bool)
            or not isinstance(self.max_nonprogress_recovery_decisions, int)
            or self.max_nonprogress_recovery_decisions < 0
        ):
            raise ValueError(
                "max_nonprogress_recovery_decisions must be non-negative"
            )
        if not isinstance(self.force_recovery_witness_when_due, bool):
            raise ValueError("force_recovery_witness_when_due must be boolean")
        if (
            isinstance(self.message_passing_steps, bool)
            or not isinstance(self.message_passing_steps, int)
            or self.message_passing_steps < 0
        ):
            raise ValueError("message_passing_steps must be non-negative")
        for name in (
            "tau_accept",
            "tau_recover",
            "tau_defer",
            "tau_mode",
            "timing_scale",
            "reward_scale",
            "learning_rate",
            "grad_clip",
            "huber_delta",
        ):
            _positive_finite(getattr(self, name), name)
        if not math.isfinite(float(self.gamma)) or not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be finite and in [0, 1)")

    @property
    def within_temperatures(self) -> tuple[float, float, float]:
        return (self.tau_accept, self.tau_recover, self.tau_defer)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ViabilityGraphConfig":
        allowed = cls.__dataclass_fields__
        return cls(**{name: values[name] for name in allowed if name in values})


def _fraction(cell, rows: int, cols: int) -> tuple[float, float]:
    return (
        float(cell[0]) / max(rows - 1, 1),
        float(cell[1]) / max(cols - 1, 1),
    )


def candidate_action_features(
    candidate: ViabilityActionCandidate,
    *,
    guard_context: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> tuple[float, ...]:
    """Encode action identity/parameters without a policy or baseline score."""

    if candidate.certificate.status is not ViabilityStatus.SAFE:
        raise ValueError("action features may only encode an exact SAFE candidate")
    state = candidate.successor_state
    mode_id = MODE_TO_ID[candidate.mode.value]
    mode_one_hot = [float(index == mode_id) for index in range(len(MODE_NAMES))]
    action_types = tuple(ViabilityActionType)
    type_one_hot = [
        float(candidate.action_type is action_type)
        for action_type in action_types
    ]

    source_present = candidate.source is not None
    source_row, source_column = (
        _fraction(candidate.source, state.rows, state.cols)
        if source_present
        else (0.0, 0.0)
    )
    destination_present = candidate.destination is not None
    destination_row, destination_column = (
        _fraction(candidate.destination, state.rows, state.cols)
        if destination_present
        else (0.0, 0.0)
    )
    yard_cells = max(state.rows * state.cols, 1)
    storage_cells = max(len(state.storage_cells), 1)

    macro_steps = None
    if candidate.recovery_action is not None:
        macro_steps = int(candidate.recovery_action.steps)
    elif candidate.horizon_steps is not None:
        macro_steps = int(candidate.horizon_steps)

    rank_before = candidate.recovery_rank_before
    rank_after = candidate.recovery_rank_after
    rank_delta = candidate.rank_delta
    guard_context = tuple(float(value) for value in guard_context)
    if len(guard_context) != 4 or not all(
        math.isfinite(value) for value in guard_context
    ):
        raise ValueError("guard_context must contain four finite values")
    features = (
        *mode_one_hot,
        *type_one_hot,
        float(candidate.target_label is not None),
        float(source_present),
        source_row,
        source_column,
        float(destination_present),
        destination_row,
        destination_column,
        float(macro_steps is not None),
        0.0 if macro_steps is None else float(macro_steps) / yard_cells,
        float(candidate.horizon_steps is not None),
        (
            0.0
            if candidate.horizon_steps is None
            else float(candidate.horizon_steps) / yard_cells
        ),
        float(rank_before is not None),
        0.0 if rank_before is None else float(rank_before) / storage_cells,
        float(rank_after is not None),
        0.0 if rank_after is None else float(rank_after) / storage_cells,
        float(rank_delta is not None),
        0.0 if rank_delta is None else float(rank_delta) / storage_cells,
        *guard_context,
    )
    if len(features) != ACTION_FEATURE_DIM:  # defensive schema assertion
        raise RuntimeError("candidate action feature schema is inconsistent")
    if not all(math.isfinite(value) for value in features):
        raise ValueError("candidate action features must be finite")
    return tuple(float(value) for value in features)


@dataclass(frozen=True)
class ViabilityGraphCandidateRecord:
    """Replay-safe representation of one certified counterfactual action."""

    key: str
    mode_id: int
    action_type: str
    current_state: RecoveryState
    successor_state: RecoveryState
    action_features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("candidate key cannot be empty")
        if self.mode_id not in ID_TO_MODE:
            raise ValueError("candidate record has an unknown mode")
        if len(self.action_features) != ACTION_FEATURE_DIM:
            raise ValueError("candidate record action feature dimension mismatch")
        if not isinstance(self.current_state, RecoveryState) or not isinstance(
            self.successor_state, RecoveryState
        ):
            raise TypeError("candidate record requires current/successor RecoveryState")


@dataclass(frozen=True)
class PreparedViabilityGraphSnapshot:
    episode_instance_id: Optional[str]
    decision_epoch: int
    records: tuple[ViabilityGraphCandidateRecord, ...]
    source_indices: tuple[int, ...]
    interface_rejection_count: int
    exact_safe_candidate_count: int
    liveness_restricted: bool = False

    @property
    def mode_ids(self) -> tuple[int, ...]:
        return tuple(record.mode_id for record in self.records)


@dataclass(frozen=True)
class ViabilityGraphDecision:
    """Auditable binding between a SAFE live option and its scored record."""

    candidate: ViabilityActionCandidate
    record: ViabilityGraphCandidateRecord
    prepared_snapshot: PreparedViabilityGraphSnapshot
    q_values: tuple[float, ...]
    mode_values: tuple[tuple[str, float], ...]
    explored: bool
    selection_source: str
    liveness_forced: bool = False
    exact_rank_progress: bool = False

    @property
    def option(self):
        return self.candidate.option


@dataclass(frozen=True)
class ViabilityGraphTransition:
    chosen: ViabilityGraphCandidateRecord
    reward: float
    duration: int
    next_candidates: tuple[ViabilityGraphCandidateRecord, ...]
    done: bool

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reward)):
            raise ValueError("transition reward must be finite")
        if (
            isinstance(self.duration, bool)
            or not isinstance(self.duration, int)
            or self.duration < 1
        ):
            raise ValueError("transition duration must be a positive integer")
        if self.done and self.next_candidates:
            raise ValueError("terminal transition cannot retain next candidates")
        if not self.done and not self.next_candidates:
            raise ValueError("nonterminal transition needs a SAFE next frontier")

    def numeric_copy(self) -> "ViabilityGraphTransition":
        # RecoveryState and candidate records are recursively immutable.
        return ViabilityGraphTransition(
            chosen=self.chosen,
            reward=float(self.reward),
            duration=int(self.duration),
            next_candidates=tuple(self.next_candidates),
            done=bool(self.done),
        )


class ViabilityGraphReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.memory = deque(maxlen=self.capacity)

    def add(self, transition: ViabilityGraphTransition) -> None:
        if not isinstance(transition, ViabilityGraphTransition):
            raise TypeError("replay accepts ViabilityGraphTransition instances")
        self.memory.append(transition.numeric_copy())

    def sample(self, batch_size: int, rng: random.Random):
        batch_size = int(batch_size)
        if batch_size <= 0 or batch_size > len(self.memory):
            raise ValueError("invalid replay batch size")
        return rng.sample(list(self.memory), batch_size)

    def sample_mode_balanced(self, batch_size: int, rng: random.Random):
        """Balance chosen transitions across the three represented modes."""

        batch_size = int(batch_size)
        memory = list(self.memory)
        if batch_size <= 0 or batch_size > len(memory):
            raise ValueError("invalid replay batch size")
        buckets: dict[int, list[int]] = {}
        for index, transition in enumerate(memory):
            buckets.setdefault(transition.chosen.mode_id, []).append(index)
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
        missing = batch_size - len(selected)
        if missing:
            remaining = [
                index for index in range(len(memory)) if index not in selected_set
            ]
            selected.extend(rng.sample(remaining, missing))
        rng.shuffle(selected)
        return [memory[index] for index in selected]

    def mode_counts(self) -> dict[int, int]:
        return dict(
            Counter(transition.chosen.mode_id for transition in self.memory)
        )

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


class ViabilityGraphQNetwork(nn.Module):
    """Shared task-Q scorer for all modes and parameterized SAFE actions."""

    def __init__(self, config: ViabilityGraphConfig) -> None:
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
        self.q_head = nn.Sequential(
            nn.Linear(
                3 * config.graph_embedding_dim + config.action_embedding_dim,
                config.head_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.head_hidden_dim, 1),
        )

    def forward(
        self,
        current_states: Sequence[RecoveryState],
        successor_states: Sequence[RecoveryState],
        action_features: Tensor | Sequence[Sequence[float]],
    ) -> Tensor:
        current_states = tuple(current_states)
        successor_states = tuple(successor_states)
        if not current_states:
            raise ValueError("Q network requires at least one candidate state")
        if len(current_states) != len(successor_states):
            raise ValueError("current and successor state counts must match")
        if not all(
            isinstance(state, RecoveryState)
            for state in current_states + successor_states
        ):
            raise TypeError("every Q candidate state must be a RecoveryState")
        parameter = next(self.parameters())
        action_features = torch.as_tensor(
            action_features, dtype=parameter.dtype, device=parameter.device
        )
        if action_features.ndim == 1:
            action_features = action_features.unsqueeze(0)
        if action_features.shape != (len(current_states), ACTION_FEATURE_DIM):
            raise ValueError(
                "action features must have shape "
                f"[{len(current_states)}, {ACTION_FEATURE_DIM}]"
            )
        graphs = tuple(
            encode_recovery_state(
                state, timing_scale=self.config.timing_scale
            )
            for state in current_states + successor_states
        )
        graph_embedding = self.graph_encoder(pad_yard_graphs(graphs))
        count = len(current_states)
        current_embedding = graph_embedding[:count]
        successor_embedding = graph_embedding[count:]
        action_embedding = self.action_encoder(action_features)
        return self.q_head(
            torch.cat(
                (
                    current_embedding,
                    successor_embedding,
                    successor_embedding - current_embedding,
                    action_embedding,
                ),
                dim=-1,
            )
        ).squeeze(-1)


def _record(
    candidate: ViabilityActionCandidate,
    *,
    current_state: RecoveryState,
    guard_context: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> ViabilityGraphCandidateRecord:
    return ViabilityGraphCandidateRecord(
        key=candidate.key,
        mode_id=MODE_TO_ID[candidate.mode.value],
        action_type=candidate.action_type.value,
        current_state=current_state,
        successor_state=candidate.successor_state,
        action_features=candidate_action_features(
            candidate, guard_context=guard_context
        ),
    )


def prepare_viability_snapshot(
    snapshot: ViabilityCandidateSnapshot,
    *,
    guard_context: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> PreparedViabilityGraphSnapshot:
    """Apply the exact SAFE mask fail-closed and strip live option objects."""

    if not isinstance(snapshot, ViabilityCandidateSnapshot):
        raise TypeError("expected a ViabilityCandidateSnapshot")
    records = []
    source_indices = []
    rejected = 0
    for index, candidate in enumerate(snapshot.candidates):
        # This repeats the interface dataclass invariant deliberately.  The
        # exact status remains authoritative even if a deserialized or foreign
        # snapshot bypassed normal construction.
        if candidate.certificate.status is not ViabilityStatus.SAFE:
            rejected += 1
            continue
        records.append(
            _record(
                candidate,
                current_state=snapshot.recovery_state,
                guard_context=guard_context,
            )
        )
        source_indices.append(index)
    return PreparedViabilityGraphSnapshot(
        episode_instance_id=snapshot.episode_instance_id,
        decision_epoch=int(snapshot.decision_epoch),
        records=tuple(records),
        source_indices=tuple(source_indices),
        interface_rejection_count=rejected,
        exact_safe_candidate_count=len(records),
    )


def _physical_recovery_key(state: RecoveryState) -> tuple:
    """Configuration identity that ignores time elapsed inside one macro."""

    return (
        state.rows,
        state.cols,
        state.traversable,
        state.storage_cells,
        state.exits,
        tuple((block.label, block.position) for block in state.blocks),
        state.agent_position,
        state.fixed_obstacles,
        state.reserved_cells,
        state.pickup_cells,
        state.wait_cells,
    )


class RecoveryWitnessGuard:
    """Bound cyclic recovery, then execute a retained dynamics witness.

    Normal task learning may use any exact-SAFE recovery candidate for a
    bounded number of nonprogress recovery decisions.  Once the bound is
    reached (or the optional due trigger fires), the guard retains the current
    exact certificate witness and selects its next transition-model action
    until that finite witness is exhausted.  The witness is a feasibility
    certificate produced by recovery search, not a baseline behavior label.

    ``recovery_rank_before/after`` is treated as a rank only when the snapshot
    explicitly authenticates breadth-first search.  A goal-directed witness
    length is never interpreted as :math:`rho`.
    """

    def __init__(
        self,
        *,
        max_nonprogress_recovery_decisions: int,
        force_when_due: bool,
    ) -> None:
        self.max_nonprogress_recovery_decisions = int(
            max_nonprogress_recovery_decisions
        )
        self.force_when_due = bool(force_when_due)
        self.nonprogress_recovery_decisions = 0
        self.active_witness: tuple[RecoveryAction, ...] = ()
        self.witness_cursor = 0
        self.activations = 0
        self.forced_decisions = 0
        self.completed_witnesses = 0
        self.mismatches = 0

    @property
    def active(self) -> bool:
        return self.witness_cursor < len(self.active_witness)

    def reset(self, *, reset_nonprogress: bool = True) -> None:
        self.active_witness = ()
        self.witness_cursor = 0
        if reset_nonprogress:
            self.nonprogress_recovery_decisions = 0

    def _due_trigger(self, snapshot: ViabilityCandidateSnapshot) -> bool:
        return self.force_when_due and any(
            float(block.remaining_time) <= 0.0
            for block in snapshot.recovery_state.blocks
        )

    def feature_context(
        self,
        snapshot: ViabilityCandidateSnapshot,
        *,
        forced_frontier: bool,
    ) -> tuple[float, float, float, float]:
        allowance = max(self.max_nonprogress_recovery_decisions, 1)
        storage_count = max(len(snapshot.recovery_state.storage_cells), 1)
        remaining = max(len(self.active_witness) - self.witness_cursor, 0)
        return (
            float(self.active),
            float(self.nonprogress_recovery_decisions) / allowance,
            float(remaining) / storage_count,
            float(forced_frontier),
        )

    def _activate(self, snapshot: ViabilityCandidateSnapshot) -> bool:
        certificate = snapshot.current_certificate
        if certificate.status is not ViabilityStatus.SAFE:
            raise RecoveryWitnessMismatch(
                "cannot activate liveness guard without a SAFE certificate"
            )
        witness = tuple(certificate.witness)
        if not witness:
            if snapshot.recovery_state.blocks:
                raise RecoveryWitnessMismatch(
                    "SAFE nonempty recovery state has an empty witness"
                )
            self.reset()
            return False
        self.active_witness = witness
        self.witness_cursor = 0
        self.activations += 1
        return True

    def forced_prepared_index(
        self,
        snapshot: ViabilityCandidateSnapshot,
        prepared: PreparedViabilityGraphSnapshot,
    ) -> Optional[int]:
        should_activate = (
            self.nonprogress_recovery_decisions
            >= self.max_nonprogress_recovery_decisions
            or self._due_trigger(snapshot)
        )
        # A zero allowance means any recovery opportunity is immediately
        # guarded, not that empty states require a witness.
        has_recovery = any(
            candidate.mode is ViabilityMode.RECOVER
            for candidate in snapshot.candidates
        )
        if not self.active and should_activate and has_recovery:
            self._activate(snapshot)
        if not self.active:
            return None

        expected = self.active_witness[self.witness_cursor]
        matches = []
        for prepared_index, source_index in enumerate(prepared.source_indices):
            candidate = snapshot.candidates[source_index]
            if candidate.recovery_action == expected:
                matches.append(prepared_index)
        if len(matches) != 1:
            self.mismatches += 1
            self.reset(reset_nonprogress=False)
            raise RecoveryWitnessMismatch(
                "retained exact witness action did not bind uniquely to the "
                f"SAFE frontier (matches={len(matches)})"
            )
        return matches[0]

    def observe(
        self,
        decision: ViabilityGraphDecision,
        *,
        next_snapshot: Optional[ViabilityCandidateSnapshot],
        done: bool,
    ) -> None:
        candidate = decision.candidate
        if decision.liveness_forced:
            if not self.active:
                self.mismatches += 1
                raise RecoveryWitnessMismatch(
                    "forced decision completed without an active witness"
                )
            if next_snapshot is not None and not done:
                expected_key = _physical_recovery_key(candidate.successor_state)
                observed_key = _physical_recovery_key(
                    next_snapshot.recovery_state
                )
                if observed_key != expected_key:
                    self.mismatches += 1
                    self.reset(reset_nonprogress=False)
                    raise RecoveryWitnessMismatch(
                        "executed witness macro did not reach its certified "
                        "physical successor"
                    )
            self.forced_decisions += 1
            self.witness_cursor += 1
            self.nonprogress_recovery_decisions = 0
            if self.witness_cursor >= len(self.active_witness) or done:
                self.completed_witnesses += 1
                self.reset()
            return

        if candidate.mode is ViabilityMode.ACCEPT:
            # Admission changes the recovery problem, invalidating any retained
            # sequence and beginning a new bounded-learning interval.
            self.reset()
        elif candidate.mode is ViabilityMode.RECOVER:
            if (
                candidate.action_type is ViabilityActionType.DELIVER
                or decision.exact_rank_progress
            ):
                self.nonprogress_recovery_decisions = 0
            else:
                self.nonprogress_recovery_decisions += 1
        if done:
            self.reset()

    def state_dict(self) -> dict:
        return {
            "max_nonprogress_recovery_decisions": (
                self.max_nonprogress_recovery_decisions
            ),
            "force_when_due": self.force_when_due,
            "nonprogress_recovery_decisions": (
                self.nonprogress_recovery_decisions
            ),
            "active_witness": self.active_witness,
            "witness_cursor": self.witness_cursor,
            "activations": self.activations,
            "forced_decisions": self.forced_decisions,
            "completed_witnesses": self.completed_witnesses,
            "mismatches": self.mismatches,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = (
            self.max_nonprogress_recovery_decisions,
            self.force_when_due,
        )
        observed = (
            int(state["max_nonprogress_recovery_decisions"]),
            bool(state["force_when_due"]),
        )
        if observed != expected:
            raise ValueError("recovery witness guard configuration mismatch")
        self.nonprogress_recovery_decisions = int(
            state.get("nonprogress_recovery_decisions", 0)
        )
        self.active_witness = tuple(state.get("active_witness", ()))
        self.witness_cursor = int(state.get("witness_cursor", 0))
        self.activations = int(state.get("activations", 0))
        self.forced_decisions = int(state.get("forced_decisions", 0))
        self.completed_witnesses = int(state.get("completed_witnesses", 0))
        self.mismatches = int(state.get("mismatches", 0))

    def audit_dict(self) -> dict:
        return {
            "max_nonprogress_recovery_decisions": (
                self.max_nonprogress_recovery_decisions
            ),
            "force_when_due": self.force_when_due,
            "nonprogress_recovery_decisions": (
                self.nonprogress_recovery_decisions
            ),
            "active": self.active,
            "remaining_witness_actions": max(
                len(self.active_witness) - self.witness_cursor, 0
            ),
            "activations": self.activations,
            "forced_decisions": self.forced_decisions,
            "completed_witnesses": self.completed_witnesses,
            "mismatches": self.mismatches,
        }


class ViabilityGraphHierarchyAgent:
    """Exact-safe, teacher-free graph Double-DQN over parameterized macros."""

    def __init__(
        self,
        *,
        config: Optional[ViabilityGraphConfig] = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
        epsilon: float = 0.9,
    ) -> None:
        self.config = config or ViabilityGraphConfig()
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.epsilon = self._validate_epsilon(epsilon)
        self.rng = random.Random(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = ViabilityGraphQNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=self.config.learning_rate
        )
        self.replay = ViabilityGraphReplayBuffer(self.config.replay_capacity)
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
        self.decision_log: list[dict] = []

    @staticmethod
    def _validate_epsilon(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("epsilon must be finite and in [0, 1]")
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
        """Intersect exact safety with the current liveness shield."""

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
        # Activation changes the augmented Markov state, so rebuild every
        # record with the post-trigger guard context before Q evaluation or
        # replay insertion.
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
        except ValueError as error:  # exact status changed within one call
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
        current_states = tuple(record.current_state for record in records)
        successor_states = tuple(record.successor_state for record in records)
        features = tuple(record.action_features for record in records)
        return current_states, successor_states, features

    def _score_records(
        self,
        records: Sequence[ViabilityGraphCandidateRecord],
        *,
        network: Optional[ViabilityGraphQNetwork] = None,
        grad: bool = False,
    ) -> Tensor:
        records = tuple(records)
        if not records:
            raise NoCertifiedViableAction("no exact SAFE candidate to score")
        network = network or self.Q_local
        current_states, successor_states, features = self._record_tensors(records)
        if grad:
            return network(current_states, successor_states, features)
        with torch.no_grad():
            return network(current_states, successor_states, features)

    def score_snapshot(self, snapshot: ViabilityCandidateSnapshot) -> Tensor:
        """Return online task-Q values for the exact SAFE frontier only."""

        prepared = prepare_viability_snapshot(snapshot)
        return self._score_records(prepared.records).detach().cpu()

    def select(
        self,
        snapshot: ViabilityCandidateSnapshot,
        *,
        training: bool = True,
        epsilon: Optional[float] = None,
    ) -> ViabilityGraphDecision:
        prepared, liveness_forced = self._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "exact verifier exposed no SAFE executable macro"
            )
        q = self._score_records(prepared.records)
        modes = torch.as_tensor(
            prepared.mode_ids, dtype=torch.long, device=q.device
        )
        mode_values, live_modes = regularized_mode_values(
            q, modes, self.within_temperatures
        )
        epsilon_value = self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        explored = bool(
            not liveness_forced
            and training
            and self.rng.random() < epsilon_value
        )

        if liveness_forced:
            selected_index = 0
            selection_source = "exact_recovery_witness_guard"
        elif explored:
            # Uniform mode, then uniform candidate: exploration itself is not
            # biased toward the mode with the largest representation.
            selected_mode = self.rng.choice(live_modes.tolist())
            indices = [
                index
                for index, mode_id in enumerate(prepared.mode_ids)
                if mode_id == selected_mode
            ]
            selected_index = self.rng.choice(indices)
            selection_source = "safe_mode_balanced_epsilon"
        else:
            maximum_mode_value = float(mode_values.max().item())
            best_mode_positions = [
                index
                for index, value in enumerate(mode_values.tolist())
                if value == maximum_mode_value
            ]
            # Stable mode-ID tie break, independent of candidate enumeration.
            mode_position = min(
                best_mode_positions,
                key=lambda index: int(live_modes[index].item()),
            )
            selected_mode = int(live_modes[mode_position].item())
            mode_indices = [
                index
                for index, mode_id in enumerate(prepared.mode_ids)
                if mode_id == selected_mode
            ]
            maximum_q = max(float(q[index].item()) for index in mode_indices)
            selected_index = min(
                (
                    index
                    for index in mode_indices
                    if float(q[index].item()) == maximum_q
                ),
                key=lambda index: prepared.records[index].key,
            )
            selection_source = (
                "singleton_safe"
                if len(prepared.records) == 1
                else "regularized_mode_map_candidate_map"
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
            (
                ID_TO_MODE[int(mode_id.item())],
                float(value.item()),
            )
            for value, mode_id in zip(mode_values, live_modes)
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=record,
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in q.tolist()),
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
        self.exact_rejections_seen += int(
            snapshot.audit.fail_closed_rejection_count
        )
        self.decision_log.append(
            {
                "decision_index": self.decision_count - 1,
                "episode_instance_id": snapshot.episode_instance_id,
                "decision_epoch": int(snapshot.decision_epoch),
                "exact_safe_candidate_count": (
                    prepared.exact_safe_candidate_count
                ),
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
        """Convenience adapter for option-driven environment loops."""

        return self.select(snapshot, **kwargs).option

    def reset_episode_state(self) -> None:
        """Clear retained liveness state at a real environment reset."""

        self.recovery_witness_guard.reset()

    # Semantic alias for option-loop integrations.
    on_episode_reset = reset_episode_state

    def observe_outcome(
        self,
        decision: ViabilityGraphDecision,
        *,
        next_snapshot: Optional[ViabilityCandidateSnapshot],
        done: bool,
    ) -> Optional[PreparedViabilityGraphSnapshot]:
        """Advance the liveness shield after a macro in training or evaluation.

        The returned next frontier is already intersected with the updated
        guard and is therefore the frontier that the Bellman backup must use.
        Evaluation loops call this method even though they do not write replay.
        """

        if not isinstance(decision, ViabilityGraphDecision):
            raise TypeError("observe_outcome requires a ViabilityGraphDecision")
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
        decision: ViabilityGraphDecision,
        *,
        reward: float,
        duration: int,
        next_snapshot: Optional[ViabilityCandidateSnapshot],
        done: bool,
        outcome_already_observed: bool = False,
    ) -> ViabilityGraphTransition:
        """Store one complete macro transition.

        ``reward`` is the discounted reward accumulated inside the selected
        macro.  An infeasible/failure boundary must be passed as ``done=True``;
        a nonterminal transition is required to expose a nonempty exact SAFE
        next frontier.
        """

        if not isinstance(decision, ViabilityGraphDecision):
            raise TypeError("remember requires a ViabilityGraphDecision")
        done = bool(done)
        if outcome_already_observed:
            if done:
                next_prepared = None
            else:
                if next_snapshot is None:
                    raise ValueError(
                        "nonterminal transition requires next_snapshot"
                    )
                next_prepared, _ = self._admissible_prepared(next_snapshot)
                if not next_prepared.records:
                    raise NoCertifiedViableAction(
                        "nonterminal next state has no liveness-admissible "
                        "SAFE macro"
                    )
        else:
            next_prepared = self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=done
            )
        next_records = () if next_prepared is None else next_prepared.records
        transition = ViabilityGraphTransition(
            chosen=decision.record,
            reward=float(reward),
            duration=int(duration),
            next_candidates=tuple(next_records),
            done=done,
        )
        self.replay.add(transition)
        self.transition_count += 1
        return transition

    def _td_batch(
        self, transitions: Sequence[ViabilityGraphTransition]
    ) -> tuple[Tensor, Tensor]:
        transitions = tuple(transitions)
        prediction = self._score_records(
            tuple(item.chosen for item in transitions),
            network=self.Q_local,
            grad=True,
        )
        next_values = torch.zeros(
            len(transitions), dtype=prediction.dtype, device=self.device
        )
        nonterminal_records = [
            record
            for transition in transitions
            if not transition.done
            for record in transition.next_candidates
        ]
        if nonterminal_records:
            with torch.no_grad():
                online_next = self._score_records(
                    nonterminal_records, network=self.Q_local
                )
                target_next = self._score_records(
                    nonterminal_records, network=self.Q_target
                )
                offset = 0
                for index, transition in enumerate(transitions):
                    if transition.done:
                        continue
                    count = len(transition.next_candidates)
                    modes = torch.as_tensor(
                        [
                            record.mode_id
                            for record in transition.next_candidates
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                    next_values[index] = double_dqn_regularized_continuation(
                        online_next[offset : offset + count],
                        target_next[offset : offset + count],
                        modes,
                        self.within_temperatures,
                        self.config.tau_mode,
                    )
                    offset += count

        rewards = torch.as_tensor(
            [item.reward for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        durations = torch.as_tensor(
            [item.duration for item in transitions],
            dtype=torch.long,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        targets = common_smdp_target(
            rewards,
            durations,
            dones,
            next_values,
            gamma=self.config.gamma,
            reward_scale=self.config.reward_scale,
        )
        return prediction, targets.detach()

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        transitions = self.replay.sample_mode_balanced(
            self.config.batch_size, self.rng
        )
        prediction, target = self._td_batch(transitions)
        loss = F.smooth_l1_loss(
            prediction, target, beta=self.config.huber_delta
        )
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
        value = float(loss.detach().item())
        self.loss_history.append(value)
        return value

    def checkpoint_metadata(self) -> dict:
        return {
            "controller_architecture": VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE,
            "candidate_interface": VIABILITY_CANDIDATE_INTERFACE,
            "network_architecture": VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
            "replay_version": VIABILITY_GRAPH_REPLAY_VERSION,
            "backup_version": VIABILITY_GRAPH_BACKUP_VERSION,
            "policy_version": VIABILITY_GRAPH_POLICY_VERSION,
            "modes": MODE_NAMES,
            "action_types": tuple(item.value for item in ViabilityActionType),
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "graph_node_feature_dim": NODE_FEATURE_DIM,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "within_temperatures": self.config.within_temperatures,
            "mode_temperature": self.config.tau_mode,
            "reference_distribution": (
                "uniform_renormalized_within_live_mode_and_over_live_modes_v1"
            ),
            "common_smdp_continuation": True,
            "double_dqn": True,
            "double_q_semantics": (
                "online_regularized_policy_target_network_evaluation_v1"
            ),
            "exact_safe_mask_authoritative": True,
            "unsafe_unknown_fail_closed": True,
            "exact_verifier_authority": EXACT_VERIFIER_AUTHORITY,
            "certification_contract": FAIL_CLOSED_CERTIFICATION_CONTRACT,
            "counterfactual_successor_graph": True,
            "current_and_successor_graphs_shared_encoder": True,
            "q_context_composition": (
                "current_successor_successor_minus_current_action_v1"
            ),
            "deployment_policy": "mode_regularized_argmax_then_candidate_argmax",
            "training_exploration": "uniform_live_mode_then_uniform_safe_candidate",
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "behavior_cloning": False,
            "teacher_prior": False,
            "reward_scale": self.config.reward_scale,
            "gamma": self.config.gamma,
            "recovery_liveness_guard": (
                "bounded_nonprogress_then_retained_exact_dynamics_witness_v1"
            ),
            "max_nonprogress_recovery_decisions": (
                self.config.max_nonprogress_recovery_decisions
            ),
            "force_recovery_witness_when_due": (
                self.config.force_recovery_witness_when_due
            ),
            "goal_directed_witness_length_used_as_exact_rank": False,
            "liveness_state_in_action_features": True,
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
        guard_state = state.get("recovery_witness_guard")
        if resumable and guard_state is not None:
            self.recovery_witness_guard.load_state_dict(guard_state)
        else:
            # Deployment starts at an explicit episode reset; a witness bound
            # to a training episode must never leak across checkpoint loading.
            self.recovery_witness_guard.reset()
        if resumable:
            required = ("optimizer", "replay", "rng_state")
            missing = [name for name in required if name not in state]
            if missing:
                raise ValueError(
                    f"resumable checkpoint is missing {tuple(missing)!r}"
                )
            self.optimizer.load_state_dict(state["optimizer"])
            self.replay.load_state_dict(state["replay"])
            self.rng.setstate(state["rng_state"])

    def checkpoint(self, *, include_replay: bool = True, **metadata) -> dict:
        payload = {
            "checkpoint_schema_version": VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
            **self.checkpoint_metadata(),
            "config": self.config.to_dict(),
            "agent_state": self.checkpoint_state(include_replay=include_replay),
        }
        collisions = sorted(set(payload).intersection(metadata))
        if collisions:
            raise ValueError(
                f"checkpoint metadata cannot override reserved keys: {collisions!r}"
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
    ) -> "ViabilityGraphHierarchyAgent":
        expected = {
            "checkpoint_schema_version": VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE,
            "candidate_interface": VIABILITY_CANDIDATE_INTERFACE,
            "network_architecture": VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
            "backup_version": VIABILITY_GRAPH_BACKUP_VERSION,
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
                f"incompatible viability graph checkpoint: {mismatches!r}"
            )
        agent = cls(
            config=ViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(
            payload["agent_state"], resumable=resumable
        )
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
            "mode_decisions": dict(self.mode_decisions),
            "action_decisions": dict(self.action_decisions),
            "selection_sources": dict(self.selection_sources),
            "exact_verifier_rejections_seen": self.exact_rejections_seen,
            "interface_non_safe_rejections": self.interface_rejections,
            "safe_candidates_scored": self.safe_candidates_scored,
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "recovery_witness_guard_audit": (
                self.recovery_witness_guard.audit_dict()
            ),
        }
        if include_decisions:
            audit["decisions"] = tuple(self.decision_log)
        return audit


__all__ = [
    "ACTION_FEATURE_DIM",
    "ACTION_FEATURE_NAMES",
    "ID_TO_MODE",
    "MODE_NAMES",
    "MODE_TO_ID",
    "NoCertifiedViableAction",
    "PreparedViabilityGraphSnapshot",
    "RecoveryWitnessGuard",
    "RecoveryWitnessMismatch",
    "VIABILITY_GRAPH_BACKUP_VERSION",
    "VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION",
    "VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE",
    "VIABILITY_GRAPH_NETWORK_ARCHITECTURE",
    "VIABILITY_GRAPH_POLICY_VERSION",
    "VIABILITY_GRAPH_REPLAY_VERSION",
    "ViabilityGraphCandidateRecord",
    "ViabilityGraphConfig",
    "ViabilityGraphDecision",
    "ViabilityGraphHierarchyAgent",
    "ViabilityGraphQNetwork",
    "ViabilityGraphReplayBuffer",
    "ViabilityGraphTransition",
    "candidate_action_features",
    "cardinality_normalized_continuation",
    "common_smdp_target",
    "double_dqn_regularized_continuation",
    "normalized_logmeanexp",
    "prepare_viability_snapshot",
    "regularized_mode_values",
]
