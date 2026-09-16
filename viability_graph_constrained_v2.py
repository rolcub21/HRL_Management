"""Teacher-free, exact-safe constrained VCG-SMDP V2.

This module is intentionally incompatible with every V1/V1.2 checkpoint.  It
keeps exact certification and the retained recovery-witness guard
lexicographically above learning, then ranks the live frontier with two raw
critics:

* operational return, and
* undiscounted finite-episode physical storage rehandles.

The online Lagrangian merit induces one cardinality-normalized policy.  Both
target critics evaluate that same policy; there is no return-only maximum and
no cost-only minimum.  The raw component targets contain no entropy bonus, so
the method is regularized policy improvement with vector Expected-SARSA /
Double-Q evaluation rather than two independent soft-Bellman fixed points.
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

from PSLAP.viability import RecoveryState, ViabilityStatus
from PSLAP.viability_candidates import (
    EXACT_VERIFIER_AUTHORITY,
    FAIL_CLOSED_CERTIFICATION_CONTRACT,
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityCertificateCache,
    ViabilityMode,
)
from PSLAP.viability_candidates_hold_v2 import (
    CERTIFIED_HOLD_INTERFACE_V2,
    CERTIFIED_HOLD_RULE_V2,
    CertifiedHoldCandidateSnapshotV2,
    CertifiedHoldRuleV2,
    PrimitiveIdleBudgetStateV2,
    enumerate_viability_candidates_hold_v2,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    YardGraphEncoder,
    encode_recovery_state,
    pad_yard_graphs,
)
from viability_graph_hierarchy import (
    ACTION_FEATURE_NAMES,
    NoCertifiedViableAction,
    RecoveryWitnessGuard,
    RecoveryWitnessMismatch,
    ViabilityGraphDecision,
    candidate_action_features,
    normalized_logmeanexp,
)
from viability_graph_episodic_audit import episodic_common_smdp_target


CONSTRAINED_V2_CHECKPOINT_SCHEMA_VERSION = 1
CONSTRAINED_V2_CHECKPOINT_FAMILY = "vcg_constrained_vector_smdp_v2"
CONSTRAINED_V2_CONTROLLER_ARCHITECTURE = (
    "exact_safe_shared_graph_vector_q_primal_dual_v2"
)
CONSTRAINED_V2_NETWORK_ARCHITECTURE = (
    "shared_graph_trunk_operational_and_nonnegative_rehandle_heads_v2"
)
CONSTRAINED_V2_REPLAY_VERSION = "raw_vector_variable_candidate_smdp_replay_v2"
CONSTRAINED_V2_BACKUP_VERSION = (
    "elapsed_weighted_single_lagrangian_policy_vector_expected_double_q_two_discount_smdp_v2"
)
CONSTRAINED_V2_POLICY_VERSION = (
    "four_group_cardinality_normalized_lagrangian_map_v2"
)
CONSTRAINED_V2_COST_DEFINITION = (
    "primitive_completed_storage_to_storage_rehandle_count_v1"
)
CONSTRAINED_V2_TIME_CONTEXT = (
    "elapsed_remaining_horizon_and_primitive_idle_budget_fraction_v2"
)

CONTROL_GROUP_NAMES = ("accept", "deliver", "reconfigure", "hold")
CONTROL_GROUP_TO_ID = {
    name: index for index, name in enumerate(CONTROL_GROUP_NAMES)
}
ID_TO_CONTROL_GROUP = {
    index: name for name, index in CONTROL_GROUP_TO_ID.items()
}
ACTION_TYPE_TO_GROUP = {
    ViabilityActionType.ACCEPT.value: CONTROL_GROUP_TO_ID["accept"],
    ViabilityActionType.DELIVER.value: CONTROL_GROUP_TO_ID["deliver"],
    ViabilityActionType.RECONFIGURE.value: CONTROL_GROUP_TO_ID["reconfigure"],
    ViabilityActionType.DEFER.value: CONTROL_GROUP_TO_ID["hold"],
}

V2_ACTION_FEATURE_NAMES = ACTION_FEATURE_NAMES + (
    "elapsed_episode_fraction",
    "remaining_episode_fraction",
    "remaining_primitive_idle_budget_fraction",
)
V2_ACTION_FEATURE_DIM = len(V2_ACTION_FEATURE_NAMES)


def _positive_finite(value, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_finite(value, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class ConstrainedV2Config:
    graph_hidden_dim: int = 64
    graph_embedding_dim: int = 64
    message_passing_steps: int = 3
    action_embedding_dim: int = 32
    head_hidden_dim: int = 128
    tau_accept: float = 0.1
    tau_deliver: float = 0.1
    tau_reconfigure: float = 0.1
    tau_hold: float = 0.1
    tau_group: float = 1.0
    timing_scale: float = 100.0
    gamma_operational: float = 0.99
    gamma_rehandle: float = 1.0
    operational_reward_scale: float = 0.01
    cost_loss_weight: float = 1.0
    learning_rate: float = 5.0e-5
    batch_size: int = 128
    replay_capacity: int = 20_000
    update_every: int = 1
    target_update_every: int = 200
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    episode_horizon_steps: int = 2_000
    max_idle_steps: int = 20
    max_hold_steps: int = 10
    max_nonprogress_recovery_decisions: int = 2
    force_recovery_witness_when_due: bool = False
    lambda_initial: float = 0.0
    lambda_max: float = 20.0

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
            "episode_horizon_steps",
            "max_idle_steps",
            "max_hold_steps",
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
            raise ValueError(
                "max_nonprogress_recovery_decisions must be non-negative"
            )
        if not isinstance(self.force_recovery_witness_when_due, bool):
            raise TypeError("force_recovery_witness_when_due must be boolean")
        for name in (
            "tau_accept",
            "tau_deliver",
            "tau_reconfigure",
            "tau_hold",
            "tau_group",
            "timing_scale",
            "operational_reward_scale",
            "cost_loss_weight",
            "learning_rate",
            "grad_clip",
            "huber_delta",
            "lambda_max",
        ):
            _positive_finite(getattr(self, name), name=name)
        if float(self.gamma_operational) != 0.99:
            raise ValueError("this frozen V2 family requires gamma_operational=0.99")
        if float(self.gamma_rehandle) != 1.0:
            raise ValueError(
                "raw finite-episode total-rehandle semantics require gamma_rehandle=1"
            )
        initial = _nonnegative_finite(self.lambda_initial, name="lambda_initial")
        if initial > float(self.lambda_max):
            raise ValueError("lambda_initial cannot exceed lambda_max")

    @property
    def within_group_temperatures(self) -> tuple[float, float, float, float]:
        return (
            self.tau_accept,
            self.tau_deliver,
            self.tau_reconfigure,
            self.tau_hold,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping) -> "ConstrainedV2Config":
        if not isinstance(values, Mapping):
            raise TypeError("V2 configuration must be a mapping")
        expected = set(cls.__dataclass_fields__)
        observed = set(values)
        if observed != expected:
            raise ValueError(
                "V2 configuration field mismatch: "
                f"missing={sorted(expected - observed)!r}, "
                f"extra={sorted(observed - expected)!r}"
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})


def _validate_policy_inputs(
    online_operational: Tensor,
    online_rehandle: Tensor,
    target_operational: Tensor,
    target_rehandle: Tensor,
    group_ids: Tensor,
    within_temperatures: Tensor | Sequence[float],
    group_temperature: float | Tensor,
    dual_lambda: float | Tensor,
    operational_weight: float | Tensor,
):
    tensors = (
        online_operational,
        online_rehandle,
        target_operational,
        target_rehandle,
    )
    if any(not torch.is_tensor(value) or value.ndim != 1 for value in tensors):
        raise ValueError("all critic vectors must be one-dimensional tensors")
    if not online_operational.numel():
        raise ValueError("critic vectors cannot be empty")
    if any(value.shape != online_operational.shape for value in tensors[1:]):
        raise ValueError("all critic vectors must align")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("critic vectors must be finite")
    device = online_operational.device
    dtype = online_operational.dtype
    values = tuple(value.to(device=device, dtype=dtype) for value in tensors)
    groups = torch.as_tensor(group_ids, dtype=torch.long, device=device)
    if groups.shape != online_operational.shape:
        raise ValueError("group_ids must align with critic vectors")
    if bool((groups < 0).any()) or bool((groups >= len(CONTROL_GROUP_NAMES)).any()):
        raise ValueError("group_ids contain an unknown control group")
    temperatures = torch.as_tensor(
        within_temperatures, dtype=dtype, device=device
    ).flatten()
    if temperatures.numel() != len(CONTROL_GROUP_NAMES):
        raise ValueError("within_temperatures must have four entries")
    if not bool(torch.isfinite(temperatures).all()) or bool(
        (temperatures <= 0.0).any()
    ):
        raise ValueError("temperatures must be finite and positive")
    tau_group = torch.as_tensor(group_temperature, dtype=dtype, device=device)
    lam = torch.as_tensor(dual_lambda, dtype=dtype, device=device)
    op_weight = torch.as_tensor(
        operational_weight, dtype=dtype, device=device
    )
    for name, scalar in (
        ("group_temperature", tau_group),
        ("dual_lambda", lam),
        ("operational_weight", op_weight),
    ):
        if scalar.numel() != 1 or not bool(torch.isfinite(scalar)):
            raise ValueError(f"{name} must be a finite scalar")
    if (
        float(tau_group) <= 0.0
        or float(lam) < 0.0
        or not 0.0 < float(op_weight) <= 1.0
    ):
        raise ValueError(
            "group temperature must be positive, lambda non-negative, and "
            "operational_weight in (0, 1]"
        )
    return (*values, groups, temperatures, tau_group, lam, op_weight)


def shared_lagrangian_policy_values(
    online_operational: Tensor,
    online_rehandle: Tensor,
    target_operational: Tensor,
    target_rehandle: Tensor,
    group_ids: Tensor,
    within_temperatures: Tensor | Sequence[float],
    group_temperature: float | Tensor,
    dual_lambda: float | Tensor,
    *,
    operational_weight: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate both target heads under one online-induced nested policy.

    Returns ``(V_op, V_phys, live_group_merits, live_group_ids)``.  The two
    component values are raw expectations and deliberately exclude KL terms.
    """

    (
        op_online,
        cost_online,
        op_target,
        cost_target,
        groups,
        temperatures,
        tau_group,
        lam,
        op_weight,
    ) = _validate_policy_inputs(
        online_operational,
        online_rehandle,
        target_operational,
        target_rehandle,
        group_ids,
        within_temperatures,
        group_temperature,
        dual_lambda,
        operational_weight,
    )
    # Q_op and Q_phys are both expressed at the current decision boundary, but
    # the declared Lagrangian is measured at episode start.  With mixed
    # discounts the operational component must therefore be weighted by
    # gamma_op**elapsed_steps while the raw finite-episode cost is not.
    merit = op_weight * op_online - lam * cost_online
    group_merits = []
    group_op_values = []
    group_cost_values = []
    live_ids = []
    for group_id in range(len(CONTROL_GROUP_NAMES)):
        mask = groups == group_id
        if not bool(mask.any()):
            continue
        tau = temperatures[group_id]
        within_policy = F.softmax(merit[mask] / tau, dim=0)
        group_merits.append(normalized_logmeanexp(merit[mask], tau))
        group_op_values.append(torch.sum(within_policy * op_target[mask]))
        group_cost_values.append(torch.sum(within_policy * cost_target[mask]))
        live_ids.append(group_id)
    group_merits = torch.stack(group_merits)
    group_policy = F.softmax(group_merits / tau_group, dim=0)
    return (
        torch.sum(group_policy * torch.stack(group_op_values)),
        torch.sum(group_policy * torch.stack(group_cost_values)),
        group_merits,
        torch.as_tensor(live_ids, dtype=torch.long, device=groups.device),
    )


def constrained_component_targets(
    operational_rewards: Tensor,
    rehandle_costs: Tensor,
    durations: Tensor,
    dones: Tensor,
    next_operational_values: Tensor,
    next_rehandle_values: Tensor,
    *,
    gamma_operational: float,
    gamma_rehandle: float,
    operational_reward_scale: float,
) -> tuple[Tensor, Tensor]:
    op_target = episodic_common_smdp_target(
        operational_rewards,
        durations,
        dones,
        next_operational_values,
        gamma=gamma_operational,
        reward_scale=operational_reward_scale,
    )
    cost_target = episodic_common_smdp_target(
        rehandle_costs,
        durations,
        dones,
        next_rehandle_values,
        gamma=gamma_rehandle,
        reward_scale=1.0,
    )
    return op_target, cost_target


@dataclass(frozen=True)
class ConstrainedV2CandidateRecord:
    key: str
    group_id: int
    action_type: str
    current_state: RecoveryState
    successor_state: RecoveryState
    action_features: tuple[float, ...]
    elapsed_steps: int
    remaining_episode_steps: int
    idle_budget_remaining_steps: int
    idle_budget_max_steps: int

    @property
    def mode_id(self) -> int:
        # Compatibility with the retained-witness preparation container.
        return self.group_id

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("candidate key cannot be empty")
        if self.group_id not in ID_TO_CONTROL_GROUP:
            raise ValueError("unknown V2 control group")
        if self.action_type not in ACTION_TYPE_TO_GROUP:
            raise ValueError("unknown action type")
        if ACTION_TYPE_TO_GROUP[self.action_type] != self.group_id:
            raise ValueError("action type and group disagree")
        if len(self.action_features) != V2_ACTION_FEATURE_DIM:
            raise ValueError("V2 action feature dimension mismatch")
        for name in (
            "elapsed_steps",
            "remaining_episode_steps",
            "idle_budget_remaining_steps",
            "idle_budget_max_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.elapsed_steps < 0 or self.remaining_episode_steps <= 0:
            raise ValueError("candidate horizon context is invalid")
        if self.idle_budget_max_steps <= 0 or not (
            0 <= self.idle_budget_remaining_steps <= self.idle_budget_max_steps
        ):
            raise ValueError("candidate idle-budget context is invalid")


@dataclass(frozen=True)
class ConstrainedV2PreparedSnapshot:
    episode_instance_id: Optional[str]
    decision_epoch: int
    records: tuple[ConstrainedV2CandidateRecord, ...]
    source_indices: tuple[int, ...]
    interface_rejection_count: int
    exact_safe_candidate_count: int
    liveness_restricted: bool = False

    @property
    def mode_ids(self) -> tuple[int, ...]:
        return tuple(record.group_id for record in self.records)


def _unwrap_snapshot(snapshot):
    if isinstance(snapshot, CertifiedHoldCandidateSnapshotV2):
        return snapshot.frontier, snapshot.hold_audit
    if isinstance(snapshot, ViabilityCandidateSnapshot):
        return snapshot, None
    raise TypeError("expected V1 or certified-Hold V2 candidate snapshot")


def prepare_constrained_snapshot(
    snapshot: CertifiedHoldCandidateSnapshotV2,
    *,
    config: ConstrainedV2Config,
    guard_context: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> ConstrainedV2PreparedSnapshot:
    if not isinstance(snapshot, CertifiedHoldCandidateSnapshotV2):
        raise TypeError(
            "constrained V2 requires an authenticated certified-Hold V2 snapshot"
        )
    if not isinstance(config, ConstrainedV2Config):
        raise TypeError("config must be ConstrainedV2Config")
    frontier, hold_audit = _unwrap_snapshot(snapshot)
    assert hold_audit is not None
    if hold_audit.interface != CERTIFIED_HOLD_INTERFACE_V2:
        raise ValueError("certified Hold frontier interface mismatch")
    if hold_audit.rule != CERTIFIED_HOLD_RULE_V2:
        raise ValueError("certified Hold rule mismatch")
    horizon = int(config.episode_horizon_steps)
    epoch = int(frontier.decision_epoch)
    if not 0 <= epoch < horizon:
        raise ValueError("V2 decision epoch lies outside the episode horizon")
    remaining = horizon - epoch
    if int(hold_audit.remaining_episode_steps) != remaining:
        raise ValueError("Hold audit remaining horizon does not match decision epoch")
    if int(hold_audit.idle_budget_max_steps) != int(config.max_idle_steps):
        raise ValueError("Hold audit idle-budget maximum does not match V2 config")
    idle_remaining = int(hold_audit.idle_budget_remaining_steps)
    idle_used = int(hold_audit.idle_steps_since_progress)
    if not 0 <= idle_remaining <= config.max_idle_steps:
        raise ValueError("Hold audit remaining idle budget is invalid")
    if idle_used + idle_remaining != config.max_idle_steps:
        raise ValueError("Hold audit idle-budget partition is inconsistent")
    if any(
        not 1 <= int(value) <= min(
            config.max_hold_steps, idle_remaining, remaining
        )
        for value in hold_audit.hold_horizons
    ):
        raise ValueError("Hold audit contains a horizon outside the V2 contract")
    hold_candidates = tuple(
        candidate
        for candidate in frontier.candidates
        if candidate.action_type is ViabilityActionType.DEFER
    )
    if len(hold_candidates) != int(hold_audit.hold_candidate_count):
        raise ValueError("Hold audit candidate count does not match the frontier")
    if tuple(candidate.horizon_steps for candidate in hold_candidates) != tuple(
        hold_audit.hold_horizons
    ):
        raise ValueError("Hold audit horizons do not match the frontier")
    elapsed_fraction = epoch / horizon
    remaining_fraction = remaining / horizon
    idle_fraction = idle_remaining / config.max_idle_steps
    records = []
    source_indices = []
    rejected = 0
    for index, candidate in enumerate(frontier.candidates):
        if candidate.certificate.status is not ViabilityStatus.SAFE:
            rejected += 1
            continue
        features = (
            *candidate_action_features(candidate, guard_context=guard_context),
            elapsed_fraction,
            remaining_fraction,
            idle_fraction,
        )
        records.append(
            ConstrainedV2CandidateRecord(
                key=str(candidate.key),
                group_id=ACTION_TYPE_TO_GROUP[candidate.action_type.value],
                action_type=candidate.action_type.value,
                current_state=frontier.recovery_state,
                successor_state=candidate.successor_state,
                action_features=tuple(float(value) for value in features),
                elapsed_steps=epoch,
                remaining_episode_steps=remaining,
                idle_budget_remaining_steps=idle_remaining,
                idle_budget_max_steps=config.max_idle_steps,
            )
        )
        source_indices.append(index)
    return ConstrainedV2PreparedSnapshot(
        episode_instance_id=frontier.episode_instance_id,
        decision_epoch=epoch,
        records=tuple(records),
        source_indices=tuple(source_indices),
        interface_rejection_count=rejected,
        exact_safe_candidate_count=len(records),
    )


@dataclass(frozen=True)
class ConstrainedV2Transition:
    chosen: ConstrainedV2CandidateRecord
    operational_reward: float
    physical_rehandle_cost: float
    duration: int
    next_candidates: tuple[ConstrainedV2CandidateRecord, ...]
    done: bool
    behavior_lambda: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.operational_reward)):
            raise ValueError("operational reward must be finite")
        physical_cost = _nonnegative_finite(
            self.physical_rehandle_cost, name="physical cost"
        )
        if not physical_cost.is_integer():
            raise ValueError("physical cost must be an integer event count")
        if isinstance(self.duration, bool) or not isinstance(self.duration, int) or self.duration < 1:
            raise ValueError("duration must be a positive integer")
        if self.done and self.next_candidates:
            raise ValueError("terminal transition cannot retain a frontier")
        if not self.done and not self.next_candidates:
            raise ValueError("nonterminal transition requires a frontier")
        _nonnegative_finite(self.behavior_lambda, name="behavior_lambda")

    def numeric_copy(self) -> "ConstrainedV2Transition":
        return ConstrainedV2Transition(
            chosen=self.chosen,
            operational_reward=float(self.operational_reward),
            physical_rehandle_cost=float(self.physical_rehandle_cost),
            duration=int(self.duration),
            next_candidates=tuple(self.next_candidates),
            done=bool(self.done),
            behavior_lambda=float(self.behavior_lambda),
        )


class ConstrainedV2ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.memory = deque(maxlen=self.capacity)

    def add(self, transition: ConstrainedV2Transition) -> None:
        if not isinstance(transition, ConstrainedV2Transition):
            raise TypeError("V2 replay requires vector transitions")
        self.memory.append(transition.numeric_copy())

    def sample_group_balanced(self, batch_size: int, rng: random.Random):
        memory = list(self.memory)
        batch_size = int(batch_size)
        if batch_size <= 0 or batch_size > len(memory):
            raise ValueError("invalid batch size")
        buckets = {}
        for index, transition in enumerate(memory):
            buckets.setdefault(transition.chosen.group_id, []).append(index)
        live = sorted(buckets)
        base, remainder = divmod(batch_size, len(live))
        selected = []
        selected_set = set()
        for position, group_id in enumerate(live):
            count = base + int(position < remainder)
            chosen = rng.sample(buckets[group_id], min(count, len(buckets[group_id])))
            selected.extend(chosen)
            selected_set.update(chosen)
        missing = batch_size - len(selected)
        if missing:
            remaining = [i for i in range(len(memory)) if i not in selected_set]
            selected.extend(rng.sample(remaining, missing))
        rng.shuffle(selected)
        return [memory[index] for index in selected]

    def group_counts(self) -> dict[int, int]:
        return dict(Counter(item.chosen.group_id for item in self.memory))

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


class ConstrainedV2QNetwork(nn.Module):
    """Shared graph/action trunk with raw operational and nonnegative cost heads."""

    def __init__(self, config: ConstrainedV2Config) -> None:
        super().__init__()
        self.config = config
        self.graph_encoder = YardGraphEncoder(
            input_dim=NODE_FEATURE_DIM,
            hidden_dim=config.graph_hidden_dim,
            output_dim=config.graph_embedding_dim,
            message_passing_steps=config.message_passing_steps,
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(V2_ACTION_FEATURE_DIM, config.action_embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(config.action_embedding_dim),
        )
        width = 3 * config.graph_embedding_dim + config.action_embedding_dim
        self.shared_head = nn.Sequential(
            nn.Linear(width, config.head_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
            nn.SiLU(),
        )
        self.operational_head = nn.Linear(config.head_hidden_dim, 1)
        self.rehandle_head = nn.Linear(config.head_hidden_dim, 1)

    def forward(
        self,
        current_states: Sequence[RecoveryState],
        successor_states: Sequence[RecoveryState],
        action_features: Tensor | Sequence[Sequence[float]],
    ) -> tuple[Tensor, Tensor]:
        current_states = tuple(current_states)
        successor_states = tuple(successor_states)
        if not current_states or len(current_states) != len(successor_states):
            raise ValueError("current/successor state batches must be nonempty and align")
        parameter = next(self.parameters())
        features = torch.as_tensor(
            action_features, dtype=parameter.dtype, device=parameter.device
        )
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape != (len(current_states), V2_ACTION_FEATURE_DIM):
            raise ValueError("V2 action feature tensor has the wrong shape")
        graphs = tuple(
            encode_recovery_state(state, timing_scale=self.config.timing_scale)
            for state in current_states + successor_states
        )
        embedding = self.graph_encoder(pad_yard_graphs(graphs))
        count = len(current_states)
        current = embedding[:count]
        successor = embedding[count:]
        action = self.action_encoder(features)
        shared = self.shared_head(
            torch.cat((current, successor, successor - current, action), dim=-1)
        )
        operational = self.operational_head(shared).squeeze(-1)
        # Softplus prevents a negative predicted resource from becoming an
        # artificial Lagrangian action bonus.
        rehandle = F.softplus(self.rehandle_head(shared).squeeze(-1))
        return operational, rehandle


class ConstrainedV2HierarchyAgent:
    """Exact-safe four-group vector critic with a projected-dual policy.

    The recovery witness remains a hard liveness shield.  Learning sees the
    already-shielded frontier and therefore backs up the same action set used
    at deployment.  ``dual_lambda`` changes action ranking only; it never
    rewrites the two raw replay signals.
    """

    def __init__(
        self,
        *,
        config: Optional[ConstrainedV2Config] = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
        epsilon: float = 0.0,
        dual_lambda: Optional[float] = None,
    ) -> None:
        self.config = config or ConstrainedV2Config()
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.epsilon = self._validate_epsilon(epsilon)
        self.dual_lambda = self._validate_lambda(
            self.config.lambda_initial if dual_lambda is None else dual_lambda
        )
        self.rng = random.Random(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = ConstrainedV2QNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=self.config.learning_rate
        )
        self.replay = ConstrainedV2ReplayBuffer(self.config.replay_capacity)
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
        self.group_decisions: Counter[str] = Counter()
        self.action_decisions: Counter[str] = Counter()
        self.selection_sources: Counter[str] = Counter()
        self.exact_rejections_seen = 0
        self.interface_rejections = 0
        self.safe_candidates_scored = 0
        self.operational_loss_history: list[float] = []
        self.rehandle_loss_history: list[float] = []
        self.total_loss_history: list[float] = []
        self.decision_log: list[dict] = []

    @staticmethod
    def _validate_epsilon(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value != 0.0:
            raise ValueError(
                "constrained V2 uses its induced soft policy for training; "
                "epsilon must be exactly zero"
            )
        return value

    def _validate_lambda(self, value: float) -> float:
        value = _nonnegative_finite(value, name="dual_lambda")
        if value > float(self.config.lambda_max):
            raise ValueError("dual_lambda exceeds the configured upper bound")
        return value

    @property
    def within_group_temperature_values(self) -> tuple[float, float, float, float]:
        return self.config.within_group_temperatures

    @property
    def within_group_temperatures(self) -> Tensor:
        return torch.as_tensor(
            self.within_group_temperature_values,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def group_temperature(self) -> float:
        return float(self.config.tau_group)

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = self._validate_epsilon(epsilon)

    def set_dual_lambda(self, value: float) -> None:
        self.dual_lambda = self._validate_lambda(value)

    def _operational_weight(self, elapsed_steps: int) -> float:
        if (
            isinstance(elapsed_steps, bool)
            or not isinstance(elapsed_steps, int)
            or not 0 <= elapsed_steps < self.config.episode_horizon_steps
        ):
            raise ValueError("elapsed_steps lie outside the V2 episode horizon")
        return float(self.config.gamma_operational) ** int(elapsed_steps)

    @staticmethod
    def _record_tensors(records: Sequence[ConstrainedV2CandidateRecord]):
        records = tuple(records)
        return (
            tuple(record.current_state for record in records),
            tuple(record.successor_state for record in records),
            tuple(record.action_features for record in records),
        )

    def _score_records(
        self,
        records: Sequence[ConstrainedV2CandidateRecord],
        *,
        network: Optional[ConstrainedV2QNetwork] = None,
        grad: bool = False,
    ) -> tuple[Tensor, Tensor]:
        records = tuple(records)
        if not records:
            raise NoCertifiedViableAction("no exact SAFE candidate to score")
        network = network or self.Q_local
        current, successor, features = self._record_tensors(records)
        if grad:
            return network(current, successor, features)
        with torch.no_grad():
            return network(current, successor, features)

    def _admissible_prepared(
        self, snapshot
    ) -> tuple[ConstrainedV2PreparedSnapshot, bool]:
        frontier, _ = _unwrap_snapshot(snapshot)
        initial = prepare_constrained_snapshot(
            snapshot,
            config=self.config,
            guard_context=self.recovery_witness_guard.feature_context(
                frontier, forced_frontier=False
            ),
        )
        if not initial.records:
            return initial, False
        forced_index = self.recovery_witness_guard.forced_prepared_index(
            frontier, initial
        )
        forced = forced_index is not None
        prepared = prepare_constrained_snapshot(
            snapshot,
            config=self.config,
            guard_context=self.recovery_witness_guard.feature_context(
                frontier, forced_frontier=forced
            ),
        )
        if forced_index is None:
            return prepared, False
        source_index = initial.source_indices[forced_index]
        try:
            restricted_index = prepared.source_indices.index(source_index)
        except ValueError as error:
            raise RecoveryWitnessMismatch(
                "forced witness candidate vanished while preparing V2 frontier"
            ) from error
        return (
            ConstrainedV2PreparedSnapshot(
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

    def score_snapshot(self, snapshot) -> tuple[Tensor, Tensor, Tensor]:
        prepared, _ = self._admissible_prepared(snapshot)
        operational, rehandle = self._score_records(prepared.records)
        elapsed_steps = {record.elapsed_steps for record in prepared.records}
        if len(elapsed_steps) != 1:
            raise ValueError("one V2 frontier contains mixed elapsed-step contexts")
        merit = (
            self._operational_weight(elapsed_steps.pop()) * operational
            - self.dual_lambda * rehandle
        )
        return (
            operational.detach().cpu(),
            rehandle.detach().cpu(),
            merit.detach().cpu(),
        )

    def select(
        self,
        snapshot,
        *,
        training: bool = True,
        epsilon: Optional[float] = None,
    ) -> ViabilityGraphDecision:
        frontier, _ = _unwrap_snapshot(snapshot)
        prepared, liveness_forced = self._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "exact verifier exposed no SAFE executable V2 macro"
            )
        operational, rehandle = self._score_records(prepared.records)
        elapsed_steps = {record.elapsed_steps for record in prepared.records}
        if len(elapsed_steps) != 1:
            raise ValueError("one V2 frontier contains mixed elapsed-step contexts")
        operational_weight = self._operational_weight(elapsed_steps.pop())
        merit = operational_weight * operational - self.dual_lambda * rehandle
        groups = torch.as_tensor(
            prepared.mode_ids, dtype=torch.long, device=self.device
        )
        # Only the merits/group IDs are used here.  The raw target values are
        # immaterial to MAP selection, so online heads are passed twice.
        _, _, group_values, live_groups = shared_lagrangian_policy_values(
            operational,
            rehandle,
            operational,
            rehandle,
            groups,
            self.within_group_temperatures,
            self.group_temperature,
            self.dual_lambda,
            operational_weight=operational_weight,
        )
        epsilon_value = self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        if epsilon_value != 0.0:  # retained as a fail-closed invariant
            raise RuntimeError("V2 epsilon invariant was violated")
        explored = bool(not liveness_forced and training)
        if liveness_forced:
            selected_index = 0
            selection_source = "exact_recovery_witness_guard"
        elif training:
            outer_probabilities = F.softmax(
                group_values / self.group_temperature, dim=0
            ).detach().cpu().tolist()
            group_position = self.rng.choices(
                range(len(live_groups)), weights=outer_probabilities, k=1
            )[0]
            selected_group = int(live_groups[group_position].item())
            indices = [
                index
                for index, group_id in enumerate(prepared.mode_ids)
                if group_id == selected_group
            ]
            within_probabilities = F.softmax(
                merit[indices]
                / self.within_group_temperatures[selected_group],
                dim=0,
            ).detach().cpu().tolist()
            selected_index = self.rng.choices(
                indices, weights=within_probabilities, k=1
            )[0]
            selection_source = "induced_regularized_lagrangian_policy_sample"
        else:
            maximum_group = float(group_values.max().item())
            positions = [
                index
                for index, value in enumerate(group_values.tolist())
                if value == maximum_group
            ]
            group_position = min(
                positions, key=lambda index: int(live_groups[index].item())
            )
            selected_group = int(live_groups[group_position].item())
            indices = [
                index
                for index, group_id in enumerate(prepared.mode_ids)
                if group_id == selected_group
            ]
            maximum_merit = max(float(merit[index].item()) for index in indices)
            selected_index = min(
                (
                    index
                    for index in indices
                    if float(merit[index].item()) == maximum_merit
                ),
                key=lambda index: prepared.records[index].key,
            )
            selection_source = (
                "singleton_safe"
                if len(prepared.records) == 1
                else "regularized_group_map_lagrangian_candidate_map"
            )

        source_index = prepared.source_indices[selected_index]
        candidate = frontier.candidates[source_index]
        record = prepared.records[selected_index]
        exact_rank_progress = bool(
            frontier.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        group_pairs = tuple(
            (
                ID_TO_CONTROL_GROUP[int(group_id.item())],
                float(value.item()),
            )
            for value, group_id in zip(group_values, live_groups)
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=record,
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merit.tolist()),
            mode_values=group_pairs,
            explored=explored,
            selection_source=selection_source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        self.decision_count += 1
        self.group_decisions[ID_TO_CONTROL_GROUP[record.group_id]] += 1
        self.action_decisions[candidate.action_type.value] += 1
        self.selection_sources[selection_source] += 1
        self.safe_candidates_scored += len(prepared.records)
        self.interface_rejections += prepared.interface_rejection_count
        self.exact_rejections_seen += int(
            frontier.audit.fail_closed_rejection_count
        )
        self.decision_log.append(
            {
                "decision_index": self.decision_count - 1,
                "episode_instance_id": frontier.episode_instance_id,
                "decision_epoch": int(frontier.decision_epoch),
                "dual_lambda": float(self.dual_lambda),
                "selected_key": candidate.key,
                "selected_group": ID_TO_CONTROL_GROUP[record.group_id],
                "selected_action_type": candidate.action_type.value,
                "selected_operational_q": float(operational[selected_index]),
                "selected_rehandle_q": float(rehandle[selected_index]),
                "selected_lagrangian_merit": float(merit[selected_index]),
                "selection_source": selection_source,
                "explored": explored,
                "liveness_forced": liveness_forced,
                "exact_rank_progress": exact_rank_progress,
                "live_groups": tuple(name for name, _ in group_pairs),
                "admissible_candidate_count": len(prepared.records),
            }
        )
        return decision

    def select_option(self, snapshot, **kwargs):
        return self.select(snapshot, **kwargs).option

    def reset_episode_state(self) -> None:
        self.recovery_witness_guard.reset()

    on_episode_reset = reset_episode_state

    def observe_outcome(
        self,
        decision: ViabilityGraphDecision,
        *,
        next_snapshot,
        done: bool,
        prepare_next: bool = True,
    ) -> Optional[ConstrainedV2PreparedSnapshot]:
        done = bool(done)
        if not done and next_snapshot is None:
            raise ValueError("nonterminal outcome requires next_snapshot")
        next_frontier = (
            None if next_snapshot is None else _unwrap_snapshot(next_snapshot)[0]
        )
        self.recovery_witness_guard.observe(
            decision, next_snapshot=next_frontier, done=done
        )
        if done or not prepare_next:
            return None
        prepared, _ = self._admissible_prepared(next_snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "nonterminal next state has no liveness-admissible SAFE V2 macro"
            )
        return prepared

    def remember(
        self,
        decision: ViabilityGraphDecision,
        *,
        operational_reward: float,
        physical_rehandle_cost: float,
        duration: int,
        next_snapshot,
        done: bool,
        outcome_already_observed: bool = False,
    ) -> ConstrainedV2Transition:
        if outcome_already_observed:
            if done:
                next_prepared = None
            else:
                if next_snapshot is None:
                    raise ValueError("nonterminal transition requires next_snapshot")
                next_prepared, _ = self._admissible_prepared(next_snapshot)
                if not next_prepared.records:
                    raise NoCertifiedViableAction(
                        "nonterminal next state has no SAFE V2 macro"
                    )
        else:
            next_prepared = self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=done
            )
        transition = ConstrainedV2Transition(
            chosen=decision.record,
            operational_reward=float(operational_reward),
            physical_rehandle_cost=float(physical_rehandle_cost),
            duration=int(duration),
            next_candidates=(
                () if next_prepared is None else tuple(next_prepared.records)
            ),
            done=bool(done),
            behavior_lambda=float(self.dual_lambda),
        )
        self.replay.add(transition)
        self.transition_count += 1
        return transition

    def _td_batch(
        self, transitions: Sequence[ConstrainedV2Transition]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        transitions = tuple(transitions)
        operational_prediction, rehandle_prediction = self._score_records(
            tuple(item.chosen for item in transitions),
            network=self.Q_local,
            grad=True,
        )
        next_operational = torch.zeros_like(operational_prediction)
        next_rehandle = torch.zeros_like(rehandle_prediction)
        flat_next = [
            record
            for transition in transitions
            if not transition.done
            for record in transition.next_candidates
        ]
        if flat_next:
            with torch.no_grad():
                online_op, online_cost = self._score_records(
                    flat_next, network=self.Q_local
                )
                target_op, target_cost = self._score_records(
                    flat_next, network=self.Q_target
                )
                offset = 0
                for index, transition in enumerate(transitions):
                    if transition.done:
                        continue
                    count = len(transition.next_candidates)
                    groups = torch.as_tensor(
                        [record.group_id for record in transition.next_candidates],
                        dtype=torch.long,
                        device=self.device,
                    )
                    elapsed_steps = {
                        record.elapsed_steps
                        for record in transition.next_candidates
                    }
                    if len(elapsed_steps) != 1:
                        raise ValueError(
                            "one replay frontier contains mixed elapsed-step contexts"
                        )
                    value_op, value_cost, _, _ = shared_lagrangian_policy_values(
                        online_op[offset : offset + count],
                        online_cost[offset : offset + count],
                        target_op[offset : offset + count],
                        target_cost[offset : offset + count],
                        groups,
                        self.within_group_temperatures,
                        self.group_temperature,
                        self.dual_lambda,
                        operational_weight=self._operational_weight(
                            elapsed_steps.pop()
                        ),
                    )
                    next_operational[index] = value_op
                    next_rehandle[index] = value_cost
                    offset += count
        rewards = torch.as_tensor(
            [item.operational_reward for item in transitions],
            dtype=operational_prediction.dtype,
            device=self.device,
        )
        costs = torch.as_tensor(
            [item.physical_rehandle_cost for item in transitions],
            dtype=rehandle_prediction.dtype,
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
        operational_target, rehandle_target = constrained_component_targets(
            rewards,
            costs,
            durations,
            dones,
            next_operational,
            next_rehandle,
            gamma_operational=self.config.gamma_operational,
            gamma_rehandle=self.config.gamma_rehandle,
            operational_reward_scale=self.config.operational_reward_scale,
        )
        return (
            operational_prediction,
            operational_target.detach(),
            rehandle_prediction,
            rehandle_target.detach(),
        )

    def learn(self) -> Optional[dict]:
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        transitions = self.replay.sample_group_balanced(
            self.config.batch_size, self.rng
        )
        op_prediction, op_target, cost_prediction, cost_target = self._td_batch(
            transitions
        )
        operational_loss = F.smooth_l1_loss(
            op_prediction, op_target, beta=self.config.huber_delta
        )
        rehandle_loss = F.smooth_l1_loss(
            cost_prediction, cost_target, beta=self.config.huber_delta
        )
        loss = operational_loss + self.config.cost_loss_weight * rehandle_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.Q_local.parameters(), self.config.grad_clip
        )
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())
            self.Q_target.requires_grad_(False)
            self.Q_target.eval()
            self.target_updates += 1
        values = {
            "total": float(loss.detach().item()),
            "operational": float(operational_loss.detach().item()),
            "rehandle": float(rehandle_loss.detach().item()),
            "gradient_norm": float(gradient_norm),
        }
        self.total_loss_history.append(values["total"])
        self.operational_loss_history.append(values["operational"])
        self.rehandle_loss_history.append(values["rehandle"])
        return values

    def checkpoint_metadata(self) -> dict:
        return {
            "checkpoint_schema_version": CONSTRAINED_V2_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_family": CONSTRAINED_V2_CHECKPOINT_FAMILY,
            "controller_architecture": CONSTRAINED_V2_CONTROLLER_ARCHITECTURE,
            "network_architecture": CONSTRAINED_V2_NETWORK_ARCHITECTURE,
            "replay_version": CONSTRAINED_V2_REPLAY_VERSION,
            "backup_version": CONSTRAINED_V2_BACKUP_VERSION,
            "policy_version": CONSTRAINED_V2_POLICY_VERSION,
            "candidate_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "hold_rule": CERTIFIED_HOLD_RULE_V2,
            "cost_definition": CONSTRAINED_V2_COST_DEFINITION,
            "time_context": CONSTRAINED_V2_TIME_CONTEXT,
            "control_groups": CONTROL_GROUP_NAMES,
            "group_mapping": dict(ACTION_TYPE_TO_GROUP),
            "action_types": tuple(item.value for item in ViabilityActionType),
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "action_feature_names": V2_ACTION_FEATURE_NAMES,
            "within_group_temperatures": self.within_group_temperature_values,
            "group_temperature": self.group_temperature,
            "reference_distribution": (
                "uniform_renormalized_within_live_group_and_over_live_groups_v2"
            ),
            "single_shared_lagrangian_policy_for_both_target_heads": True,
            "entropy_or_kl_added_to_raw_component_targets": False,
            "component_backup": "vector_expected_sarsa_double_q_v2",
            "operational_gamma": self.config.gamma_operational,
            "rehandle_gamma": self.config.gamma_rehandle,
            "operational_reward_scale": self.config.operational_reward_scale,
            "deployment_policy": (
                "regularized_group_map_then_lagrangian_candidate_map"
            ),
            "training_policy": "induced_nested_regularized_lagrangian_sample",
            "epsilon_exploration": False,
            "map_budget_guarantee": "empirical_only",
            "exact_safe_mask_authoritative": True,
            "unsafe_unknown_fail_closed": True,
            "exact_verifier_authority": EXACT_VERIFIER_AUTHORITY,
            "certification_contract": FAIL_CLOSED_CERTIFICATION_CONTRACT,
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "scalarized_reward_stored_in_replay": False,
            "cost_head_nonnegative": True,
        }

    def checkpoint_state(self, *, include_replay: bool = True) -> dict:
        state = {
            "Q_local": self.Q_local.state_dict(),
            "Q_target": self.Q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "dual_lambda": self.dual_lambda,
            "transition_count": self.transition_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "group_decisions": dict(self.group_decisions),
            "action_decisions": dict(self.action_decisions),
            "selection_sources": dict(self.selection_sources),
            "exact_rejections_seen": self.exact_rejections_seen,
            "interface_rejections": self.interface_rejections,
            "safe_candidates_scored": self.safe_candidates_scored,
            "operational_loss_history": tuple(self.operational_loss_history),
            "rehandle_loss_history": tuple(self.rehandle_loss_history),
            "total_loss_history": tuple(self.total_loss_history),
            "rng_state": self.rng.getstate(),
            "recovery_witness_guard": self.recovery_witness_guard.state_dict(),
        }
        if include_replay:
            state["replay"] = self.replay.state_dict()
        return state

    def load_checkpoint_state(self, state: Mapping, *, resumable: bool) -> None:
        self.Q_local.load_state_dict(state["Q_local"])
        self.Q_target.load_state_dict(state.get("Q_target", state["Q_local"]))
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.epsilon = self._validate_epsilon(state.get("epsilon", self.epsilon))
        self.dual_lambda = self._validate_lambda(state["dual_lambda"])
        self.transition_count = int(state.get("transition_count", 0))
        self.decision_count = int(state.get("decision_count", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.target_updates = int(state.get("target_updates", 0))
        self.group_decisions = Counter(state.get("group_decisions", {}))
        self.action_decisions = Counter(state.get("action_decisions", {}))
        self.selection_sources = Counter(state.get("selection_sources", {}))
        self.exact_rejections_seen = int(state.get("exact_rejections_seen", 0))
        self.interface_rejections = int(state.get("interface_rejections", 0))
        self.safe_candidates_scored = int(state.get("safe_candidates_scored", 0))
        self.operational_loss_history = list(
            state.get("operational_loss_history", ())
        )
        self.rehandle_loss_history = list(state.get("rehandle_loss_history", ()))
        self.total_loss_history = list(state.get("total_loss_history", ()))
        guard_state = state.get("recovery_witness_guard")
        if resumable and guard_state is not None:
            self.recovery_witness_guard.load_state_dict(guard_state)
        else:
            self.recovery_witness_guard.reset()
        if resumable:
            missing = [
                name
                for name in ("optimizer", "replay", "rng_state")
                if name not in state
            ]
            if missing:
                raise ValueError(f"resumable V2 checkpoint is missing {missing!r}")
            self.optimizer.load_state_dict(state["optimizer"])
            self.replay.load_state_dict(state["replay"])
            self.rng.setstate(state["rng_state"])

    def checkpoint(self, *, include_replay: bool = True, **metadata) -> dict:
        payload = {
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
        payload: Mapping,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "ConstrainedV2HierarchyAgent":
        expected = {
            "checkpoint_schema_version": CONSTRAINED_V2_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_family": CONSTRAINED_V2_CHECKPOINT_FAMILY,
            "controller_architecture": CONSTRAINED_V2_CONTROLLER_ARCHITECTURE,
            "network_architecture": CONSTRAINED_V2_NETWORK_ARCHITECTURE,
            "replay_version": CONSTRAINED_V2_REPLAY_VERSION,
            "backup_version": CONSTRAINED_V2_BACKUP_VERSION,
            "policy_version": CONSTRAINED_V2_POLICY_VERSION,
            "candidate_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "cost_definition": CONSTRAINED_V2_COST_DEFINITION,
            "action_feature_names": V2_ACTION_FEATURE_NAMES,
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "single_shared_lagrangian_policy_for_both_target_heads": True,
            "entropy_or_kl_added_to_raw_component_targets": False,
            "exact_safe_mask_authoritative": True,
            "baseline_teacher": False,
            "scalarized_reward_stored_in_replay": False,
        }
        mismatches = {
            name: (payload.get(name), required)
            for name, required in expected.items()
            if payload.get(name) != required
        }
        if mismatches:
            raise ValueError(f"incompatible constrained V2 checkpoint: {mismatches!r}")
        agent = cls(
            config=ConstrainedV2Config.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload["agent_state"].get("epsilon", 0.0)),
            dual_lambda=float(payload["agent_state"]["dual_lambda"]),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent

    def audit(self, *, include_decisions: bool = False) -> dict:
        result = {
            **self.checkpoint_metadata(),
            "epsilon": self.epsilon,
            "dual_lambda": self.dual_lambda,
            "transition_count": self.transition_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "replay_size": len(self.replay),
            "replay_group_counts": {
                ID_TO_CONTROL_GROUP[group_id]: count
                for group_id, count in self.replay.group_counts().items()
            },
            "group_decisions": dict(self.group_decisions),
            "action_decisions": dict(self.action_decisions),
            "selection_sources": dict(self.selection_sources),
            "recovery_witness_guard": self.recovery_witness_guard.audit_dict(),
        }
        if include_decisions:
            result["decisions"] = tuple(self.decision_log)
        return result


class ConstrainedV2SmokeRuntime:
    """Small, real environment adapter for the isolated development trainer.

    This is deliberately an integration smoke rather than the eventual proper
    trainer.  It executes only exact-certified bound options, records the two
    raw SMDP outcomes, and never queries a baseline or fallback controller.
    """

    def __init__(self, *, args, contract: Mapping) -> None:
        from train_vcg_constrained_v2 import (
            BACKUP_VERSION,
            CHECKPOINT_FAMILY,
            COST_DEFINITION,
            FROZEN_OBJECTIVE_SPEC,
        )
        from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec

        if not isinstance(contract, Mapping):
            raise TypeError("V2 smoke contract must be a mapping")
        if contract.get("checkpoint_family") != CHECKPOINT_FAMILY:
            raise ValueError("V2 smoke checkpoint family mismatch")
        if contract.get("backup_version") != BACKUP_VERSION:
            raise ValueError("V2 smoke backup contract mismatch")
        if contract.get("cost_definition") != COST_DEFINITION:
            raise ValueError("V2 smoke physical-cost contract mismatch")
        if contract.get("dense_objective_spec") != FROZEN_OBJECTIVE_SPEC.to_dict():
            raise ValueError("V2 smoke dense-objective contract mismatch")
        if float(contract.get("gamma_operational")) != 0.99:
            raise ValueError("V2 smoke requires gamma_operational=0.99")
        if float(contract.get("gamma_rehandle")) != 1.0:
            raise ValueError("V2 smoke requires raw finite-episode rehandle cost")
        if float(contract.get("reward_scale")) != 0.01:
            raise ValueError("V2 smoke requires operational reward scale 0.01")
        hold_contract = contract.get("certified_hold", {})
        if hold_contract.get("frontier_interface") != CERTIFIED_HOLD_INTERFACE_V2:
            raise ValueError("V2 smoke requires the certified Hold frontier")
        if hold_contract.get("rule") != CERTIFIED_HOLD_RULE_V2:
            raise ValueError("V2 smoke Hold rule mismatch")

        requested_device = str(contract.get("device", getattr(args, "device", "cpu")))
        if requested_device.lower() == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        environment = dict(contract["environment"])
        self.objective_spec = TimingObjectiveSpec.from_dict(
            contract["dense_objective_spec"]
        )
        self.env = ObjectiveAuditSmallRoomsEnv(
            timing_objective=self.objective_spec,
            grid_rows=int(environment["grid_rows"]),
            grid_cols=int(environment["grid_cols"]),
            number_blocks=int(environment["number_blocks"]),
            choose_storage=False,
            arrival_rate=float(environment["arrival_rate"]),
            proc_mean=float(environment["proc_mean"]),
        )
        dual = dict(contract["dual"])
        self.config = ConstrainedV2Config(
            gamma_operational=float(contract["gamma_operational"]),
            gamma_rehandle=float(contract["gamma_rehandle"]),
            operational_reward_scale=float(contract["reward_scale"]),
            episode_horizon_steps=int(contract["max_steps"]),
            max_hold_steps=int(hold_contract["max_option_steps"]),
            max_idle_steps=int(hold_contract["max_idle_steps"]),
            lambda_initial=float(dual["lambda_initial"]),
            lambda_max=float(dual["lambda_max"]),
            # A smoke should exercise learning on a small two-block episode.
            batch_size=16,
            replay_capacity=2_000,
            target_update_every=50,
        )
        self.agent = ConstrainedV2HierarchyAgent(
            config=self.config,
            seed=int(contract["model_seed"]),
            device=self.device,
            epsilon=0.0,
            dual_lambda=float(dual["lambda_initial"]),
        )
        self.search_config = ViabilitySearchConfig(
            max_depth=None,
            max_nodes=100_000,
            max_primitive_steps=None,
            reserve_queue_cells=True,
            search_order="goal_directed",
        )
        self.hold_rule = CertifiedHoldRuleV2(
            max_option_steps=self.config.max_hold_steps,
            max_idle_steps=self.config.max_idle_steps,
        )
        self.max_steps = int(contract["max_steps"])
        self._core_contract = {
            "controller": CONSTRAINED_V2_CONTROLLER_ARCHITECTURE,
            "checkpoint_family": CONSTRAINED_V2_CHECKPOINT_FAMILY,
            "backup_version": CONSTRAINED_V2_BACKUP_VERSION,
            "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "hold_rule": CERTIFIED_HOLD_RULE_V2,
            "cost_definition": CONSTRAINED_V2_COST_DEFINITION,
            "scalarized_reward_stored_in_replay": False,
            "training_policy": "induced_nested_regularized_lagrangian_sample",
            "deployment_policy": "nested_lagrangian_map",
            "mixed_discount_policy_merit": (
                "gamma_operational_pow_elapsed_times_qop_minus_lambda_qphys"
            ),
            "operational_gamma": self.config.gamma_operational,
            "rehandle_gamma": self.config.gamma_rehandle,
            "operational_reward_scale": self.config.operational_reward_scale,
            "dual_update_authority": "trainer_projected_episode_residual",
            "exact_safe_frontier_authoritative": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
        }

    @property
    def core_contract(self) -> Mapping:
        return dict(self._core_contract)

    def set_dual_lambda(self, value: float) -> None:
        self.agent.set_dual_lambda(value)

    @property
    def dual_lambda(self) -> float:
        return float(self.agent.dual_lambda)

    def _evaluation_agent(self, *, seed: int) -> ConstrainedV2HierarchyAgent:
        evaluation = ConstrainedV2HierarchyAgent(
            config=self.config,
            seed=int(seed),
            device=self.device,
            epsilon=0.0,
            dual_lambda=self.agent.dual_lambda,
        )
        evaluation.load_checkpoint_state(
            self.agent.checkpoint_state(include_replay=False),
            resumable=False,
        )
        evaluation.Q_local.eval()
        return evaluation

    @staticmethod
    def _failure(prefix: str, error: Exception) -> str:
        return f"{prefix}:{type(error).__name__}:{error}"

    @staticmethod
    def _option_failure(option) -> Optional[str]:
        outcome = getattr(option, "last_outcome", None)
        if isinstance(outcome, Mapping) and outcome.get("success") is False:
            return str(outcome.get("reason", "option_reported_failure"))
        if bool(getattr(option, "failed", False)):
            return str(getattr(option, "failure_reason", "option_failed"))
        return None

    @dataclass(frozen=True)
    class _MacroExecution:
        discounted_return: float
        raw_return: float
        duration: int
        env_terminal: bool
        truncated: bool
        option_success: bool
        failure_reason: Optional[str]
        delivery_deviations: tuple[float, ...]
        relocations: int
        illegal_drops: int
        action_type: str

    def _execute_macro(self, candidate, *, remaining_steps: int, evaluation: bool):
        """Execute one bound option with strict primitive event validation."""

        from train_vcg_constrained_v2 import primitive_physical_rehandle_cost

        if candidate.certificate.status is not ViabilityStatus.SAFE:
            raise ValueError("only exactly SAFE candidates may be executed")
        if remaining_steps <= 0:
            raise ValueError("remaining_steps must be positive")
        option = candidate.option
        try:
            initiable = bool(option.initiation(self.env.get_current_state()))
        except (RuntimeError, ValueError, TypeError) as error:
            return self._MacroExecution(
                0.0, 0.0, 0, False, False, False,
                self._failure("option_initiation", error), (), 0, 0,
                candidate.action_type.value,
            )
        if not initiable:
            return self._MacroExecution(
                0.0, 0.0, 0, False, False, False,
                "bound_option_not_initiable", (), 0, 0,
                candidate.action_type.value,
            )
        discounted = 0.0
        raw = 0.0
        duration = 0
        terminal = False
        terminated = False
        failure = None
        deviations = []
        relocations = 0
        illegal_drops = 0
        while duration < remaining_steps and not terminal and not terminated:
            try:
                action = option.policy(
                    self.env.get_current_state(), test=bool(evaluation)
                )
                next_state, reward, terminal, info = self.env.step(action)
                duration += 1
                reward = float(reward)
                if not math.isfinite(reward):
                    raise ValueError("environment reward is not finite")
                discounted += (
                    self.config.gamma_operational ** (duration - 1)
                ) * reward
                raw += reward
                relocations += primitive_physical_rehandle_cost(info)
                illegal_drops += int(bool(info.get("illegal_drop")))
                if "delivery_error_time" in info:
                    deviation = float(info["delivery_error_time"])
                    if not math.isfinite(deviation):
                        raise ValueError("delivery deviation is not finite")
                    deviations.append(deviation)
                terminated = bool(option.termination(next_state))
            except (RuntimeError, ValueError, TypeError) as error:
                failure = self._failure("macro_runtime", error)
                break
        truncated = bool(
            duration >= remaining_steps and not terminal and not terminated
        )
        failure = failure or self._option_failure(option)
        if terminal and not terminated and failure is None:
            failure = "environment_terminated_before_option_contract"
        if truncated and failure is None:
            failure = "step_limit_during_bound_macro"
        return self._MacroExecution(
            discounted_return=float(discounted),
            raw_return=float(raw),
            duration=int(duration),
            env_terminal=bool(terminal),
            truncated=truncated,
            option_success=bool(failure is None and (terminated or terminal)),
            failure_reason=failure,
            delivery_deviations=tuple(deviations),
            relocations=int(relocations),
            illegal_drops=int(illegal_drops),
            action_type=candidate.action_type.value,
        )

    def _enumerate_frontier(
        self,
        episode_agent: ConstrainedV2HierarchyAgent,
        idle_budget: PrimitiveIdleBudgetStateV2,
        *,
        remaining_steps: int,
        force_override: Optional[bool] = None,
        certificate_cache,
    ) -> CertifiedHoldCandidateSnapshotV2:
        forced = (
            episode_agent.recovery_witness_guard.active
            if force_override is None
            else bool(force_override)
        )
        return enumerate_viability_candidates_hold_v2(
            self.env,
            idle_budget=idle_budget,
            remaining_episode_steps=int(remaining_steps),
            recovery_witness_forced=forced,
            search_config=self.search_config,
            hold_rule=self.hold_rule,
            cache=certificate_cache,
        )

    def _authoritative_post_guard_frontier(
        self,
        episode_agent: ConstrainedV2HierarchyAgent,
        idle_budget: PrimitiveIdleBudgetStateV2,
        *,
        remaining_steps: int,
        certificate_cache,
    ) -> CertifiedHoldCandidateSnapshotV2:
        """Enumerate, trigger any due/nonprogress guard, then rebind Hold."""

        snapshot = self._enumerate_frontier(
            episode_agent,
            idle_budget,
            remaining_steps=remaining_steps,
            certificate_cache=certificate_cache,
        )
        active_before = episode_agent.recovery_witness_guard.active
        prepared, _ = episode_agent._admissible_prepared(snapshot)
        active_after = episode_agent.recovery_witness_guard.active
        if active_after != active_before:
            snapshot = self._enumerate_frontier(
                episode_agent,
                idle_budget,
                remaining_steps=remaining_steps,
                certificate_cache=certificate_cache,
            )
            prepared, _ = episode_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "post-guard frontier has no liveness-admissible SAFE macro"
            )
        return snapshot

    def run_episode(
        self, *, instance_seed: int, training: bool, max_steps: int
    ) -> Mapping:
        from example.helper.timing_metrics import summarize_delivery_timing
        from vcg_objective_audit import DENSE_PIECEWISE

        max_steps = int(max_steps)
        if max_steps != self.max_steps:
            raise ValueError("runtime max_steps differs from the frozen V2 contract")
        episode_agent = (
            self.agent
            if training
            else self._evaluation_agent(seed=int(instance_seed))
        )
        episode_agent.Q_local.train(bool(training))
        instance = self.env.sample_episode_instance(int(instance_seed))
        self.env.reset(instance=instance)
        episode_agent.reset_episode_state()
        idle_budget = PrimitiveIdleBudgetStateV2(
            max_idle_steps=self.config.max_idle_steps,
            idle_steps_since_progress=0,
        )
        certificate_cache = ViabilityCertificateCache()
        total_return = 0.0
        episode_start_discounted_return = 0.0
        steps = 0
        macros = 0
        physical_rehandles = 0
        illegal_drops = 0
        fallbacks = 0
        delivery_deviations: list[float] = []
        method_failure_reason = None
        pending_snapshot = None
        losses = []
        all_exact_safe = True
        witness_mismatches_before = episode_agent.recovery_witness_guard.mismatches
        selected_counts: Counter[str] = Counter()
        hold_outcomes: Counter[str] = Counter()

        while steps < max_steps and not self.env.is_state_terminal(
            self.env.current_state
        ):
            if pending_snapshot is None:
                try:
                    pending_snapshot = self._authoritative_post_guard_frontier(
                        episode_agent,
                        idle_budget,
                        remaining_steps=max_steps - steps,
                        certificate_cache=certificate_cache,
                    )
                except (RuntimeError, ValueError, TypeError) as error:
                    method_failure_reason = self._failure(
                        "frontier_construction", error
                    )
                    break
            snapshot = pending_snapshot
            pending_snapshot = None
            if not snapshot.candidates:
                method_failure_reason = "no_exact_safe_candidate"
                break
            try:
                decision = episode_agent.select(
                    snapshot,
                    training=bool(training),
                    epsilon=0.0,
                )
            except (NoCertifiedViableAction, RuntimeError, ValueError, TypeError) as error:
                method_failure_reason = self._failure("safe_selection", error)
                break
            all_exact_safe = bool(
                all_exact_safe
                and decision.candidate.certificate.status is ViabilityStatus.SAFE
            )
            selected_counts[decision.candidate.action_type.value] += 1
            steps_before = steps
            execution = self._execute_macro(
                decision.candidate,
                remaining_steps=max_steps - steps,
                evaluation=not training,
            )
            macros += 1
            steps += int(execution.duration)
            total_return += float(execution.raw_return)
            episode_start_discounted_return += (
                (self.config.gamma_operational ** steps_before)
                * float(execution.discounted_return)
            )
            physical_rehandles += int(execution.relocations)
            illegal_drops += int(execution.illegal_drops)
            delivery_deviations.extend(execution.delivery_deviations)

            observed_event = False
            if decision.candidate.action_type is ViabilityActionType.DEFER:
                outcome = getattr(decision.option, "last_outcome", None)
                if isinstance(outcome, Mapping):
                    hold_outcomes[str(outcome.get("reason"))] += 1
                    observed_event = outcome.get("reason") == "observed_event"

            time_limit = bool(
                steps >= max_steps
                and not self.env.is_state_terminal(self.env.current_state)
            )
            terminal_boundary = bool(
                execution.env_terminal
                or execution.truncated
                or not execution.option_success
                or time_limit
                or execution.duration <= 0
            )
            if execution.option_success and execution.duration > 0:
                try:
                    idle_budget = idle_budget.after_macro(
                        decision.candidate.action_type,
                        duration=int(execution.duration),
                        observed_event=observed_event,
                        exact_rank_progress=decision.exact_rank_progress,
                        recovery_witness_forced=decision.liveness_forced,
                    )
                except (RuntimeError, ValueError, TypeError) as error:
                    terminal_boundary = True
                    method_failure_reason = self._failure("idle_budget", error)

            outcome_observed = False
            if execution.duration > 0:
                try:
                    if terminal_boundary:
                        episode_agent.observe_outcome(
                            decision,
                            next_snapshot=None,
                            done=True,
                            prepare_next=False,
                        )
                    else:
                        # The probe authenticates the physical successor for a
                        # retained witness without imposing the *old* guard on
                        # Hold.  Only after the guard advances do we construct
                        # the authoritative Bellman/deployment frontier.
                        probe_snapshot = self._enumerate_frontier(
                            episode_agent,
                            idle_budget,
                            remaining_steps=max_steps - steps,
                            force_override=False,
                            certificate_cache=certificate_cache,
                        )
                        episode_agent.observe_outcome(
                            decision,
                            next_snapshot=probe_snapshot,
                            done=False,
                            prepare_next=False,
                        )
                        pending_snapshot = self._authoritative_post_guard_frontier(
                            episode_agent,
                            idle_budget,
                            remaining_steps=max_steps - steps,
                            certificate_cache=certificate_cache,
                        )
                    outcome_observed = True
                except (NoCertifiedViableAction, RuntimeError, ValueError, TypeError) as error:
                    terminal_boundary = True
                    pending_snapshot = None
                    outcome_observed = True
                    method_failure_reason = self._failure(
                        "liveness_outcome", error
                    )

            if training and execution.duration > 0:
                try:
                    episode_agent.remember(
                        decision,
                        operational_reward=execution.discounted_return,
                        physical_rehandle_cost=int(execution.relocations),
                        duration=int(execution.duration),
                        next_snapshot=(
                            None if terminal_boundary else pending_snapshot
                        ),
                        done=terminal_boundary,
                        outcome_already_observed=outcome_observed,
                    )
                    learned = episode_agent.learn()
                    if learned is not None:
                        losses.append(dict(learned))
                except (NoCertifiedViableAction, RuntimeError, ValueError, TypeError) as error:
                    terminal_boundary = True
                    pending_snapshot = None
                    method_failure_reason = self._failure("learning", error)

            if not execution.option_success and method_failure_reason is None:
                method_failure_reason = (
                    f"macro_failure:{execution.action_type}:"
                    f"{execution.failure_reason or 'unknown'}"
                )
            if time_limit and method_failure_reason is None:
                method_failure_reason = "episode_step_limit"
            if terminal_boundary:
                break

        terminal = bool(self.env.is_state_terminal(self.env.current_state))
        if not terminal and method_failure_reason is None:
            method_failure_reason = "episode_step_limit"
        required_deliveries = int(instance.number_blocks)
        delivery_count = len(delivery_deviations)
        if terminal and delivery_count != required_deliveries and method_failure_reason is None:
            method_failure_reason = "terminal_incomplete_required_workload"
        timing = summarize_delivery_timing(
            delivery_deviations, self.objective_spec.window
        )
        mean_absolute_error = (
            None
            if not delivery_deviations
            else float(timing["mean_absolute_error"])
        )
        objective_audit = self.env.objective_audit_summary()
        audited_return = float(
            objective_audit["return_by_objective"][DENSE_PIECEWISE]
        )
        if not math.isclose(
            total_return, audited_return, rel_tol=0.0, abs_tol=1.0e-9
        ):
            method_failure_reason = "dense_objective_audit_mismatch"
        if int(objective_audit["delivery_count"]) != delivery_count:
            method_failure_reason = "delivery_count_audit_mismatch"
        witness_mismatches = (
            episode_agent.recovery_witness_guard.mismatches
            - witness_mismatches_before
        )
        success = bool(
            terminal
            and delivery_count == required_deliveries
            and method_failure_reason is None
        )
        strict = bool(
            success
            and illegal_drops == 0
            and fallbacks == 0
            and witness_mismatches == 0
            and all_exact_safe
        )
        return {
            "method_version": "vcg_constrained_v2",
            "split": "training" if training else "development_validation",
            "instance_seed": int(instance_seed),
            "episode_instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "strict_method_success": strict,
            "success": success,
            "completion_rate": delivery_count / required_deliveries,
            "physical_rehandles": int(physical_rehandles),
            "physical_storage_relocations": int(physical_rehandles),
            "required_deliveries": required_deliveries,
            "delivery_count": delivery_count,
            "dense_return": float(total_return),
            "episode_start_discounted_dense_return": float(
                episode_start_discounted_return
            ),
            "mean_absolute_error": mean_absolute_error,
            "delivery_deviations": tuple(float(x) for x in delivery_deviations),
            "steps": int(steps),
            "macro_decisions": int(macros),
            "method_failure_reason": method_failure_reason,
            "illegal_drops": int(illegal_drops),
            "fallbacks": int(fallbacks),
            "witness_mismatches": int(witness_mismatches),
            "all_selected_candidates_exact_safe": bool(all_exact_safe),
            "selected_action_counts": dict(selected_counts),
            "hold_outcome_counts": dict(hold_outcomes),
            "loss_updates": len(losses),
            "last_loss": None if not losses else losses[-1],
            "dual_lambda": float(episode_agent.dual_lambda),
            "certificate_cache_entries": len(certificate_cache),
            "baseline_teacher": False,
            "baseline_policy_query": False,
        }

    def checkpoint_state(self, *, include_replay: bool) -> Mapping:
        return self.agent.checkpoint(include_replay=bool(include_replay))


def build_smoke_runtime(*, args, contract: Mapping) -> ConstrainedV2SmokeRuntime:
    """Trainer integration hook; no V1 runtime fallback is permitted."""

    return ConstrainedV2SmokeRuntime(args=args, contract=contract)
