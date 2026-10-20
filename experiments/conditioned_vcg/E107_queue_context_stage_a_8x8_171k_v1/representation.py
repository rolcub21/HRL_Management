"""Online-visible queue context for the E107 Stage-A critic.

The certified frontier and graph representation remain unchanged.  E107 adds
one bounded public context vector to Q(s, a), and retains that vector inside
the immutable current state stored in replay.  No future arrival time,
duration, or unreleased job attribute is read.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math
from typing import Optional, Sequence
from unittest.mock import patch

import torch
from torch import Tensor, nn

from PSLAP.viability import RecoveryState
from PSLAP.yard_graph import NODE_FEATURE_NAMES
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ACTION_FEATURE_NAMES,
    ViabilityGraphCandidateRecord,
    ViabilityGraphQNetwork,
)
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import (
    acceleration as v3_acceleration,
)


CONTROLLER_ARCHITECTURE = "e107_exact_safe_queue_observed_graph_smdp_q_v1"
NETWORK_ARCHITECTURE = "e107_current_public_queue_context_graph_q_v1"
TIME_CONTEXT_CONTRACT = (
    "bounded_current_boundary_public_queue_and_progress_context_no_future_schedule_v1"
)
QUEUE_CONTEXT_NAMES = (
    "waiting_fraction",
    "oldest_wait_age_fraction_of_horizon",
    "inbound_wait_age_fraction_of_horizon",
    "over_cap_fraction",
    "current_excess_wait_fraction_of_maximum",
    "accepted_fraction",
    "delivered_fraction",
    "elapsed_fraction",
)
QUEUE_CONTEXT_DIM = len(QUEUE_CONTEXT_NAMES)


@dataclass(frozen=True)
class PublicQueueContext:
    values: tuple[float, ...]
    waiting_jobs: int
    oldest_wait_age: int
    inbound_wait_age: Optional[int]
    over_cap_jobs: int
    accepted_jobs: int
    delivered_jobs: int
    time_step: int

    def __post_init__(self) -> None:
        if len(self.values) != QUEUE_CONTEXT_DIM:
            raise ValueError("public queue context has the wrong width")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0
                   for value in self.values):
            raise ValueError("public queue context must be finite and bounded")


def public_queue_context(
    env,
    snapshot,
    *,
    wait_cap: int,
    required_jobs: int,
    max_steps: int,
) -> PublicQueueContext:
    """Build context from facts observable at the current decision boundary."""

    if wait_cap <= 0 or required_jobs <= 0 or max_steps <= wait_cap:
        raise ValueError("invalid queue-context normalization contract")
    time_step = int(env.time_steps)
    if int(snapshot.decision_epoch) != time_step:
        raise ValueError("snapshot and live queue context are from different times")
    blocks = tuple(env.blocks)
    if len(blocks) != required_jobs:
        raise ValueError("environment workload differs from queue-context contract")

    # The arrival comparison is the information boundary: no property other
    # than arrival_step is read for an unreleased job.
    released = tuple(
        block for block in blocks if int(block.arrival_step) <= time_step
    )
    outside = tuple(
        block
        for block in released
        if block.stored_time_step is None and not bool(block.delivered)
    )
    ages = {
        str(block.label): max(0, time_step - int(block.arrival_step))
        for block in outside
    }
    oldest = max(ages.values(), default=0)
    inbound_age = ages.get(str(snapshot.inbound_label))
    over_cap = sum(age > wait_cap for age in ages.values())
    excess = sum(max(0, age - wait_cap) for age in ages.values())
    accepted = sum(block.stored_time_step is not None for block in released)
    delivered = sum(bool(block.delivered) for block in released)

    jobs = float(required_jobs)
    maximum_excess = float((max_steps - wait_cap) * required_jobs)
    values = (
        len(outside) / jobs,
        min(oldest / max_steps, 1.0),
        0.0 if inbound_age is None else min(inbound_age / max_steps, 1.0),
        over_cap / jobs,
        min(excess / maximum_excess, 1.0),
        accepted / jobs,
        delivered / jobs,
        min(time_step / max_steps, 1.0),
    )
    return PublicQueueContext(
        values=tuple(float(value) for value in values),
        waiting_jobs=len(outside),
        oldest_wait_age=oldest,
        inbound_wait_age=inbound_age,
        over_cap_jobs=over_cap,
        accepted_jobs=accepted,
        delivered_jobs=delivered,
        time_step=time_step,
    )


@dataclass(frozen=True)
class QueueObservedRecoveryState(RecoveryState):
    """Recovery state plus the public context at its current boundary."""

    queue_context: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.queue_context) != QUEUE_CONTEXT_DIM:
            raise ValueError("queue-observed recovery state has wrong context width")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0
                   for value in self.queue_context):
            raise ValueError("queue-observed context must be finite and bounded")


def attach_queue_context(
    state: RecoveryState,
    context: Sequence[float],
) -> QueueObservedRecoveryState:
    values = tuple(float(value) for value in context)
    return QueueObservedRecoveryState(
        **{field.name: getattr(state, field.name) for field in fields(RecoveryState)},
        queue_context=values,
    )


class QueueContextQNetwork(ViabilityGraphQNetwork):
    """Native graph/action Q network conditioned on current public workload."""

    def __init__(self, config) -> None:
        super().__init__(config)
        context_width = config.action_embedding_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(QUEUE_CONTEXT_DIM, context_width),
            nn.SiLU(),
            nn.LayerNorm(context_width),
        )
        self.q_head = nn.Sequential(
            nn.Linear(
                3 * config.graph_embedding_dim
                + config.action_embedding_dim
                + context_width,
                config.head_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.head_hidden_dim, 1),
        )

    def context_tensor(self, current_states: Sequence[RecoveryState]) -> Tensor:
        values = []
        for state in current_states:
            if not isinstance(state, QueueObservedRecoveryState):
                raise TypeError("E107 current state lacks replayed queue context")
            values.append(state.queue_context)
        parameter = next(self.parameters())
        result = torch.as_tensor(
            values, dtype=parameter.dtype, device=parameter.device
        )
        if result.shape != (len(values), QUEUE_CONTEXT_DIM):
            raise ValueError("E107 queue context tensor has the wrong shape")
        return result

    def score_embeddings(
        self,
        current_embedding: Tensor,
        successor_embedding: Tensor,
        action_features,
        queue_context: Tensor,
    ) -> Tensor:
        parameter = next(self.parameters())
        action = torch.as_tensor(
            action_features, dtype=parameter.dtype, device=parameter.device
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)
        count = current_embedding.shape[0]
        if action.shape != (count, ACTION_FEATURE_DIM):
            raise ValueError("E107 action feature matrix has the wrong shape")
        if queue_context.shape != (count, QUEUE_CONTEXT_DIM):
            raise ValueError("E107 queue context matrix has the wrong shape")
        return self.q_head(
            torch.cat(
                (
                    current_embedding,
                    successor_embedding,
                    successor_embedding - current_embedding,
                    self.action_encoder(action),
                    self.context_encoder(queue_context),
                ),
                dim=-1,
            )
        ).squeeze(-1)

    def forward(self, current_states, successor_states, action_features) -> Tensor:
        current = tuple(current_states)
        successor = tuple(successor_states)
        if not current or len(current) != len(successor):
            raise ValueError("E107 candidate state counts must be nonempty and equal")
        if not all(isinstance(state, RecoveryState) for state in current + successor):
            raise TypeError("E107 candidate states must be RecoveryState instances")
        embeddings = fast_inputs._embed(self, current + successor)
        count = len(current)
        return self.score_embeddings(
            embeddings[:count],
            embeddings[count:],
            action_features,
            self.context_tensor(current),
        )


class QueueContextAgent(EpisodicViabilityGraphHierarchyAgent):
    """Exact-safe episodic agent whose replay retains public queue context."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = QueueContextQNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=self.config.learning_rate
        )
        self.context_environment = None
        self.context_wait_cap = 80
        self.context_required_jobs = 36
        self.context_max_steps = self.config.episode_horizon_steps

    def bind_context_environment(
        self,
        env,
        *,
        wait_cap: int,
        required_jobs: int,
        max_steps: int,
    ) -> None:
        self.context_environment = env
        self.context_wait_cap = int(wait_cap)
        self.context_required_jobs = int(required_jobs)
        self.context_max_steps = int(max_steps)

    def _admissible_prepared(self, snapshot):
        prepared, forced = super()._admissible_prepared(snapshot)
        if self.context_environment is None:
            raise RuntimeError("E107 agent is not bound to its live environment")
        context = public_queue_context(
            self.context_environment,
            snapshot,
            wait_cap=self.context_wait_cap,
            required_jobs=self.context_required_jobs,
            max_steps=self.context_max_steps,
        )
        current = attach_queue_context(
            snapshot.recovery_state, context.values
        )
        records = tuple(
            replace(record, current_state=current)
            for record in prepared.records
        )
        return replace(prepared, records=records), forced

    def checkpoint_metadata(self) -> dict:
        result = super().checkpoint_metadata()
        result.update(
            controller_architecture=CONTROLLER_ARCHITECTURE,
            network_architecture=NETWORK_ARCHITECTURE,
            time_context_contract=TIME_CONTEXT_CONTRACT,
            remaining_budget_observed_by_q=True,
            elapsed_time_observed_by_q=True,
            representation_difference_from_vcg_v1=(
                "bounded current public queue/progress context"
            ),
            queue_context_names=QUEUE_CONTEXT_NAMES,
            queue_context_dim=QUEUE_CONTEXT_DIM,
            queue_context_stored_in_replay=True,
            q_context_composition=(
                "current_graph_successor_graph_delta_action_current_queue_context_v1"
            ),
        )
        return result

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "QueueContextAgent":
        expected = {
            "checkpoint_schema_version": (
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION
            ),
            "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "time_context_contract": TIME_CONTEXT_CONTRACT,
            "queue_context_names": QUEUE_CONTEXT_NAMES,
            "queue_context_dim": QUEUE_CONTEXT_DIM,
            "queue_context_stored_in_replay": True,
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatches:
            raise ValueError(f"incompatible E107 checkpoint: {mismatches!r}")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E107 checkpoint exposes the future schedule")
        agent = cls(
            config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


def deduplicated_queue_scores(records, network, audit=None) -> Tensor:
    """Exact no-gradient scoring with state-identity graph reuse."""

    if torch.is_grad_enabled():
        raise RuntimeError("accelerated E107 scoring requires no_grad")
    records = tuple(records)
    if not records:
        raise ValueError("cannot score an empty E107 frontier")
    if not isinstance(network, QueueContextQNetwork):
        raise TypeError("E107 accelerator received another network family")
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
    parameter = next(network.parameters())
    embeddings = fast_inputs._embed(network, unique).index_select(
        0,
        torch.as_tensor(inverse, dtype=torch.long, device=parameter.device),
    )
    count = len(records)
    contexts = network.context_tensor(
        tuple(record.current_state for record in records)
    )
    scores = network.score_embeddings(
        embeddings[:count],
        embeddings[count:],
        tuple(record.action_features for record in records),
        contexts,
    )
    if audit is not None:
        audit["calls"] = int(audit.get("calls", 0)) + 1
        audit["state_slots"] = int(audit.get("state_slots", 0)) + len(states)
        audit["unique_state_identities"] = int(
            audit.get("unique_state_identities", 0)
        ) + len(unique)
    return scores


@contextmanager
def acceleration_enabled():
    """Route both online and frozen-target continuation scoring through E107."""

    with patch.object(
        v3_acceleration,
        "_deduplicated_no_grad_scores",
        deduplicated_queue_scores,
    ):
        yield


__all__ = [
    "CONTROLLER_ARCHITECTURE",
    "NETWORK_ARCHITECTURE",
    "PublicQueueContext",
    "QUEUE_CONTEXT_DIM",
    "QUEUE_CONTEXT_NAMES",
    "QueueContextAgent",
    "QueueContextQNetwork",
    "QueueObservedRecoveryState",
    "TIME_CONTEXT_CONTRACT",
    "acceleration_enabled",
    "attach_queue_context",
    "deduplicated_queue_scores",
    "public_queue_context",
]
