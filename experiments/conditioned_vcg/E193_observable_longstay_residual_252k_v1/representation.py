"""Observable, long-stay-scaled representation for E193.

This module is intentionally isolated from E191.  It retains E191's graph and
action inputs, but replaces its eight-field short-stay queue context with ten
bounded fields whose time units match the Poisson-700 workload.  The two new
fields contain observed arrival history only; neither a workload label nor a
future release time is exposed to the policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from PSLAP.viability import RecoveryState
from viability_graph_hierarchy import ACTION_FEATURE_DIM
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191,
)

from .defer import observed_labels


CONTROLLER_ARCHITECTURE = (
    "e193_observable_longstay_choice_preserving_progress_guard_v1"
)
NETWORK_ARCHITECTURE = (
    "e193_e191_graph_observed_arrival_longstay_context_scalar_q_v1"
)
TIME_CONTEXT_CONTRACT = (
    "bounded_observed_arrival_history_longstay_scaled_no_future_schedule_v1"
)

# These are units conversions, not learned workload identifiers.
REQUIRED_JOBS = 36
DURATION_SCALE_STEPS = 700
EPISODE_HORIZON_STEPS = 9_000
ARRIVAL_WINDOW_STEPS = 200
FASTEST_ARRIVAL_GAP_STEPS = 25
SLOWEST_ARRIVAL_GAP_STEPS = 100
MAX_OBSERVED_ARRIVAL_RATE = 1.0 / FASTEST_ARRIVAL_GAP_STEPS
TIMING_SCALE = float(DURATION_SCALE_STEPS)

QUEUE_CONTEXT_NAMES = (
    "waiting_fraction",
    "oldest_wait_age_fraction_of_mean_stay",
    "inbound_wait_age_fraction_of_mean_stay",
    "total_wait_age_fraction_of_mean_stay_capacity",
    "released_fraction",
    "accepted_fraction",
    "delivered_fraction",
    "elapsed_fraction_of_episode_horizon",
    "recent_observed_arrival_rate_fraction_of_maximum",
    "time_since_last_observed_arrival_fraction_of_max_gap",
)
QUEUE_CONTEXT_DIM = len(QUEUE_CONTEXT_NAMES)
if QUEUE_CONTEXT_DIM != 10:
    raise RuntimeError("E193 public context width changed")


def _unit_interval(value: float) -> float:
    """Return a finite value clipped to the representation's public range."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("E193 context value must be finite")
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True)
class PublicLongStayContext:
    """Current public queue/progress facts and observed arrival history."""

    values: tuple[float, ...]
    waiting_jobs: int
    oldest_wait_age: int
    inbound_wait_age: int | None
    total_wait_age: int
    released_jobs: int
    accepted_jobs: int
    delivered_jobs: int
    observed_arrival_times: tuple[int, ...]
    recent_observed_arrivals: int
    time_since_last_observed_arrival: int
    time_step: int

    def __post_init__(self) -> None:
        if len(self.values) != QUEUE_CONTEXT_DIM:
            raise ValueError("E193 public context has the wrong width")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.values
        ):
            raise ValueError("E193 public context must be finite and bounded")
        if any(value > self.time_step for value in self.observed_arrival_times):
            raise ValueError("E193 context contains a future arrival")


def public_longstay_context(
    env,
    snapshot,
    *,
    required_jobs: int = REQUIRED_JOBS,
    max_steps: int = EPISODE_HORIZON_STEPS,
) -> PublicLongStayContext:
    """Build the E193 context without reading unreleased job attributes.

    Public inventory status is the information boundary.  Only after a job is
    visible at a position, carried, stored, or delivered do we read its
    release timestamp as observed history.  In particular, this function does
    not even compare an unreleased job's ``arrival_step`` with the clock.
    """

    if required_jobs <= 0 or max_steps <= 0:
        raise ValueError("invalid E193 context normalization contract")
    time_step = int(env.time_steps)
    if int(snapshot.decision_epoch) != time_step:
        raise ValueError("snapshot and E193 context are from different times")
    blocks = tuple(env.blocks)
    if len(blocks) != int(required_jobs):
        raise ValueError("E193 workload differs from its context contract")

    visible_labels = observed_labels(env)
    released = tuple(
        block for block in blocks if str(block.label) in visible_labels
    )
    observed_arrival_times = tuple(
        sorted(int(block.arrival_step) for block in released)
    )
    outside = tuple(
        block
        for block in released
        if getattr(block, "stored_time_step", None) is None
        and not bool(block.delivered)
    )
    ages = {
        str(block.label): max(0, time_step - int(block.arrival_step))
        for block in outside
    }
    oldest = max(ages.values(), default=0)
    inbound_age = ages.get(str(snapshot.inbound_label))
    total_age = sum(ages.values())
    accepted = sum(
        getattr(block, "stored_time_step", None) is not None
        for block in released
    )
    delivered = sum(bool(block.delivered) for block in released)

    recent_count = sum(
        arrival > time_step - ARRIVAL_WINDOW_STEPS
        for arrival in observed_arrival_times
    )
    recent_rate = recent_count / float(ARRIVAL_WINDOW_STEPS)
    since_last = (
        time_step - observed_arrival_times[-1]
        if observed_arrival_times
        else time_step
    )

    jobs = float(required_jobs)
    values = (
        len(outside) / jobs,
        _unit_interval(oldest / float(DURATION_SCALE_STEPS)),
        _unit_interval(
            0.0
            if inbound_age is None
            else inbound_age / float(DURATION_SCALE_STEPS)
        ),
        _unit_interval(
            total_age / float(DURATION_SCALE_STEPS * required_jobs)
        ),
        len(released) / jobs,
        accepted / jobs,
        delivered / jobs,
        _unit_interval(time_step / float(max_steps)),
        _unit_interval(recent_rate / MAX_OBSERVED_ARRIVAL_RATE),
        _unit_interval(
            since_last / float(SLOWEST_ARRIVAL_GAP_STEPS)
        ),
    )
    return PublicLongStayContext(
        values=tuple(float(value) for value in values),
        waiting_jobs=len(outside),
        oldest_wait_age=int(oldest),
        inbound_wait_age=(None if inbound_age is None else int(inbound_age)),
        total_wait_age=int(total_age),
        released_jobs=len(released),
        accepted_jobs=int(accepted),
        delivered_jobs=int(delivered),
        observed_arrival_times=observed_arrival_times,
        recent_observed_arrivals=int(recent_count),
        time_since_last_observed_arrival=int(since_last),
        time_step=time_step,
    )


@dataclass(frozen=True)
class ObservableLongStayState(RecoveryState):
    """Recovery state carrying E193's public context into replay."""

    queue_context: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.queue_context) != QUEUE_CONTEXT_DIM:
            raise ValueError("E193 recovery state has the wrong context width")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.queue_context
        ):
            raise ValueError("E193 recovery context must be finite and bounded")


def attach_longstay_context(
    state: RecoveryState,
    context: Sequence[float],
) -> ObservableLongStayState:
    values = tuple(float(value) for value in context)
    return ObservableLongStayState(
        **{
            field.name: getattr(state, field.name)
            for field in fields(RecoveryState)
        },
        queue_context=values,
    )


class ObservableLongStayQNetwork(e107.QueueContextQNetwork):
    """E191 graph/action Q network with E193's ten public inputs."""

    def __init__(self, config) -> None:
        super().__init__(config)
        context_width = config.action_embedding_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(QUEUE_CONTEXT_DIM, context_width),
            nn.SiLU(),
            nn.LayerNorm(context_width),
        )

    def context_tensor(
        self,
        current_states: Sequence[RecoveryState],
    ) -> Tensor:
        values = []
        for state in current_states:
            if not isinstance(state, ObservableLongStayState):
                raise TypeError("E193 current state lacks replayed public context")
            values.append(state.queue_context)
        parameter = next(self.parameters())
        result = torch.as_tensor(
            values, dtype=parameter.dtype, device=parameter.device
        )
        if result.shape != (len(values), QUEUE_CONTEXT_DIM):
            raise ValueError("E193 public context tensor has the wrong shape")
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
            raise ValueError("E193 action feature matrix has the wrong shape")
        if queue_context.shape != (count, QUEUE_CONTEXT_DIM):
            raise ValueError("E193 public context matrix has the wrong shape")
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


class ObservableLongStayAgent(e191.FreshChoicePreservingQueueAgent):
    """Fresh E191 controller with observable, correctly scaled context."""

    # A residual-value subclass may change its network identity while keeping
    # this representation/checkpoint loader.  Using class attributes here
    # prevents the base loader from silently accepting the wrong value head.
    CONTROLLER_ARCHITECTURE_ID = CONTROLLER_ARCHITECTURE
    NETWORK_ARCHITECTURE_ID = NETWORK_ARCHITECTURE

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not math.isclose(
            float(self.config.timing_scale), TIMING_SCALE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"E193 requires timing_scale={TIMING_SCALE:g}, got "
                f"{self.config.timing_scale!r}"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = ObservableLongStayQNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=self.config.learning_rate
        )

    def _decorate_observed_records(self, snapshot, prepared):
        """Extension seam for E193's exact-known-cost residual layer."""

        return prepared

    def _admissible_prepared(self, snapshot):
        prepared, forced = self.recovery_witness_guard.admissible_prepared(snapshot)
        if self.context_environment is None:
            raise RuntimeError("E193 agent is not bound to its live environment")
        context = public_longstay_context(
            self.context_environment,
            snapshot,
            required_jobs=self.context_required_jobs,
            max_steps=self.context_max_steps,
        )
        current = attach_longstay_context(
            snapshot.recovery_state, context.values
        )
        records = tuple(
            replace(record, current_state=current)
            for record in prepared.records
        )
        observed = replace(prepared, records=records)
        return self._decorate_observed_records(snapshot, observed), forced

    def checkpoint_metadata(self) -> dict:
        result = super().checkpoint_metadata()
        result.update(
            controller_architecture=self.CONTROLLER_ARCHITECTURE_ID,
            network_architecture=self.NETWORK_ARCHITECTURE_ID,
            time_context_contract=TIME_CONTEXT_CONTRACT,
            queue_context_names=QUEUE_CONTEXT_NAMES,
            queue_context_dim=QUEUE_CONTEXT_DIM,
            queue_context_stored_in_replay=True,
            q_context_composition=(
                "current_graph_successor_graph_delta_action_"
                "observed_longstay_context_v1"
            ),
            timing_scale=TIMING_SCALE,
            e188_only_method_delta_from_e187=False,
            e191_architecture_identical_to_e189=False,
            e193_fresh_random_initialization=True,
            e193_observed_arrival_history=True,
            e193_future_schedule_visible_to_policy=False,
            e193_longstay_time_normalization=True,
            e193_representation_version=NETWORK_ARCHITECTURE,
            representation_difference_from_vcg_v1=(
                "ten-field observed-arrival and long-stay-scaled public context"
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
    ) -> "ObservableLongStayAgent":
        expected = {
            "checkpoint_schema_version": (
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION
            ),
            "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
            "controller_architecture": cls.CONTROLLER_ARCHITECTURE_ID,
            "network_architecture": cls.NETWORK_ARCHITECTURE_ID,
            "time_context_contract": TIME_CONTEXT_CONTRACT,
            "queue_context_names": QUEUE_CONTEXT_NAMES,
            "queue_context_dim": QUEUE_CONTEXT_DIM,
            "queue_context_stored_in_replay": True,
            "e193_fresh_random_initialization": True,
            "e193_observed_arrival_history": True,
            "e193_future_schedule_visible_to_policy": False,
            "e193_longstay_time_normalization": True,
            "e193_representation_version": NETWORK_ARCHITECTURE,
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(f"incompatible E193 checkpoint: {mismatches!r}")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E193 checkpoint exposes the future schedule")
        if payload.get("e187_warm_start_manifest") is not None:
            raise ValueError("E193 checkpoint contains transferred weights")
        config = EpisodicViabilityGraphConfig.from_dict(payload["config"])
        if not math.isclose(
            float(config.timing_scale), TIMING_SCALE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("E193 checkpoint has another timing scale")
        agent = cls(
            config=config,
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


def with_longstay_timing_scale(
    config: EpisodicViabilityGraphConfig,
) -> EpisodicViabilityGraphConfig:
    """Return the authenticated E193 graph config without mutating its caller."""

    if not isinstance(config, EpisodicViabilityGraphConfig):
        raise TypeError("E193 requires an EpisodicViabilityGraphConfig")
    return replace(config, timing_scale=TIMING_SCALE)


def architecture_signature(agent) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(int(value) for value in tensor.shape)
        for name, tensor in agent.Q_local.state_dict().items()
    }


# The E107 scorer dispatches through the network's overridden context methods,
# so its exact state-identity graph reuse remains valid for this subclass.
deduplicated_residual_scores = e107.deduplicated_queue_scores
deduplicated_queue_scores = deduplicated_residual_scores
acceleration_enabled = e107.acceleration_enabled
attach_queue_context = attach_longstay_context
public_queue_context = public_longstay_context


__all__ = [
    "ARRIVAL_WINDOW_STEPS",
    "CONTROLLER_ARCHITECTURE",
    "DURATION_SCALE_STEPS",
    "EPISODE_HORIZON_STEPS",
    "FASTEST_ARRIVAL_GAP_STEPS",
    "MAX_OBSERVED_ARRIVAL_RATE",
    "NETWORK_ARCHITECTURE",
    "ObservableLongStayAgent",
    "ObservableLongStayQNetwork",
    "ObservableLongStayState",
    "PublicLongStayContext",
    "QUEUE_CONTEXT_DIM",
    "QUEUE_CONTEXT_NAMES",
    "REQUIRED_JOBS",
    "SLOWEST_ARRIVAL_GAP_STEPS",
    "TIME_CONTEXT_CONTRACT",
    "TIMING_SCALE",
    "acceleration_enabled",
    "architecture_signature",
    "attach_longstay_context",
    "attach_queue_context",
    "deduplicated_queue_scores",
    "deduplicated_residual_scores",
    "public_longstay_context",
    "public_queue_context",
    "with_longstay_timing_scale",
]
