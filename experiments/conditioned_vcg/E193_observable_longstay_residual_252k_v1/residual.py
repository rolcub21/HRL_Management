"""Observable nominal-cost residual critic for E193.

The neural network predicts only the part of action value that is not already
known at the decision boundary.  Every policy and TD path nevertheless sees
the same *total* value

    Q_total(s, a) = K_observed(s, a) + Q_residual(s, a),

where ``K_observed`` is the scaled negative objective cost of the nominal
macro under currently observable jobs.  Arrivals that occur while a macro is
executing are deliberately absent from K and remain a TD residual; consulting
their timestamps here would leak the future schedule.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
import math
from typing import Sequence
from unittest.mock import patch

import torch
from torch import Tensor

from viability_graph_hierarchy import ViabilityGraphCandidateRecord
from experiments.conditioned_vcg.E180_wait_cost_unmasked_pilot_239k_v1.objective import (
    nominal_components,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1.objective import (
    COMPONENT_NAMES,
    scalar_cost,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import (
    acceleration as v3_acceleration,
)

from . import representation


RESIDUAL_VALUE_CONTRACT = (
    "observable_nominal_five_component_K_plus_scalar_residual_q_v1"
)
CONTROLLER_ARCHITECTURE = (
    "e193_observable_longstay_choice_preserving_nominal_residual_v1"
)
NETWORK_ARCHITECTURE = (
    "e193_observable_longstay_context_scalar_nominal_cost_residual_q_v1"
)


@dataclass(frozen=True)
class ObservableCostRecord(ViabilityGraphCandidateRecord):
    """Replay-safe candidate plus its decision-time observable nominal cost."""

    nominal_components: tuple[float, ...]
    nominal_duration: int

    def __post_init__(self) -> None:
        super().__post_init__()
        values = tuple(float(value) for value in self.nominal_components)
        if len(values) != len(COMPONENT_NAMES):
            raise ValueError("E193 nominal component vector has the wrong width")
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("E193 nominal components must be finite and nonnegative")
        if (
            isinstance(self.nominal_duration, bool)
            or not isinstance(self.nominal_duration, int)
            or self.nominal_duration < 1
        ):
            raise ValueError("E193 nominal duration must be a positive integer")

    @property
    def nominal_J(self) -> float:
        return scalar_cost(self.nominal_components)


def known_q_tensor(
    records: Sequence[ObservableCostRecord],
    like: Tensor,
    *,
    reward_scale: float,
) -> Tensor:
    """Return the fixed nominal reward contribution in learned-Q units.

    The environment returns negative cost and the controller maximizes Q, so
    the sign is negative.  Applying ``reward_scale`` here is essential: the
    Bellman target applies the same scale to the observed macro return.
    """

    records = tuple(records)
    if not records or not all(isinstance(item, ObservableCostRecord) for item in records):
        raise TypeError("E193 scoring requires ObservableCostRecord instances")
    scale = float(reward_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("reward_scale must be finite and positive")
    return torch.as_tensor(
        [-scale * item.nominal_J for item in records],
        dtype=like.dtype,
        device=like.device,
    )


class ObservableResidualAgent(representation.ObservableLongStayAgent):
    """E193 observable critic with known nominal K added to every Q path."""

    CONTROLLER_ARCHITECTURE_ID = CONTROLLER_ARCHITECTURE
    NETWORK_ARCHITECTURE_ID = NETWORK_ARCHITECTURE

    def _decorate_observed_records(self, snapshot, prepared):
        # ``ObservableLongStayAgent`` invokes this hook only after attaching
        # the ten-field public context to every current state.
        records = []
        base_fields = fields(ViabilityGraphCandidateRecord)
        for record, source_index in zip(
            prepared.records, prepared.source_indices, strict=True
        ):
            prediction = nominal_components(
                self.context_environment,
                snapshot.candidates[source_index],
                self.config.gamma,
            )
            records.append(
                ObservableCostRecord(
                    **{
                        item.name: getattr(record, item.name)
                        for item in base_fields
                    },
                    nominal_components=tuple(
                        float(value) for value in prediction["components"]
                    ),
                    nominal_duration=int(prediction["duration"]),
                )
            )
        return replace(prepared, records=tuple(records))

    def _score_records(self, records, *, network=None, grad=False):
        records = tuple(records)
        residual = super()._score_records(
            records, network=network, grad=grad
        )
        return residual + known_q_tensor(
            records,
            residual,
            reward_scale=self.config.reward_scale,
        )

    def checkpoint_metadata(self) -> dict:
        value = super().checkpoint_metadata()
        value.update(
            network_architecture=NETWORK_ARCHITECTURE,
            e193_residual_value_contract=RESIDUAL_VALUE_CONTRACT,
            e193_nominal_components=COMPONENT_NAMES,
            e193_nominal_costs_stored_in_replay=True,
            e193_nominal_cost_information=(
                "current observable stored/queued jobs and nominal bound path; "
                "unobserved arrivals during macro excluded"
            ),
            e193_total_q_used_for_selection_and_td=True,
            e193_known_q_sign="negative_cost_reward",
            e193_known_q_scale=float(self.config.reward_scale),
        )
        return value

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "ObservableResidualAgent":
        expected = {
            "e193_residual_value_contract": RESIDUAL_VALUE_CONTRACT,
            "e193_nominal_components": COMPONENT_NAMES,
            "e193_nominal_costs_stored_in_replay": True,
            "e193_total_q_used_for_selection_and_td": True,
            "e193_known_q_sign": "negative_cost_reward",
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(
                f"incompatible E193 residual checkpoint: {mismatches!r}"
            )
        agent = super().from_checkpoint(
            payload, device=device, resumable=resumable, seed=seed
        )
        if resumable:
            replay_records = [
                record
                for transition in agent.replay.memory
                for record in (transition.chosen, *transition.next_candidates)
            ]
            if not all(
                isinstance(record, ObservableCostRecord)
                for record in replay_records
            ):
                raise ValueError("E193 resumable replay lacks nominal cost records")
        return agent


def deduplicated_total_scores(records, network, audit=None) -> Tensor:
    """Accelerated no-gradient residual scoring plus immutable nominal K."""

    records = tuple(records)
    residual = representation.deduplicated_residual_scores(
        records, network, audit=audit
    )
    return residual + known_q_tensor(
        records,
        residual,
        reward_scale=network.config.reward_scale,
    )


@contextmanager
def acceleration_enabled():
    """Make online continuation and frozen-target caches store total Q.

    E187's v4 TD accelerator calls the v3 scorer directly instead of calling
    ``agent._score_records``.  Installing the total scorer here prevents the
    known K term from silently disappearing in Double-DQN continuation values.
    """

    with patch.object(
        v3_acceleration,
        "_deduplicated_no_grad_scores",
        deduplicated_total_scores,
    ):
        yield


__all__ = [
    "CONTROLLER_ARCHITECTURE",
    "NETWORK_ARCHITECTURE",
    "ObservableCostRecord",
    "ObservableResidualAgent",
    "RESIDUAL_VALUE_CONTRACT",
    "acceleration_enabled",
    "deduplicated_total_scores",
    "known_q_tensor",
]
