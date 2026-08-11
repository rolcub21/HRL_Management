"""Learned recoverability estimates from policy-independent yard graphs.

The critic in this module is deliberately narrower than a controller.  It
maps a current or counterfactual :class:`~PSLAP.yard_graph.YardGraph` batch to
estimates of the physical recovery problem defined in :mod:`PSLAP.viability`:

* whether the state is recoverable under the stated dynamics contract;
* the number of strict macros in a recovery witness; and
* the number of primitive steps in that witness.

Labels come from dynamics search certificates, never from a storage baseline
or a behavior-policy action.  Computational ``UNKNOWN`` certificates are
masked from every supervised target.  They are not negative examples and, in
particular, are never silently promoted to ``SAFE``.

The optional ensemble helper reports an empirical lower confidence bound
``mean - scale * standard_deviation``.  This is a conservative ranking or
screening statistic, not a formal safety certificate.  Exact certification is
still required wherever a hard recoverability guarantee is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from PSLAP.viability import RecoverabilityCertificate, ViabilityStatus
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    PaddedYardGraphBatch,
    YardGraphEncoder,
)


DYNAMICS_LABEL_CONTRACT = (
    "exact_recovery_status_bfs_rank_and_valid_witness_steps_v2"
)


def _optional_nonnegative_finite(
    value: Optional[float],
    *,
    name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-negative finite number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number or None")
    return normalized


@dataclass(frozen=True)
class DynamicsViabilityLabel:
    """One policy-independent training label produced by dynamics search.

    ``recovery_rank`` is the minimum number of strict macros only when the
    verifier used breadth-first search. ``primitive_steps`` is the movement,
    pickup, and putdown count of the returned valid witness; it need not be a
    globally shortest primitive route.
    Regression targets are meaningful only for ``SAFE`` certificates.  A
    ``SAFE`` label may omit either target when only classification supervision
    is available, but ``UNSAFE`` and ``UNKNOWN`` labels may not invent them.
    """

    status: ViabilityStatus
    recovery_rank: Optional[float] = None
    primitive_steps: Optional[float] = None
    contract: str = DYNAMICS_LABEL_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.status, ViabilityStatus):
            raise TypeError("status must be a ViabilityStatus")
        rank = _optional_nonnegative_finite(
            self.recovery_rank,
            name="recovery_rank",
        )
        steps = _optional_nonnegative_finite(
            self.primitive_steps,
            name="primitive_steps",
        )
        object.__setattr__(self, "recovery_rank", rank)
        object.__setattr__(self, "primitive_steps", steps)
        if self.status is not ViabilityStatus.SAFE and (
            rank is not None or steps is not None
        ):
            raise ValueError(
                "only SAFE dynamics labels may contain recovery targets"
            )
        if not isinstance(self.contract, str) or not self.contract:
            raise ValueError("label contract must be a non-empty string")

    @classmethod
    def from_certificate(
        cls,
        certificate: RecoverabilityCertificate,
    ) -> "DynamicsViabilityLabel":
        """Convert exact-search evidence without changing its status.

        In particular, a resource-limited ``UNKNOWN`` certificate remains an
        unknown label.  Its empty witness is not interpreted as zero recovery
        work and its search cutoff is not interpreted as evidence of failure.
        """

        if not isinstance(certificate, RecoverabilityCertificate):
            raise TypeError("certificate must be a RecoverabilityCertificate")
        if certificate.status is not ViabilityStatus.SAFE:
            return cls(status=certificate.status)
        if certificate.witness_primitive_steps is None:  # pragma: no cover
            raise ValueError("SAFE certificate is missing witness step count")
        return cls(
            status=certificate.status,
            recovery_rank=certificate.exact_recovery_rank,
            primitive_steps=certificate.witness_primitive_steps,
        )

    @property
    def has_safety_target(self) -> bool:
        return self.status is not ViabilityStatus.UNKNOWN

    @property
    def safety_target(self) -> Optional[float]:
        if self.status is ViabilityStatus.UNKNOWN:
            return None
        return float(self.status is ViabilityStatus.SAFE)


# A more explicit semantic spelling for callers that use recoverability
# terminology.  Both names identify the same immutable data contract.
DynamicsRecoverabilityLabel = DynamicsViabilityLabel


@dataclass(frozen=True)
class ViabilityCriticOutput:
    """Batched, differentiable predictions from one critic."""

    safety_logit: Tensor
    safety_probability: Tensor
    recovery_rank: Tensor
    primitive_steps: Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.safety_logit,
            self.safety_probability,
            self.recovery_rank,
            self.primitive_steps,
        )
        if not all(torch.is_tensor(value) for value in tensors):
            raise TypeError("all critic outputs must be tensors")
        shape = self.safety_logit.shape
        if any(value.shape != shape for value in tensors[1:]):
            raise ValueError("all critic outputs must have identical shapes")

    @property
    def safe_probability(self) -> Tensor:
        """Readable alias used at conservative screening call sites."""

        return self.safety_probability


def _prediction_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )


class CounterfactualViabilityCritic(nn.Module):
    """Graph critic with separate safety, rank, and primitive-step heads.

    The supplied :class:`YardGraphEncoder` determines the variable-size graph
    representation.  A caller may share no weights, share an externally
    managed encoder, or instantiate independent critics for an ensemble.  The
    three prediction heads never consume baseline scores or selected actions.
    """

    def __init__(
        self,
        *,
        encoder: Optional[YardGraphEncoder] = None,
        input_dim: int = NODE_FEATURE_DIM,
        graph_hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        message_passing_steps: int = 2,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if encoder is not None and not isinstance(encoder, YardGraphEncoder):
            raise TypeError("encoder must be a YardGraphEncoder or None")
        for value, name in (
            (input_dim, "input_dim"),
            (graph_hidden_dim, "graph_hidden_dim"),
            (graph_embedding_dim, "graph_embedding_dim"),
            (head_hidden_dim, "head_hidden_dim"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(message_passing_steps, bool)
            or not isinstance(message_passing_steps, int)
            or message_passing_steps < 0
        ):
            raise ValueError(
                "message_passing_steps must be a non-negative integer"
            )

        self.encoder = encoder or YardGraphEncoder(
            input_dim=input_dim,
            hidden_dim=graph_hidden_dim,
            output_dim=graph_embedding_dim,
            message_passing_steps=message_passing_steps,
        )
        embedding_dim = self.encoder.output_dim
        self.safety_head = _prediction_head(embedding_dim, head_hidden_dim)
        self.recovery_rank_head = _prediction_head(
            embedding_dim, head_hidden_dim
        )
        self.primitive_steps_head = _prediction_head(
            embedding_dim, head_hidden_dim
        )

    def forward(
        self,
        batch_or_features: PaddedYardGraphBatch | Tensor,
        node_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> ViabilityCriticOutput:
        embedding = self.encoder(
            batch_or_features,
            node_mask=node_mask,
            edge_index=edge_index,
        )
        safety_logit = self.safety_head(embedding).squeeze(-1)
        # Softplus preserves gradients while guaranteeing estimates >= 0.
        recovery_rank = F.softplus(
            self.recovery_rank_head(embedding).squeeze(-1)
        )
        primitive_steps = F.softplus(
            self.primitive_steps_head(embedding).squeeze(-1)
        )
        return ViabilityCriticOutput(
            safety_logit=safety_logit,
            safety_probability=torch.sigmoid(safety_logit),
            recovery_rank=recovery_rank,
            primitive_steps=primitive_steps,
        )


# Concise public name for integrations that do not need to repeat the
# counterfactual qualifier at every call site.
ViabilityCritic = CounterfactualViabilityCritic


@dataclass(frozen=True)
class ViabilityCriticLoss:
    """Differentiable loss components plus auditable mask counts."""

    total: Tensor
    safety: Tensor
    recovery_rank: Tensor
    primitive_steps: Tensor
    known_safety_count: int
    recovery_rank_count: int
    primitive_steps_count: int


def _validate_loss_scalar(
    value: float,
    *,
    name: str,
    minimum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number >= {minimum}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return normalized


def _masked_smooth_l1(
    prediction: Tensor,
    targets: Sequence[Optional[float]],
    *,
    beta: float,
) -> tuple[Tensor, int]:
    indices = tuple(index for index, value in enumerate(targets) if value is not None)
    if not indices:
        return prediction.sum() * 0.0, 0
    index_tensor = torch.tensor(
        indices,
        dtype=torch.long,
        device=prediction.device,
    )
    target_tensor = prediction.new_tensor(
        tuple(float(targets[index]) for index in indices)
    )
    return (
        F.smooth_l1_loss(
            prediction.index_select(0, index_tensor),
            target_tensor,
            reduction="mean",
            beta=beta,
        ),
        len(indices),
    )


def compute_viability_critic_loss(
    output: ViabilityCriticOutput,
    labels: Sequence[DynamicsViabilityLabel],
    *,
    false_safe_weight: float = 5.0,
    safety_weight: float = 1.0,
    recovery_rank_weight: float = 1.0,
    primitive_steps_weight: float = 1.0,
    huber_beta: float = 1.0,
) -> ViabilityCriticLoss:
    """Compute masked asymmetric classification and witness regression loss.

    ``SAFE`` is target one and ``UNSAFE`` is target zero.  Multiplying the
    negative examples by ``false_safe_weight`` specifically increases the
    penalty and gradient when an unsafe state receives a high safe logit.
    Loss is normalized by the number of known labels rather than the sum of
    sample weights, so increasing this parameter cannot cancel itself out.

    ``UNKNOWN`` labels are absent from the safety target and from both
    regression targets.  Rank and primitive-step regression additionally
    require a ``SAFE`` label carrying the corresponding exact witness value.
    """

    if not isinstance(output, ViabilityCriticOutput):
        raise TypeError("output must be a ViabilityCriticOutput")
    labels = tuple(labels)
    if not all(isinstance(label, DynamicsViabilityLabel) for label in labels):
        raise TypeError("labels must contain DynamicsViabilityLabel instances")
    if output.safety_logit.ndim != 1:
        raise ValueError("critic loss expects one-dimensional batched outputs")
    if len(labels) != output.safety_logit.shape[0]:
        raise ValueError("label count must equal critic batch size")

    false_safe_weight = _validate_loss_scalar(
        false_safe_weight,
        name="false_safe_weight",
        minimum=1.0,
    )
    safety_weight = _validate_loss_scalar(
        safety_weight,
        name="safety_weight",
        minimum=0.0,
    )
    recovery_rank_weight = _validate_loss_scalar(
        recovery_rank_weight,
        name="recovery_rank_weight",
        minimum=0.0,
    )
    primitive_steps_weight = _validate_loss_scalar(
        primitive_steps_weight,
        name="primitive_steps_weight",
        minimum=0.0,
    )
    huber_beta = _validate_loss_scalar(
        huber_beta,
        name="huber_beta",
        minimum=0.0,
    )
    if huber_beta == 0.0:
        raise ValueError("huber_beta must be strictly positive")

    known_indices = tuple(
        index for index, label in enumerate(labels) if label.has_safety_target
    )
    if known_indices:
        index_tensor = torch.tensor(
            known_indices,
            dtype=torch.long,
            device=output.safety_logit.device,
        )
        logits = output.safety_logit.index_select(0, index_tensor)
        targets = logits.new_tensor(
            tuple(labels[index].safety_target for index in known_indices)
        )
        per_example = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        asymmetric_weights = torch.where(
            targets == 0.0,
            per_example.new_full(per_example.shape, false_safe_weight),
            torch.ones_like(per_example),
        )
        safety_loss = (per_example * asymmetric_weights).mean()
    else:
        safety_loss = output.safety_logit.sum() * 0.0

    rank_targets = tuple(
        label.recovery_rank
        if label.status is ViabilityStatus.SAFE
        else None
        for label in labels
    )
    primitive_targets = tuple(
        label.primitive_steps
        if label.status is ViabilityStatus.SAFE
        else None
        for label in labels
    )
    rank_loss, rank_count = _masked_smooth_l1(
        output.recovery_rank,
        rank_targets,
        beta=huber_beta,
    )
    primitive_loss, primitive_count = _masked_smooth_l1(
        output.primitive_steps,
        primitive_targets,
        beta=huber_beta,
    )
    total = (
        safety_weight * safety_loss
        + recovery_rank_weight * rank_loss
        + primitive_steps_weight * primitive_loss
    )
    return ViabilityCriticLoss(
        total=total,
        safety=safety_loss,
        recovery_rank=rank_loss,
        primitive_steps=primitive_loss,
        known_safety_count=len(known_indices),
        recovery_rank_count=rank_count,
        primitive_steps_count=primitive_count,
    )


# Short alias matching the module's single loss definition.
viability_critic_loss = compute_viability_critic_loss


@dataclass(frozen=True)
class ViabilityEnsembleEstimate:
    """Empirical member statistics for one graph batch."""

    safety_probability_mean: Tensor
    safety_probability_std: Tensor
    safety_probability_lcb: Tensor
    recovery_rank_mean: Tensor
    recovery_rank_std: Tensor
    primitive_steps_mean: Tensor
    primitive_steps_std: Tensor
    member_count: int

    @property
    def conservative_safe_probability(self) -> Tensor:
        return self.safety_probability_lcb


def summarize_viability_ensemble(
    outputs: Sequence[ViabilityCriticOutput],
    *,
    lcb_scale: float = 2.0,
) -> ViabilityEnsembleEstimate:
    """Aggregate independent critics and form a bounded empirical LCB.

    The population standard deviation (``unbiased=False``) makes the utility
    well defined for one member.  ``lcb_scale`` is a tunable conservatism
    coefficient; it does not turn this statistic into an exact certificate.
    """

    outputs = tuple(outputs)
    if not outputs:
        raise ValueError("at least one critic output is required")
    if not all(isinstance(output, ViabilityCriticOutput) for output in outputs):
        raise TypeError("outputs must contain ViabilityCriticOutput instances")
    lcb_scale = _validate_loss_scalar(
        lcb_scale,
        name="lcb_scale",
        minimum=0.0,
    )
    reference = outputs[0].safety_probability
    if any(
        output.safety_probability.shape != reference.shape
        or output.safety_probability.device != reference.device
        for output in outputs[1:]
    ):
        raise ValueError("ensemble member outputs must share shape and device")

    probabilities = torch.stack(
        tuple(output.safety_probability for output in outputs), dim=0
    )
    ranks = torch.stack(tuple(output.recovery_rank for output in outputs), dim=0)
    steps = torch.stack(tuple(output.primitive_steps for output in outputs), dim=0)
    probability_mean = probabilities.mean(dim=0)
    probability_std = probabilities.std(dim=0, unbiased=False)
    return ViabilityEnsembleEstimate(
        safety_probability_mean=probability_mean,
        safety_probability_std=probability_std,
        safety_probability_lcb=(
            probability_mean - lcb_scale * probability_std
        ).clamp(0.0, 1.0),
        recovery_rank_mean=ranks.mean(dim=0),
        recovery_rank_std=ranks.std(dim=0, unbiased=False),
        primitive_steps_mean=steps.mean(dim=0),
        primitive_steps_std=steps.std(dim=0, unbiased=False),
        member_count=len(outputs),
    )


class ViabilityCriticEnsemble(nn.Module):
    """Thin container for independently parameterized viability critics."""

    def __init__(
        self,
        members: Sequence[CounterfactualViabilityCritic],
    ) -> None:
        super().__init__()
        members = tuple(members)
        if not members:
            raise ValueError("an ensemble requires at least one critic")
        if not all(
            isinstance(member, CounterfactualViabilityCritic)
            for member in members
        ):
            raise TypeError(
                "ensemble members must be CounterfactualViabilityCritic instances"
            )
        self.members = nn.ModuleList(members)

    def forward(
        self,
        batch_or_features: PaddedYardGraphBatch | Tensor,
        node_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> tuple[ViabilityCriticOutput, ...]:
        return tuple(
            member(
                batch_or_features,
                node_mask=node_mask,
                edge_index=edge_index,
            )
            for member in self.members
        )

    def estimate(
        self,
        batch_or_features: PaddedYardGraphBatch | Tensor,
        node_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        *,
        lcb_scale: float = 2.0,
    ) -> ViabilityEnsembleEstimate:
        return summarize_viability_ensemble(
            self(
                batch_or_features,
                node_mask=node_mask,
                edge_index=edge_index,
            ),
            lcb_scale=lcb_scale,
        )


__all__ = [
    "DYNAMICS_LABEL_CONTRACT",
    "CounterfactualViabilityCritic",
    "DynamicsRecoverabilityLabel",
    "DynamicsViabilityLabel",
    "ViabilityCritic",
    "ViabilityCriticEnsemble",
    "ViabilityCriticLoss",
    "ViabilityCriticOutput",
    "ViabilityEnsembleEstimate",
    "compute_viability_critic_loss",
    "summarize_viability_ensemble",
    "viability_critic_loss",
]
