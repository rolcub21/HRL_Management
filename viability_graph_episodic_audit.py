"""Finite-episode discount audit for the VCG controller.

This module changes exactly one learning assumption from
``viability_graph_hierarchy``: the SMDP discount may be ``gamma == 1`` when a
finite episode horizon is declared.  The exact-safe candidate frontier, graph
and action representations, shared Q network, regularized continuation,
exploration rule, replay sampling, liveness shield, and deployment policy are
otherwise inherited unchanged from VCG-v1.

The variant has its own controller/checkpoint/backup identities.  A VCG-v1
loader must therefore reject one of these checkpoints, and this loader must
reject a VCG-v1 checkpoint.  This prevents an objective-audit checkpoint from
silently entering the main VCG result stream.

Remaining episode budget is deliberately *not* appended to the observation in
this audit.  Adding it only to the learned arm would confound the discount
comparison by changing the information set.  A later time-aware experiment
must expose the same observable clock/budget to every learned and deterministic
comparison arm and must receive a separate protocol identity.

For ``gamma == 1`` this is an episodic return estimator, not a discounted
infinite-horizon contraction.  Training integrations must mark completion or
time-limit truncation as terminal.  ``remember`` enforces that no transition
bootstraps at or beyond the declared horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
from torch import Tensor

from PSLAP.viability_candidates import (
    EXACT_VERIFIER_AUTHORITY,
    FAIL_CLOSED_CERTIFICATION_CONTRACT,
    VIABILITY_CANDIDATE_INTERFACE,
)
from PSLAP.yard_graph import (
    NODE_FEATURE_NAMES,
)
from viability_graph_hierarchy import (
    ACTION_FEATURE_NAMES,
    VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
    ViabilityGraphConfig,
    ViabilityGraphDecision,
    ViabilityGraphHierarchyAgent,
    ViabilityGraphTransition,
    common_smdp_target,
    double_dqn_regularized_continuation,
)


EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION = 2
EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY = (
    "episodic_viability_graph_objective_audit_v1"
)
EPISODIC_VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE = (
    "exact_viability_constrained_graph_smdp_q_episodic_gamma_capable_audit_v1"
)
EPISODIC_VIABILITY_GRAPH_BACKUP_VERSION = (
    "nested_cardinality_normalized_regularized_double_q_common_"
    "finite_episode_gamma_capable_smdp_v1"
)
EPISODIC_TERMINAL_BOUNDARY_CONTRACT = (
    "completion_or_time_limit_is_terminal_no_bootstrap_at_declared_horizon_v1"
)
EPISODIC_TIME_CONTEXT_CONTRACT = (
    "no_remaining_budget_feature_architecture_and_information_match_vcg_v1"
)


def episodic_common_smdp_target(
    rewards: Tensor,
    durations: Tensor,
    dones: Tensor,
    next_values: Tensor,
    *,
    gamma: float,
    reward_scale: float = 1.0,
) -> Tensor:
    """Common finite-episode SMDP target allowing ``0 <= gamma <= 1``.

    ``rewards`` must already contain the discounted intra-macro return under
    the same ``gamma``.  With ``gamma == 1`` that is the undiscounted sum of
    rewards inside the macro.  Nonterminal continuation remains shared across
    Accept, Recover, and Defer; no mode-specific target is introduced.

    For every ``gamma < 1`` this delegates to the VCG-v1 operator, making the
    audit variant numerically identical rather than merely algebraically
    similar.  The explicit ``gamma == 1`` branch is the only extension.
    """

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if gamma < 1.0:
        return common_smdp_target(
            rewards,
            durations,
            dones,
            next_values,
            gamma=gamma,
            reward_scale=reward_scale,
        )

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
    # Written explicitly, rather than dropping the duration input, so the
    # gamma=1 audit retains the same variable-duration operator signature.
    discount = torch.pow(
        rewards.new_tensor(1.0), durations.to(dtype=rewards.dtype)
    )
    return (
        reward_scale * rewards
        + discount * (~dones).to(rewards.dtype) * next_values
    )


@dataclass(frozen=True)
class EpisodicViabilityGraphConfig(ViabilityGraphConfig):
    """VCG-v1 hyperparameters plus an authenticated finite horizon."""

    episode_horizon_steps: int = 4_000

    def __post_init__(self) -> None:
        gamma = float(self.gamma)
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if (
            isinstance(self.episode_horizon_steps, bool)
            or not isinstance(self.episode_horizon_steps, int)
            or self.episode_horizon_steps <= 0
        ):
            raise ValueError("episode_horizon_steps must be a positive integer")

        # Reuse every V1 validation rule without weakening the original class.
        # Its only narrower rule is gamma < 1, so validate a representable
        # value immediately below one when this audit requests gamma == 1.
        base_values = {
            name: getattr(self, name)
            for name in ViabilityGraphConfig.__dataclass_fields__
        }
        if gamma == 1.0:
            base_values["gamma"] = math.nextafter(1.0, 0.0)
        ViabilityGraphConfig(**base_values)


class EpisodicViabilityGraphHierarchyAgent(ViabilityGraphHierarchyAgent):
    """VCG-v1 agent with an isolated finite-episode gamma-capable backup."""

    def __init__(
        self,
        *,
        config: Optional[EpisodicViabilityGraphConfig] = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
        epsilon: float = 0.9,
    ) -> None:
        config = config or EpisodicViabilityGraphConfig()
        if not isinstance(config, EpisodicViabilityGraphConfig):
            raise TypeError("config must be an EpisodicViabilityGraphConfig")

        # V1's network never reads gamma; nevertheless its constructor checks
        # the V1 config.  Initialize the exactly identical network/agent path
        # with the adjacent valid gamma, then restore the authenticated audit
        # config before any scoring, replay, or learning occurs.
        base_values = {
            name: getattr(config, name)
            for name in ViabilityGraphConfig.__dataclass_fields__
        }
        if float(config.gamma) == 1.0:
            base_values["gamma"] = math.nextafter(1.0, 0.0)
        super().__init__(
            config=ViabilityGraphConfig(**base_values),
            seed=seed,
            device=device,
            epsilon=epsilon,
        )
        self.config = config
        # The config is retained only for timing-scale introspection in the
        # shared network, but keeping it exact makes audits unambiguous.
        self.Q_local.config = config
        self.Q_target.config = config

    def remember(
        self,
        decision: ViabilityGraphDecision,
        *,
        reward: float,
        duration: int,
        next_snapshot,
        done: bool,
        outcome_already_observed: bool = False,
    ) -> ViabilityGraphTransition:
        """Store a transition while enforcing the finite-horizon boundary."""

        if not isinstance(decision, ViabilityGraphDecision):
            raise TypeError("remember requires a ViabilityGraphDecision")
        if not bool(done):
            boundary = int(decision.prepared_snapshot.decision_epoch) + int(
                duration
            )
            if next_snapshot is not None:
                boundary = max(boundary, int(next_snapshot.decision_epoch))
            if boundary >= self.config.episode_horizon_steps:
                raise ValueError(
                    "a transition reaching the declared finite horizon must "
                    "be terminal (done=True)"
                )
        return super().remember(
            decision,
            reward=reward,
            duration=duration,
            next_snapshot=next_snapshot,
            done=done,
            outcome_already_observed=outcome_already_observed,
        )

    def _td_batch(
        self, transitions: Sequence[ViabilityGraphTransition]
    ) -> tuple[Tensor, Tensor]:
        """VCG-v1 Double-Q continuation with the episodic target operator."""

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
        targets = episodic_common_smdp_target(
            rewards,
            durations,
            dones,
            next_values,
            gamma=self.config.gamma,
            reward_scale=self.config.reward_scale,
        )
        return prediction, targets.detach()

    def checkpoint_metadata(self) -> dict:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "checkpoint_family": (
                    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY
                ),
                "controller_architecture": (
                    EPISODIC_VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE
                ),
                # Deliberately identical: this audit is not an architecture
                # ablation and must not be described as one.
                "network_architecture": VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
                "backup_version": EPISODIC_VIABILITY_GRAPH_BACKUP_VERSION,
                "finite_episode_objective_audit": True,
                "finite_horizon_required": True,
                "episode_horizon_steps": self.config.episode_horizon_steps,
                "horizon_cap_role": (
                    "protocol_safety_cap_not_task_objective_v1"
                ),
                "horizon_nonbinding_must_be_reported_empirically": True,
                "terminal_boundary_contract": (
                    EPISODIC_TERMINAL_BOUNDARY_CONTRACT
                ),
                "operator_gamma_domain": "closed_interval_0_to_1_v1",
                "gamma_one_is_infinite_horizon_contraction": False,
                "gamma_one_semantics": (
                    "finite_episode_undiscounted_return_no_fixed_point_claim"
                ),
                "time_context_contract": EPISODIC_TIME_CONTEXT_CONTRACT,
                "remaining_budget_observed_by_q": False,
                "representation_difference_from_vcg_v1": "none",
                "safe_frontier_difference_from_vcg_v1": "none",
                "policy_difference_from_vcg_v1": "none",
                "comparison_information_parity_required": True,
            }
        )
        return metadata

    def checkpoint(self, *, include_replay: bool = True, **metadata) -> dict:
        payload = {
            "checkpoint_schema_version": (
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION
            ),
            **self.checkpoint_metadata(),
            "config": self.config.to_dict(),
            "agent_state": self.checkpoint_state(include_replay=include_replay),
        }
        collisions = sorted(set(payload).intersection(metadata))
        if collisions:
            raise ValueError(
                "checkpoint metadata cannot override reserved keys: "
                f"{collisions!r}"
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
    ) -> "EpisodicViabilityGraphHierarchyAgent":
        expected = {
            "checkpoint_schema_version": (
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION
            ),
            "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
            "controller_architecture": (
                EPISODIC_VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE
            ),
            "candidate_interface": VIABILITY_CANDIDATE_INTERFACE,
            "network_architecture": VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
            "backup_version": EPISODIC_VIABILITY_GRAPH_BACKUP_VERSION,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "exact_safe_mask_authoritative": True,
            "baseline_teacher": False,
            "finite_episode_objective_audit": True,
            "finite_horizon_required": True,
            "terminal_boundary_contract": EPISODIC_TERMINAL_BOUNDARY_CONTRACT,
            "time_context_contract": EPISODIC_TIME_CONTEXT_CONTRACT,
            "remaining_budget_observed_by_q": False,
            "representation_difference_from_vcg_v1": "none",
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatches:
            raise ValueError(
                "incompatible episodic viability graph checkpoint: "
                f"{mismatches!r}"
            )
        config = EpisodicViabilityGraphConfig.from_dict(payload["config"])
        metadata_mismatches = {}
        if float(payload.get("gamma", math.nan)) != float(config.gamma):
            metadata_mismatches["gamma"] = (
                payload.get("gamma"),
                config.gamma,
            )
        if int(payload.get("episode_horizon_steps", -1)) != int(
            config.episode_horizon_steps
        ):
            metadata_mismatches["episode_horizon_steps"] = (
                payload.get("episode_horizon_steps"),
                config.episode_horizon_steps,
            )
        if metadata_mismatches:
            raise ValueError(
                "episodic checkpoint metadata/config mismatch: "
                f"{metadata_mismatches!r}"
            )
        agent = cls(
            config=config,
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(
            payload["agent_state"], resumable=resumable
        )
        return agent


__all__ = [
    "EPISODIC_TERMINAL_BOUNDARY_CONTRACT",
    "EPISODIC_TIME_CONTEXT_CONTRACT",
    "EPISODIC_VIABILITY_GRAPH_BACKUP_VERSION",
    "EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY",
    "EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION",
    "EPISODIC_VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE",
    "EpisodicViabilityGraphConfig",
    "EpisodicViabilityGraphHierarchyAgent",
    "episodic_common_smdp_target",
]
