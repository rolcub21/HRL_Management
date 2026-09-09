"""Controlled representation variants for E12.

These agents deliberately leave the exact certificate, liveness guard, action
features, temporal hierarchy, Bellman target, and deployment tie breaking
unchanged.  Only the representation under test changes:

``full_relational_successor``
    The published VCG representation: message passing over both the current
    and counterfactual-successor yard graphs.
``nonrelational_successor``
    The same node features, graph pooling, successor input, action input, and
    parameter count, but every message-passing layer receives an empty edge
    set.  Its transformations are therefore node-local.
``relational_current_candidate``
    The same relational encoder and action input, but no counterfactual
    successor information.  The current graph is supplied to both graph slots,
    so network width and parameter count match the full arm exactly.

The module is separate from the frozen final controller because E12 retrains
all three variants independently; it does not modify any authenticated E1--E11
checkpoint or result.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from PSLAP.viability_candidates import ViabilityMode
from PSLAP.yard_graph import encode_recovery_state, pad_yard_graphs
from vcg_v11_conditioned_handling import (
    CHECKPOINT_SCHEMA_VERSION as HANDLING_CHECKPOINT_SCHEMA_VERSION,
    TARGET_CONTRACT,
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
    ConditionedHandlingConfig,
    ConditionedHandlingDecision,
)
from viability_graph_hierarchy import (
    ACTION_FEATURE_NAMES,
    ID_TO_MODE,
    MODE_NAMES,
    NODE_FEATURE_NAMES,
    VIABILITY_GRAPH_BACKUP_VERSION,
    VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    VIABILITY_GRAPH_REPLAY_VERSION,
    NoCertifiedViableAction,
    ViabilityGraphConfig,
    ViabilityGraphDecision,
    ViabilityGraphHierarchyAgent,
)
from viability_graph_preference_conditioned import select_hierarchical_index


FULL_RELATIONAL_SUCCESSOR = "full_relational_successor"
NONRELATIONAL_SUCCESSOR = "nonrelational_successor"
RELATIONAL_CURRENT_CANDIDATE = "relational_current_candidate"
REPRESENTATION_VARIANTS = (
    FULL_RELATIONAL_SUCCESSOR,
    NONRELATIONAL_SUCCESSOR,
    RELATIONAL_CURRENT_CANDIDATE,
)

OPERATIONAL_CONTROLLER_ARCHITECTURE = (
    "e12_exact_safe_representation_ablation_graph_smdp_q_v1"
)
OPERATIONAL_NETWORK_ARCHITECTURE = {
    FULL_RELATIONAL_SUCCESSOR: (
        "relational_current_successor_shared_action_conditional_q_v1"
    ),
    NONRELATIONAL_SUCCESSOR: (
        "nonrelational_pooled_current_successor_action_conditional_q_v1"
    ),
    RELATIONAL_CURRENT_CANDIDATE: (
        "relational_current_graph_action_conditional_equal_width_q_v1"
    ),
}
HANDLING_CONTROLLER_ARCHITECTURE = (
    "e12_representation_matched_conditioned_future_handling_v1"
)
HANDLING_NETWORK_ARCHITECTURE = (
    "representation_matched_lambda_conditioned_future_cost_v1"
)


class RepresentationAblationError(RuntimeError):
    pass


class EdgeBlindGraphEncoder(nn.Module):
    """Run a normal yard encoder with identical weights but no graph edges."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, batch_or_features, node_mask=None, edge_index=None):
        features, valid, _observed_edges = self.encoder._prepare_inputs(
            batch_or_features, node_mask, edge_index
        )
        empty_edges = torch.empty(
            (2, 0), dtype=torch.long, device=features.device
        )
        return self.encoder(features, valid, empty_edges)


class RepresentationAblationAgent(ViabilityGraphHierarchyAgent):
    """Base class whose concrete subclasses bind one representation arm."""

    representation_variant: Optional[str] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        variant = self._variant()
        steps = int(self.config.message_passing_steps)
        if steps <= 0:
            raise ValueError(
                "E12 keeps three parameter-matched encoder layers in every arm"
            )
        if variant == NONRELATIONAL_SUCCESSOR:
            self.Q_local.graph_encoder = EdgeBlindGraphEncoder(
                self.Q_local.graph_encoder
            )
            self.Q_target.graph_encoder = EdgeBlindGraphEncoder(
                self.Q_target.graph_encoder
            )

    @classmethod
    def _variant(cls) -> str:
        value = cls.representation_variant
        if value not in REPRESENTATION_VARIANTS:
            raise ValueError("a concrete E12 representation variant is required")
        return str(value)

    @staticmethod
    def _record_tensors(records):
        records = tuple(records)
        if not records:
            raise NoCertifiedViableAction("representation arm received no candidate")
        current = tuple(record.current_state for record in records)
        successor = tuple(record.successor_state for record in records)
        action = tuple(record.action_features for record in records)
        return current, successor, action

    def checkpoint_metadata(self) -> dict:
        result = super().checkpoint_metadata()
        variant = self._variant()
        successor = variant != RELATIONAL_CURRENT_CANDIDATE
        relational = variant != NONRELATIONAL_SUCCESSOR
        result.update(
            {
                "controller_architecture": OPERATIONAL_CONTROLLER_ARCHITECTURE,
                "network_architecture": OPERATIONAL_NETWORK_ARCHITECTURE[variant],
                "representation_ablation_protocol": "e12_v1",
                "representation_variant": variant,
                "relational_message_passing": relational,
                "encoder_parameter_matched": True,
                "message_passing_edge_input": (
                    "yard_adjacency" if relational else "empty_edge_set"
                ),
                "counterfactual_successor_graph": successor,
                "current_and_successor_graphs_shared_encoder": successor,
                "q_context_composition": (
                    "current_successor_successor_minus_current_action_v1"
                    if successor
                    else "current_current_zero_delta_action_equal_width_v1"
                ),
                "exact_checker_changed": False,
                "candidate_generator_changed": False,
                "policy_semantics_changed": False,
            }
        )
        return result

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "RepresentationAblationAgent":
        variant = cls._variant()
        expected = {
            "checkpoint_schema_version": VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": OPERATIONAL_CONTROLLER_ARCHITECTURE,
            "network_architecture": OPERATIONAL_NETWORK_ARCHITECTURE[variant],
            "representation_ablation_protocol": "e12_v1",
            "representation_variant": variant,
            "backup_version": VIABILITY_GRAPH_BACKUP_VERSION,
            "replay_version": VIABILITY_GRAPH_REPLAY_VERSION,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "graph_node_feature_names": NODE_FEATURE_NAMES,
            "modes": MODE_NAMES,
            "exact_safe_mask_authoritative": True,
            "baseline_teacher": False,
            "exact_checker_changed": False,
            "candidate_generator_changed": False,
            "policy_semantics_changed": False,
            "encoder_parameter_matched": True,
        }
        mismatch = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatch:
            raise ValueError(f"incompatible E12 operational checkpoint: {mismatch}")
        agent = cls(
            config=ViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


class FullRelationalSuccessorAgent(RepresentationAblationAgent):
    representation_variant = FULL_RELATIONAL_SUCCESSOR


class NonrelationalSuccessorAgent(RepresentationAblationAgent):
    representation_variant = NONRELATIONAL_SUCCESSOR


class RelationalCurrentCandidateAgent(RepresentationAblationAgent):
    representation_variant = RELATIONAL_CURRENT_CANDIDATE

    @staticmethod
    def _record_tensors(records):
        records = tuple(records)
        if not records:
            raise NoCertifiedViableAction("representation arm received no candidate")
        current = tuple(record.current_state for record in records)
        # Repeating the current graph preserves the full network's dimensions
        # while making both the successor slot and difference slot independent
        # of the candidate's modeled successor.
        return (
            current,
            current,
            tuple(record.action_features for record in records),
        )


AGENT_CLASS_BY_VARIANT = {
    FULL_RELATIONAL_SUCCESSOR: FullRelationalSuccessorAgent,
    NONRELATIONAL_SUCCESSOR: NonrelationalSuccessorAgent,
    RELATIONAL_CURRENT_CANDIDATE: RelationalCurrentCandidateAgent,
}


def agent_class_for_variant(variant: str):
    try:
        return AGENT_CLASS_BY_VARIANT[str(variant)]
    except KeyError as error:
        raise ValueError(f"unknown E12 representation variant: {variant!r}") from error


def detached_representation_features(
    base_agent: RepresentationAblationAgent,
    records: Sequence,
) -> Tensor:
    """Return the exact feature prefix used by this arm's operational Q."""

    records = tuple(records)
    if not records:
        raise NoCertifiedViableAction("handling model received an empty frontier")
    current_states, successor_states, action_features = base_agent._record_tensors(
        records
    )
    network = base_agent.Q_local
    parameter = next(network.parameters())
    action_features = torch.as_tensor(
        action_features, dtype=parameter.dtype, device=parameter.device
    )
    with torch.no_grad():
        graphs = tuple(
            encode_recovery_state(
                state, timing_scale=network.config.timing_scale
            )
            for state in current_states + successor_states
        )
        embedding = network.graph_encoder(pad_yard_graphs(graphs))
        count = len(current_states)
        current = embedding[:count]
        successor = embedding[count:]
        action = network.action_encoder(action_features)
        features = torch.cat(
            (current, successor, successor - current, action), dim=-1
        )
    return features.detach()


class RepresentationConditionedHandlingAgent(ConditionedHandlingAgent):
    """Conditioned handling head using the matched E12 representation."""

    base_agent: RepresentationAblationAgent

    def select(
        self,
        snapshot,
        *,
        preference_lambda: float,
        training: bool = False,
        epsilon: Optional[float] = None,
    ):
        value = self._validate_lambda(preference_lambda)
        epsilon_value = (
            self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        )
        if value == 0.0:
            if training and epsilon_value != 0.0:
                raise ValueError("lambda-zero collection must preserve base Qop")
            return self.base_agent.select(snapshot, training=False, epsilon=0.0)

        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE conditioned candidate")
        features = detached_representation_features(
            self.base_agent, prepared.records
        )
        with torch.no_grad():
            operational = self.base_agent.Q_local.q_head(features).squeeze(-1)
            immediate = torch.as_tensor(
                [
                    1.0 if item.action_type == "reconfigure" else 0.0
                    for item in prepared.records
                ],
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
            training
            and not liveness_forced
            and self.rng.random() < epsilon_value
        )
        if liveness_forced:
            selected_index = 0
            source = "exact_recovery_witness_guard"
        elif explored:
            live_modes = tuple(mode for mode, _ in hierarchy.mode_values)
            selected_mode = self.rng.choice(live_modes)
            indices = [
                index
                for index, mode in enumerate(prepared.mode_ids)
                if mode == selected_mode
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
        base.exact_rejections_seen += int(
            snapshot.audit.fail_closed_rejection_count
        )
        self.decision_count += 1
        self.preference_decisions[str(value)] += 1
        return ConditionedHandlingDecision(
            base_decision=base_decision,
            preference_lambda=value,
            operational_values=tuple(
                float(item) for item in operational.detach().cpu()
            ),
            handling_values=tuple(
                float(item) for item in handling.detach().cpu()
            ),
            chosen_feature=features[selected_index].detach().cpu().clone(),
            chosen_immediate=int(immediate[selected_index].item()),
        )

    def checkpoint(
        self,
        *,
        base_checkpoint_sha256: str,
        base_policy_digest: str,
    ) -> dict:
        if self.pending_observations or self.completed_samples is not None:
            raise ValueError("handling checkpoint requires an episode boundary")
        return {
            "checkpoint_schema_version": HANDLING_CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": HANDLING_CONTROLLER_ARCHITECTURE,
            "network_architecture": HANDLING_NETWORK_ARCHITECTURE,
            "target_contract": TARGET_CONTRACT,
            "representation_variant": self.base_agent._variant(),
            "config": self.config.to_dict(),
            "base_checkpoint_sha256": str(base_checkpoint_sha256),
            "base_policy_digest": str(base_policy_digest),
            "lambda_zero_direct_base_delegation": True,
            "operational_critic_frozen": True,
            "exact_immediate_rehandle_term": True,
            "random_initialization_no_warm_start": True,
            "network_state": self.handling_network.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping,
        *,
        base_agent: RepresentationAblationAgent,
        expected_base_checkpoint_sha256: str,
        expected_base_policy_digest: str,
        seed: int,
    ) -> "RepresentationConditionedHandlingAgent":
        expected = {
            "checkpoint_schema_version": HANDLING_CHECKPOINT_SCHEMA_VERSION,
            "controller_architecture": HANDLING_CONTROLLER_ARCHITECTURE,
            "network_architecture": HANDLING_NETWORK_ARCHITECTURE,
            "target_contract": TARGET_CONTRACT,
            "representation_variant": base_agent._variant(),
            "base_checkpoint_sha256": expected_base_checkpoint_sha256,
            "base_policy_digest": expected_base_policy_digest,
            "lambda_zero_direct_base_delegation": True,
            "operational_critic_frozen": True,
            "exact_immediate_rehandle_term": True,
            "random_initialization_no_warm_start": True,
        }
        mismatch = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatch:
            raise ValueError(f"incompatible E12 handling checkpoint: {mismatch}")
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
            "controller_architecture": HANDLING_CONTROLLER_ARCHITECTURE,
            "representation_variant": self.base_agent._variant(),
            "operational_critic_frozen": True,
            "lambda_zero_direct_base_delegation": True,
            "exact_immediate_rehandle_term": True,
            "future_handling_nonnegative": True,
            "target_contract": TARGET_CONTRACT,
            "decision_count": self.decision_count,
            "transition_count": self.transition_count,
            "preference_decisions": dict(self.preference_decisions),
            "base_audit": self.base_agent.audit(include_decisions=False),
        }


__all__ = [
    "AGENT_CLASS_BY_VARIANT",
    "FULL_RELATIONAL_SUCCESSOR",
    "HANDLING_CONTROLLER_ARCHITECTURE",
    "HANDLING_NETWORK_ARCHITECTURE",
    "NONRELATIONAL_SUCCESSOR",
    "OPERATIONAL_CONTROLLER_ARCHITECTURE",
    "OPERATIONAL_NETWORK_ARCHITECTURE",
    "RELATIONAL_CURRENT_CANDIDATE",
    "REPRESENTATION_VARIANTS",
    "FullRelationalSuccessorAgent",
    "NonrelationalSuccessorAgent",
    "RelationalCurrentCandidateAgent",
    "RepresentationAblationAgent",
    "RepresentationConditionedHandlingAgent",
    "agent_class_for_variant",
    "detached_representation_features",
]
