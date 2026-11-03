"""Fresh E215 architecture with the complete exact-SAFE frontier exposed."""

from __future__ import annotations

import torch

from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191,
)
from experiments.conditioned_vcg.E225_e215_seed0_guard_free_development_284k_v1.guard_free import (
    FrozenGuardFreeQueueAgent,
)


CONTROLLER_ARCHITECTURE = (
    "e226_e215_fresh_guard_free_full_certified_frontier_v1"
)
NETWORK_ARCHITECTURE = e191.NETWORK_ARCHITECTURE
TIME_CONTEXT_CONTRACT = e191.TIME_CONTEXT_CONTRACT
QUEUE_CONTEXT_NAMES = e191.QUEUE_CONTEXT_NAMES
QUEUE_CONTEXT_DIM = e191.QUEUE_CONTEXT_DIM
GUARD_FREE_CONTRACT = (
    "passive_e215_nonprogress_features_no_mask_no_witness_no_bypass_v1"
)


class GuardFreeTrainingQueueAgent(FrozenGuardFreeQueueAgent):
    """E215 neural agent whose critic always sees the full certified frontier."""

    def checkpoint_metadata(self) -> dict:
        value = super().checkpoint_metadata()
        for name in (
            "e188_choice_preserving_progress_guard",
            "e188_only_method_delta_from_e187",
            "e188_reconfigure_mask_at_threshold",
            "e188_critic_retains_accept_deliver_defer_choice",
            "e188_one_retained_witness_step_only_when_progress_empty",
            "e188_retained_witness_interruptible",
            "e188_guard_audit",
        ):
            value.pop(name, None)
        value.update(
            controller_architecture=CONTROLLER_ARCHITECTURE,
            network_architecture=NETWORK_ARCHITECTURE,
            recovery_liveness_guard=GUARD_FREE_CONTRACT,
            e188_guard_contract=GUARD_FREE_CONTRACT,
            e226_guard_intervention_enabled=False,
            e226_reconfigure_mask_enabled=False,
            e226_witness_fallback_enabled=False,
            e226_critic_bypass_enabled=False,
            e226_full_certified_frontier_in_selection=True,
            e226_full_certified_frontier_in_td_targets=True,
            e226_guard_context_features_retained=True,
            e226_passive_nonprogress_counter_retained=True,
            e226_only_behavior_delta_from_e215=True,
            safe_frontier_difference_from_vcg_v1=(
                "none after exact certification; all exact-SAFE actions are scored"
            ),
            policy_difference_from_vcg_v1=(
                "critic selects from the complete exact-SAFE frontier at every "
                "decision; no liveness mask or forced witness exists"
            ),
            bellman_frontier_matches_deployment_shield=True,
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
    ) -> "GuardFreeTrainingQueueAgent":
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
            "e187_value_head_reset": True,
            "e187_optimizer_replay_counters_reset": True,
            "recovery_liveness_guard": GUARD_FREE_CONTRACT,
            "e188_guard_contract": GUARD_FREE_CONTRACT,
            "e226_guard_intervention_enabled": False,
            "e226_reconfigure_mask_enabled": False,
            "e226_witness_fallback_enabled": False,
            "e226_critic_bypass_enabled": False,
            "e226_full_certified_frontier_in_selection": True,
            "e226_full_certified_frontier_in_td_targets": True,
            "e226_only_behavior_delta_from_e215": True,
            "e191_fresh_random_initialization": True,
            "e191_transferred_modules": (),
            "e191_architecture_identical_to_e189": True,
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(f"incompatible E226 checkpoint: {mismatches!r}")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E226 checkpoint exposes the future schedule")
        if payload.get("e187_warm_start_manifest") is not None:
            raise ValueError("E226 checkpoint contains transferred weights")

        agent = cls(
            config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


architecture_signature = e191.architecture_signature
PublicQueueContext = e191.PublicQueueContext
public_queue_context = e191.public_queue_context
attach_queue_context = e191.attach_queue_context
deduplicated_queue_scores = e191.deduplicated_queue_scores
acceleration_enabled = e191.acceleration_enabled


__all__ = [
    "CONTROLLER_ARCHITECTURE", "GUARD_FREE_CONTRACT",
    "GuardFreeTrainingQueueAgent", "NETWORK_ARCHITECTURE",
    "PublicQueueContext", "QUEUE_CONTEXT_DIM", "QUEUE_CONTEXT_NAMES",
    "TIME_CONTEXT_CONTRACT", "acceleration_enabled", "architecture_signature",
    "attach_queue_context", "deduplicated_queue_scores", "public_queue_context",
]

