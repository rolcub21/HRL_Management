"""E187 queue representation with E188's guard as the sole delta."""

from __future__ import annotations

from dataclasses import replace

import torch

from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    representation as e187,
)

from . import guard as guard_module
from .guard import ChoicePreservingWitnessGuard, GUARD_CONTRACT


CONTROLLER_ARCHITECTURE = (
    "e188_wait_corrected_queue_choice_preserving_progress_guard_v1"
)
# The neural architecture and inputs are unchanged from the matched E187 arm.
NETWORK_ARCHITECTURE = e187.NETWORK_ARCHITECTURE
TIME_CONTEXT_CONTRACT = e187.TIME_CONTEXT_CONTRACT
QUEUE_CONTEXT_NAMES = e187.QUEUE_CONTEXT_NAMES
QUEUE_CONTEXT_DIM = e187.QUEUE_CONTEXT_DIM
SOURCE_CHECKPOINTS = e187.SOURCE_CHECKPOINTS
TRANSFER_PREFIXES = e187.TRANSFER_PREFIXES
PublicQueueContext = e187.PublicQueueContext
public_queue_context = e187.public_queue_context
attach_queue_context = e187.attach_queue_context
deduplicated_queue_scores = e187.deduplicated_queue_scores
acceleration_enabled = e187.acceleration_enabled
authenticate_sources = e187.authenticate_sources
source_manifest = e187.source_manifest
# Reuse the parent context manager itself. It toggles the inherited class flag
# and therefore suppresses warm start for this subclass in E187's self-check.
warm_start_disabled = e187.warm_start_disabled


class ChoicePreservingQueueAgent(e187.WaitCorrectedQueueAgent):
    """Matched E187 agent whose only behavior change is guard admissibility."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.recovery_witness_guard = ChoicePreservingWitnessGuard(
            max_nonprogress_recovery_decisions=(
                self.config.max_nonprogress_recovery_decisions
            ),
            force_when_due=self.config.force_recovery_witness_when_due,
        )

    def _admissible_prepared(self, snapshot):
        prepared, forced = self.recovery_witness_guard.admissible_prepared(snapshot)
        if self.context_environment is None:
            raise RuntimeError("E188 agent is not bound to its live environment")
        context = public_queue_context(
            self.context_environment,
            snapshot,
            required_jobs=self.context_required_jobs,
            max_steps=self.context_max_steps,
        )
        current = attach_queue_context(snapshot.recovery_state, context.values)
        records = tuple(
            replace(record, current_state=current) for record in prepared.records
        )
        return replace(prepared, records=records), forced

    def select(self, snapshot, *, training=True, epsilon=None):
        decision = super().select(snapshot, training=training, epsilon=epsilon)
        if decision.prepared_snapshot.liveness_restricted:
            self.recovery_witness_guard.record_selection(
                action_type=decision.candidate.action_type.value,
                forced=decision.liveness_forced,
            )
        return decision

    def checkpoint_metadata(self) -> dict:
        result = super().checkpoint_metadata()
        result.update(
            controller_architecture=CONTROLLER_ARCHITECTURE,
            network_architecture=NETWORK_ARCHITECTURE,
            recovery_liveness_guard=GUARD_CONTRACT,
            e188_guard_contract=GUARD_CONTRACT,
            e188_choice_preserving_progress_guard=True,
            e188_only_method_delta_from_e187=True,
            e188_reconfigure_mask_at_threshold=True,
            e188_critic_retains_accept_deliver_defer_choice=True,
            e188_one_retained_witness_step_only_when_progress_empty=True,
            e188_retained_witness_interruptible=True,
            safe_frontier_difference_from_vcg_v1=(
                "exact-safe frontier masked only for bounded liveness"
            ),
            policy_difference_from_vcg_v1=(
                "mask Reconfigure at the E187 nonprogress bound; preserve "
                "critic choice among certified objective actions"
            ),
            e188_guard_audit=self.recovery_witness_guard.audit(),
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
    ) -> "ChoicePreservingQueueAgent":
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
            "recovery_liveness_guard": GUARD_CONTRACT,
            "e188_guard_contract": GUARD_CONTRACT,
            "e188_choice_preserving_progress_guard": True,
            "e188_only_method_delta_from_e187": True,
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatches:
            raise ValueError(f"incompatible E188 checkpoint: {mismatches!r}")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E188 checkpoint exposes the future schedule")
        manifest = payload.get("e187_warm_start_manifest")
        source = SOURCE_CHECKPOINTS.get(int(seed))
        if (
            source is None
            or not isinstance(manifest, dict)
            or manifest.get("sha256") != source["sha256"]
        ):
            raise ValueError("E188 checkpoint has another warm-start identity")

        # Toggle the inherited owner, not a subclass-local flag, so the E187
        # and E188 warm-start suppression contexts remain interoperable.
        previous = e187.WaitCorrectedQueueAgent._suppress_warm_start
        e187.WaitCorrectedQueueAgent._suppress_warm_start = True
        try:
            agent = cls(
                config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
                seed=seed,
                device=device,
                epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
            )
        finally:
            e187.WaitCorrectedQueueAgent._suppress_warm_start = previous
        agent.e187_warm_start_manifest = dict(manifest)
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


__all__ = [
    "CONTROLLER_ARCHITECTURE", "ChoicePreservingQueueAgent",
    "NETWORK_ARCHITECTURE", "PublicQueueContext", "QUEUE_CONTEXT_DIM",
    "QUEUE_CONTEXT_NAMES", "SOURCE_CHECKPOINTS", "TIME_CONTEXT_CONTRACT",
    "TRANSFER_PREFIXES", "acceleration_enabled", "attach_queue_context",
    "authenticate_sources", "deduplicated_queue_scores", "guard_module",
    "public_queue_context", "source_manifest", "warm_start_disabled",
]
