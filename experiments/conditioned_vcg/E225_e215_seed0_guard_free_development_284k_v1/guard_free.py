"""Exact-SAFE frontier adapter with no liveness masking or forced action."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from PSLAP.viability_candidates import ViabilityActionType
from viability_graph_hierarchy import prepare_viability_snapshot
from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    representation as e188_representation,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191_representation,
)


class FrozenGuardFreeQueueAgent(
    e191_representation.FreshChoicePreservingQueueAgent
):
    """Load E215 weights but expose every exact-SAFE action to the critic.

    The four frozen guard-context inputs remain in the network input because
    removing them would change the trained representation.  Their passive
    non-progress counter continues to update, but it never masks a candidate,
    activates a witness, or chooses an action.  Thus every executed action is
    selected by the frozen critic (or is the sole certified action).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.guard_free_prepared_frontier_calls = 0
        self.guard_free_frontiers = 0
        self.guard_free_threshold_frontiers = 0
        self.guard_free_would_mask_frontiers = 0
        self.guard_free_would_force_frontiers = 0
        self.guard_free_safe_candidates = 0
        self.guard_free_safe_reconfigure_candidates = 0
        self.guard_free_selected_actions = Counter()
        self.guard_free_threshold_selected_actions = Counter()
        self.guard_free_reconfigure_decisions = 0
        self.guard_free_singleton_reconfigure_decisions = 0
        self.guard_free_critic_reconfigure_decisions = 0
        self.guard_free_decision_records: list[dict] = []
        self._guard_free_pending: dict | None = None

    @staticmethod
    def _action_type(snapshot, prepared, index):
        return snapshot.candidates[prepared.source_indices[index]].action_type

    def _admissible_prepared(self, snapshot):
        guard = self.recovery_witness_guard
        # Preserve the observation presented during training, but explicitly
        # report a non-forced frontier.  No guard admissibility function is
        # called on this path.
        prepared = prepare_viability_snapshot(
            snapshot,
            guard_context=guard.feature_context(
                snapshot, forced_frontier=False
            ),
        )
        if self.context_environment is None:
            raise RuntimeError("E225 agent is not bound to its live environment")

        action_types = tuple(
            self._action_type(snapshot, prepared, index)
            for index in range(len(prepared.records))
        )
        reconfigure = sum(
            value is ViabilityActionType.RECONFIGURE for value in action_types
        )
        objective = len(action_types) - reconfigure
        threshold = bool(
            guard.active
            or guard.nonprogress_recovery_decisions
            >= guard.max_nonprogress_recovery_decisions
            or guard._due_trigger(snapshot)
        )
        would_mask = bool(threshold and objective and reconfigure)
        would_force = bool(threshold and action_types and not objective)

        self.guard_free_prepared_frontier_calls += 1
        self._guard_free_pending = {
            "threshold_reached": threshold,
            "old_guard_would_mask_reconfigure": would_mask,
            "old_guard_would_force_witness": would_force,
            "exact_safe_candidates": len(action_types),
            "safe_reconfigure_candidates": reconfigure,
            "safe_nonreconfigure_candidates": objective,
            "nonprogress_recovery_decisions": int(
                guard.nonprogress_recovery_decisions
            ),
        }

        context = e188_representation.public_queue_context(
            self.context_environment,
            snapshot,
            required_jobs=self.context_required_jobs,
            max_steps=self.context_max_steps,
        )
        current = e188_representation.attach_queue_context(
            snapshot.recovery_state, context.values
        )
        records = tuple(
            replace(record, current_state=current) for record in prepared.records
        )
        # prepare_viability_snapshot already returns liveness_restricted=False.
        return replace(prepared, records=records), False

    def select(self, snapshot, *, training=True, epsilon=None):
        decision = super().select(
            snapshot, training=training, epsilon=epsilon
        )
        if decision.liveness_forced:
            raise RuntimeError("E225 guard-free replay received a forced action")
        if decision.prepared_snapshot.liveness_restricted:
            raise RuntimeError("E225 guard-free replay received a masked frontier")
        if self._guard_free_pending is None:
            raise RuntimeError("E225 guard-free frontier telemetry is missing")

        self.guard_free_frontiers += 1
        self.guard_free_threshold_frontiers += int(
            self._guard_free_pending["threshold_reached"]
        )
        self.guard_free_would_mask_frontiers += int(
            self._guard_free_pending["old_guard_would_mask_reconfigure"]
        )
        self.guard_free_would_force_frontiers += int(
            self._guard_free_pending["old_guard_would_force_witness"]
        )
        self.guard_free_safe_candidates += int(
            self._guard_free_pending["exact_safe_candidates"]
        )
        self.guard_free_safe_reconfigure_candidates += int(
            self._guard_free_pending["safe_reconfigure_candidates"]
        )
        action = str(decision.candidate.action_type.value)
        self.guard_free_selected_actions[action] += 1
        if self._guard_free_pending["threshold_reached"]:
            self.guard_free_threshold_selected_actions[action] += 1
        if decision.candidate.action_type is ViabilityActionType.RECONFIGURE:
            self.guard_free_reconfigure_decisions += 1
            if self._guard_free_pending["exact_safe_candidates"] == 1:
                self.guard_free_singleton_reconfigure_decisions += 1
            else:
                self.guard_free_critic_reconfigure_decisions += 1

        self.guard_free_decision_records.append({
            **self._guard_free_pending,
            "decision_epoch": int(snapshot.decision_epoch),
            "selected_key": str(decision.candidate.key),
            "selected_action_type": action,
            "selection_source": str(decision.selection_source),
            "liveness_forced": False,
            "liveness_restricted": False,
        })
        self._guard_free_pending = None
        return decision

    def guard_free_audit(self) -> dict:
        return {
            "contract": "exact_safe_frontier_critic_only_no_guard_v1",
            "prepared_frontier_calls": self.guard_free_prepared_frontier_calls,
            "frontiers": self.guard_free_frontiers,
            "threshold_frontiers": self.guard_free_threshold_frontiers,
            "old_guard_would_mask_frontiers": (
                self.guard_free_would_mask_frontiers
            ),
            "old_guard_would_force_frontiers": (
                self.guard_free_would_force_frontiers
            ),
            "safe_candidates": self.guard_free_safe_candidates,
            "safe_reconfigure_candidates": (
                self.guard_free_safe_reconfigure_candidates
            ),
            "selected_actions": dict(self.guard_free_selected_actions),
            "threshold_selected_actions": dict(
                self.guard_free_threshold_selected_actions
            ),
            "reconfigure_decisions": self.guard_free_reconfigure_decisions,
            "singleton_reconfigure_decisions": (
                self.guard_free_singleton_reconfigure_decisions
            ),
            "critic_reconfigure_decisions": (
                self.guard_free_critic_reconfigure_decisions
            ),
            "forced_decisions": 0,
            "masked_frontiers": 0,
        }


__all__ = ["FrozenGuardFreeQueueAgent"]

