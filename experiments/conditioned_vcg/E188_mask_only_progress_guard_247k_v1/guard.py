"""Choice-preserving liveness guard for E188."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from PSLAP.viability_candidates import ViabilityActionType
from viability_graph_hierarchy import (
    RecoveryWitnessGuard,
    RecoveryWitnessMismatch,
    prepare_viability_snapshot,
)

GUARD_CONTRACT = (
    "bounded_nonprogress_mask_reconfigure_then_interruptible_"
    "retained_exact_witness_step_v1"
)


def _subset(prepared, indices):
    return replace(
        prepared,
        records=tuple(prepared.records[i] for i in indices),
        source_indices=tuple(prepared.source_indices[i] for i in indices),
        liveness_restricted=True,
    )


class ChoicePreservingWitnessGuard(RecoveryWitnessGuard):
    """Keep E187 counters; mask Reconfigure at the trigger boundary."""

    def __init__(self, *, max_nonprogress_recovery_decisions, force_when_due):
        super().__init__(
            max_nonprogress_recovery_decisions=max_nonprogress_recovery_decisions,
            force_when_due=force_when_due,
        )
        self.masked_frontier_calls = 0
        self.reconfigure_candidates_removed = 0
        self.progress_frontier_calls = 0
        self.fallback_frontier_calls = 0
        self.witness_interruptions = 0
        self.masked_decisions = 0
        self.masked_action_decisions = Counter()
        self.fallback_decisions = 0

    def _triggered(self, snapshot):
        return (
            self.active
            or self.nonprogress_recovery_decisions
            >= self.max_nonprogress_recovery_decisions
            or self._due_trigger(snapshot)
        )

    @staticmethod
    def _action_type(snapshot, prepared, index):
        return snapshot.candidates[prepared.source_indices[index]].action_type

    def _witness_index(self, snapshot, prepared):
        if not self.active and not self._activate(snapshot):
            raise RecoveryWitnessMismatch("E188 last-resort witness is empty")
        expected = self.active_witness[self.witness_cursor]
        matches = tuple(
            i for i, source in enumerate(prepared.source_indices)
            if snapshot.candidates[source].recovery_action == expected
        )
        if len(matches) != 1:
            self.mismatches += 1
            self.reset(reset_nonprogress=False)
            raise RecoveryWitnessMismatch(
                "E188 retained witness did not bind uniquely "
                f"(matches={len(matches)})"
            )
        return matches[0]

    def admissible_prepared(self, snapshot):
        prepared = prepare_viability_snapshot(
            snapshot,
            guard_context=self.feature_context(snapshot, forced_frontier=False),
        )
        if not prepared.records or not self._triggered(snapshot):
            return prepared, False
        objective = tuple(
            i for i in range(len(prepared.records))
            if self._action_type(snapshot, prepared, i)
            is not ViabilityActionType.RECONFIGURE
        )
        removed = len(prepared.records) - len(objective)
        if objective:
            if self.active:
                self.reset(reset_nonprogress=False)
                self.witness_interruptions += 1
                prepared = prepare_viability_snapshot(
                    snapshot,
                    guard_context=self.feature_context(
                        snapshot, forced_frontier=False
                    ),
                )
                objective = tuple(
                    i for i in range(len(prepared.records))
                    if self._action_type(snapshot, prepared, i)
                    is not ViabilityActionType.RECONFIGURE
                )
                removed = len(prepared.records) - len(objective)
            if not removed:
                return prepared, False
            self.masked_frontier_calls += 1
            self.progress_frontier_calls += 1
            self.reconfigure_candidates_removed += removed
            return _subset(prepared, objective), False
        if not self.active:
            self._activate(snapshot)
        prepared = prepare_viability_snapshot(
            snapshot,
            guard_context=self.feature_context(snapshot, forced_frontier=True),
        )
        index = self._witness_index(snapshot, prepared)
        self.masked_frontier_calls += 1
        self.fallback_frontier_calls += 1
        self.reconfigure_candidates_removed += max(len(prepared.records) - 1, 0)
        return _subset(prepared, (index,)), True

    def record_selection(self, *, action_type, forced):
        if forced:
            self.fallback_decisions += 1
        else:
            self.masked_decisions += 1
            self.masked_action_decisions[str(action_type)] += 1

    def state_dict(self):
        result = super().state_dict()
        result.update(
            guard_contract=GUARD_CONTRACT,
            masked_frontier_calls=self.masked_frontier_calls,
            reconfigure_candidates_removed=self.reconfigure_candidates_removed,
            progress_frontier_calls=self.progress_frontier_calls,
            fallback_frontier_calls=self.fallback_frontier_calls,
            witness_interruptions=self.witness_interruptions,
            masked_decisions=self.masked_decisions,
            masked_action_decisions=dict(self.masked_action_decisions),
            fallback_decisions=self.fallback_decisions,
        )
        return result

    def load_state_dict(self, state):
        if state.get("guard_contract") != GUARD_CONTRACT:
            raise ValueError("E188 guard checkpoint identity mismatch")
        super().load_state_dict(state)
        for name in (
            "masked_frontier_calls", "reconfigure_candidates_removed",
            "progress_frontier_calls", "fallback_frontier_calls",
            "witness_interruptions", "masked_decisions", "fallback_decisions",
        ):
            setattr(self, name, int(state.get(name, 0)))
        self.masked_action_decisions = Counter(
            state.get("masked_action_decisions", {})
        )

    def audit_dict(self):
        result = super().audit_dict()
        result.update(
            guard_contract=GUARD_CONTRACT,
            masked_frontier_calls=self.masked_frontier_calls,
            reconfigure_candidates_removed=self.reconfigure_candidates_removed,
            progress_frontier_calls=self.progress_frontier_calls,
            fallback_frontier_calls=self.fallback_frontier_calls,
            witness_interruptions=self.witness_interruptions,
            masked_decisions=self.masked_decisions,
            masked_action_decisions=dict(self.masked_action_decisions),
            fallback_decisions=self.fallback_decisions,
        )
        return result

    def audit(self):
        return self.audit_dict()


__all__ = ["ChoicePreservingWitnessGuard", "GUARD_CONTRACT"]
