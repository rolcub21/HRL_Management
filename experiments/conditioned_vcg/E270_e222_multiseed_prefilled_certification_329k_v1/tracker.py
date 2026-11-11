"""Matched rankers and certification telemetry for E270."""

from __future__ import annotations

from collections import Counter

from PSLAP.viability import ViabilityStatus
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import (
    CertifiedDynamicAgent,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191_representation,
)
from experiments.conditioned_vcg.E198_e195_shadow_certification_ablation_257k_v1 import (
    frontier as e198_frontier,
)
from viability_graph_hierarchy import RecoveryWitnessMismatch


class CertificationTracker:
    """Read-only exact labels for certified and physical frontiers."""

    def __init__(self, arm: str) -> None:
        self.arm = str(arm)
        self.frontier_calls = 0
        self.physical_candidates = 0
        self.safe_candidates = 0
        self.unsafe_candidates = 0
        self.unknown_candidates = 0
        self.frontiers_with_non_safe = 0
        self.current_status_counts = Counter()
        self.selected_status_counts = Counter()
        self.first_non_safe_current = None
        self.first_exhaustive_unsafe_current = None
        self.first_non_safe_selection = None
        self.first_exhaustive_unsafe_selection = None
        self.guard_witness_unavailable = 0
        self.guard_witness_errors = []
        self.events = []
        self.latest = None

    @staticmethod
    def _certificate_row(certificate) -> dict:
        return {
            "status": certificate.status.value,
            "exhaustive": bool(certificate.exhaustive),
            "reason": str(certificate.reason),
            "witness_length": len(certificate.witness),
        }

    def record_frontier(self, certified, physical=None) -> None:
        self.frontier_calls += 1
        current = certified.current_certificate
        current_row = {
            "kind": "current_state",
            "decision_epoch": int(certified.decision_epoch),
            **self._certificate_row(current),
        }
        self.current_status_counts[current.status.value] += 1
        if current.status is not ViabilityStatus.SAFE:
            if self.first_non_safe_current is None:
                self.first_non_safe_current = dict(current_row)
            if (
                current.status is ViabilityStatus.UNSAFE
                and current.exhaustive
                and self.first_exhaustive_unsafe_current is None
            ):
                self.first_exhaustive_unsafe_current = dict(current_row)
        if physical is not None:
            counts = Counter(
                e198_frontier.shadow_certificate(item).status.value
                for item in physical.candidates
            )
            self.physical_candidates += len(physical.candidates)
            self.safe_candidates += counts[ViabilityStatus.SAFE.value]
            self.unsafe_candidates += counts[ViabilityStatus.UNSAFE.value]
            self.unknown_candidates += counts[ViabilityStatus.UNKNOWN.value]
            self.frontiers_with_non_safe += int(
                counts[ViabilityStatus.UNSAFE.value]
                + counts[ViabilityStatus.UNKNOWN.value] > 0
            )
        self.latest = (certified, physical)

    def record_selection(self, decision) -> None:
        candidate = decision.candidate
        certificate = e198_frontier.shadow_certificate(candidate)
        row = {
            "kind": "selected_successor",
            "selection_index": len(self.events),
            "candidate_key": candidate.key,
            "action_type": candidate.action_type.value,
            "liveness_forced": bool(decision.liveness_forced),
            **self._certificate_row(certificate),
        }
        self.selected_status_counts[certificate.status.value] += 1
        if certificate.status is not ViabilityStatus.SAFE:
            if self.first_non_safe_selection is None:
                self.first_non_safe_selection = dict(row)
            if (
                certificate.status is ViabilityStatus.UNSAFE
                and certificate.exhaustive
                and self.first_exhaustive_unsafe_selection is None
            ):
                self.first_exhaustive_unsafe_selection = dict(row)
        self.events.append(row)

    def record_guard_witness_error(self, error: Exception) -> None:
        self.guard_witness_unavailable += 1
        self.guard_witness_errors.append(f"{type(error).__name__}:{error}")

    def audit(self) -> dict:
        return {
            "arm": self.arm,
            "frontier_calls": int(self.frontier_calls),
            "physical_candidates": int(self.physical_candidates),
            "safe_candidates": int(self.safe_candidates),
            "unsafe_candidates": int(self.unsafe_candidates),
            "unknown_candidates": int(self.unknown_candidates),
            "frontiers_with_non_safe": int(self.frontiers_with_non_safe),
            "current_status_counts": dict(self.current_status_counts),
            "selected_successor_status_counts": dict(self.selected_status_counts),
            "first_non_safe_current": self.first_non_safe_current,
            "first_exhaustive_unsafe_current": self.first_exhaustive_unsafe_current,
            "first_non_safe_selection": self.first_non_safe_selection,
            "first_exhaustive_unsafe_selection": (
                self.first_exhaustive_unsafe_selection
            ),
            "guard_witness_unavailable": int(self.guard_witness_unavailable),
            "guard_witness_errors": list(self.guard_witness_errors),
            "events": list(self.events),
        }


class TrackedE191Agent(e191_representation.FreshChoicePreservingQueueAgent):
    """Frozen E191 critic with local certification telemetry."""

    def configure_e233(self, tracker: CertificationTracker) -> None:
        self.e233_tracker = tracker

    def select(self, snapshot, *, training=True, epsilon=None):
        try:
            decision = super().select(
                snapshot, training=training, epsilon=epsilon
            )
        except RecoveryWitnessMismatch as error:
            self.e233_tracker.record_guard_witness_error(error)
            raise
        self.e233_tracker.record_selection(decision)
        return decision


class PrefillTracker(CertificationTracker):
    """Retain the first current-state certificate as the start-state gate."""

    def __init__(self, arm: str) -> None:
        super().__init__(arm)
        self.initial_current_certificate = None

    def record_frontier(self, certified, physical=None) -> None:
        if self.initial_current_certificate is None:
            self.initial_current_certificate = self._certificate_row(
                certified.current_certificate
            )
        super().record_frontier(certified, physical)

    def audit(self) -> dict:
        value = super().audit()
        value["monitor_is_non_authoritative"] = self.arm in (
            "vcg_physical_unfiltered", "dynamic_physical_unfiltered",
            "a3c_physical_unfiltered",
        )
        value["initial_current_certificate"] = self.initial_current_certificate
        return value


class MatchedDynamicE191Agent(TrackedE191Agent):
    """One Dynamic ranker used unchanged on certified or physical frontiers."""

    MATCHED_POLICY = "e270_identical_dynamic_ranker_frontier_filter_only_v1"

    def configure_e270(self, tracker: PrefillTracker) -> None:
        self.configure_e233(tracker)
        self.e270_dynamic = CertifiedDynamicAgent(self)

    def select(self, snapshot, *, training=True, epsilon=None):
        try:
            decision = self.e270_dynamic.select(
                snapshot, training=False, epsilon=0.0
            )
        except RecoveryWitnessMismatch as error:
            self.e233_tracker.record_guard_witness_error(error)
            raise
        self.e233_tracker.record_selection(decision)
        return decision


__all__ = [
    "CertificationTracker", "MatchedDynamicE191Agent", "PrefillTracker",
    "TrackedE191Agent",
]
