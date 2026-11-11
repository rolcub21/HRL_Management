"""Matched rankers and certification telemetry for E270."""

from __future__ import annotations

from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import (
    CertifiedDynamicAgent,
)
from experiments.conditioned_vcg.E233_certification_causal_ablation_292k_v1.tracker import (
    CertificationTracker,
    TrackedE191Agent,
)
from viability_graph_hierarchy import RecoveryWitnessMismatch


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


__all__ = ["MatchedDynamicE191Agent", "PrefillTracker", "TrackedE191Agent"]
