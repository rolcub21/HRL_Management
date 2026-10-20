"""Liveness-only Defer bounds: no release-age, inbound or due-date preference."""

from dataclasses import dataclass, field

from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import BoundedEventDeferRule, DeferLivenessDecision


@dataclass(frozen=True)
class ProgressOnlyDeferRule(BoundedEventDeferRule):
    version: str = field(default="e122_safe_fixed_horizon_and_persistent_progress_bounds_v1", init=False)

    def evaluate(self, env, yard, *, inbound_present, current_certificate,
                 consecutive_defer_decisions, terminal):
        if terminal:
            return DeferLivenessDecision(False, None, "terminal_state")
        if current_certificate.status is not ViabilityStatus.SAFE:
            return DeferLivenessDecision(False, None, "current_recoverability_not_certified_safe")
        if consecutive_defer_decisions >= self.max_consecutive_defer_decisions:
            return DeferLivenessDecision(False, None, "consecutive_defer_bound_reached")
        if not yard.blocks and not any(not block.delivered for block in env.blocks):
            return DeferLivenessDecision(False, None, "no_unfinished_work")
        return DeferLivenessDecision(True, self.max_option_steps, "e122_bounded_wait_without_timing_exclusions")
