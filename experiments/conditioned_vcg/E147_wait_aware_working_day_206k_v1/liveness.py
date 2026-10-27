"""Working-day liveness without a false deadline during external idle time."""

from contextlib import contextmanager
from dataclasses import dataclass, field

from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import DeferLivenessDecision
from experiments.conditioned_vcg.E121_blocked_pickup_wait_8x8_185k_v1.policy import (
    BlockedPickupWaitRule,
)
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.liveness import (
    available_work,
)
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1 import controller as base


@dataclass(frozen=True)
class WorkingDayDeferRule(BlockedPickupWaitRule):
    """Permit bounded waiting when no released work can physically progress.

    The inherited consecutive-Defer cap remains authoritative whenever any
    released block is available.  A nonterminal state with no released work
    has no controller action other than waiting for an external arrival, so
    exhausting that cap would create an empty frontier rather than liveness.
    This rule reads no future release time or workload label.
    """

    version: str = field(
        default="e147_working_day_external_idle_exception_v1", init=False
    )

    def evaluate(
        self,
        env,
        yard,
        *,
        inbound_present,
        current_certificate,
        consecutive_defer_decisions,
        terminal,
    ):
        if (
            not terminal
            and not inbound_present
            and not available_work(env)
        ):
            if current_certificate.status is not ViabilityStatus.SAFE:
                return DeferLivenessDecision(
                    False, None, "current_recoverability_not_certified_safe"
                )
            return DeferLivenessDecision(
                True,
                self.max_option_steps,
                "external_idle_no_released_work",
            )
        return super().evaluate(
            env,
            yard,
            inbound_present=inbound_present,
            current_certificate=current_certificate,
            consecutive_defer_decisions=consecutive_defer_decisions,
            terminal=terminal,
        )


@contextmanager
def interface(env=None):
    """Retain E136 information adapters; replace only the Defer rule."""

    with base.interface(env):
        yield WorkingDayDeferRule()

