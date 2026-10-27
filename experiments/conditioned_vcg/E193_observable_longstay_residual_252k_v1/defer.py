"""Observable, event-based Defer semantics for E193's storage clock.

This module is deliberately local to E193.  The frozen V1 frontier still
constructs and certifies every physical candidate; E193 patches only its
Defer rule and execution object while its adapter is active.

The controller may wait until the earliest *currently knowable* event:

* a stored block's delivery window opens; or
* a not-yet-observed job is released.

An exact future release timestamp is never read.  When an unreleased job
exists, the option is bound by the episode error boundary and is interrupted
as soon as the job becomes observable.  The declared workload size reveals
that some release remains, but not when it will occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from example.Options.ExplicitDeferOption import ExplicitDeferOption
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    DeferLivenessDecision,
)


DEFER_VERSION = "e193_observable_release_or_storage_window_defer_v1"
OPTION_VERSION = "e193_observable_release_or_storage_window_option_v1"


def observed_labels(env) -> frozenset[str]:
    """Return labels whose existence is public at the current boundary.

    In particular, this function never reads ``arrival_step``.  A job becomes
    observable only through its current inventory status/position.
    """

    return frozenset(
        str(block.label)
        for block in env.blocks
        if (
            block.position is not None
            or bool(block.carrying)
            or bool(block.stored)
            or bool(block.delivered)
        )
    )


def unresolved_release_exists(env) -> bool:
    """Whether the declared finite workload contains an unseen job."""

    return len(observed_labels(env)) < len(env.blocks)


def next_storage_window_open(
    env,
    *,
    at_time: int | None = None,
    delivery_window: int = 20,
) -> int | None:
    """Earliest future ``stored_time + duration - window`` among stored jobs.

    Only a stored job has a storage-clock deadline.  The order of the guards
    below is intentional: no duration or stored timestamp is inspected for an
    unreleased/outside job.
    """

    now = int(env.time_steps if at_time is None else at_time)
    window = int(delivery_window)
    if window < 0:
        raise ValueError("delivery_window must be non-negative")

    openings: list[int] = []
    for block in env.blocks:
        if not bool(block.stored) or bool(block.delivered):
            continue
        stored = getattr(block, "stored_time_step", None)
        if stored is None:
            raise RuntimeError("a stored block has no storage-clock start")
        opening = int(stored) + int(block.storage_steps_needed) - window
        if opening > now:
            openings.append(opening)
    return min(openings) if openings else None


class ObservableLongStayDeferOption(ExplicitDeferOption):
    """WAIT to a known stored-window boundary or an observed release."""

    VERSION = OPTION_VERSION
    EVENT_CONTRACT = "observable_inventory_release_or_known_storage_window_v1"

    def __init__(self, env, horizon_steps: int):
        self.initial_observed_labels = observed_labels(env)
        opening = next_storage_window_open(env)
        endpoint = int(env.time_steps) + int(horizon_steps)
        self.boundary_reason = (
            "storage_window_open"
            if opening is not None and endpoint == int(opening)
            else "episode_error_boundary"
        )
        super().__init__(env, horizon_steps=int(horizon_steps))

    def _observed_signature(self):
        # The base signature already contains all visible positions/statuses.
        # Including the label set makes the intended release event explicit
        # and remains independent of every future timestamp.
        return (observed_labels(self.env), super()._observed_signature())

    def termination(self, state):
        del state
        if not self.started:
            return False
        if self.failed:
            self._record(success=False, reason=self.failure_reason)
            return True
        if self.env.is_state_terminal(self.env.current_state):
            self._record(success=True, reason="environment_terminal")
            return True
        if any(block.carrying for block in self.env.blocks):
            self.failed = True
            self.failure_reason = "carry_state_changed"
            self._record(success=False, reason=self.failure_reason)
            return True
        if observed_labels(self.env) != self.initial_observed_labels:
            # Keep the inherited runner's generic event spelling so its
            # diagnostic counter resets without any shared-code change.
            self._record(success=True, reason="observed_event")
            return True
        if self.steps >= self.horizon_steps:
            self._record(success=True, reason=self.boundary_reason)
            return True
        return False


@dataclass(frozen=True)
class ObservableLongStayDeferRule(BoundedEventDeferRule):
    """Expose one Defer that jumps to the next observable/known event.

    ``max_option_steps`` and ``max_consecutive_defer_decisions`` are retained
    only for subtype compatibility with the frozen frontier.  This rule never
    consults either value: finite-event liveness replaces the numeric streak.
    """

    max_option_steps: int = 1
    max_consecutive_defer_decisions: int = 1
    episode_horizon_steps: int = 9_000
    delivery_window: int = 20
    version: str = field(default=DEFER_VERSION, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.episode_horizon_steps, bool)
            or not isinstance(self.episode_horizon_steps, int)
            or self.episode_horizon_steps <= 0
        ):
            raise ValueError("episode_horizon_steps must be a positive integer")
        if (
            isinstance(self.delivery_window, bool)
            or not isinstance(self.delivery_window, int)
            or self.delivery_window < 0
        ):
            raise ValueError("delivery_window must be a non-negative integer")

    def evaluate(
        self,
        env,
        yard,
        *,
        inbound_present,
        current_certificate,
        consecutive_defer_decisions,
        terminal,
    ) -> DeferLivenessDecision:
        # These inputs are intentionally irrelevant to eligibility.  Inbound
        # waiting is priced by J, while the old streak is hidden controller
        # state that E193 removes from the decision problem.
        del yard, inbound_present, consecutive_defer_decisions

        if terminal:
            return DeferLivenessDecision(False, None, "terminal_state")
        if current_certificate.status is not ViabilityStatus.SAFE:
            return DeferLivenessDecision(
                False, None, "current_recoverability_not_certified_safe"
            )

        now = int(env.time_steps)
        episode_remaining = int(self.episode_horizon_steps) - now
        if episode_remaining <= 0:
            return DeferLivenessDecision(
                False, None, "episode_error_boundary_reached"
            )

        opening = next_storage_window_open(
            env,
            at_time=now,
            delivery_window=self.delivery_window,
        )
        unseen_release = unresolved_release_exists(env)
        if opening is None and not unseen_release:
            return DeferLivenessDecision(False, None, "no_future_public_event")

        if opening is None:
            horizon = episode_remaining
            reason = "wait_for_observed_release"
        else:
            horizon = min(int(opening) - now, episode_remaining)
            reason = (
                "wait_for_storage_window_or_observed_release"
                if unseen_release
                else "wait_for_storage_window"
            )
        if horizon <= 0:  # defensive; strict-future openings are positive
            return DeferLivenessDecision(False, None, "no_positive_event_horizon")
        return DeferLivenessDecision(True, int(horizon), reason)


__all__ = [
    "DEFER_VERSION",
    "OPTION_VERSION",
    "ObservableLongStayDeferOption",
    "ObservableLongStayDeferRule",
    "next_storage_window_open",
    "observed_labels",
    "unresolved_release_exists",
]
