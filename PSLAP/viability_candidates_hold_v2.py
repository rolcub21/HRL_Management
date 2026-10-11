"""Isolated V2 exact frontier with certified, bounded Hold candidates.

The frozen V1 frontier is used unchanged for Accept/Recover certification.
Its single Defer candidate is removed and replaced by a small family of
event-interruptible Hold horizons.  Hold remains the existing ``DEFER`` action
type/mode, so this module introduces no fourth temporal mode.

Safety and liveness have deliberately separate contracts:

* physical safety is inherited from the exact current-state recovery
  certificate because a no-event WAIT stutters the physical configuration;
* any observable inventory/status change interrupts Hold and requires a fresh
  exact frontier;
* a primitive-step idle budget bounds repeated Holds independently of how the
  horizon family is represented; and
* due work, an exhausted budget, an unsafe/unknown current certificate, or an
  active recovery-witness force all remove Hold fail closed.

No future arrival time or unarrived processing requirement is read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import floor, isfinite
from time import perf_counter
from typing import Iterable, MutableMapping, Optional

from example.Options.ExplicitHoldOption import ExplicitHoldOption
from PSLAP.viability import RecoverabilityCertificate, ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityCertificateCache,
    ViabilityMode,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import CertificateCacheKey, ViabilitySearchConfig
from PSLAP.viability_prioritizer import RecoveryStatePrioritizer


CERTIFIED_HOLD_INTERFACE_V2 = (
    "strict_parameterized_viability_constrained_action_frontier_with_hold_v2"
)
CERTIFIED_HOLD_RULE_V2 = (
    "positive_slack_event_interruptible_primitive_idle_budget_hold_v2"
)
ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2 = (
    "current_exact_safe_physical_certificate_preserved_by_no_event_wait_"
    "and_recertified_on_observed_change_v2"
)
HIDDEN_SCHEDULE_CONTRACT_V2 = (
    "online_status_stored_clock_and_exact_macro_duration_only_no_future_schedule_v2"
)
HOLD_ACTION_KEY_PREFIX = "hold"

# This rule is passed only to the frozen V1 enumerator to suppress its V1
# Defer candidate.  Accept/Recover construction is independent of the streak.
_V1_DEFER_DISABLED_RULE = BoundedEventDeferRule(
    max_option_steps=1,
    max_consecutive_defer_decisions=1,
)


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class PrimitiveIdleBudgetStateV2:
    """Observable cross-macro liveness state measured in primitive steps."""

    max_idle_steps: int = 20
    idle_steps_since_progress: int = 0

    def __post_init__(self) -> None:
        maximum = _positive_int(self.max_idle_steps, name="max_idle_steps")
        used = _nonnegative_int(
            self.idle_steps_since_progress,
            name="idle_steps_since_progress",
        )
        if used > maximum:
            raise ValueError("idle steps cannot exceed the configured budget")

    @property
    def remaining_steps(self) -> int:
        return int(self.max_idle_steps - self.idle_steps_since_progress)

    @property
    def remaining_fraction(self) -> float:
        return float(self.remaining_steps) / float(self.max_idle_steps)

    def after_macro(
        self,
        action_type: ViabilityActionType | str,
        *,
        duration: int,
        observed_event: bool = False,
        exact_rank_progress: bool = False,
        recovery_witness_forced: bool = False,
    ) -> "PrimitiveIdleBudgetStateV2":
        """Advance the liveness clock after one successful strict macro.

        A nonprogress Reconfigure deliberately does not reset the idle clock;
        otherwise alternating Hold/Reconfigure could evade the bound.  A
        retained exact-witness step is authenticated structural progress.
        """

        action_type = ViabilityActionType(action_type)
        duration = _positive_int(duration, name="duration")
        observed_event = bool(observed_event)
        exact_rank_progress = bool(exact_rank_progress)
        recovery_witness_forced = bool(recovery_witness_forced)
        if observed_event and action_type is not ViabilityActionType.DEFER:
            raise ValueError("observed_event is meaningful only after Hold")

        if action_type is ViabilityActionType.DEFER:
            if observed_event:
                used = 0
            else:
                used = self.idle_steps_since_progress + duration
                if used > self.max_idle_steps:
                    raise ValueError(
                        "executed Hold exceeded the remaining primitive idle budget"
                    )
        elif action_type in (
            ViabilityActionType.ACCEPT,
            ViabilityActionType.DELIVER,
        ):
            used = 0
        elif exact_rank_progress or recovery_witness_forced:
            used = 0
        else:
            used = self.idle_steps_since_progress
        return replace(self, idle_steps_since_progress=int(used))


@dataclass(frozen=True)
class CertifiedHoldDecisionV2:
    allowed: bool
    horizons: tuple[int, ...]
    reason: str
    nearest_positive_deadline_steps: Optional[float]
    critical_delivery_start_horizon_steps: Optional[int]
    maximum_horizon_steps: Optional[int]

    def __post_init__(self) -> None:
        horizons = tuple(int(value) for value in self.horizons)
        object.__setattr__(self, "horizons", horizons)
        if horizons != tuple(sorted(set(horizons))):
            raise ValueError("Hold horizons must be sorted and unique")
        if any(value <= 0 for value in horizons):
            raise ValueError("Hold horizons must be positive")
        if self.allowed != bool(horizons):
            raise ValueError("allowed must agree with the horizon family")
        if self.allowed:
            if self.maximum_horizon_steps != max(horizons):
                raise ValueError("maximum horizon disagrees with Hold family")
        elif self.maximum_horizon_steps is not None:
            raise ValueError("rejected Hold cannot expose a maximum horizon")


@dataclass(frozen=True)
class CertifiedHoldRuleV2:
    """Construct a small online, positive-slack Hold horizon family."""

    max_option_steps: int = 10
    max_idle_steps: int = 20
    include_one_step: bool = True
    version: str = CERTIFIED_HOLD_RULE_V2

    def __post_init__(self) -> None:
        _positive_int(self.max_option_steps, name="max_option_steps")
        _positive_int(self.max_idle_steps, name="max_idle_steps")
        if not isinstance(self.include_one_step, bool):
            raise TypeError("include_one_step must be boolean")
        if self.version != CERTIFIED_HOLD_RULE_V2:
            raise ValueError("CertifiedHoldRuleV2 version is immutable")

    @staticmethod
    def _unfinished_external_work(env) -> bool:
        # Intentionally online-only.  Do not add arrival_step or processing
        # fields here: absence/status is observable, the future schedule is not.
        return any(
            block.position is None
            and not block.carrying
            and not block.stored
            and not block.delivered
            for block in env.blocks
        )

    @staticmethod
    def _critical_delivery_start_horizon(snapshot) -> Optional[int]:
        remaining_by_label = {
            str(block.label): float(block.remaining_time)
            for block in snapshot.yard.blocks
        }
        positive = []
        for candidate in snapshot.candidates:
            if candidate.action_type is not ViabilityActionType.DELIVER:
                continue
            remaining = remaining_by_label.get(str(candidate.target_label))
            action = candidate.recovery_action
            if remaining is None or action is None:
                continue
            slack = remaining - int(action.steps)
            if isfinite(slack) and slack >= 1.0:
                positive.append(int(floor(slack)))
        return min(positive) if positive else None

    def evaluate(
        self,
        env,
        snapshot: ViabilityCandidateSnapshot,
        *,
        idle_budget: PrimitiveIdleBudgetStateV2,
        remaining_episode_steps: int,
        recovery_witness_forced: bool,
    ) -> CertifiedHoldDecisionV2:
        if not isinstance(snapshot, ViabilityCandidateSnapshot):
            raise TypeError("snapshot must be a ViabilityCandidateSnapshot")
        if not isinstance(idle_budget, PrimitiveIdleBudgetStateV2):
            raise TypeError("idle_budget must be PrimitiveIdleBudgetStateV2")
        if idle_budget.max_idle_steps != self.max_idle_steps:
            raise ValueError("Hold rule and idle budget maximum disagree")
        remaining_episode_steps = _nonnegative_int(
            remaining_episode_steps, name="remaining_episode_steps"
        )
        recovery_witness_forced = bool(recovery_witness_forced)

        def rejected(reason, nearest=None):
            return CertifiedHoldDecisionV2(
                False, (), reason, nearest, None, None
            )

        if snapshot.audit.terminal:
            return rejected("terminal_state")
        if recovery_witness_forced:
            return rejected("exact_recovery_witness_forced")
        if snapshot.current_certificate.status is not ViabilityStatus.SAFE:
            return rejected("current_recoverability_not_certified_safe")
        if idle_budget.remaining_steps <= 0:
            return rejected("primitive_idle_budget_exhausted")
        if remaining_episode_steps <= 0:
            return rejected("episode_horizon_exhausted")

        stored = tuple(snapshot.yard.blocks)
        inbound_present = snapshot.inbound_label is not None
        if stored:
            nearest = min(float(block.remaining_time) for block in stored)
            if not isfinite(nearest):
                return rejected("stored_deadline_not_finite")
            if nearest <= 0.0:
                return rejected("admitted_work_due_requires_recovery_progress", nearest)
            deadline_cap = int(floor(nearest))
            if deadline_cap <= 0:
                return rejected("no_positive_whole_step_before_deadline", nearest)
        else:
            nearest = None
            deadline_cap = self.max_option_steps
            if inbound_present:
                return rejected("inbound_without_stored_work_requires_accept")
            if not self._unfinished_external_work(env):
                return rejected("no_unfinished_work")

        cap = min(
            self.max_option_steps,
            idle_budget.remaining_steps,
            remaining_episode_steps,
            deadline_cap,
        )
        if cap <= 0:
            return rejected("no_positive_certified_hold_horizon", nearest)

        critical = self._critical_delivery_start_horizon(snapshot)
        horizons = {int(cap)}
        if self.include_one_step:
            horizons.add(1)
        if critical is not None:
            horizons.add(min(int(critical), int(cap)))
        horizons = tuple(sorted(value for value in horizons if 1 <= value <= cap))
        reason = (
            "bounded_inbound_hold_with_positive_stored_slack"
            if inbound_present
            else (
                "bounded_hold_with_positive_stored_slack"
                if stored
                else "bounded_hold_for_observable_external_event"
            )
        )
        return CertifiedHoldDecisionV2(
            True,
            horizons,
            reason,
            nearest,
            critical,
            max(horizons),
        )


@dataclass(frozen=True)
class CertifiedHoldAuditV2:
    interface: str
    rule: str
    robust_stutter_contract: str
    hidden_schedule_contract: str
    hidden_schedule_fields_read: tuple[str, ...]
    physical_certificate_status: str
    recovery_witness_forced: bool
    inbound_present: bool
    idle_budget_max_steps: int
    idle_steps_since_progress: int
    idle_budget_remaining_steps: int
    remaining_episode_steps: int
    hold_allowed: bool
    hold_horizons: tuple[int, ...]
    hold_reason: str
    nearest_positive_deadline_steps: Optional[float]
    critical_delivery_start_horizon_steps: Optional[int]
    hold_candidate_count: int

    def audit_dict(self) -> dict:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CertifiedHoldCandidateSnapshotV2:
    """V2 snapshot plus explicit idle-budget/liveness evidence."""

    frontier: ViabilityCandidateSnapshot
    hold_audit: CertifiedHoldAuditV2

    @property
    def candidates(self):
        return self.frontier.candidates

    @property
    def audit(self):
        return self.frontier.audit

    @property
    def recovery_state(self):
        return self.frontier.recovery_state

    @property
    def current_certificate(self):
        return self.frontier.current_certificate

    @property
    def decision_epoch(self):
        return self.frontier.decision_epoch

    @property
    def episode_instance_id(self):
        return self.frontier.episode_instance_id

    @property
    def inbound_label(self):
        return self.frontier.inbound_label

    def candidates_for_mode(self, mode):
        return self.frontier.candidates_for_mode(mode)

    def candidates_of_type(self, action_type):
        return self.frontier.candidates_of_type(action_type)


def enumerate_viability_candidates_hold_v2(
    env,
    *,
    idle_budget: PrimitiveIdleBudgetStateV2,
    remaining_episode_steps: int,
    recovery_witness_forced: bool,
    search_config: ViabilitySearchConfig = ViabilitySearchConfig(),
    hold_rule: CertifiedHoldRuleV2 = CertifiedHoldRuleV2(),
    reserved_cells: Iterable[tuple[int, int]] = (),
    cache: Optional[
        MutableMapping[CertificateCacheKey, RecoverabilityCertificate]
    ] = None,
    max_replans: int = 8,
    state_prioritizer: Optional[RecoveryStatePrioritizer] = None,
) -> CertifiedHoldCandidateSnapshotV2:
    """Return the unchanged exact V1 physical frontier plus certified Holds."""

    if not isinstance(idle_budget, PrimitiveIdleBudgetStateV2):
        raise TypeError("idle_budget must be PrimitiveIdleBudgetStateV2")
    if not isinstance(hold_rule, CertifiedHoldRuleV2):
        raise TypeError("hold_rule must be CertifiedHoldRuleV2")
    started = perf_counter()
    base = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=1,
        search_config=search_config,
        liveness_rule=_V1_DEFER_DISABLED_RULE,
        reserved_cells=reserved_cells,
        cache=cache,
        max_replans=max_replans,
        state_prioritizer=state_prioritizer,
    )
    physical = tuple(
        candidate
        for candidate in base.candidates
        if candidate.action_type is not ViabilityActionType.DEFER
    )
    if len(physical) != len(base.candidates):
        raise RuntimeError("frozen V1 Defer suppression failed closed")

    decision = hold_rule.evaluate(
        env,
        base,
        idle_budget=idle_budget,
        remaining_episode_steps=remaining_episode_steps,
        recovery_witness_forced=recovery_witness_forced,
    )
    holds = []
    for horizon in decision.horizons:
        option = ExplicitHoldOption(
            env,
            horizon_steps=int(horizon),
            idle_budget_steps=idle_budget.remaining_steps,
        )
        if not option.initiation(env.get_current_state()):
            raise RuntimeError("certified Hold option is not executable at its boundary")
        holds.append(
            ViabilityActionCandidate(
                key=f"{HOLD_ACTION_KEY_PREFIX}:{int(horizon):04d}",
                mode=ViabilityMode.DEFER,
                action_type=ViabilityActionType.DEFER,
                target_label=None,
                source=tuple(base.agent_position),
                destination=None,
                successor_state=base.recovery_state,
                certificate=base.current_certificate,
                option=option,
                horizon_steps=int(horizon),
                recovery_rank_before=base.audit.current_recovery_rank,
                recovery_rank_after=base.audit.current_recovery_rank,
            )
        )
    candidates = physical + tuple(holds)
    updated_audit = replace(
        base.audit,
        interface=CERTIFIED_HOLD_INTERFACE_V2,
        liveness_rule=hold_rule.version,
        defer_allowed=decision.allowed,
        defer_horizon_steps=decision.maximum_horizon_steps,
        defer_reason=decision.reason,
        # This V1 compatibility field is not the V2 liveness authority.  The
        # canonical primitive counter is in ``hold_audit`` below.
        consecutive_defer_decisions=0,
        candidate_count=len(candidates),
        defer_candidate_count=len(holds),
        # ``started`` precedes the frozen V1 call, so elapsed wall time already
        # includes its exact analysis.  Adding ``base.audit.analysis_seconds``
        # here would count that work twice.
        analysis_seconds=perf_counter() - started,
    )
    frontier = replace(base, candidates=candidates, audit=updated_audit)
    hold_audit = CertifiedHoldAuditV2(
        interface=CERTIFIED_HOLD_INTERFACE_V2,
        rule=hold_rule.version,
        robust_stutter_contract=ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
        hidden_schedule_contract=HIDDEN_SCHEDULE_CONTRACT_V2,
        hidden_schedule_fields_read=(),
        physical_certificate_status=base.current_certificate.status.value,
        recovery_witness_forced=bool(recovery_witness_forced),
        inbound_present=base.inbound_label is not None,
        idle_budget_max_steps=idle_budget.max_idle_steps,
        idle_steps_since_progress=idle_budget.idle_steps_since_progress,
        idle_budget_remaining_steps=idle_budget.remaining_steps,
        remaining_episode_steps=int(remaining_episode_steps),
        hold_allowed=decision.allowed,
        hold_horizons=decision.horizons,
        hold_reason=decision.reason,
        nearest_positive_deadline_steps=(
            decision.nearest_positive_deadline_steps
        ),
        critical_delivery_start_horizon_steps=(
            decision.critical_delivery_start_horizon_steps
        ),
        hold_candidate_count=len(holds),
    )
    return CertifiedHoldCandidateSnapshotV2(frontier, hold_audit)


__all__ = [
    "CERTIFIED_HOLD_INTERFACE_V2",
    "CERTIFIED_HOLD_RULE_V2",
    "HIDDEN_SCHEDULE_CONTRACT_V2",
    "HOLD_ACTION_KEY_PREFIX",
    "ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2",
    "CertifiedHoldAuditV2",
    "CertifiedHoldCandidateSnapshotV2",
    "CertifiedHoldDecisionV2",
    "CertifiedHoldRuleV2",
    "PrimitiveIdleBudgetStateV2",
    "enumerate_viability_candidates_hold_v2",
]
