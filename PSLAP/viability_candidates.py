"""Policy-independent viable macro candidates at strict decision boundaries.

This module is the action-interface layer between the exact recovery model in
``PSLAP.viability`` and an eventual constrained SMDP controller.  It does not
score, rank, or select an action.  In particular, no REG, dynamic-PSLAP,
nearest-free, urgency, or GA policy is consulted.

Only successor states certified ``SAFE`` enter the returned frontier.  An
``UNSAFE`` result and a computational ``UNKNOWN`` result are both fail-closed;
their counts remain visible in the immutable audit.  The resulting option
objects are bound to the current episode and decision epoch, so a snapshot is
one-shot and must not be reused after the environment advances.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from time import perf_counter
from typing import Iterable, Optional

from example.Options.DirectDeliverOption import DirectDeliverOption
from example.Options.ExplicitDeferOption import ExplicitDeferOption
from example.Options.ExplicitAcceptOption import ExplicitAcceptOption
from example.Options.ReconfigureOption import ReconfigureOption
from PSLAP.dynamic_yard import BlockView, Cell, YardSnapshot
from PSLAP.neutral_protocol import shared_candidate_mask
from PSLAP.viability import (
    RecoverabilityCertificate,
    RecoveryAction,
    RecoveryActionKind,
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
    apply_recovery_action,
    legal_recovery_actions,
)
from PSLAP.viability_filter import (
    CertificateCacheKey,
    POST_ACCEPT_CERTIFICATION_CONTRACT,
    ROBUST_QUEUE_OBSTACLE_CONTRACT,
    CandidateViabilityReport,
    ViabilitySearchConfig,
    certify_post_accept_candidates,
    online_fixed_obstacles,
)
from PSLAP.viability_prioritizer import (
    RecoveryStatePrioritizer,
    StatePriorityBatch,
    validate_priority_batch,
)
from PSLAP.relocation_family_certification import (
    RELOCATION_FAMILY_CERTIFICATION,
    ValidatedRelocationFamilyAnchor,
    try_relocation_family,
)


VIABILITY_CANDIDATE_INTERFACE = (
    "strict_parameterized_viability_constrained_action_frontier_v1"
)
STRICT_DECISION_BOUNDARY_CONTRACT = (
    "empty_hand_no_half_committed_assignment_episode_epoch_boundary_v1"
)
FAIL_CLOSED_CERTIFICATION_CONTRACT = (
    "safe_only_unknown_and_unsafe_excluded_v1"
)
EXACT_VERIFIER_AUTHORITY = "exact_recovery_transition_search_v1"
BOUNDARY_FIXED_OBSTACLE_CONTRACT = (
    ROBUST_QUEUE_OBSTACLE_CONTRACT + "_except_agent_start_cell_v1"
)
BOUNDARY_AGENT_START_EXCEPTION = "_except_agent_start_cell_v1"
CANONICAL_CERTIFICATION_ORDER = "canonical_exact_full_v1"
EXACT_ONLY_RECOVERY_CERTIFICATION = "native_exact_search_only_v1"
RECOVERY_CERTIFICATION_STRATEGIES = (
    EXACT_ONLY_RECOVERY_CERTIFICATION,
    RELOCATION_FAMILY_CERTIFICATION,
)


class ViabilityMode(str, Enum):
    ACCEPT = "accept"
    RECOVER = "recover"
    DEFER = "defer"


class ViabilityActionType(str, Enum):
    ACCEPT = "accept"
    DELIVER = "deliver"
    RECONFIGURE = "reconfigure"
    DEFER = "defer"


MODE_FOR_ACTION_TYPE = {
    ViabilityActionType.ACCEPT: ViabilityMode.ACCEPT,
    ViabilityActionType.DELIVER: ViabilityMode.RECOVER,
    ViabilityActionType.RECONFIGURE: ViabilityMode.RECOVER,
    ViabilityActionType.DEFER: ViabilityMode.DEFER,
}


class StrictDecisionBoundaryError(ValueError):
    """The live environment is not at the action model's atomic boundary."""


class ViabilityCertificateCache(
    MutableMapping[CertificateCacheKey, RecoverabilityCertificate]
):
    """Reusable exact-certificate cache shared by all candidate classes.

    Cache entries are keyed by the complete immutable recovery state and every
    search budget.  The cache stores certificates, never action preferences.
    """

    def __init__(self):
        self._entries: dict[
            CertificateCacheKey, RecoverabilityCertificate
        ] = {}

    def __getitem__(self, key: CertificateCacheKey) -> RecoverabilityCertificate:
        return self._entries[key]

    def __setitem__(
        self,
        key: CertificateCacheKey,
        value: RecoverabilityCertificate,
    ) -> None:
        if not isinstance(value, RecoverabilityCertificate):
            raise TypeError("viability cache values must be certificates")
        self._entries[key] = value

    def __delitem__(self, key: CertificateCacheKey) -> None:
        del self._entries[key]

    def __iter__(self) -> Iterator[CertificateCacheKey]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class DeferLivenessDecision:
    allowed: bool
    horizon_steps: Optional[int]
    reason: str

    def __post_init__(self) -> None:
        if self.allowed:
            if self.horizon_steps is None or self.horizon_steps <= 0:
                raise ValueError("an allowed defer requires a positive horizon")
        elif self.horizon_steps is not None:
            raise ValueError("a rejected defer cannot have a horizon")


@dataclass(frozen=True)
class BoundedEventDeferRule:
    """Online-observable liveness rule for the otherwise self-looping mode.

    With stored work, waiting is bounded by the nearest positive known storage
    deadline.  Once any admitted block is due, recovery must make progress.
    With no admitted work, a bounded wait may expose an arrival event.  The
    explicit streak bound prevents an integration from proposing idle waits
    forever when no event occurs; the caller must reset the streak only after
    a non-defer macro or an observed arrival/status event.
    """

    max_option_steps: int = 10
    max_consecutive_defer_decisions: int = 16
    version: str = field(
        default="bounded_observable_event_or_positive_deadline_defer_v1",
        init=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_option_steps, bool)
            or not isinstance(self.max_option_steps, int)
            or self.max_option_steps <= 0
        ):
            raise ValueError("max_option_steps must be a positive integer")
        if (
            isinstance(self.max_consecutive_defer_decisions, bool)
            or not isinstance(self.max_consecutive_defer_decisions, int)
            or self.max_consecutive_defer_decisions <= 0
        ):
            raise ValueError(
                "max_consecutive_defer_decisions must be a positive integer"
            )

    def evaluate(
        self,
        env,
        yard: YardSnapshot,
        *,
        inbound_present: bool,
        current_certificate: RecoverabilityCertificate,
        consecutive_defer_decisions: int,
        terminal: bool,
    ) -> DeferLivenessDecision:
        """Apply the stated rule without reading an unarrived job's schedule."""

        if terminal:
            return DeferLivenessDecision(False, None, "terminal_state")
        if inbound_present:
            return DeferLivenessDecision(
                False, None, "inbound_requires_accept_or_capacity_release"
            )
        if current_certificate.status is not ViabilityStatus.SAFE:
            return DeferLivenessDecision(
                False, None, "current_recoverability_not_certified_safe"
            )
        if (
            consecutive_defer_decisions
            >= self.max_consecutive_defer_decisions
        ):
            return DeferLivenessDecision(
                False, None, "consecutive_defer_bound_reached"
            )

        if yard.blocks:
            nearest_deadline = min(
                float(block.remaining_time) for block in yard.blocks
            )
            if nearest_deadline <= 0.0:
                return DeferLivenessDecision(
                    False, None, "admitted_work_due_requires_recovery_progress"
                )
            horizon = min(
                self.max_option_steps,
                max(1, int(ceil(nearest_deadline))),
            )
            return DeferLivenessDecision(
                True, horizon, "bounded_wait_toward_known_positive_deadline"
            )

        # Only currently observable status is used.  We deliberately do not
        # inspect arrival_step or storage_steps_needed for an unarrived block.
        unresolved_external_work = any(
            not block.delivered and not block.stored and not block.carrying
            for block in env.blocks
        )
        if not unresolved_external_work:
            return DeferLivenessDecision(False, None, "no_unfinished_work")
        return DeferLivenessDecision(
            True,
            self.max_option_steps,
            "bounded_wait_for_observable_arrival_event",
        )


@dataclass(frozen=True)
class ViabilityActionCandidate:
    """One SAFE successor and its exact, current-epoch option binding."""

    key: str
    mode: ViabilityMode
    action_type: ViabilityActionType
    target_label: Optional[str]
    source: Optional[Cell]
    destination: Optional[Cell]
    successor_state: RecoveryState
    certificate: RecoverabilityCertificate
    option: object = field(compare=False, hash=False, repr=False)
    recovery_action: Optional[RecoveryAction] = None
    horizon_steps: Optional[int] = None
    recovery_rank_before: Optional[int] = None
    recovery_rank_after: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("candidate key cannot be empty")
        if self.mode is not MODE_FOR_ACTION_TYPE[self.action_type]:
            raise ValueError("candidate mode/action type mismatch")
        if self.certificate.status is not ViabilityStatus.SAFE:
            raise ValueError("only SAFE candidates may enter the frontier")
        if self.action_type is ViabilityActionType.DEFER:
            if self.horizon_steps is None or self.horizon_steps <= 0:
                raise ValueError("defer candidate requires a positive horizon")
            if self.target_label is not None or self.destination is not None:
                raise ValueError("defer candidate cannot bind inventory")
        elif self.horizon_steps is not None:
            raise ValueError("only defer candidates have a horizon")
        if self.action_type in (
            ViabilityActionType.DELIVER,
            ViabilityActionType.RECONFIGURE,
        ):
            if self.recovery_action is None:
                raise ValueError("recovery candidate requires a recovery action")
        elif self.recovery_action is not None:
            raise ValueError("non-recovery candidate cannot carry a recovery action")

    @property
    def rank_delta(self) -> Optional[int]:
        """Positive means the selected macro reduces exact recovery rank."""

        if (
            self.recovery_rank_before is None
            or self.recovery_rank_after is None
        ):
            return None
        return self.recovery_rank_before - self.recovery_rank_after


@dataclass(frozen=True)
class ViabilityCandidateAudit:
    interface: str
    boundary_contract: str
    certification_contract: str
    verifier_authority: str
    search_order: str
    post_accept_contract: str
    fixed_obstacle_contract: str
    liveness_rule: str
    baseline_viability_teacher: bool
    terminal: bool
    current_recovery_status: ViabilityStatus
    current_recovery_rank: Optional[int]
    recovery_rank_exact: bool
    physical_accept_count: int
    executable_accept_count: int
    accept_executor_rejections: int
    safe_accept_count: int
    unsafe_accept_count: int
    unknown_accept_count: int
    legal_recovery_count: int
    recovery_executor_rejections: int
    safe_recovery_count: int
    unsafe_recovery_count: int
    unknown_recovery_count: int
    defer_allowed: bool
    defer_horizon_steps: Optional[int]
    defer_reason: str
    consecutive_defer_decisions: int
    candidate_count: int
    accept_candidate_count: int
    deliver_candidate_count: int
    reconfigure_candidate_count: int
    defer_candidate_count: int
    rank_reducing_recovery_count: int
    cache_hits: int
    cache_misses: int
    cache_entries: int
    analysis_seconds: float
    exact_analysis_seconds: float = 0.0
    certification_order: str = CANONICAL_CERTIFICATION_ORDER
    priority_states_scored: int = 0
    priority_pass_count: int = 0
    priority_inference_seconds: float = 0.0
    priority_checkpoint_sha256: Optional[str] = None
    complete_frontier_exactly_verified: bool = True
    recovery_certification_strategy: str = EXACT_ONLY_RECOVERY_CERTIFICATION
    relocation_family_anchor_available: bool = False
    relocation_family_attempt_count: int = 0
    relocation_family_proof_count: int = 0
    relocation_family_miss_count: int = 0
    relocation_family_setup_seconds: float = 0.0
    relocation_family_connection_seconds: float = 0.0
    native_recovery_search_count: int = 0

    @property
    def fail_closed_rejection_count(self) -> int:
        return self.unsafe_accept_count + self.unknown_accept_count + (
            self.unsafe_recovery_count + self.unknown_recovery_count
        )

    def audit_dict(self) -> dict:
        return {
            "interface": self.interface,
            "boundary_contract": self.boundary_contract,
            "certification_contract": self.certification_contract,
            "verifier_authority": self.verifier_authority,
            "search_order": self.search_order,
            "post_accept_contract": self.post_accept_contract,
            "fixed_obstacle_contract": self.fixed_obstacle_contract,
            "liveness_rule": self.liveness_rule,
            "baseline_viability_teacher": self.baseline_viability_teacher,
            "terminal": self.terminal,
            "current_recovery_status": self.current_recovery_status.value,
            "current_recovery_rank": self.current_recovery_rank,
            "recovery_rank_exact": self.recovery_rank_exact,
            "physical_accept_count": self.physical_accept_count,
            "executable_accept_count": self.executable_accept_count,
            "accept_executor_rejections": self.accept_executor_rejections,
            "safe_accept_count": self.safe_accept_count,
            "unsafe_accept_count": self.unsafe_accept_count,
            "unknown_accept_count": self.unknown_accept_count,
            "legal_recovery_count": self.legal_recovery_count,
            "recovery_executor_rejections": self.recovery_executor_rejections,
            "safe_recovery_count": self.safe_recovery_count,
            "unsafe_recovery_count": self.unsafe_recovery_count,
            "unknown_recovery_count": self.unknown_recovery_count,
            "defer_allowed": self.defer_allowed,
            "defer_horizon_steps": self.defer_horizon_steps,
            "defer_reason": self.defer_reason,
            "consecutive_defer_decisions": self.consecutive_defer_decisions,
            "candidate_count": self.candidate_count,
            "accept_candidate_count": self.accept_candidate_count,
            "deliver_candidate_count": self.deliver_candidate_count,
            "reconfigure_candidate_count": self.reconfigure_candidate_count,
            "defer_candidate_count": self.defer_candidate_count,
            "rank_reducing_recovery_count": self.rank_reducing_recovery_count,
            "fail_closed_rejection_count": self.fail_closed_rejection_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_entries": self.cache_entries,
            "analysis_seconds": float(self.analysis_seconds),
            "exact_analysis_seconds": float(self.exact_analysis_seconds),
            "certification_order": self.certification_order,
            "priority_states_scored": self.priority_states_scored,
            "priority_pass_count": self.priority_pass_count,
            "priority_inference_seconds": float(
                self.priority_inference_seconds
            ),
            "priority_checkpoint_sha256": self.priority_checkpoint_sha256,
            "complete_frontier_exactly_verified": (
                self.complete_frontier_exactly_verified
            ),
            "recovery_certification_strategy": (
                self.recovery_certification_strategy
            ),
            "relocation_family_anchor_available": (
                self.relocation_family_anchor_available
            ),
            "relocation_family_attempt_count": (
                self.relocation_family_attempt_count
            ),
            "relocation_family_proof_count": (
                self.relocation_family_proof_count
            ),
            "relocation_family_miss_count": (
                self.relocation_family_miss_count
            ),
            "relocation_family_setup_seconds": float(
                self.relocation_family_setup_seconds
            ),
            "relocation_family_connection_seconds": float(
                self.relocation_family_connection_seconds
            ),
            "native_recovery_search_count": self.native_recovery_search_count,
        }


@dataclass(frozen=True)
class ViabilityCandidateSnapshot:
    """Immutable, one-epoch candidate frontier and certification evidence."""

    episode_instance_id: Optional[str]
    decision_epoch: int
    agent_position: Cell
    yard: YardSnapshot
    recovery_state: RecoveryState
    current_certificate: RecoverabilityCertificate
    inbound_label: Optional[str]
    candidates: tuple[ViabilityActionCandidate, ...]
    audit: ViabilityCandidateAudit

    def __post_init__(self) -> None:
        keys = tuple(candidate.key for candidate in self.candidates)
        if len(keys) != len(set(keys)):
            raise ValueError("candidate identities must be unique")
        if len(self.candidates) != self.audit.candidate_count:
            raise ValueError("candidate count disagrees with audit")
        if tuple(self.agent_position) != self.recovery_state.agent_position:
            raise ValueError("snapshot and recovery agent positions disagree")

    def candidates_for_mode(
        self, mode: ViabilityMode | str
    ) -> tuple[ViabilityActionCandidate, ...]:
        mode = ViabilityMode(mode)
        return tuple(item for item in self.candidates if item.mode is mode)

    def candidates_of_type(
        self, action_type: ViabilityActionType | str
    ) -> tuple[ViabilityActionCandidate, ...]:
        action_type = ViabilityActionType(action_type)
        return tuple(
            item for item in self.candidates if item.action_type is action_type
        )


def _certificate_key(
    state: RecoveryState, config: ViabilitySearchConfig
) -> CertificateCacheKey:
    return (
        state,
        config.max_depth,
        config.max_nodes,
        config.max_primitive_steps,
        config.search_order,
    )


def _analyze_cached(
    state: RecoveryState,
    config: ViabilitySearchConfig,
    cache: MutableMapping[CertificateCacheKey, RecoverabilityCertificate],
) -> tuple[RecoverabilityCertificate, bool, float]:
    key = _certificate_key(state, config)
    cached = cache.get(key)
    if cached is not None:
        return cached, True, 0.0
    started = perf_counter()
    certificate = analyze_recoverability(
        state,
        max_depth=config.max_depth,
        max_nodes=config.max_nodes,
        max_primitive_steps=config.max_primitive_steps,
        search_order=config.search_order,
    )
    elapsed = perf_counter() - started
    cache[key] = certificate
    return certificate, False, elapsed


def _certificate_rank(
    certificate: RecoverabilityCertificate,
) -> Optional[int]:
    return certificate.exact_recovery_rank


def _recovery_candidate_key(action: RecoveryAction) -> str:
    prefix = (
        "deliver"
        if action.kind is RecoveryActionKind.DELIVERY
        else "reconfigure"
    )
    return (
        f"{prefix}:{action.block_label}:"
        f"{action.destination[0]}:{action.destination[1]}"
    )


def _strict_inbound(env):
    inbound = [
        block
        for block in env.blocks
        if (
            block.position == env.pickup_cell
            and not block.stored
            and not block.delivered
            and not block.carrying
        )
    ]
    if len(inbound) > 1:
        raise StrictDecisionBoundaryError(
            "strict boundary has multiple physical inbound blocks"
        )
    return inbound[0] if inbound else None


def _validate_strict_boundary(env) -> None:
    if getattr(env, "current_state", None) is None:
        raise StrictDecisionBoundaryError("environment has not been reset")
    if any(block.carrying for block in env.blocks):
        raise StrictDecisionBoundaryError(
            "strict decision boundary cannot carry inventory"
        )
    labels = [str(block.label) for block in env.blocks]
    if len(labels) != len(set(labels)):
        raise StrictDecisionBoundaryError("block labels must be unique")
    for block in env.blocks:
        if (
            not block.stored
            and not block.delivered
            and block.storage_location is not None
        ):
            raise StrictDecisionBoundaryError(
                "strict boundary contains a half-committed inbound assignment"
            )
        if block.stored and not block.delivered:
            if (
                block.position is None
                or block.storage_location is None
                or tuple(block.position) != tuple(block.storage_location)
            ):
                raise StrictDecisionBoundaryError(
                    "stored inventory violates the boundary location invariant"
                )
    _strict_inbound(env)


def _inbound_view(env, block) -> BlockView:
    remaining = None
    signed = getattr(env, "signed_remaining_storage_time", None)
    if callable(signed):
        remaining = signed(block)
    if remaining is None:
        remaining = max(
            0.0,
            float(block.storage_steps_needed)
            - float(block.storage_steps_elapsed),
        )
    return BlockView(
        str(block.label), tuple(block.position), float(remaining)
    )


def enumerate_viability_candidates(
    env,
    *,
    consecutive_defer_decisions: int,
    search_config: ViabilitySearchConfig = ViabilitySearchConfig(),
    liveness_rule: BoundedEventDeferRule = BoundedEventDeferRule(),
    reserved_cells: Iterable[Cell] = (),
    cache: Optional[
        MutableMapping[CertificateCacheKey, RecoverabilityCertificate]
    ] = None,
    max_replans: int = 8,
    state_prioritizer: Optional[RecoveryStatePrioritizer] = None,
    recovery_certification_strategy: str = EXACT_ONLY_RECOVERY_CERTIFICATION,
) -> ViabilityCandidateSnapshot:
    """Enumerate, certify, and bind every viable parameterized macro.

    ``consecutive_defer_decisions`` is intentionally explicit.  The future
    controller must carry this observable liveness state across snapshots and
    reset it only on progress/an observed event; silently defaulting it to zero
    would make a finite per-option wait compatible with infinite deferral.

    ``state_prioritizer`` may only permute uncached exact checks.  The complete
    physical frontier is still certified and reconstructed in canonical order.

    ``recovery_certification_strategy`` may additionally serve relocation
    successors with an exact constructive witness derived from the native
    current-state certificate.  Such proofs are never inserted into the
    outcome cache; every unresolved case retains native exact fallback.
    """

    if (
        isinstance(consecutive_defer_decisions, bool)
        or not isinstance(consecutive_defer_decisions, int)
        or consecutive_defer_decisions < 0
    ):
        raise ValueError(
            "consecutive_defer_decisions must be a non-negative integer"
        )
    if (
        isinstance(max_replans, bool)
        or not isinstance(max_replans, int)
        or max_replans <= 0
    ):
        raise ValueError("max_replans must be a positive integer")
    if not isinstance(search_config, ViabilitySearchConfig):
        raise TypeError("search_config must be a ViabilitySearchConfig")
    if not isinstance(liveness_rule, BoundedEventDeferRule):
        raise TypeError("liveness_rule must be a BoundedEventDeferRule")
    if recovery_certification_strategy not in RECOVERY_CERTIFICATION_STRATEGIES:
        raise ValueError(
            "recovery_certification_strategy must be one of "
            f"{RECOVERY_CERTIFICATION_STRATEGIES!r}"
        )

    started = perf_counter()
    _validate_strict_boundary(env)
    reservations = tuple(tuple(cell) for cell in reserved_cells)
    if len(reservations) != len(set(reservations)):
        raise ValueError("reserved_cells must be unique")
    certificate_cache = cache if cache is not None else ViabilityCertificateCache()
    cache_hits = 0
    cache_misses = 0
    exact_analysis_seconds = 0.0
    priority_batches: list[StatePriorityBatch] = []

    terminal = bool(env.is_state_terminal(env.current_state))
    yard = YardSnapshot.from_env(env)
    agent_position = tuple(env.current_state)
    fixed_obstacles = online_fixed_obstacles(
        env,
        reserve_queue_cells=search_config.reserve_queue_cells,
    )
    # A strict macro may begin on a live/structurally reserved cell and move
    # away from it.  Keeping that start in fixed_obstacles would reject the
    # RecoveryState before the first transition rather than constrain motion.
    fixed_obstacles = fixed_obstacles - {agent_position}
    recovery_state = RecoveryState.from_yard_snapshot(
        yard,
        agent_position,
        fixed_obstacles=fixed_obstacles,
        reserved_cells=reservations,
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    )
    current_certificate, current_hit, current_exact_seconds = _analyze_cached(
        recovery_state, search_config, certificate_cache
    )
    cache_hits += int(current_hit)
    cache_misses += int(not current_hit)
    exact_analysis_seconds += current_exact_seconds
    current_rank = _certificate_rank(current_certificate)
    relocation_family_anchor = None
    relocation_family_setup_seconds = 0.0
    relocation_family_connection_seconds = 0.0
    relocation_family_attempt_count = 0
    relocation_family_proof_count = 0
    relocation_family_miss_count = 0
    native_recovery_search_count = 0
    if (
        recovery_certification_strategy == RELOCATION_FAMILY_CERTIFICATION
        and current_certificate.status is ViabilityStatus.SAFE
        and current_certificate.witness
        and search_config.search_order == "goal_directed"
    ):
        family_started = perf_counter()
        relocation_family_anchor = ValidatedRelocationFamilyAnchor(
            recovery_state, current_certificate
        )
        relocation_family_setup_seconds = perf_counter() - family_started

    candidates: list[ViabilityActionCandidate] = []
    inbound = _strict_inbound(env)

    # Accept(block, cell): physical mask -> exact option executability -> exact
    # post-accept recovery certificate.  No proposal/ranking is requested.
    physical_accept_count = 0
    executable_accept_count = 0
    accept_executor_rejections = 0
    safe_accept_count = 0
    unsafe_accept_count = 0
    unknown_accept_count = 0
    accept_report: Optional[CandidateViabilityReport] = None
    if inbound is not None and not terminal:
        inbound_view = _inbound_view(env, inbound)
        physical_cells = shared_candidate_mask(
            yard,
            inbound_view,
            tuple(inbound.position),
            reserved_cells=reservations,
        )
        physical_accept_count = len(physical_cells)
        bound_accept_options = {}
        for cell in physical_cells:
            option = ExplicitAcceptOption(
                env,
                inbound.label,
                cell,
                max_replans=max_replans,
            )
            if not option.initiation(env.get_current_state()):
                accept_executor_rejections += 1
                continue
            bound_accept_options[cell] = option
        executable_cells = tuple(bound_accept_options)
        executable_accept_count = len(executable_cells)
        accept_report = certify_post_accept_candidates(
            env,
            yard,
            inbound_view,
            executable_cells,
            config=search_config,
            reserved_cells=reservations,
            cache=certificate_cache,
            state_prioritizer=state_prioritizer,
        )
        accept_audit = accept_report.audit_dict()
        cache_hits += int(accept_audit["cache_hits"])
        cache_misses += int(accept_audit["cache_misses"])
        exact_analysis_seconds += float(
            accept_audit["exact_analysis_seconds"]
        )
        if accept_report.priority_batch is not None:
            priority_batches.append(accept_report.priority_batch)
        safe_accept_count = len(accept_report.safe_cells)
        unsafe_accept_count = len(accept_report.unsafe_cells)
        unknown_accept_count = len(accept_report.unknown_cells)
        for item in accept_report.candidates:
            if item.status is not ViabilityStatus.SAFE:
                continue
            cell = item.cell
            candidates.append(
                ViabilityActionCandidate(
                    key=(
                        f"accept:{inbound.label}:{cell[0]}:{cell[1]}"
                    ),
                    mode=ViabilityMode.ACCEPT,
                    action_type=ViabilityActionType.ACCEPT,
                    target_label=str(inbound.label),
                    source=tuple(inbound.position),
                    destination=cell,
                    successor_state=item.recovery_state,
                    certificate=item.certificate,
                    option=bound_accept_options[cell],
                    recovery_rank_before=current_rank,
                    recovery_rank_after=_certificate_rank(item.certificate),
                )
            )

    # Deliver/Reconfigure: enumerate the transition model itself, then certify
    # each deterministic successor before binding the corresponding option.
    recovery_status_counts = Counter()
    recovery_executor_rejections = 0
    legal_actions = () if terminal else legal_recovery_actions(recovery_state)
    recovery_specs = tuple(
        (
            action,
            apply_recovery_action(recovery_state, action),
            _recovery_candidate_key(action),
        )
        for action in legal_actions
    )
    recovery_results: list[
        Optional[tuple[RecoverabilityCertificate, bool]]
    ] = [None] * len(recovery_specs)
    missing_recovery_indices = []
    for index, (_, successor, _) in enumerate(recovery_specs):
        cached = certificate_cache.get(
            _certificate_key(successor, search_config)
        )
        if cached is None:
            missing_recovery_indices.append(index)
        else:
            recovery_results[index] = (cached, True)
            cache_hits += 1

    ordered_recovery_misses = tuple(missing_recovery_indices)
    if state_prioritizer is not None and missing_recovery_indices:
        priority_request = tuple(
            (recovery_specs[index][2], recovery_specs[index][1])
            for index in missing_recovery_indices
        )
        priority_batch = validate_priority_batch(
            state_prioritizer.prioritize(priority_request),
            priority_request,
            prioritizer=state_prioritizer,
        )
        priority_batches.append(priority_batch)
        ordered_recovery_misses = tuple(
            missing_recovery_indices[index]
            for index in priority_batch.ordered_indices
        )

    for index in ordered_recovery_misses:
        action, successor, _ = recovery_specs[index]
        if (
            relocation_family_anchor is not None
            and action.kind is RecoveryActionKind.RELOCATION
        ):
            family_started = perf_counter()
            attempt = try_relocation_family(
                relocation_family_anchor,
                action,
                successor,
                search_config,
            )
            relocation_family_connection_seconds += (
                perf_counter() - family_started
            )
            relocation_family_attempt_count += 1
            if attempt.certificate is not None:
                relocation_family_proof_count += 1
                cache_misses += 1
                recovery_results[index] = (attempt.certificate, False)
                continue
            relocation_family_miss_count += 1
        certificate, cache_hit, elapsed = _analyze_cached(
            successor, search_config, certificate_cache
        )
        exact_analysis_seconds += elapsed
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        native_recovery_search_count += int(not cache_hit)
        recovery_results[index] = (certificate, cache_hit)

    if any(result is None for result in recovery_results):  # pragma: no cover
        raise RuntimeError("recovery certification left an unresolved state")

    # Candidate construction remains in canonical transition-model order.  A
    # critic may change only the order in which cache misses reach the exact
    # verifier, never the frontier supplied to the controller.
    for (action, successor, key), result in zip(
        recovery_specs, recovery_results
    ):
        certificate, _ = result  # type: ignore[misc]
        recovery_status_counts[certificate.status] += 1
        if certificate.status is not ViabilityStatus.SAFE:
            continue

        if action.kind is RecoveryActionKind.DELIVERY:
            action_type = ViabilityActionType.DELIVER
            option = DirectDeliverOption.from_recovery_action(
                env, action, max_replans=max_replans
            )
        elif action.kind is RecoveryActionKind.RELOCATION:
            action_type = ViabilityActionType.RECONFIGURE
            option = ReconfigureOption(
                env,
                action.block_label,
                action.destination,
                max_replans=max_replans,
            )
        else:  # pragma: no cover - enum is closed
            raise RuntimeError(f"unknown recovery action kind: {action.kind}")
        if not option.initiation(env.get_current_state()):
            # Transition/executor disagreement is never promoted to a viable
            # action.  The explicit count keeps this fail-closed mismatch
            # distinguishable from verifier UNSAFE/UNKNOWN results.
            recovery_executor_rejections += 1
            continue
        candidates.append(
            ViabilityActionCandidate(
                key=key,
                mode=ViabilityMode.RECOVER,
                action_type=action_type,
                target_label=action.block_label,
                source=action.source,
                destination=action.destination,
                successor_state=successor,
                certificate=certificate,
                option=option,
                recovery_action=action,
                recovery_rank_before=current_rank,
                recovery_rank_after=_certificate_rank(certificate),
            )
        )

    # Defer is a bounded liveness action, not an automatically safe self-loop.
    defer_decision = liveness_rule.evaluate(
        env,
        yard,
        inbound_present=inbound is not None,
        current_certificate=current_certificate,
        consecutive_defer_decisions=consecutive_defer_decisions,
        terminal=terminal,
    )
    if defer_decision.allowed:
        defer_option = ExplicitDeferOption(
            env, horizon_steps=defer_decision.horizon_steps
        )
        if defer_option.initiation(env.get_current_state()):
            candidates.append(
                ViabilityActionCandidate(
                    key=f"defer:{defer_decision.horizon_steps}",
                    mode=ViabilityMode.DEFER,
                    action_type=ViabilityActionType.DEFER,
                    target_label=None,
                    source=agent_position,
                    destination=None,
                    successor_state=recovery_state,
                    certificate=current_certificate,
                    option=defer_option,
                    horizon_steps=defer_decision.horizon_steps,
                    recovery_rank_before=current_rank,
                    recovery_rank_after=current_rank,
                )
            )
        else:  # defensive: rule and executable option must agree
            defer_decision = DeferLivenessDecision(
                False, None, "defer_option_not_executable"
            )

    type_counts = Counter(item.action_type for item in candidates)
    rank_reducing = sum(
        item.mode is ViabilityMode.RECOVER
        and item.rank_delta is not None
        and item.rank_delta > 0
        for item in candidates
    )
    priority_checkpoint_shas = {
        batch.checkpoint_sha256 for batch in priority_batches
    }
    if len(priority_checkpoint_shas) > 1:
        raise RuntimeError("one frontier used multiple critic checkpoints")
    priority_checkpoint_sha256 = (
        next(iter(priority_checkpoint_shas))
        if priority_checkpoint_shas
        else (
            None
            if state_prioritizer is None
            else str(state_prioritizer.checkpoint_sha256)
        )
    )
    audit = ViabilityCandidateAudit(
        interface=VIABILITY_CANDIDATE_INTERFACE,
        boundary_contract=STRICT_DECISION_BOUNDARY_CONTRACT,
        certification_contract=FAIL_CLOSED_CERTIFICATION_CONTRACT,
        verifier_authority=EXACT_VERIFIER_AUTHORITY,
        search_order=search_config.search_order,
        post_accept_contract=POST_ACCEPT_CERTIFICATION_CONTRACT,
        fixed_obstacle_contract=(
            search_config.fixed_obstacle_contract
            + BOUNDARY_AGENT_START_EXCEPTION
        ),
        liveness_rule=liveness_rule.version,
        baseline_viability_teacher=False,
        terminal=terminal,
        current_recovery_status=current_certificate.status,
        current_recovery_rank=current_rank,
        recovery_rank_exact=current_certificate.recovery_rank_is_exact,
        physical_accept_count=physical_accept_count,
        executable_accept_count=executable_accept_count,
        accept_executor_rejections=accept_executor_rejections,
        safe_accept_count=safe_accept_count,
        unsafe_accept_count=unsafe_accept_count,
        unknown_accept_count=unknown_accept_count,
        legal_recovery_count=len(legal_actions),
        recovery_executor_rejections=recovery_executor_rejections,
        safe_recovery_count=recovery_status_counts[ViabilityStatus.SAFE],
        unsafe_recovery_count=recovery_status_counts[ViabilityStatus.UNSAFE],
        unknown_recovery_count=recovery_status_counts[ViabilityStatus.UNKNOWN],
        defer_allowed=defer_decision.allowed,
        defer_horizon_steps=defer_decision.horizon_steps,
        defer_reason=defer_decision.reason,
        consecutive_defer_decisions=consecutive_defer_decisions,
        candidate_count=len(candidates),
        accept_candidate_count=type_counts[ViabilityActionType.ACCEPT],
        deliver_candidate_count=type_counts[ViabilityActionType.DELIVER],
        reconfigure_candidate_count=type_counts[
            ViabilityActionType.RECONFIGURE
        ],
        defer_candidate_count=type_counts[ViabilityActionType.DEFER],
        rank_reducing_recovery_count=int(rank_reducing),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_entries=len(certificate_cache),
        analysis_seconds=perf_counter() - started,
        exact_analysis_seconds=exact_analysis_seconds,
        certification_order=(
            CANONICAL_CERTIFICATION_ORDER
            if state_prioritizer is None
            else str(state_prioritizer.protocol)
        ),
        priority_states_scored=sum(
            len(batch.entries) for batch in priority_batches
        ),
        priority_pass_count=sum(
            batch.priority_pass_count for batch in priority_batches
        ),
        priority_inference_seconds=sum(
            float(batch.inference_seconds) for batch in priority_batches
        ),
        priority_checkpoint_sha256=priority_checkpoint_sha256,
        complete_frontier_exactly_verified=True,
        recovery_certification_strategy=recovery_certification_strategy,
        relocation_family_anchor_available=(
            relocation_family_anchor is not None
        ),
        relocation_family_attempt_count=relocation_family_attempt_count,
        relocation_family_proof_count=relocation_family_proof_count,
        relocation_family_miss_count=relocation_family_miss_count,
        relocation_family_setup_seconds=relocation_family_setup_seconds,
        relocation_family_connection_seconds=(
            relocation_family_connection_seconds
        ),
        native_recovery_search_count=native_recovery_search_count,
    )
    instance = getattr(env, "current_episode_instance", None)
    return ViabilityCandidateSnapshot(
        episode_instance_id=(
            None if instance is None else str(instance.instance_id)
        ),
        decision_epoch=int(env.time_steps),
        agent_position=agent_position,
        yard=yard,
        recovery_state=recovery_state,
        current_certificate=current_certificate,
        inbound_label=None if inbound is None else str(inbound.label),
        candidates=tuple(candidates),
        audit=audit,
    )


__all__ = [
    "BOUNDARY_FIXED_OBSTACLE_CONTRACT",
    "BOUNDARY_AGENT_START_EXCEPTION",
    "CANONICAL_CERTIFICATION_ORDER",
    "EXACT_VERIFIER_AUTHORITY",
    "FAIL_CLOSED_CERTIFICATION_CONTRACT",
    "EXACT_ONLY_RECOVERY_CERTIFICATION",
    "RECOVERY_CERTIFICATION_STRATEGIES",
    "RELOCATION_FAMILY_CERTIFICATION",
    "STRICT_DECISION_BOUNDARY_CONTRACT",
    "VIABILITY_CANDIDATE_INTERFACE",
    "BoundedEventDeferRule",
    "DeferLivenessDecision",
    "StrictDecisionBoundaryError",
    "ViabilityActionCandidate",
    "ViabilityActionType",
    "ViabilityCandidateAudit",
    "ViabilityCandidateSnapshot",
    "ViabilityCertificateCache",
    "ViabilityMode",
    "enumerate_viability_candidates",
]
