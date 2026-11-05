"""Expose E195's executable physical frontier without certificate filtering.

Exact certificates are still computed so this matched ablation can report the
status of every exposed and selected candidate.  They have no eligibility
authority: the executor sees a SAFE proxy while ``shadow_certificate`` retains
the exact label.  This is the same causal design used by E03, updated for the
current E195 action interface and workload.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional

from PSLAP.viability import (
    RecoverabilityCertificate,
    RecoveryAction,
    ViabilityStatus,
)
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityCertificateCache,
    ViabilityMode,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP import viability_candidates as vc
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import (
    legal,
)


PHYSICAL_INTERFACE = "e198_executable_physical_frontier_v1"
SHADOW_CONTRACT = "exact_certificate_shadow_label_no_eligibility_authority_v1"
SHADOW_AUTHORITY = "none_for_eligibility_exact_for_diagnosis_v1"
ELIGIBILITY_PROXY_REASON = "e198_physical_eligibility_proxy"


@dataclass(frozen=True)
class PhysicalCandidate:
    """Executor-compatible candidate with a non-authoritative exact label."""

    key: str
    mode: ViabilityMode
    action_type: ViabilityActionType
    target_label: Optional[str]
    source: Optional[tuple[int, int]]
    destination: Optional[tuple[int, int]]
    successor_state: object
    certificate: RecoverabilityCertificate
    shadow_certificate: RecoverabilityCertificate
    option: object
    recovery_action: Optional[RecoveryAction] = None
    horizon_steps: Optional[int] = None
    recovery_rank_before: Optional[int] = None
    recovery_rank_after: Optional[int] = None

    @property
    def rank_delta(self) -> Optional[int]:
        if self.recovery_rank_before is None or self.recovery_rank_after is None:
            return None
        return self.recovery_rank_before - self.recovery_rank_after


def shadow_certificate(candidate) -> RecoverabilityCertificate:
    return getattr(candidate, "shadow_certificate", candidate.certificate)


def _eligibility_certificate(
    certificate: RecoverabilityCertificate,
) -> RecoverabilityCertificate:
    if certificate.status is ViabilityStatus.SAFE:
        return certificate
    return replace(
        certificate,
        status=ViabilityStatus.SAFE,
        reason=(
            f"{ELIGIBILITY_PROXY_REASON};"
            f"shadow_status={certificate.status.value};"
            f"shadow_reason={certificate.reason}"
        ),
    )


def _from_certified(candidate) -> PhysicalCandidate:
    return PhysicalCandidate(
        key=candidate.key,
        mode=candidate.mode,
        action_type=candidate.action_type,
        target_label=candidate.target_label,
        source=candidate.source,
        destination=candidate.destination,
        successor_state=candidate.successor_state,
        certificate=candidate.certificate,
        shadow_certificate=candidate.certificate,
        option=candidate.option,
        recovery_action=candidate.recovery_action,
        horizon_steps=candidate.horizon_steps,
        recovery_rank_before=candidate.recovery_rank_before,
        recovery_rank_after=candidate.recovery_rank_after,
    )


def _from_proposal(
    proposal: legal.LegalProposal,
    *,
    shadow: RecoverabilityCertificate,
    current_rank: Optional[int],
) -> PhysicalCandidate:
    after = shadow.exact_recovery_rank
    if proposal.action_type is ViabilityActionType.DEFER:
        after = current_rank
    return PhysicalCandidate(
        key=proposal.key,
        mode=proposal.mode,
        action_type=proposal.action_type,
        target_label=proposal.target_label,
        source=proposal.source,
        destination=proposal.destination,
        successor_state=proposal.successor_state,
        certificate=_eligibility_certificate(shadow),
        shadow_certificate=shadow,
        option=proposal.option,
        recovery_action=proposal.recovery_action,
        horizon_steps=proposal.horizon_steps,
        recovery_rank_before=current_rank,
        recovery_rank_after=after,
    )


def expose_physical_snapshot(
    env,
    certified: ViabilityCandidateSnapshot,
    *,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    consecutive_defer_decisions: int,
    cache: ViabilityCertificateCache,
    reserved_cells=(),
    max_replans: int = 8,
) -> ViabilityCandidateSnapshot:
    """Replace the SAFE-only frontier by all executable physical proposals."""

    if tuple(reserved_cells):
        raise RuntimeError("E198 does not alter E195's empty reservation set")
    if int(max_replans) != 8:
        raise RuntimeError("E198 requires E195's max_replans=8 contract")
    safe_by_key = {candidate.key: candidate for candidate in certified.candidates}
    proposals = legal.physical_frontier(
        env,
        current_state=certified.recovery_state,
        current_certificate=certified.current_certificate,
        search=search_config,
        liveness=liveness_rule,
        consecutive_defer=int(consecutive_defer_decisions),
    )
    current_rank = certified.audit.current_recovery_rank
    candidates = []
    for proposal in proposals:
        safe = safe_by_key.get(proposal.key)
        if safe is not None:
            candidates.append(_from_certified(safe))
            continue
        shadow = cache.get(
            vc._certificate_key(proposal.successor_state, search_config)
        )
        if shadow is None:
            raise RuntimeError(
                "E198 physical proposal lacks the exact shadow label computed "
                f"by the matched E195 enumerator: {proposal.key}"
            )
        candidates.append(
            _from_proposal(
                proposal,
                shadow=shadow,
                current_rank=current_rank,
            )
        )

    keys = tuple(candidate.key for candidate in candidates)
    if len(keys) != len(set(keys)):
        raise RuntimeError("E198 physical frontier has duplicate keys")
    safe_subsequence = tuple(
        candidate.key
        for candidate in candidates
        if candidate.shadow_certificate.status is ViabilityStatus.SAFE
    )
    certified_keys = tuple(candidate.key for candidate in certified.candidates)
    if safe_subsequence != certified_keys:
        raise RuntimeError(
            "E198 physical frontier does not preserve E195's certified subsequence"
        )

    type_counts = Counter(candidate.action_type for candidate in candidates)
    audit = replace(
        certified.audit,
        interface=PHYSICAL_INTERFACE,
        certification_contract=SHADOW_CONTRACT,
        verifier_authority=SHADOW_AUTHORITY,
        candidate_count=len(candidates),
        accept_candidate_count=type_counts[ViabilityActionType.ACCEPT],
        deliver_candidate_count=type_counts[ViabilityActionType.DELIVER],
        reconfigure_candidate_count=type_counts[ViabilityActionType.RECONFIGURE],
        defer_candidate_count=type_counts[ViabilityActionType.DEFER],
        rank_reducing_recovery_count=sum(
            candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
            for candidate in candidates
        ),
    )
    return replace(certified, candidates=tuple(candidates), audit=audit)


def status_counts(snapshot: ViabilityCandidateSnapshot) -> dict[str, int]:
    counts = Counter(
        shadow_certificate(candidate).status.value
        for candidate in snapshot.candidates
    )
    return {status.value: int(counts[status.value]) for status in ViabilityStatus}


def shadow_physical_enumerator(original):
    """Wrap the matched E195 certifier without giving labels authority."""

    def wrapped(env, **kwargs):
        certified = original(env, **kwargs)
        cache = kwargs.get("cache")
        if cache is None:
            raise RuntimeError("E198 requires E195's per-episode certificate cache")
        return expose_physical_snapshot(
            env,
            certified,
            search_config=kwargs.get("search_config", ViabilitySearchConfig()),
            liveness_rule=kwargs.get("liveness_rule", BoundedEventDeferRule()),
            consecutive_defer_decisions=kwargs["consecutive_defer_decisions"],
            cache=cache,
            reserved_cells=kwargs.get("reserved_cells", ()),
            max_replans=kwargs.get("max_replans", 8),
        )

    return wrapped


__all__ = [
    "ELIGIBILITY_PROXY_REASON",
    "PHYSICAL_INTERFACE",
    "SHADOW_AUTHORITY",
    "SHADOW_CONTRACT",
    "PhysicalCandidate",
    "expose_physical_snapshot",
    "shadow_certificate",
    "shadow_physical_enumerator",
    "status_counts",
]
