"""Certified Accept-first frontier for fixed-order, immediate admission."""

from dataclasses import replace

from PSLAP.viability_candidates import ViabilityActionType
from experiments.conditioned_vcg.E159_lazy_joint_training_218k_v1 import run as e159
from experiments.conditioned_vcg.E162_certified_guard_mask_221k_v1 import frontier as e162


def _safe_accept_exists(agent, physical):
    indices = tuple(
        index
        for index, proposal in enumerate(physical.proposals)
        if proposal.action_type is ViabilityActionType.ACCEPT
    )
    if not indices:
        return False, indices
    for index in indices:
        candidate, hit, elapsed = e159._candidate(agent, physical, index)
        agent.e164_accept_certificate_queries += 1
        agent.e164_accept_certificate_hits += int(hit)
        agent.e164_accept_certificate_seconds += float(elapsed)
        if candidate is not None:
            return True, indices
    return False, indices


def build(agent, env, **kwargs):
    physical = e162.build(agent, env, **kwargs)
    if physical.forced_witness:
        return physical
    safe, indices = _safe_accept_exists(agent, physical)
    if not safe:
        return physical
    agent.e164_accept_priority_epochs += 1
    return replace(
        physical,
        proposals=tuple(physical.proposals[index] for index in indices),
        records=tuple(physical.records[index] for index in indices),
        reconfigure_masked=False,
    )


def counters(agent):
    return {
        "accept_priority_epochs": int(getattr(agent, "e164_accept_priority_epochs", 0)),
        "accept_certificate_queries": int(getattr(agent, "e164_accept_certificate_queries", 0)),
        "accept_certificate_hits": int(getattr(agent, "e164_accept_certificate_hits", 0)),
        "accept_certificate_seconds": float(getattr(agent, "e164_accept_certificate_seconds", 0.0)),
    }
