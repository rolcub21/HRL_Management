"""Add bounded waiting only at a physically blocked inbound pickup."""

from dataclasses import dataclass, field, replace

from PSLAP.neutral_protocol import shared_candidate_mask
from PSLAP.viability_candidates import BoundedEventDeferRule, DeferLivenessDecision, _strict_inbound, _inbound_view


@dataclass(frozen=True)
class BlockedPickupWaitRule(BoundedEventDeferRule):
    max_inbound_release_age: int = 80
    version: str = field(default="e121_blocked_pickup_nominal_due_release_age_bounded_wait_v1", init=False)

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.max_inbound_release_age, bool)
                or not isinstance(self.max_inbound_release_age, int) or self.max_inbound_release_age <= 0):
            raise ValueError("max_inbound_release_age must be a positive integer")

    def evaluate(self, env, yard, *, inbound_present, **kwargs):
        if not inbound_present or kwargs["terminal"]:
            return super().evaluate(env, yard, inbound_present=inbound_present, **kwargs)
        # All remaining parent exclusions still apply: SAFE current recovery,
        # nominal due cutoff, streak bound and known unfinished work.
        parent = super().evaluate(env, yard, inbound_present=False, **kwargs)
        if not parent.allowed:
            return parent
        inbound = _strict_inbound(env)
        if inbound is None:
            return DeferLivenessDecision(False, None, "e121_inbound_identity_unavailable")
        remaining = self.max_inbound_release_age - (int(env.time_steps) - int(inbound.arrival_step))
        if remaining <= 0:
            return DeferLivenessDecision(False, None, "e121_inbound_absolute_release_age_bound_reached")
        # Restrict the intervention to E120's physical routing/capacity trap.
        # Do not suppress an already available Accept or add uncertified actions.
        if shared_candidate_mask(yard, _inbound_view(env, inbound), tuple(inbound.position)):
            return DeferLivenessDecision(False, None, "inbound_requires_accept_or_capacity_release")
        return replace(parent, horizon_steps=min(parent.horizon_steps, remaining),
                       reason="e121_bounded_inbound_wait_at_physically_blocked_pickup")
