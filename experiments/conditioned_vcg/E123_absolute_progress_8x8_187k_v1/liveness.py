"""Deadline enforcement for actual admission/delivery events, not arrivals.

The 3600-step bound is conditional on a verified <=20-step next-progress
certificate, <=30-step release gaps, first release at zero, 36 finite jobs,
and deterministic execution of the certified strict macros. No observed
maximum macro duration is substituted for a certified upper bound.
"""

from dataclasses import dataclass, field

from PSLAP.viability import RecoveryActionKind, ViabilityStatus
from PSLAP.viability_candidates import BoundedEventDeferRule, DeferLivenessDecision
from viability_graph_hierarchy import RecoveryWitnessGuard, NoCertifiedViableAction
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.rewards import ReleaseETCostEnv

WAIT_BUDGET = 30
PROGRESS_BUDGET = 20
HORIZON = 3600


@dataclass
class ProgressClock:
    last_progress_time: int = 0
    admissions: int = 0
    deliveries: int = 0
    events: int = 0
    maximum_gap: int = 0

    def age(self, now):
        return int(now) - self.last_progress_time

    def observe(self, now, admissions, deliveries):
        if admissions < self.admissions or deliveries < self.deliveries:
            raise RuntimeError("E123 progress counts must be monotone")
        increment = admissions - self.admissions + deliveries - self.deliveries
        if increment:
            self.maximum_gap = max(self.maximum_gap, self.age(now))
            self.last_progress_time = int(now)
            self.events += increment
        self.admissions, self.deliveries = int(admissions), int(deliveries)


class ProgressEnv(ReleaseETCostEnv):
    enforce_absolute_bound = False

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.progress_clock = ProgressClock()
        return result

    def step(self, action):
        result = super().step(action)
        admissions = sum(block.stored_time_step is not None for block in self.blocks)
        deliveries = sum(block.delivered for block in self.blocks)
        clock = self.progress_clock
        before_gap = clock.age(self.time_steps)
        clock.observe(self.time_steps, admissions, deliveries)
        if self.enforce_absolute_bound and before_gap > WAIT_BUDGET + PROGRESS_BUDGET:
            raise RuntimeError("E123 certified absolute progress bound breached")
        return result


def available_work(env):
    return any(not block.delivered and (block.position is not None or block.stored or block.carrying)
               for block in env.blocks)


def macro_steps(candidate):
    if candidate.recovery_action is not None:
        return int(candidate.recovery_action.steps)
    if candidate.horizon_steps is not None:
        return int(candidate.horizon_steps)
    if candidate.action_type.value == "accept":
        option = candidate.option
        actions = option._strict_actions(option._block())
        return len(actions) if actions else None
    return None


def progress_prefix(certificate):
    if certificate.status is not ViabilityStatus.SAFE:
        return ()
    prefix = []
    for action in certificate.witness:
        prefix.append(action)
        if action.kind is RecoveryActionKind.DELIVERY:
            return tuple(prefix)
    return ()


def bounded_prefix(certificate):
    prefix = progress_prefix(certificate)
    return prefix if prefix and sum(action.steps for action in prefix) <= PROGRESS_BUDGET else ()


def successor_supported(candidate):
    return not candidate.successor_state.blocks or bool(bounded_prefix(candidate.certificate))


def direct_progress(candidate):
    duration = macro_steps(candidate)
    return (candidate.action_type.value in ("accept", "deliver") and duration is not None
            and duration <= PROGRESS_BUDGET and successor_supported(candidate))


@dataclass(frozen=True)
class AbsoluteProgressDeferRule(BoundedEventDeferRule):
    version: str = field(default="e123_absolute_time_since_actual_flow_progress_v1", init=False)

    def evaluate(self, env, yard, *, inbound_present, current_certificate,
                 consecutive_defer_decisions, terminal):
        if terminal:
            return DeferLivenessDecision(False, None, "terminal_state")
        if current_certificate.status is not ViabilityStatus.SAFE:
            return DeferLivenessDecision(False, None, "current_recoverability_not_certified_safe")
        remaining = WAIT_BUDGET - env.progress_clock.age(env.time_steps)
        if remaining <= 0:
            if available_work(env):
                return DeferLivenessDecision(False, None, "absolute_flow_progress_deadline")
            # An absent release cannot be manufactured by the controller.
            # This exception is audited against the declared release-gap bound.
            return DeferLivenessDecision(True, 1, "external_idle_no_released_work")
        return DeferLivenessDecision(True, min(self.max_option_steps, remaining),
                                    "bounded_wait_toward_absolute_flow_progress_deadline")


class AbsoluteProgressGuard(RecoveryWitnessGuard):
    """At expiry force a direct flow event or a certified prefix to delivery."""

    def __init__(self):
        super().__init__(max_nonprogress_recovery_decisions=2, force_when_due=False)
        self.environment = None
        self.commit_time = None

    def reset(self, *, reset_nonprogress=True):
        super().reset(reset_nonprogress=reset_nonprogress)
        self.commit_time = None

    def state_dict(self):
        return {**super().state_dict(), "absolute_wait_budget": WAIT_BUDGET,
                "progress_prefix_budget": PROGRESS_BUDGET, "commit_time": self.commit_time}

    def load_state_dict(self, state):
        if (state.get("absolute_wait_budget") != WAIT_BUDGET
                or state.get("progress_prefix_budget") != PROGRESS_BUDGET):
            raise ValueError("E123 absolute guard budget differs")
        super().load_state_dict(state)
        self.commit_time = state.get("commit_time")
        if self.active and self.commit_time is None:
            raise ValueError("E123 committed progress prefix lacks its clock")

    def forced_prepared_index(self, snapshot, prepared):
        env = self.environment
        if env is None:
            raise RuntimeError("E123 progress guard is not bound")
        overdue = env.progress_clock.age(snapshot.decision_epoch) >= WAIT_BUDGET
        if not self.active and overdue and available_work(env):
            direct = any(direct_progress(snapshot.candidates[index]) for index in prepared.source_indices)
            if not direct:
                prefix = bounded_prefix(snapshot.current_certificate)
                if not prefix:
                    raise NoCertifiedViableAction("E123 next progress within 20 steps is not certified")
                self.active_witness = prefix
                self.witness_cursor = 0
                self.activations += 1
                self.commit_time = snapshot.decision_epoch
        if not self.active:
            return None
        expected = self.active_witness[self.witness_cursor]
        matches = [i for i, index in enumerate(prepared.source_indices)
                   if snapshot.candidates[index].recovery_action == expected]
        if len(matches) != 1:
            raise NoCertifiedViableAction("E123 committed progress-prefix action disappeared")
        return matches[0]


def eligible_indices(snapshot, prepared, env, forced):
    if forced:
        return tuple(range(len(prepared.records)))
    clock = env.progress_clock
    remaining = WAIT_BUDGET - clock.age(snapshot.decision_epoch)
    allowed = []
    for i, source in enumerate(prepared.source_indices):
        candidate = snapshot.candidates[source]
        if direct_progress(candidate):
            allowed.append(i)
        elif remaining > 0 and candidate.action_type.value in ("defer", "reconfigure"):
            steps = macro_steps(candidate)
            if steps is not None and 0 < steps <= remaining and successor_supported(candidate):
                allowed.append(i)
        elif not available_work(env) and candidate.action_type.value == "defer":
            allowed.append(i)
    if not allowed:
        raise NoCertifiedViableAction("E123 bounded-progress eligible frontier is empty")
    return tuple(allowed)


def quantitative_contract(arrival_steps, jobs=36):
    arrivals = tuple(int(value) for value in arrival_steps)
    release_ok = (len(arrivals) == jobs and arrivals[0] == 0
                  and all(0 <= b - a <= WAIT_BUDGET for a, b in zip(arrivals, arrivals[1:])))
    bound = 2 * jobs * (WAIT_BUDGET + PROGRESS_BUDGET)
    return {"release_assumption_passed": release_ok, "progress_events": 2 * jobs,
            "wait_budget": WAIT_BUDGET, "certified_progress_budget": PROGRESS_BUDGET,
            "conditional_completion_bound": bound, "horizon": HORIZON,
            "fits_horizon": release_ok and bound <= HORIZON,
            "conditions": "all selected successors retain a <=20-step progress certificate; nonprogress macros fit remaining wait; strict macro execution succeeds"}
