"""Same certified shell for learned and rule rankers; old sources unmodified."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from unittest.mock import patch

from experiments.conditioned_vcg.E130_exact_macro_cost_diagnostic_194k_v1.costs import (
    predict as legacy_predict, components, ReleasedJob)
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1 import model as parent
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as shared
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import CertifiedDynamicAgent
from experiments.conditioned_vcg.E126_release_clock_informed_baselines_190k_v1 import clock
from .environment import due

HORIZON = 6000  # Development execution budget only; NOT a proved completion bound.


def predict(env, candidate, gamma=1.):
    result = legacy_predict(env, candidate, gamma)
    jobs = tuple(ReleasedJob(str(b.label), env.public_due_date(b)) for b in env.blocks
                 if b.arrival_step <= env.time_steps and not b.delivered)
    result['components'] = components(env.time_steps, result['duration'], jobs,
                                     candidate.action_type.value, candidate.target_label, gamma)
    result['assumption'] = 'known released jobs, nominal path; unobserved future arrivals excluded'
    return result


class GridAgent(parent.PilotAgent):
    def _admissible_prepared(self, snapshot):
        with patch.object(parent, 'predict', predict):
            prepared, forced = super()._admissible_prepared(snapshot)
        # The inherited progress observation has a fixed 3600 normalization.
        # Keep width and semantics, but normalize elapsed age by THIS budget.
        env = self.context_environment
        age = env.progress_clock.age(snapshot.decision_epoch)
        records = tuple(replace(r, current_state=replace(r.current_state,
            progress_context=(min(age/HORIZON, 1.), *r.current_state.progress_context[1:])))
            for r in prepared.records)
        return replace(prepared, records=records), forced

    @classmethod
    def bound(cls, env):
        agent = cls(config=parent.EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=HORIZON),
                    seed=0, device='cpu', epsilon=0., arm='A_outflow_bounded',
                    absolute_progress=False, variant='residual_q')
        agent.bind_context_environment(env, wait_cap=80, required_jobs=len(env.blocks), max_steps=HORIZON)
        return agent

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(), 'e136_external_due_dates': True,
                'deployment_checkpoint_eligible': False, 'e136_development_only': True}

    @classmethod
    def from_checkpoint(cls, payload, **kwargs):
        if not payload.get('e136_external_due_dates'):
            raise ValueError('GridAgent rejects checkpoints lacking external-date contract')
        return super().from_checkpoint(payload, **kwargs)


def certified_pair(env):
    # Both literally use GridAgent._admissible_prepared and its identical guard,
    # cost records, input schema and liveness rule. Only ranking is different.
    return GridAgent.bound(env), CertifiedDynamicAgent(GridAgent.bound(env))


@contextmanager
def interface(env=None):
    # E126's planning-view/projected-admission adapters call this same function.
    # Their otherwise unchanged rules now see EXPLICIT dates rather than a+p.
    # No environment duration field, baseline logic or old source is modified.
    with ExitStack() as stack:
        stack.enter_context(patch.object(clock, 'remaining', due))
        if env is not None:
            def view(*args, **kwargs):
                return clock.planning_view(env, 'e136_external_date_view', *args, **kwargs)
            for module in (clock.selector, clock.accept, clock.ga, clock.capacity_ga):
                stack.enter_context(patch.object(module, 'BlockView', view))
            stack.enter_context(patch.object(clock.scheduler.DurationAwareAtomicScheduler,
                                            '_projected_plan', clock.projected_plan))
        rule = stack.enter_context(shared.controller('A_outflow_bounded'))
        yield rule


def horizon_audit(instance, initial_count):
    arrivals = instance.arrival_steps[initial_count:]
    events = initial_count + 2*len(arrivals)
    return {'initial_jobs': initial_count, 'incoming_jobs': len(arrivals),
            'required_new_flow_events': events,
            'maximum_incoming_release_gap': max((b-a for a,b in zip(arrivals,arrivals[1:])), default=0),
            'development_horizon': HORIZON,
            'legacy_3600_bound_applicable': False,
            'completion_guarantee_established': False,
            'reason': 'A-outflow bounded-wait guard has no verified uniform next-progress duration; '
                      'a generous execution horizon is not a liveness proof'}
