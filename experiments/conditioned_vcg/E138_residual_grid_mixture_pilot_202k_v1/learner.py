import math
from time import perf_counter

import torch
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.controller import GridAgent, interface, HORIZON
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1.model import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1.rollout import RecordedGridEnv, collect
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as shared
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from .protocol import epsilon


class MixtureAgent(GridAgent):
    def _td_batch(self, transitions):
        result = super()._td_batch(transitions)
        prediction, target = result[:2]
        if float((prediction.detach()-target).abs().max()) > 1e6:
            raise RuntimeError('TD error exceeded frozen runtime sanity gate')
        return result

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(), 'e138_mixture_pilot':True,
                'e136_development_only':False, 'paper_evidence':False}

    @classmethod
    def from_checkpoint(cls, payload, **kwargs):
        if not payload.get('e138_mixture_pilot'):
            raise ValueError('E138 rejects a foreign checkpoint')
        return super().from_checkpoint(payload, **kwargs)


def fresh(device='cpu', config=None):
    return MixtureAgent(config=config or EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=HORIZON),
                        seed=0, device=device, epsilon=.9, arm='A_outflow_bounded',
                        absolute_progress=False, variant='residual_q')


def episode(agent, env, *, training, emit=lambda _:None, macro_limit=1000):
    agent.bind_context_environment(env,wait_cap=80,required_jobs=len(env.blocks),max_steps=HORIZON)
    agent.reset_episode_state()
    started = perf_counter()
    count, streak, reason = 0, 0, None
    audit = shared.original.replay.audit
    before_updates = agent.gradient_steps
    with interface(env) as rule:
        cache = shared.original.replay.reuse.TimingInvariantCertificateCache()
        def snapshot():
            return audit.enumerate_viability_candidates(env, consecutive_defer_decisions=streak,
                search_config=ViabilitySearchConfig(max_nodes=2000), liveness_rule=rule, cache=cache,
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
        current = snapshot()
        while count < macro_limit and env.time_steps < HORIZON and not all(b.delivered for b in env.blocks):
            eps = epsilon(agent.decision_count) if training else 0.
            if training:
                agent.set_epsilon(eps)
            decision = agent.select(current, training=training, epsilon=eps)
            execution = audit._execute_audit_macro(env,decision.candidate,gamma=1.,
                remaining_steps=HORIZON-env.time_steps,evaluation=not training)
            count += 1
            if not execution.option_success or execution.duration < 1:
                raise RuntimeError(f'macro failed: {execution.failure_reason}')
            outcome = getattr(decision.option,'last_outcome',None)
            if decision.candidate.action_type.value == 'defer':
                event = isinstance(outcome,dict) and outcome.get('reason') == 'observed_event'
                streak = 0 if event else streak+1
            else:
                streak = 0
            complete = all(b.delivered for b in env.blocks)
            terminal = complete or env.time_steps >= HORIZON or count >= macro_limit
            # The decision cap is a real truncation boundary, never a completed
            # trajectory. Failed/truncated episodes are not committed to resume.
            nxt = None if terminal else snapshot()
            if training:
                agent.remember(decision,reward=execution.discounted_return,duration=execution.duration,
                               next_snapshot=nxt,done=terminal)
                loss = agent.learn()
                if loss is not None and not math.isfinite(loss):
                    raise RuntimeError('nonfinite learning loss')
                if agent.td_diagnostics and agent.td_diagnostics[-1]['max_abs_td_error'] > 1e6:
                    raise RuntimeError('TD error exceeded frozen runtime sanity gate')
            else:
                agent.observe_outcome(decision,next_snapshot=nxt,done=terminal)
            emit(dict(decisions=count,steps=env.time_steps,delivered=sum(b.delivered for b in env.blocks),
                      required=len(env.blocks),epsilon=eps,gradient_steps=agent.gradient_steps))
            if terminal:
                if not complete:
                    reason = 'primitive_horizon' if env.time_steps >= HORIZON else 'decision_cap'
                break
            current = nxt
    strict = all(b.delivered for b in env.blocks) and reason is None
    return dict(strict_complete=strict,reason=reason,metrics=collect(env,strict),decisions=count,
                gradient_updates=agent.gradient_steps-before_updates,wall_seconds=perf_counter()-started,
                epsilon=agent.epsilon if training else 0.)
