"""Observed arrivals only; fixed 120-step window, never regime identifiers."""
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math

import torch
from torch import nn

from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import ProgressObservedState, INPUT_DIM
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1.model import SignedNetwork
from . import run as v1

WINDOW = 120
FEATURES = ('incoming_arrivals_last_120_primitive_steps_per_step',
            'primitive_steps_since_last_observed_incoming_arrival_or_episode_start')


def observed_arrivals(env):
    now = int(env.time_steps)
    initial = env.initial_labels
    times = [int(b.arrival_step) for b in env.blocks
             if b.arrival_step <= now and b.label not in initial]
    # Window is (t-120,t], with release at zero included during startup.
    rate = sum(t > now-WINDOW for t in times)/float(WINDOW)
    age = now-max(times) if times else now
    return (rate, float(age))


@dataclass(frozen=True)
class ArrivalState(ProgressObservedState):
    arrival_context: tuple[float,...] = (0.,0.)

    def __post_init__(self):
        super().__post_init__()
        if len(self.arrival_context)!=2 or not all(math.isfinite(v) and v>=0 for v in self.arrival_context):
            raise ValueError('Invalid observed arrival context')


class ArrivalNetwork(SignedNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.arrival_projection = nn.Linear(2, config.action_embedding_dim, bias=False)
        nn.init.zeros_(self.arrival_projection.weight)

    def context_tensor(self, states):
        states = tuple(states)
        if not all(isinstance(s, ArrivalState) for s in states):
            raise TypeError('Arrival history must be stored in replay')
        parameter = next(self.parameters())
        return torch.as_tensor([s.queue_context+s.progress_context+s.arrival_context for s in states],
                               dtype=parameter.dtype,device=parameter.device)

    def score_component_embeddings(self, current_embedding, successor_embedding, action_features, queue_context):
        parameter = next(self.parameters())
        action = torch.as_tensor(action_features,dtype=parameter.dtype,device=parameter.device)
        if action.ndim==1:
            action=action.unsqueeze(0)
        count=current_embedding.shape[0]
        if queue_context.shape!=(count,INPUT_DIM+2):
            raise ValueError('Arrival context dimensions differ')
        context=self.context_encoder(queue_context[:,:INPUT_DIM])+self.arrival_projection(queue_context[:,INPUT_DIM:])
        latent=self.consequence_trunk(torch.cat((current_embedding,successor_embedding,
            successor_embedding-current_embedding,self.action_encoder(action),context),dim=-1))
        return torch.cat(tuple(head(latent) for head in
            (self.service_head,self.wait_head,self.step_head,self.rehandle_head)),dim=-1)


class ArrivalAgent(v1.WarmAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local=ArrivalNetwork(self.config).to(self.device)
        self.Q_target=deepcopy(self.Q_local).requires_grad_(False).eval()
        self.optimizer=torch.optim.Adam(self.Q_local.parameters(),lr=5e-5)

    def _admissible_prepared(self, snapshot):
        prepared, forced=super()._admissible_prepared(snapshot)
        context=observed_arrivals(self.context_environment)
        records=tuple(replace(r,current_state=ArrivalState(
            **{f.name:getattr(r.current_state,f.name) for f in fields(ProgressObservedState)},
            arrival_context=context)) for r in prepared.records)
        return replace(prepared,records=records),forced

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(),'e144_arrival_v2':True,
                'arrival_context_names':FEATURES,'arrival_window_steps':WINDOW,
                'arrival_context_in_replay':True,'future_release_schedule_visible':False}

    @classmethod
    def from_checkpoint(cls,payload,**kwargs):
        if not payload.get('e144_arrival_v2') or payload.get('arrival_window_steps')!=WINDOW:
            raise ValueError('Requires arrival-aware E144 v2 checkpoint')
        return super().from_checkpoint(payload,**kwargs)


def initialize(payload,device='cpu'):
    config=replace(v1.EpisodicViabilityGraphConfig.from_dict(payload['config']),episode_horizon_steps=6000)
    agent=ArrivalAgent(config=config,seed=0,device=device,epsilon=.9,
        variant='residual_q',arm='A_outflow_bounded',absolute_progress=False)
    for network,key in ((agent.Q_local,'Q_local'),(agent.Q_target,'Q_target')):
        missing,unexpected=network.load_state_dict(payload['agent_state'][key],strict=False)
        if missing!=['arrival_projection.weight'] or unexpected:
            raise ValueError('Unexpected source network migration')
        if bool(network.arrival_projection.weight.count_nonzero()):
            raise ValueError('New arrival connections must start at zero')
    return agent


def defer_streak(previous, action_type, events_before, events_after):
    """Same flow-event reset as E137/E141 and the certified baseline."""
    return previous+1 if action_type=='defer' and events_after==events_before else 0


def episode(agent,env,*,training,emit=lambda _:None,macro_limit=1000):
    """Native learner, with the baseline-matched E137 Defer-counter semantics."""
    from time import perf_counter
    from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.controller import interface,HORIZON
    from PSLAP.viability_filter import ViabilitySearchConfig
    from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
    from experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1.rollout import collect
    shared=v1.base.shared
    audit=shared.original.replay.audit
    agent.bind_context_environment(env,wait_cap=80,required_jobs=len(env.blocks),max_steps=HORIZON)
    agent.reset_episode_state()
    started=perf_counter()
    count,streak,reason=0,0,None
    before_updates=agent.gradient_steps
    with interface(env) as rule:
        cache=shared.original.replay.reuse.TimingInvariantCertificateCache()
        def snapshot():
            return audit.enumerate_viability_candidates(env,consecutive_defer_decisions=streak,
                search_config=ViabilitySearchConfig(max_nodes=2000),liveness_rule=rule,cache=cache,
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
        current=snapshot()
        while count<macro_limit and env.time_steps<HORIZON and not all(b.delivered for b in env.blocks):
            eps=v1.base.p.epsilon(agent.decision_count) if training else 0.
            if training:
                agent.set_epsilon(eps)
            events_before=env.progress_clock.events
            decision=agent.select(current,training=training,epsilon=eps)
            execution=audit._execute_audit_macro(env,decision.candidate,gamma=1.,
                remaining_steps=HORIZON-env.time_steps,evaluation=not training)
            count+=1
            if not execution.option_success or execution.duration<1:
                raise RuntimeError(f'macro failed: {execution.failure_reason}')
            streak=defer_streak(streak,decision.candidate.action_type.value,
                                events_before,env.progress_clock.events)
            complete=all(b.delivered for b in env.blocks)
            terminal=complete or env.time_steps>=HORIZON or count>=macro_limit
            nxt=None if terminal else snapshot()
            if training:
                agent.remember(decision,reward=execution.discounted_return,duration=execution.duration,
                               next_snapshot=nxt,done=terminal)
                loss=agent.learn()
                if loss is not None and not math.isfinite(loss):
                    raise RuntimeError('nonfinite learning loss')
            else:
                agent.observe_outcome(decision,next_snapshot=nxt,done=terminal)
            emit(dict(decisions=count,steps=env.time_steps,delivered=sum(b.delivered for b in env.blocks),
                      required=len(env.blocks),epsilon=eps,gradient_steps=agent.gradient_steps))
            if terminal:
                if not complete:
                    reason='primitive_horizon' if env.time_steps>=HORIZON else 'decision_cap'
                break
            current=nxt
    strict=all(b.delivered for b in env.blocks) and reason is None
    return dict(strict_complete=strict,reason=reason,metrics=collect(env,strict),
        decisions=count,gradient_updates=agent.gradient_steps-before_updates,
        wall_seconds=perf_counter()-started,epsilon=agent.epsilon if training else 0.)
