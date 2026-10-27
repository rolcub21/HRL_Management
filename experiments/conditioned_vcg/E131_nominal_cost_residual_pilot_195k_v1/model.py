from copy import deepcopy
from dataclasses import dataclass,fields,replace
import math
import torch
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.acceleration import SharedProgressAgent
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import ProgressNetwork,INPUT_DIM
from viability_graph_hierarchy import ViabilityGraphCandidateRecord,ACTION_FEATURE_DIM
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E130_exact_macro_cost_diagnostic_194k_v1.costs import predict

ARMS=('total_q','residual_q')
NETWORK='e131_matched_signed_four_head_v1'


@dataclass(frozen=True)
class CostRecord(ViabilityGraphCandidateRecord):
    nominal_components: tuple[float,...]
    nominal_duration: int

    def __post_init__(self):
        super().__post_init__()
        if len(self.nominal_components)!=4 or not all(math.isfinite(v) and v>=0 for v in self.nominal_components):
            raise ValueError('Invalid nominal cost vector')
        if self.nominal_duration<1: raise ValueError('Invalid nominal duration')


def known_tensor(records,like):
    return torch.as_tensor([r.nominal_components for r in records],dtype=like.dtype,device=like.device)


def residual_target(observed,known_current,next_total,duration,done,gamma=1.):
    """Explicit algebra used to verify the equivalent total-space TD loss."""
    discount=observed.new_tensor(gamma)**duration
    return observed-known_current+(discount*(1.-done)).unsqueeze(-1)*next_total


class SignedNetwork(ProgressNetwork):
    def score_component_embeddings(self,current_embedding,successor_embedding,action_features,queue_context):
        parameter=next(self.parameters())
        action=torch.as_tensor(action_features,dtype=parameter.dtype,device=parameter.device)
        if action.ndim==1: action=action.unsqueeze(0)
        count=current_embedding.shape[0]
        if action.shape!=(count,ACTION_FEATURE_DIM) or queue_context.shape!=(count,INPUT_DIM):
            raise ValueError('E131 input dimensions differ')
        latent=self.consequence_trunk(torch.cat((current_embedding,successor_embedding,
            successor_embedding-current_embedding,self.action_encoder(action),
            self.context_encoder(queue_context)),dim=-1))
        return torch.cat(tuple(head(latent) for head in
            (self.service_head,self.wait_head,self.step_head,self.rehandle_head)),dim=-1)


class PilotAgent(SharedProgressAgent):
    def __init__(self,*,variant='total_q',**kwargs):
        if variant not in ARMS: raise ValueError('Unknown E131 variant')
        super().__init__(**kwargs)
        self.variant=variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local=SignedNetwork(self.config).to(self.device)
        self.Q_target=deepcopy(self.Q_local).requires_grad_(False).eval()
        self.optimizer=torch.optim.Adam(self.Q_local.parameters(),lr=5e-5)
        self.td_diagnostics=[]

    def _admissible_prepared(self,snapshot):
        prepared,forced=super()._admissible_prepared(snapshot)
        records=[]
        for record,index in zip(prepared.records,prepared.source_indices):
            prediction=predict(self.context_environment,snapshot.candidates[index],self.config.gamma)
            records.append(CostRecord(**{f.name:getattr(record,f.name) for f in fields(ViabilityGraphCandidateRecord)},
                nominal_components=tuple(prediction['components']),nominal_duration=prediction['duration']))
        return replace(prepared,records=tuple(records)),forced

    def _component_scores(self,records,*,network=None,grad=False):
        records=tuple(records)
        if not all(isinstance(r,CostRecord) for r in records):
            raise TypeError('E131 replay lacks decision-time nominal costs')
        learned=super()._component_scores(records,network=network,grad=grad)
        # ALWAYS expose total costs to selection, Double-DQN target evaluation,
        # frozen-target caching and the inherited conservative preference term.
        return learned+known_tensor(records,learned) if self.variant=='residual_q' else learned

    def _td_batch(self,transitions):
        prediction,target,anchor,pairs=super()._td_batch(transitions)
        # Optimizing Huber((K+R), C+next_total) is exactly the same objective
        # and gradient as Huber(R, C-K+next_total). K is detached fixed data.
        if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
            raise RuntimeError('Nonfinite E131 TD values')
        if len(self.td_diagnostics)<20 or self.gradient_steps%200==0:
            k=known_tensor([t.chosen for t in transitions],prediction)
            learned=prediction-k if self.variant=='residual_q' else prediction
            self.td_diagnostics.append(dict(gradient_step=self.gradient_steps,
                max_abs_total_prediction=float(prediction.detach().abs().max()),
                max_abs_learned_prediction=float(learned.detach().abs().max()),
                max_abs_target=float(target.abs().max()),max_abs_td_error=float((target-prediction.detach()).abs().max())))
        return prediction,target,anchor,pairs

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(),'network_architecture':NETWORK,
                'e131_variant':self.variant,'nominal_costs_in_replay':True,'signed_heads_both_arms':True,
                'cost_heads_nonnegative':False}

    def checkpoint_state(self,*,include_replay=True):
        return {**super().checkpoint_state(include_replay=include_replay),'e131_td_diagnostics':self.td_diagnostics}

    def load_checkpoint_state(self,state,*,resumable=False):
        super().load_checkpoint_state(state,resumable=resumable)
        self.td_diagnostics=list(state.get('e131_td_diagnostics',[]))

    @classmethod
    def from_checkpoint(cls,payload,*,device='cpu',resumable=False,seed=0):
        if payload.get('network_architecture')!=NETWORK or not payload.get('nominal_costs_in_replay'):
            raise ValueError('E131 rejects foreign or total-Q legacy checkpoints')
        agent=cls(config=EpisodicViabilityGraphConfig.from_dict(payload['config']),seed=seed,device=device,
            epsilon=payload['agent_state'].get('epsilon',0.),variant=payload['e131_variant'],
            arm='A_outflow_bounded',absolute_progress=False)
        agent.load_checkpoint_state(payload['agent_state'],resumable=resumable)
        return agent
