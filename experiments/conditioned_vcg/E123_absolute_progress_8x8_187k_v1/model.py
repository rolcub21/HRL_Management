"""Same four-head graph critic, with explicit replayed progress-clock inputs."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from PSLAP.viability import RecoveryState
from viability_graph_hierarchy import ACTION_FEATURE_DIM
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import QueueObservedRecoveryState
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import model as base
from .liveness import AbsoluteProgressGuard, eligible_indices, WAIT_BUDGET, PROGRESS_BUDGET, HORIZON

PROGRESS_CONTEXT_NAMES = (
    "absolute_time_since_last_flow_progress_fraction_of_horizon",
    "remaining_absolute_wait_fraction",
    "legacy_defer_streak_fraction",
    "remaining_committed_progress_budget_fraction",
)
PROGRESS_DIM = len(PROGRESS_CONTEXT_NAMES)
INPUT_DIM = base.QUEUE_CONTEXT_DIM + PROGRESS_DIM
NETWORK = "e123_progress_observed_four_cost_graph_q_v1"


@dataclass(frozen=True)
class ProgressObservedState(QueueObservedRecoveryState):
    progress_context: tuple[float, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if len(self.progress_context) != PROGRESS_DIM or not all(
                math.isfinite(v) and 0 <= v <= 1 for v in self.progress_context):
            raise ValueError("E123 invalid progress context")


def attach_progress(state, context):
    return ProgressObservedState(**{f.name: getattr(state, f.name) for f in fields(RecoveryState)},
                                 queue_context=tuple(state.queue_context), progress_context=tuple(context))


class ProgressNetwork(base.FactorizedQNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.context_encoder = nn.Sequential(nn.Linear(INPUT_DIM, config.action_embedding_dim),
                                             nn.SiLU(), nn.LayerNorm(config.action_embedding_dim))

    def context_tensor(self, states):
        states = tuple(states)
        if not all(isinstance(state, ProgressObservedState) for state in states):
            raise TypeError("E123 replay lacks progress-clock inputs")
        parameter = next(self.parameters())
        return torch.as_tensor([s.queue_context + s.progress_context for s in states],
                               dtype=parameter.dtype, device=parameter.device)

    def score_component_embeddings(self, current_embedding, successor_embedding, action_features, queue_context):
        parameter = next(self.parameters())
        action = torch.as_tensor(action_features, dtype=parameter.dtype, device=parameter.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        count = current_embedding.shape[0]
        if action.shape != (count, ACTION_FEATURE_DIM) or queue_context.shape != (count, INPUT_DIM):
            raise ValueError("E123 feature dimensions differ")
        latent = self.consequence_trunk(torch.cat((current_embedding, successor_embedding,
            successor_embedding - current_embedding, self.action_encoder(action),
            self.context_encoder(queue_context)), dim=-1))
        return torch.cat(tuple(F.softplus(head(latent)) for head in
                         (self.service_head, self.wait_head, self.step_head, self.rehandle_head)), dim=-1)


class ProgressAgent(base.FactorizedConservativeAgent):
    def __init__(self, *, absolute_progress=False, arm="A_outflow_bounded", **kwargs):
        super().__init__(**kwargs)
        self.absolute_progress, self.arm = bool(absolute_progress), arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = ProgressNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.Q_local.parameters(), lr=base.learning_rate_at(0))
        if self.absolute_progress:
            self.recovery_witness_guard = AbsoluteProgressGuard()

    def _admissible_prepared(self, snapshot):
        env = self.context_environment
        if self.absolute_progress:
            self.recovery_witness_guard.environment = env
        prepared, forced = super()._admissible_prepared(snapshot)
        if self.absolute_progress:
            keep = eligible_indices(snapshot, prepared, env, forced)
            prepared = replace(prepared, records=tuple(prepared.records[i] for i in keep),
                source_indices=tuple(prepared.source_indices[i] for i in keep),
                liveness_restricted=prepared.liveness_restricted or len(keep) != len(prepared.records))
        age = env.progress_clock.age(snapshot.decision_epoch)
        guard = self.recovery_witness_guard
        commit = getattr(guard, "commit_time", None)
        context = (min(age / HORIZON, 1.), max(0., WAIT_BUDGET - age) / WAIT_BUDGET,
                   min(snapshot.audit.consecutive_defer_decisions / 16., 1.),
                   max(0., PROGRESS_BUDGET - (snapshot.decision_epoch - commit)) / PROGRESS_BUDGET
                   if guard.active and commit is not None else 1.)
        records = tuple(replace(record, current_state=attach_progress(record.current_state, context))
                        for record in prepared.records)
        return replace(prepared, records=records), forced

    def checkpoint_metadata(self):
        value = super().checkpoint_metadata()
        value.update(network_architecture=NETWORK, e123_arm=self.arm,
                     e123_absolute_progress=self.absolute_progress,
                     progress_context_names=PROGRESS_CONTEXT_NAMES,
                     progress_context_stored_in_replay=True, context_input_dim=INPUT_DIM,
                     e123_no_foreign_checkpoint_initialization=True,
                     learning_rate_schedule_by_decisions=((8000, 5e-5), (math.inf, 1e-5)))
        return value

    @classmethod
    def from_checkpoint(cls, payload, *, device="cpu", resumable=False, seed=0):
        if (payload.get("network_architecture") != NETWORK or payload.get("context_input_dim") != INPUT_DIM
                or tuple(payload.get("progress_context_names", ())) != PROGRESS_CONTEXT_NAMES):
            raise ValueError("E123 rejects a foreign or clock-unobserved checkpoint")
        agent = cls(config=EpisodicViabilityGraphConfig.from_dict(payload["config"]), seed=seed,
                    device=device, epsilon=float(payload["agent_state"].get("epsilon", 0.)),
                    arm=payload["e123_arm"], absolute_progress=payload["e123_absolute_progress"])
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent
