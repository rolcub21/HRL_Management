"""Policy/operator-consistent constrained VCG V2.2 core and runtime.

The safety, Hold, graph, critic, replay-record, and SMDP mechanics are inherited
from the audited V2/V2.1 implementation.  Policy realization is new and
isolated: V2.2 always samples the induced nested Lagrangian policy.  The
external ``training`` flag controls mutation and option evaluation mode only;
it never switches the controller to MAP.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math
import random
from typing import Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F

from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityCertificateCache,
    ViabilityMode,
)
from PSLAP.viability_candidates_hold_v2 import (
    CERTIFIED_HOLD_INTERFACE_V2,
    CERTIFIED_HOLD_RULE_V2,
    HIDDEN_SCHEDULE_CONTRACT_V2,
    ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
    CertifiedHoldRuleV2,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from viability_graph_hierarchy import (
    NoCertifiedViableAction,
    RecoveryWitnessMismatch,
    ViabilityGraphDecision,
)
import viability_graph_constrained_v2 as v2
from train_vcg_constrained_v2 import COST_DEFINITION, FROZEN_OBJECTIVE_SPEC
from train_vcg_constrained_v2_2 import (
    BACKUP_VERSION,
    CHECKPOINT_FAMILY,
    CHECKPOINT_SCHEMA_VERSION,
    CONTROLLER_ARCHITECTURE,
    FROZEN_VIABILITY_SEARCH_CONFIG,
    METHOD_VERSION,
    POLICY_DIAGNOSTIC_DISTRIBUTION,
    POLICY_REALIZATION,
    POLICY_SCHEDULE_PROTOCOL,
    REPLAY_RNG_SEED,
    VALIDATION_PROTOCOL,
    schedule_for_episode,
)


V2_2_POLICY_VERSION = "four_group_induced_nested_stochastic_lagrangian_policy_v2_2"
V2_2_DIAGNOSTIC_VERSION = "executed_outer_and_selected_within_distribution_v2_2"


def _validate_schedule_mapping(state: Mapping) -> dict:
    if not isinstance(state, Mapping):
        raise TypeError("V2.2 schedule state must be a mapping")
    received = dict(state)
    episode = received.get("episode_number")
    if isinstance(episode, bool) or not isinstance(episode, int):
        raise ValueError("V2.2 schedule requires integer episode_number")
    context = received.get("execution_context")
    if context == "training":
        expected = schedule_for_episode(episode, validation=False).to_dict()
    elif context == "validation_deployment":
        expected = schedule_for_episode(episode, validation=True).to_dict()
    else:
        raise ValueError("V2.2 schedule execution_context is invalid")
    if received != expected:
        differences = {
            key: (received.get(key), expected.get(key))
            for key in sorted(set(received) | set(expected))
            if received.get(key) != expected.get(key)
        }
        raise ValueError(f"V2.2 schedule does not match frozen clock: {differences!r}")
    return deepcopy(expected)


class ConstrainedV22HierarchyAgent(v2.ConstrainedV2HierarchyAgent):
    """V2 vector critic with independent behavior and replay RNG streams."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._v2_2_schedule_state: Optional[dict] = None
        self.behavior_rng = random.Random(0)
        self.replay_rng = random.Random(REPLAY_RNG_SEED)
        self.behavior_rng_seed: Optional[int] = None
        self.replay_rng_seed = REPLAY_RNG_SEED
        self.behavior_rng_reset_count = 0
        self._policy_diagnostic_count = 0
        self._outer_entropy_sum = 0.0
        self._outer_map_probability_sum = 0.0
        self._within_entropy_sum = 0.0
        self._within_map_probability_sum = 0.0
        self._last_policy_diagnostic: Optional[dict] = None

    @property
    def schedule_state(self) -> Mapping:
        return {} if self._v2_2_schedule_state is None else deepcopy(self._v2_2_schedule_state)

    def set_schedule_state(self, state: Mapping) -> None:
        self._v2_2_schedule_state = _validate_schedule_mapping(state)

    def _require_schedule(self) -> dict:
        if self._v2_2_schedule_state is None:
            raise RuntimeError("V2.2 policy schedule has not been installed")
        return self._v2_2_schedule_state

    @property
    def within_group_temperature_values(self) -> tuple[float, float, float, float]:
        values = tuple(float(x) for x in self._require_schedule()["within_group_temperatures"])
        if len(values) != 4:
            raise RuntimeError("V2.2 schedule lost a within-group temperature")
        return values

    @property
    def group_temperature(self) -> float:
        value = float(self._require_schedule()["group_temperature"])
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("V2.2 group temperature is invalid")
        return value

    def reset_behavior_rng(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("V2.2 behavior RNG seed must be a non-negative integer")
        self.behavior_rng = random.Random(int(seed))
        self.behavior_rng_seed = int(seed)
        self.behavior_rng_reset_count += 1

    @staticmethod
    def _distribution_diagnostic(logits: torch.Tensor) -> tuple[float, float]:
        probabilities = F.softmax(logits, dim=0)
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12)))
        return float(entropy.item()), float(probabilities.max().item())

    def _record_policy_diagnostic(self, decision) -> None:
        outer = torch.as_tensor([value for _, value in decision.mode_values], dtype=torch.float64)
        outer_entropy, outer_max = self._distribution_diagnostic(outer / self.group_temperature)
        group = int(decision.record.group_id)
        indices = [i for i, gid in enumerate(decision.prepared_snapshot.mode_ids) if int(gid) == group]
        within = torch.as_tensor([decision.q_values[i] for i in indices], dtype=torch.float64)
        within_entropy, within_max = self._distribution_diagnostic(
            within / self.within_group_temperature_values[group]
        )
        diagnostic = {
            "protocol": V2_2_DIAGNOSTIC_VERSION,
            "distribution_semantics": POLICY_DIAGNOSTIC_DISTRIBUTION,
            "policy_realization": POLICY_REALIZATION,
            "behavior_rng_seed": self.behavior_rng_seed,
            "outer_entropy": outer_entropy,
            "outer_map_probability": outer_max,
            "selected_within_entropy": within_entropy,
            "selected_within_map_probability": within_max,
        }
        self._policy_diagnostic_count += 1
        self._outer_entropy_sum += outer_entropy
        self._outer_map_probability_sum += outer_max
        self._within_entropy_sum += within_entropy
        self._within_map_probability_sum += within_max
        self._last_policy_diagnostic = diagnostic
        if self.decision_log:
            self.decision_log[-1]["v2_2_policy_diagnostic"] = dict(diagnostic)

    def policy_diagnostic_state(self) -> dict:
        count = int(self._policy_diagnostic_count)
        mean = lambda value: None if count == 0 else float(value / count)
        return {
            "protocol": V2_2_DIAGNOSTIC_VERSION,
            "distribution_semantics": POLICY_DIAGNOSTIC_DISTRIBUTION,
            "count": count,
            "outer_entropy_sum": self._outer_entropy_sum,
            "outer_map_probability_sum": self._outer_map_probability_sum,
            "selected_within_entropy_sum": self._within_entropy_sum,
            "selected_within_map_probability_sum": self._within_map_probability_sum,
            "mean_outer_entropy": mean(self._outer_entropy_sum),
            "mean_outer_map_probability": mean(self._outer_map_probability_sum),
            "mean_selected_within_entropy": mean(self._within_entropy_sum),
            "mean_selected_within_map_probability": mean(self._within_map_probability_sum),
            "last": None if self._last_policy_diagnostic is None else dict(self._last_policy_diagnostic),
        }

    def select(self, snapshot, *, training: bool = True, epsilon=None):
        """Sample V2.2's policy; ``training`` authorizes mutation elsewhere only."""

        schedule = self._require_schedule()
        expected_training = schedule["execution_context"] == "training"
        if bool(training) != expected_training:
            raise RuntimeError("V2.2 external training flag disagrees with execution context")
        if self.behavior_rng_seed is None:
            raise RuntimeError("V2.2 behavior RNG was not installed for this rollout")
        epsilon_value = self.epsilon if epsilon is None else self._validate_epsilon(epsilon)
        if epsilon_value != 0.0:
            raise RuntimeError("V2.2 epsilon invariant was violated")

        frontier, _ = v2._unwrap_snapshot(snapshot)
        prepared, liveness_forced = self._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("exact verifier exposed no SAFE V2.2 macro")
        operational, rehandle = self._score_records(prepared.records)
        elapsed = {record.elapsed_steps for record in prepared.records}
        if len(elapsed) != 1:
            raise ValueError("one V2.2 frontier contains mixed elapsed contexts")
        op_weight = self._operational_weight(elapsed.pop())
        merit = op_weight * operational - self.dual_lambda * rehandle
        groups = torch.as_tensor(prepared.mode_ids, dtype=torch.long, device=self.device)
        _, _, group_values, live_groups = v2.shared_lagrangian_policy_values(
            operational, rehandle, operational, rehandle, groups,
            self.within_group_temperatures, self.group_temperature,
            self.dual_lambda, operational_weight=op_weight,
        )
        if liveness_forced:
            selected_index = 0
            selection_source = "exact_recovery_witness_guard"
        else:
            outer_probabilities = F.softmax(group_values / self.group_temperature, dim=0).detach().cpu().tolist()
            group_position = self.behavior_rng.choices(
                range(len(live_groups)), weights=outer_probabilities, k=1
            )[0]
            selected_group = int(live_groups[group_position].item())
            indices = [i for i, gid in enumerate(prepared.mode_ids) if gid == selected_group]
            within_probabilities = F.softmax(
                merit[indices] / self.within_group_temperatures[selected_group], dim=0
            ).detach().cpu().tolist()
            selected_index = self.behavior_rng.choices(indices, weights=within_probabilities, k=1)[0]
            selection_source = "induced_nested_stochastic_lagrangian_policy"

        source_index = prepared.source_indices[selected_index]
        candidate = frontier.candidates[source_index]
        record = prepared.records[selected_index]
        exact_rank_progress = bool(
            frontier.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None and candidate.rank_delta > 0
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=record,
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merit.tolist()),
            mode_values=tuple(
                (v2.ID_TO_CONTROL_GROUP[int(group_id.item())], float(value.item()))
                for value, group_id in zip(group_values, live_groups)
            ),
            explored=False,
            selection_source=selection_source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        self.decision_count += 1
        self.group_decisions[v2.ID_TO_CONTROL_GROUP[record.group_id]] += 1
        self.action_decisions[candidate.action_type.value] += 1
        self.selection_sources[selection_source] += 1
        self.safe_candidates_scored += len(prepared.records)
        self.interface_rejections += prepared.interface_rejection_count
        self.exact_rejections_seen += int(frontier.audit.fail_closed_rejection_count)
        self.decision_log.append({
            "decision_index": self.decision_count - 1,
            "episode_instance_id": frontier.episode_instance_id,
            "decision_epoch": int(frontier.decision_epoch),
            "candidate_key": record.key,
            "selection_source": selection_source,
            "policy_realization": POLICY_REALIZATION,
            "behavior_rng_seed": self.behavior_rng_seed,
        })
        self._record_policy_diagnostic(decision)
        return decision

    def learn(self) -> Optional[dict]:
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        transitions = self.replay.sample_group_balanced(self.config.batch_size, self.replay_rng)
        op_prediction, op_target, cost_prediction, cost_target = self._td_batch(transitions)
        operational_loss = F.smooth_l1_loss(op_prediction, op_target, beta=self.config.huber_delta)
        rehandle_loss = F.smooth_l1_loss(cost_prediction, cost_target, beta=self.config.huber_delta)
        loss = operational_loss + self.config.cost_loss_weight * rehandle_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())
            self.Q_target.requires_grad_(False)
            self.Q_target.eval()
            self.target_updates += 1
        values = {
            "total": float(loss.detach().item()),
            "operational": float(operational_loss.detach().item()),
            "rehandle": float(rehandle_loss.detach().item()),
            "gradient_norm": float(gradient_norm),
        }
        self.total_loss_history.append(values["total"])
        self.operational_loss_history.append(values["operational"])
        self.rehandle_loss_history.append(values["rehandle"])
        return values

    def checkpoint_metadata(self) -> dict:
        metadata = super().checkpoint_metadata()
        metadata.update({
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "policy_version": V2_2_POLICY_VERSION,
            "method_version": METHOD_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "policy_realization": POLICY_REALIZATION,
            "training_policy": POLICY_REALIZATION,
            "backup_policy": POLICY_REALIZATION,
            "validation_policy": POLICY_REALIZATION,
            "deployment_policy": POLICY_REALIZATION,
            "same_policy_for_behavior_backup_validation_deployment": True,
            "map_policy_available": False,
            "policy_realization_decoupled_from_training_flag": True,
            "behavior_rng_scope": "rollout_action_sampling_only",
            "replay_rng_scope": "persistent_replay_sampling_only",
            "replay_rng_seed": REPLAY_RNG_SEED,
            "policy_diagnostic_version": V2_2_DIAGNOSTIC_VERSION,
            "policy_diagnostic_distribution": POLICY_DIAGNOSTIC_DISTRIBUTION,
            "schedule_state": dict(self._require_schedule()),
            "within_group_temperatures": self.within_group_temperature_values,
            "group_temperature": self.group_temperature,
        })
        return metadata

    def checkpoint_state(self, *, include_replay: bool = True) -> dict:
        state = super().checkpoint_state(include_replay=include_replay)
        state["schedule_state"] = dict(self._require_schedule())
        state["policy_diagnostics"] = self.policy_diagnostic_state()
        state["behavior_rng_state"] = self.behavior_rng.getstate()
        state["behavior_rng_seed"] = self.behavior_rng_seed
        state["behavior_rng_reset_count"] = self.behavior_rng_reset_count
        state["replay_rng_state"] = self.replay_rng.getstate()
        state["replay_rng_seed"] = self.replay_rng_seed
        return state

    def load_checkpoint_state(self, state: Mapping, *, resumable: bool) -> None:
        if "schedule_state" not in state:
            raise ValueError("V2.2 agent state is missing schedule_state")
        schedule = _validate_schedule_mapping(state["schedule_state"])
        v2.ConstrainedV2HierarchyAgent.load_checkpoint_state(self, state, resumable=resumable)
        self._v2_2_schedule_state = schedule
        if state.get("replay_rng_seed") != REPLAY_RNG_SEED:
            raise ValueError("V2.2 replay RNG seed mismatch")
        if resumable:
            for key in ("behavior_rng_state", "replay_rng_state"):
                if key not in state:
                    raise ValueError(f"resumable V2.2 state is missing {key}")
            self.behavior_rng.setstate(state["behavior_rng_state"])
            self.replay_rng.setstate(state["replay_rng_state"])
            self.behavior_rng_seed = state.get("behavior_rng_seed")
            self.behavior_rng_reset_count = int(state.get("behavior_rng_reset_count", 0))
        else:
            # A weight-only clone must install a new rollout-specific action
            # stream explicitly.  Never label constructor RNG state as the
            # checkpoint's saved behavior stream.
            self.behavior_rng = random.Random(0)
            self.behavior_rng_seed = None
            self.behavior_rng_reset_count = 0
        diagnostics = state.get("policy_diagnostics", {})
        if diagnostics:
            if diagnostics.get("protocol") != V2_2_DIAGNOSTIC_VERSION:
                raise ValueError("V2.2 policy diagnostic version mismatch")
            self._policy_diagnostic_count = int(diagnostics.get("count", 0))
            self._outer_entropy_sum = float(diagnostics.get("outer_entropy_sum", 0.0))
            self._outer_map_probability_sum = float(diagnostics.get("outer_map_probability_sum", 0.0))
            self._within_entropy_sum = float(diagnostics.get("selected_within_entropy_sum", 0.0))
            self._within_map_probability_sum = float(diagnostics.get("selected_within_map_probability_sum", 0.0))
            last = diagnostics.get("last")
            self._last_policy_diagnostic = None if last is None else dict(last)

    @classmethod
    def from_checkpoint(cls, payload: Mapping, *, device="cpu", resumable=False, seed=10):
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "policy_version": V2_2_POLICY_VERSION,
            "method_version": METHOD_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "policy_realization": POLICY_REALIZATION,
            "training_policy": POLICY_REALIZATION,
            "backup_policy": POLICY_REALIZATION,
            "validation_policy": POLICY_REALIZATION,
            "deployment_policy": POLICY_REALIZATION,
            "same_policy_for_behavior_backup_validation_deployment": True,
            "map_policy_available": False,
            "policy_realization_decoupled_from_training_flag": True,
            "replay_rng_seed": REPLAY_RNG_SEED,
            "policy_diagnostic_version": V2_2_DIAGNOSTIC_VERSION,
            "policy_diagnostic_distribution": POLICY_DIAGNOSTIC_DISTRIBUTION,
            "single_shared_lagrangian_policy_for_both_target_heads": True,
            "entropy_or_kl_added_to_raw_component_targets": False,
            "exact_safe_mask_authoritative": True,
            "candidate_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "cost_definition": COST_DEFINITION,
            "baseline_teacher": False,
            "scalarized_reward_stored_in_replay": False,
        }
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches:
            raise ValueError(f"incompatible constrained V2.2 checkpoint: {mismatches!r}")
        agent = cls(
            config=v2.ConstrainedV2Config.from_dict(payload["config"]),
            seed=int(seed), device=device, epsilon=0.0,
            dual_lambda=float(payload["agent_state"]["dual_lambda"]),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=bool(resumable))
        if dict(payload.get("schedule_state", {})) != dict(agent.schedule_state):
            raise ValueError("V2.2 checkpoint metadata/state schedule mismatch")
        return agent


class _FrozenV22EvaluationAgent(ConstrainedV22HierarchyAgent):
    def remember(self, *args, **kwargs):
        raise RuntimeError("V2.2 frozen validation clone forbids replay mutation")

    def learn(self, *args, **kwargs):
        raise RuntimeError("V2.2 frozen validation clone forbids learning")


class ConstrainedV22DevelopmentRuntime(v2.ConstrainedV2SmokeRuntime):
    """Real environment adapter with explicit stochastic policy RNG control."""

    def __init__(self, *, args, contract: Mapping) -> None:
        from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec

        expected = {
            "checkpoint_family": CHECKPOINT_FAMILY,
            "controller": CONTROLLER_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "method_version": METHOD_VERSION,
            "validation_protocol": VALIDATION_PROTOCOL,
            "cost_definition": COST_DEFINITION,
            "gamma_operational": 0.99,
            "gamma_rehandle": 1.0,
            "reward_scale": 0.01,
            "resumable": False,
            "development_only": True,
            "no_v1_v2_v2_1_fallback": True,
        }
        mismatches = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
        if mismatches:
            raise ValueError(f"V2.2 runtime contract mismatch: {mismatches!r}")
        if contract.get("dense_objective_spec") != FROZEN_OBJECTIVE_SPEC.to_dict():
            raise ValueError("V2.2 dense-objective contract mismatch")
        if dict(contract.get("viability_search_config", {})) != FROZEN_VIABILITY_SEARCH_CONFIG:
            raise ValueError("V2.2 viability-search contract mismatch")
        hold = dict(contract.get("certified_hold", {}))
        hold_expected = {
            "frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "rule": CERTIFIED_HOLD_RULE_V2,
            "robust_stutter_contract": ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
            "hidden_schedule_contract": HIDDEN_SCHEDULE_CONTRACT_V2,
            "base_v1_defer_frontier_allowed": False,
            "observed_successor_required": True,
        }
        hold_mismatches = {key: (hold.get(key), value) for key, value in hold_expected.items() if hold.get(key) != value}
        if hold_mismatches:
            raise ValueError(f"V2.2 Hold contract mismatch: {hold_mismatches!r}")

        requested_device = str(contract.get("device", getattr(args, "device", "cpu")))
        if requested_device.lower() == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        environment = dict(contract["environment"])
        self.objective_spec = TimingObjectiveSpec.from_dict(contract["dense_objective_spec"])
        self.env = ObjectiveAuditSmallRoomsEnv(
            timing_objective=self.objective_spec,
            grid_rows=int(environment["grid_rows"]), grid_cols=int(environment["grid_cols"]),
            number_blocks=int(environment["number_blocks"]), choose_storage=False,
            arrival_rate=float(environment["arrival_rate"]), proc_mean=float(environment["proc_mean"]),
        )
        self.config = v2.ConstrainedV2Config.from_dict(contract["agent_config"])
        self.model_seed = int(contract["model_seed"])
        self.agent = ConstrainedV22HierarchyAgent(
            config=self.config, seed=self.model_seed, device=self.device, epsilon=0.0,
            dual_lambda=float(contract["dual"]["lambda_initial"]),
        )
        self.search_config = ViabilitySearchConfig(**FROZEN_VIABILITY_SEARCH_CONFIG)
        self.hold_rule = CertifiedHoldRuleV2(
            max_option_steps=self.config.max_hold_steps,
            max_idle_steps=self.config.max_idle_steps,
        )
        self.max_steps = int(contract["max_steps"])
        self._schedule_state = {}
        self._pending_policy_rng_seed: Optional[int] = None
        self._last_evaluation_agent: Optional[_FrozenV22EvaluationAgent] = None
        self._last_evaluation_before: Optional[dict] = None
        self._last_evaluation_selection_before: Optional[Counter] = None
        self._validation_batch_signature_before: Optional[tuple] = None
        self._core_contract = {
            "controller": CONTROLLER_ARCHITECTURE,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "backup_version": BACKUP_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "validation_protocol": VALIDATION_PROTOCOL,
            "training_policy": POLICY_REALIZATION,
            "backup_policy": POLICY_REALIZATION,
            "validation_policy": POLICY_REALIZATION,
            "deployment_policy": POLICY_REALIZATION,
            "same_policy_for_behavior_backup_validation_deployment": True,
            "map_policy_available": False,
            "policy_realization_decoupled_from_training_flag": True,
            "single_shared_lagrangian_policy_for_both_target_heads": True,
            "entropy_or_kl_added_to_raw_component_targets": False,
            "behavior_rng_scope": "per_rollout_action_sampling_only",
            "replay_rng_scope": "persistent_replay_sampling_only",
            "replay_rng_seed": REPLAY_RNG_SEED,
            "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "hold_rule": CERTIFIED_HOLD_RULE_V2,
            "cost_definition": COST_DEFINITION,
            "scalarized_reward_stored_in_replay": False,
            "operational_gamma": self.config.gamma_operational,
            "rehandle_gamma": self.config.gamma_rehandle,
            "operational_reward_scale": self.config.operational_reward_scale,
            "dual_update_authority": "trainer_complete_active_block_residual",
            "exact_safe_frontier_authoritative": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "agent_config": self.config.to_dict(),
            "viability_search_config": dict(FROZEN_VIABILITY_SEARCH_CONFIG),
            "fresh_certificate_cache_per_episode": True,
        }

    @property
    def core_contract(self) -> Mapping:
        return deepcopy(self._core_contract)

    @property
    def schedule_state(self) -> Mapping:
        return deepcopy(self._schedule_state)

    def set_schedule_state(self, state: Mapping) -> None:
        canonical = _validate_schedule_mapping(state)
        self.agent.set_schedule_state(canonical)
        self._schedule_state = canonical

    def set_dual_lambda(self, value: float) -> None:
        self.agent.set_dual_lambda(value)

    @property
    def dual_lambda(self) -> float:
        return float(self.agent.dual_lambda)

    @staticmethod
    def _module_signature(module: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in module.state_dict().items():
            value = tensor.detach().contiguous().cpu()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def _state_digest(cls, value) -> str:
        digest = hashlib.sha256()

        def update(item) -> None:
            if torch.is_tensor(item):
                tensor = item.detach().contiguous().cpu()
                digest.update(b"tensor")
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
            elif isinstance(item, Mapping):
                digest.update(b"mapping")
                for key in sorted(item, key=lambda key: repr(key)):
                    update(key)
                    update(item[key])
            elif isinstance(item, (tuple, list)):
                digest.update(type(item).__name__.encode())
                digest.update(str(len(item)).encode())
                for child in item:
                    update(child)
            elif isinstance(item, (str, bytes, int, float, bool, type(None))):
                digest.update(type(item).__name__.encode())
                digest.update(repr(item).encode())
            elif hasattr(item, "__dict__"):
                digest.update(type(item).__qualname__.encode())
                update(vars(item))
            else:
                digest.update(type(item).__qualname__.encode())
                digest.update(repr(item).encode())

        update(value)
        return digest.hexdigest()

    def _training_signature(self) -> tuple:
        state = self.agent.checkpoint_state(include_replay=False)
        replay_signature = (
            int(self.agent.replay.capacity),
            len(self.agent.replay.memory),
            tuple(
                (
                    id(item),
                    item.chosen.key,
                    float(item.operational_reward),
                    float(item.physical_rehandle_cost),
                    int(item.duration),
                    bool(item.done),
                    float(item.behavior_lambda),
                    tuple(record.key for record in item.next_candidates),
                )
                for item in self.agent.replay.memory
            ),
        )
        return (
            self._state_digest(state),
            replay_signature,
            bool(self.agent.Q_local.training),
            bool(self.agent.Q_target.training),
            tuple(parameter.requires_grad for parameter in self.agent.Q_local.parameters()),
            tuple(parameter.requires_grad for parameter in self.agent.Q_target.parameters()),
        )

    def _full_training_signature(self) -> tuple:
        """Expensive byte-level state signature, used only at batch boundaries."""

        state = self.agent.checkpoint_state(include_replay=True)
        return (
            self._state_digest(state),
            bool(self.agent.Q_local.training),
            bool(self.agent.Q_target.training),
            tuple(parameter.requires_grad for parameter in self.agent.Q_local.parameters()),
            tuple(parameter.requires_grad for parameter in self.agent.Q_target.parameters()),
        )

    def begin_validation_batch(self) -> Mapping:
        if self._validation_batch_signature_before is not None:
            raise RuntimeError("V2.2 validation batch is already active")
        if self._schedule_state.get("execution_context") != "validation_deployment":
            raise RuntimeError("V2.2 validation batch requires validation schedule")
        self._validation_batch_signature_before = self._full_training_signature()
        return {"validation_batch_started": True, "full_training_state_digest": self._validation_batch_signature_before[0]}

    def end_validation_batch(self) -> Mapping:
        before = self._validation_batch_signature_before
        if before is None:
            raise RuntimeError("V2.2 validation batch is not active")
        after = self._full_training_signature()
        self._validation_batch_signature_before = None
        return {
            "validation_batch_ended": True,
            "training_agent_unchanged": before == after,
            "full_training_state_digest_before": before[0],
            "full_training_state_digest_after": after[0],
        }

    def _evaluation_agent(self, *, seed: int) -> _FrozenV22EvaluationAgent:
        del seed
        if self._pending_policy_rng_seed is None:
            raise RuntimeError("V2.2 validation action RNG was not installed")
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            clone = _FrozenV22EvaluationAgent(
                config=self.config, seed=self.model_seed, device=self.device, epsilon=0.0,
                dual_lambda=self.agent.dual_lambda,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        clone.load_checkpoint_state(self.agent.checkpoint_state(include_replay=False), resumable=False)
        clone.set_schedule_state(self._schedule_state)
        clone.reset_behavior_rng(self._pending_policy_rng_seed)
        clone.Q_local.eval()
        clone.Q_target.eval()
        clone.Q_local.requires_grad_(False)
        clone.Q_target.requires_grad_(False)
        self._last_evaluation_agent = clone
        self._last_evaluation_before = clone.policy_diagnostic_state()
        self._last_evaluation_selection_before = Counter(clone.selection_sources)
        return clone

    @staticmethod
    def _diagnostic_delta(before: Mapping, after: Mapping) -> dict:
        count = int(after["count"]) - int(before["count"])
        def delta(field):
            return None if count <= 0 else (float(after[field]) - float(before[field])) / count
        return {
            "policy_diagnostic_decisions": count,
            "mean_outer_policy_entropy": delta("outer_entropy_sum"),
            "mean_outer_map_probability": delta("outer_map_probability_sum"),
            "mean_selected_within_policy_entropy": delta("selected_within_entropy_sum"),
            "mean_selected_within_map_probability": delta("selected_within_map_probability_sum"),
        }

    def run_episode(self, *, instance_seed: int, training: bool, max_steps: int,
                    policy_rng_index: Optional[int] = None,
                    policy_rng_seed: Optional[int] = None) -> Mapping:
        schedule = _validate_schedule_mapping(self._schedule_state)
        expected_training = schedule["execution_context"] == "training"
        if bool(training) != expected_training:
            raise RuntimeError("V2.2 run flag disagrees with execution context")
        if isinstance(policy_rng_seed, bool) or not isinstance(policy_rng_seed, int) or policy_rng_seed < 0:
            raise ValueError("V2.2 requires an explicit action RNG seed per rollout")
        if isinstance(policy_rng_index, bool) or not isinstance(policy_rng_index, int) or policy_rng_index < 0:
            raise ValueError("V2.2 requires an explicit action RNG index per rollout")
        if 622_000_000 <= policy_rng_seed < 622_000_120:
            raise ValueError("V2.2 refuses the unopened final action-RNG panel")
        if 86_000 <= int(instance_seed) < 86_030:
            raise ValueError("V2.2 refuses the unopened final EpisodeInstance panel")
        if 83_000 <= int(instance_seed) < 84_000:
            raise ValueError("V2.2 refuses the protected prior prospective 83xxx panel")

        canonical_instance = self.env.sample_episode_instance(int(instance_seed))
        canonical_text = canonical_instance.to_json()
        instance_sha = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        if training:
            self.agent.reset_behavior_rng(int(policy_rng_seed))
            before = self.agent.policy_diagnostic_state()
            selection_before = Counter(self.agent.selection_sources)
            training_signature = None
        else:
            if self._validation_batch_signature_before is None:
                raise RuntimeError("V2.2 validation rollout requires an authenticated batch")
            self._pending_policy_rng_seed = int(policy_rng_seed)
            self._last_evaluation_agent = None
            self._last_evaluation_before = None
            self._last_evaluation_selection_before = None
            training_signature = self._training_signature()
            before = None
        result = dict(v2.ConstrainedV2SmokeRuntime.run_episode(
            self, instance_seed=int(instance_seed), training=bool(training), max_steps=int(max_steps)
        ))
        episode_agent = self.agent if training else self._last_evaluation_agent
        if episode_agent is None:
            raise RuntimeError("V2.2 validation clone was not retained")
        if result.get("episode_instance_id") != canonical_instance.instance_id or result.get("schedule_id") != canonical_instance.schedule_id:
            raise RuntimeError("V2.2 runtime EpisodeInstance identity mismatch")
        if not training:
            if self._last_evaluation_before is None:
                raise RuntimeError("V2.2 validation diagnostic baseline missing")
            before = self._last_evaluation_before
            if self._last_evaluation_selection_before is None:
                raise RuntimeError("V2.2 validation selection baseline missing")
            selection_before = self._last_evaluation_selection_before
        assert before is not None
        result.update(self._diagnostic_delta(before, episode_agent.policy_diagnostic_state()))
        selection_after = Counter(episode_agent.selection_sources)
        selection_counts = {
            name: int(selection_after[name] - selection_before[name])
            for name in set(selection_after) | set(selection_before)
            if selection_after[name] - selection_before[name]
        }
        allowed_sources = {
            "induced_nested_stochastic_lagrangian_policy",
            "exact_recovery_witness_guard",
        }
        map_used = any("map" in str(name).lower() and count for name, count in selection_counts.items())
        stochastic_only = bool(
            not map_used
            and set(selection_counts).issubset(allowed_sources)
            and sum(selection_counts.values()) == int(result.get("macro_decisions", -1))
        )
        result.update({
            "method_version": METHOD_VERSION,
            "split": "training" if training else "development_validation",
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "policy_mode": POLICY_REALIZATION,
            "execution_context": schedule["execution_context"],
            "within_group_temperatures": tuple(schedule["within_group_temperatures"]),
            "group_temperature": float(schedule["group_temperature"]),
            "dual_lambda": float(episode_agent.dual_lambda),
            "policy_rng_index": int(policy_rng_index),
            "policy_rng_seed": int(policy_rng_seed),
            "policy_rng_scope": "action_sampling_only",
            "behavior_policy_rng_reset": True,
            "replay_rng_seed": REPLAY_RNG_SEED,
            "episode_instance_sha256": instance_sha,
            "map_selection_used": bool(map_used),
            "selection_source_counts": selection_counts,
            "stochastic_selection_only": stochastic_only,
            "option_evaluation_mode": False if training else True,
            "evaluation_learning": None if training else False,
            "fresh_evaluation_clone": None if training else True,
            "training_agent_unchanged": (
                None if training else training_signature == self._training_signature()
            ),
        })
        self._pending_policy_rng_seed = None
        return result

    def checkpoint_state(self, *, include_replay: bool) -> Mapping:
        return self.agent.checkpoint(include_replay=bool(include_replay))


def build_v2_2_development_runtime(*, args, contract: Mapping) -> ConstrainedV22DevelopmentRuntime:
    return ConstrainedV22DevelopmentRuntime(args=args, contract=contract)


__all__ = [
    "ConstrainedV22DevelopmentRuntime", "ConstrainedV22HierarchyAgent",
    "V2_2_DIAGNOSTIC_VERSION", "V2_2_POLICY_VERSION",
    "build_v2_2_development_runtime",
]
