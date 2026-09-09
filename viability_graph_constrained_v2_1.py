"""Isolated two-timescale controller/runtime for constrained VCG V2.1.

V2.1 reuses the audited V2 graph encoder, raw vector replay, exact viability
frontier, and option execution.  Its policy state is new and fail-closed: a
trainer-authenticated block schedule supplies the temperatures used by both
behavior selection and every Bellman continuation.  No V2 checkpoint or
runtime contract is accepted as a V2.1 artifact.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Real
from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from PSLAP.viability_candidates_hold_v2 import (
    CERTIFIED_HOLD_INTERFACE_V2,
    CERTIFIED_HOLD_RULE_V2,
    HIDDEN_SCHEDULE_CONTRACT_V2,
    ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
    CertifiedHoldRuleV2,
)
from PSLAP.viability_filter import ViabilitySearchConfig
import viability_graph_constrained_v2 as v2
from train_vcg_constrained_v2 import COST_DEFINITION, FROZEN_OBJECTIVE_SPEC
from train_vcg_constrained_v2_1 import (
    BACKUP_VERSION,
    CHECKPOINT_FAMILY,
    CHECKPOINT_SCHEMA_VERSION,
    CONTROLLER_ARCHITECTURE,
    FROZEN_VIABILITY_SEARCH_CONFIG,
    METHOD_VERSION,
    POLICY_SCHEDULE_PROTOCOL,
    schedule_for_episode,
)


V2_1_POLICY_VERSION = (
    "four_group_block_annealed_shared_lagrangian_policy_v2_1"
)
V2_1_DIAGNOSTIC_VERSION = (
    "outer_and_selected_within_entropy_and_map_probability_v1"
)
V2_1_DIAGNOSTIC_DISTRIBUTION = (
    "regularized_operator_distribution_not_executed_map_randomness"
)


def _validate_schedule_mapping(state: Mapping) -> dict:
    """Return the canonical trainer schedule or fail on any mismatch."""

    if not isinstance(state, Mapping):
        raise TypeError("V2.1 schedule state must be a mapping")
    received = dict(state)
    episode = received.get("episode_number")
    if isinstance(episode, bool) or not isinstance(episode, int):
        raise ValueError("V2.1 schedule requires an integer episode_number")
    policy_mode = received.get("policy_mode")
    if policy_mode == "regularized_sample":
        expected = schedule_for_episode(episode, validation=False).to_dict()
    elif policy_mode == "map":
        expected = schedule_for_episode(episode, validation=True).to_dict()
    else:
        raise ValueError("V2.1 schedule policy_mode is invalid")
    if received != expected:
        differences = {
            key: (received.get(key), expected.get(key))
            for key in sorted(set(received) | set(expected))
            if received.get(key) != expected.get(key)
        }
        raise ValueError(f"V2.1 schedule does not match frozen clock: {differences!r}")
    return deepcopy(expected)


class ConstrainedV21HierarchyAgent(v2.ConstrainedV2HierarchyAgent):
    """V2 vector critic with trainer-owned blockwise policy temperatures."""

    def __init__(self, *args, **kwargs) -> None:
        self._v2_1_schedule_state: Optional[dict] = None
        self._policy_diagnostic_count = 0
        self._outer_entropy_sum = 0.0
        self._outer_map_probability_sum = 0.0
        self._within_entropy_sum = 0.0
        self._within_map_probability_sum = 0.0
        self._last_policy_diagnostic: Optional[dict] = None
        super().__init__(*args, **kwargs)

    @property
    def schedule_state(self) -> Mapping:
        return (
            {}
            if self._v2_1_schedule_state is None
            else deepcopy(self._v2_1_schedule_state)
        )

    def set_schedule_state(self, state: Mapping) -> None:
        self._v2_1_schedule_state = _validate_schedule_mapping(state)

    def _require_schedule(self) -> dict:
        if self._v2_1_schedule_state is None:
            raise RuntimeError("V2.1 policy schedule has not been installed")
        return self._v2_1_schedule_state

    @property
    def within_group_temperature_values(self) -> tuple[float, float, float, float]:
        schedule = self._require_schedule()
        values = tuple(float(value) for value in schedule["within_group_temperatures"])
        if len(values) != 4:
            raise RuntimeError("V2.1 schedule lost a within-group temperature")
        return values

    @property
    def group_temperature(self) -> float:
        value = float(self._require_schedule()["group_temperature"])
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("V2.1 group temperature is invalid")
        return value

    @staticmethod
    def _distribution_diagnostic(logits: torch.Tensor) -> tuple[float, float]:
        probabilities = F.softmax(logits, dim=0)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1.0e-12))
        )
        return float(entropy.item()), float(probabilities.max().item())

    def _record_policy_diagnostic(self, decision) -> None:
        outer_values = torch.as_tensor(
            [value for _, value in decision.mode_values], dtype=torch.float64
        )
        outer_entropy, outer_map_probability = self._distribution_diagnostic(
            outer_values / self.group_temperature
        )
        selected_group = int(decision.record.group_id)
        selected_indices = [
            index
            for index, group_id in enumerate(decision.prepared_snapshot.mode_ids)
            if int(group_id) == selected_group
        ]
        within_values = torch.as_tensor(
            [decision.q_values[index] for index in selected_indices],
            dtype=torch.float64,
        )
        within_entropy, within_map_probability = self._distribution_diagnostic(
            within_values / self.within_group_temperature_values[selected_group]
        )
        diagnostic = {
            "protocol": V2_1_DIAGNOSTIC_VERSION,
            "distribution_semantics": V2_1_DIAGNOSTIC_DISTRIBUTION,
            "episode_number": self._require_schedule()["episode_number"],
            "block_number": self._require_schedule()["block_number"],
            "policy_mode": self._require_schedule()["policy_mode"],
            "outer_entropy": outer_entropy,
            "outer_map_probability": outer_map_probability,
            "selected_within_entropy": within_entropy,
            "selected_within_map_probability": within_map_probability,
        }
        self._policy_diagnostic_count += 1
        self._outer_entropy_sum += outer_entropy
        self._outer_map_probability_sum += outer_map_probability
        self._within_entropy_sum += within_entropy
        self._within_map_probability_sum += within_map_probability
        self._last_policy_diagnostic = diagnostic
        if self.decision_log:
            self.decision_log[-1]["v2_1_policy_diagnostic"] = dict(diagnostic)

    def select(self, snapshot, *, training: bool = True, epsilon=None):
        schedule = self._require_schedule()
        expected_training = schedule["policy_mode"] == "regularized_sample"
        if bool(training) != expected_training:
            raise RuntimeError(
                "V2.1 select training flag disagrees with authenticated policy_mode"
            )
        decision = super().select(
            snapshot, training=training, epsilon=epsilon
        )
        self._record_policy_diagnostic(decision)
        return decision

    def policy_diagnostic_state(self) -> dict:
        count = int(self._policy_diagnostic_count)

        def mean(total: float) -> Optional[float]:
            return None if count == 0 else float(total / count)

        return {
            "protocol": V2_1_DIAGNOSTIC_VERSION,
            "distribution_semantics": V2_1_DIAGNOSTIC_DISTRIBUTION,
            "count": count,
            "outer_entropy_sum": float(self._outer_entropy_sum),
            "outer_map_probability_sum": float(self._outer_map_probability_sum),
            "selected_within_entropy_sum": float(self._within_entropy_sum),
            "selected_within_map_probability_sum": float(
                self._within_map_probability_sum
            ),
            "mean_outer_entropy": mean(self._outer_entropy_sum),
            "mean_outer_map_probability": mean(
                self._outer_map_probability_sum
            ),
            "mean_selected_within_entropy": mean(self._within_entropy_sum),
            "mean_selected_within_map_probability": mean(
                self._within_map_probability_sum
            ),
            "last": (
                None
                if self._last_policy_diagnostic is None
                else dict(self._last_policy_diagnostic)
            ),
        }

    def checkpoint_metadata(self) -> dict:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_family": CHECKPOINT_FAMILY,
                "controller_architecture": CONTROLLER_ARCHITECTURE,
                "backup_version": BACKUP_VERSION,
                "policy_version": V2_1_POLICY_VERSION,
                "method_version": METHOD_VERSION,
                "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
                "schedule_state": dict(self._require_schedule()),
                "within_group_temperatures": self.within_group_temperature_values,
                "group_temperature": self.group_temperature,
                "same_temperatures_for_behavior_and_backup": True,
                "training_policy": (
                    "block_annealed_nested_regularized_lagrangian_sample"
                ),
                "deployment_policy": "nested_lagrangian_map",
                "policy_diagnostic_version": V2_1_DIAGNOSTIC_VERSION,
                "policy_diagnostic_distribution": (
                    V2_1_DIAGNOSTIC_DISTRIBUTION
                ),
            }
        )
        return metadata

    def checkpoint_state(self, *, include_replay: bool = True) -> dict:
        state = super().checkpoint_state(include_replay=include_replay)
        state["schedule_state"] = dict(self._require_schedule())
        state["policy_diagnostics"] = self.policy_diagnostic_state()
        return state

    def load_checkpoint_state(self, state: Mapping, *, resumable: bool) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("V2.1 agent state must be a mapping")
        if "schedule_state" not in state:
            raise ValueError("V2.1 agent state is missing schedule_state")
        schedule = _validate_schedule_mapping(state["schedule_state"])
        super().load_checkpoint_state(state, resumable=resumable)
        self._v2_1_schedule_state = schedule
        diagnostics = state.get("policy_diagnostics", {})
        if diagnostics:
            if diagnostics.get("protocol") != V2_1_DIAGNOSTIC_VERSION:
                raise ValueError("V2.1 policy diagnostic version mismatch")
            self._policy_diagnostic_count = int(diagnostics.get("count", 0))
            self._outer_entropy_sum = float(
                diagnostics.get("outer_entropy_sum", 0.0)
            )
            self._outer_map_probability_sum = float(
                diagnostics.get("outer_map_probability_sum", 0.0)
            )
            self._within_entropy_sum = float(
                diagnostics.get("selected_within_entropy_sum", 0.0)
            )
            self._within_map_probability_sum = float(
                diagnostics.get("selected_within_map_probability_sum", 0.0)
            )
            last = diagnostics.get("last")
            self._last_policy_diagnostic = None if last is None else dict(last)

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "ConstrainedV21HierarchyAgent":
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "policy_version": V2_1_POLICY_VERSION,
            "method_version": METHOD_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "same_temperatures_for_behavior_and_backup": True,
            "candidate_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "cost_definition": COST_DEFINITION,
            "scalarized_reward_stored_in_replay": False,
            "baseline_teacher": False,
            "single_shared_lagrangian_policy_for_both_target_heads": True,
            "entropy_or_kl_added_to_raw_component_targets": False,
            "exact_safe_mask_authoritative": True,
            "training_policy": (
                "block_annealed_nested_regularized_lagrangian_sample"
            ),
            "deployment_policy": "nested_lagrangian_map",
            "policy_diagnostic_version": V2_1_DIAGNOSTIC_VERSION,
            "policy_diagnostic_distribution": V2_1_DIAGNOSTIC_DISTRIBUTION,
        }
        mismatches = {
            key: (payload.get(key), required)
            for key, required in expected.items()
            if payload.get(key) != required
        }
        if mismatches:
            raise ValueError(
                f"incompatible constrained V2.1 checkpoint: {mismatches!r}"
            )
        agent = cls(
            config=v2.ConstrainedV2Config.from_dict(payload["config"]),
            seed=int(seed),
            device=device,
            epsilon=float(payload["agent_state"].get("epsilon", 0.0)),
            dual_lambda=float(payload["agent_state"]["dual_lambda"]),
        )
        agent.load_checkpoint_state(
            payload["agent_state"], resumable=bool(resumable)
        )
        if dict(payload.get("schedule_state", {})) != dict(agent.schedule_state):
            raise ValueError("V2.1 checkpoint metadata/state schedule mismatch")
        return agent


class ConstrainedV21DevelopmentRuntime(v2.ConstrainedV2SmokeRuntime):
    """Real V2.1 environment adapter with an authenticated policy clock."""

    def __init__(self, *, args, contract: Mapping) -> None:
        from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec

        if not isinstance(contract, Mapping):
            raise TypeError("V2.1 development contract must be a mapping")
        expected = {
            "checkpoint_family": CHECKPOINT_FAMILY,
            "controller": CONTROLLER_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "method_version": METHOD_VERSION,
            "cost_definition": COST_DEFINITION,
            "gamma_operational": 0.99,
            "gamma_rehandle": 1.0,
            "reward_scale": 0.01,
            "resumable": False,
            "development_only": True,
            "no_v1_or_v2_fallback": True,
        }
        mismatches = {
            key: (contract.get(key), required)
            for key, required in expected.items()
            if contract.get(key) != required
        }
        if mismatches:
            raise ValueError(f"V2.1 runtime contract mismatch: {mismatches!r}")
        if contract.get("dense_objective_spec") != FROZEN_OBJECTIVE_SPEC.to_dict():
            raise ValueError("V2.1 dense-objective contract mismatch")
        if dict(contract.get("viability_search_config", {})) != dict(
            FROZEN_VIABILITY_SEARCH_CONFIG
        ):
            raise ValueError("V2.1 viability-search contract mismatch")
        if contract.get("fresh_certificate_cache_per_episode") is not True:
            raise ValueError("V2.1 requires a fresh certificate cache per episode")
        hold_contract = contract.get("certified_hold", {})
        hold_expected = {
            "frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "rule": CERTIFIED_HOLD_RULE_V2,
            "robust_stutter_contract": ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
            "hidden_schedule_contract": HIDDEN_SCHEDULE_CONTRACT_V2,
            "base_v1_defer_frontier_allowed": False,
            "observed_successor_required": True,
        }
        hold_mismatches = {
            key: (hold_contract.get(key), required)
            for key, required in hold_expected.items()
            if hold_contract.get(key) != required
        }
        if hold_mismatches:
            raise ValueError(
                f"V2.1 certified-Hold contract mismatch: {hold_mismatches!r}"
            )

        requested_device = str(contract.get("device", getattr(args, "device", "cpu")))
        if requested_device.lower() == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        environment = dict(contract["environment"])
        self.objective_spec = TimingObjectiveSpec.from_dict(
            contract["dense_objective_spec"]
        )
        self.env = ObjectiveAuditSmallRoomsEnv(
            timing_objective=self.objective_spec,
            grid_rows=int(environment["grid_rows"]),
            grid_cols=int(environment["grid_cols"]),
            number_blocks=int(environment["number_blocks"]),
            choose_storage=False,
            arrival_rate=float(environment["arrival_rate"]),
            proc_mean=float(environment["proc_mean"]),
        )
        self.config = v2.ConstrainedV2Config.from_dict(contract["agent_config"])
        if self.config.to_dict() != dict(contract["agent_config"]):
            raise ValueError("V2.1 agent configuration did not round-trip exactly")
        self.agent = ConstrainedV21HierarchyAgent(
            config=self.config,
            seed=int(contract["model_seed"]),
            device=self.device,
            epsilon=0.0,
            dual_lambda=float(contract["dual"]["lambda_initial"]),
        )
        self.search_config = ViabilitySearchConfig(
            **dict(FROZEN_VIABILITY_SEARCH_CONFIG)
        )
        self.hold_rule = CertifiedHoldRuleV2(
            max_option_steps=self.config.max_hold_steps,
            max_idle_steps=self.config.max_idle_steps,
        )
        self.max_steps = int(contract["max_steps"])
        self._schedule_state: dict = {}
        self._last_evaluation_agent: Optional[ConstrainedV21HierarchyAgent] = None
        self._last_evaluation_diagnostic_before: Optional[dict] = None
        self._core_contract = {
            "controller": CONTROLLER_ARCHITECTURE,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "backup_version": BACKUP_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "hold_rule": CERTIFIED_HOLD_RULE_V2,
            "robust_stutter_contract": ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
            "hidden_schedule_contract": HIDDEN_SCHEDULE_CONTRACT_V2,
            "base_v1_defer_frontier_allowed": False,
            "cost_definition": COST_DEFINITION,
            "scalarized_reward_stored_in_replay": False,
            "same_temperatures_for_behavior_and_backup": True,
            "training_policy": (
                "block_annealed_nested_regularized_lagrangian_sample"
            ),
            "deployment_policy": "nested_lagrangian_map",
            "mixed_discount_policy_merit": (
                "gamma_operational_pow_elapsed_times_qop_minus_lambda_qphys"
            ),
            "operational_gamma": self.config.gamma_operational,
            "rehandle_gamma": self.config.gamma_rehandle,
            "operational_reward_scale": self.config.operational_reward_scale,
            "dual_update_authority": "trainer_complete_active_block_residual",
            "exact_safe_frontier_authoritative": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "agent_config": self.config.to_dict(),
            "policy_diagnostic_version": V2_1_DIAGNOSTIC_VERSION,
            "policy_diagnostic_distribution": V2_1_DIAGNOSTIC_DISTRIBUTION,
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

    def _evaluation_agent(self, *, seed: int) -> ConstrainedV21HierarchyAgent:
        schedule = _validate_schedule_mapping(self._schedule_state)
        if schedule["policy_mode"] != "map":
            raise RuntimeError("V2.1 evaluation requires an authenticated MAP schedule")
        evaluation = ConstrainedV21HierarchyAgent(
            config=self.config,
            seed=int(seed),
            device=self.device,
            epsilon=0.0,
            dual_lambda=self.agent.dual_lambda,
        )
        evaluation.load_checkpoint_state(
            self.agent.checkpoint_state(include_replay=False), resumable=False
        )
        evaluation.set_schedule_state(schedule)
        evaluation.Q_local.eval()
        self._last_evaluation_agent = evaluation
        self._last_evaluation_diagnostic_before = (
            evaluation.policy_diagnostic_state()
        )
        return evaluation

    @staticmethod
    def _diagnostic_delta(before: Mapping, after: Mapping) -> dict:
        count = int(after["count"]) - int(before["count"])

        def delta_mean(key: str) -> Optional[float]:
            if count <= 0:
                return None
            return float((float(after[key]) - float(before[key])) / count)

        return {
            "policy_diagnostic_version": V2_1_DIAGNOSTIC_VERSION,
            "policy_diagnostic_distribution": V2_1_DIAGNOSTIC_DISTRIBUTION,
            "policy_diagnostic_decisions": count,
            "mean_outer_policy_entropy": delta_mean("outer_entropy_sum"),
            "mean_outer_map_probability": delta_mean(
                "outer_map_probability_sum"
            ),
            "mean_selected_within_policy_entropy": delta_mean(
                "selected_within_entropy_sum"
            ),
            "mean_selected_within_map_probability": delta_mean(
                "selected_within_map_probability_sum"
            ),
        }

    @staticmethod
    def _module_signature(module: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in module.state_dict().items():
            value = tensor.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _training_agent_isolation_signature(self) -> tuple:
        """Exact-enough immutable signature of all requested training state."""

        replay = tuple(
            (
                id(item),
                item.chosen.key,
                float(item.operational_reward),
                float(item.physical_rehandle_cost),
                int(item.duration),
                tuple(record.key for record in item.next_candidates),
                bool(item.done),
                float(item.behavior_lambda),
            )
            for item in self.agent.replay.memory
        )
        counters = (
            int(self.agent.transition_count),
            int(self.agent.decision_count),
            int(self.agent.gradient_steps),
            int(self.agent.target_updates),
            tuple(sorted(self.agent.group_decisions.items())),
            tuple(sorted(self.agent.action_decisions.items())),
            tuple(sorted(self.agent.selection_sources.items())),
            int(self.agent.exact_rejections_seen),
            int(self.agent.interface_rejections),
            int(self.agent.safe_candidates_scored),
            self.agent.policy_diagnostic_state()["count"],
        )
        return (
            self._module_signature(self.agent.Q_local),
            self._module_signature(self.agent.Q_target),
            replay,
            counters,
            repr(self.agent.rng.getstate()),
            repr(self.agent.recovery_witness_guard.state_dict()),
            bool(self.agent.Q_local.training),
            bool(self.agent.Q_target.training),
            tuple(sorted(dict(self.agent.schedule_state).items())),
        )

    def run_episode(
        self, *, instance_seed: int, training: bool, max_steps: int
    ) -> Mapping:
        schedule = _validate_schedule_mapping(self._schedule_state)
        expected_training = schedule["policy_mode"] == "regularized_sample"
        if bool(training) != expected_training:
            raise RuntimeError(
                "V2.1 run training flag disagrees with authenticated policy_mode"
            )
        if training:
            before = self.agent.policy_diagnostic_state()
            training_signature_before = None
        else:
            before = {
                "count": 0,
                "outer_entropy_sum": 0.0,
                "outer_map_probability_sum": 0.0,
                "selected_within_entropy_sum": 0.0,
                "selected_within_map_probability_sum": 0.0,
            }
            self._last_evaluation_agent = None
            self._last_evaluation_diagnostic_before = None
            training_signature_before = self._training_agent_isolation_signature()
        result = dict(
            super().run_episode(
                instance_seed=int(instance_seed),
                training=bool(training),
                max_steps=int(max_steps),
            )
        )
        episode_agent = (
            self.agent if training else self._last_evaluation_agent
        )
        if episode_agent is None:
            raise RuntimeError("V2.1 evaluation clone was not retained")
        training_agent_unchanged = (
            None
            if training
            else training_signature_before
            == self._training_agent_isolation_signature()
        )
        if not training:
            if self._last_evaluation_diagnostic_before is None:
                raise RuntimeError("V2.1 evaluation diagnostic baseline was not retained")
            before = self._last_evaluation_diagnostic_before
        after = episode_agent.policy_diagnostic_state()
        result.update(self._diagnostic_delta(before, after))
        result.update(
            {
                "method_version": METHOD_VERSION,
                "split": (
                    "training" if training else "development_validation"
                ),
                "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
                "schedule_phase": schedule["phase"],
                "policy_mode": schedule["policy_mode"],
                "within_group_temperatures": tuple(
                    schedule["within_group_temperatures"]
                ),
                "group_temperature": float(schedule["group_temperature"]),
                "dual_lambda": float(episode_agent.dual_lambda),
                "same_temperatures_for_behavior_and_backup": True,
                "policy_diagnostic_distribution": (
                    V2_1_DIAGNOSTIC_DISTRIBUTION
                ),
                "evaluation_learning": False if not training else None,
                "training_agent_unchanged": training_agent_unchanged,
            }
        )
        return result

    def checkpoint_state(self, *, include_replay: bool) -> Mapping:
        return self.agent.checkpoint(include_replay=bool(include_replay))


def build_v2_1_development_runtime(
    *, args, contract: Mapping
) -> ConstrainedV21DevelopmentRuntime:
    """Build only the authenticated V2.1 runtime; no V2 fallback exists."""

    return ConstrainedV21DevelopmentRuntime(args=args, contract=contract)


__all__ = [
    "ConstrainedV21DevelopmentRuntime",
    "ConstrainedV21HierarchyAgent",
    "V2_1_DIAGNOSTIC_VERSION",
    "V2_1_DIAGNOSTIC_DISTRIBUTION",
    "V2_1_POLICY_VERSION",
    "build_v2_1_development_runtime",
]
