"""Final-panel adapter for the frozen VCG V2.3 evaluation runtime.

The development runtime intentionally refuses the reserved 86xxx instance and
622m action-RNG namespaces.  This adapter changes only that lifecycle guard:
it requires a caller-supplied, already serialized/reloaded EpisodeInstance and
executes the same frozen V2.3 agent, exact frontier, option runtime, and
stochastic policy.  It never samples an EpisodeInstance itself.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Mapping

import torch

import viability_graph_constrained_v2 as v2
import viability_graph_constrained_v2_3 as v23_core
from viability_graph_constrained_v2_3 import (
    ConstrainedV23DevelopmentRuntime,
)


FINAL_INSTANCE_SEEDS = tuple(range(86_000, 86_030))
FINAL_POLICY_RNG_SEEDS = tuple(range(622_000_000, 622_000_120))


class _BoundInstanceEnvironment:
    """Delegate an environment while binding its sampler to one frozen object."""

    def __init__(self, environment, *, instance, instance_seed: int) -> None:
        self._environment = environment
        self._instance = instance
        self._instance_seed = int(instance_seed)

    def sample_episode_instance(self, seed: int):
        if int(seed) != self._instance_seed:
            raise RuntimeError("bound final EpisodeInstance seed mismatch")
        # The object was freshly reloaded by the orchestrator for this one row.
        # The base runtime passes it directly to env.reset and does not mutate
        # the immutable EpisodeInstance value.
        return self._instance

    def __getattr__(self, name):
        return getattr(self._environment, name)


class FinalV23Runtime(ConstrainedV23DevelopmentRuntime):
    """VCG V2.3 validation runtime accepting only the frozen final namespaces."""

    def run_final_episode(
        self,
        *,
        episode_instance,
        instance_seed: int,
        max_steps: int,
        policy_rng_index: int,
        policy_rng_seed: int,
    ) -> Mapping:
        schedule = v23_core._validate_schedule_mapping(self._schedule_state)
        if schedule["execution_context"] != "validation_deployment":
            raise RuntimeError("final V2.3 rollout requires the validation schedule")
        if type(instance_seed) is not int or instance_seed not in FINAL_INSTANCE_SEEDS:
            raise ValueError("final V2.3 instance seed is outside 86000..86029")
        if type(policy_rng_index) is not int or policy_rng_index not in range(4):
            raise ValueError("final V2.3 policy RNG index must be 0..3")
        instance_index = instance_seed - FINAL_INSTANCE_SEEDS[0]
        expected_rng = 622_000_000 + 4 * instance_index + policy_rng_index
        if type(policy_rng_seed) is not int or policy_rng_seed != expected_rng:
            raise ValueError("final V2.3 action RNG seed violates the frozen grid")
        if type(max_steps) is not int or max_steps != 2000:
            raise ValueError("final V2.3 max_steps must be the frozen integer 2000")
        if policy_rng_seed not in FINAL_POLICY_RNG_SEEDS:
            raise ValueError("final V2.3 action RNG seed is outside the final namespace")
        if getattr(episode_instance, "seed", None) != instance_seed:
            raise ValueError("final V2.3 EpisodeInstance seed mismatch")
        episode_instance.validate_for(self.env)
        canonical_text = episode_instance.to_json()
        instance_sha = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()

        if self._validation_batch_signature_before is None:
            raise RuntimeError("final V2.3 rollout requires an authenticated batch")
        self._pending_policy_rng_seed = policy_rng_seed
        self._last_evaluation_agent = None
        self._last_evaluation_before = None
        self._last_evaluation_selection_before = None
        training_signature = self._training_signature()
        original_environment = self.env
        self.env = _BoundInstanceEnvironment(
            original_environment,
            instance=episode_instance,
            instance_seed=instance_seed,
        )
        try:
            result = dict(
                v2.ConstrainedV2SmokeRuntime.run_episode(
                    self,
                    instance_seed=instance_seed,
                    training=False,
                    max_steps=max_steps,
                )
            )
        finally:
            self.env = original_environment
            self._pending_policy_rng_seed = None

        episode_agent = self._last_evaluation_agent
        if episode_agent is None:
            raise RuntimeError("final V2.3 evaluation clone was not retained")
        if (
            result.get("episode_instance_id") != episode_instance.instance_id
            or result.get("schedule_id") != episode_instance.schedule_id
        ):
            raise RuntimeError("final V2.3 runtime EpisodeInstance identity mismatch")
        if self._last_evaluation_before is None:
            raise RuntimeError("final V2.3 validation diagnostic baseline missing")
        if self._last_evaluation_selection_before is None:
            raise RuntimeError("final V2.3 validation selection baseline missing")

        result.update(
            self._diagnostic_delta(
                self._last_evaluation_before,
                episode_agent.policy_diagnostic_state(),
            )
        )
        selection_before = self._last_evaluation_selection_before
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
        map_used = any(
            "map" in str(name).lower() and count
            for name, count in selection_counts.items()
        )
        stochastic_only = bool(
            not map_used
            and set(selection_counts).issubset(allowed_sources)
            and sum(selection_counts.values())
            == int(result.get("macro_decisions", -1))
        )
        result.update(
            {
                "method_version": v23_core.METHOD_VERSION,
                "split": "final_confirmatory_evaluation",
                "policy_schedule_protocol": v23_core.POLICY_SCHEDULE_PROTOCOL,
                "policy_mode": v23_core.POLICY_REALIZATION,
                "execution_context": schedule["execution_context"],
                "within_group_temperatures": tuple(
                    schedule["within_group_temperatures"]
                ),
                "group_temperature": float(schedule["group_temperature"]),
                "dual_lambda": float(episode_agent.dual_lambda),
                "policy_rng_index": policy_rng_index,
                "policy_rng_seed": policy_rng_seed,
                "policy_rng_scope": "action_sampling_only",
                "behavior_policy_rng_reset": True,
                "replay_rng_seed": v23_core.REPLAY_RNG_SEED,
                "episode_instance_sha256": instance_sha,
                "map_selection_used": bool(map_used),
                "selection_source_counts": selection_counts,
                "stochastic_selection_only": stochastic_only,
                "option_evaluation_mode": True,
                "evaluation_learning": False,
                "fresh_evaluation_clone": True,
                "training_agent_unchanged": (
                    training_signature == self._training_signature()
                ),
                "external_serialized_episode_instance": True,
                "final_namespace_prospectively_opened": True,
            }
        )
        if result["training_agent_unchanged"] is not True:
            raise RuntimeError("final V2.3 rollout mutated the frozen training agent")
        if result["stochastic_selection_only"] is not True:
            raise RuntimeError("final V2.3 rollout did not use only its frozen policy")
        if result["map_selection_used"] is not False:
            raise RuntimeError("final V2.3 rollout used forbidden MAP extraction")
        return result


__all__ = ["FINAL_INSTANCE_SEEDS", "FINAL_POLICY_RNG_SEEDS", "FinalV23Runtime"]
