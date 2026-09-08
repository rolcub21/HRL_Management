#!/usr/bin/env python3
"""Paired full-frontier benchmark for learned viability-check ordering.

The benchmark compares two evaluation-only realizations of one frozen
viability-constrained graph controller:

``exact_full``
    Certify uncached counterfactual states in their canonical order.

``critic_ordered_full``
    Use a frozen, calibrated viability ensemble to permute uncached checks,
    then run the same exact verifier on *every* state and restore the canonical
    candidate order before task-Q selection.

The critic never screens a state and never supplies a safety certificate.
Consequently the two arms are required to expose identical complete SAFE
frontiers, select identical actions, follow identical macro trajectories, and
perform the same number of exact cache misses.  Timing is descriptive: the
ordered arm adds critic inference and cannot reduce total verifier calls under
this complete-frontier contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Mapping, Optional, Sequence

import torch

from contention_metrics import contention_metric_record
from example.helper.timing_metrics import summarize_delivery_timing
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.viability import RecoveryAction, RecoveryState, ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    EXACT_ONLY_RECOVERY_CERTIFICATION,
    ViabilityActionType,
    ViabilityCandidateSnapshot,
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import (
    ViabilitySearchConfig,
    online_fixed_obstacles,
)
from PSLAP.viability_prioritizer import ViabilityCriticPrioritizer
from train_viability_graph_smdp import (
    SEALED_STRESS_V1_HOLDOUT_SEEDS,
    execute_certified_macro,
    resolve_device,
    seed_everything,
)
from viability_graph_hierarchy import ViabilityGraphHierarchyAgent
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EpisodicViabilityGraphHierarchyAgent,
)


PROTOCOL = "paired_exact_full_vs_critic_ordered_full_v1"
EXACT_FULL = "exact_full"
CRITIC_ORDERED_FULL = "critic_ordered_full"
ARMS = (EXACT_FULL, CRITIC_ORDERED_FULL)
RESULT_NAME = "benchmark.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _canonical_digest(value) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _load_controller_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("controller checkpoint must contain a dictionary")
    payload = dict(payload)

    # Proper-training v1 stores its immutable execution contract as one nested
    # object.  Earlier benchmark artifacts stored the same fields as top-level
    # aliases.  Normalize in memory so the authenticated checkpoint bytes do
    # not need to be rewritten, and fail if two representations disagree.
    resume_contract = payload.get("resume_contract")
    if isinstance(resume_contract, Mapping):
        aliases = {
            "environment": "environment",
            "viability_search": "search_config",
            "liveness_rule": "liveness_rule",
        }
        for target, source in aliases.items():
            nested = resume_contract.get(source)
            current = payload.get(target)
            if current is None and isinstance(nested, Mapping):
                payload[target] = dict(nested)
            elif (
                isinstance(current, Mapping)
                and isinstance(nested, Mapping)
                and dict(current) != dict(nested)
            ):
                raise ValueError(
                    f"controller checkpoint {target} disagrees with "
                    f"resume_contract.{source}"
                )
        if payload.get("protocol") is None and payload.get(
            "training_protocol"
        ) is not None:
            payload["protocol"] = payload["training_protocol"]
        if payload.get("instance_regime") is None:
            payload["instance_regime"] = "contention_proper_training"
        if payload.get("model_seed") is None and resume_contract.get(
            "model_seed"
        ) is not None:
            payload["model_seed"] = int(resume_contract["model_seed"])
        if "train_instance_seeds" not in payload:
            base = resume_contract.get("train_instance_seed_base")
            completed = payload.get("completed_training_episodes")
            if base is not None and completed is not None:
                payload["train_instance_seeds"] = tuple(
                    range(int(base), int(base) + int(completed))
                )
    for name in ("environment", "viability_search", "liveness_rule"):
        if not isinstance(payload.get(name), Mapping):
            raise ValueError(f"controller checkpoint is missing {name}")
    if payload.get("exact_safe_mask_authoritative") is not True:
        raise ValueError("controller checkpoint does not preserve exact safety")
    if payload.get("baseline_teacher") is not False:
        raise ValueError("benchmark requires a baseline-free controller")
    return payload


def _make_env(payload: Mapping) -> SmallRoomsEnv:
    config = payload["environment"]
    required = (
        "grid_rows",
        "grid_cols",
        "number_blocks",
        "arrival_rate",
        "proc_mean",
    )
    missing = tuple(name for name in required if name not in config)
    if missing:
        raise ValueError(f"controller environment metadata is incomplete: {missing}")
    return SmallRoomsEnv(
        grid_rows=int(config["grid_rows"]),
        grid_cols=int(config["grid_cols"]),
        number_blocks=int(config["number_blocks"]),
        choose_storage=False,
        arrival_rate=float(config["arrival_rate"]),
        proc_mean=float(config["proc_mean"]),
    )


def _search_config(payload: Mapping) -> ViabilitySearchConfig:
    allowed = ViabilitySearchConfig.__dataclass_fields__
    values = {
        name: payload["viability_search"][name]
        for name in allowed
        if name in payload["viability_search"]
    }
    return ViabilitySearchConfig(**values)


def _liveness_rule(payload: Mapping) -> BoundedEventDeferRule:
    values = payload["liveness_rule"]
    return BoundedEventDeferRule(
        max_option_steps=int(values["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            values["max_consecutive_defer_decisions"]
        ),
    )


def _freeze_agent(payload: dict, *, device: torch.device, seed: int):
    checkpoint_family = payload.get("checkpoint_family")
    if checkpoint_family == EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY:
        agent = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
            payload,
            device=device,
            resumable=False,
            seed=seed,
        )
    else:
        agent = ViabilityGraphHierarchyAgent.from_checkpoint(
            payload,
            device=device,
            resumable=False,
            seed=seed,
        )
    if float(agent.config.gamma) >= 1.0:
        raise ValueError(
            "this benchmark executor requires gamma < 1; gamma-one audit "
            "checkpoints need the finite-episode evaluation executor"
        )
    for network in (agent.Q_local, agent.Q_target):
        network.requires_grad_(False)
        network.eval()
    agent.set_epsilon(0.0)
    return agent


def _recovery_action_signature(action: RecoveryAction) -> dict:
    return {
        "kind": action.kind.value,
        "block_label": action.block_label,
        "source": action.source,
        "destination": action.destination,
        "steps": int(action.steps),
    }


def _certificate_signature(certificate) -> dict:
    return {
        "status": certificate.status.value,
        "search_order": certificate.search_order,
        "exhaustive": bool(certificate.exhaustive),
        "reason": certificate.reason,
        "explored_nodes": int(certificate.explored_nodes),
        "generated_states": int(certificate.generated_states),
        "witness_primitive_steps": certificate.witness_primitive_steps,
        "witness": tuple(
            _recovery_action_signature(action) for action in certificate.witness
        ),
    }


def _environment_signature(env: SmallRoomsEnv) -> dict:
    blocks = []
    for block in sorted(env.blocks, key=lambda item: str(item.label)):
        blocks.append(
            {
                "label": str(block.label),
                "position": None if block.position is None else tuple(block.position),
                "storage_location": (
                    None
                    if block.storage_location is None
                    else tuple(block.storage_location)
                ),
                "carrying": bool(block.carrying),
                "stored": bool(block.stored),
                "delivered": bool(block.delivered),
                "storage_steps_elapsed": int(block.storage_steps_elapsed),
                "stored_time_step": (
                    None
                    if block.stored_time_step is None
                    else int(block.stored_time_step)
                ),
            }
        )
    return {
        "time_steps": int(env.time_steps),
        "agent_position": tuple(env.current_state),
        "terminal": bool(env.is_state_terminal(env.current_state)),
        "blocks": tuple(blocks),
    }


def _frontier_record(
    snapshot: ViabilityCandidateSnapshot,
    *,
    wall_seconds: float,
) -> dict:
    audit = snapshot.audit.audit_dict()
    candidates = tuple(
        {
            "key": candidate.key,
            "mode": candidate.mode.value,
            "action_type": candidate.action_type.value,
            "target_label": candidate.target_label,
            "source": candidate.source,
            "destination": candidate.destination,
            "certificate": _certificate_signature(candidate.certificate),
        }
        for candidate in snapshot.candidates
    )
    return {
        "decision_epoch": int(snapshot.decision_epoch),
        "episode_instance_id": snapshot.episode_instance_id,
        "candidate_keys": tuple(item["key"] for item in candidates),
        "candidate_frontier_digest": _canonical_digest(candidates),
        "candidates": candidates,
        "current_certificate": _certificate_signature(
            snapshot.current_certificate
        ),
        "cache_hits": int(audit["cache_hits"]),
        "cache_misses": int(audit["cache_misses"]),
        "cache_entries": int(audit["cache_entries"]),
        "safe_candidate_count": int(audit["candidate_count"]),
        "unsafe_candidates_rejected": int(audit["unsafe_accept_count"])
        + int(audit["unsafe_recovery_count"]),
        "unknown_candidates_rejected": int(audit["unknown_accept_count"])
        + int(audit["unknown_recovery_count"]),
        "exact_search_seconds": float(audit.get("exact_analysis_seconds", 0.0)),
        "recovery_certification_strategy": audit.get(
            "recovery_certification_strategy",
            EXACT_ONLY_RECOVERY_CERTIFICATION,
        ),
        "relocation_family_anchor_available": bool(
            audit.get("relocation_family_anchor_available", False)
        ),
        "relocation_family_attempt_count": int(
            audit.get("relocation_family_attempt_count", 0)
        ),
        "relocation_family_proof_count": int(
            audit.get("relocation_family_proof_count", 0)
        ),
        "relocation_family_miss_count": int(
            audit.get("relocation_family_miss_count", 0)
        ),
        "relocation_family_setup_seconds": float(
            audit.get("relocation_family_setup_seconds", 0.0)
        ),
        "relocation_family_connection_seconds": float(
            audit.get("relocation_family_connection_seconds", 0.0)
        ),
        "native_recovery_search_count": int(
            audit.get("native_recovery_search_count", 0)
        ),
        "critic_inference_seconds": float(
            audit.get("priority_inference_seconds", 0.0)
        ),
        "priority_states_scored": int(audit.get("priority_states_scored", 0)),
        "priority_pass_count": int(audit.get("priority_pass_count", 0)),
        "certification_order": audit.get("certification_order"),
        "total_frontier_seconds": float(wall_seconds),
        "complete_frontier_exactly_verified": bool(
            audit.get("complete_frontier_exactly_verified", True)
        ),
        "exact_verifier_authority": audit["verifier_authority"],
    }


def _enumerate_frontier(
    env,
    *,
    consecutive_defer: int,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    cache: ViabilityCertificateCache,
    prioritizer: Optional[ViabilityCriticPrioritizer],
    recovery_certification_strategy: str = EXACT_ONLY_RECOVERY_CERTIFICATION,
) -> tuple[ViabilityCandidateSnapshot, dict]:
    started = perf_counter()
    snapshot = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=consecutive_defer,
        search_config=search_config,
        liveness_rule=liveness_rule,
        cache=cache,
        state_prioritizer=prioritizer,
        recovery_certification_strategy=recovery_certification_strategy,
    )
    elapsed = perf_counter() - started
    return snapshot, _frontier_record(snapshot, wall_seconds=elapsed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sum_field(records: Sequence[dict], name: str) -> float:
    return float(sum(float(record[name]) for record in records))


def run_arm(
    *,
    arm: str,
    controller_payload: dict,
    instance,
    instance_seed: int,
    search_config: ViabilitySearchConfig,
    liveness_rule: BoundedEventDeferRule,
    prioritizer: ViabilityCriticPrioritizer,
    max_steps: int,
    device: torch.device,
    recovery_certification_strategy: str = EXACT_ONLY_RECOVERY_CERTIFICATION,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown benchmark arm: {arm}")
    env = _make_env(controller_payload)
    env.current_episode = 1
    env.reset(instance=instance)
    if env.current_episode_instance.instance_id != instance.instance_id:
        raise RuntimeError("arm did not consume the paired EpisodeInstance")
    agent = _freeze_agent(
        controller_payload,
        device=device,
        seed=int(controller_payload.get("model_seed", 0)),
    )
    agent.reset_episode_state()
    cache = ViabilityCertificateCache()
    active_prioritizer = prioritizer if arm == CRITIC_ORDERED_FULL else None

    steps = 0
    total_return = 0.0
    consecutive_defer = 0
    delivery_deviations = []
    frontiers = []
    decisions = []
    method_failure_reason = None
    macro_failures = 0
    illegal_drops = 0
    relocations = 0
    standalone_reconfigurations = 0
    standalone_with_direct_delivery_available = 0
    standalone_without_direct_delivery_available = 0
    directly_deliverable_self_reconfigurations = 0
    decision_epochs_with_direct_delivery_available = 0
    reconfiguration_decision_epochs = 0
    reconfiguration_decision_epochs_with_direct_delivery_available = 0
    reconfiguration_decision_epochs_without_direct_delivery_available = 0
    directly_deliverable_self_reconfiguration_decision_epochs = 0
    pending = None
    pending_index = None
    _sync(device)
    arm_started = perf_counter()

    while steps < max_steps and not env.is_state_terminal(env.current_state):
        if pending is None:
            pending, record = _enumerate_frontier(
                env,
                consecutive_defer=consecutive_defer,
                search_config=search_config,
                liveness_rule=liveness_rule,
                cache=cache,
                prioritizer=active_prioritizer,
                recovery_certification_strategy=(
                    recovery_certification_strategy
                ),
            )
            frontiers.append(record)
            pending_index = len(frontiers) - 1
        snapshot = pending
        frontier_index = pending_index
        pending = None
        pending_index = None
        if not snapshot.candidates:
            method_failure_reason = "no_exact_safe_candidate"
            break
        if not all(
            candidate.certificate.status is ViabilityStatus.SAFE
            for candidate in snapshot.candidates
        ):
            raise RuntimeError("frontier contains a non-SAFE candidate")

        _sync(device)
        selection_started = perf_counter()
        decision = agent.select(snapshot, training=False, epsilon=0.0)
        _sync(device)
        selection_seconds = perf_counter() - selection_started
        direct_delivery_labels = {
            candidate.target_label
            for candidate in snapshot.candidates
            if candidate.action_type is ViabilityActionType.DELIVER
        }
        direct_delivery_available = bool(direct_delivery_labels)
        selected_reconfiguration = (
            decision.candidate.action_type is ViabilityActionType.RECONFIGURE
        )
        selected_reconfigure_block_directly_deliverable = bool(
            selected_reconfiguration
            and decision.candidate.target_label in direct_delivery_labels
        )
        decision_epochs_with_direct_delivery_available += int(
            direct_delivery_available
        )
        if selected_reconfiguration:
            reconfiguration_decision_epochs += 1
            reconfiguration_decision_epochs_with_direct_delivery_available += int(
                direct_delivery_available
            )
            reconfiguration_decision_epochs_without_direct_delivery_available += int(
                not direct_delivery_available
            )
            directly_deliverable_self_reconfiguration_decision_epochs += int(
                selected_reconfigure_block_directly_deliverable
            )
        execution = execute_certified_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=max_steps - steps,
            evaluation=True,
        )
        steps += execution.duration
        total_return += execution.raw_return
        delivery_deviations.extend(execution.delivery_deviations)
        macro_failures += int(not execution.option_success)
        illegal_drops += execution.illegal_drops
        relocations += execution.relocations
        if selected_reconfiguration:
            standalone_reconfigurations += execution.relocations
            if direct_delivery_available:
                standalone_with_direct_delivery_available += execution.relocations
            else:
                standalone_without_direct_delivery_available += execution.relocations
            if selected_reconfigure_block_directly_deliverable:
                directly_deliverable_self_reconfigurations += execution.relocations
        elif execution.relocations:
            raise RuntimeError(
                "VCG physical storage relocation occurred outside a standalone "
                "Reconfigure macro; contention mechanism is unattributable"
            )

        if execution.action_type == ViabilityActionType.DEFER.value:
            outcome = getattr(decision.option, "last_outcome", None)
            observed_event = bool(
                isinstance(outcome, dict)
                and outcome.get("reason") == "observed_event"
            )
            consecutive_defer = 0 if observed_event else consecutive_defer + 1
        else:
            consecutive_defer = 0

        boundary_done = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or execution.duration == 0
        )
        if not boundary_done:
            pending, record = _enumerate_frontier(
                env,
                consecutive_defer=consecutive_defer,
                search_config=search_config,
                liveness_rule=liveness_rule,
                cache=cache,
                prioritizer=active_prioritizer,
                recovery_certification_strategy=(
                    recovery_certification_strategy
                ),
            )
            frontiers.append(record)
            pending_index = len(frontiers) - 1
            if not pending.candidates:
                boundary_done = True
                method_failure_reason = "no_exact_safe_candidate"
                pending = None
                pending_index = None

        agent.observe_outcome(
            decision,
            next_snapshot=None if boundary_done else pending,
            done=boundary_done,
        )
        decisions.append(
            {
                "decision_index": len(decisions),
                "frontier_index": frontier_index,
                "decision_epoch": int(snapshot.decision_epoch),
                "selected_key": decision.candidate.key,
                "selected_mode": decision.candidate.mode.value,
                "selected_action_type": decision.candidate.action_type.value,
                "selection_source": decision.selection_source,
                "liveness_forced": bool(decision.liveness_forced),
                "selected_certificate": _certificate_signature(
                    decision.candidate.certificate
                ),
                "duration": int(execution.duration),
                "discounted_return": float(execution.discounted_return),
                "raw_return": float(execution.raw_return),
                "option_success": bool(execution.option_success),
                "option_terminated": bool(execution.option_terminated),
                "failure_reason": execution.failure_reason,
                "delivery_deviations": tuple(execution.delivery_deviations),
                "relocations": int(execution.relocations),
                "physical_storage_relocations": int(execution.relocations),
                "safe_direct_delivery_candidate_available": bool(
                    direct_delivery_available
                ),
                "selected_reconfigure_block_directly_deliverable": bool(
                    selected_reconfigure_block_directly_deliverable
                ),
                "illegal_drops": int(execution.illegal_drops),
                "selection_seconds": float(selection_seconds),
                "post_macro_state": _environment_signature(env),
            }
        )
        if not execution.option_success and method_failure_reason is None:
            method_failure_reason = (
                f"macro_failure:{execution.action_type}:"
                f"{execution.failure_reason or 'unknown'}"
            )
        if boundary_done:
            break

    _sync(device)
    wall_seconds = perf_counter() - arm_started
    terminal = bool(env.is_state_terminal(env.current_state))
    if steps >= max_steps and not terminal and method_failure_reason is None:
        method_failure_reason = "episode_step_limit"
    success = bool(terminal and method_failure_reason is None)
    timing = summarize_delivery_timing(
        delivery_deviations,
        env.DELIVERY_TARGET_WINDOW,
    )
    contention_metrics = contention_metric_record(
        physical_storage_relocations=relocations,
        target_bound_obstruction_clearances=0,
        standalone_reconfigurations=standalone_reconfigurations,
        standalone_with_direct_delivery_available=(
            standalone_with_direct_delivery_available
        ),
        standalone_without_direct_delivery_available=(
            standalone_without_direct_delivery_available
        ),
        directly_deliverable_self_reconfigurations=(
            directly_deliverable_self_reconfigurations
        ),
    )
    behavior = {
        "instance_id": instance.instance_id,
        "frontiers": tuple(
            {
                "decision_epoch": item["decision_epoch"],
                "candidate_keys": item["candidate_keys"],
                "candidate_frontier_digest": item["candidate_frontier_digest"],
                "current_certificate": item["current_certificate"],
                "candidates": item["candidates"],
            }
            for item in frontiers
        ),
        "decisions": tuple(
            {
                key: value
                for key, value in item.items()
                if key != "selection_seconds"
            }
            for item in decisions
        ),
        "return": float(total_return),
        "steps": int(steps),
        "terminal": terminal,
        "success": success,
        "method_failure_reason": method_failure_reason,
        "final_state": _environment_signature(env),
        "delivery_deviations": tuple(delivery_deviations),
    }
    return {
        "arm": arm,
        "instance_seed": int(instance_seed),
        "instance_id": instance.instance_id,
        "return": float(total_return),
        "steps": int(steps),
        "macro_decisions": len(decisions),
        "terminal": terminal,
        "success": success,
        "strict_method_success": bool(
            success and macro_failures == 0 and illegal_drops == 0
        ),
        "method_failure_reason": method_failure_reason,
        "delivery_deviations": tuple(delivery_deviations),
        **timing,
        **contention_metrics,
        "relocations": int(relocations),
        # Backward-compatible aliases.  ``obstructive_moves`` is deliberately
        # zero for VCG: standalone Reconfigure macros are physical relocation
        # events, not target-bound retrieval-executor clearances.
        "obstructive_moves": 0,
        "decision_epochs_with_direct_delivery_available": int(
            decision_epochs_with_direct_delivery_available
        ),
        "reconfiguration_decision_epochs": int(
            reconfiguration_decision_epochs
        ),
        "reconfiguration_decision_epochs_with_direct_delivery_available": int(
            reconfiguration_decision_epochs_with_direct_delivery_available
        ),
        "reconfiguration_decision_epochs_without_direct_delivery_available": int(
            reconfiguration_decision_epochs_without_direct_delivery_available
        ),
        "directly_deliverable_self_reconfiguration_decision_epochs": int(
            directly_deliverable_self_reconfiguration_decision_epochs
        ),
        "illegal_drops": int(illegal_drops),
        "macro_failures": int(macro_failures),
        "frontiers": tuple(frontiers),
        "decisions": tuple(decisions),
        "exact_cache_hits": int(sum(item["cache_hits"] for item in frontiers)),
        "exact_cache_misses": int(
            sum(item["cache_misses"] for item in frontiers)
        ),
        "exact_search_seconds": _sum_field(frontiers, "exact_search_seconds"),
        "recovery_certification_strategy": recovery_certification_strategy,
        "relocation_family_attempt_count": int(
            sum(item["relocation_family_attempt_count"] for item in frontiers)
        ),
        "relocation_family_proof_count": int(
            sum(item["relocation_family_proof_count"] for item in frontiers)
        ),
        "relocation_family_miss_count": int(
            sum(item["relocation_family_miss_count"] for item in frontiers)
        ),
        "relocation_family_setup_seconds": _sum_field(
            frontiers, "relocation_family_setup_seconds"
        ),
        "relocation_family_connection_seconds": _sum_field(
            frontiers, "relocation_family_connection_seconds"
        ),
        "native_recovery_search_count": int(
            sum(item["native_recovery_search_count"] for item in frontiers)
        ),
        "critic_inference_seconds": _sum_field(
            frontiers, "critic_inference_seconds"
        ),
        "total_frontier_seconds": _sum_field(
            frontiers, "total_frontier_seconds"
        ),
        "episode_wall_seconds": float(wall_seconds),
        "certificate_cache_entries": len(cache),
        "complete_frontier_exactly_verified": all(
            item["complete_frontier_exactly_verified"] for item in frontiers
        ),
        "exact_verifier_authoritative": True,
        "critic_certificate_authority": False,
        "training": False,
        "gradient_steps": 0,
        "behavior_digest": _canonical_digest(behavior),
    }


def _frontier_semantics(run: dict) -> tuple:
    return tuple(
        {
            "decision_epoch": item["decision_epoch"],
            "episode_instance_id": item["episode_instance_id"],
            "candidate_keys": item["candidate_keys"],
            "candidate_frontier_digest": item["candidate_frontier_digest"],
            "candidates": item["candidates"],
            "current_certificate": item["current_certificate"],
        }
        for item in run["frontiers"]
    )


def _decision_semantics(run: dict) -> tuple:
    return tuple(
        {key: value for key, value in item.items() if key != "selection_seconds"}
        for item in run["decisions"]
    )


def compare_pair(exact: dict, prioritized: dict) -> dict:
    exact_misses = tuple(
        item["cache_misses"] for item in exact["frontiers"]
    )
    prioritized_misses = tuple(
        item["cache_misses"] for item in prioritized["frontiers"]
    )
    exact_hits = tuple(item["cache_hits"] for item in exact["frontiers"])
    prioritized_hits = tuple(
        item["cache_hits"] for item in prioritized["frontiers"]
    )
    assertions = {
        "same_instance_id": exact["instance_id"] == prioritized["instance_id"],
        "complete_exact_frontiers": bool(
            exact["complete_frontier_exactly_verified"]
            and prioritized["complete_frontier_exactly_verified"]
        ),
        "identical_frontiers": (
            _frontier_semantics(exact) == _frontier_semantics(prioritized)
        ),
        "identical_selected_actions_and_trajectory": (
            _decision_semantics(exact) == _decision_semantics(prioritized)
        ),
        "identical_behavior_digest": (
            exact["behavior_digest"] == prioritized["behavior_digest"]
        ),
        "identical_outcomes": all(
            exact[name] == prioritized[name]
            for name in (
                "return",
                "steps",
                "macro_decisions",
                "terminal",
                "success",
                "strict_method_success",
                "method_failure_reason",
                "delivery_deviations",
                "relocations",
                "physical_storage_relocations",
                "target_bound_obstruction_clearances",
                "standalone_reconfigurations",
                "standalone_with_direct_delivery_available",
                "standalone_without_direct_delivery_available",
                "directly_deliverable_self_reconfigurations",
                "illegal_drops",
                "macro_failures",
            )
        ),
        "equal_exact_cache_misses": (
            exact["exact_cache_misses"] == prioritized["exact_cache_misses"]
        ),
        "equal_per_frontier_exact_cache_misses": (
            exact_misses == prioritized_misses
        ),
        "equal_exact_cache_hits": (
            exact["exact_cache_hits"] == prioritized["exact_cache_hits"]
        ),
        "equal_per_frontier_exact_cache_hits": exact_hits == prioritized_hits,
        "equal_final_cache_entries": (
            exact["certificate_cache_entries"]
            == prioritized["certificate_cache_entries"]
        ),
        "critic_ordering_exercised": (
            sum(
                item["priority_states_scored"]
                for item in prioritized["frontiers"]
            )
            > 0
            and prioritized["critic_inference_seconds"] > 0.0
            and exact["critic_inference_seconds"] == 0.0
        ),
        "every_selected_action_exactly_safe": all(
            item["selected_certificate"]["status"]
            == ViabilityStatus.SAFE.value
            for run in (exact, prioritized)
            for item in run["decisions"]
        ),
    }
    return {
        "instance_seed": exact["instance_seed"],
        "instance_id": exact["instance_id"],
        "assertions": assertions,
        "passed": all(assertions.values()),
        "exact_cache_misses": exact["exact_cache_misses"],
        "critic_ordered_cache_misses": prioritized["exact_cache_misses"],
        "exact_search_seconds": exact["exact_search_seconds"],
        "critic_ordered_exact_search_seconds": prioritized[
            "exact_search_seconds"
        ],
        "critic_inference_seconds": prioritized["critic_inference_seconds"],
        "exact_total_frontier_seconds": exact["total_frontier_seconds"],
        "critic_ordered_total_frontier_seconds": prioritized[
            "total_frontier_seconds"
        ],
    }


def _warm_prioritizer(
    prioritizer: ViabilityCriticPrioritizer,
    controller_payload: Mapping,
    instance,
    search_config: ViabilitySearchConfig,
) -> None:
    env = _make_env(controller_payload)
    env.reset(instance=instance)
    yard = YardSnapshot.from_env(env)
    agent_position = tuple(env.current_state)
    fixed = online_fixed_obstacles(
        env,
        reserve_queue_cells=search_config.reserve_queue_cells,
    ) - {agent_position}
    state = RecoveryState.from_yard_snapshot(
        yard,
        agent_position=agent_position,
        fixed_obstacles=fixed,
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    )
    prioritizer.prioritize((("untimed-warmup", state),))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pair exact canonical versus critic-ordered complete exact "
            "frontiers under one frozen VCG controller"
        )
    )
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--critic-expected-sha256",
        default=None,
        help="optional pinned SHA-256 identity for the critic checkpoint",
    )
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[74_000])
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be unique")
    sealed = set(args.seeds).intersection(SEALED_STRESS_V1_HOLDOUT_SEEDS)
    if sealed:
        raise ValueError(
            "priority benchmark refuses sealed stress_v1 holdout seeds: "
            f"{tuple(sorted(sealed))}"
        )


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = resolve_device(args.device)
    seed_everything(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = args.output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    controller_payload = _load_controller_checkpoint(
        args.controller_checkpoint
    )
    search_config = _search_config(controller_payload)
    liveness_rule = _liveness_rule(controller_payload)
    manifest = args.dataset_manifest
    if manifest is None:
        root_candidate = args.critic_checkpoint.parent / "dataset-manifest.json"
        nested_candidate = (
            args.critic_checkpoint.parent / "data" / "dataset-manifest.json"
        )
        manifest = (
            root_candidate if root_candidate.is_file() else nested_candidate
        )
    prioritizer = ViabilityCriticPrioritizer.from_checkpoint(
        args.critic_checkpoint,
        device=device,
        dataset_manifest=manifest,
        expected_checkpoint_sha256=args.critic_expected_sha256,
    )

    instances = {}
    for seed in args.seeds:
        env = _make_env(controller_payload)
        instance = env.sample_episode_instance(int(seed))
        instances[int(seed)] = instance
        (instances_dir / f"seed-{seed}.json").write_text(
            instance.to_json() + "\n", encoding="utf-8"
        )
    _warm_prioritizer(
        prioritizer,
        controller_payload,
        instances[int(args.seeds[0])],
        search_config,
    )

    runs = []
    pairs = []
    failed_pairs = []
    for index, seed in enumerate(args.seeds):
        # Alternation reduces systematic first-arm timing bias without sharing
        # a verifier cache or mutable controller/environment state.
        order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
        by_arm = {}
        for arm in order:
            print(f"[{arm}] seed={seed}", flush=True)
            run = run_arm(
                arm=arm,
                controller_payload=controller_payload,
                instance=instances[int(seed)],
                instance_seed=int(seed),
                search_config=search_config,
                liveness_rule=liveness_rule,
                prioritizer=prioritizer,
                max_steps=args.max_steps,
                device=device,
            )
            by_arm[arm] = run
            runs.append(run)
            print(
                f"[{arm}] R={run['return']:.2f} "
                f"strict={int(run['strict_method_success'])} "
                f"dec={run['macro_decisions']} misses={run['exact_cache_misses']} "
                f"exact_s={run['exact_search_seconds']:.4f} "
                f"critic_s={run['critic_inference_seconds']:.4f} "
                f"frontier_s={run['total_frontier_seconds']:.4f}",
                flush=True,
            )
        pair = compare_pair(by_arm[EXACT_FULL], by_arm[CRITIC_ORDERED_FULL])
        pair["execution_order"] = order
        pairs.append(pair)
        if not pair["passed"]:
            failed_pairs.append(
                {
                    "instance_seed": int(seed),
                    "failed_assertions": tuple(
                        name
                        for name, passed in pair["assertions"].items()
                        if not passed
                    ),
                }
            )

    exact_runs = [item for item in runs if item["arm"] == EXACT_FULL]
    priority_runs = [
        item for item in runs if item["arm"] == CRITIC_ORDERED_FULL
    ]
    result = {
        "protocol": PROTOCOL,
        "arms": {
            EXACT_FULL: "canonical_order_complete_exact_frontier",
            CRITIC_ORDERED_FULL: (
                "calibrated_lcb_permutation_complete_exact_frontier"
            ),
        },
        "screening": False,
        "complete_frontier_exactly_verified": True,
        "critic_certificate_authority": False,
        "exact_verifier_authoritative": True,
        "training": False,
        "future_schedule_visible_to_policy": False,
        "device": str(device),
        "controller_checkpoint": str(args.controller_checkpoint.resolve()),
        "controller_checkpoint_sha256": _sha256_file(
            args.controller_checkpoint
        ),
        "critic_checkpoint": str(args.critic_checkpoint.resolve()),
        "critic": prioritizer.audit_dict(),
        "dataset_manifest": str(Path(manifest).resolve()),
        "environment": dict(controller_payload["environment"]),
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "seeds": tuple(int(seed) for seed in args.seeds),
        "pairing_key": "EpisodeInstance.instance_id",
        "cache_contract": "separate_empty_cache_per_arm_per_episode_v1",
        "timing_note": (
            "critic inference is CUDA-synchronized; full-frontier ordering "
            "cannot reduce verifier calls and timing is descriptive"
        ),
        "runs": tuple(runs),
        "pairs": tuple(pairs),
        "summary": {
            "pairs": len(pairs),
            "passed_pairs": sum(item["passed"] for item in pairs),
            "all_invariants_passed": not failed_pairs,
            "mean_exact_cache_misses": fmean(
                item["exact_cache_misses"] for item in exact_runs
            ),
            "mean_critic_ordered_cache_misses": fmean(
                item["exact_cache_misses"] for item in priority_runs
            ),
            "mean_exact_search_seconds": fmean(
                item["exact_search_seconds"] for item in exact_runs
            ),
            "mean_critic_ordered_exact_search_seconds": fmean(
                item["exact_search_seconds"] for item in priority_runs
            ),
            "mean_critic_inference_seconds": fmean(
                item["critic_inference_seconds"] for item in priority_runs
            ),
            "mean_exact_total_frontier_seconds": fmean(
                item["total_frontier_seconds"] for item in exact_runs
            ),
            "mean_critic_ordered_total_frontier_seconds": fmean(
                item["total_frontier_seconds"] for item in priority_runs
            ),
            "failed_pairs": tuple(failed_pairs),
        },
    }
    result_path = args.output_dir / RESULT_NAME
    result_path.write_text(
        json.dumps(
            _json_safe(result),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Results: {result_path}", flush=True)
    if failed_pairs:
        raise RuntimeError(
            "paired priority benchmark invariant failure; inspect "
            f"{result_path}: {failed_pairs}"
        )
    return result


if __name__ == "__main__":
    main()
