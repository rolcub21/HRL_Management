#!/usr/bin/env python3
"""Development-only counterfactual audit of VCG proactive relocations.

For every eligible factual standalone ``Reconfigure`` decision, this runner
forks the exact same strict decision state into:

``R``
    the factual VCG reconfiguration;
``D_Q``
    the highest frozen-Q exact-SAFE delivery that remains admissible after the
    recovery-witness guard; and
``D_SELF``
    the highest-Q admissible delivery of the reconfigured block itself, when
    that block is directly deliverable; and
``W``
    the exact admissible Defer candidate, when one actually exists.

No WAIT action is synthesized.  After the forced first macro, every branch is
controlled by the same frozen epsilon-zero VCG policy.  The script is pinned
to the already-opened V1.1 development panel (80000--80029) and the finalized
selected-best model seeds 0/1/2.  It cannot open a prospective or sealed panel
and cannot authorize a performance claim.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence
import uuid

import torch

from benchmark_viability_critic_priority import (
    _canonical_digest,
    _environment_signature,
    _enumerate_frontier,
    _freeze_agent,
    _liveness_rule,
    _search_config,
    _sha256_file,
)
from compare_vcg_dense_pareto import (
    EVALUATION_SEEDS,
    MAX_STEPS,
    MODEL_SEEDS,
    PROTOCOL as SOURCE_PROTOCOL,
    SELECTED_BEST,
    _authenticate_training_bundle,
)
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import summarize_delivery_timing
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionType,
    ViabilityCertificateCache,
)
from train_vcg_dense_proper import FROZEN_OBJECTIVE_SPEC
from train_viability_graph_smdp import execute_certified_macro, resolve_device
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    ObjectiveAuditSmallRoomsEnv,
)
from viability_graph_hierarchy import ViabilityGraphDecision
from vcg_relocation_amortization_analysis import (
    build_relocation_amortization_report,
    validate_branch_rows,
)


PROTOCOL = "vcg_proactive_relocation_amortization_development_v1"
PROTOCOL_SCHEMA_VERSION = 1
BRANCH_SCHEMA_VERSION = 1
SOURCE_DIR_DEFAULT = Path(
    "results/vcg-dense-v1-1-pareto-development-30seed"
)
REANALYSIS_AUDIT_DEFAULT = Path(
    "results/vcg-contention-metric-reanalysis-development-30seed/"
    "contention-metrics-audit.json"
)
OUTPUT_DIR_DEFAULT = Path(
    "results/vcg-proactive-relocation-amortization-development-v1"
)
TRAINING_DIR_DEFAULTS = tuple(
    Path(f"results/vcg-dense-v1-1-seed{seed}-500ep")
    for seed in MODEL_SEEDS
)
EXPECTED_SELECTED_CHECKPOINT_SHA256 = {
    0: "aa52ee14612ae39bb40fb9bdfecba8b7a44c40665628d26bb51f9f8f7a5355e8",
    1: "575326ddb3dbd535f4849e0f9d85bc55b8063279ebcfb6a872107f459dc56bb4",
    2: "3f681965feaca098a8d7fc2b58b41a8f0784191417e964a1fe7d5be2f5781917",
}
EXPECTED_FULL_PRIMARY_EVENT_COUNT = 167
EXPECTED_FULL_DIRECT_SELF_EVENT_COUNT = 166
BRANCH_IDS = ("R", "D_Q", "D_SELF", "W")


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_safe(value), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
    )


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_equal(name: str, observed, expected) -> None:
    if _json_safe(observed) != _json_safe(expected):
        raise RuntimeError(
            f"{name} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise RuntimeError(
            f"{name} mismatch: observed={observed}, expected={expected}"
        )


def _make_audit_env(payload: Mapping) -> ObjectiveAuditSmallRoomsEnv:
    config = payload["environment"]
    return ObjectiveAuditSmallRoomsEnv(
        timing_objective=FROZEN_OBJECTIVE_SPEC,
        grid_rows=int(config["grid_rows"]),
        grid_cols=int(config["grid_cols"]),
        number_blocks=int(config["number_blocks"]),
        choose_storage=False,
        arrival_rate=float(config["arrival_rate"]),
        proc_mean=float(config["proc_mean"]),
    )


def _block_signature(block) -> dict:
    names = (
        "label",
        "position",
        "carrying",
        "delivered",
        "stored",
        "picked",
        "storage_location",
        "storage_steps_needed",
        "storage_steps_elapsed",
        "stored_time_step",
        "delivered_time_step",
        "delivery_error_time",
        "arrival_step",
        "cleared",
        "hold",
        "hold_counter",
    )
    return {name: getattr(block, name, None) for name in names}


def _branch_state_signature(env) -> dict:
    return {
        "episode_instance_id": env.current_episode_instance.instance_id,
        "time_steps": int(env.time_steps),
        "agent_position": tuple(env.current_state),
        "terminal": bool(env.is_state_terminal(env.current_state)),
        "storage_counts": tuple(
            sorted((tuple(cell), int(count)) for cell, count in env.storage_counts.items())
        ),
        "blocks": tuple(
            _block_signature(block)
            for block in sorted(env.blocks, key=lambda item: str(item.label))
        ),
    }


def _guard_signature(state: Mapping) -> dict:
    witness = []
    for action in state.get("active_witness", ()):
        witness.append(
            {
                "kind": action.kind.value,
                "block_label": action.block_label,
                "source": action.source,
                "destination": action.destination,
                "steps": int(action.steps),
            }
        )
    return {
        "max_nonprogress_recovery_decisions": int(
            state["max_nonprogress_recovery_decisions"]
        ),
        "force_when_due": bool(state["force_when_due"]),
        "nonprogress_recovery_decisions": int(
            state.get("nonprogress_recovery_decisions", 0)
        ),
        "active_witness": tuple(witness),
        "witness_cursor": int(state.get("witness_cursor", 0)),
        "activations": int(state.get("activations", 0)),
        "forced_decisions": int(state.get("forced_decisions", 0)),
        "completed_witnesses": int(state.get("completed_witnesses", 0)),
        "mismatches": int(state.get("mismatches", 0)),
    }


def _new_deliveries(env, delivered_before: set[str]) -> tuple[dict, ...]:
    rows = []
    for block in env.blocks:
        label = str(block.label)
        if label in delivered_before or not block.delivered:
            continue
        if block.delivery_error_time is None or block.delivered_time_step is None:
            raise RuntimeError("delivered block lacks timing provenance")
        rows.append(
            {
                "label": label,
                "signed_error": float(block.delivery_error_time),
                "delivered_time_step": int(block.delivered_time_step),
            }
        )
    return tuple(sorted(rows, key=lambda row: row["delivered_time_step"]))


def _discounted(records: Sequence[Mapping], objective: str, gamma: float) -> float:
    return float(
        sum(
            (float(gamma) ** index)
            * float(record["step_reward_by_objective"][objective])
            for index, record in enumerate(records)
        )
    )


def _objective_totals(records: Sequence[Mapping], gamma: float) -> dict:
    records = tuple(records)
    return {
        "legacy_return_raw": float(
            sum(
                float(record["step_reward_by_objective"][LEGACY_CLIPPED])
                for record in records
            )
        ),
        "dense_return_raw": float(
            sum(
                float(record["step_reward_by_objective"][DENSE_PIECEWISE])
                for record in records
            )
        ),
        "legacy_discounted_return": _discounted(
            records, LEGACY_CLIPPED, gamma
        ),
        "dense_discounted_return": _discounted(
            records, DENSE_PIECEWISE, gamma
        ),
    }


def _stable_best_index(
    snapshot,
    prepared,
    q_values: Sequence[float],
    action_type: ViabilityActionType,
) -> Optional[int]:
    eligible = [
        index
        for index, source_index in enumerate(prepared.source_indices)
        if snapshot.candidates[source_index].action_type is action_type
    ]
    if not eligible:
        return None
    maximum = max(float(q_values[index]) for index in eligible)
    return min(
        (
            index
            for index in eligible
            if float(q_values[index]) == maximum
        ),
        key=lambda index: snapshot.candidates[
            prepared.source_indices[index]
        ].key,
    )


def _stable_best_delivery_for_label(
    snapshot,
    prepared,
    q_values: Sequence[float],
    target_label: str,
) -> Optional[int]:
    eligible = [
        index
        for index, source_index in enumerate(prepared.source_indices)
        if (
            snapshot.candidates[source_index].action_type
            is ViabilityActionType.DELIVER
            and snapshot.candidates[source_index].target_label == target_label
        )
    ]
    if not eligible:
        return None
    maximum = max(float(q_values[index]) for index in eligible)
    return min(
        (
            index
            for index in eligible
            if float(q_values[index]) == maximum
        ),
        key=lambda index: snapshot.candidates[
            prepared.source_indices[index]
        ].key,
    )


def _manual_decision(
    agent,
    snapshot,
    *,
    candidate_key: str,
    selection_source: str,
) -> tuple[ViabilityGraphDecision, float, tuple[str, ...]]:
    prepared, forced = agent._admissible_prepared(snapshot)
    if forced:
        raise RuntimeError(
            "counterfactual initial action is not post-guard admissible"
        )
    q_tensor = agent._score_records(prepared.records)
    q_values = tuple(float(value) for value in q_tensor.detach().cpu().tolist())
    matches = [
        index
        for index, source_index in enumerate(prepared.source_indices)
        if snapshot.candidates[source_index].key == candidate_key
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"counterfactual candidate {candidate_key!r} is not uniquely "
            "post-guard admissible"
        )
    index = matches[0]
    source_index = prepared.source_indices[index]
    candidate = snapshot.candidates[source_index]
    exact_rank_progress = bool(
        snapshot.audit.recovery_rank_exact
        and candidate.mode.value == "recover"
        and candidate.rank_delta is not None
        and candidate.rank_delta > 0
    )
    decision = ViabilityGraphDecision(
        candidate=candidate,
        record=prepared.records[index],
        prepared_snapshot=prepared,
        q_values=q_values,
        mode_values=(),
        explored=False,
        selection_source=selection_source,
        liveness_forced=False,
        exact_rank_progress=exact_rank_progress,
    )
    admissible_keys = tuple(
        snapshot.candidates[source].key for source in prepared.source_indices
    )
    return decision, q_values[index], admissible_keys


def _update_consecutive_defer(decision, execution, current: int) -> int:
    if execution.action_type != ViabilityActionType.DEFER.value:
        return 0
    outcome = getattr(decision.option, "last_outcome", None)
    observed_event = bool(
        isinstance(outcome, dict) and outcome.get("reason") == "observed_event"
    )
    return 0 if observed_event else int(current) + 1


def _run_branch(
    *,
    branch_id: str,
    initial_candidate_key: str,
    pre_env,
    pre_guard_state: Mapping,
    consecutive_defer: int,
    episode_steps_before: int,
    controller_payload: Mapping,
    search_config,
    liveness_rule,
    expected_frontier_digest: str,
    expected_state_digest: str,
    expected_guard_digest: str,
    device: torch.device,
) -> dict:
    if branch_id not in BRANCH_IDS:
        raise ValueError(f"unknown branch {branch_id}")
    env = deepcopy(pre_env)
    if _digest(_branch_state_signature(env)) != expected_state_digest:
        raise RuntimeError("deep-copied branch environment changed state")
    agent = _freeze_agent(
        dict(controller_payload),
        device=device,
        seed=int(controller_payload.get("model_seed", 0)),
    )
    agent.recovery_witness_guard.load_state_dict(deepcopy(dict(pre_guard_state)))
    if _digest(_guard_signature(agent.recovery_witness_guard.state_dict())) != (
        expected_guard_digest
    ):
        raise RuntimeError("restored branch guard changed hidden state")
    cache = ViabilityCertificateCache()
    record_start = len(env.objective_audit_records)
    branch_epoch = int(env.time_steps)
    remaining_labels = tuple(
        str(block.label) for block in env.blocks if not block.delivered
    )
    guard_start = agent.recovery_witness_guard.audit_dict()
    steps = 0
    consecutive = int(consecutive_defer)
    physical_relocations = 0
    macro_failures = 0
    illegal_drops = 0
    method_failure_reason = None
    decisions = []
    deliveries = []
    frontier_records = []
    pending = None
    pending_record = None
    initial_duration = None
    initial_relocations = None
    initial_q = None

    while (
        episode_steps_before + steps < MAX_STEPS
        and not env.is_state_terminal(env.current_state)
    ):
        if pending is None:
            pending, pending_record = _enumerate_frontier(
                env,
                consecutive_defer=consecutive,
                search_config=search_config,
                liveness_rule=liveness_rule,
                cache=cache,
                prioritizer=None,
            )
        snapshot = pending
        frontier_record = pending_record
        pending = None
        pending_record = None
        frontier_records.append(frontier_record)
        if not snapshot.candidates:
            method_failure_reason = "no_exact_safe_candidate"
            break
        if not all(
            candidate.certificate.status is ViabilityStatus.SAFE
            for candidate in snapshot.candidates
        ):
            raise RuntimeError("branch frontier contains non-SAFE candidate")

        first = not decisions
        if first and frontier_record["candidate_frontier_digest"] != (
            expected_frontier_digest
        ):
            raise RuntimeError("fresh branch frontier differs at branch point")
        if first and branch_id == "R":
            decision = agent.select(snapshot, training=False, epsilon=0.0)
            if decision.candidate.key != initial_candidate_key:
                raise RuntimeError(
                    "R branch did not reproduce the factual VCG decision"
                )
            try:
                prepared_index = decision.prepared_snapshot.source_indices.index(
                    snapshot.candidates.index(decision.candidate)
                )
            except ValueError as error:
                raise RuntimeError("factual candidate missing from prepared frontier") from error
            initial_q = float(decision.q_values[prepared_index])
            admissible_keys = tuple(
                snapshot.candidates[source].key
                for source in decision.prepared_snapshot.source_indices
            )
        elif first:
            decision, initial_q, admissible_keys = _manual_decision(
                agent,
                snapshot,
                candidate_key=initial_candidate_key,
                selection_source=f"counterfactual_forced_{branch_id}",
            )
        else:
            decision = agent.select(snapshot, training=False, epsilon=0.0)
            admissible_keys = tuple(
                snapshot.candidates[source].key
                for source in decision.prepared_snapshot.source_indices
            )

        delivered_before = {
            str(block.label) for block in env.blocks if block.delivered
        }
        execution = execute_certified_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=MAX_STEPS - episode_steps_before - steps,
            evaluation=True,
        )
        macro_start = steps
        steps += int(execution.duration)
        physical_relocations += int(execution.relocations)
        macro_failures += int(not execution.option_success)
        illegal_drops += int(execution.illegal_drops)
        new_deliveries = _new_deliveries(env, delivered_before)
        deliveries.extend(new_deliveries)
        if first:
            initial_duration = int(execution.duration)
            initial_relocations = int(execution.relocations)

        consecutive = _update_consecutive_defer(
            decision, execution, consecutive
        )
        boundary_done = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or execution.duration == 0
        )
        if not boundary_done:
            pending, pending_record = _enumerate_frontier(
                env,
                consecutive_defer=consecutive,
                search_config=search_config,
                liveness_rule=liveness_rule,
                cache=cache,
                prioritizer=None,
            )
            if not pending.candidates:
                boundary_done = True
                method_failure_reason = "no_exact_safe_candidate"
                pending = None
                pending_record = None
        agent.observe_outcome(
            decision,
            next_snapshot=None if boundary_done else pending,
            done=boundary_done,
        )
        decisions.append(
            {
                "decision_index": len(decisions),
                "decision_epoch": int(snapshot.decision_epoch),
                "selected_key": decision.candidate.key,
                "selected_action_type": decision.candidate.action_type.value,
                "selected_target_label": decision.candidate.target_label,
                "selection_source": decision.selection_source,
                "liveness_forced": bool(decision.liveness_forced),
                "all_frontier_candidates_exact_safe": True,
                "admissible_candidate_keys": admissible_keys,
                "macro_start_step": int(macro_start),
                "duration": int(execution.duration),
                "physical_relocations": int(execution.relocations),
                "option_success": bool(execution.option_success),
                "failure_reason": execution.failure_reason,
                "illegal_drops": int(execution.illegal_drops),
                "new_deliveries": new_deliveries,
            }
        )
        if not execution.option_success and method_failure_reason is None:
            method_failure_reason = (
                f"macro_failure:{execution.action_type}:"
                f"{execution.failure_reason or 'unknown'}"
            )
        if boundary_done:
            break

    if initial_duration is None or initial_relocations is None:
        raise RuntimeError("branch executed no initial macro")
    terminal = bool(env.is_state_terminal(env.current_state))
    if (
        episode_steps_before + steps >= MAX_STEPS
        and not terminal
        and method_failure_reason is None
    ):
        method_failure_reason = "episode_step_limit"
    success = bool(terminal and method_failure_reason is None)
    strict = bool(success and macro_failures == 0 and illegal_drops == 0)

    records = tuple(env.objective_audit_records[record_start:])
    if len(records) != steps:
        raise RuntimeError("primitive objective record count disagrees with steps")
    immediate_records = records[:initial_duration]
    future_records = records[initial_duration:]
    total_objectives = _objective_totals(records, agent.config.gamma)
    immediate_objectives = _objective_totals(
        immediate_records, agent.config.gamma
    )
    future_successor_objectives = _objective_totals(
        future_records, agent.config.gamma
    )
    future_branch_objectives = {
        "legacy_discounted_return": float(agent.config.gamma**initial_duration)
        * future_successor_objectives["legacy_discounted_return"],
        "dense_discounted_return": float(agent.config.gamma**initial_duration)
        * future_successor_objectives["dense_discounted_return"],
    }
    labeled_errors = {
        str(item["label"]): float(item["signed_error"])
        for item in deliveries
    }
    delivery_values = tuple(labeled_errors.values())
    timing = summarize_delivery_timing(
        delivery_values, FROZEN_OBJECTIVE_SPEC.window
    )
    guard_end = agent.recovery_witness_guard.audit_dict()
    completed_labels = tuple(sorted(labeled_errors))
    expected_labels = tuple(sorted(remaining_labels))
    if strict and completed_labels != expected_labels:
        raise RuntimeError(
            "strict branch did not complete exactly the branch-point labels"
        )
    return {
        "branch_id": branch_id,
        "execution_reused": False,
        "derived_from_branch_id": None,
        "initial_candidate_key": initial_candidate_key,
        "initial_action_type": decisions[0]["selected_action_type"],
        "initial_target_label": decisions[0]["selected_target_label"],
        "initial_q": float(initial_q),
        "initial_macro_duration": int(initial_duration),
        "initial_macro_physical_relocations": int(initial_relocations),
        "initial_macro_legacy_return_raw": immediate_objectives[
            "legacy_return_raw"
        ],
        "initial_macro_dense_return_raw": immediate_objectives[
            "dense_return_raw"
        ],
        "initial_macro_legacy_discounted_return": immediate_objectives[
            "legacy_discounted_return"
        ],
        "initial_macro_dense_discounted_return": immediate_objectives[
            "dense_discounted_return"
        ],
        "future_only_steps": int(steps - initial_duration),
        "total_steps": int(steps),
        "future_only_physical_relocations": int(
            physical_relocations - initial_relocations
        ),
        "total_physical_relocations": int(physical_relocations),
        "future_only_legacy_return_raw": future_successor_objectives[
            "legacy_return_raw"
        ],
        "future_only_dense_return_raw": future_successor_objectives[
            "dense_return_raw"
        ],
        "total_legacy_return_raw": total_objectives["legacy_return_raw"],
        "total_dense_return_raw": total_objectives["dense_return_raw"],
        "future_only_legacy_discounted_at_branch_start": (
            future_branch_objectives["legacy_discounted_return"]
        ),
        "future_only_dense_discounted_at_branch_start": (
            future_branch_objectives["dense_discounted_return"]
        ),
        "future_only_legacy_discounted_at_successor": (
            future_successor_objectives["legacy_discounted_return"]
        ),
        "future_only_dense_discounted_at_successor": (
            future_successor_objectives["dense_discounted_return"]
        ),
        "total_legacy_discounted_return": total_objectives[
            "legacy_discounted_return"
        ],
        "total_dense_discounted_return": total_objectives[
            "dense_discounted_return"
        ],
        "terminal": terminal,
        "success": success,
        "strict_method_success": strict,
        "method_failure_reason": method_failure_reason,
        "macro_failures": int(macro_failures),
        "illegal_drops": int(illegal_drops),
        "expected_remaining_labels": expected_labels,
        "remaining_delivery_count": len(expected_labels),
        "completed_remaining_delivery_count": len(completed_labels),
        "completed_remaining_labels": completed_labels,
        "labeled_errors": labeled_errors,
        "delivery_sequence": tuple(deliveries),
        **timing,
        "liveness_forced_decisions": int(
            sum(bool(item["liveness_forced"]) for item in decisions)
        ),
        "guard_activations_delta": int(
            guard_end["activations"] - guard_start["activations"]
        ),
        "guard_forced_decisions_delta": int(
            guard_end["forced_decisions"] - guard_start["forced_decisions"]
        ),
        "guard_mismatches_delta": int(
            guard_end["mismatches"] - guard_start["mismatches"]
        ),
        "all_candidates_exact_safe": bool(
            all(
                item["all_frontier_candidates_exact_safe"]
                for item in decisions
            )
        ),
        "empty_safe_frontiers": int(
            method_failure_reason == "no_exact_safe_candidate"
        ),
        "fresh_certificate_cache": True,
        "exact_cache_hits": int(
            sum(item["cache_hits"] for item in frontier_records)
        ),
        "exact_cache_misses": int(
            sum(item["cache_misses"] for item in frontier_records)
        ),
        "decision_keys": tuple(item["selected_key"] for item in decisions),
        "decision_records": tuple(decisions),
        "final_state": _environment_signature(env),
        "final_branch_state_digest": _digest(_branch_state_signature(env)),
        "branch_epoch": branch_epoch,
        "primitive_objective_record_count": len(records),
    }


def _matched_timing(left: Mapping, right: Mapping) -> dict:
    left_errors = {
        str(label): float(value)
        for label, value in left["labeled_errors"].items()
    }
    right_errors = {
        str(label): float(value)
        for label, value in right["labeled_errors"].items()
    }
    left_expected = tuple(sorted(str(value) for value in left["expected_remaining_labels"]))
    right_expected = tuple(sorted(str(value) for value in right["expected_remaining_labels"]))
    labels = tuple(sorted(set(left_errors).intersection(right_errors)))
    eligible = bool(
        left.get("strict_method_success")
        and right.get("strict_method_success")
        and left_expected == right_expected
        and tuple(sorted(left_errors)) == left_expected
        and tuple(sorted(right_errors)) == right_expected
    )
    if not eligible:
        return {
            "timing_comparison_eligible": False,
            "timing_ineligibility_reason": (
                "both branches must be strict and complete exactly the same "
                "branch-point labels"
            ),
            "matched_labels": labels,
            "matched_label_count": len(labels),
            "R_missing_labels": tuple(sorted(set(right_expected) - set(left_errors))),
            "alternative_missing_labels": tuple(
                sorted(set(left_expected) - set(right_errors))
            ),
            "mean_absolute_error_improvement_for_R": None,
            "mean_tardiness_improvement_for_R": None,
            "per_label": (),
        }
    rows = tuple(
        {
            "label": label,
            "R_signed_error": left_errors[label],
            "alternative_signed_error": right_errors[label],
            "absolute_error_improvement_for_R": (
                abs(right_errors[label]) - abs(left_errors[label])
            ),
            "tardiness_improvement_for_R": (
                max(0.0, right_errors[label])
                - max(0.0, left_errors[label])
            ),
        }
        for label in labels
    )
    return {
        "timing_comparison_eligible": True,
        "timing_ineligibility_reason": None,
        "matched_labels": labels,
        "matched_label_count": len(labels),
        "R_missing_labels": tuple(sorted(set(right_errors) - set(left_errors))),
        "alternative_missing_labels": tuple(
            sorted(set(left_errors) - set(right_errors))
        ),
        "mean_absolute_error_improvement_for_R": (
            float(fmean(item["absolute_error_improvement_for_R"] for item in rows))
            if rows
            else None
        ),
        "mean_tardiness_improvement_for_R": (
            float(fmean(item["tardiness_improvement_for_R"] for item in rows))
            if rows
            else None
        ),
        "per_label": rows,
    }


def _branch_comparison(R: Mapping, alternative: Mapping) -> dict:
    timing = _matched_timing(R, alternative)
    return {
        "alternative_branch_id": alternative["branch_id"],
        "future_relocations_avoided_by_R": int(
            alternative["future_only_physical_relocations"]
            - R["future_only_physical_relocations"]
        ),
        "total_relocations_avoided_by_R": int(
            alternative["total_physical_relocations"]
            - R["total_physical_relocations"]
        ),
        "steps_avoided_by_R": int(
            alternative["total_steps"] - R["total_steps"]
        ),
        "dense_return_advantage_for_R": float(
            R["total_dense_return_raw"]
            - alternative["total_dense_return_raw"]
        ),
        "legacy_return_advantage_for_R": float(
            R["total_legacy_return_raw"]
            - alternative["total_legacy_return_raw"]
        ),
        "dense_discounted_return_advantage_for_R": float(
            R["total_dense_discounted_return"]
            - alternative["total_dense_discounted_return"]
        ),
        "legacy_discounted_return_advantage_for_R": float(
            R["total_legacy_discounted_return"]
            - alternative["total_legacy_discounted_return"]
        ),
        **timing,
    }


def _event_ledger_path(
    output_dir: Path, method_id: str, instance_seed: int, decision_index: int
) -> Path:
    return (
        output_dir
        / "event-ledger"
        / method_id
        / f"seed-{instance_seed}"
        / f"decision-{decision_index:04d}.json"
    )


def _load_reanalysis_map(path: Path) -> tuple[dict, str]:
    audit = _load_json(path)
    _require_equal(
        "contention reanalysis opened no new rollouts",
        audit.get("rollouts_executed"),
        0,
    )
    rows = {}
    for row in audit.get("run_reconstruction_audits", ()):
        if row.get("method_group") != "vcg_dense_v1_1_selected_3seed":
            continue
        key = (str(row["method_id"]), int(row["instance_seed"]))
        if key in rows:
            raise RuntimeError("duplicate corrected contention reconstruction")
        rows[key] = row
    expected = len(MODEL_SEEDS) * len(EVALUATION_SEEDS)
    if len(rows) != expected:
        raise RuntimeError(
            f"corrected contention reconstruction has {len(rows)}/{expected} "
            "primary runs"
        )
    return rows, _sha256_file(path)


def _authenticate_source_panel_chain(source_dir: Path) -> dict:
    names = (
        "protocol-manifest.json",
        "instance-manifest.json",
        "pareto-report.json",
        "pareto-audit.json",
        "pareto-runs.csv",
    )
    paths = {name: source_dir / name for name in names}
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    protocol = _load_json(paths["protocol-manifest.json"])
    instances = _load_json(paths["instance-manifest.json"])
    report = _load_json(paths["pareto-report.json"])
    audit = _load_json(paths["pareto-audit.json"])
    _require_equal("source protocol", protocol.get("protocol"), SOURCE_PROTOCOL)
    _require_equal("source report protocol", report.get("runner_protocol"), SOURCE_PROTOCOL)
    _require_equal("source audit protocol", audit.get("protocol"), SOURCE_PROTOCOL)
    for name, artifact in (("protocol", protocol), ("report", report)):
        _require_equal(
            f"source {name} development scope",
            artifact.get("scope"),
            "development_only_sealed_panels_unopened",
        )
        _require_equal(
            f"source {name} performance authorization",
            artifact.get("performance_claim_authorized"),
            False,
        )
    _require_equal("source evaluation seeds", protocol.get("evaluation_seeds"), EVALUATION_SEEDS)
    _require_equal("source audit protocol manifest", audit.get("protocol_manifest"), protocol)
    _require_equal("source audit instance manifest", audit.get("instance_manifest"), instances)
    _require_equal("source report instance manifest", report.get("instance_manifest"), instances)
    ledger_manifest = audit.get("run_ledger_manifest")
    if not isinstance(ledger_manifest, list) or len(ledger_manifest) != 270:
        raise RuntimeError("source panel must expose 270 physical ledger hashes")
    authenticated = []
    keys = set()
    for entry in ledger_manifest:
        key = str(entry.get("run_key"))
        if not key or key in keys:
            raise RuntimeError(f"invalid source ledger manifest key: {key}")
        keys.add(key)
        path = Path(str(entry.get("path"))).resolve()
        if source_dir != path and source_dir not in path.parents:
            raise RuntimeError(f"source ledger escapes source root: {path}")
        sha = _sha256_file(path)
        _require_equal("source ledger manifest SHA", sha, entry.get("sha256"))
        ledger = _load_json(path)
        _require_equal("source ledger manifest run key", ledger.get("run_key"), key)
        _require_equal(
            "source ledger manifest input fingerprint",
            ledger.get("input_fingerprint"),
            entry.get("input_fingerprint"),
        )
        if not isinstance(ledger.get("run"), Mapping) or not isinstance(
            ledger.get("method_audit"), Mapping
        ):
            raise RuntimeError(f"source ledger is incomplete: {path}")
        authenticated.append(
            {
                "run_key": key,
                "path": str(path),
                "sha256": sha,
                "input_fingerprint": entry.get("input_fingerprint"),
            }
        )
    return {
        "artifact_sha256": hashes,
        "run_ledger_count": len(authenticated),
        "run_ledger_manifest_digest": _digest(authenticated),
        "run_ledger_manifest": tuple(authenticated),
    }


def _authenticate_instances(source_dir: Path, payload: Mapping):
    manifest_path = source_dir / "instance-manifest.json"
    manifest = _load_json(manifest_path)
    _require_equal(
        "source instance protocol",
        manifest.get("contract", {}).get("protocol"),
        SOURCE_PROTOCOL,
    )
    _require_equal(
        "source instance seeds",
        manifest.get("contract", {}).get("seeds"),
        EVALUATION_SEEDS,
    )
    result = {}
    records = manifest.get("instances", {})
    for seed in EVALUATION_SEEDS:
        record = records.get(str(seed))
        if not isinstance(record, Mapping):
            raise RuntimeError(f"source instance manifest lacks seed {seed}")
        expected_path = (source_dir / "instances" / f"seed-{seed}.json").resolve()
        path = Path(record["path"]).resolve()
        _require_equal(f"seed {seed} instance path", path, expected_path)
        _require_equal(
            f"seed {seed} instance SHA-256",
            _sha256_file(path),
            record["sha256"],
        )
        instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        instance.validate_for(_make_audit_env(payload))
        expected = _make_audit_env(payload).sample_episode_instance(seed)
        _require_equal(f"seed {seed} deterministic instance", instance, expected)
        _require_equal(f"seed {seed} instance id", instance.instance_id, record["instance_id"])
        _require_equal(f"seed {seed} schedule id", instance.schedule_id, record["schedule_id"])
        result[seed] = (instance, dict(record))
    return result, manifest, _sha256_file(manifest_path)


def _source_ledger(
    source_dir: Path,
    arm,
    instance_seed: int,
    instance_record: Mapping,
    reconstruction: Mapping,
) -> tuple[dict, str]:
    path = (
        source_dir
        / "run-ledger"
        / arm.method_id
        / f"seed-{instance_seed}.json"
    )
    ledger = _load_json(path)
    sha = _sha256_file(path)
    _require_equal("source ledger protocol", ledger.get("protocol"), SOURCE_PROTOCOL)
    _require_equal(
        "source ledger run key",
        ledger.get("run_key"),
        f"{arm.method_id}:{instance_seed}",
    )
    contract = ledger.get("input_contract", {})
    _require_equal("source method id", contract.get("method_id"), arm.method_id)
    _require_equal("source checkpoint SHA", contract.get("checkpoint_sha256"), arm.checkpoint_sha256)
    _require_equal("source policy digest", contract.get("deployment_policy_digest"), arm.deployment_policy_digest)
    _require_equal("source instance SHA", contract.get("instance_sha256"), instance_record["sha256"])
    _require_equal("corrected audit source ledger SHA", reconstruction.get("source_ledger_sha256"), sha)
    method_audit = ledger.get("method_audit", {})
    _require_equal(
        "source VCG complete exact frontier",
        method_audit.get("complete_frontier_exactly_verified"),
        True,
    )
    _require_equal(
        "source VCG exact verifier authority",
        method_audit.get("exact_verifier_authoritative"),
        True,
    )
    _require_equal("source VCG strict completion", ledger["run"].get("strict_method_success"), 1.0)
    return ledger, sha


def _validate_source_factual(
    factual: Mapping,
    source_ledger: Mapping,
    reconstruction: Mapping,
) -> None:
    source_run = source_ledger["run"]
    source_audit = source_ledger["method_audit"]
    _close("factual legacy return", factual["legacy_return"], source_run["return"])
    _close("factual dense return", factual["dense_return"], source_run["dense_rescored_return"])
    _require_equal("factual steps", factual["steps"], source_run["steps"])
    _require_equal("factual deviations", factual["delivery_deviations"], source_run["delivery_deviations"])
    _require_equal("factual relocations", factual["relocations"], source_run["relocations"])
    observed_frontiers = tuple(
        (
            int(item["decision_epoch"]),
            tuple(item["candidate_keys"]),
            item["candidate_frontier_digest"],
        )
        for item in factual["frontiers"]
    )
    expected_frontiers = tuple(
        (
            int(item["decision_epoch"]),
            tuple(item["candidate_keys"]),
            item["candidate_frontier_digest"],
        )
        for item in source_audit["frontiers"]
    )
    _require_equal("factual exact frontiers", observed_frontiers, expected_frontiers)
    transitions = reconstruction["reconstruction"]["transitions"]
    _require_equal("factual transition count", len(factual["decisions"]), len(transitions))
    for observed, expected in zip(factual["decisions"], transitions):
        _require_equal("semantic transition epoch", observed["decision_epoch"], expected["decision_epoch"])
        _require_equal("semantic transition type", observed["action_type"], expected["action_type"])
        if observed["candidate_key"] not in expected["equivalent_candidate_keys"]:
            raise RuntimeError(
                "factual selected key is outside authenticated equivalent "
                f"transition set: {observed['candidate_key']}"
            )


def _validate_R_suffix(event: Mapping, factual: Mapping) -> None:
    R = event["branches"]["R"]
    index = int(event["decision_index"])
    suffix = factual["decisions"][index:]
    _require_equal(
        "R factual decision suffix",
        R["decision_keys"],
        tuple(item["candidate_key"] for item in suffix),
    )
    _require_equal(
        "R factual suffix steps",
        R["total_steps"],
        sum(int(item["duration"]) for item in suffix),
    )
    _require_equal(
        "R factual suffix relocations",
        R["total_physical_relocations"],
        sum(int(item["relocations"]) for item in suffix),
    )
    records = factual["objective_records"][
        int(suffix[0]["objective_record_start"]):
    ]
    totals = _objective_totals(records, factual["gamma"])
    _close("R factual dense suffix", R["total_dense_return_raw"], totals["dense_return_raw"])
    _close("R factual legacy suffix", R["total_legacy_return_raw"], totals["legacy_return_raw"])
    _close("R factual dense discounted suffix", R["total_dense_discounted_return"], totals["dense_discounted_return"])
    _close("R factual legacy discounted suffix", R["total_legacy_discounted_return"], totals["legacy_discounted_return"])
    _require_equal("R factual final state", R["final_state"], factual["final_state"])


def _csv_text(rows: Sequence[Mapping]) -> str:
    rows = tuple(rows)
    if not rows:
        return ""
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    _canonical_json(value)
                    if isinstance(value, (Mapping, tuple, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    return buffer.getvalue()


def _event_rows(events: Sequence[Mapping]):
    event_rows = []
    branch_rows = []
    for event in events:
        identity = {
            key: event[key]
            for key in (
                "event_id",
                "method_id",
                "model_seed",
                "instance_seed",
                "instance_id",
                "schedule_id",
                "event_index",
                "decision_index",
                "decision_epoch",
            )
        }
        identity["checkpoint_sha256"] = event["checkpoint_sha256"]
        identity["eligible"] = bool(event["eligible"])
        D = event["comparisons"]["D_Q_minus_R"]
        event_rows.append(
            {
                **identity,
                "factual_candidate_key": event["factual_candidate_key"],
                "directly_deliverable_self_reconfiguration": event[
                    "directly_deliverable_self_reconfiguration"
                ],
                "defer_branch_admissible": event["defer_branch_admissible"],
                "future_relocations_avoided_by_R_vs_D_Q": D[
                    "future_relocations_avoided_by_R"
                ],
                "total_relocations_avoided_by_R_vs_D_Q": D[
                    "total_relocations_avoided_by_R"
                ],
                "steps_avoided_by_R_vs_D_Q": D["steps_avoided_by_R"],
                "dense_return_advantage_for_R_vs_D_Q": D[
                    "dense_return_advantage_for_R"
                ],
                "dense_discounted_return_advantage_for_R_vs_D_Q": D[
                    "dense_discounted_return_advantage_for_R"
                ],
                "matched_MAE_improvement_for_R_vs_D_Q": D[
                    "mean_absolute_error_improvement_for_R"
                ],
            }
        )
        for branch_id, branch in event["branches"].items():
            branch_rows.append({**identity, **branch})
    return tuple(event_rows), tuple(branch_rows)


def _summary(events: Sequence[Mapping], *, complete_scope: bool) -> dict:
    comparisons = tuple(
        event["comparisons"]["D_Q_minus_R"] for event in events
    )

    def mean(name):
        values = [item[name] for item in comparisons if item.get(name) is not None]
        return float(fmean(float(value) for value in values)) if values else None

    return {
        "event_count": len(events),
        "directly_deliverable_self_event_count": sum(
            bool(item["directly_deliverable_self_reconfiguration"])
            for item in events
        ),
        "defer_branch_event_count": sum(
            bool(item["defer_branch_admissible"]) for item in events
        ),
        "strict_R_branch_rate": (
            float(fmean(item["branches"]["R"]["strict_method_success"] for item in events))
            if events
            else None
        ),
        "strict_D_Q_branch_rate": (
            float(fmean(item["branches"]["D_Q"]["strict_method_success"] for item in events))
            if events
            else None
        ),
        "timing_comparison_eligible_event_count": sum(
            bool(item["timing_comparison_eligible"])
            for item in comparisons
        ),
        "timing_comparison_ineligible_event_count": sum(
            not bool(item["timing_comparison_eligible"])
            for item in comparisons
        ),
        "productive_future_rehandle_event_count": sum(
            item["future_relocations_avoided_by_R"] >= 1
            for item in comparisons
        ),
        "mean_future_relocations_avoided_by_R": mean(
            "future_relocations_avoided_by_R"
        ),
        "mean_total_relocations_avoided_by_R": mean(
            "total_relocations_avoided_by_R"
        ),
        "mean_steps_avoided_by_R": mean("steps_avoided_by_R"),
        "mean_dense_return_advantage_for_R": mean(
            "dense_return_advantage_for_R"
        ),
        "mean_dense_discounted_return_advantage_for_R": mean(
            "dense_discounted_return_advantage_for_R"
        ),
        "mean_matched_MAE_improvement_for_R": mean(
            "mean_absolute_error_improvement_for_R"
        ),
        "complete_declared_scope": bool(complete_scope),
        "statistical_unit_warning": (
            "events are nested within crossed model-seed x EpisodeInstance "
            "trajectories and are not independent samples"
        ),
    }


def run_audit(args) -> dict:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    training_dirs = tuple(path.resolve() for path in args.training_dirs)
    if tuple(EVALUATION_SEEDS) != tuple(range(80_000, 80_030)):
        raise RuntimeError("development panel constant changed unexpectedly")
    bundles = tuple(_authenticate_training_bundle(path) for path in training_dirs)
    by_seed = {bundle.model_seed: bundle for bundle in bundles}
    if set(by_seed) != set(MODEL_SEEDS) or len(by_seed) != len(bundles):
        raise RuntimeError("runner requires selected-best model seeds 0/1/2")
    arms = tuple(by_seed[seed].selected for seed in MODEL_SEEDS)
    for arm in arms:
        _require_equal("selected policy group", arm.policy_group, SELECTED_BEST)
        _require_equal(
            f"model seed {arm.model_seed} selected checkpoint SHA",
            arm.checkpoint_sha256,
            EXPECTED_SELECTED_CHECKPOINT_SHA256[arm.model_seed],
        )
    payload = arms[0].payload
    source_chain = _authenticate_source_panel_chain(source_dir)
    instances, instance_manifest, instance_manifest_sha = _authenticate_instances(
        source_dir, payload
    )
    _require_equal(
        "source-chain instance manifest SHA",
        source_chain["artifact_sha256"]["instance-manifest.json"],
        instance_manifest_sha,
    )
    reanalysis, reanalysis_sha = _load_reanalysis_map(
        args.reanalysis_audit.resolve()
    )
    device = resolve_device(args.device)
    event_limit = args.max_events
    if event_limit is not None and event_limit <= 0:
        raise ValueError("--max-events must be positive")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    protocol_contract = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "development_only": True,
        "performance_claim_authorized": False,
        "source_protocol": SOURCE_PROTOCOL,
        "source_dir": str(source_dir),
        "source_artifact_sha256": source_chain["artifact_sha256"],
        "source_run_ledger_count": source_chain["run_ledger_count"],
        "source_run_ledger_manifest_digest": source_chain[
            "run_ledger_manifest_digest"
        ],
        "source_instance_manifest_sha256": instance_manifest_sha,
        "source_contention_reanalysis_audit": str(
            args.reanalysis_audit.resolve()
        ),
        "source_contention_reanalysis_sha256": reanalysis_sha,
        "evaluation_seeds": EVALUATION_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "checkpoint_sha256": EXPECTED_SELECTED_CHECKPOINT_SHA256,
        "branches": BRANCH_IDS,
        "D_Q": "highest_Q_local_post_guard_exact_SAFE_delivery_stable_key_tie",
        "D_SELF": (
            "highest_Q_local_post_guard_exact_SAFE_delivery_of_the_"
            "reconfigured_block_when_available"
        ),
        "W": "exact_post_guard_Defer_only_no_synthetic_WAIT",
        "continuation": "same_frozen_epsilon_zero_VCG_policy",
        "fresh_certificate_cache_per_branch": True,
        "state_authority": (
            "immutable_EpisodeInstance_prefix_replay_with_authenticated_"
            "deepcopy_branch_state"
        ),
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "max_steps": MAX_STEPS,
        "max_events": event_limit,
        "analysis_bootstrap_samples": int(args.bootstrap_samples),
        "analysis_bootstrap_seed": int(args.bootstrap_seed),
    }
    manifest_path = output_dir / "protocol-manifest.json"
    if manifest_path.is_file():
        _require_equal(
            "resume protocol manifest",
            _load_json(manifest_path),
            protocol_contract,
        )
    else:
        _atomic_json(manifest_path, protocol_contract)

    events = []
    source_runs = []
    eligible_seen = 0
    skipped_due_limit = 0
    stop_after_episode = False

    for arm in arms:
        search_config = _search_config(arm.payload)
        liveness_rule = _liveness_rule(arm.payload)
        for instance_seed in EVALUATION_SEEDS:
            instance, instance_record = instances[instance_seed]
            reconstruction = reanalysis[(arm.method_id, instance_seed)]
            source_ledger, source_ledger_sha = _source_ledger(
                source_dir,
                arm,
                instance_seed,
                instance_record,
                reconstruction,
            )
            env = _make_audit_env(arm.payload)
            env.current_episode = 1
            env.reset(instance=instance)
            agent = _freeze_agent(
                arm.payload,
                device=device,
                seed=arm.model_seed,
            )
            agent.reset_episode_state()
            cache = ViabilityCertificateCache()
            steps = 0
            consecutive = 0
            relocations = 0
            macro_failures = 0
            illegal_drops = 0
            method_failure_reason = None
            delivery_deviations = []
            decisions = []
            frontiers = []
            pending = None
            pending_record = None
            pending_events = []

            while steps < MAX_STEPS and not env.is_state_terminal(env.current_state):
                if pending is None:
                    pending, pending_record = _enumerate_frontier(
                        env,
                        consecutive_defer=consecutive,
                        search_config=search_config,
                        liveness_rule=liveness_rule,
                        cache=cache,
                        prioritizer=None,
                    )
                snapshot = pending
                frontier_record = pending_record
                pending = None
                pending_record = None
                frontiers.append(frontier_record)
                if not snapshot.candidates:
                    method_failure_reason = "no_exact_safe_candidate"
                    break
                guard_before = deepcopy(
                    agent.recovery_witness_guard.state_dict()
                )
                decision = agent.select(snapshot, training=False, epsilon=0.0)
                post_guard_candidates = tuple(
                    snapshot.candidates[source]
                    for source in decision.prepared_snapshot.source_indices
                )
                post_guard_deliveries = tuple(
                    candidate
                    for candidate in post_guard_candidates
                    if candidate.action_type is ViabilityActionType.DELIVER
                )
                potential = bool(
                    decision.candidate.action_type
                    is ViabilityActionType.RECONFIGURE
                    and not decision.liveness_forced
                    and post_guard_deliveries
                )
                pre_env = deepcopy(env) if potential else None
                pre_state_signature = (
                    _branch_state_signature(pre_env) if potential else None
                )
                pre_record_count = len(env.objective_audit_records)
                delivered_before = {
                    str(block.label) for block in env.blocks if block.delivered
                }
                execution = execute_certified_macro(
                    env,
                    decision.candidate,
                    gamma=agent.config.gamma,
                    remaining_steps=MAX_STEPS - steps,
                    evaluation=True,
                )
                decision_start_step = steps
                steps += int(execution.duration)
                relocations += int(execution.relocations)
                macro_failures += int(not execution.option_success)
                illegal_drops += int(execution.illegal_drops)
                delivery_deviations.extend(execution.delivery_deviations)
                consecutive = _update_consecutive_defer(
                    decision, execution, consecutive
                )
                boundary_done = bool(
                    execution.env_terminal
                    or execution.truncated
                    or not execution.option_success
                    or execution.duration == 0
                )
                if not boundary_done:
                    pending, pending_record = _enumerate_frontier(
                        env,
                        consecutive_defer=consecutive,
                        search_config=search_config,
                        liveness_rule=liveness_rule,
                        cache=cache,
                        prioritizer=None,
                    )
                    if not pending.candidates:
                        boundary_done = True
                        method_failure_reason = "no_exact_safe_candidate"
                        pending = None
                        pending_record = None
                agent.observe_outcome(
                    decision,
                    next_snapshot=None if boundary_done else pending,
                    done=boundary_done,
                )
                new_deliveries = _new_deliveries(env, delivered_before)
                decision_row = {
                    "decision_index": len(decisions),
                    "decision_epoch": int(snapshot.decision_epoch),
                    "candidate_key": decision.candidate.key,
                    "action_type": decision.candidate.action_type.value,
                    "target_label": decision.candidate.target_label,
                    "liveness_forced": bool(decision.liveness_forced),
                    "duration": int(execution.duration),
                    "relocations": int(execution.relocations),
                    "option_success": bool(execution.option_success),
                    "failure_reason": execution.failure_reason,
                    "objective_record_start": int(pre_record_count),
                    "objective_record_end": int(len(env.objective_audit_records)),
                    "decision_start_step": int(decision_start_step),
                    "new_deliveries": new_deliveries,
                }
                decisions.append(decision_row)

                eligible = bool(
                    potential
                    and execution.option_success
                    and execution.relocations == 1
                    and execution.illegal_drops == 0
                )
                if eligible:
                    eligible_seen += 1
                    if event_limit is not None and len(events) + len(pending_events) >= event_limit:
                        skipped_due_limit += 1
                        stop_after_episode = True
                    else:
                        prepared = decision.prepared_snapshot
                        q_values = decision.q_values
                        best_delivery_index = _stable_best_index(
                            snapshot,
                            prepared,
                            q_values,
                            ViabilityActionType.DELIVER,
                        )
                        if best_delivery_index is None:
                            raise RuntimeError("eligible event lost its delivery alternative")
                        best_delivery = snapshot.candidates[
                            prepared.source_indices[best_delivery_index]
                        ]
                        self_delivery_index = _stable_best_delivery_for_label(
                            snapshot,
                            prepared,
                            q_values,
                            str(decision.candidate.target_label),
                        )
                        self_delivery = (
                            None
                            if self_delivery_index is None
                            else snapshot.candidates[
                                prepared.source_indices[self_delivery_index]
                            ]
                        )
                        defer_index = _stable_best_index(
                            snapshot,
                            prepared,
                            q_values,
                            ViabilityActionType.DEFER,
                        )
                        defer_candidate = (
                            None
                            if defer_index is None
                            else snapshot.candidates[
                                prepared.source_indices[defer_index]
                            ]
                        )
                        event_index = len(events) + len(pending_events)
                        event_id = (
                            f"m{arm.model_seed}-s{instance_seed}-"
                            f"d{decision_row['decision_index']}-"
                            f"t{snapshot.decision_epoch}"
                        )
                        state_digest = _digest(pre_state_signature)
                        guard_digest = _digest(_guard_signature(guard_before))
                        input_contract = {
                            "protocol": PROTOCOL,
                            "branch_schema_version": BRANCH_SCHEMA_VERSION,
                            "event_id": event_id,
                            "method_id": arm.method_id,
                            "model_seed": arm.model_seed,
                            "checkpoint_sha256": arm.checkpoint_sha256,
                            "deployment_policy_digest": arm.deployment_policy_digest,
                            "instance_seed": instance_seed,
                            "instance_id": instance.instance_id,
                            "schedule_id": instance.schedule_id,
                            "instance_sha256": instance_record["sha256"],
                            "source_ledger_sha256": source_ledger_sha,
                            "decision_index": decision_row["decision_index"],
                            "decision_epoch": int(snapshot.decision_epoch),
                            "factual_candidate_key": decision.candidate.key,
                            "D_Q_candidate_key": best_delivery.key,
                            "D_SELF_candidate_key": (
                                None if self_delivery is None else self_delivery.key
                            ),
                            "W_candidate_key": (
                                None if defer_candidate is None else defer_candidate.key
                            ),
                            "candidate_frontier_digest": frontier_record[
                                "candidate_frontier_digest"
                            ],
                            "branch_state_digest": state_digest,
                            "guard_state_digest": guard_digest,
                            "consecutive_defer": int(
                                snapshot.audit.consecutive_defer_decisions
                            ),
                            "episode_steps_before": int(decision_start_step),
                            "max_steps": MAX_STEPS,
                            "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
                        }
                        ledger_path = _event_ledger_path(
                            output_dir,
                            arm.method_id,
                            instance_seed,
                            decision_row["decision_index"],
                        )
                        fingerprint = _digest(input_contract)
                        if ledger_path.is_file():
                            ledger_record = _load_json(ledger_path)
                            _require_equal(
                                "event resume input contract",
                                ledger_record.get("input_contract"),
                                input_contract,
                            )
                            _require_equal(
                                "event resume fingerprint",
                                ledger_record.get("input_fingerprint"),
                                fingerprint,
                            )
                            event = ledger_record["event"]
                            pending_events.append((event, None, None))
                            if (
                                event_limit is not None
                                and len(events) + len(pending_events) >= event_limit
                            ):
                                stop_after_episode = True
                        else:
                            branches = {}
                            candidates = {
                                "R": decision.candidate.key,
                                "D_Q": best_delivery.key,
                            }
                            self_aliases_D_Q = bool(
                                self_delivery is not None
                                and self_delivery.key == best_delivery.key
                            )
                            if self_delivery is not None and not self_aliases_D_Q:
                                candidates["D_SELF"] = self_delivery.key
                            if defer_candidate is not None:
                                candidates["W"] = defer_candidate.key
                            for branch_id, candidate_key in candidates.items():
                                print(
                                    f"[{arm.method_id}] seed={instance_seed} "
                                    f"event={event_id} branch={branch_id}",
                                    flush=True,
                                )
                                branch = _run_branch(
                                    branch_id=branch_id,
                                    initial_candidate_key=candidate_key,
                                    pre_env=pre_env,
                                    pre_guard_state=guard_before,
                                    consecutive_defer=int(
                                        snapshot.audit.consecutive_defer_decisions
                                    ),
                                    episode_steps_before=decision_start_step,
                                    controller_payload=arm.payload,
                                    search_config=search_config,
                                    liveness_rule=liveness_rule,
                                    expected_frontier_digest=frontier_record[
                                        "candidate_frontier_digest"
                                    ],
                                    expected_state_digest=state_digest,
                                    expected_guard_digest=guard_digest,
                                    device=device,
                                )
                                branches[branch_id] = branch
                            if self_aliases_D_Q:
                                branches["D_SELF"] = {
                                    **deepcopy(branches["D_Q"]),
                                    "branch_id": "D_SELF",
                                    "execution_reused": True,
                                    "derived_from_branch_id": "D_Q",
                                }
                            directly_deliverable = bool(
                                decision.candidate.target_label
                                in {
                                    candidate.target_label
                                    for candidate in post_guard_deliveries
                                }
                            )
                            comparisons = {
                                "D_Q_minus_R": _branch_comparison(
                                    branches["R"], branches["D_Q"]
                                )
                            }
                            if "W" in branches:
                                comparisons["W_minus_R"] = _branch_comparison(
                                    branches["R"], branches["W"]
                                )
                            if "D_SELF" in branches:
                                comparisons["D_SELF_minus_R"] = _branch_comparison(
                                    branches["R"], branches["D_SELF"]
                                )
                            event = {
                                "event_id": event_id,
                                "method_id": arm.method_id,
                                "model_seed": arm.model_seed,
                                "checkpoint_sha256": arm.checkpoint_sha256,
                                "instance_seed": instance_seed,
                                "instance_id": instance.instance_id,
                                "schedule_id": instance.schedule_id,
                                "event_index": event_index,
                                "decision_index": decision_row["decision_index"],
                                "decision_epoch": int(snapshot.decision_epoch),
                                "eligible": True,
                                "factual_candidate_key": decision.candidate.key,
                                "factual_target_label": decision.candidate.target_label,
                                "factual_source": decision.candidate.source,
                                "factual_destination": decision.candidate.destination,
                                "directly_deliverable_self_reconfiguration": directly_deliverable,
                                "post_guard_delivery_candidate_count": len(post_guard_deliveries),
                                "D_Q_candidate_key": best_delivery.key,
                                "D_Q_target_label": best_delivery.target_label,
                                "D_SELF_candidate_key": (
                                    None if self_delivery is None else self_delivery.key
                                ),
                                "defer_branch_admissible": defer_candidate is not None,
                                "W_candidate_key": (
                                    None if defer_candidate is None else defer_candidate.key
                                ),
                                "no_synthetic_wait": True,
                                "branches": branches,
                                "comparisons": comparisons,
                                "input_fingerprint": fingerprint,
                            }
                            ledger_record = {
                                "protocol": PROTOCOL,
                                "input_contract": input_contract,
                                "input_fingerprint": fingerprint,
                                "event": event,
                            }
                            pending_events.append(
                                (event, ledger_path, ledger_record)
                            )
                            if (
                                event_limit is not None
                                and len(events) + len(pending_events) >= event_limit
                            ):
                                stop_after_episode = True

                if not execution.option_success and method_failure_reason is None:
                    method_failure_reason = (
                        f"macro_failure:{execution.action_type}:"
                        f"{execution.failure_reason or 'unknown'}"
                    )
                if boundary_done:
                    break

            objective_records = env.objective_audit_records
            objective_summary = env.objective_audit_summary()
            factual = {
                "method_id": arm.method_id,
                "instance_seed": instance_seed,
                "steps": steps,
                "legacy_return": objective_summary["return_by_objective"][
                    LEGACY_CLIPPED
                ],
                "dense_return": objective_summary["return_by_objective"][
                    DENSE_PIECEWISE
                ],
                "gamma": float(agent.config.gamma),
                "delivery_deviations": tuple(delivery_deviations),
                "relocations": relocations,
                "macro_failures": macro_failures,
                "illegal_drops": illegal_drops,
                "method_failure_reason": method_failure_reason,
                "decisions": tuple(decisions),
                "frontiers": tuple(frontiers),
                "objective_records": objective_records,
                "final_state": _environment_signature(env),
            }
            _validate_source_factual(factual, source_ledger, reconstruction)
            for event, ledger_path, ledger_record in pending_events:
                _validate_R_suffix(event, factual)
                event["R_factual_continuation_reproduced"] = True
                if ledger_record is not None:
                    ledger_record["event"] = event
                    _atomic_json(ledger_path, ledger_record)
                events.append(event)
            # Existing transactional events were loaded directly rather than
            # queued for a write; validate them here as well.
            existing_for_run = [
                event
                for event in events
                if event["method_id"] == arm.method_id
                and int(event["instance_seed"]) == instance_seed
            ]
            for event in existing_for_run:
                _validate_R_suffix(event, factual)
            source_runs.append(
                {
                    "method_id": arm.method_id,
                    "model_seed": arm.model_seed,
                    "instance_seed": instance_seed,
                    "instance_id": instance.instance_id,
                    "source_ledger_sha256": source_ledger_sha,
                    "factual_prefix_replay_authenticated": True,
                }
            )
            print(
                f"[factual] {arm.method_id} seed={instance_seed} "
                f"R={factual['dense_return']:.2f} reloc={relocations} "
                f"events_total={len(events)}",
                flush=True,
            )
            if stop_after_episode:
                break
        if stop_after_episode:
            break

    complete_scope = bool(
        event_limit is None
        and len(source_runs) == len(MODEL_SEEDS) * len(EVALUATION_SEEDS)
    )
    if complete_scope:
        _require_equal(
            "full primary event count",
            len(events),
            EXPECTED_FULL_PRIMARY_EVENT_COUNT,
        )
        _require_equal(
            "full direct-self event count",
            sum(
                bool(event["directly_deliverable_self_reconfiguration"])
                for event in events
            ),
            EXPECTED_FULL_DIRECT_SELF_EVENT_COUNT,
        )
    event_rows, branch_rows = _event_rows(events)
    summary = _summary(events, complete_scope=complete_scope)
    if complete_scope:
        analysis_report = build_relocation_amortization_report(
            branch_rows,
            target_window=FROZEN_OBJECTIVE_SPEC.window,
            expected_model_seeds=MODEL_SEEDS,
            bootstrap_samples=args.bootstrap_samples,
            rng_seed=args.bootstrap_seed,
        )
    else:
        smoke_validation = validate_branch_rows(
            branch_rows,
            target_window=FROZEN_OBJECTIVE_SPEC.window,
            allow_partial_smoke=True,
        )
        analysis_report = {
            "analysis_performed": False,
            "reason": (
                "bounded smoke does not expose all three fixed model seeds; "
                "the three-seed clustered estimand is intentionally withheld"
            ),
            "events_validated_by_runner": len(events),
            "smoke_schema_validation": smoke_validation,
        }
    audit = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "development_only": True,
        "performance_claim_authorized": False,
        "sealed_or_prospective_panels_opened": False,
        "new_episode_instances_generated": 0,
        "source_instance_manifest_sha256": instance_manifest_sha,
        "source_contention_reanalysis_sha256": reanalysis_sha,
        "source_runs_authenticated": len(source_runs),
        "expected_source_runs_for_full_scope": len(MODEL_SEEDS)
        * len(EVALUATION_SEEDS),
        "eligible_events_seen": eligible_seen,
        "events_audited": len(events),
        "events_skipped_due_limit": skipped_due_limit,
        "complete_declared_scope": complete_scope,
        "all_R_factual_continuations_reproduced": all(
            bool(event.get("R_factual_continuation_reproduced"))
            for event in events
        ),
        "all_branches_fresh_cache": all(
            branch["fresh_certificate_cache"]
            for event in events
            for branch in event["branches"].values()
        ),
        "all_branches_exact_safe": all(
            branch["all_candidates_exact_safe"]
            for event in events
            for branch in event["branches"].values()
        ),
        "no_synthetic_wait": all(event["no_synthetic_wait"] for event in events),
        "source_runs": tuple(source_runs),
        "source_artifact_sha256": source_chain["artifact_sha256"],
        "source_run_ledger_count": source_chain["run_ledger_count"],
        "source_run_ledger_manifest_digest": source_chain[
            "run_ledger_manifest_digest"
        ],
        "source_run_ledger_manifest": source_chain["run_ledger_manifest"],
        "protocol_contract": protocol_contract,
    }
    report = {
        "protocol": PROTOCOL,
        "scope": (
            "complete_development_panel"
            if complete_scope
            else "bounded_development_smoke"
        ),
        "summary": summary,
        "interpretation": {
            "positive_delta_direction": "positive values favor factual R",
            "future_relocations_avoided": (
                "D_Q future-only physical relocations minus R future-only "
                "physical relocations; R's immediate relocation is excluded"
            ),
            "total_relocations_avoided": (
                "D_Q total physical relocations minus R total physical "
                "relocations; R's immediate relocation is included"
            ),
            "causal_scope": (
                "one forced first exact-SAFE macro, followed by the same frozen "
                "greedy VCG continuation"
            ),
            "defer_scope": (
                "reported only when exact Defer survives the liveness guard"
            ),
        },
        "performance_claim_authorized": False,
    }
    _atomic_json(output_dir / "events.json", {"events": events})
    _atomic_json(output_dir / "branches.json", {"branches": branch_rows})
    _atomic_text(output_dir / "events.csv", _csv_text(event_rows))
    _atomic_text(output_dir / "branches.csv", _csv_text(branch_rows))
    _atomic_json(output_dir / "report.json", report)
    _atomic_json(output_dir / "analysis-report.json", analysis_report)
    _atomic_json(output_dir / "audit.json", audit)
    return {
        "summary": summary,
        "complete_declared_scope": complete_scope,
        "events_path": str((output_dir / "events.json").resolve()),
        "branches_path": str((output_dir / "branches.json").resolve()),
        "report_path": str((output_dir / "report.json").resolve()),
        "analysis_report_path": str(
            (output_dir / "analysis-report.json").resolve()
        ),
        "audit_path": str((output_dir / "audit.json").resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dirs",
        nargs=3,
        type=Path,
        default=TRAINING_DIR_DEFAULTS,
    )
    parser.add_argument(
        "--source-dir", type=Path, default=SOURCE_DIR_DEFAULT
    )
    parser.add_argument(
        "--reanalysis-audit", type=Path, default=REANALYSIS_AUDIT_DEFAULT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help=(
            "Development smoke bound. Use 5 with a separate output directory; "
            "omit for the complete fixed development panel."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_808)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_audit(args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
