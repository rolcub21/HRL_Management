#!/usr/bin/env python3
"""Full-dynamic 5x5 recovery filtering with one carried primitive budget.

This additive v2 runner keeps the v1 frozen policies, EpisodeInstances, live
executor, disturbance family, and selector fixed.  Its scientific correction
is that a finite closed-admission recovery budget belongs to a *recovery
segment*: after the frozen selector first chooses recovery, the current stored
block cohort and budget are committed until that cohort is empty.  The budget
is carried across live replanning boundaries and charged for every recovery
macro.  Accept and Defer are masked while a segment is active; outside a
segment they remain under the nominal live frontier.

A successful recovery execution must have exactly the strict RecoveryAction
duration plus its requested injected primitives.
The declared disturbed physical successor is checked immediately and again at
the next enumerated live boundary, ignoring only block ``remaining_time``.

This is still a recovery-only dynamic stress experiment.  Accept and Defer do
not acquire a recursive robust-viability theorem, and future arrivals remain
outside the closed recovery abstraction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.viability import RecoveryAction, RecoveryState, ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCertificateCache,
)
from PSLAP.viability_filter import online_fixed_obstacles
import run_vcg_dynamic_robust_filter_panel_5x5 as v1
from vcg_exact_recovery_execution import (
    EXACT_RECOVERY_EXECUTION_CONTRACT,
    exact_execution_to_dict,
    execute_exact_bounded_recovery_realization,
)
from vcg_robust_recovery_snapshot_5x5 import (
    ROBUST_CONTRACT,
    SolveStatus,
    canonical_actions,
    certify_snapshot,
    execution_envelope,
    method_to_dict,
    recovery_action_key,
)


PROTOCOL = "vcg_5x5_full_dynamic_carried_budget_recovery_filter_panel_v2"
SCHEMA_VERSION = 2
METHODS = v1.METHODS
HANDLING_LAMBDAS = v1.HANDLING_LAMBDAS
MODEL_SEED = v1.MODEL_SEED
EXPECTED_BLOCKS = v1.EXPECTED_BLOCKS
MAX_EXPANSIONS = v1.MAX_EXPANSIONS
BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO = (
    v1.BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO
)
PILOT_SEEDS = v1.PILOT_SEEDS
FULL_SEEDS = v1.FULL_SEEDS

HERE = Path(__file__).resolve().parent
DEFAULT_PILOT_OUTPUT = (
    HERE / "results/vcg-dynamic-budgeted-robust-filter-panel-5x5-pilot-v2"
)
DEFAULT_FULL_OUTPUT = (
    HERE / "results/vcg-dynamic-budgeted-robust-filter-panel-5x5-v2"
)
CONTRACT_NAME = "dynamic-budgeted-filter-contract.json"
REPORT_NAME = "dynamic-budgeted-filter-report.json"

TIMING_METRIC_FIELDS = (
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "tardy_delivery_rate",
    "mean_tardiness_when_tardy",
    "p90_tardiness",
    "p90_absolute_error",
    "signed_deviation_std",
    "absolute_error_std",
)


class BudgetedDynamicFilterError(v1.DynamicFilterError):
    pass


def _nullable_timing_summary(deviations: Sequence[float]) -> dict:
    summary = dict(
        v1.summarize_delivery_timing(
            deviations,
            v1.v11.FROZEN_OBJECTIVE_SPEC.window,
        )
    )
    for key in TIMING_METRIC_FIELDS:
        value = summary.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            summary[key] = None
    if not deviations and any(summary[key] is not None for key in TIMING_METRIC_FIELDS):
        raise BudgetedDynamicFilterError(
            "zero-delivery timing metrics must all be undefined"
        )
    return summary


@dataclass(frozen=True)
class PendingPhysicalSuccessor:
    decision_index: int
    candidate_key: str
    expected_key: Mapping[str, object]


@dataclass
class CommittedRecoverySegment:
    """Episode-local budget for one theorem-aligned committed stored cohort."""

    next_epoch_index: int = 0
    epoch_index: Optional[int] = None
    initial_budget: Optional[int] = None
    remaining_budget: Optional[int] = None
    initial_workload_labels: tuple[str, ...] = ()
    current_workload_labels: tuple[str, ...] = ()
    activation_count: int = 0
    total_consumed_steps: int = 0
    completion_count: int = 0

    @property
    def active(self) -> bool:
        return self.remaining_budget is not None

    def boundary(self, state: RecoveryState) -> dict:
        labels = tuple(sorted(block.label for block in state.blocks))
        if len(labels) != len(set(labels)):
            raise BudgetedDynamicFilterError(
                "recovery boundary contains duplicate block labels"
            )
        if not self.active:
            return {
                "event": "inactive_boundary",
                "active": False,
                "segment_index": None,
                "initial_budget": None,
                "remaining_budget": None,
                "cohort_labels": [],
                "observed_recovery_labels": list(labels),
            }
        if not labels:
            raise BudgetedDynamicFilterError(
                "active segment reached an empty boundary before explicit close"
            )
        if not set(labels).issubset(self.current_workload_labels):
            raise BudgetedDynamicFilterError(
                "committed recovery cohort gained an unexpected stored label"
            )
        self.current_workload_labels = labels
        return {
            "event": "active_continuity_verified",
            "active": True,
            "segment_index": self.epoch_index,
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "cohort_labels": list(self.initial_workload_labels),
            "observed_recovery_labels": list(labels),
        }

    def activate(
        self,
        state: RecoveryState,
        *,
        primitive_budget: int,
        nominal_witness_primitive_steps: int,
        nominal_witness_macro_count: int,
        remaining_episode_horizon: int,
    ) -> dict:
        if self.active:
            raise BudgetedDynamicFilterError("cannot activate an active segment")
        labels = tuple(sorted(block.label for block in state.blocks))
        if not labels:
            raise BudgetedDynamicFilterError("cannot commit an empty recovery cohort")
        if (
            isinstance(primitive_budget, bool)
            or not isinstance(primitive_budget, int)
            or primitive_budget < 1
        ):
            raise BudgetedDynamicFilterError(
                "committed recovery segment requires positive finite budget"
            )
        budget = int(
            nominal_witness_primitive_steps
            + BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO
            * nominal_witness_macro_count
        )
        if primitive_budget != min(budget, remaining_episode_horizon):
            raise BudgetedDynamicFilterError("segment activation budget changed")
        self.epoch_index = self.next_epoch_index
        self.next_epoch_index += 1
        self.initial_budget = primitive_budget
        self.remaining_budget = primitive_budget
        self.initial_workload_labels = labels
        self.current_workload_labels = labels
        self.activation_count += 1
        return {
            "event": "committed_recovery_segment_activated",
            "active": True,
            "segment_index": self.epoch_index,
            "initial_budget": primitive_budget,
            "remaining_budget": primitive_budget,
            "cohort_labels": list(labels),
            "nominal_witness_primitive_steps": nominal_witness_primitive_steps,
            "nominal_witness_macro_count": nominal_witness_macro_count,
            "remaining_episode_horizon": remaining_episode_horizon,
            "uncapped_budget": budget,
            "budget_rule": "min(witness_steps+3*witness_macros,live_horizon)",
        }

    def close_completed(self) -> dict:
        if not self.active:
            raise BudgetedDynamicFilterError("cannot close inactive segment")
        audit = {
            "event": "committed_recovery_segment_completed",
            "completed_segment_index": self.epoch_index,
            "discarded_leftover_budget": self.remaining_budget,
            "initial_budget": self.initial_budget,
            "cohort_labels": list(self.initial_workload_labels),
        }
        self.completion_count += 1
        self.epoch_index = None
        self.initial_budget = None
        self.remaining_budget = None
        self.initial_workload_labels = ()
        self.current_workload_labels = ()
        return audit

    def consume(self, duration: int, *, action_type: str) -> dict:
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            raise TypeError("carried-budget duration must be non-negative int")
        if not self.active:
            return {
                "charged": False,
                "action_type": action_type,
                "duration": duration,
                "before": None,
                "after": None,
            }
        before = int(self.remaining_budget)
        if duration > before:
            raise BudgetedDynamicFilterError(
                "executed recovery macro exceeded carried segment budget"
            )
        self.remaining_budget = before - duration
        self.total_consumed_steps += duration
        return {
            "charged": True,
            "action_type": action_type,
            "duration": duration,
            "before": before,
            "after": self.remaining_budget,
            "segment_index": self.epoch_index,
        }



def _timing_erased_state(state: RecoveryState) -> RecoveryState:
    """Project clocks out of the theorem-facing recovery state."""

    if not isinstance(state, RecoveryState):
        raise TypeError("timing erasure requires a RecoveryState")
    return replace(
        state,
        blocks=tuple(replace(block, remaining_time=0.0) for block in state.blocks),
    )


def _provisional_budget(snapshot, *, remaining_episode_horizon: int) -> dict:
    state = _timing_erased_state(snapshot.recovery_state)
    if not state.blocks:
        return {
            "status": "NOT_APPLICABLE_EMPTY_RECOVERY_STATE",
            "primitive_budget": None,
            "nominal_witness_primitive_steps": None,
            "nominal_witness_macro_count": None,
            "remaining_episode_horizon": remaining_episode_horizon,
        }
    current = snapshot.current_certificate
    steps = current.witness_primitive_steps
    macros = current.witness_macro_count
    valid = bool(
        current.status is ViabilityStatus.SAFE
        and not isinstance(steps, bool)
        and isinstance(steps, int)
        and steps >= 1
        and not isinstance(macros, bool)
        and isinstance(macros, int)
        and macros >= 1
        and remaining_episode_horizon >= 1
    )
    if not valid:
        return {
            "status": "UNKNOWN_NO_FINITE_CURRENT_NOMINAL_WITNESS",
            "primitive_budget": None,
            "nominal_witness_primitive_steps": steps,
            "nominal_witness_macro_count": macros,
            "remaining_episode_horizon": remaining_episode_horizon,
        }
    uncapped = int(
        steps + BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO * macros
    )
    return {
        "status": "FINITE_PROVISIONAL_BUDGET",
        "primitive_budget": min(uncapped, int(remaining_episode_horizon)),
        "uncapped_budget": uncapped,
        "nominal_witness_primitive_steps": int(steps),
        "nominal_witness_macro_count": int(macros),
        "remaining_episode_horizon": int(remaining_episode_horizon),
        "timing_erased": True,
    }


def _panel_seeds(panel: str) -> tuple[int, ...]:
    return v1._panel_seeds(panel)


def _default_output(panel: str) -> Path:
    return DEFAULT_PILOT_OUTPUT if panel == "pilot" else DEFAULT_FULL_OUTPUT


def _lambda_key(value: float) -> str:
    return v1._lambda_key(value)


def _recovery_physical_key(state: RecoveryState) -> dict:
    """Physical RecoveryState identity with only block clocks projected out."""

    if not isinstance(state, RecoveryState):
        raise TypeError("physical key requires a RecoveryState")
    return {
        "rows": int(state.rows),
        "cols": int(state.cols),
        "traversable": [list(cell) for cell in sorted(state.traversable)],
        "storage_cells": [list(cell) for cell in sorted(state.storage_cells)],
        "exits": [list(cell) for cell in state.exits],
        "blocks": [
            {"label": block.label, "position": list(block.position)}
            for block in sorted(state.blocks, key=lambda item: item.label)
        ],
        "agent_position": list(state.agent_position),
        "fixed_obstacles": [list(cell) for cell in sorted(state.fixed_obstacles)],
        "reserved_cells": [list(cell) for cell in sorted(state.reserved_cells)],
        "pickup_cells": [list(cell) for cell in sorted(state.pickup_cells)],
        "wait_cells": [list(cell) for cell in sorted(state.wait_cells)],
    }


def _physical_comparison(expected: Mapping, actual: Mapping) -> dict:
    fields = tuple(expected)
    if set(actual) != set(expected):
        raise BudgetedDynamicFilterError("physical key fields changed")
    mismatched = [field for field in fields if actual[field] != expected[field]]
    return {
        "matched": not mismatched,
        "ignored_fields": ["blocks[].remaining_time"],
        "mismatched_fields": mismatched,
        "expected_key": dict(expected),
        "actual_key": dict(actual),
    }


def _live_recovery_state(env, *, template: RecoveryState, search_config) -> RecoveryState:
    yard = YardSnapshot.from_env(env)
    agent_position = tuple(env.current_state)
    fixed = online_fixed_obstacles(
        env,
        reserve_queue_cells=search_config.reserve_queue_cells,
    ) - {agent_position}
    return _timing_erased_state(RecoveryState.from_yard_snapshot(
        yard,
        agent_position,
        fixed_obstacles=fixed,
        reserved_cells=template.reserved_cells,
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    ))


def _action_to_dict(action: RecoveryAction) -> dict:
    return {
        "kind": action.kind.value,
        "block_label": action.block_label,
        "source": list(action.source),
        "destination": list(action.destination),
        "approach_path": [list(cell) for cell in action.approach_path],
        "transport_path": [list(cell) for cell in action.transport_path],
        "steps": int(action.steps),
    }


def _authenticated_recovery_actions(snapshot) -> tuple[RecoveryState, dict[str, RecoveryAction]]:
    """Bind every live recovery candidate to the identical canonical action."""

    state = _timing_erased_state(snapshot.recovery_state)
    canonical = {recovery_action_key(action): action for action in canonical_actions(state)}
    if len(canonical) != len(canonical_actions(state)):
        raise BudgetedDynamicFilterError("canonical recovery action keys collided")
    authenticated = {}
    for candidate in snapshot.candidates:
        if candidate.action_type not in (
            ViabilityActionType.DELIVER,
            ViabilityActionType.RECONFIGURE,
        ):
            continue
        action = candidate.recovery_action
        if action is None:
            raise BudgetedDynamicFilterError(
                "live recovery candidate has no bound RecoveryAction"
            )
        expected = canonical.get(candidate.key)
        if expected is None:
            raise BudgetedDynamicFilterError(
                "live recovery action is absent from canonical solver actions"
            )
        if action != expected:
            raise BudgetedDynamicFilterError(
                "live RecoveryAction differs from canonical solver action"
            )
        if candidate.key != recovery_action_key(action):
            raise BudgetedDynamicFilterError(
                "live recovery candidate key disagrees with full action"
            )
        authenticated[candidate.key] = expected
    return state, authenticated


def _declared_recovery_outcome(
    state: RecoveryState,
    action: RecoveryAction,
    *,
    realization,
    primitive_budget: int,
) -> dict:
    outcomes = execution_envelope(
        state,
        action,
        remaining_primitive_steps=primitive_budget,
        contract=ROBUST_CONTRACT,
    )
    stop = (
        tuple(action.destination)
        if realization.adjacent_stop is None
        else tuple(realization.adjacent_stop)
    )
    matches = tuple(
        item
        for item in outcomes
        if item.delay_steps == realization.delay_steps and item.stop_cell == stop
    )
    if len(matches) != 1:
        raise BudgetedDynamicFilterError(
            "requested live realization is absent or duplicated in declared envelope"
        )
    outcome = matches[0]
    result = {
        "disturbance_id": outcome.disturbance_id,
        "delay_steps": outcome.delay_steps,
        "stop_cell": list(outcome.stop_cell),
        "stop_extra_steps": outcome.stop_extra_steps,
        "primitive_steps": outcome.primitive_steps,
        "horizon_safe": outcome.horizon_safe,
        "tube_safe": outcome.tube_safe,
        "nominal": outcome.nominal,
        "expected_physical_key": (
            None
            if outcome.successor is None
            else _recovery_physical_key(outcome.successor)
        ),
    }
    return result


def _filter_recovery_frontier(
    snapshot,
    *,
    method: str,
    primitive_budget: Optional[int],
    segment_active: bool,
    memo: dict,
    memo_namespace: int,
) -> tuple[frozenset[str], dict]:
    """Filter at the carried budget; never derive a new local horizon."""

    if method not in METHODS:
        raise ValueError(f"unknown filter method: {method}")
    timing_erased_state, authenticated_actions = _authenticated_recovery_actions(
        snapshot
    )
    recovery = tuple(
        candidate
        for candidate in snapshot.candidates
        if candidate.action_type
        in (ViabilityActionType.DELIVER, ViabilityActionType.RECONFIGURE)
    )
    accept_keys = tuple(
        item.key
        for item in snapshot.candidates
        if item.action_type is ViabilityActionType.ACCEPT
    )
    defer_candidates = tuple(
        item
        for item in snapshot.candidates
        if item.action_type is ViabilityActionType.DEFER
    )
    masked_nonrecovery = (
        accept_keys + tuple(item.key for item in defer_candidates)
        if segment_active
        else ()
    )
    pass_through = (
        ()
        if segment_active
        else accept_keys + tuple(item.key for item in defer_candidates)
    )

    common = {
        "method": method,
        "primitive_budget": primitive_budget,
        "budget_source": (
            "carried_committed_segment"
            if segment_active
            else "provisional_precommitment"
        ),
        "segment_active": bool(segment_active),
        "timing_erased_recovery_state": True,
        "original_recovery_keys": [item.key for item in recovery],
        "pass_through_keys": list(pass_through),
        "masked_nonrecovery_keys": list(masked_nonrecovery),
        "authenticated_recovery_actions": {
            key: _action_to_dict(action)
            for key, action in sorted(authenticated_actions.items())
        },
        "full_recovery_action_equality_authenticated": True,
    }
    if not recovery:
        return frozenset(pass_through), {
            **common,
            "status": (
                "ACTIVE_NO_RECOVERY_CANDIDATE"
                if segment_active
                else "NOT_APPLICABLE_NO_RECOVERY_CANDIDATE"
            ),
            "admitted_recovery_keys": [],
            "rejected_recovery_keys": [],
            "certification_wall_seconds": 0.0,
            "memo_hit": False,
        }
    if (
        isinstance(primitive_budget, bool)
        or not isinstance(primitive_budget, int)
        or primitive_budget < 0
    ):
        return frozenset(pass_through), {
            **common,
            "status": "UNKNOWN_NO_ACTIVE_CARRIED_RECOVERY_BUDGET",
            "admitted_recovery_keys": [],
            "rejected_recovery_keys": [item.key for item in recovery],
            "certification_wall_seconds": 0.0,
            "memo_hit": False,
        }

    cache_key = (int(memo_namespace), timing_erased_state, primitive_budget)
    started = perf_counter()
    memo_hit = cache_key in memo
    if memo_hit:
        certificate = memo[cache_key]
    else:
        certificate = certify_snapshot(
            timing_erased_state,
            primitive_budget=primitive_budget,
            max_expansions=MAX_EXPANSIONS,
        )
        memo[cache_key] = certificate
    elapsed = perf_counter() - started
    method_certificate = getattr(certificate, method)
    solver_admitted = frozenset(method_certificate.admitted_action_keys)
    original = frozenset(item.key for item in recovery)
    admitted_recovery = solver_admitted & original
    admitted = frozenset(pass_through) | admitted_recovery
    solver_status = method_certificate.state.status.value
    if admitted_recovery:
        live_status = solver_status
    elif solver_status == SolveStatus.UNKNOWN.value:
        live_status = SolveStatus.UNKNOWN.value
    else:
        live_status = "NO_EXECUTABLE_ADMITTED_RECOVERY"
    return admitted, {
        **common,
        "status": live_status,
        "solver_state_status": solver_status,
        "semantic_digest": certificate.semantic_digest,
        "solver_admitted_recovery_keys": sorted(solver_admitted),
        "admitted_recovery_keys": sorted(admitted_recovery),
        "rejected_recovery_keys": sorted(original - admitted_recovery),
        "method_certificate": method_to_dict(method_certificate),
        "certification_wall_seconds": float(elapsed),
        "memo_hit": memo_hit,
    }


def _v2_semantic_source_sha256() -> dict:
    paths = {
        "exact_recovery_executor": HERE / "vcg_exact_recovery_execution.py",
        "macro_execution_core": HERE / "train_viability_graph_smdp.py",
        "frozen_selector_hierarchy": HERE / "viability_graph_hierarchy.py",
        "recovery_model": HERE / "PSLAP/viability.py",
        "live_candidate_interface": HERE / "PSLAP/viability_candidates.py",
        "live_viability_filter": HERE / "PSLAP/viability_filter.py",
        "dynamic_yard_projection": HERE / "PSLAP/dynamic_yard.py",
        "reported_timing_metrics": HERE / "example/helper/timing_metrics.py",
    }
    return {name: v1._sha(path.resolve()) for name, path in paths.items()}


def _contract(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
) -> dict:
    runtime_versions = v1.final86._configure_runtime()
    # Reuse the authenticated source/grid description, then replace every
    # v1-specific scientific semantic with this independently hashed v2.
    payload = json.loads(
        json.dumps(
            v1._json_safe(
                v1._contract(
                    project_root,
                    output_dir,
                    panel=panel,
                    device_name=device_name,
                )
            )
        )
    )
    payload.pop("contract_sha256", None)
    payload["schema_version"] = SCHEMA_VERSION
    payload["protocol"] = PROTOCOL
    payload["scientific_role"] = (
        "dynamic_committed_stored_cohort_recursive_completion_stress"
    )
    payload["runtime_versions"] = runtime_versions
    payload["recovery_filter"].update(
        {
            "replanned_at_every_live_decision_boundary": True,
            "unknown_fails_closed": True,
            "max_expansions_per_method_per_decision": MAX_EXPANSIONS,
            "timing_state_projection": (
                "all_BlockView_remaining_time_canonicalized_to_zero"
            ),
            "timing_metrics_outside_robust_state": True,
            "inactive_budget_formula": (
                "min(current_nominal_witness_primitive_steps_plus_3_times_"
                "current_nominal_witness_macro_count,remaining_live_episode_"
                "horizon)"
            ),
            "budget_formula": (
                "inactive_min(witness_steps+3*witness_macros,live_horizon)_"
                "then_carried_without_replenishment_after_recovery_commitment"
            ),
            "budget_role": (
                "finite_primitive_horizon_for_one_committed_stored_cohort"
            ),
            "active_budget_semantics": (
                "frozen_at_first_selected_recovery_then_decremented_by_each_"
                "exact_realized_recovery_macro_duration_without_replenishment"
            ),
            "active_action_mask": "deliver_and_reconfigure_only",
            "segment_completion": (
                "committed_stored_label_cohort_empty_then_discard_leftover_"
                "and_reopen_nominal_admission_next_boundary"
            ),
            "recovery_action_authentication": (
                "full_dataclass_equality_kind_label_source_destination_"
                "approach_path_transport_path_and_steps"
            ),
        }
    )
    payload["selector"].update(
        {
            "inactive_selector_scope": (
                "full_provisionally_filtered_frontier_accept_recover_defer"
            ),
            "active_selector_scope": "filtered_recovery_candidates_only",
            "selector_unchanged_after_masking": True,
        }
    )
    payload["dynamic_frontier"].update(
        {
            "accept_semantics": (
                "nominal_exact_safe_only_while_no_committed_segment_is_active"
            ),
            "defer_semantics": (
                "nominal_bounded_event_only_while_no_committed_segment_is_active"
            ),
        }
    )
    payload["committed_recovery_segment"] = {
        "activation": (
            "inactive_frozen_selector_selects_deliver_or_reconfigure"
        ),
        "cohort": "exact_stored_block_label_set_at_activation",
        "continuity": (
            "next_timing_erased_RecoveryState_labels_must_be_subset_of_cohort"
        ),
        "new_stored_labels_during_active_segment": "fail_closed",
        "accept_and_defer_while_active": "masked_fail_closed",
        "arrived_but_unstored_external_work": "outside_RecoveryState_cohort",
        "budget_replenishment": False,
        "relocation_reset": False,
        "delivery_reset": False,
    }
    payload["live_disturbance"].update(
        {
            "execution_contract": EXACT_RECOVERY_EXECUTION_CONTRACT,
            "base_execution_contract": EXACT_RECOVERY_EXECUTION_CONTRACT,
            "live_path_execution": (
                "literal_certified_approach_PICKUP_transport_PUTDOWN_no_"
                "planner_no_replanning"
            ),
            "base_duration_check": "actual_equals_RecoveryAction.steps",
            "total_duration_check": (
                "actual_equals_RecoveryAction.steps_plus_requested_injection"
            ),
            "declared_successor_check": (
                "immediate_and_next_boundary_timing_erased_exact_RecoveryState"
            ),
            "successor_equal_fields": [
                "rows",
                "cols",
                "traversable",
                "storage_cells",
                "exits",
                "block_label_to_position",
                "agent_position",
                "fixed_obstacles",
                "reserved_cells",
                "pickup_cells",
                "wait_cells",
            ],
            "successor_ignored_fields": ["blocks[].remaining_time"],
            "realization_sampling_scope": (
                "one_deterministic_action_conditional_member_not_adversarial_"
                "not_worst_case_not_exhaustive_W"
            ),
            "nonterminal_fixed_member": (
                "delay_1_plus_lexicographically_first_clear_adjacent_stop_if_"
                "available"
            ),
            "terminal_fixed_member": "delay_0_nominal_endpoint",
        }
    )
    payload["claims"] = {
        "full_dynamic_episode_evaluation": True,
        "conditional_recursive_completion_of_committed_stored_cohort": True,
        "unconditional_recursive_completion_claim": False,
        "full_dynamic_robust_viability": False,
        "full_episode_completion_theorem": False,
        "arrival_robustness": False,
        "timing_robustness": False,
        "accept_or_defer_robustness": False,
        "hardware_robustness": False,
    }
    payload["recursive_claim_conditions"] = [
        "recursive_method_action_is_WINNING_not_UNKNOWN_or_LOSING",
        "realized_disturbance_is_the_declared_fixed_envelope_member",
        "exact_RecoveryAction_path_duration_and_endpoint_match_live_execution",
        "next_timing_erased_physical_successor_matches_declared_outcome",
        "committed_budget_chain_has_no_reset_or_replenishment",
        "solver_did_not_hit_an_UNRESOLVED_UNKNOWN_cutoff_for_admission",
    ]
    payload["method_interpretation"] = {
        "nominal": (
            "finite_nominal_completion_filter_under_disturbed_live_execution_"
            "and_may_fail"
        ),
        "one_step": (
            "one_macro_disturbance_containment_then_nominal_completion_and_"
            "may_fail_later"
        ),
        "recursive": (
            "conditionally_theorem_facing_recursive_completion_filter_for_"
            "committed_cohort_when_all_declared_correspondence_checks_hold"
        ),
    }
    payload["computation_reporting"].update(
        {
            "memo_scope": "one_instance_shared_across_six_method_lambda_arms",
            "memo_cleared_between_instances": True,
            "memo_hit_fields_comparative": False,
            "memo_hit_reason": (
                "depends_on_arm_order_process_restart_and_resume_ledger_state"
            ),
            "wall_time_fields_comparative": False,
            "wall_time_reason": (
                "depends_on_memo_order_hardware_runtime_and_resume_state"
            ),
        }
    )
    payload["source_sha256"]["v1_helper_runner"] = payload[
        "source_sha256"
    ]["runner"]
    payload["source_sha256"]["runner"] = v1._sha(Path(__file__).resolve())
    payload["source_sha256"].update(_v2_semantic_source_sha256())
    payload["contract_sha256"] = v1._digest(payload)
    return payload


def prepare_contract(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
) -> dict:
    expected = _contract(
        project_root,
        output_dir,
        panel=panel,
        device_name=device_name,
    )
    path = output_dir.resolve() / CONTRACT_NAME
    if path.exists():
        observed = v1._read(path)
        v1._verify_self_hash(
            observed, "contract_sha256", label="budgeted dynamic contract"
        )
        if observed != expected:
            raise BudgetedDynamicFilterError(
                "budgeted dynamic contract or bound sources changed"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise BudgetedDynamicFilterError(
                "nonempty output has no budgeted dynamic contract"
            )
        v1._atomic_json(path, expected)
    return expected


def _duration_verification(
    candidate: ViabilityActionCandidate,
    *,
    realization,
    execution: v1.ObservedExecution,
) -> dict:
    action = candidate.recovery_action
    if action is None:
        raise BudgetedDynamicFilterError("recovery duration check lacks action")
    audit = execution.audit
    expected_base = int(action.steps)
    expected_injected = int(realization.requested_injected_steps)
    expected_total = expected_base + expected_injected
    observed_base = audit.get("base_primitive_steps")
    observed_injected = audit.get("injected_primitive_steps")
    observed_total = audit.get("total_primitive_steps")
    base_replans = audit.get("base_replan_count")
    exact_path = audit.get("exact_path", {})
    exact_trace_match = exact_path.get("trace_matches_certified_action")
    planner_calls = exact_path.get("planner_calls")
    exact_replans = exact_path.get("replan_count")
    matched = bool(
        observed_base == expected_base
        and observed_injected == expected_injected
        and observed_total == expected_total
        and execution.total_primitive_steps == expected_total
        and base_replans == 0
        and exact_trace_match is True
        and planner_calls == 0
        and exact_replans == 0
    )
    return {
        "matched": matched,
        "expected_base_duration": expected_base,
        "observed_base_duration": observed_base,
        "expected_injected_duration": expected_injected,
        "observed_injected_duration": observed_injected,
        "expected_total_duration": expected_total,
        "observed_total_duration": observed_total,
        "observed_wrapper_total_duration": execution.total_primitive_steps,
        "base_replan_count": base_replans,
        "exact_trace_matches_certified_action": exact_trace_match,
        "planner_calls": planner_calls,
        "exact_executor_replan_count": exact_replans,
    }


def _execute_exact_selected(
    env,
    candidate: ViabilityActionCandidate,
    *,
    realization,
    gamma: float,
    remaining_steps: int,
) -> v1.ObservedExecution:
    execution = execute_exact_bounded_recovery_realization(
        env,
        candidate,
        realization=realization,
        gamma=gamma,
        remaining_steps=remaining_steps,
        evaluation=True,
    )
    audit = exact_execution_to_dict(execution)
    audit.update(
        {
            "execution_layer": EXACT_RECOVERY_EXECUTION_CONTRACT,
            "recovery_disturbance_applied": True,
        }
    )
    return v1.ObservedExecution(
        total_primitive_steps=int(execution.total_primitive_steps),
        total_raw_return=float(execution.total_raw_return),
        delivery_deviations=tuple(
            float(value)
            for value in execution.base_execution.delivery_deviations
        ),
        relocations=int(execution.base_execution.relocations),
        illegal_drops=int(execution.base_execution.illegal_drops),
        env_terminal=bool(execution.env_terminal),
        success=bool(execution.realization_complete),
        failure_reason=execution.failure_reason,
        replay_terminal=bool(execution.replay_terminal),
        audit=audit,
    )


def _selected_failure_execution(candidate, reason: str, **audit) -> dict:
    return {
        "candidate_key": candidate.key,
        "action_type": candidate.action_type.value,
        "success": False,
        "failure_reason": reason,
        **audit,
    }


def _run_episode(
    *,
    arm,
    instance,
    method: str,
    handling_lambda: float,
    device: torch.device,
    base_agent,
    cost_network,
    certification_memo: dict,
) -> dict:
    env = benchmark._make_env(arm.payload)
    env.current_episode = 1
    env.reset(instance=instance)
    if env.current_episode_instance.instance_id != instance.instance_id:
        raise BudgetedDynamicFilterError(
            "environment did not consume frozen EpisodeInstance"
        )
    search_config = benchmark._search_config(arm.payload)
    liveness_rule = benchmark._liveness_rule(arm.payload)
    frontier_cache = ViabilityCertificateCache()
    segment = CommittedRecoverySegment()
    pending: Optional[PendingPhysicalSuccessor] = None

    steps = 0
    total_return = 0.0
    consecutive_defer = 0
    delivery_deviations: list[float] = []
    physical_rehandles = 0
    standalone_reconfigurations = 0
    standalone_with_direct_delivery_available = 0
    standalone_without_direct_delivery_available = 0
    directly_deliverable_self_reconfigurations = 0
    illegal_drops = 0
    macro_failures = 0
    decisions = []
    method_failure_reason = None
    complete_frontier = True
    duration_verification_count = 0
    duration_mismatch_count = 0
    immediate_successor_verification_count = 0
    boundary_successor_verification_count = 0
    physical_successor_mismatch_count = 0
    started = perf_counter()

    while steps < v1.v11.MAX_STEPS and not env.is_state_terminal(
        env.current_state
    ):
        snapshot, frontier = benchmark._enumerate_frontier(
            env,
            consecutive_defer=consecutive_defer,
            search_config=search_config,
            liveness_rule=liveness_rule,
            cache=frontier_cache,
            prioritizer=None,
        )
        complete_frontier = complete_frontier and bool(
            frontier["complete_frontier_exactly_verified"]
        )
        timing_erased_state = _timing_erased_state(snapshot.recovery_state)

        if pending is not None:
            boundary_check = _physical_comparison(
                pending.expected_key,
                _recovery_physical_key(timing_erased_state),
            )
            boundary_successor_verification_count += 1
            decisions[pending.decision_index][
                "next_boundary_successor_verification"
            ] = boundary_check
            if not boundary_check["matched"]:
                physical_successor_mismatch_count += 1
                method_failure_reason = (
                    "declared_recovery_successor_mismatch_at_next_boundary"
                )
                pending = None
                break
            pending = None

        try:
            segment_boundary = segment.boundary(timing_erased_state)
        except BudgetedDynamicFilterError:
            method_failure_reason = "committed_recovery_cohort_continuity_failure"
            break

        if not snapshot.candidates:
            method_failure_reason = "no_nominal_exact_safe_candidate"
            break
        if not all(
            candidate.certificate.status is ViabilityStatus.SAFE
            for candidate in snapshot.candidates
        ):
            raise BudgetedDynamicFilterError(
                "nominal exact frontier contains non-SAFE action"
            )

        remaining_live_horizon = int(v1.v11.MAX_STEPS - steps)
        if segment.active:
            budget_audit = {
                "status": "ACTIVE_CARRIED_BUDGET",
                "primitive_budget": int(segment.remaining_budget),
                "segment_index": segment.epoch_index,
                "remaining_episode_horizon": remaining_live_horizon,
                "timing_erased": True,
            }
        else:
            budget_audit = _provisional_budget(
                snapshot,
                remaining_episode_horizon=remaining_live_horizon,
            )
        primitive_budget = budget_audit["primitive_budget"]
        admitted, filter_audit = _filter_recovery_frontier(
            snapshot,
            method=method,
            primitive_budget=primitive_budget,
            segment_active=segment.active,
            memo=certification_memo,
            memo_namespace=int(instance.seed),
        )
        if not admitted:
            method_failure_reason = (
                "empty_active_recovery_frontier"
                if segment.active
                else "empty_method_filtered_frontier"
            )
            decisions.append(
                {
                    "decision_index": len(decisions),
                    "decision_epoch": int(snapshot.decision_epoch),
                    "original_candidate_keys": [
                        item.key for item in snapshot.candidates
                    ],
                    "segment_boundary": segment_boundary,
                    "budget": budget_audit,
                    "filter": filter_audit,
                    "selection": None,
                    "execution": None,
                }
            )
            break

        selection = v1._select_frozen(
            snapshot,
            admitted_keys=admitted,
            base_agent=base_agent,
            cost_network=cost_network,
            handling_lambda=handling_lambda,
        )
        candidate = selection.candidate
        recovery = candidate.action_type in (
            ViabilityActionType.DELIVER,
            ViabilityActionType.RECONFIGURE,
        )
        if segment.active and not recovery:
            raise BudgetedDynamicFilterError(
                "active committed segment selected a masked non-recovery action"
            )

        _, authenticated_actions = _authenticated_recovery_actions(snapshot)
        activation = None
        if recovery and not segment.active:
            if not isinstance(primitive_budget, int) or primitive_budget < 1:
                method_failure_reason = (
                    "selected_recovery_without_finite_provisional_budget"
                )
            else:
                activation = segment.activate(
                    timing_erased_state,
                    primitive_budget=primitive_budget,
                    nominal_witness_primitive_steps=budget_audit[
                        "nominal_witness_primitive_steps"
                    ],
                    nominal_witness_macro_count=budget_audit[
                        "nominal_witness_macro_count"
                    ],
                    remaining_episode_horizon=remaining_live_horizon,
                )

        decision = {
            "decision_index": len(decisions),
            "decision_epoch": int(snapshot.decision_epoch),
            "original_candidate_keys": [item.key for item in snapshot.candidates],
            "segment_boundary": segment_boundary,
            "segment_activation": activation,
            "budget": budget_audit,
            "filter": filter_audit,
            "selection": selection.audit,
            "execution": None,
        }
        decisions.append(decision)
        if method_failure_reason is not None:
            decision["execution"] = _selected_failure_execution(
                candidate, method_failure_reason
            )
            break

        realization = None
        declared_outcome = None
        if recovery:
            action = authenticated_actions[candidate.key]
            realization = v1._realization(env, candidate)
            declared_outcome = _declared_recovery_outcome(
                timing_erased_state,
                action,
                realization=realization,
                primitive_budget=int(segment.remaining_budget),
            )
            decision["declared_recovery_outcome"] = declared_outcome
            expected_total = int(action.steps) + int(
                realization.requested_injected_steps
            )
            if declared_outcome["primitive_steps"] != expected_total:
                raise BudgetedDynamicFilterError(
                    "declared outcome cost disagrees with live realization"
                )
            if (
                not declared_outcome["tube_safe"]
                or not declared_outcome["horizon_safe"]
                or declared_outcome["expected_physical_key"] is None
            ):
                method_failure_reason = (
                    "selected_declared_realization_outside_carried_budget"
                )
                decision["execution"] = _selected_failure_execution(
                    candidate,
                    method_failure_reason,
                    realization=v1.asdict(realization),
                )
                break
            if expected_total > remaining_live_horizon:
                raise BudgetedDynamicFilterError(
                    "carried budget exceeded remaining live episode horizon"
                )

        direct_delivery_labels = {
            item.target_label
            for item in snapshot.candidates
            if item.action_type is ViabilityActionType.DELIVER
        }
        selected_reconfiguration = (
            candidate.action_type is ViabilityActionType.RECONFIGURE
        )
        selected_reconfigure_block_directly_deliverable = bool(
            selected_reconfiguration
            and candidate.target_label in direct_delivery_labels
        )
        if recovery:
            execution = _execute_exact_selected(
                env,
                candidate,
                realization=realization,
                gamma=base_agent.config.gamma,
                remaining_steps=remaining_live_horizon,
            )
        else:
            execution = v1._execute_selected(
                env,
                candidate,
                gamma=base_agent.config.gamma,
                remaining_steps=remaining_live_horizon,
            )
        decision["execution"] = execution.audit

        duration_check = None
        immediate_check = None
        budget_charge = None
        segment_completion = None
        if recovery:
            duration_check = _duration_verification(
                candidate,
                realization=realization,
                execution=execution,
            )
            decision["recovery_duration_verification"] = duration_check
            duration_verification_count += 1
            if not duration_check["matched"]:
                duration_mismatch_count += 1
                method_failure_reason = "recovery_macro_duration_mismatch"
            else:
                budget_charge = segment.consume(
                    int(execution.total_primitive_steps),
                    action_type=candidate.action_type.value,
                )
            expected_key = declared_outcome["expected_physical_key"]
            try:
                actual_state = _live_recovery_state(
                    env,
                    template=timing_erased_state,
                    search_config=search_config,
                )
                immediate_check = _physical_comparison(
                    expected_key,
                    _recovery_physical_key(actual_state),
                )
            except (TypeError, ValueError, RuntimeError) as error:
                immediate_check = {
                    "matched": False,
                    "status": "live_state_not_a_strict_recovery_boundary",
                    "error_type": type(error).__name__,
                    "expected_key": expected_key,
                    "actual_key": None,
                    "ignored_fields": ["blocks[].remaining_time"],
                    "mismatched_fields": ["strict_boundary_unavailable"],
                }
            immediate_successor_verification_count += 1
            if not immediate_check["matched"]:
                physical_successor_mismatch_count += 1
                method_failure_reason = (
                    "declared_recovery_successor_mismatch_immediate"
                )
            elif (
                duration_check["matched"]
                and execution.success
                and not expected_key["blocks"]
            ):
                segment_completion = segment.close_completed()
            if (
                duration_check["matched"]
                and execution.success
                and immediate_check["matched"]
                and not execution.env_terminal
            ):
                pending = PendingPhysicalSuccessor(
                    decision_index=decision["decision_index"],
                    candidate_key=candidate.key,
                    expected_key=expected_key,
                )
            elif (
                duration_check["matched"]
                and execution.success
                and immediate_check["matched"]
                and execution.env_terminal
            ):
                decision["next_boundary_successor_verification"] = {
                    "matched": True,
                    "status": "terminal_boundary_vacuous",
                    "expected_key": expected_key,
                    "actual_key": immediate_check["actual_key"],
                    "ignored_fields": ["blocks[].remaining_time"],
                    "mismatched_fields": [],
                }
            else:
                decision["next_boundary_successor_verification"] = {
                    "matched": False,
                    "status": "next_boundary_not_established_after_failed_recovery",
                    "expected_key": expected_key,
                    "actual_key": None,
                    "ignored_fields": ["blocks[].remaining_time"],
                    "mismatched_fields": ["boundary_not_observed"],
                }
                physical_successor_mismatch_count += 1
        decision["budget_charge"] = budget_charge
        decision["immediate_successor_verification"] = immediate_check
        decision["segment_completion"] = segment_completion

        steps += int(execution.total_primitive_steps)
        total_return += float(execution.total_raw_return)
        delivery_deviations.extend(execution.delivery_deviations)
        physical_rehandles += int(execution.relocations)
        illegal_drops += int(execution.illegal_drops)
        macro_failures += int(not execution.success)
        if selected_reconfiguration:
            standalone_reconfigurations += int(execution.relocations)
            if direct_delivery_labels:
                standalone_with_direct_delivery_available += int(
                    execution.relocations
                )
            else:
                standalone_without_direct_delivery_available += int(
                    execution.relocations
                )
            if selected_reconfigure_block_directly_deliverable:
                directly_deliverable_self_reconfigurations += int(
                    execution.relocations
                )
        elif execution.relocations:
            raise BudgetedDynamicFilterError(
                "physical rehandle occurred outside Reconfigure"
            )

        if candidate.action_type is ViabilityActionType.DEFER:
            outcome = getattr(candidate.option, "last_outcome", None)
            observed_event = bool(
                isinstance(outcome, dict)
                and outcome.get("reason") == "observed_event"
            )
            consecutive_defer = 0 if observed_event else consecutive_defer + 1
        else:
            consecutive_defer = 0
        decision["post_execution_environment"] = benchmark._environment_signature(env)

        if method_failure_reason is not None:
            break
        if not execution.success:
            method_failure_reason = (
                f"macro_execution_failure:{execution.failure_reason or 'unknown'}"
            )
            break
        if execution.total_primitive_steps <= 0:
            method_failure_reason = "zero_duration_macro"
            break
        if execution.replay_terminal:
            break

    terminal = bool(env.is_state_terminal(env.current_state))
    if pending is not None:
        decisions[pending.decision_index][
            "next_boundary_successor_verification"
        ] = {
            "matched": False,
            "status": "next_live_replanning_boundary_not_reached",
            "expected_key": dict(pending.expected_key),
            "actual_key": None,
            "ignored_fields": ["blocks[].remaining_time"],
            "mismatched_fields": ["boundary_not_observed"],
        }
        physical_successor_mismatch_count += 1
        if method_failure_reason is None:
            method_failure_reason = "missing_post_recovery_replanning_boundary"
    if segment.active and method_failure_reason is None:
        method_failure_reason = "episode_ended_with_active_committed_segment"
    if (
        steps >= v1.v11.MAX_STEPS
        and not terminal
        and method_failure_reason is None
    ):
        method_failure_reason = "episode_step_limit"

    success = bool(terminal and method_failure_reason is None)
    strict = bool(
        success
        and macro_failures == 0
        and illegal_drops == 0
        and complete_frontier
        and len(delivery_deviations) == EXPECTED_BLOCKS
        and duration_mismatch_count == 0
        and physical_successor_mismatch_count == 0
        and not segment.active
    )
    timing = _nullable_timing_summary(delivery_deviations)
    contention = v1.contention_metric_record(
        physical_storage_relocations=physical_rehandles,
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
    legacy_return, dense_return = v1._dual_rescore_from_legacy_return(
        total_return,
        delivery_deviations,
        v1.v11.FROZEN_OBJECTIVE_SPEC,
    )
    behavior = {
        "instance_seed": int(instance.seed),
        "episode_instance_id": instance.instance_id,
        "method": method,
        "handling_lambda": float(handling_lambda),
        "decisions": decisions,
        "final_environment": benchmark._environment_signature(env),
        "return": float(total_return),
        "steps": int(steps),
        "terminal": terminal,
        "failure": method_failure_reason,
    }
    robust_decisions = [
        item
        for item in decisions
        if item.get("filter", {}).get("original_recovery_keys")
    ]
    status_counts = Counter(item["filter"]["status"] for item in robust_decisions)
    return {
        "instance_seed": int(instance.seed),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "method": method,
        "handling_lambda": float(handling_lambda),
        "device": str(device),
        "strict_safe_complete": strict,
        "terminal": terminal,
        "success": success,
        "method_failure_reason": method_failure_reason,
        "delivery_count": len(delivery_deviations),
        "delivery_deviations": delivery_deviations,
        "raw_environment_return": float(total_return),
        "return": float(legacy_return),
        "dense_return": float(dense_return),
        "steps": int(steps),
        "physical_rehandles": int(physical_rehandles),
        "physical_rehandles_per_100": float(
            100.0 * physical_rehandles / EXPECTED_BLOCKS
        ),
        **timing,
        **contention,
        "illegal_drops": int(illegal_drops),
        "macro_failures": int(macro_failures),
        "complete_nominal_frontier_exactly_verified": complete_frontier,
        "recovery_filter_decision_count": len(robust_decisions),
        "recovery_filter_status_counts": dict(sorted(status_counts.items())),
        "recovery_candidates_removed": sum(
            len(item["filter"].get("rejected_recovery_keys", ()))
            for item in robust_decisions
        ),
        "logical_recovery_expanded_nodes": sum(
            item["filter"].get("method_certificate", {})
            .get("computation", {})
            .get("expanded_nodes", 0)
            for item in robust_decisions
        ),
        "recovery_decisions_hitting_compute_cutoff": sum(
            item["filter"].get("method_certificate", {})
            .get("computation", {})
            .get("cutoff_count", 0)
            > 0
            for item in robust_decisions
        ),
        "recovery_certification_wall_seconds": float(
            sum(
                item["filter"].get("certification_wall_seconds", 0.0)
                for item in robust_decisions
            )
        ),
        "recovery_certification_memo_hits": sum(
            bool(item["filter"].get("memo_hit")) for item in robust_decisions
        ),
        "committed_segment_activation_count": segment.activation_count,
        "committed_segment_completion_count": segment.completion_count,
        "committed_segment_open_at_end": segment.active,
        "committed_segment_total_consumed_steps": segment.total_consumed_steps,
        "recovery_duration_verification_count": duration_verification_count,
        "recovery_duration_mismatch_count": duration_mismatch_count,
        "immediate_successor_verification_count": (
            immediate_successor_verification_count
        ),
        "boundary_successor_verification_count": (
            boundary_successor_verification_count
        ),
        "physical_successor_mismatch_count": physical_successor_mismatch_count,
        "timing_erased_recovery_certification": True,
        "episode_wall_seconds": float(perf_counter() - started),
        "behavior_digest": v1._digest(v1._behavior_semantics(behavior)),
        "final_environment": behavior["final_environment"],
        "decisions": decisions,
        "training_or_learning": False,
    }


def _ledger_path(
    output_dir: Path,
    *,
    method: str,
    handling_lambda: float,
    instance_seed: int,
) -> Path:
    return (
        output_dir.resolve()
        / "run-ledger"
        / method
        / f"lambda-{_lambda_key(handling_lambda)}"
        / f"instance-{int(instance_seed)}.json"
    )


def _replay_committed_segment_audit(row: Mapping) -> dict:
    """Authenticate the complete carried-budget state machine from decisions."""

    decisions = row.get("decisions")
    if not isinstance(decisions, list):
        raise BudgetedDynamicFilterError("segment replay requires decision list")
    active = False
    segment_index = None
    next_segment_index = 0
    initial_budget = None
    remaining = None
    cohort: tuple[str, ...] = ()
    current_labels: tuple[str, ...] = ()
    activations = 0
    completions = 0
    consumed_steps = 0
    duration_checks = 0
    duration_mismatches = 0
    immediate_checks = 0
    boundary_checks = 0
    physical_mismatches = 0

    for expected_index, decision in enumerate(decisions):
        if decision.get("decision_index") != expected_index:
            raise BudgetedDynamicFilterError("decision indices are not contiguous")
        boundary = decision.get("segment_boundary")
        budget_audit = decision.get("budget")
        filter_audit = decision.get("filter")
        if (
            not isinstance(boundary, Mapping)
            or not isinstance(budget_audit, Mapping)
            or not isinstance(filter_audit, Mapping)
        ):
            raise BudgetedDynamicFilterError(
                "decision lacks segment boundary or filter audit"
            )
        if filter_audit.get("segment_active") is not active:
            raise BudgetedDynamicFilterError(
                "filter active flag disagrees with replayed segment state"
            )
        if boundary.get("active") is not active:
            raise BudgetedDynamicFilterError(
                "boundary active flag disagrees with replayed segment state"
            )
        if active:
            if boundary.get("event") != "active_continuity_verified":
                raise BudgetedDynamicFilterError("active segment continuity was skipped")
            if boundary.get("segment_index") != segment_index:
                raise BudgetedDynamicFilterError("active segment identity changed")
            if boundary.get("initial_budget") != initial_budget:
                raise BudgetedDynamicFilterError("active initial budget changed")
            if boundary.get("remaining_budget") != remaining:
                raise BudgetedDynamicFilterError(
                    "active boundary replenished or reset carried budget"
                )
            if filter_audit.get("primitive_budget") != remaining:
                raise BudgetedDynamicFilterError(
                    "active filter did not use carried remaining budget"
                )
            if budget_audit.get("primitive_budget") != remaining:
                raise BudgetedDynamicFilterError(
                    "active decision budget did not equal carried budget"
                )
            if filter_audit.get("pass_through_keys"):
                raise BudgetedDynamicFilterError(
                    "active segment retained non-recovery candidates"
                )
            observed = tuple(boundary.get("observed_recovery_labels", ()))
            if tuple(sorted(observed)) != observed or not set(observed).issubset(
                current_labels
            ):
                raise BudgetedDynamicFilterError(
                    "committed cohort did not shrink monotonically"
                )
            if tuple(boundary.get("cohort_labels", ())) != cohort:
                raise BudgetedDynamicFilterError("committed cohort identity changed")
            current_labels = observed
        else:
            if boundary.get("event") != "inactive_boundary":
                raise BudgetedDynamicFilterError("inactive boundary audit changed")
            if boundary.get("segment_index") is not None:
                raise BudgetedDynamicFilterError("inactive boundary has segment id")

        selection = decision.get("selection")
        execution = decision.get("execution")
        selected = selection is not None
        action_type = None
        recovery_selected = False
        if selected:
            if not isinstance(execution, Mapping):
                raise BudgetedDynamicFilterError(
                    "selected decision has no execution audit"
                )
            action_type = execution.get("action_type")
            recovery_selected = action_type in ("deliver", "reconfigure")
            if active and not recovery_selected:
                raise BudgetedDynamicFilterError(
                    "active segment selected Accept or Defer"
                )

        activation = decision.get("segment_activation")
        if activation is not None:
            if active or not recovery_selected:
                raise BudgetedDynamicFilterError("illegal segment activation")
            if activation.get("event") != "committed_recovery_segment_activated":
                raise BudgetedDynamicFilterError("activation event changed")
            if activation.get("segment_index") != next_segment_index:
                raise BudgetedDynamicFilterError("segment indices changed")
            initial = activation.get("initial_budget")
            witness_steps = activation.get("nominal_witness_primitive_steps")
            witness_macros = activation.get("nominal_witness_macro_count")
            live_horizon = activation.get("remaining_episode_horizon")
            uncapped = activation.get("uncapped_budget")
            if (
                isinstance(initial, bool)
                or not isinstance(initial, int)
                or initial < 1
                or initial != filter_audit.get("primitive_budget")
                or initial != budget_audit.get("primitive_budget")
                or activation.get("remaining_budget") != initial
                or isinstance(witness_steps, bool)
                or not isinstance(witness_steps, int)
                or witness_steps < 1
                or isinstance(witness_macros, bool)
                or not isinstance(witness_macros, int)
                or witness_macros < 1
                or isinstance(live_horizon, bool)
                or not isinstance(live_horizon, int)
                or live_horizon < 1
                or uncapped
                != witness_steps
                + BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO * witness_macros
                or initial != min(uncapped, live_horizon)
            ):
                raise BudgetedDynamicFilterError("activation budget is invalid")
            cohort = tuple(activation.get("cohort_labels", ()))
            if (
                not cohort
                or tuple(sorted(cohort)) != cohort
                or cohort
                != tuple(boundary.get("observed_recovery_labels", ()))
            ):
                raise BudgetedDynamicFilterError("activation cohort is invalid")
            current_labels = cohort
            active = True
            segment_index = next_segment_index
            next_segment_index += 1
            initial_budget = initial
            remaining = initial
            activations += 1
        elif recovery_selected and not active:
            raise BudgetedDynamicFilterError(
                "inactive recovery selection did not activate a segment"
            )

        duration = decision.get("recovery_duration_verification")
        immediate = decision.get("immediate_successor_verification")
        next_boundary = decision.get("next_boundary_successor_verification")
        charge = decision.get("budget_charge")
        completion = decision.get("segment_completion")
        actually_executed_recovery = bool(
            recovery_selected
            and isinstance(execution.get("total_primitive_steps"), int)
        )
        if actually_executed_recovery:
            if not isinstance(duration, Mapping):
                raise BudgetedDynamicFilterError(
                    "executed recovery lacks duration verification"
                )
            if not isinstance(immediate, Mapping):
                raise BudgetedDynamicFilterError(
                    "executed recovery lacks immediate successor verification"
                )
            if not isinstance(next_boundary, Mapping):
                raise BudgetedDynamicFilterError(
                    "executed recovery lacks next-boundary coverage"
                )
            exact_path = execution.get("exact_path")
            if not isinstance(exact_path, Mapping):
                raise BudgetedDynamicFilterError(
                    "executed recovery lacks exact certified-path trace"
                )
            expected_names = exact_path.get("expected_action_names")
            emitted_names = exact_path.get("emitted_action_names")
            expected_cells = exact_path.get("expected_cell_trace")
            observed_cells = exact_path.get("observed_cell_trace")
            derived_trace_match = bool(
                isinstance(expected_names, (tuple, list))
                and isinstance(emitted_names, (tuple, list))
                and isinstance(expected_cells, (tuple, list))
                and isinstance(observed_cells, (tuple, list))
                and list(expected_names) == list(emitted_names)
                and list(expected_cells) == list(observed_cells)
                and exact_path.get("planner_calls") == 0
                and exact_path.get("replan_count") == 0
            )
            if exact_path.get("trace_matches_certified_action") is not derived_trace_match:
                raise BudgetedDynamicFilterError(
                    "exact-path trace flag disagrees with emitted trace"
                )
            recovery_action = exact_path.get("recovery_action")
            if not isinstance(recovery_action, Mapping) or recovery_action.get(
                "steps"
            ) != len(expected_names):
                raise BudgetedDynamicFilterError(
                    "exact-path trace length disagrees with RecoveryAction"
                )
            duration_checks += 1
            immediate_checks += 1
            duration_mismatches += int(duration.get("matched") is not True)
            physical_mismatches += int(immediate.get("matched") is not True)
            physical_mismatches += int(next_boundary.get("matched") is not True)
            if "status" not in next_boundary:
                boundary_checks += 1
            if duration.get("matched") is True:
                if (
                    derived_trace_match is not True
                    or duration.get("exact_trace_matches_certified_action")
                    is not True
                    or duration.get("planner_calls") != 0
                    or duration.get("exact_executor_replan_count") != 0
                ):
                    raise BudgetedDynamicFilterError(
                        "matched duration lacks exact-path correspondence"
                    )
                if not isinstance(charge, Mapping) or charge.get("charged") is not True:
                    raise BudgetedDynamicFilterError(
                        "exact-duration recovery was not charged"
                    )
                if charge.get("segment_index") != segment_index:
                    raise BudgetedDynamicFilterError("budget charge segment changed")
                spent = charge.get("duration")
                after = charge.get("after")
                if (
                    charge.get("before") != remaining
                    or isinstance(spent, bool)
                    or not isinstance(spent, int)
                    or spent < 0
                    or spent != execution.get("total_primitive_steps")
                    or charge.get("action_type") != action_type
                    or after != remaining - spent
                    or after < 0
                ):
                    raise BudgetedDynamicFilterError(
                        "carried budget charge chain is invalid"
                    )
                remaining = after
                consumed_steps += spent
            elif charge is not None:
                raise BudgetedDynamicFilterError(
                    "duration-mismatched recovery was charged as certified"
                )
        elif any(item is not None for item in (duration, immediate, charge, completion)):
            raise BudgetedDynamicFilterError(
                "nonexecuted recovery/nonrecovery has execution verification"
            )

        if completion is not None:
            if not active or not actually_executed_recovery:
                raise BudgetedDynamicFilterError("illegal segment completion")
            if completion.get("event") != "committed_recovery_segment_completed":
                raise BudgetedDynamicFilterError("completion event changed")
            if (
                execution.get("realization_complete") is not True
                or not isinstance(duration, Mapping)
                or duration.get("matched") is not True
                or not isinstance(charge, Mapping)
                or charge.get("charged") is not True
                or not isinstance(immediate, Mapping)
                or immediate.get("matched") is not True
            ):
                raise BudgetedDynamicFilterError(
                    "failed or unverified recovery cannot complete a segment"
                )
            if completion.get("completed_segment_index") != segment_index:
                raise BudgetedDynamicFilterError("completed segment identity changed")
            if completion.get("initial_budget") != initial_budget:
                raise BudgetedDynamicFilterError("completion initial budget changed")
            if tuple(completion.get("cohort_labels", ())) != cohort:
                raise BudgetedDynamicFilterError("completion cohort changed")
            if completion.get("discarded_leftover_budget") != remaining:
                raise BudgetedDynamicFilterError(
                    "completion leftover disagrees with carried budget"
                )
            expected_key = immediate.get("expected_key")
            if (
                immediate.get("matched") is not True
                or not isinstance(expected_key, Mapping)
                or expected_key.get("blocks") != []
            ):
                raise BudgetedDynamicFilterError(
                    "segment completed before committed cohort became empty"
                )
            completions += 1
            active = False
            segment_index = None
            remaining = None
            initial_budget = None
            cohort = ()
            current_labels = ()
        elif (
            actually_executed_recovery
            and execution.get("realization_complete") is True
            and isinstance(duration, Mapping)
            and duration.get("matched") is True
            and isinstance(immediate, Mapping)
            and immediate.get("matched") is True
            and isinstance(immediate.get("expected_key"), Mapping)
            and immediate["expected_key"].get("blocks") == []
        ):
            raise BudgetedDynamicFilterError(
                "empty committed cohort did not close its segment"
            )

    replay = {
        "committed_segment_activation_count": activations,
        "committed_segment_completion_count": completions,
        "committed_segment_open_at_end": active,
        "committed_segment_total_consumed_steps": consumed_steps,
        "recovery_duration_verification_count": duration_checks,
        "recovery_duration_mismatch_count": duration_mismatches,
        "immediate_successor_verification_count": immediate_checks,
        "boundary_successor_verification_count": boundary_checks,
        "physical_successor_mismatch_count": physical_mismatches,
    }
    for key, expected in replay.items():
        if row.get(key) != expected:
            raise BudgetedDynamicFilterError(
                f"row count disagrees with segment replay: {key}"
            )
    return replay


def _validate_ledger(
    ledger: Mapping,
    *,
    contract: Mapping,
    method: str,
    handling_lambda: float,
    record: Mapping,
) -> dict:
    v1._verify_self_hash(
        ledger, "ledger_sha256", label="budgeted dynamic run ledger"
    )
    expected_outer = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": method,
        "handling_lambda": float(handling_lambda),
        "instance_seed": int(record["instance_seed"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "checkpoint_sha256": v1.source89.SELECTED_CHECKPOINT_SHA256[MODEL_SEED],
        "cost_head_sha256": v1.source89.COST_HEAD_SHA256[MODEL_SEED],
    }
    for key, value in expected_outer.items():
        if ledger.get(key) != value:
            raise BudgetedDynamicFilterError(
                f"budgeted dynamic ledger identity changed: {key}"
            )

    # Reuse the exhaustive v1 common-row validator after changing only the
    # additive outer protocol identity and recomputing that temporary hash.
    original_row = ledger.get("row")
    if not isinstance(original_row, Mapping):
        raise BudgetedDynamicFilterError("budgeted ledger has no result row")
    legacy_row = dict(original_row)
    if legacy_row.get("delivery_count") == 0:
        for key in TIMING_METRIC_FIELDS:
            if legacy_row.get(key) is None:
                legacy_row[key] = 0.0
    legacy = dict(ledger)
    legacy["row"] = legacy_row
    legacy["schema_version"] = v1.SCHEMA_VERSION
    legacy["protocol"] = v1.PROTOCOL
    legacy["ledger_sha256"] = v1._digest(legacy, drop="ledger_sha256")
    v1._validate_ledger(
        legacy,
        contract={
            "contract_sha256": contract["contract_sha256"],
            "device": contract["device"],
        },
        method=method,
        handling_lambda=handling_lambda,
        record=record,
    )
    row = dict(original_row)

    delivery_count = row.get("delivery_count")
    if any(key not in row for key in TIMING_METRIC_FIELDS):
        raise BudgetedDynamicFilterError("row is missing declared timing metrics")
    if delivery_count == 0:
        if any(row.get(key) is not None for key in TIMING_METRIC_FIELDS):
            raise BudgetedDynamicFilterError(
                "zero-delivery timing metrics must be JSON null"
            )
    else:
        required_finite = tuple(
            key
            for key in TIMING_METRIC_FIELDS
            if key != "mean_tardiness_when_tardy"
        )
        if any(
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), (int, float))
            or not math.isfinite(float(row[key]))
            for key in required_finite
        ):
            raise BudgetedDynamicFilterError(
                "delivered-row timing metrics must be finite"
            )
        conditional = row.get("mean_tardiness_when_tardy")
        if conditional is not None and (
            isinstance(conditional, bool)
            or not isinstance(conditional, (int, float))
            or not math.isfinite(float(conditional))
        ):
            raise BudgetedDynamicFilterError(
                "conditional tardiness metric is invalid"
            )

    def require_json_finite(value, path="row") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                require_json_finite(item, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                require_json_finite(item, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise BudgetedDynamicFilterError(
                f"nonfinite float is forbidden in v2 ledger: {path}"
            )

    require_json_finite(row)

    nonnegative_integer_fields = (
        "committed_segment_activation_count",
        "committed_segment_completion_count",
        "committed_segment_total_consumed_steps",
        "recovery_duration_verification_count",
        "recovery_duration_mismatch_count",
        "immediate_successor_verification_count",
        "boundary_successor_verification_count",
        "physical_successor_mismatch_count",
    )
    if any(
        isinstance(row.get(key), bool)
        or not isinstance(row.get(key), int)
        or row[key] < 0
        for key in nonnegative_integer_fields
    ):
        raise BudgetedDynamicFilterError(
            "budgeted dynamic row has invalid v2 counts"
        )
    if not isinstance(row.get("committed_segment_open_at_end"), bool):
        raise BudgetedDynamicFilterError("invalid final segment state")
    if row.get("timing_erased_recovery_certification") is not True:
        raise BudgetedDynamicFilterError("row did not use timing-erased recovery")

    for decision in row["decisions"]:
        filter_audit = decision.get("filter", {})
        if filter_audit.get("timing_erased_recovery_state") is not True:
            raise BudgetedDynamicFilterError(
                "decision filter did not authenticate timing erasure"
            )
        if filter_audit.get("segment_active"):
            if filter_audit.get("pass_through_keys"):
                raise BudgetedDynamicFilterError(
                    "active segment retained Accept or Defer"
                )
            selection = decision.get("selection")
            execution = decision.get("execution")
            if selection is not None and execution is not None and execution.get(
                "action_type"
            ) not in ("deliver", "reconfigure"):
                raise BudgetedDynamicFilterError(
                    "active segment executed a non-recovery action"
                )
        activation = decision.get("segment_activation")
        if activation is not None:
            if activation.get("initial_budget") != decision.get("filter", {}).get(
                "primitive_budget"
            ):
                raise BudgetedDynamicFilterError(
                    "segment activation/filter budget mismatch"
                )
        duration = decision.get("recovery_duration_verification")
        if duration is not None and duration.get("matched"):
            if duration.get("expected_total_duration") != duration.get(
                "observed_total_duration"
            ):
                raise BudgetedDynamicFilterError(
                    "matched recovery duration audit is inconsistent"
                )
        charge = decision.get("budget_charge")
        if charge is not None and charge.get("charged"):
            if charge["after"] != charge["before"] - charge["duration"]:
                raise BudgetedDynamicFilterError(
                    "carried budget did not decrement exactly"
                )
            if charge["after"] < 0:
                raise BudgetedDynamicFilterError("carried budget became negative")

    _replay_committed_segment_audit(row)

    if row.get("strict_safe_complete"):
        v2_strict = (
            row["committed_segment_open_at_end"] is False,
            row["committed_segment_activation_count"]
            == row["committed_segment_completion_count"],
            row["recovery_duration_mismatch_count"] == 0,
            row["physical_successor_mismatch_count"] == 0,
            row["recovery_duration_verification_count"]
            == row["immediate_successor_verification_count"],
        )
        if not all(v2_strict):
            raise BudgetedDynamicFilterError(
                "strict row violates committed-segment invariants"
            )
    return row


def evaluate(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
) -> list[dict]:
    contract = prepare_contract(
        project_root,
        output_dir,
        panel=panel,
        device_name=device_name,
    )
    auth = v1.bridge._historical_source_auth(project_root)
    device = v1.pilot._device(device_name)
    v1.final86._configure_runtime()
    base_agent = v1.pilot._fresh_base(auth["arm"], device)
    cost_network = v1.source89._load_bound_cost(
        project_root,
        auth["arm"],
        device=device,
        config=base_agent.config,
    )
    seeds = set(_panel_seeds(panel))
    records = [
        record
        for record in auth["manifest"]["instances"]
        if int(record["instance_seed"]) in seeds
    ]
    rows = []
    total = len(records) * len(METHODS) * len(HANDLING_LAMBDAS)
    completed = 0
    for record in records:
        # Certificates are preference independent, so the six arms for one
        # frozen instance share a memo.  Clearing here bounds memory and makes
        # memo diagnostics explicitly instance/run/resume local.
        certification_memo: dict = {}
        instance = None
        for method in METHODS:
            for handling_lambda in HANDLING_LAMBDAS:
                path = _ledger_path(
                    output_dir,
                    method=method,
                    handling_lambda=handling_lambda,
                    instance_seed=int(record["instance_seed"]),
                )
                if path.exists():
                    row = _validate_ledger(
                        v1._read(path),
                        contract=contract,
                        method=method,
                        handling_lambda=handling_lambda,
                        record=record,
                    )
                else:
                    if instance is None:
                        instance = v1.source89._load_instance(
                            auth["source_output"], record
                        )
                    row = _run_episode(
                        arm=auth["arm"],
                        instance=instance,
                        method=method,
                        handling_lambda=handling_lambda,
                        device=device,
                        base_agent=base_agent,
                        cost_network=cost_network,
                        certification_memo=certification_memo,
                    )
                    ledger = {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "method": method,
                        "handling_lambda": float(handling_lambda),
                        "instance_seed": int(record["instance_seed"]),
                        "episode_instance_id": record["episode_instance_id"],
                        "schedule_id": record["schedule_id"],
                        "checkpoint_sha256": (
                            v1.source89.SELECTED_CHECKPOINT_SHA256[MODEL_SEED]
                        ),
                        "cost_head_sha256": v1.source89.COST_HEAD_SHA256[MODEL_SEED],
                        "row": row,
                    }
                    ledger["ledger_sha256"] = v1._digest(ledger)
                    _validate_ledger(
                        ledger,
                        contract=contract,
                        method=method,
                        handling_lambda=handling_lambda,
                        record=record,
                    )
                    v1._atomic_json(path, ledger)
                rows.append(row)
                completed += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{completed}/{total}",
                            "method": method,
                            "handling_lambda": handling_lambda,
                            "instance_seed": record["instance_seed"],
                            "strict_safe_complete": row["strict_safe_complete"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return rows


def _cell_summary(rows: Sequence[Mapping]) -> dict:
    summary = v1._cell_summary(rows)
    status_counts = Counter()
    for row in rows:
        status_counts.update(row.get("recovery_filter_status_counts", {}))
    duration_checks = sum(
        row["recovery_duration_verification_count"] for row in rows
    )
    immediate_checks = sum(
        row["immediate_successor_verification_count"] for row in rows
    )
    boundary_checks = sum(
        row["boundary_successor_verification_count"] for row in rows
    )
    summary.update(
        {
            "aggregate_recovery_filter_status_counts": dict(
                sorted(status_counts.items())
            ),
            "total_unknown_recovery_filter_decisions": int(
                status_counts.get(SolveStatus.UNKNOWN.value, 0)
            ),
            "total_committed_segment_activations": sum(
                row["committed_segment_activation_count"] for row in rows
            ),
            "total_committed_segment_completions": sum(
                row["committed_segment_completion_count"] for row in rows
            ),
            "rows_ending_with_open_segment": sum(
                bool(row["committed_segment_open_at_end"]) for row in rows
            ),
            "total_committed_segment_consumed_steps": sum(
                row["committed_segment_total_consumed_steps"] for row in rows
            ),
            "total_recovery_duration_mismatches": sum(
                row["recovery_duration_mismatch_count"] for row in rows
            ),
            "total_physical_successor_mismatches": sum(
                row["physical_successor_mismatch_count"] for row in rows
            ),
            "total_recovery_duration_verifications": duration_checks,
            "total_immediate_successor_verifications": immediate_checks,
            "total_observed_next_boundary_verifications": boundary_checks,
        }
    )
    return summary


def summarize(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
) -> dict:
    contract = prepare_contract(
        project_root,
        output_dir,
        panel=panel,
        device_name=device_name,
    )
    auth = v1.bridge._historical_source_auth(project_root)
    seeds = set(_panel_seeds(panel))
    records = [
        record
        for record in auth["manifest"]["instances"]
        if int(record["instance_seed"]) in seeds
    ]
    rows = []
    for method in METHODS:
        for handling_lambda in HANDLING_LAMBDAS:
            for record in records:
                path = _ledger_path(
                    output_dir,
                    method=method,
                    handling_lambda=handling_lambda,
                    instance_seed=int(record["instance_seed"]),
                )
                if not path.is_file():
                    raise BudgetedDynamicFilterError(
                        f"missing budgeted dynamic ledger: {path}"
                    )
                rows.append(
                    _validate_ledger(
                        v1._read(path),
                        contract=contract,
                        method=method,
                        handling_lambda=handling_lambda,
                        record=record,
                    )
                )
    expected = int(contract["expected_rows"])
    if len(rows) != expected:
        raise BudgetedDynamicFilterError("budgeted dynamic row grid is incomplete")
    method_summary = {
        method: {
            str(value): _cell_summary(
                [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["handling_lambda"] == value
                ]
            )
            for value in HANDLING_LAMBDAS
        }
        for method in METHODS
    }
    checks = {
        "exact_row_grid_complete": len(rows) == expected,
        "all_rows_no_training": all(
            row["training_or_learning"] is False for row in rows
        ),
        "all_instances_from_frozen_panel": {
            row["instance_seed"] for row in rows
        }
        == seeds,
        "all_rows_timing_erased_recovery": all(
            row["timing_erased_recovery_certification"] is True for row in rows
        ),
        "strict_rows_have_exact_duration_and_successor_checks": all(
            not row["strict_safe_complete"]
            or (
                row["recovery_duration_mismatch_count"] == 0
                and row["physical_successor_mismatch_count"] == 0
            )
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise BudgetedDynamicFilterError(
            f"budgeted dynamic panel checks failed: {checks!r}"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "status_meaning": "protocol_completed_not_all_episodes_succeeded",
        "contract_sha256": contract["contract_sha256"],
        "panel": panel,
        "row_count": len(rows),
        "episode_instance_count": len(seeds),
        "method_summary": method_summary,
        "checks": checks,
        "claim": (
            "conditional_descriptive_live_stress_of_recursive_completion_for_"
            "committed_stored_cohorts_under_one_fixed_modeled_realization_and_"
            "carried_finite_budget"
        ),
        "limitations": {
            "full_dynamic_robust_viability": False,
            "full_episode_completion_theorem": False,
            "arrival_robustness": False,
            "timing_robustness": False,
            "accept_robustness": False,
            "defer_robustness": False,
            "hardware_validation": False,
            "live_realizations_exhaustive_over_W": False,
            "live_realizations_adversarial_or_worst_case": False,
            "one_fixed_action_conditional_realization_per_recovery": True,
            "recursive_claim_is_conditional_on_WINNING_no_UNKNOWN_and_all_"
            "live_model_correspondence_checks": True,
            "timing_erased_recovery_state": True,
            "timing_metrics_reported_outside_robust_state": True,
            "finite_solver_max_expansions": MAX_EXPANSIONS,
            "unknown_fails_closed": True,
            "row_wall_times_noncomparative": True,
            "pilot_supports_confirmatory_claim": False,
            "full_panel_descriptive_stress_completed": panel == "full",
        },
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
        "rows": rows,
    }
    report["report_sha256"] = v1._digest(report)
    path = output_dir.resolve() / REPORT_NAME
    if path.exists():
        observed = v1._read(path)
        v1._verify_self_hash(
            observed, "report_sha256", label="budgeted dynamic report"
        )
        if observed != report:
            raise BudgetedDynamicFilterError(
                "existing budgeted dynamic report changed"
            )
    else:
        v1._atomic_json(path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "evaluate", "summarize", "run"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--panel", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--project-root", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        _default_output(args.panel)
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    if args.command == "prepare":
        result = prepare_contract(
            project_root,
            output_dir,
            panel=args.panel,
            device_name=args.device,
        )
        summary = {
            "status": result["status"],
            "panel": args.panel,
            "contract": str((output_dir / CONTRACT_NAME).resolve()),
        }
    elif args.command == "evaluate":
        rows = evaluate(
            project_root,
            output_dir,
            panel=args.panel,
            device_name=args.device,
        )
        summary = {
            "status": "evaluated",
            "panel": args.panel,
            "row_count": len(rows),
            "output_dir": str(output_dir),
        }
    else:
        if args.command == "run":
            evaluate(
                project_root,
                output_dir,
                panel=args.panel,
                device_name=args.device,
            )
        result = summarize(
            project_root,
            output_dir,
            panel=args.panel,
            device_name=args.device,
        )
        summary = {
            "status": result["status"],
            "panel": args.panel,
            "row_count": result["row_count"],
            "report": str((output_dir / REPORT_NAME).resolve()),
        }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
