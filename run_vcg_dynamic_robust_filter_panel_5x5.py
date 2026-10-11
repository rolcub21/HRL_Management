#!/usr/bin/env python3
"""Full-dynamic 5x5 stress test for bounded robust recovery filtering.

This additive runner reuses the frozen model-seed-0 VCG 1.1 checkpoint, its
detached handling-cost head, and the already-opened 89k EpisodeInstances.  It
does not train or select anything.  At every live decision boundary it:

1. enumerates the repository's exact-SAFE dynamic frontier;
2. applies a nominal, one-step, or recursive *recovery-only* certificate to
   Deliver/Reconfigure candidates;
3. ranks the retained frontier with one shared stateless frozen mode selector;
4. executes the selected live option; and
5. for nonterminal recovery macros, realizes a predeclared bounded delay and
   canonical adjacent-stop disturbance through the real ``env.step`` API.

Accept and Defer remain under the existing nominal exact-SAFE layer and are
not disturbed.  Consequently this is a full-dynamic recovery-filter stress
experiment, not a full dynamic robust-viability theorem.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tempfile
from time import perf_counter
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

import benchmark_viability_critic_priority as benchmark
import compare_vcg_dense_pareto as v11
from compare_viability_graph_baselines import _dual_rescore_from_legacy_return
from contention_metrics import contention_metric_record
from example.helper.timing_metrics import summarize_delivery_timing
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCertificateCache,
)
import run_vcg_final86_four_method as final86
import run_vcg_robust_recovery_snapshot_panel_5x5 as bridge
import run_vcg_v11_nested_handling_pilot as pilot
import run_vcg_v11_nested_lambda_frontier_confirmation_89k as source89
from vcg_bounded_macro_execution import (
    BoundedMacroRealization,
    LIVE_BOUNDED_MACRO_EXECUTION_CONTRACT,
    declared_clear_adjacent_stops,
    execute_bounded_macro_realization,
)
from vcg_robust_recovery_snapshot_5x5 import (
    PROTOCOL as RECOVERY_SOLVER_PROTOCOL,
    SolveStatus,
    UNCERTAINTY_CONTRACT,
    certify_snapshot,
    method_to_dict,
    recovery_action_key,
)
from vcg_v11_nested_handling import score_cost_records
from train_viability_graph_smdp import execute_certified_macro
from viability_graph_hierarchy import (
    ID_TO_MODE,
    prepare_viability_snapshot,
    regularized_mode_values,
)


PROTOCOL = "vcg_5x5_full_dynamic_bounded_recovery_filter_panel_v1"
SCHEMA_VERSION = 1
METHODS = ("nominal", "one_step", "recursive")
HANDLING_LAMBDAS = (0.0, 0.2)
MODEL_SEED = 0
EXPECTED_BLOCKS = 8
MAX_EXPANSIONS = 250
BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO = 3
UNFORCED_GUARD_CONTEXT = (0.0, 0.0, 0.0, 0.0)
PILOT_SEEDS = tuple(source89.INSTANCE_SEEDS[:4])
FULL_SEEDS = tuple(source89.INSTANCE_SEEDS)

HERE = Path(__file__).resolve().parent
DEFAULT_PILOT_OUTPUT = (
    HERE / "results/vcg-dynamic-robust-filter-panel-5x5-pilot"
)
DEFAULT_FULL_OUTPUT = HERE / "results/vcg-dynamic-robust-filter-panel-5x5"
CONTRACT_NAME = "dynamic-filter-contract.json"
REPORT_NAME = "dynamic-filter-report.json"


class DynamicFilterError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenSelection:
    candidate: ViabilityActionCandidate
    audit: Mapping[str, object]


@dataclass(frozen=True)
class ObservedExecution:
    total_primitive_steps: int
    total_raw_return: float
    delivery_deviations: tuple[float, ...]
    relocations: int
    illegal_drops: int
    env_terminal: bool
    success: bool
    failure_reason: Optional[str]
    replay_terminal: bool
    audit: Mapping[str, object]


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical(value: Mapping, *, drop: Optional[str] = None) -> bytes:
    payload = dict(value)
    if drop is not None:
        payload.pop(drop, None)
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping, *, drop: Optional[str] = None) -> str:
    return hashlib.sha256(_canonical(value, drop=drop)).hexdigest()


def _behavior_semantics(value):
    """Remove runtime/memo diagnostics from the deterministic behavior hash."""

    if isinstance(value, Mapping):
        return {
            str(key): _behavior_semantics(item)
            for key, item in value.items()
            if key not in {"certification_wall_seconds", "memo_hit"}
        }
    if isinstance(value, (tuple, list)):
        return [_behavior_semantics(item) for item in value]
    return value


def _sha(path: Path) -> str:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise DynamicFilterError(f"expected canonical regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise DynamicFilterError(f"expected canonical regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DynamicFilterError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise DynamicFilterError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping) -> None:
    path = Path(path).absolute()
    if path.is_symlink():
        raise DynamicFilterError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent:
        raise DynamicFilterError(f"output parent must be canonical: {path.parent}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_self_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, drop=field):
        raise DynamicFilterError(f"{label} self hash mismatch")


def _panel_seeds(panel: str) -> tuple[int, ...]:
    if panel == "pilot":
        return PILOT_SEEDS
    if panel == "full":
        return FULL_SEEDS
    raise ValueError(f"unknown panel: {panel}")


def _default_output(panel: str) -> Path:
    return DEFAULT_PILOT_OUTPUT if panel == "pilot" else DEFAULT_FULL_OUTPUT


def _lambda_key(value: float) -> str:
    return "0" if float(value) == 0.0 else "0p2"


def _contract(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
) -> dict:
    auth = bridge._historical_source_auth(project_root)
    seeds = _panel_seeds(panel)
    records = [
        record
        for record in auth["manifest"]["instances"]
        if int(record["instance_seed"]) in seeds
    ]
    if [int(record["instance_seed"]) for record in records] != list(seeds):
        raise DynamicFilterError("89k manifest does not contain the requested panel")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared",
        "scientific_role": (
            "full_dynamic_recovery_filter_stress_not_full_robust_viability"
        ),
        "panel": panel,
        "output_dir": str(output_dir.resolve()),
        "device": str(torch.device(device_name)),
        "instance_seeds": list(seeds),
        "episode_instance_count": len(seeds),
        "methods": list(METHODS),
        "handling_lambdas": list(HANDLING_LAMBDAS),
        "expected_rows": len(seeds) * len(METHODS) * len(HANDLING_LAMBDAS),
        "source": {
            "protocol": source89.PROTOCOL,
            "output": str(auth["source_output"]),
            "contract_sha256": auth["contract"]["contract_sha256"],
            "activation_sha256": auth["activation"]["activation_sha256"],
            "manifest_sha256": auth["manifest"]["manifest_sha256"],
            "report_sha256": auth["report"]["report_sha256"],
            "model_seed": MODEL_SEED,
            "checkpoint_sha256": source89.SELECTED_CHECKPOINT_SHA256[MODEL_SEED],
            "cost_head_sha256": source89.COST_HEAD_SHA256[MODEL_SEED],
            "records": records,
        },
        "dynamic_frontier": {
            "existing_exact_safe_verifier_authoritative": True,
            "bounded_event_defer_rule_retained": True,
            "future_schedule_hidden_from_controller": True,
            "accept_semantics": "existing_nominal_exact_safe_pass_through",
            "defer_semantics": "existing_nominal_bounded_event_pass_through",
        },
        "recovery_filter": {
            "solver_protocol": RECOVERY_SOLVER_PROTOCOL,
            "uncertainty_contract": UNCERTAINTY_CONTRACT,
            "applies_to": ["deliver", "reconfigure"],
            "replanned_at_every_live_decision_boundary": True,
            "unknown_fails_closed": True,
            "max_expansions_per_method_per_decision": MAX_EXPANSIONS,
            "budget_formula": (
                "current_nominal_witness_primitive_steps_plus_3_times_"
                "current_nominal_witness_macro_count"
            ),
            "budget_role": (
                "state_local_physical_recovery_horizon_not_block_deadline"
            ),
        },
        "selector": {
            "policy": (
                "shared_stateless_frozen_regularized_mode_map_candidate_map"
            ),
            "guard_context": list(UNFORCED_GUARD_CONTEXT),
            "retained_nominal_recovery_witness_guard": False,
            "reason": (
                "a_nominal_retained_witness_can_be_removed_by_robust_filtering_"
                "or_invalidated_by_a_realized_endpoint"
            ),
            "operational_q_frozen": True,
            "handling_q_frozen": True,
            "merit": "q_operational-lambda*q_predicted_rehandles",
            "same_selector_for_every_method_and_lambda": True,
            "epsilon": 0.0,
        },
        "live_disturbance": {
            "execution_contract": LIVE_BOUNDED_MACRO_EXECUTION_CONTRACT,
            "applies_to": ["deliver", "reconfigure"],
            "nonterminal_recovery_realization": (
                "delay_1_then_lexicographically_first_declared_clear_adjacent_"
                "stop_if_one_exists"
            ),
            "terminal_completion_realization": (
                "delay_0_only_when_selected_delivery_is_the_sole_undelivered_"
                "episode_block"
            ),
            "accept_and_defer_disturbed": False,
            "same_action_conditional_rule_for_every_arm": True,
            "live_wait_and_move_advance_clocks_and_arrivals": True,
        },
        "episode_max_steps": int(v11.MAX_STEPS),
        "training_or_learning": False,
        "checkpoint_selection": False,
        "lambda_selection": False,
        "aggregate_policy": (
            "whole_cell_metrics_only_no_complete_case_performance_average"
        ),
        "computation_reporting": {
            "comparative_measure": (
                "counterfactual_cold_selected_method_expanded_nodes_and_"
                "cutoffs_summed_per_decision"
            ),
            "wall_time_role": (
                "row_diagnostic_only_not_method_comparable_due_to_shared_memo_"
                "loop_order_and_resume_state"
            ),
        },
        "claims": {
            "full_dynamic_episode_evaluation": True,
            "full_dynamic_robust_viability": False,
            "arrival_robustness": False,
            "accept_or_defer_robustness": False,
            "hardware_robustness": False,
        },
        "source_sha256": {
            "runner": _sha(Path(__file__).resolve()),
            "recovery_solver": _sha(
                HERE / "vcg_robust_recovery_snapshot_5x5.py"
            ),
            "live_execution_adapter": _sha(
                HERE / "vcg_bounded_macro_execution.py"
            ),
            "historical_authenticator": _sha(
                HERE / "run_vcg_robust_recovery_snapshot_panel_5x5.py"
            ),
        },
    }
    payload["contract_sha256"] = _digest(payload)
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
        observed = _read(path)
        _verify_self_hash(observed, "contract_sha256", label="dynamic contract")
        if observed != expected:
            raise DynamicFilterError("dynamic contract or bound sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise DynamicFilterError("nonempty output has no dynamic contract")
        _atomic_json(path, expected)
    return expected


def _filter_recovery_frontier(
    snapshot,
    *,
    method: str,
    memo: dict,
    memo_namespace: int,
) -> tuple[frozenset[str], dict]:
    if method not in METHODS:
        raise ValueError(f"unknown filter method: {method}")
    recovery = tuple(
        candidate
        for candidate in snapshot.candidates
        if candidate.action_type
        in (ViabilityActionType.DELIVER, ViabilityActionType.RECONFIGURE)
    )
    for candidate in recovery:
        if candidate.recovery_action is None:
            raise DynamicFilterError(
                "live recovery candidate has no bound RecoveryAction"
            )
        expected_key = recovery_action_key(candidate.recovery_action)
        if candidate.key != expected_key:
            raise DynamicFilterError(
                "live recovery candidate key disagrees with robust solver key: "
                f"{candidate.key!r} != {expected_key!r}"
            )
    pass_through = tuple(
        candidate.key
        for candidate in snapshot.candidates
        if candidate.action_type
        in (ViabilityActionType.ACCEPT, ViabilityActionType.DEFER)
    )
    if not recovery:
        return frozenset(pass_through), {
            "method": method,
            "status": "NOT_APPLICABLE_NO_RECOVERY_CANDIDATE",
            "primitive_budget": None,
            "original_recovery_keys": [],
            "admitted_recovery_keys": [],
            "rejected_recovery_keys": [],
            "pass_through_keys": list(pass_through),
            "certification_wall_seconds": 0.0,
            "memo_hit": False,
        }

    current = snapshot.current_certificate
    witness_steps = current.witness_primitive_steps
    witness_macros = current.witness_macro_count
    if (
        current.status is not ViabilityStatus.SAFE
        or isinstance(witness_steps, bool)
        or not isinstance(witness_steps, int)
        or witness_steps < 1
        or isinstance(witness_macros, bool)
        or not isinstance(witness_macros, int)
        or witness_macros < 1
    ):
        return frozenset(pass_through), {
            "method": method,
            "status": "UNKNOWN_NO_FINITE_CURRENT_NOMINAL_WITNESS",
            "primitive_budget": None,
            "original_recovery_keys": [item.key for item in recovery],
            "admitted_recovery_keys": [],
            "rejected_recovery_keys": [item.key for item in recovery],
            "pass_through_keys": list(pass_through),
            "certification_wall_seconds": 0.0,
            "memo_hit": False,
        }
    budget = int(
        witness_steps
        + BUDGET_SLACK_PER_NOMINAL_WITNESS_MACRO * witness_macros
    )
    cache_key = (int(memo_namespace), snapshot.recovery_state, budget)
    started = perf_counter()
    memo_hit = cache_key in memo
    if memo_hit:
        certificate = memo[cache_key]
    else:
        certificate = certify_snapshot(
            snapshot.recovery_state,
            primitive_budget=budget,
            max_expansions=MAX_EXPANSIONS,
        )
        memo[cache_key] = certificate
    elapsed = perf_counter() - started
    method_certificate = getattr(certificate, method)
    solver_admitted_recovery = frozenset(
        method_certificate.admitted_action_keys
    )
    original_recovery = frozenset(item.key for item in recovery)
    # The recovery solver starts from every physically legal root action.  The
    # live exact-SAFE frontier can be a strict subset because it also requires
    # an executable option and an exact nominally recoverable successor.
    admitted_recovery = solver_admitted_recovery & original_recovery
    admitted = frozenset(pass_through) | admitted_recovery
    serialized = method_to_dict(method_certificate)
    solver_status = method_certificate.state.status.value
    if admitted_recovery:
        live_status = solver_status
    elif solver_status == SolveStatus.UNKNOWN.value:
        live_status = SolveStatus.UNKNOWN.value
    else:
        live_status = "NO_EXECUTABLE_ADMITTED_RECOVERY"
    return admitted, {
        "method": method,
        "status": live_status,
        "solver_state_status": solver_status,
        "primitive_budget": budget,
        "semantic_digest": certificate.semantic_digest,
        "original_recovery_keys": sorted(original_recovery),
        "solver_admitted_recovery_keys": sorted(solver_admitted_recovery),
        "admitted_recovery_keys": sorted(admitted_recovery),
        "rejected_recovery_keys": sorted(original_recovery - admitted_recovery),
        "pass_through_keys": list(pass_through),
        "method_certificate": serialized,
        "certification_wall_seconds": float(elapsed),
        "memo_hit": memo_hit,
    }


def _select_frozen(
    snapshot,
    *,
    admitted_keys: frozenset[str],
    base_agent,
    cost_network,
    handling_lambda: float,
) -> FrozenSelection:
    prepared = prepare_viability_snapshot(
        snapshot,
        guard_context=UNFORCED_GUARD_CONTEXT,
    )
    retained = tuple(
        (record, source_index)
        for record, source_index in zip(
            prepared.records, prepared.source_indices
        )
        if record.key in admitted_keys
    )
    if not retained:
        raise DynamicFilterError("method filter exposed no executable candidate")
    records = tuple(item[0] for item in retained)
    source_indices = tuple(item[1] for item in retained)
    operational = base_agent._score_records(records)
    if handling_lambda == 0.0:
        handling = torch.zeros_like(operational)
        cost_head_called = False
    else:
        handling = score_cost_records(
            cost_network,
            base_agent.Q_local,
            records,
        )
        cost_head_called = True
    merit = operational - float(handling_lambda) * handling
    if not bool(torch.isfinite(merit).all()):
        raise DynamicFilterError("frozen selector produced nonfinite merit")
    modes = torch.as_tensor(
        tuple(record.mode_id for record in records),
        dtype=torch.long,
        device=merit.device,
    )
    mode_values, live_modes = regularized_mode_values(
        merit,
        modes,
        base_agent.within_temperatures,
    )
    maximum_mode_value = float(mode_values.max().item())
    best_mode_positions = [
        index
        for index, value in enumerate(mode_values.tolist())
        if float(value) == maximum_mode_value
    ]
    mode_position = min(
        best_mode_positions,
        key=lambda index: int(live_modes[index].item()),
    )
    selected_mode = int(live_modes[mode_position].item())
    mode_indices = [
        index
        for index, mode_id in enumerate(modes.tolist())
        if int(mode_id) == selected_mode
    ]
    maximum_merit = max(float(merit[index].item()) for index in mode_indices)
    selected_index = min(
        (
            index
            for index in mode_indices
            if float(merit[index].item()) == maximum_merit
        ),
        key=lambda index: records[index].key,
    )
    candidate = snapshot.candidates[source_indices[selected_index]]
    return FrozenSelection(
        candidate=candidate,
        audit={
            "selector": (
                "shared_stateless_frozen_regularized_mode_map_candidate_map"
            ),
            "handling_lambda": float(handling_lambda),
            "guard_context": list(UNFORCED_GUARD_CONTEXT),
            "cost_head_called": cost_head_called,
            "retained_candidate_keys": [record.key for record in records],
            "q_operational": [float(value) for value in operational.tolist()],
            "q_predicted_rehandles": [
                float(value) for value in handling.tolist()
            ],
            "merit": [float(value) for value in merit.tolist()],
            "mode_values": [
                {
                    "mode": ID_TO_MODE[int(mode.item())],
                    "value": float(value.item()),
                }
                for value, mode in zip(mode_values, live_modes)
            ],
            "selected_key": candidate.key,
            "selected_operational_q": float(operational[selected_index]),
            "selected_handling_q": float(handling[selected_index]),
            "selected_merit": float(merit[selected_index]),
        },
    )


def _would_complete_live_episode(env, candidate: ViabilityActionCandidate) -> bool:
    """Return whether this delivery is the final undelivered episode block.

    The closed-admission RecoveryState deliberately excludes unarrived work, so
    an empty candidate successor is not sufficient to establish live episode
    completion.  This check reads only current delivered/identity flags, never
    an arrival time or storage duration.
    """

    if candidate.action_type is not ViabilityActionType.DELIVER:
        return False
    undelivered = tuple(block for block in env.blocks if not block.delivered)
    return bool(
        len(undelivered) == 1
        and str(undelivered[0].label) == str(candidate.target_label)
    )


def _realization(
    env,
    candidate: ViabilityActionCandidate,
) -> BoundedMacroRealization:
    if candidate.action_type not in (
        ViabilityActionType.DELIVER,
        ViabilityActionType.RECONFIGURE,
    ):
        return BoundedMacroRealization(delay_steps=0)
    # Disturbance after workload completion is vacuous and cannot be injected
    # after an authoritative terminal transition.
    if _would_complete_live_episode(env, candidate):
        return BoundedMacroRealization(delay_steps=0)
    stops = declared_clear_adjacent_stops(candidate)
    return BoundedMacroRealization(
        delay_steps=1,
        adjacent_stop=None if not stops else min(stops),
    )


def _execution_to_dict(execution) -> dict:
    return {
        "candidate_key": execution.candidate_key,
        "action_type": execution.action_type,
        "realization": asdict(execution.realization),
        "realization_id": execution.realization.realization_id,
        "declared_adjacent_stops": list(execution.declared_adjacent_stops),
        "base_primitive_steps": execution.base_primitive_steps,
        "base_replan_count": execution.base_replan_count,
        "injected_primitive_steps": execution.injected_primitive_steps,
        "total_primitive_steps": execution.total_primitive_steps,
        "start_agent_position": execution.start_agent_position,
        "base_endpoint": execution.base_endpoint,
        "realized_endpoint": execution.realized_endpoint,
        "start_time_step": execution.start_time_step,
        "base_end_time_step": execution.base_end_time_step,
        "realized_end_time_step": execution.realized_end_time_step,
        "injected_transitions": [asdict(item) for item in execution.injected_transitions],
        "injected_raw_return": execution.injected_raw_return,
        "injected_discounted_return_from_macro_start": (
            execution.injected_discounted_return_from_macro_start
        ),
        "total_raw_return": execution.total_raw_return,
        "total_discounted_return": execution.total_discounted_return,
        "env_terminal": execution.env_terminal,
        "realization_complete": execution.realization_complete,
        "failure_reason": execution.failure_reason,
        "base": {
            "duration": execution.base_execution.duration,
            "option_success": execution.base_execution.option_success,
            "failure_reason": execution.base_execution.failure_reason,
            "delivery_deviations": list(
                execution.base_execution.delivery_deviations
            ),
            "relocations": execution.base_execution.relocations,
            "illegal_drops": execution.base_execution.illegal_drops,
        },
    }


def _execute_selected(
    env,
    candidate: ViabilityActionCandidate,
    *,
    gamma: float,
    remaining_steps: int,
) -> ObservedExecution:
    recovery = candidate.action_type in (
        ViabilityActionType.DELIVER,
        ViabilityActionType.RECONFIGURE,
    )
    if not recovery:
        base = execute_certified_macro(
            env,
            candidate,
            gamma=gamma,
            remaining_steps=remaining_steps,
            evaluation=True,
        )
        audit = {
            "execution_layer": "existing_nominal_live_macro_executor",
            "recovery_disturbance_applied": False,
            "candidate_key": candidate.key,
            "action_type": candidate.action_type.value,
            "base": {
                "duration": int(base.duration),
                "discounted_return": float(base.discounted_return),
                "raw_return": float(base.raw_return),
                "env_terminal": bool(base.env_terminal),
                "truncated": bool(base.truncated),
                "option_terminated": bool(base.option_terminated),
                "option_success": bool(base.option_success),
                "failure_reason": base.failure_reason,
                "delivery_deviations": list(base.delivery_deviations),
                "relocations": int(base.relocations),
                "illegal_drops": int(base.illegal_drops),
            },
        }
        return ObservedExecution(
            total_primitive_steps=int(base.duration),
            total_raw_return=float(base.raw_return),
            delivery_deviations=tuple(float(v) for v in base.delivery_deviations),
            relocations=int(base.relocations),
            illegal_drops=int(base.illegal_drops),
            env_terminal=bool(base.env_terminal),
            success=bool(base.option_success),
            failure_reason=base.failure_reason,
            replay_terminal=bool(base.replay_terminal),
            audit=audit,
        )

    realization = _realization(env, candidate)
    if remaining_steps <= realization.requested_injected_steps:
        return ObservedExecution(
            total_primitive_steps=0,
            total_raw_return=0.0,
            delivery_deviations=(),
            relocations=0,
            illegal_drops=0,
            env_terminal=False,
            success=False,
            failure_reason="episode_horizon_cannot_fit_declared_realization",
            replay_terminal=True,
            audit={
                "execution_layer": LIVE_BOUNDED_MACRO_EXECUTION_CONTRACT,
                "recovery_disturbance_applied": True,
                "candidate_key": candidate.key,
                "action_type": candidate.action_type.value,
                "realization": asdict(realization),
                "realization_id": realization.realization_id,
                "failure_reason": (
                    "episode_horizon_cannot_fit_declared_realization"
                ),
            },
        )
    execution = execute_bounded_macro_realization(
        env,
        candidate,
        realization=realization,
        gamma=gamma,
        remaining_steps=remaining_steps,
        evaluation=True,
    )
    return ObservedExecution(
        total_primitive_steps=int(execution.total_primitive_steps),
        total_raw_return=float(execution.total_raw_return),
        delivery_deviations=tuple(
            float(v) for v in execution.base_execution.delivery_deviations
        ),
        relocations=int(execution.base_execution.relocations),
        illegal_drops=int(execution.base_execution.illegal_drops),
        env_terminal=bool(execution.env_terminal),
        success=bool(execution.realization_complete),
        failure_reason=execution.failure_reason,
        replay_terminal=bool(execution.replay_terminal),
        audit={
            "execution_layer": LIVE_BOUNDED_MACRO_EXECUTION_CONTRACT,
            "recovery_disturbance_applied": True,
            **_execution_to_dict(execution),
        },
    )


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
        raise DynamicFilterError("environment did not consume frozen EpisodeInstance")
    search_config = benchmark._search_config(arm.payload)
    liveness_rule = benchmark._liveness_rule(arm.payload)
    frontier_cache = ViabilityCertificateCache()
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
    started = perf_counter()

    while steps < v11.MAX_STEPS and not env.is_state_terminal(env.current_state):
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
        if not snapshot.candidates:
            method_failure_reason = "no_nominal_exact_safe_candidate"
            break
        if not all(
            candidate.certificate.status is ViabilityStatus.SAFE
            for candidate in snapshot.candidates
        ):
            raise DynamicFilterError("nominal exact frontier contains non-SAFE action")

        admitted, filter_audit = _filter_recovery_frontier(
            snapshot,
            method=method,
            memo=certification_memo,
            memo_namespace=int(instance.seed),
        )
        if not admitted:
            method_failure_reason = "empty_method_filtered_frontier"
            decisions.append(
                {
                    "decision_index": len(decisions),
                    "decision_epoch": int(snapshot.decision_epoch),
                    "original_candidate_keys": [
                        item.key for item in snapshot.candidates
                    ],
                    "filter": filter_audit,
                    "selected": None,
                    "execution": None,
                }
            )
            break
        selection = _select_frozen(
            snapshot,
            admitted_keys=admitted,
            base_agent=base_agent,
            cost_network=cost_network,
            handling_lambda=handling_lambda,
        )
        candidate = selection.candidate
        remaining_steps = int(v11.MAX_STEPS - steps)
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
        execution = _execute_selected(
            env,
            candidate,
            gamma=base_agent.config.gamma,
            remaining_steps=remaining_steps,
        )
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
            raise DynamicFilterError(
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

        decisions.append(
            {
                "decision_index": len(decisions),
                "decision_epoch": int(snapshot.decision_epoch),
                "original_candidate_keys": [item.key for item in snapshot.candidates],
                "filter": filter_audit,
                "selection": selection.audit,
                "execution": execution.audit,
                "post_execution_environment": benchmark._environment_signature(env),
            }
        )
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
    if steps >= v11.MAX_STEPS and not terminal and method_failure_reason is None:
        method_failure_reason = "episode_step_limit"
    success = bool(terminal and method_failure_reason is None)
    strict = bool(
        success
        and macro_failures == 0
        and illegal_drops == 0
        and complete_frontier
        and len(delivery_deviations) == EXPECTED_BLOCKS
    )
    timing = summarize_delivery_timing(
        delivery_deviations,
        v11.FROZEN_OBJECTIVE_SPEC.window,
    )
    contention = contention_metric_record(
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
    legacy_return, dense_return = _dual_rescore_from_legacy_return(
        total_return,
        delivery_deviations,
        v11.FROZEN_OBJECTIVE_SPEC,
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
        if item.get("filter", {}).get("status")
        != "NOT_APPLICABLE_NO_RECOVERY_CANDIDATE"
    ]
    status_counts = Counter(
        item["filter"]["status"] for item in robust_decisions
    )
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
        "episode_wall_seconds": float(perf_counter() - started),
        "behavior_digest": _digest(_behavior_semantics(behavior)),
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


def _validate_ledger(
    ledger: Mapping,
    *,
    contract: Mapping,
    method: str,
    handling_lambda: float,
    record: Mapping,
) -> dict:
    _verify_self_hash(ledger, "ledger_sha256", label="dynamic run ledger")
    row = ledger.get("row")
    if not isinstance(row, Mapping):
        raise DynamicFilterError("dynamic ledger has no result row")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": method,
        "handling_lambda": float(handling_lambda),
        "instance_seed": int(record["instance_seed"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "checkpoint_sha256": source89.SELECTED_CHECKPOINT_SHA256[MODEL_SEED],
        "cost_head_sha256": source89.COST_HEAD_SHA256[MODEL_SEED],
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise DynamicFilterError(f"dynamic ledger identity changed: {key}")
    if (
        row.get("instance_seed") != expected["instance_seed"]
        or row.get("episode_instance_id") != expected["episode_instance_id"]
        or row.get("schedule_id") != expected["schedule_id"]
        or row.get("method") != method
        or row.get("handling_lambda") != float(handling_lambda)
    ):
        raise DynamicFilterError("dynamic row identity changed")
    row = dict(row)
    if row.get("device") != contract.get("device"):
        raise DynamicFilterError("dynamic row device changed")
    deviations = row.get("delivery_deviations")
    decisions = row.get("decisions")
    if not isinstance(deviations, list) or not isinstance(decisions, list):
        raise DynamicFilterError("dynamic row has malformed episode sequences")
    if row.get("delivery_count") != len(deviations):
        raise DynamicFilterError("dynamic row delivery count mismatch")
    integer_fields = (
        "steps",
        "physical_rehandles",
        "illegal_drops",
        "macro_failures",
    )
    if any(
        isinstance(row.get(key), bool)
        or not isinstance(row.get(key), int)
        or row[key] < 0
        for key in integer_fields
    ):
        raise DynamicFilterError("dynamic row has invalid non-negative counts")
    numeric_fields = (
        "raw_environment_return",
        "return",
        "dense_return",
        "mean_absolute_error",
        "physical_rehandles_per_100",
    )
    if any(
        isinstance(row.get(key), bool)
        or not isinstance(row.get(key), (int, float))
        or not math.isfinite(float(row[key]))
        for key in numeric_fields
    ):
        raise DynamicFilterError("dynamic row has invalid finite metrics")
    expected_rehandles = 100.0 * row["physical_rehandles"] / EXPECTED_BLOCKS
    if not math.isclose(
        float(row["physical_rehandles_per_100"]),
        expected_rehandles,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DynamicFilterError("dynamic row rehandle normalization mismatch")
    expected_legacy, expected_dense = _dual_rescore_from_legacy_return(
        float(row["raw_environment_return"]),
        deviations,
        v11.FROZEN_OBJECTIVE_SPEC,
    )
    if not math.isclose(float(row["return"]), expected_legacy, abs_tol=1e-9):
        raise DynamicFilterError("dynamic row legacy return mismatch")
    if not math.isclose(
        float(row["dense_return"]), expected_dense, abs_tol=1e-9
    ):
        raise DynamicFilterError("dynamic row dense return mismatch")
    if row.get("strict_safe_complete"):
        strict_invariants = (
            row.get("terminal") is True,
            row.get("success") is True,
            row.get("method_failure_reason") is None,
            len(deviations) == EXPECTED_BLOCKS,
            row["macro_failures"] == 0,
            row["illegal_drops"] == 0,
            row.get("complete_nominal_frontier_exactly_verified") is True,
        )
        if not all(strict_invariants):
            raise DynamicFilterError("strict dynamic row violates completion invariants")
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise DynamicFilterError("dynamic decision audit is malformed")
        selection = decision.get("selection")
        if selection is None:
            continue
        selected_key = selection.get("selected_key")
        if selected_key not in decision.get("original_candidate_keys", ()):
            raise DynamicFilterError("selected key was absent from nominal frontier")
        filter_audit = decision.get("filter", {})
        admitted = set(filter_audit.get("pass_through_keys", ())) | set(
            filter_audit.get("admitted_recovery_keys", ())
        )
        if selected_key not in admitted:
            raise DynamicFilterError("selected key was absent from filtered frontier")
        execution = decision.get("execution")
        if not isinstance(execution, Mapping) or execution.get(
            "candidate_key"
        ) != selected_key:
            raise DynamicFilterError("execution audit disagrees with selection")
    behavior = {
        "instance_seed": row["instance_seed"],
        "episode_instance_id": row["episode_instance_id"],
        "method": row["method"],
        "handling_lambda": row["handling_lambda"],
        "decisions": decisions,
        "final_environment": row.get("final_environment"),
        "return": float(row["raw_environment_return"]),
        "steps": row["steps"],
        "terminal": row["terminal"],
        "failure": row["method_failure_reason"],
    }
    if row.get("behavior_digest") != _digest(_behavior_semantics(behavior)):
        raise DynamicFilterError("dynamic row behavior digest mismatch")
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
    auth = bridge._historical_source_auth(project_root)
    device = pilot._device(device_name)
    final86._configure_runtime()
    base_agent = pilot._fresh_base(auth["arm"], device)
    cost_network = source89._load_bound_cost(
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
    # Semantic certificates are immutable and independent of preference.  The
    # namespace in each key prevents accidental cross-instance aliasing.  This
    # shared memo makes the fixed 3x2 comparison practical without changing a
    # certificate or its reported logical solver work.
    certification_memo: dict = {}
    total = len(records) * len(METHODS) * len(HANDLING_LAMBDAS)
    completed = 0
    for method in METHODS:
        for handling_lambda in HANDLING_LAMBDAS:
            for record in records:
                path = _ledger_path(
                    output_dir,
                    method=method,
                    handling_lambda=handling_lambda,
                    instance_seed=int(record["instance_seed"]),
                )
                if path.exists():
                    ledger = _read(path)
                    row = _validate_ledger(
                        ledger,
                        contract=contract,
                        method=method,
                        handling_lambda=handling_lambda,
                        record=record,
                    )
                else:
                    instance = source89._load_instance(auth["source_output"], record)
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
                        "checkpoint_sha256": source89.SELECTED_CHECKPOINT_SHA256[
                            MODEL_SEED
                        ],
                        "cost_head_sha256": source89.COST_HEAD_SHA256[MODEL_SEED],
                        "row": row,
                    }
                    ledger["ledger_sha256"] = _digest(ledger)
                    _validate_ledger(
                        ledger,
                        contract=contract,
                        method=method,
                        handling_lambda=handling_lambda,
                        record=record,
                    )
                    _atomic_json(path, ledger)
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
    rows = tuple(rows)
    strict = [row for row in rows if row["strict_safe_complete"]]
    all_strict = len(strict) == len(rows)
    failures = Counter(
        str(row.get("method_failure_reason") or "none") for row in rows
    )
    metrics = None
    if all_strict:
        metrics = {
            "mean_dense_return": float(fmean(row["dense_return"] for row in rows)),
            "mean_absolute_error": float(
                fmean(row["mean_absolute_error"] for row in rows)
            ),
            "mean_steps": float(fmean(row["steps"] for row in rows)),
            "physical_rehandles_per_100": float(
                100.0
                * sum(row["physical_rehandles"] for row in rows)
                / (len(rows) * EXPECTED_BLOCKS)
            ),
            "within_target_window_rate": float(
                fmean(row["within_target_window_rate"] for row in rows)
            ),
        }
    return {
        "row_count": len(rows),
        "strict_safe_complete_count": len(strict),
        "terminal_count": sum(bool(row["terminal"]) for row in rows),
        "whole_cell_performance_eligible": all_strict,
        "metrics": metrics,
        "failure_reason_counts": dict(sorted(failures.items())),
        "total_recovery_filter_decisions": sum(
            row["recovery_filter_decision_count"] for row in rows
        ),
        "total_recovery_candidates_removed": sum(
            row["recovery_candidates_removed"] for row in rows
        ),
        "total_macro_failures": sum(row["macro_failures"] for row in rows),
        "total_illegal_drops": sum(row["illegal_drops"] for row in rows),
        "total_logical_recovery_expanded_nodes": sum(
            row["logical_recovery_expanded_nodes"] for row in rows
        ),
        "total_recovery_decisions_hitting_compute_cutoff": sum(
            row["recovery_decisions_hitting_compute_cutoff"] for row in rows
        ),
    }


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
    auth = bridge._historical_source_auth(project_root)
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
                    raise DynamicFilterError(f"missing dynamic ledger: {path}")
                rows.append(
                    _validate_ledger(
                        _read(path),
                        contract=contract,
                        method=method,
                        handling_lambda=handling_lambda,
                        record=record,
                    )
                )
    expected = int(contract["expected_rows"])
    if len(rows) != expected:
        raise DynamicFilterError("dynamic row grid is incomplete")
    cell_summary = {
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
        "unknown_never_described_as_winning": all(
            not (
                decision.get("filter", {}).get("status")
                == SolveStatus.UNKNOWN.value
                and decision.get("selection", {}).get("selected_key")
                in decision.get("filter", {}).get("rejected_recovery_keys", ())
            )
            for row in rows
            for decision in row["decisions"]
        ),
    }
    if not all(checks.values()):
        raise DynamicFilterError(f"dynamic panel checks failed: {checks!r}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "status_meaning": "protocol_completed_not_all_episodes_succeeded",
        "contract_sha256": contract["contract_sha256"],
        "panel": panel,
        "row_count": len(rows),
        "episode_instance_count": len(seeds),
        "method_summary": cell_summary,
        "checks": checks,
        "claim": (
            "frozen-policy_full-dynamic_descriptive_stress_under_recovery-only_"
            "bounded_filtering_and_live_recovery_disturbances"
        ),
        "limitations": {
            "full_dynamic_robust_viability": False,
            "accept_robustness": False,
            "defer_robustness": False,
            "future_arrivals_in_recursive_certificate": False,
            "live_wait_advances_clocks_but_snapshot_delay_only_charges_budget": True,
            "retained_nominal_witness_guard_used": False,
            "selector_is_shared_stateless_frozen_mode_aggregation": True,
            "row_wall_times_are_noncomparative_memo_order_resume_diagnostics": (
                True
            ),
            "hardware_validation": False,
            "pilot_supports_confirmatory_claim": False,
            "full_panel_descriptive_stress_completed": panel == "full",
        },
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
        "rows": rows,
    }
    report["report_sha256"] = _digest(report)
    path = output_dir.resolve() / REPORT_NAME
    if path.exists():
        observed = _read(path)
        _verify_self_hash(observed, "report_sha256", label="dynamic report")
        if observed != report:
            raise DynamicFilterError("existing dynamic report changed")
    else:
        _atomic_json(path, report)
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
