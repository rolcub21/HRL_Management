#!/usr/bin/env python3
"""Exhaustive live 5x5 disturbance trees for one committed recovery cohort.

This additive evaluation reuses the frozen policies, frozen 89k
EpisodeInstances, exact RecoveryAction executor, finite disturbance envelope,
and carried-budget recovery filters from the completed v2 dynamic panel.  It
does not train or select a checkpoint, lambda, instance, or root state.

For every selected panel instance, an authenticated pre-action snapshot from
the source lambda-zero trajectory is replayed through the authoritative
environment.  Admission is then closed exogenously, the blocks already in
storage become the committed cohort, and that common state and primitive
budget are crossed with all declared filter/lambda arms.
At every reachable active recovery boundary the frozen policy selects once,
the environment is deep-copied once per declared disturbance, the exact bound
macro and disturbance are sent through ``env.step``, and every realized
successor is independently replanned until the committed cohort is empty.

The result is a memoized live-state DAG, not a sampled trajectory.  A resource
cutoff is UNKNOWN, never PASS or FAIL.  Globally terminal delivery receives an
explicit absorbing-terminal quotient: every abstract post-terminal outcome is
accounted for, but all such outcomes map to the sole executable terminal
transition because the authoritative environment forbids actions after
termination.  The report therefore distinguishes abstract branch coverage,
live equivalence classes, and terminal aliases.

The scope remains recovery-only.  Accept/Defer, future-arrival uncertainty,
block-clock disturbances, interior path errors, and hardware faults are not in
the declared uncertainty set and receive no robustness claim here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from PSLAP.viability import RecoveryAction, RecoveryState, ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCertificateCache,
)
import run_vcg_dynamic_budgeted_robust_filter_panel_5x5 as v2
import run_vcg_dynamic_robust_filter_panel_5x5 as v1
import run_vcg_robust_recovery_snapshot_panel_5x5 as snapshot_bridge
from vcg_bounded_macro_execution import BoundedMacroRealization
from vcg_exact_recovery_execution import EXACT_RECOVERY_EXECUTION_CONTRACT
from vcg_robust_recovery_snapshot_5x5 import (
    ROBUST_CONTRACT,
    SolveStatus,
    execution_envelope,
    recovery_state_to_dict,
)


PROTOCOL = "vcg_5x5_exhaustive_live_committed_recovery_tree_v1"
SCHEMA_VERSION = 1
CARRIER_METHOD = "source_vcg_lambda_zero"
CARRIER_LAMBDA = 0.0
ROOT_INDEX = 0
METHODS = v2.METHODS
HANDLING_LAMBDAS = v2.HANDLING_LAMBDAS
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_INVALID = "INVALID"
STATUS_ORDER = {
    STATUS_PASS: 0,
    STATUS_UNKNOWN: 1,
    STATUS_FAIL: 2,
    STATUS_INVALID: 3,
}
TERMINAL_EQUIVALENCE_CONTRACT = (
    "global_terminal_delivery_maps_all_declared_post_macro_outcomes_to_one_"
    "absorbing_live_transition_with_every_abstract_id_accounted_v1"
)

HERE = Path(__file__).resolve().parent
PARENT_SNAPSHOT_OUTPUT = (
    HERE / "results/vcg-robust-recovery-snapshot-panel-5x5"
)
SMOKE_SEEDS = (89000,)
PILOT_SEEDS = (89000, 89004, 89008, 89012, 89016, 89020, 89024, 89028)
DEFAULT_SMOKE_OUTPUT = (
    HERE / "results/vcg-exhaustive-live-recovery-tree-5x5-smoke-v1"
)
DEFAULT_PILOT_OUTPUT = (
    HERE / "results/vcg-exhaustive-live-recovery-tree-5x5-pilot-v1"
)
DEFAULT_FULL_OUTPUT = (
    HERE / "results/vcg-exhaustive-live-recovery-tree-5x5-v1"
)
CONTRACT_NAME = "exhaustive-live-tree-contract.json"
REPORT_NAME = "exhaustive-live-tree-report.json"


class ExhaustiveLiveTreeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeLimits:
    max_unique_nodes_per_root: int

    def __post_init__(self) -> None:
        value = self.max_unique_nodes_per_root
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_unique_nodes_per_root must be positive")


@dataclass(frozen=True)
class RootSnapshot:
    instance_seed: int
    episode_instance_id: str
    schedule_id: str
    initial_budget: int
    cohort_labels: tuple[str, ...]
    timing_erased_state: RecoveryState
    environment: object
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class VisitResult:
    node_id: Optional[str]
    status: str
    summary: Mapping[str, object]


def _panel_seeds(panel: str) -> tuple[int, ...]:
    if panel == "smoke":
        return SMOKE_SEEDS
    if panel == "pilot":
        return PILOT_SEEDS
    if panel == "full":
        return tuple(v2.FULL_SEEDS)
    raise ValueError(f"unknown panel: {panel}")


def _default_output(panel: str) -> Path:
    if panel == "smoke":
        return DEFAULT_SMOKE_OUTPUT
    if panel == "pilot":
        return DEFAULT_PILOT_OUTPUT
    if panel == "full":
        return DEFAULT_FULL_OUTPUT
    raise ValueError(f"unknown panel: {panel}")


def _default_node_limit(panel: str) -> int:
    if panel == "smoke":
        return 256
    if panel == "pilot":
        return 20_000
    if panel == "full":
        return 100_000
    raise ValueError(f"unknown panel: {panel}")


def _lambda_key(value: float) -> str:
    return v2._lambda_key(value)


def _block_causal_state(block) -> dict:
    fields = (
        "label",
        "position",
        "storage_location",
        "carrying",
        "stored",
        "delivered",
        "picked",
        "cleared",
        "hold",
        "hold_counter",
        "storage_steps_needed",
        "storage_steps_elapsed",
        "arrival_step",
        "stored_time_step",
        "delivered_time_step",
        "delivery_error_time",
        "last_position",
    )
    return {
        name: v1._json_safe(getattr(block, name, None))
        for name in fields
    }


def live_causal_state(
    env,
    *,
    remaining_budget: int,
    cohort_labels: Sequence[str],
    method: str,
    handling_lambda: float,
) -> dict:
    """Return every live value that can affect active recovery continuation."""

    return {
        "episode_instance_id": env.current_episode_instance.instance_id,
        "time_steps": int(env.time_steps),
        "agent_position": list(env.current_state),
        "terminal": bool(env.is_state_terminal(env.current_state)),
        "blocks": [
            _block_causal_state(block)
            for block in sorted(env.blocks, key=lambda item: str(item.label))
        ],
        "storage_counts": [
            [list(cell), int(count)]
            for cell, count in sorted(env.storage_counts.items())
        ],
        "store_events": v1._json_safe(env.store_events),
        "delivery_error_times": v1._json_safe(env.delivery_error_times),
        "remaining_budget": int(remaining_budget),
        "cohort_labels": list(sorted(str(value) for value in cohort_labels)),
        "method": str(method),
        "handling_lambda": float(handling_lambda),
        "consecutive_defer": 0,
    }


def outcome_realization(
    action: RecoveryAction,
    outcome,
) -> BoundedMacroRealization:
    stop = tuple(outcome.stop_cell)
    destination = tuple(action.destination)
    return BoundedMacroRealization(
        delay_steps=int(outcome.delay_steps),
        adjacent_stop=None if stop == destination else stop,
    )


def _structural_envelope(
    state: RecoveryState,
    action: RecoveryAction,
) -> tuple:
    outcomes = execution_envelope(
        state,
        action,
        remaining_primitive_steps=10**9,
        contract=ROBUST_CONTRACT,
    )
    return tuple(sorted(outcomes, key=lambda item: item.disturbance_id))


def _empty_path_summary(*, status: str, reason: str) -> dict:
    completed = int(status == STATUS_PASS)
    return {
        "total_path_count": 1,
        "completed_path_count": completed,
        "status_path_counts": {status: 1},
        "witnesses": {} if status == STATUS_PASS else {status: []},
        "completed_extrema": (
            {
                "minimum_raw_return": {"value": 0.0, "path": []},
                "maximum_live_steps": {"value": 0, "path": []},
                "maximum_physical_rehandles": {"value": 0, "path": []},
                "maximum_absolute_error_sum": {
                    "value": 0.0,
                    "delivery_count": 0,
                    "path": [],
                },
            }
            if completed
            else None
        ),
        "leaf_reason": reason,
    }


def _status_from_children(statuses: Sequence[str]) -> str:
    if not statuses:
        return STATUS_INVALID
    return max(statuses, key=lambda value: STATUS_ORDER[value])


def _prepend_path(path, disturbance_id: str) -> list[str]:
    return [str(disturbance_id), *[str(value) for value in path]]


def _combine_edge_summaries(edges: Sequence[Mapping]) -> dict:
    total_paths = 0
    completed_paths = 0
    status_counts: Counter[str] = Counter()
    witnesses: dict[str, list[str]] = {}
    completed_candidates = []
    for edge in edges:
        disturbance_id = str(edge["disturbance_id"])
        child = edge["child_summary"]
        total_paths += int(child["total_path_count"])
        completed_paths += int(child["completed_path_count"])
        for status, count in child["status_path_counts"].items():
            status_counts[str(status)] += int(count)
        for status, path in child.get("witnesses", {}).items():
            candidate = _prepend_path(path, disturbance_id)
            current = witnesses.get(status)
            if current is None or (len(candidate), candidate) < (len(current), current):
                witnesses[status] = candidate
        extrema = child.get("completed_extrema")
        if extrema is None:
            continue
        execution = edge["execution_metrics"]
        completed_candidates.append(
            {
                "path_prefix": disturbance_id,
                "return": float(execution["raw_return"])
                + float(extrema["minimum_raw_return"]["value"]),
                "return_path": _prepend_path(
                    extrema["minimum_raw_return"]["path"], disturbance_id
                ),
                "steps": int(execution["live_steps"])
                + int(extrema["maximum_live_steps"]["value"]),
                "steps_path": _prepend_path(
                    extrema["maximum_live_steps"]["path"], disturbance_id
                ),
                "rehandles": int(execution["physical_rehandles"])
                + int(extrema["maximum_physical_rehandles"]["value"]),
                "rehandles_path": _prepend_path(
                    extrema["maximum_physical_rehandles"]["path"],
                    disturbance_id,
                ),
                "absolute_error_sum": float(execution["absolute_error_sum"])
                + float(extrema["maximum_absolute_error_sum"]["value"]),
                "delivery_count": int(execution["delivery_count"])
                + int(extrema["maximum_absolute_error_sum"]["delivery_count"]),
                "absolute_error_path": _prepend_path(
                    extrema["maximum_absolute_error_sum"]["path"],
                    disturbance_id,
                ),
            }
        )
    extrema = None
    if completed_candidates:
        minimum_return = min(
            completed_candidates,
            key=lambda item: (item["return"], item["return_path"]),
        )
        maximum_steps = max(
            completed_candidates,
            key=lambda item: (item["steps"], tuple(reversed(item["steps_path"]))),
        )
        maximum_rehandles = max(
            completed_candidates,
            key=lambda item: (
                item["rehandles"],
                tuple(reversed(item["rehandles_path"])),
            ),
        )
        maximum_abs = max(
            completed_candidates,
            key=lambda item: (
                item["absolute_error_sum"],
                tuple(reversed(item["absolute_error_path"])),
            ),
        )
        extrema = {
            "minimum_raw_return": {
                "value": minimum_return["return"],
                "path": minimum_return["return_path"],
            },
            "maximum_live_steps": {
                "value": maximum_steps["steps"],
                "path": maximum_steps["steps_path"],
            },
            "maximum_physical_rehandles": {
                "value": maximum_rehandles["rehandles"],
                "path": maximum_rehandles["rehandles_path"],
            },
            "maximum_absolute_error_sum": {
                "value": maximum_abs["absolute_error_sum"],
                "delivery_count": maximum_abs["delivery_count"],
                "mean_absolute_error": (
                    None
                    if maximum_abs["delivery_count"] == 0
                    else maximum_abs["absolute_error_sum"]
                    / maximum_abs["delivery_count"]
                ),
                "path": maximum_abs["absolute_error_path"],
            },
        }
    return {
        "total_path_count": int(total_paths),
        "completed_path_count": int(completed_paths),
        "status_path_counts": dict(sorted(status_counts.items())),
        "witnesses": dict(sorted(witnesses.items())),
        "completed_extrema": extrema,
        "leaf_reason": None,
    }


def _read_authenticated_parent(
    project_root: Path,
    panel: str,
) -> tuple[dict, dict, dict]:
    del project_root
    parent_dir = PARENT_SNAPSHOT_OUTPUT.resolve()
    contract_path = parent_dir / snapshot_bridge.CONTRACT_NAME
    panel_path = parent_dir / snapshot_bridge.PANEL_NAME
    report_path = parent_dir / snapshot_bridge.REPORT_NAME
    if not contract_path.exists() or not panel_path.exists() or not report_path.exists():
        raise ExhaustiveLiveTreeError(
            f"authenticated snapshot-panel artifacts are missing from {parent_dir}"
        )
    contract = v1._read(contract_path)
    snapshot_panel = v1._read(panel_path)
    report = v1._read(report_path)
    snapshot_bridge._verify_self_hash(
        contract,
        "contract_sha256",
        label="parent snapshot bridge contract",
    )
    snapshot_bridge._verify_self_hash(
        snapshot_panel,
        "panel_sha256",
        label="parent snapshot panel",
    )
    snapshot_bridge._verify_self_hash(
        report,
        "report_sha256",
        label="parent snapshot bridge report",
    )
    if any(
        item.get("protocol") != snapshot_bridge.PROTOCOL
        for item in (contract, snapshot_panel, report)
    ):
        raise ExhaustiveLiveTreeError("snapshot parent protocol changed")
    if snapshot_panel.get("contract_sha256") != contract.get("contract_sha256"):
        raise ExhaustiveLiveTreeError("snapshot panel/contract binding changed")
    if report.get("contract_sha256") != contract.get("contract_sha256"):
        raise ExhaustiveLiveTreeError("snapshot report/contract binding changed")
    if report.get("panel_sha256") != snapshot_panel.get("panel_sha256"):
        raise ExhaustiveLiveTreeError("snapshot report/panel binding changed")
    if report.get("status") != "passed" or snapshot_panel.get("snapshot_count") != 30:
        raise ExhaustiveLiveTreeError("snapshot parent did not pass its protocol")
    expected = set(_panel_seeds(panel))
    observed = {
        int(item["instance_seed"])
        for item in snapshot_panel["snapshots"]
        if int(item["instance_seed"]) in expected
    }
    if observed != expected:
        raise ExhaustiveLiveTreeError("snapshot panel lacks requested roots")
    if panel in ("smoke", "pilot") and any(
        int(item["observed_occupancy"]) != 2
        for item in snapshot_panel["snapshots"]
        if int(item["instance_seed"]) in expected
    ):
        raise ExhaustiveLiveTreeError("staged smoke/pilot roots are not occupancy two")
    return contract, snapshot_panel, report


def _snapshot_row(snapshot_panel: Mapping, instance_seed: int) -> Mapping:
    rows = tuple(
        row
        for row in snapshot_panel["snapshots"]
        if int(row["instance_seed"]) == int(instance_seed)
    )
    if len(rows) != 1:
        raise ExhaustiveLiveTreeError(
            f"snapshot panel has {len(rows)} roots for seed {instance_seed}"
        )
    return rows[0]


class _RootCaptured(Exception):
    def __init__(self, environment, snapshot, selected_key: str) -> None:
        super().__init__("authenticated live root captured")
        self.environment = environment
        self.snapshot = snapshot
        self.selected_key = selected_key


def _root_capture_factory(*, target: Mapping, env_holder: dict):
    def factory(base_agent):
        original_select = base_agent.select
        local_decision_index = 0

        def select(_self, snapshot, *, training=True, epsilon=None):
            nonlocal local_decision_index
            decision_index = local_decision_index
            local_decision_index += 1
            decision = original_select(
                snapshot,
                training=training,
                epsilon=epsilon,
            )
            if decision_index != int(target["source_decision_index"]):
                return decision
            checks = {
                "decision_epoch": int(snapshot.decision_epoch)
                == int(target["decision_epoch"]),
                "occupancy": len(snapshot.recovery_state.blocks)
                == int(target["observed_occupancy"]),
                "recovery_state_digest": snapshot_bridge._digest(
                    recovery_state_to_dict(snapshot.recovery_state)
                )
                == target["source_recovery_state_digest"],
                "candidate_keys": [item.key for item in snapshot.candidates]
                == list(target["source_frontier"]["candidate_keys"]),
                "frontier_complete": bool(
                    snapshot.audit.complete_frontier_exactly_verified
                ),
                "selected_key": decision.candidate.key
                == target["authenticated_source_selected_key"],
                "liveness_forced": bool(decision.liveness_forced)
                == bool(target["authenticated_source_liveness_forced"]),
            }
            if not all(checks.values()):
                raise ExhaustiveLiveTreeError(
                    f"authenticated snapshot root replay changed: {checks!r}"
                )
            env = env_holder.get("environment")
            if env is None:
                raise ExhaustiveLiveTreeError("live root replay did not expose env")
            raise _RootCaptured(
                deepcopy(env),
                snapshot,
                decision.candidate.key,
            )

        base_agent.select = MethodType(select, base_agent)
        return base_agent

    return factory


def replay_common_root(
    *,
    arm,
    instance,
    snapshot_row: Mapping,
    device: torch.device,
) -> RootSnapshot:
    """Replay and capture one predeclared, authenticated live snapshot root."""

    env_holder: dict = {}
    original_make_env = benchmark._make_env

    def exposing_make_env(payload):
        env = original_make_env(payload)
        env_holder["environment"] = env
        return env

    try:
        with patch.object(benchmark, "_make_env", exposing_make_env):
            v1.pilot._run_raw(
                arm,
                instance,
                device=device,
                wrapper_factory=_root_capture_factory(
                    target=snapshot_row,
                    env_holder=env_holder,
                ),
            )
    except _RootCaptured as captured:
        timing_state = v2._timing_erased_state(captured.snapshot.recovery_state)
        cohort = tuple(sorted(block.label for block in timing_state.blocks))
        if len(cohort) != int(snapshot_row["observed_occupancy"]):
            raise ExhaustiveLiveTreeError("captured root cohort size changed")
        causal = live_causal_state(
            captured.environment,
            remaining_budget=int(snapshot_row["primitive_budget"]),
            cohort_labels=cohort,
            method=CARRIER_METHOD,
            handling_lambda=CARRIER_LAMBDA,
        )
        provenance = {
            "source_root_replay_authenticated": True,
            "carrier_method": CARRIER_METHOD,
            "carrier_handling_lambda": CARRIER_LAMBDA,
            "source_decision_index": int(snapshot_row["source_decision_index"]),
            "source_decision_epoch": int(snapshot_row["decision_epoch"]),
            "source_selected_key_before_admission_closure": captured.selected_key,
            "source_recovery_state_digest": snapshot_row[
                "source_recovery_state_digest"
            ],
            "source_behavior_digest": snapshot_row["source_behavior_digest"],
            "source_ledger_sha256": snapshot_row["source_ledger_sha256"],
            "observed_occupancy": int(snapshot_row["observed_occupancy"]),
            "root_environment_digest": v1._digest(causal),
            "root_environment": causal,
        }
        return RootSnapshot(
            instance_seed=int(instance.seed),
            episode_instance_id=instance.instance_id,
            schedule_id=instance.schedule_id,
            initial_budget=int(snapshot_row["primitive_budget"]),
            cohort_labels=cohort,
            timing_erased_state=timing_state,
            environment=captured.environment,
            provenance=provenance,
        )
    raise ExhaustiveLiveTreeError("source replay passed the target without capture")


class SegmentTreeExplorer:
    """Memoized exhaustive live disturbance explorer for one common root."""

    def __init__(
        self,
        *,
        root: RootSnapshot,
        method: str,
        handling_lambda: float,
        base_agent,
        cost_network,
        search_config,
        liveness_rule,
        limits: TreeLimits,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        if float(handling_lambda) not in HANDLING_LAMBDAS:
            raise ValueError(f"unknown handling lambda: {handling_lambda}")
        self.root = root
        self.method = method
        self.handling_lambda = float(handling_lambda)
        self.base_agent = base_agent
        self.cost_network = cost_network
        self.search_config = search_config
        self.liveness_rule = liveness_rule
        self.limits = limits
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._seen: dict[str, str] = {}
        self._results: dict[str, VisitResult] = {}
        self._certification_memo: dict = {}
        self.node_cutoff_count = 0
        self.deduplicated_successor_count = 0
        self.abstract_outcome_count = 0
        self.live_execution_count = 0
        self.terminal_alias_count = 0
        self.started = perf_counter()

    def _causal(self, env, remaining_budget: int) -> dict:
        return live_causal_state(
            env,
            remaining_budget=remaining_budget,
            cohort_labels=self.root.cohort_labels,
            method=self.method,
            handling_lambda=self.handling_lambda,
        )

    def _node_key(self, env, remaining_budget: int) -> tuple[str, dict]:
        causal = self._causal(env, remaining_budget)
        return v1._digest(causal), causal

    def _new_node_id(self, digest: str) -> str:
        base = f"node-{digest[:20]}"
        if base not in self.nodes:
            return base
        if self.nodes[base]["causal_digest"] != digest:
            raise ExhaustiveLiveTreeError("live node digest prefix collided")
        return base

    def _cutoff(self, causal_digest: str) -> VisitResult:
        self.node_cutoff_count += 1
        return VisitResult(
            node_id=None,
            status=STATUS_UNKNOWN,
            summary=_empty_path_summary(
                status=STATUS_UNKNOWN,
                reason=(
                    "deterministic_unique_node_limit_reached:"
                    f"{causal_digest[:16]}"
                ),
            ),
        )

    def _leaf(
        self,
        node: dict,
        *,
        status: str,
        reason: str,
    ) -> VisitResult:
        node["status"] = status
        node["reason"] = reason
        node["expected_disturbance_ids"] = []
        node["edge_indices"] = []
        summary = _empty_path_summary(status=status, reason=reason)
        node["path_summary"] = summary
        result = VisitResult(node["node_id"], status, summary)
        self._results[node["node_id"]] = result
        return result

    def _execution_metrics(self, execution) -> dict:
        deviations = [float(value) for value in execution.delivery_deviations]
        return {
            "raw_return": float(execution.total_raw_return),
            "live_steps": int(execution.total_primitive_steps),
            "physical_rehandles": int(execution.relocations),
            "delivery_count": len(deviations),
            "absolute_error_sum": float(sum(abs(value) for value in deviations)),
            "delivery_deviations": deviations,
        }

    def _exact_summary(self, execution) -> dict:
        audit = execution.audit
        exact_path = audit["exact_path"]
        summary = {
            "candidate_key": audit["candidate_key"],
            "realization_id": audit["realization_id"],
            "base_primitive_steps": audit["base_primitive_steps"],
            "injected_primitive_steps": audit["injected_primitive_steps"],
            "total_primitive_steps": audit["total_primitive_steps"],
            "realization_complete": audit["realization_complete"],
            "failure_reason": audit["failure_reason"],
            "trace_matches_certified_action": exact_path[
                "trace_matches_certified_action"
            ],
            "planner_calls": exact_path["planner_calls"],
            "exact_executor_replan_count": exact_path["replan_count"],
            "primitive_trace_length": len(exact_path["expected_action_names"]),
            "exact_trace_sha256": v1._digest(
                {
                    "expected_action_names": exact_path["expected_action_names"],
                    "emitted_action_names": exact_path["emitted_action_names"],
                    "expected_cell_trace": exact_path["expected_cell_trace"],
                    "observed_cell_trace": exact_path["observed_cell_trace"],
                }
            ),
        }
        if not exact_path["trace_matches_certified_action"]:
            summary["mismatched_trace"] = {
                "expected_action_names": exact_path["expected_action_names"],
                "emitted_action_names": exact_path["emitted_action_names"],
                "expected_cell_trace": exact_path["expected_cell_trace"],
                "observed_cell_trace": exact_path["observed_cell_trace"],
            }
        return summary

    def _execute_branch(
        self,
        *,
        env,
        state: RecoveryState,
        candidate: ViabilityActionCandidate,
        action: RecoveryAction,
        outcome,
        remaining_budget: int,
        depth: int,
    ) -> dict:
        branch_env = deepcopy(env)
        before_digest, _ = self._node_key(branch_env, remaining_budget)
        expected_before, _ = self._node_key(env, remaining_budget)
        if before_digest != expected_before:
            raise ExhaustiveLiveTreeError("deep-copied branch changed causal state")

        declared_realization = outcome_realization(action, outcome)
        global_terminal_delivery = v1._would_complete_live_episode(
            branch_env,
            candidate,
        )
        terminal_alias = bool(
            global_terminal_delivery
            and declared_realization.requested_injected_steps > 0
        )
        effective_realization = (
            BoundedMacroRealization(delay_steps=0)
            if terminal_alias
            else declared_realization
        )
        live_horizon = int(v1.v11.MAX_STEPS - branch_env.time_steps)
        required_live = int(action.steps + effective_realization.requested_injected_steps)
        if required_live > live_horizon:
            summary = _empty_path_summary(
                status=STATUS_FAIL,
                reason="declared_realization_exceeds_remaining_live_episode_horizon",
            )
            return {
                "status": STATUS_FAIL,
                "child_id": None,
                "child_summary": summary,
                "execution_metrics": {
                    "raw_return": 0.0,
                    "live_steps": 0,
                    "physical_rehandles": 0,
                    "delivery_count": 0,
                    "absolute_error_sum": 0.0,
                    "delivery_deviations": [],
                },
                "audit": {
                    "live_execution_attempted": False,
                    "reason": "insufficient_live_episode_horizon",
                },
                "terminal_alias": terminal_alias,
            }

        execution = v2._execute_exact_selected(
            branch_env,
            candidate,
            realization=effective_realization,
            gamma=self.base_agent.config.gamma,
            remaining_steps=live_horizon,
        )
        self.live_execution_count += 1
        if terminal_alias:
            self.terminal_alias_count += 1
        exact = self._exact_summary(execution)
        metrics = self._execution_metrics(execution)
        expected_live_steps = required_live
        checks = {
            "execution_success": bool(execution.success),
            "exact_trace_matches": bool(exact["trace_matches_certified_action"]),
            "planner_calls_zero": exact["planner_calls"] == 0,
            "replans_zero": exact["exact_executor_replan_count"] == 0,
            "live_duration_matches": int(execution.total_primitive_steps)
            == expected_live_steps,
            "illegal_drops_zero": int(execution.illegal_drops) == 0,
        }
        actual_state = None
        physical = None
        try:
            actual_state = v2._live_recovery_state(
                branch_env,
                template=state,
                search_config=self.search_config,
            )
            expected_state = (
                v2._timing_erased_state(candidate.successor_state)
                if terminal_alias
                else outcome.successor
            )
            if expected_state is None:
                raise ExhaustiveLiveTreeError(
                    "structural outcome unexpectedly lacks a successor"
                )
            expected_key = v2._recovery_physical_key(expected_state)
            actual_key = v2._recovery_physical_key(actual_state)
            comparison = v2._physical_comparison(expected_key, actual_key)
            physical = {
                "matched": bool(comparison["matched"]),
                "mismatched_fields": comparison["mismatched_fields"],
                "expected_key_sha256": v1._digest(expected_key),
                "actual_key_sha256": v1._digest(actual_key),
                "ignored_fields": comparison["ignored_fields"],
            }
            if not comparison["matched"]:
                physical["expected_key"] = expected_key
                physical["actual_key"] = actual_key
            checks["physical_successor_matches"] = bool(comparison["matched"])
        except (TypeError, ValueError, RuntimeError) as error:
            physical = {
                "matched": False,
                "error_type": type(error).__name__,
                "mismatched_fields": ["strict_live_recovery_state_unavailable"],
            }
            checks["physical_successor_matches"] = False
        if terminal_alias:
            checks["absorbing_environment_terminal"] = bool(
                branch_env.is_state_terminal(branch_env.current_state)
            )
            checks["absorbing_cohort_empty"] = bool(
                actual_state is not None and not actual_state.blocks
            )

        audit = {
            "live_execution_attempted": True,
            "declared_realization": asdict(declared_realization),
            "effective_live_realization": asdict(effective_realization),
            "terminal_absorbing_alias": terminal_alias,
            "abstract_primitive_cost": int(outcome.primitive_steps),
            "effective_live_primitive_cost": expected_live_steps,
            "remaining_budget_before": int(remaining_budget),
            "remaining_budget_after_abstract_charge": int(
                remaining_budget - outcome.primitive_steps
            ),
            "checks": checks,
            "physical_successor": physical,
            "exact_execution": exact,
        }
        if not all(checks.values()):
            summary = _empty_path_summary(
                status=STATUS_INVALID,
                reason="live_execution_or_model_correspondence_mismatch",
            )
            return {
                "status": STATUS_INVALID,
                "child_id": None,
                "child_summary": summary,
                "execution_metrics": metrics,
                "audit": audit,
                "terminal_alias": terminal_alias,
            }

        child_budget = int(remaining_budget - outcome.primitive_steps)
        if child_budget < 0:
            summary = _empty_path_summary(
                status=STATUS_FAIL,
                reason="carried_recovery_budget_exhausted_by_declared_outcome",
            )
            return {
                "status": STATUS_FAIL,
                "child_id": None,
                "child_summary": summary,
                "execution_metrics": metrics,
                "audit": audit,
                "terminal_alias": terminal_alias,
            }
        child = self._visit(branch_env, child_budget, depth=depth + 1)
        return {
            "status": child.status,
            "child_id": child.node_id,
            "child_summary": child.summary,
            "execution_metrics": metrics,
            "audit": audit,
            "terminal_alias": terminal_alias,
        }

    def _visit(self, env, remaining_budget: int, *, depth: int) -> VisitResult:
        digest, _ = self._node_key(env, remaining_budget)
        existing = self._seen.get(digest)
        if existing is not None:
            self.deduplicated_successor_count += 1
            result = self._results.get(existing)
            if result is None:
                raise ExhaustiveLiveTreeError("live tree contains a cycle")
            return result
        if len(self.nodes) >= self.limits.max_unique_nodes_per_root:
            return self._cutoff(digest)

        node_id = self._new_node_id(digest)
        node = {
            "node_id": node_id,
            "causal_digest": digest,
            "depth": int(depth),
            "remaining_budget": int(remaining_budget),
            "time_steps": int(env.time_steps),
            "agent_position": list(env.current_state),
            "environment_terminal": bool(env.is_state_terminal(env.current_state)),
            "status": None,
            "reason": None,
        }
        self.nodes[node_id] = node
        self._seen[digest] = node_id

        try:
            live_state = v2._live_recovery_state(
                env,
                template=self.root.timing_erased_state,
                search_config=self.search_config,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            node["state_error_type"] = type(error).__name__
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="strict_live_recovery_state_unavailable",
            )
        labels = tuple(sorted(block.label for block in live_state.blocks))
        node["observed_cohort_labels"] = list(labels)
        if not set(labels).issubset(self.root.cohort_labels):
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="committed_cohort_gained_a_stored_label",
            )
        if not labels:
            return self._leaf(
                node,
                status=STATUS_PASS,
                reason="committed_stored_cohort_complete",
            )
        if remaining_budget <= 0:
            return self._leaf(
                node,
                status=STATUS_FAIL,
                reason="positive_workload_with_exhausted_carried_budget",
            )
        if env.is_state_terminal(env.current_state):
            return self._leaf(
                node,
                status=STATUS_FAIL,
                reason="environment_terminal_with_committed_work_remaining",
            )

        frontier_cache = ViabilityCertificateCache()
        snapshot, frontier = benchmark._enumerate_frontier(
            env,
            consecutive_defer=0,
            search_config=self.search_config,
            liveness_rule=self.liveness_rule,
            cache=frontier_cache,
            prioritizer=None,
        )
        node["decision_epoch"] = int(snapshot.decision_epoch)
        node["nominal_frontier_size"] = len(snapshot.candidates)
        node["nominal_frontier_complete"] = bool(
            frontier["complete_frontier_exactly_verified"]
        )
        if not node["nominal_frontier_complete"]:
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="nominal_frontier_not_exactly_complete",
            )
        if not snapshot.candidates:
            return self._leaf(
                node,
                status=STATUS_FAIL,
                reason="no_nominal_exact_safe_candidate",
            )
        if not all(
            item.certificate.status is ViabilityStatus.SAFE
            for item in snapshot.candidates
        ):
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="nominal_frontier_contains_non_safe_candidate",
            )
        snapshot_state, authenticated = v2._authenticated_recovery_actions(snapshot)
        state_check = v2._physical_comparison(
            v2._recovery_physical_key(live_state),
            v2._recovery_physical_key(snapshot_state),
        )
        node["live_snapshot_state_match"] = state_check
        if not state_check["matched"]:
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="live_state_and_replanned_snapshot_disagree",
            )

        admitted, filter_audit = v2._filter_recovery_frontier(
            snapshot,
            method=self.method,
            primitive_budget=int(remaining_budget),
            segment_active=True,
            memo=self._certification_memo,
            memo_namespace=int(self.root.instance_seed),
        )
        node["filter"] = {
            "status": filter_audit["status"],
            "solver_state_status": filter_audit.get("solver_state_status"),
            "original_recovery_keys": filter_audit["original_recovery_keys"],
            "admitted_recovery_keys": filter_audit["admitted_recovery_keys"],
            "rejected_recovery_keys": filter_audit["rejected_recovery_keys"],
            "masked_nonrecovery_keys": filter_audit["masked_nonrecovery_keys"],
            "semantic_digest": filter_audit.get("semantic_digest"),
            "memo_hit": filter_audit["memo_hit"],
        }
        if not admitted:
            status = (
                STATUS_UNKNOWN
                if filter_audit["status"] == SolveStatus.UNKNOWN.value
                or filter_audit.get("solver_state_status")
                == SolveStatus.UNKNOWN.value
                or str(filter_audit["status"]).startswith("UNKNOWN")
                else STATUS_FAIL
            )
            return self._leaf(
                node,
                status=status,
                reason="active_recovery_filter_exposed_no_action",
            )

        selection = v1._select_frozen(
            snapshot,
            admitted_keys=admitted,
            base_agent=self.base_agent,
            cost_network=self.cost_network,
            handling_lambda=self.handling_lambda,
        )
        candidate = selection.candidate
        node["selection"] = dict(selection.audit)
        if candidate.action_type not in (
            ViabilityActionType.DELIVER,
            ViabilityActionType.RECONFIGURE,
        ):
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="active_segment_selected_non_recovery_action",
            )
        action = authenticated.get(candidate.key)
        if action is None:
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="selected_recovery_action_was_not_authenticated",
            )
        outcomes = _structural_envelope(snapshot_state, action)
        expected_ids = [item.disturbance_id for item in outcomes]
        if len(expected_ids) != len(set(expected_ids)):
            raise ExhaustiveLiveTreeError("declared disturbance ids collided")
        node["expected_disturbance_ids"] = expected_ids
        node["selected_recovery_action"] = v2._action_to_dict(action)
        self.abstract_outcome_count += len(outcomes)

        edge_records = []
        edge_indices = []
        for outcome in outcomes:
            branch = self._execute_branch(
                env=env,
                state=snapshot_state,
                candidate=candidate,
                action=action,
                outcome=outcome,
                remaining_budget=remaining_budget,
                depth=depth,
            )
            edge_index = len(self.edges)
            edge = {
                "edge_index": edge_index,
                "parent_node_id": node_id,
                "child_node_id": branch["child_id"],
                "disturbance_id": outcome.disturbance_id,
                "delay_steps": int(outcome.delay_steps),
                "stop_cell": list(outcome.stop_cell),
                "stop_extra_steps": int(outcome.stop_extra_steps),
                "abstract_primitive_steps": int(outcome.primitive_steps),
                "status": branch["status"],
                "terminal_absorbing_alias": bool(branch["terminal_alias"]),
                "execution_metrics": branch["execution_metrics"],
                "execution_audit": branch["audit"],
            }
            self.edges.append(edge)
            edge_indices.append(edge_index)
            edge_records.append(
                {
                    "disturbance_id": outcome.disturbance_id,
                    "status": branch["status"],
                    "child_summary": branch["child_summary"],
                    "execution_metrics": branch["execution_metrics"],
                }
            )

        observed_ids = [self.edges[index]["disturbance_id"] for index in edge_indices]
        node["edge_indices"] = edge_indices
        node["disturbance_coverage_complete"] = observed_ids == expected_ids
        if not node["disturbance_coverage_complete"]:
            return self._leaf(
                node,
                status=STATUS_INVALID,
                reason="declared_disturbance_coverage_incomplete",
            )
        summary = _combine_edge_summaries(edge_records)
        status = _status_from_children([item["status"] for item in edge_records])
        node["status"] = status
        node["reason"] = "all_declared_branches_closed" if status == STATUS_PASS else (
            "at_least_one_declared_branch_did_not_complete"
        )
        node["path_summary"] = summary
        result = VisitResult(node_id, status, summary)
        self._results[node_id] = result
        return result

    def run(self) -> dict:
        root_env = deepcopy(self.root.environment)
        root_result = self._visit(
            root_env,
            self.root.initial_budget,
            depth=0,
        )

        edge_statuses = Counter(item["status"] for item in self.edges)
        node_statuses = Counter(item["status"] for item in self.nodes.values())
        maximum_depth = max((item["depth"] for item in self.nodes.values()), default=0)
        maximum_fanout = max(
            (len(item.get("edge_indices", ())) for item in self.nodes.values()),
            default=0,
        )
        all_expanded_covered = all(
            item.get("disturbance_coverage_complete", True)
            for item in self.nodes.values()
        )
        elapsed = perf_counter() - self.started
        return {
            "instance_seed": self.root.instance_seed,
            "episode_instance_id": self.root.episode_instance_id,
            "schedule_id": self.root.schedule_id,
            "method": self.method,
            "handling_lambda": self.handling_lambda,
            "root_provenance": dict(self.root.provenance),
            "initial_budget": self.root.initial_budget,
            "cohort_labels": list(self.root.cohort_labels),
            "cohort_size": len(self.root.cohort_labels),
            "universal_status": root_result.status,
            "universal_pass": root_result.status == STATUS_PASS,
            "root_node_id": root_result.node_id,
            "coverage": {
                "all_expanded_nodes_cover_every_declared_id": all_expanded_covered,
                "abstract_outcome_count": self.abstract_outcome_count,
                "live_execution_count": self.live_execution_count,
                "live_equivalence_class_count": (
                    self.live_execution_count - self.terminal_alias_count
                ),
                "terminal_absorbing_alias_count": self.terminal_alias_count,
                "node_cutoff_count": self.node_cutoff_count,
                "deduplicated_successor_count": self.deduplicated_successor_count,
            },
            "graph": {
                "unique_node_count": len(self.nodes),
                "live_edge_count": len(self.edges),
                "maximum_depth": maximum_depth,
                "maximum_fanout": maximum_fanout,
                "node_status_counts": dict(sorted(node_statuses.items())),
                "edge_status_counts": dict(sorted(edge_statuses.items())),
                "nodes": list(self.nodes.values()),
                "edges": self.edges,
            },
            "path_summary": root_result.summary,
            "tree_wall_seconds": float(elapsed),
            "training_or_learning": False,
        }


def _semantic_sources(project_root: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "snapshot_bridge_runner": project_root / "run_vcg_robust_recovery_snapshot_panel_5x5.py",
        "parent_v1_runner": project_root / "run_vcg_dynamic_robust_filter_panel_5x5.py",
        "parent_v2_runner": project_root / "run_vcg_dynamic_budgeted_robust_filter_panel_5x5.py",
        "exact_executor": project_root / "vcg_exact_recovery_execution.py",
        "bounded_execution": project_root / "vcg_bounded_macro_execution.py",
        "macro_execution_core": project_root / "train_viability_graph_smdp.py",
        "recovery_solver": project_root / "vcg_robust_recovery_snapshot_5x5.py",
        "recovery_model": project_root / "PSLAP/viability.py",
        "candidate_interface": project_root / "PSLAP/viability_candidates.py",
        "live_viability_filter": project_root / "PSLAP/viability_filter.py",
        "dynamic_yard_projection": project_root / "PSLAP/dynamic_yard.py",
        "live_environment": project_root / "example/small_rooms_env.py",
        "reported_timing_metrics": project_root / "example/helper/timing_metrics.py",
        "frontier_enumerator": project_root / "benchmark_viability_critic_priority.py",
        "frozen_selector": project_root / "viability_graph_hierarchy.py",
    }
    return {name: v1._sha(path.resolve()) for name, path in paths.items()}


def _contract(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
    limits: TreeLimits,
) -> dict:
    runtime_versions = v1.final86._configure_runtime()
    parent_contract, parent_panel, parent_report = _read_authenticated_parent(
        project_root,
        panel,
    )
    parent_dir = PARENT_SNAPSHOT_OUTPUT.resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_role": (
            "exhaustive_live_closed_loop_confirmation_of_one_committed_"
            "recovery_cohort_under_the_declared_finite_W"
        ),
        "panel": panel,
        "instance_seeds": list(_panel_seeds(panel)),
        "methods": list(METHODS),
        "handling_lambdas": list(HANDLING_LAMBDAS),
        "carrier": {
            "method": CARRIER_METHOD,
            "handling_lambda": CARRIER_LAMBDA,
            "root_index": ROOT_INDEX,
            "root_rule": (
                "authenticated_predeclared_snapshot_panel_boundary_then_"
                "exogenous_admission_closure"
            ),
            "pilot_stratum": "observed_occupancy_2",
            "common_root_crossed_across_all_six_arms": True,
        },
        "scope": {
            "committed_stored_cohort_only": True,
            "full_episode_branching": False,
            "accept_robustness": False,
            "defer_robustness": False,
            "future_arrival_robustness": False,
            "timing_robustness": False,
            "interior_path_disturbance_robustness": False,
            "hardware_robustness": False,
        },
        "branching": {
            "disturbance_contract": asdict(ROBUST_CONTRACT),
            "enumerate_every_declared_id_at_every_reachable_selected_recovery": True,
            "deepcopy_full_live_environment_per_branch": True,
            "exact_executor": EXACT_RECOVERY_EXECUTION_CONTRACT,
            "replan_after_every_realized_successor": True,
            "active_accept_and_defer_masked": True,
            "recovery_solver_max_expansions_per_live_node": v2.MAX_EXPANSIONS,
            "solver_unknown_fails_closed": True,
            "dag_memo_key": (
                "full_causal_live_environment_plus_remaining_budget_cohort_"
                "method_lambda_and_consecutive_defer"
            ),
            "terminal_equivalence_contract": TERMINAL_EQUIVALENCE_CONTRACT,
            "terminal_abstract_ids_never_silently_omitted": True,
        },
        "classification": {
            "PASS": (
                "complete_branch_closure_all_declared_ids_accounted_every_"
                "leaf_empties_cohort_no_mismatch_no_cutoff"
            ),
            "FAIL": "concrete_modeled_disturbance_counterexample",
            "UNKNOWN": "solver_or_live_tree_resource_cutoff",
            "INVALID": "clone_contract_or_model_executor_correspondence_failure",
        },
        "limits": asdict(limits),
        "device": device_name,
        "runtime_versions": runtime_versions,
        "output_dir": str(output_dir.resolve()),
        "parent": {
            "output_dir": str(parent_dir),
            "contract_sha256": parent_contract["contract_sha256"],
            "panel_sha256": parent_panel["panel_sha256"],
            "report_sha256": parent_report["report_sha256"],
            "contract_file_sha256": v1._sha(
                parent_dir / snapshot_bridge.CONTRACT_NAME
            ),
            "panel_file_sha256": v1._sha(parent_dir / snapshot_bridge.PANEL_NAME),
            "report_file_sha256": v1._sha(
                parent_dir / snapshot_bridge.REPORT_NAME
            ),
        },
        "source_sha256": _semantic_sources(project_root),
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
    }
    payload = v1._json_safe(payload)
    payload["contract_sha256"] = v1._digest(payload)
    return payload


def prepare_contract(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
    limits: TreeLimits,
) -> dict:
    expected = _contract(
        project_root,
        output_dir,
        panel=panel,
        device_name=device_name,
        limits=limits,
    )
    path = output_dir.resolve() / CONTRACT_NAME
    if path.exists():
        observed = v1._read(path)
        v1._verify_self_hash(observed, "contract_sha256", label="exhaustive tree contract")
        if observed != expected:
            raise ExhaustiveLiveTreeError(
                "exhaustive tree contract, parent artifacts, or sources changed"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ExhaustiveLiveTreeError(
                "nonempty exhaustive tree output has no authenticated contract"
            )
        v1._atomic_json(path, expected)
    return expected


def _ledger_path(
    output_dir: Path,
    *,
    instance_seed: int,
    method: str,
    handling_lambda: float,
) -> Path:
    return (
        output_dir.resolve()
        / "root-ledger"
        / f"instance-{int(instance_seed)}"
        / method
        / f"lambda-{_lambda_key(handling_lambda)}.json"
    )


def _validate_row(
    row: Mapping,
    *,
    contract: Mapping,
    instance_seed: int,
    method: str,
    handling_lambda: float,
    expected_root_digest: str,
) -> dict:
    if not isinstance(contract.get("contract_sha256"), str):
        raise ExhaustiveLiveTreeError("tree row validation contract is invalid")
    if int(row.get("instance_seed", -1)) != int(instance_seed):
        raise ExhaustiveLiveTreeError("tree row seed changed")
    if row.get("method") != method:
        raise ExhaustiveLiveTreeError("tree row method changed")
    if float(row.get("handling_lambda", math.nan)) != float(handling_lambda):
        raise ExhaustiveLiveTreeError("tree row lambda changed")
    if row.get("root_provenance", {}).get("root_environment_digest") != expected_root_digest:
        raise ExhaustiveLiveTreeError("tree row common root changed")
    if row.get("root_provenance", {}).get("source_root_replay_authenticated") is not True:
        raise ExhaustiveLiveTreeError("tree row source root was not authenticated")
    if row.get("universal_status") not in STATUS_ORDER:
        raise ExhaustiveLiveTreeError("tree row has an unknown status")
    coverage = row.get("coverage", {})
    graph = row.get("graph", {})
    if int(graph.get("unique_node_count", -1)) != len(graph.get("nodes", ())):
        raise ExhaustiveLiveTreeError("tree node count changed")
    if int(graph.get("live_edge_count", -1)) != len(graph.get("edges", ())):
        raise ExhaustiveLiveTreeError("tree edge count changed")
    if int(coverage.get("abstract_outcome_count", -1)) != len(graph.get("edges", ())):
        raise ExhaustiveLiveTreeError("abstract outcome/edge coverage changed")
    if row.get("universal_status") == STATUS_PASS:
        invariants = (
            row.get("universal_pass") is True,
            coverage.get("all_expanded_nodes_cover_every_declared_id") is True,
            int(coverage.get("node_cutoff_count", -1)) == 0,
            graph.get("node_status_counts", {}).get(STATUS_FAIL, 0) == 0,
            graph.get("node_status_counts", {}).get(STATUS_UNKNOWN, 0) == 0,
            graph.get("node_status_counts", {}).get(STATUS_INVALID, 0) == 0,
            int(row.get("path_summary", {}).get("completed_path_count", 0))
            == int(row.get("path_summary", {}).get("total_path_count", -1)),
        )
        if not all(invariants):
            raise ExhaustiveLiveTreeError("PASS tree violates universal invariants")
    if row.get("training_or_learning") is not False:
        raise ExhaustiveLiveTreeError("tree row performed training")
    return dict(row)


def _validate_ledger(
    ledger: Mapping,
    *,
    contract: Mapping,
    instance_seed: int,
    method: str,
    handling_lambda: float,
    expected_root_digest: str,
) -> dict:
    v1._verify_self_hash(ledger, "ledger_sha256", label="exhaustive tree ledger")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "instance_seed": int(instance_seed),
        "method": method,
        "handling_lambda": float(handling_lambda),
    }
    for field, value in expected.items():
        if ledger.get(field) != value:
            raise ExhaustiveLiveTreeError(f"tree ledger {field} changed")
    return _validate_row(
        ledger["row"],
        contract=contract,
        instance_seed=instance_seed,
        method=method,
        handling_lambda=handling_lambda,
        expected_root_digest=expected_root_digest,
    )


def _compact_report_row(
    row: Mapping,
    *,
    ledger: Mapping,
    ledger_path: Path,
    output_dir: Path,
) -> dict:
    """Reference the authenticated DAG ledger without duplicating it."""

    graph = row["graph"]
    provenance = row["root_provenance"]
    return {
        "instance_seed": int(row["instance_seed"]),
        "episode_instance_id": row["episode_instance_id"],
        "schedule_id": row["schedule_id"],
        "method": row["method"],
        "handling_lambda": float(row["handling_lambda"]),
        "root_provenance": {
            "source_root_replay_authenticated": provenance[
                "source_root_replay_authenticated"
            ],
            "source_decision_index": provenance["source_decision_index"],
            "source_decision_epoch": provenance["source_decision_epoch"],
            "source_behavior_digest": provenance["source_behavior_digest"],
            "source_ledger_sha256": provenance["source_ledger_sha256"],
            "observed_occupancy": provenance["observed_occupancy"],
            "root_environment_digest": provenance["root_environment_digest"],
        },
        "initial_budget": int(row["initial_budget"]),
        "cohort_labels": list(row["cohort_labels"]),
        "cohort_size": int(row["cohort_size"]),
        "universal_status": row["universal_status"],
        "universal_pass": row["universal_pass"],
        "coverage": dict(row["coverage"]),
        "graph": {
            "unique_node_count": int(graph["unique_node_count"]),
            "live_edge_count": int(graph["live_edge_count"]),
            "maximum_depth": int(graph["maximum_depth"]),
            "maximum_fanout": int(graph["maximum_fanout"]),
            "node_status_counts": dict(graph["node_status_counts"]),
            "edge_status_counts": dict(graph["edge_status_counts"]),
        },
        "path_summary": dict(row["path_summary"]),
        "tree_wall_seconds": float(row["tree_wall_seconds"]),
        "ledger_path": ledger_path.resolve().relative_to(
            output_dir.resolve()
        ).as_posix(),
        "ledger_sha256": ledger["ledger_sha256"],
        "training_or_learning": False,
    }


def evaluate(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
    limits: TreeLimits,
) -> tuple[list[dict], list[dict]]:
    contract = prepare_contract(
        project_root,
        output_dir,
        panel=panel,
        device_name=device_name,
        limits=limits,
    )
    _, parent_panel, _ = _read_authenticated_parent(project_root, panel)
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
    wanted = set(_panel_seeds(panel))
    records = tuple(
        record
        for record in auth["manifest"]["instances"]
        if int(record["instance_seed"]) in wanted
    )
    if {int(item["instance_seed"]) for item in records} != wanted:
        raise ExhaustiveLiveTreeError("authenticated source manifest lacks panel seeds")
    search_config = benchmark._search_config(auth["arm"].payload)
    liveness_rule = benchmark._liveness_rule(auth["arm"].payload)
    rows: list[dict] = []
    missing_roots: list[dict] = []
    total_possible = len(records) * len(METHODS) * len(HANDLING_LAMBDAS)
    completed = 0

    for record in records:
        seed = int(record["instance_seed"])
        instance = v1.source89._load_instance(auth["source_output"], record)
        source_snapshot = _snapshot_row(parent_panel, seed)
        root = replay_common_root(
            arm=auth["arm"],
            instance=instance,
            snapshot_row=source_snapshot,
            device=device,
        )
        root_digest = str(root.provenance["root_environment_digest"])
        for method in METHODS:
            for handling_lambda in HANDLING_LAMBDAS:
                path = _ledger_path(
                    output_dir,
                    instance_seed=seed,
                    method=method,
                    handling_lambda=handling_lambda,
                )
                if path.exists():
                    ledger = v1._read(path)
                    row = _validate_ledger(
                        ledger,
                        contract=contract,
                        instance_seed=seed,
                        method=method,
                        handling_lambda=handling_lambda,
                        expected_root_digest=root_digest,
                    )
                else:
                    explorer = SegmentTreeExplorer(
                        root=root,
                        method=method,
                        handling_lambda=handling_lambda,
                        base_agent=base_agent,
                        cost_network=cost_network,
                        search_config=search_config,
                        liveness_rule=liveness_rule,
                        limits=limits,
                    )
                    row = explorer.run()
                    ledger = {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "instance_seed": seed,
                        "episode_instance_id": record["episode_instance_id"],
                        "schedule_id": record["schedule_id"],
                        "method": method,
                        "handling_lambda": float(handling_lambda),
                        "checkpoint_sha256": v1.source89.SELECTED_CHECKPOINT_SHA256[
                            v2.MODEL_SEED
                        ],
                        "cost_head_sha256": v1.source89.COST_HEAD_SHA256[v2.MODEL_SEED],
                        "row": row,
                    }
                    ledger["ledger_sha256"] = v1._digest(ledger)
                    _validate_ledger(
                        ledger,
                        contract=contract,
                        instance_seed=seed,
                        method=method,
                        handling_lambda=handling_lambda,
                        expected_root_digest=root_digest,
                    )
                    v1._atomic_json(path, ledger)
                rows.append(
                    _compact_report_row(
                        row,
                        ledger=ledger,
                        ledger_path=path,
                        output_dir=output_dir,
                    )
                )
                completed += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{completed}/{total_possible}",
                            "instance_seed": seed,
                            "method": method,
                            "handling_lambda": handling_lambda,
                            "universal_status": row["universal_status"],
                            "unique_nodes": row["graph"]["unique_node_count"],
                            "live_edges": row["graph"]["live_edge_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return rows, missing_roots


def _method_summary(rows: Sequence[Mapping]) -> dict:
    result = {}
    for method in METHODS:
        for handling_lambda in HANDLING_LAMBDAS:
            group = tuple(
                row
                for row in rows
                if row["method"] == method
                and float(row["handling_lambda"]) == float(handling_lambda)
            )
            statuses = Counter(row["universal_status"] for row in group)
            key = f"{method}|lambda={handling_lambda:g}"
            result[key] = {
                "common_root_count": len(group),
                "status_counts": dict(sorted(statuses.items())),
                "universal_pass_rate": (
                    None
                    if not group
                    else sum(row["universal_status"] == STATUS_PASS for row in group)
                    / len(group)
                ),
                "all_common_roots_pass": bool(
                    group and all(row["universal_status"] == STATUS_PASS for row in group)
                ),
                "unique_nodes": sum(row["graph"]["unique_node_count"] for row in group),
                "live_edges": sum(row["graph"]["live_edge_count"] for row in group),
                "abstract_outcomes": sum(
                    row["coverage"]["abstract_outcome_count"] for row in group
                ),
                "terminal_absorbing_aliases": sum(
                    row["coverage"]["terminal_absorbing_alias_count"] for row in group
                ),
                "node_cutoffs": sum(
                    row["coverage"]["node_cutoff_count"] for row in group
                ),
                "tree_wall_seconds": sum(row["tree_wall_seconds"] for row in group),
            }
    return result


def summarize(
    project_root: Path,
    output_dir: Path,
    *,
    panel: str,
    device_name: str,
    limits: TreeLimits,
) -> dict:
    contract = prepare_contract(
        project_root,
        output_dir,
        panel=panel,
        device_name=device_name,
        limits=limits,
    )
    _, parent_panel, _ = _read_authenticated_parent(project_root, panel)
    rows = []
    missing_roots = []
    for seed in _panel_seeds(panel):
        source_snapshot = _snapshot_row(parent_panel, seed)
        if int(source_snapshot["observed_occupancy"]) < 1:
            raise ExhaustiveLiveTreeError("snapshot root has an empty cohort")
        root_digest = None
        for method in METHODS:
            for handling_lambda in HANDLING_LAMBDAS:
                path = _ledger_path(
                    output_dir,
                    instance_seed=seed,
                    method=method,
                    handling_lambda=handling_lambda,
                )
                if not path.exists():
                    raise ExhaustiveLiveTreeError(
                        f"missing exhaustive tree ledger: {path}"
                    )
                ledger = v1._read(path)
                observed_digest = ledger["row"]["root_provenance"][
                    "root_environment_digest"
                ]
                if root_digest is None:
                    root_digest = observed_digest
                elif root_digest != observed_digest:
                    raise ExhaustiveLiveTreeError(
                        "method/lambda arms do not share an identical root"
                    )
                row = _validate_ledger(
                    ledger,
                    contract=contract,
                    instance_seed=seed,
                    method=method,
                    handling_lambda=handling_lambda,
                    expected_root_digest=root_digest,
                )
                rows.append(
                    _compact_report_row(
                        row,
                        ledger=ledger,
                        ledger_path=path,
                        output_dir=output_dir,
                    )
                )
    eligible = len(_panel_seeds(panel)) - len(missing_roots)
    expected_rows = eligible * len(METHODS) * len(HANDLING_LAMBDAS)
    if len(rows) != expected_rows:
        raise ExhaustiveLiveTreeError("exhaustive row grid is incomplete")
    method_summary = _method_summary(rows)
    recursive_rows = tuple(row for row in rows if row["method"] == "recursive")
    claim_ready = bool(
        recursive_rows
        and all(row["universal_status"] == STATUS_PASS for row in recursive_rows)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "status_meaning": "protocol_completed_not_every_controller_tree_passed",
        "contract_sha256": contract["contract_sha256"],
        "panel": panel,
        "requested_instance_count": len(_panel_seeds(panel)),
        "common_root_instance_count": eligible,
        "missing_common_roots": missing_roots,
        "row_count": len(rows),
        "method_summary": method_summary,
        "recursive_exhaustive_live_claim_ready": claim_ready,
        "claim": (
            "exhaustive_modulo_explicit_global_terminal_absorbing_equivalence_"
            "for_the_first_common_committed_recovery_cohort_only"
        ),
        "limitations": {
            "full_episode_robustness": False,
            "accept_or_defer_robustness": False,
            "arrival_robustness": False,
            "timing_robustness": False,
            "interior_path_disturbance_robustness": False,
            "hardware_robustness": False,
            "all_possible_real_world_disturbances": False,
            "one_common_root_per_eligible_instance": True,
            "resource_cutoff_is_unknown": True,
            "terminal_absorbing_equivalence_explicit": True,
        },
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
        "rows": rows,
    }
    report["report_sha256"] = v1._digest(report)
    v1._atomic_json(output_dir.resolve() / REPORT_NAME, report)
    return report


def _console_report(report: Mapping) -> dict:
    """Keep CLI output bounded; authenticated per-arm ledgers retain DAGs."""

    return {
        "protocol": report["protocol"],
        "status": report["status"],
        "panel": report["panel"],
        "requested_instance_count": report["requested_instance_count"],
        "common_root_instance_count": report["common_root_instance_count"],
        "row_count": report["row_count"],
        "recursive_exhaustive_live_claim_ready": report[
            "recursive_exhaustive_live_claim_ready"
        ],
        "method_summary": report["method_summary"],
        "report_sha256": report["report_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--panel", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--project-root", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-unique-nodes", type=int)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        _default_output(args.panel)
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    limits = TreeLimits(
        max_unique_nodes_per_root=(
            _default_node_limit(args.panel)
            if args.max_unique_nodes is None
            else int(args.max_unique_nodes)
        )
    )
    if args.command == "prepare":
        result = {
            "contract": prepare_contract(
                project_root,
                output_dir,
                panel=args.panel,
                device_name=args.device,
                limits=limits,
            )
        }
    elif args.command == "run":
        rows, missing = evaluate(
            project_root,
            output_dir,
            panel=args.panel,
            device_name=args.device,
            limits=limits,
        )
        report = summarize(
            project_root,
            output_dir,
            panel=args.panel,
            device_name=args.device,
            limits=limits,
        )
        result = {
            "evaluation": {
                "row_count": len(rows),
                "missing_common_roots": missing,
            },
            "report": _console_report(report),
        }
    else:
        report = summarize(
            project_root,
            output_dir,
            panel=args.panel,
            device_name=args.device,
            limits=limits,
        )
        result = {
            "report": _console_report(report)
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
