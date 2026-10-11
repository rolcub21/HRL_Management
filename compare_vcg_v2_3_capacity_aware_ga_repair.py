"""Development-only matched comparison for capacity-aware rolling-GA repairs.

The completed V2.3 matched-panel comparison is immutable input.  This runner
authenticates and reuses all 156 historical rows, executes only four new,
versioned capacity-aware GA sources on the same twelve already-opened 85xxx
EpisodeInstances, and then rebuilds the metric-specific rankings, Pareto
fronts, and the predeclared development method-capacity screen.

Nothing in this protocol opens the reserved 86xxx final panel.  The historical
GA adapter IDs, ledgers, and results remain untouched; repaired sources have
new IDs and preserve their full queue/capacity/reservation audit in each new
atomic run ledger.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable, Mapping, Optional, Sequence
import uuid

import compare_vcg_v2_3_matched_baselines as base
from compare_viability_graph_baselines import _dual_rescore_from_legacy_return
from PSLAP.ga_capacity_aware import (
    CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
    CAPACITY_AWARE_QUEUE_CONTRACT,
    CapacityAwareCompleteRollingGAAssigner,
    CapacityAwareDurationAwareRollingGAAssigner,
    CapacityAwareOperationalRollingGAAssigner,
    CapacityAwareRollingGAStorageAssigner,
)
from PSLAP.track_a import (
    TRACK_A_GA_ROLLING_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
)
from track_b_urgency_evaluate import evaluate_assignment_ablation_one, resolve_device


PROTOCOL = "vcg_v2_3_capacity_aware_partial_ga_repair_85xxx_development_v1"
SCHEMA_VERSION = 1
PANEL_SEEDS = base.PANEL_SEEDS
EXPECTED_DELIVERIES = base.EXPECTED_DELIVERIES
FINAL_PANEL_SEEDS = base.FINAL_PANEL_SEEDS

MATCHED_CONTRACT_RAW_SHA256 = (
    "60704d7fa37807f4d98b34683043a3a2efdc9fb8b8d6e4e4e8c3ff3cbdb12af0"
)
MATCHED_PREFLIGHT_RAW_SHA256 = (
    "d2e77ea915d84e4ee167fe339479ced94b93797ac3888f5ca948776ee795018e"
)
MATCHED_REPORT_RAW_SHA256 = (
    "522746ecbcdfd914b7d4998367ae55511edd820d06571e89c3fbd9122eff8caf"
)
MATCHED_RUNS_RAW_SHA256 = (
    "7812868db6da637c867cf0f9d99d1aa45a0443570ea34d805cc1fa22d07eea0f"
)
MATCHED_AUDIT_RAW_SHA256 = (
    "e32c0705bb250cf8283ac2fb9e1460a688105f617e8f5e5c7715a0f4abfad62c"
)
MATCHED_CONTRACT_CANONICAL_SHA256 = (
    "d1de9e1a38fc89d3f0c63f5661b488d85e62c65369898fc38c3ec64f56cc743c"
)
MATCHED_REPORT_CANONICAL_SHA256 = (
    "b89929fcbf7f26f62ead0117e9b780c160a3858b2cc7a15fbccb722246adeeef"
)
MATCHED_AUDIT_CANONICAL_SHA256 = (
    "ef8a31a752f61c4a6806db584841bd00f47ca502a3345fffe18e439afddd12bb"
)

REPAIRED_2009_METHOD = (
    "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
)
REPAIRED_DURATION_METHOD = (
    "duration_aware_pslap_ga_duration_aware_rolling_capacity_aware_partial"
)
REPAIRED_OPERATIONAL_METHOD = (
    "duration_aware_pslap_ga_operational_rolling_capacity_aware_partial"
)
REPAIRED_COMPLETE_METHOD = (
    "duration_aware_enhanced_complete_rolling_ga_capacity_aware_partial"
)

REPAIRED_METHOD_TO_SOURCE = {
    REPAIRED_2009_METHOD: TRACK_A_GA_ROLLING_CAPACITY_AWARE,
    REPAIRED_DURATION_METHOD: TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
    REPAIRED_OPERATIONAL_METHOD: TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
    REPAIRED_COMPLETE_METHOD: TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
}
REPAIRED_METHOD_TO_VERSION = {
    REPAIRED_2009_METHOD: CapacityAwareRollingGAStorageAssigner.VERSION,
    REPAIRED_DURATION_METHOD: CapacityAwareDurationAwareRollingGAAssigner.VERSION,
    REPAIRED_OPERATIONAL_METHOD: CapacityAwareOperationalRollingGAAssigner.VERSION,
    REPAIRED_COMPLETE_METHOD: CapacityAwareCompleteRollingGAAssigner.VERSION,
}
REPAIRED_METHOD_TO_HISTORICAL = {
    REPAIRED_2009_METHOD: base.ROLLING_2009_METHOD,
    REPAIRED_DURATION_METHOD: base.DURATION_AWARE_GA_METHOD,
    REPAIRED_OPERATIONAL_METHOD: base.OPERATIONAL_GA_METHOD,
    REPAIRED_COMPLETE_METHOD: base.ENHANCED_GA_METHOD,
}
REPAIRED_METHOD_TO_OBJECTIVE = {
    REPAIRED_2009_METHOD: "pslap_2009_rolling",
    REPAIRED_DURATION_METHOD: "duration_aware_rolling",
    REPAIRED_OPERATIONAL_METHOD: "operational_rolling",
    REPAIRED_COMPLETE_METHOD: "duration_aware_singleton_complete_rolling",
}
REPAIRED_METHODS = tuple(REPAIRED_METHOD_TO_SOURCE)
EXPANDED_PRIMARY_METHODS = (*base.PRIMARY_METHODS, *REPAIRED_METHODS)

LEGACY_LEX_FIELDS = (
    "infeasible_events",
    "obstructive_moves",
    "route_steps",
)
DURATION_LEX_FIELDS = (
    "infeasible_events",
    "priority_obstruction_cost",
    "weighted_route_steps",
    "raw_obstructive_moves",
    "raw_route_steps",
)

CONTRACT_FILENAME = "repair-contract.json"
PREFLIGHT_FILENAME = "preflight.json"
RUNS_FILENAME = "expanded-runs.csv"
REPORT_FILENAME = "expanded-report.json"
AUDIT_FILENAME = "expanded-audit.json"


class CapacityRepairComparisonError(base.MatchedComparisonError):
    """Raised when the frozen repair protocol or an audit fails closed."""


@dataclass(frozen=True)
class AuthenticatedMatchedSource:
    contract: dict
    preflight: dict
    report: dict
    audit: dict
    rows: tuple[dict, ...]
    paths: Mapping[str, Path]
    raw_sha256: Mapping[str, str]
    historical_ledger_paths: Mapping[str, Path]
    historical_ledger_raw_sha256: Mapping[str, str]


def _require_equal(name: str, observed, expected) -> None:
    try:
        base._require_equal(name, observed, expected)
    except base.MatchedComparisonError as error:
        raise CapacityRepairComparisonError(str(error)) from error


def _atomic_json(path: Path, value) -> None:
    base._atomic_json(path, value)


def _sha256_file(path: Path) -> str:
    return base._sha256_file(path)


def _digest_json(value) -> str:
    return base._digest_json(value)


def _verify_self_hash(payload: Mapping, field: str, *, name: str) -> str:
    try:
        return base._verify_self_hash(payload, field, name=name)
    except base.MatchedComparisonError as error:
        raise CapacityRepairComparisonError(str(error)) from error


def _load_json(path: Path) -> dict:
    try:
        return base._load_json(path)
    except (base.MatchedComparisonError, FileNotFoundError) as error:
        raise CapacityRepairComparisonError(str(error)) from error


def _row_key(row: Mapping) -> tuple:
    method = row.get("method_id")
    if method == base.V23_METHOD:
        return method, int(row["instance_seed"]), int(row["policy_rng_index"])
    if method == base.V11_METHOD:
        return method, int(row["model_seed"]), int(row["instance_seed"])
    return method, int(row["instance_seed"])


def _verify_old_baseline_ledgers(
    matched_dir: Path,
    *,
    report_rows: Sequence[Mapping],
    audit: Mapping,
) -> tuple[dict[str, Path], dict[str, str]]:
    expected_grid = {
        (method, seed)
        for method in base.DETERMINISTIC_METHODS
        for seed in PANEL_SEEDS
    }
    manifest = audit.get("baseline_ledger_manifest")
    if not isinstance(manifest, list) or len(manifest) != len(expected_grid):
        raise CapacityRepairComparisonError(
            "matched source baseline ledger manifest is not the exact 6x12 grid"
        )
    report_by_key = {_row_key(row): row for row in report_rows}
    observed = set()
    authenticated_paths = {}
    authenticated_hashes = {}
    ledger_root = (matched_dir / "run-ledger").resolve()
    for record in manifest:
        run_key = str(record.get("run_key"))
        try:
            method, seed_text = run_key.rsplit(":", 1)
            seed = int(seed_text)
        except (TypeError, ValueError) as error:
            raise CapacityRepairComparisonError(
                f"invalid historical run key {run_key!r}"
            ) from error
        if (method, seed) not in expected_grid:
            raise CapacityRepairComparisonError(
                f"historical ledger outside exact deterministic grid: {run_key}"
            )
        observed.add((method, seed))
        path = Path(str(record.get("path"))).resolve()
        if not path.is_relative_to(ledger_root):
            # The completed artifact records absolute paths.  Permit relocation
            # of the repository only through the exact deterministic grid path.
            path = (ledger_root / method / f"seed-{seed}.json").resolve()
        if not path.is_relative_to(ledger_root) or not path.is_file():
            raise CapacityRepairComparisonError(
                f"historical baseline ledger path is invalid: {path}"
            )
        _require_equal(
            f"historical ledger raw SHA {run_key}",
            _sha256_file(path),
            record.get("raw_sha256"),
        )
        ledger = _load_json(path)
        _require_equal(
            f"historical ledger self SHA {run_key}",
            _verify_self_hash(ledger, "ledger_sha256", name=run_key),
            record.get("ledger_sha256"),
        )
        _require_equal("historical ledger run key", ledger.get("run_key"), run_key)
        _require_equal(
            f"historical report/ledger row {run_key}",
            report_by_key.get((method, seed)),
            ledger.get("run"),
        )
        authenticated_paths[run_key] = path
        authenticated_hashes[run_key] = str(record.get("raw_sha256"))
    _require_equal("historical deterministic ledger grid", observed, expected_grid)
    return authenticated_paths, authenticated_hashes


def authenticate_completed_matched_comparison(
    matched_dir: Path,
    *,
    sources: base.AuthenticatedSources,
) -> AuthenticatedMatchedSource:
    """Authenticate the immutable 156-row comparison and its source chains."""

    matched_dir = matched_dir.resolve()
    pins = {
        "comparison-contract.json": MATCHED_CONTRACT_RAW_SHA256,
        "preflight.json": MATCHED_PREFLIGHT_RAW_SHA256,
        "matched-report.json": MATCHED_REPORT_RAW_SHA256,
        "matched-runs.csv": MATCHED_RUNS_RAW_SHA256,
        "matched-audit.json": MATCHED_AUDIT_RAW_SHA256,
    }
    paths = {name: (matched_dir / name).resolve() for name in pins}
    for name, path in paths.items():
        if not path.is_relative_to(matched_dir) or not path.is_file():
            raise CapacityRepairComparisonError(
                f"missing immutable matched source artifact: {path}"
            )
        _require_equal(f"matched source raw SHA {name}", _sha256_file(path), pins[name])

    contract = _load_json(paths["comparison-contract.json"])
    preflight = _load_json(paths["preflight.json"])
    report = _load_json(paths["matched-report.json"])
    audit = _load_json(paths["matched-audit.json"])
    _require_equal(
        "matched contract canonical SHA",
        _verify_self_hash(contract, "contract_sha256", name="matched contract"),
        MATCHED_CONTRACT_CANONICAL_SHA256,
    )
    _require_equal(
        "matched report canonical SHA",
        _verify_self_hash(report, "report_sha256", name="matched report"),
        MATCHED_REPORT_CANONICAL_SHA256,
    )
    _require_equal(
        "matched audit canonical SHA",
        _verify_self_hash(audit, "audit_sha256", name="matched audit"),
        MATCHED_AUDIT_CANONICAL_SHA256,
    )
    for payload_name, payload in (
        ("contract", contract),
        ("preflight", preflight),
        ("report", report),
        ("audit", audit),
    ):
        _require_equal(f"matched {payload_name} protocol", payload.get("protocol"), base.PROTOCOL)
        _require_equal(
            f"matched {payload_name} final panel flag",
            payload.get("final_86xxx_panel_opened"),
            False,
        )
        _require_equal(
            f"matched {payload_name} performance claim",
            payload.get("performance_claim_authorized"),
            False,
        )
    _require_equal("matched contract panel", contract.get("panel_seeds"), PANEL_SEEDS)
    _require_equal("matched contract primary methods", contract.get("primary_online_methods"), base.PRIMARY_METHODS)
    _require_equal("matched preflight status", preflight.get("status"), "complete")
    _require_equal("matched preflight missing rows", preflight.get("missing_deterministic_rows"), ())
    _require_equal("matched report run count", report.get("run_count"), 156)
    _require_equal("matched audit report raw SHA", audit.get("matched_report_raw_sha256"), MATCHED_REPORT_RAW_SHA256)
    _require_equal("matched audit CSV raw SHA", audit.get("matched_runs_raw_sha256"), MATCHED_RUNS_RAW_SHA256)

    rows = tuple(report.get("runs", ()))
    if len(rows) != 156 or len({_row_key(row) for row in rows}) != 156:
        raise CapacityRepairComparisonError(
            "matched report does not contain 156 unique source rows"
        )
    if any(int(row.get("instance_seed", -1)) in FINAL_PANEL_SEEDS for row in rows):
        raise CapacityRepairComparisonError("matched report contains a protected 86xxx row")
    if {int(row.get("instance_seed", -1)) for row in rows} != set(PANEL_SEEDS):
        raise CapacityRepairComparisonError("matched report panel is not exact 85000..85011")

    # Independently regenerate the learned-source normalized rows from their
    # authenticated ledgers, then bind them to the pinned matched report.
    regenerated = (
        *base._normalize_v23_rows(sources.v23_rows, sources.identities),
        *base._normalize_v11_rows(sources.v11_rows, sources.identities),
    )
    by_key = {_row_key(row): row for row in rows}
    for row in regenerated:
        _require_equal(
            f"matched learned row {_row_key(row)!r}",
            by_key.get(_row_key(row)),
            row,
        )
    historical_ledger_paths, historical_ledger_hashes = (
        _verify_old_baseline_ledgers(
            matched_dir, report_rows=rows, audit=audit
        )
    )

    # Recompute every historical whole-method summary and ranking under the
    # original protocol.  This detects a pinned report with internally
    # inconsistent aggregates without executing any policy.
    recomputed = tuple(
        base._summarize_method(method, rows, sources.identities)
        for method in base.PRIMARY_METHODS
    )
    _require_equal("historical method summaries", report.get("method_summaries"), recomputed)
    _require_equal("historical metric rankings", report.get("metric_rankings"), base._rank_methods(recomputed))
    _require_equal("historical Pareto fronts", report.get("pareto_fronts"), base._pareto_fronts(recomputed))

    return AuthenticatedMatchedSource(
        contract=contract,
        preflight=preflight,
        report=report,
        audit=audit,
        rows=rows,
        paths=paths,
        raw_sha256={name: pins[name] for name in sorted(pins)},
        historical_ledger_paths=historical_ledger_paths,
        historical_ledger_raw_sha256=historical_ledger_hashes,
    )


def _current_source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    paths = dict(base._baseline_source_paths())
    paths.update(
        {
            "PSLAP/ga_capacity_aware.py": root / "PSLAP/ga_capacity_aware.py",
            "compare_vcg_v2_3_capacity_aware_ga_repair.py": Path(__file__).resolve(),
        }
    )
    result = {}
    for name, path in sorted(paths.items()):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def build_repair_contract(
    *,
    sources: base.AuthenticatedSources,
    matched: AuthenticatedMatchedSource,
    device: str,
) -> dict:
    source_paths = _current_source_paths()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "final_622xxx_action_rng_panel_opened": False,
        "panel_seeds": PANEL_SEEDS,
        "episode_instance_count": len(PANEL_SEEDS),
        "expected_deliveries_per_episode": EXPECTED_DELIVERIES,
        "immutable_matched_source": {
            "protocol": base.PROTOCOL,
            "source_row_count": 156,
            "rows_reused_without_policy_execution": 156,
            "raw_sha256": matched.raw_sha256,
            "contract_canonical_sha256": MATCHED_CONTRACT_CANONICAL_SHA256,
            "report_canonical_sha256": MATCHED_REPORT_CANONICAL_SHA256,
            "audit_canonical_sha256": MATCHED_AUDIT_CANONICAL_SHA256,
            "historical_source_time_hashes_preserved": True,
            "current_optional_selector_hooks_do_not_reinterpret_historical_rows": True,
        },
        "new_repaired_methods": REPAIRED_METHODS,
        "new_assignment_sources": REPAIRED_METHOD_TO_SOURCE,
        "new_assignment_source_versions": REPAIRED_METHOD_TO_VERSION,
        "historical_lineage": REPAIRED_METHOD_TO_HISTORICAL,
        "expected_new_run_grid": {
            "methods": len(REPAIRED_METHODS),
            "instances_per_method": len(PANEL_SEEDS),
            "total_rows": len(REPAIRED_METHODS) * len(PANEL_SEEDS),
            "only_new_repaired_methods_executed": True,
        },
        "queue_capacity_contract": {
            "identifier": CAPACITY_AWARE_QUEUE_CONTRACT,
            "audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
            "mandatory_current_gene_index": 0,
            "frozen_pickup_cell": FROZEN_PICKUP_CELL,
            "planned_cohort": "earliest_arrived_FIFO_prefix_limited_by_current_shared_mask",
            "overflow": "all_excess_arrived_pending_labels_explicit_and_nonbinding",
            "overflow_is_scheduler_defer": False,
            "overflow_virtual_penalty": 0.0,
            "only_gene_zero_reserved_and_committed": True,
            "future_schedule_access": False,
            "information_regime": "online_arrived_only",
        },
        "objective_spec": base.FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_operational_return",
        "environment": sources.v23_contract["environment"],
        "execution_device": str(device),
        "fixed_runtime": {
            "max_steps": base.MAX_STEPS,
            "max_defer_steps": base.MAX_DEFER_STEPS,
            "lookahead_margin_steps": base.LOOKAHEAD_MARGIN_STEPS,
            "rolling_population": base.ROLLING_POPULATION,
            "rolling_generations": base.ROLLING_GENERATIONS,
            "rolling_ga_egress_weight": base.ROLLING_GA_EGRESS_WEIGHT,
            "ga_seed_base": base.GA_SEED_BASE,
        },
        "safety_protocol": {
            "complete_4_by_12_grid_required": True,
            "whole_method_exclusion_on_any_unsafe_row": True,
            "complete_case_filtering_permitted": False,
            "strict_success_full_completion_zero_failures_required": True,
            "source_queue_capacity_and_reservation_audit_required": True,
        },
        "development_method_capacity_screen_frozen_before_repaired_outcomes": {
            "source_v2_3_episode160_must_remain_authenticated_and_eligible": True,
            "all_48_v2_3_rows_strict_safe_complete": True,
            "pass_condition": (
                "no_whole_method_safety_eligible_matched_online_comparator_"
                "weakly_dominates_v2_3_on_MAE_and_total_physical_rehandles_"
                "per_100_with_at_least_one_strict"
            ),
            "not_a_seed_stability_or_superiority_claim": True,
            "independent_v2_3_training_seed_replication_still_required": True,
        },
        "aggregation": matched.contract["aggregation"],
        "current_repair_source_sha256": _source_hashes(source_paths),
    }
    contract["contract_sha256"] = _digest_json(contract)
    return contract


def _repair_input_contract(
    method: str,
    seed: int,
    *,
    sources: base.AuthenticatedSources,
    repair_contract: Mapping,
) -> dict:
    if method not in REPAIRED_METHODS:
        raise CapacityRepairComparisonError(f"unknown repaired method: {method}")
    if seed not in PANEL_SEEDS or seed in FINAL_PANEL_SEEDS:
        raise CapacityRepairComparisonError(
            f"repaired method seed is outside exact opened panel: {seed}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "repair_contract_sha256": repair_contract["contract_sha256"],
        "method_id": method,
        "historical_lineage_method_id": REPAIRED_METHOD_TO_HISTORICAL[method],
        "assignment_source": REPAIRED_METHOD_TO_SOURCE[method],
        "assignment_source_version": REPAIRED_METHOD_TO_VERSION[method],
        "capacity_aware_queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
        "source_audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
        "frozen_pickup_cell": FROZEN_PICKUP_CELL,
        **base._identity_fields(seed, sources.identities),
        "information_regime": "online_arrived_only",
        "objective_spec": base.FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_operational_return",
        "max_steps": base.MAX_STEPS,
        "max_defer_steps": base.MAX_DEFER_STEPS,
        "lookahead_margin_steps": base.LOOKAHEAD_MARGIN_STEPS,
        "rolling_population": base.ROLLING_POPULATION,
        "rolling_generations": base.ROLLING_GENERATIONS,
        "rolling_ga_egress_weight": base.ROLLING_GA_EGRESS_WEIGHT,
        "ga_seed": base.GA_SEED_BASE + seed,
        "future_schedule_accessed": False,
        "final_86xxx_panel_opened": False,
    }


def _candidate_mask_sha(cells: Sequence[Sequence[int]]) -> str:
    normalized = tuple(tuple(int(value) for value in cell) for cell in cells)
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _queue_partition_sha(
    observed: Sequence[str], planned: Sequence[str], overflow: Sequence[str]
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "observed": tuple(observed),
                "planned": tuple(planned),
                "deferred": tuple(overflow),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _cost_scalar(components: Mapping, fields: Sequence[str]) -> int:
    if tuple(fields) == LEGACY_LEX_FIELDS:
        return (
            int(components["infeasible_events"]) * 10**9
            + int(components["obstructive_moves"]) * 10**6
            + int(components["route_steps"])
        )
    if tuple(fields) == DURATION_LEX_FIELDS:
        return (
            int(components["infeasible_events"]) * 10**15
            + int(components["priority_obstruction_cost"]) * 10**10
            + int(components["weighted_route_steps"]) * 10**4
            + int(components["raw_obstructive_moves"]) * 10**2
            + int(components["raw_route_steps"])
        )
    raise CapacityRepairComparisonError(
        f"unrecognized objective lexicographic fields: {fields!r}"
    )


_CAPACITY_SOURCE_AUDIT_KEYS = frozenset(
    {
        "audit_schema_version", "queue_contract", "assignment_source_version",
        "objective_variant", "information_regime", "future_schedule_accessed",
        "future_schedule_fields_read", "observable_fields_used", "pending_order",
        "preview_contract", "optimizer_base_config",
        "optimizer_seed_offset_contract", "egress_weight",
        "objective_lexicographic_fields", "deferred_semantics",
        "scheduler_defer_action_introduced", "queue_overflow_virtual_penalty",
        "decision_count", "half_minted_unbound_preview_present",
        "committed_plan_count", "minted_unreserved_preview_count",
        "minted_unreserved_proposal_ids",
        "reserved_uncommitted_preview_count",
        "reserved_uncommitted_proposal_ids",
        "decisions_with_deferred_remainder",
        "observed_pending_membership_count_total",
        "planned_membership_count_total", "deferred_membership_count_total",
        "max_observed_arrived_pending_count", "max_planned_count",
        "max_deferred_count", "all_current_blocks_planned",
        "all_current_blocks_are_gene_zero",
        "all_current_blocks_are_unique_observed_pickup_heads",
        "all_current_blocks_are_fifo_heads", "only_gene_zero_ever_committed",
        "committed_gene_indices", "all_noncurrent_planned_genes_nonbinding",
        "all_capacity_limits_respected", "all_pending_partitions_exact",
        "all_deferred_remainders_explicit", "all_reservations_integral",
        "decisions",
    }
)
_CAPACITY_DECISION_KEYS = frozenset(
    {
        "audit_schema_version", "queue_contract", "source_version",
        "objective_variant", "information_regime", "future_schedule_accessed",
        "selection_time_step", "source_cell", "current_block_label",
        "observed_arrived_pending_labels", "planned_labels", "deferred_labels",
        "observed_arrived_pending_count", "planned_count", "deferred_count",
        "admissible_candidate_count", "admissible_candidate_cells",
        "candidate_mask_sha256", "queue_partition_sha256",
        "current_shared_mask_capacity", "optimizer_seed", "optimizer_config",
        "optimizer_seed_offset_contract", "egress_weight", "optimizer_mode",
        "objective_lexicographic_fields", "best_assignment",
        "best_cost_components", "best_cost_scalar", "chosen_cell",
        "current_block_planned", "current_block_plan_index",
        "current_block_is_gene_zero",
        "current_block_is_unique_observed_pickup_head",
        "current_block_is_fifo_head", "only_gene_zero_committed",
        "committed_gene_indices", "noncurrent_planned_genes_nonbinding",
        "capacity_respected", "pending_partition_exact",
        "deferred_remainder_explicit", "deferred_semantics",
        "queue_overflow_labels", "queue_overflow_count",
        "scheduler_defer_action_introduced", "queue_overflow_virtual_penalty",
        "proposal_id", "candidate_mask_id", "decision_index",
        "commit_time_step", "reservation_integrity",
    }
)
_GA_CONFIG_KEYS = frozenset(
    {
        "population_size", "generations", "elite_count", "tournament_size",
        "crossover_rate", "mutation_rate", "seed",
    }
)
FROZEN_PICKUP_CELL = (1, 3)


def _is_nonnegative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_cell(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(_is_nonnegative_int(item) for item in value)
    )


def _is_exact_zero_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def _capacity_audit_issues(
    raw: Mapping,
    *,
    method: str,
    seed: int,
    expected_pickup_cell: Sequence[int] = FROZEN_PICKUP_CELL,
    expected_delivery_count: int = EXPECTED_DELIVERIES,
) -> list[str]:
    """Return exact source/selector/reservation audit violations."""

    issues: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            issues.append(label)

    selector = raw.get("selector_audit")
    options = raw.get("scheduler_audit")
    if not isinstance(selector, Mapping):
        return ["missing_selector_audit"]
    if not isinstance(options, Mapping):
        return ["missing_scheduler_option_audit"]
    source = selector.get("source_audit")
    if not isinstance(source, Mapping):
        return ["missing_capacity_aware_source_audit"]

    expected_source = REPAIRED_METHOD_TO_SOURCE[method]
    expected_version = REPAIRED_METHOD_TO_VERSION[method]
    expected_objective = REPAIRED_METHOD_TO_OBJECTIVE[method]
    expected_lex = (
        LEGACY_LEX_FIELDS if method == REPAIRED_2009_METHOD else DURATION_LEX_FIELDS
    )
    expected_egress = (
        None if method == REPAIRED_2009_METHOD else base.ROLLING_GA_EGRESS_WEIGHT
    )
    expected_base_config = {
        "population_size": base.ROLLING_POPULATION,
        "generations": base.ROLLING_GENERATIONS,
        "elite_count": 2,
        "tournament_size": 3,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "seed": base.GA_SEED_BASE + seed,
    }
    check(set(source) == _CAPACITY_SOURCE_AUDIT_KEYS, "source_audit_schema_keys")
    check(raw.get("assignment_source") == expected_source, "raw_assignment_source_mismatch")
    check(raw.get("assignment_source_version") == expected_version, "raw_assignment_source_version_mismatch")
    check(selector.get("assignment_source") == expected_source, "selector_assignment_source_mismatch")
    check(selector.get("assignment_source_version") == expected_version, "selector_source_version_mismatch")
    check(source.get("audit_schema_version") == CAPACITY_AWARE_AUDIT_SCHEMA_VERSION, "source_audit_schema_mismatch")
    check(source.get("queue_contract") == CAPACITY_AWARE_QUEUE_CONTRACT, "source_queue_contract_mismatch")
    check(source.get("assignment_source_version") == expected_version, "source_version_mismatch")
    check(source.get("objective_variant") == expected_objective, "source_objective_variant_mismatch")
    check(tuple(source.get("objective_lexicographic_fields", ())) == expected_lex, "source_objective_lex_fields_mismatch")
    check(source.get("information_regime") == "online_arrived_only", "source_information_regime_mismatch")
    check(source.get("future_schedule_accessed") is False, "future_schedule_accessed")
    check(tuple(source.get("future_schedule_fields_read", ())) == (), "future_schedule_fields_read")
    check(
        tuple(source.get("observable_fields_used", ()))
        == (
            "position",
            "stored",
            "delivered",
            "realized_arrival_step",
            "current_remaining_storage_time",
        ),
        "source_observable_fields_mismatch",
    )
    check(
        source.get("pending_order")
        == "mandatory_current_then_earliest_arrived_stable",
        "source_pending_order_mismatch",
    )
    check(
        source.get("preview_contract")
        == (
            "environment_seed_and_decision_counter_invariant_with_"
            "replaceable_diagnostic_cache_v1"
        ),
        "source_preview_contract_mismatch",
    )
    base_config = source.get("optimizer_base_config")
    check(isinstance(base_config, Mapping), "source_optimizer_base_config_missing")
    if isinstance(base_config, Mapping):
        check(set(base_config) == _GA_CONFIG_KEYS, "source_optimizer_base_config_schema")
        check(
            all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in base_config.values()
            ),
            "source_optimizer_base_config_numeric_type",
        )
        check(dict(base_config) == expected_base_config, "source_optimizer_base_config_mismatch")
    check(
        source.get("optimizer_seed_offset_contract")
        == "base_seed_plus_committed_assignment_index",
        "source_optimizer_seed_contract_mismatch",
    )
    check(source.get("egress_weight") == expected_egress, "source_egress_weight_mismatch")
    check(
        source.get("deferred_semantics")
        == "nonbinding_rolling_horizon_overflow_not_an_executed_defer",
        "source_overflow_semantics_mismatch",
    )
    check(source.get("scheduler_defer_action_introduced") is False, "overflow_introduced_scheduler_defer")
    check(_is_exact_zero_number(source.get("queue_overflow_virtual_penalty")), "overflow_virtual_penalty_nonzero_or_invalid")
    check(source.get("half_minted_unbound_preview_present") is False, "half_minted_preview_at_episode_end")
    for field in (
        "decision_count",
        "committed_plan_count",
        "minted_unreserved_preview_count",
        "reserved_uncommitted_preview_count",
        "decisions_with_deferred_remainder",
        "observed_pending_membership_count_total",
        "planned_membership_count_total",
        "deferred_membership_count_total",
        "max_observed_arrived_pending_count",
        "max_planned_count",
        "max_deferred_count",
    ):
        check(_is_nonnegative_int(source.get(field)), f"source_invalid_nonnegative_count:{field}")
    check(source.get("reserved_uncommitted_preview_count") == 0, "reserved_preview_left_uncommitted")
    minted_unreserved_ids = tuple(
        source.get("minted_unreserved_proposal_ids", ())
    )
    reserved_uncommitted_ids = tuple(
        source.get("reserved_uncommitted_proposal_ids", ())
    )
    check(
        all(isinstance(value, str) and bool(value) for value in minted_unreserved_ids)
        and len(minted_unreserved_ids) == len(set(minted_unreserved_ids)),
        "minted_unreserved_proposal_ids_invalid",
    )
    check(
        source.get("minted_unreserved_preview_count")
        == len(minted_unreserved_ids),
        "minted_unreserved_count_id_mismatch",
    )
    check(
        all(isinstance(value, str) and bool(value) for value in reserved_uncommitted_ids)
        and len(reserved_uncommitted_ids) == len(set(reserved_uncommitted_ids)),
        "reserved_uncommitted_proposal_ids_invalid",
    )
    check(
        source.get("reserved_uncommitted_preview_count")
        == len(reserved_uncommitted_ids),
        "reserved_uncommitted_count_id_mismatch",
    )
    for field in (
        "all_current_blocks_planned",
        "all_current_blocks_are_gene_zero",
        "all_current_blocks_are_unique_observed_pickup_heads",
        "all_current_blocks_are_fifo_heads",
        "only_gene_zero_ever_committed",
        "all_noncurrent_planned_genes_nonbinding",
        "all_capacity_limits_respected",
        "all_pending_partitions_exact",
        "all_deferred_remainders_explicit",
        "all_reservations_integral",
    ):
        check(source.get(field) is True, f"source_false:{field}")
    delivery_count = raw.get("delivery_count")
    check(_is_nonnegative_int(delivery_count), "invalid_delivery_count")

    source_decisions = source.get("decisions")
    selector_decisions = selector.get("decisions")
    if not isinstance(source_decisions, list) or not isinstance(selector_decisions, list):
        return sorted(set((*issues, "missing_source_or_selector_decisions")))
    selector_valid = [item for item in selector_decisions if item.get("valid") is True]
    decision_count = len(source_decisions)
    check(decision_count == expected_delivery_count, "source_decision_count_not_expected")
    check(delivery_count == expected_delivery_count, "delivery_count_not_expected")
    check(decision_count == delivery_count, "decision_delivery_count_mismatch")
    check(
        tuple(source.get("committed_gene_indices", ())) == ((0,) if decision_count else ()),
        "source_nonzero_or_missing_gene_commit",
    )
    for name, value in (
        ("source.decision_count", source.get("decision_count")),
        ("source.committed_plan_count", source.get("committed_plan_count")),
        ("selector.valid_assignment_count", selector.get("valid_assignment_count")),
        ("selector.reserved_commit_count", selector.get("reserved_commit_count")),
        ("option.inbound_successes", options.get("inbound_successes")),
        ("option.reservation_bound_count", options.get("reservation_bound_count")),
        ("option.reservation_commit_count", options.get("reservation_commit_count")),
        ("option.reservation_execution_match_count", options.get("reservation_execution_match_count")),
    ):
        check(_is_nonnegative_int(value), f"invalid_decision_count:{name}")
        check(value == decision_count, f"decision_count_mismatch:{name}")
    check(len(selector_valid) == decision_count, "selector_valid_decision_length_mismatch")
    check(
        _is_nonnegative_int(options.get("reservation_invalidation_count")),
        "reservation_invalidation_invalid",
    )
    check(options.get("reservation_invalidation_count") == 0, "reservation_invalidation_nonzero")
    check(raw.get("reservation_integrity") is True, "raw_reservation_integrity_false")

    selector_by_proposal = {
        str(item.get("proposal_id")): item for item in selector_valid
    }
    source_ids = [str(item.get("proposal_id")) for item in source_decisions]
    check(len(source_ids) == len(set(source_ids)), "duplicate_source_proposal_id")
    check(
        set(minted_unreserved_ids).isdisjoint(source_ids),
        "discarded_preview_id_also_committed",
    )
    check(
        set(reserved_uncommitted_ids).isdisjoint(source_ids),
        "reserved_uncommitted_id_also_committed",
    )
    check(
        set(minted_unreserved_ids).isdisjoint(reserved_uncommitted_ids),
        "preview_id_both_unreserved_and_reserved",
    )
    check(set(source_ids) == set(selector_by_proposal), "source_selector_proposal_set_mismatch")
    check(list(options.get("bound_proposal_ids", ())) == source_ids, "bound_proposal_order_mismatch")
    check(list(options.get("committed_proposal_ids", ())) == source_ids, "committed_proposal_order_mismatch")

    observed_total = planned_total = overflow_total = 0
    decisions_with_overflow = 0
    maximum_observed = maximum_planned = maximum_overflow = 0
    for index, record in enumerate(source_decisions):
        prefix = f"decision[{index}]"
        observed = tuple(record.get("observed_arrived_pending_labels", ()))
        planned = tuple(record.get("planned_labels", ()))
        overflow = tuple(record.get("deferred_labels", ()))
        candidates = tuple(tuple(cell) for cell in record.get("admissible_candidate_cells", ()))
        assignment = tuple(tuple(cell) for cell in record.get("best_assignment", ()))
        current = record.get("current_block_label")
        chosen = tuple(record.get("chosen_cell", ()))
        selector_record = selector_by_proposal.get(str(record.get("proposal_id")), {})
        check(set(record) == _CAPACITY_DECISION_KEYS, f"{prefix}:schema_keys")
        check(record.get("audit_schema_version") == CAPACITY_AWARE_AUDIT_SCHEMA_VERSION, f"{prefix}:schema")
        check(record.get("queue_contract") == CAPACITY_AWARE_QUEUE_CONTRACT, f"{prefix}:queue_contract")
        check(record.get("source_version") == expected_version, f"{prefix}:source_version")
        check(record.get("objective_variant") == expected_objective, f"{prefix}:objective")
        check(record.get("information_regime") == "online_arrived_only", f"{prefix}:information_regime")
        check(record.get("future_schedule_accessed") is False, f"{prefix}:future_schedule")
        check(_is_nonnegative_int(record.get("decision_index")) and record.get("decision_index") == index, f"{prefix}:decision_index")
        check(
            all(isinstance(label, str) and bool(label) for label in observed),
            f"{prefix}:missing_or_invalid_label",
        )
        check(len(observed) == len(set(observed)), f"{prefix}:duplicate_observed_label")
        check(set(planned).isdisjoint(overflow), f"{prefix}:overlapping_partition_labels")
        check(observed == (*planned, *overflow), f"{prefix}:pending_partition")
        check(len(planned) > 0 and planned[0] == current, f"{prefix}:current_not_gene_zero")
        check(
            _is_nonnegative_int(record.get("current_block_plan_index"))
            and record.get("current_block_plan_index") == 0,
            f"{prefix}:current_index",
        )
        check(record.get("current_block_is_gene_zero") is True, f"{prefix}:gene_zero_flag")
        check(record.get("current_block_is_fifo_head") is True, f"{prefix}:fifo_flag")
        check(record.get("current_block_is_unique_observed_pickup_head") is True, f"{prefix}:pickup_flag")
        check(record.get("only_gene_zero_committed") is True, f"{prefix}:commit_flag")
        check(tuple(record.get("committed_gene_indices", ())) == (0,), f"{prefix}:committed_indices")
        check(record.get("noncurrent_planned_genes_nonbinding") is True, f"{prefix}:tail_binding")
        check(record.get("current_block_planned") is True, f"{prefix}:current_planned_flag")
        check(record.get("capacity_respected") is True, f"{prefix}:capacity_flag")
        check(record.get("pending_partition_exact") is True, f"{prefix}:partition_flag")
        check(record.get("deferred_remainder_explicit") is True, f"{prefix}:overflow_explicit_flag")
        check(_is_cell(record.get("source_cell")), f"{prefix}:invalid_source_cell")
        check(tuple(record.get("source_cell", ())) == tuple(expected_pickup_cell), f"{prefix}:source_not_frozen_pickup_cell")
        check(all(_is_cell(cell) for cell in candidates), f"{prefix}:invalid_candidate_cell")
        check(_is_cell(record.get("chosen_cell")), f"{prefix}:invalid_chosen_cell")
        check(len(planned) <= len(candidates), f"{prefix}:capacity")
        check(len(assignment) == len(planned), f"{prefix}:assignment_length")
        check(len(assignment) == len(set(assignment)), f"{prefix}:duplicate_assignment")
        check(all(cell in candidates for cell in assignment), f"{prefix}:assignment_outside_mask")
        check(bool(assignment) and chosen == assignment[0], f"{prefix}:chosen_not_gene_zero")
        check(tuple(record.get("queue_overflow_labels", ())) == overflow, f"{prefix}:overflow_labels")
        check(record.get("deferred_semantics") == "nonbinding_rolling_horizon_overflow_not_an_executed_defer", f"{prefix}:overflow_semantics")
        check(record.get("scheduler_defer_action_introduced") is False, f"{prefix}:scheduler_defer")
        check(
            _is_exact_zero_number(record.get("queue_overflow_virtual_penalty")),
            f"{prefix}:overflow_penalty",
        )
        for count_field, expected_count in (
            ("observed_arrived_pending_count", len(observed)),
            ("planned_count", len(planned)),
            ("deferred_count", len(overflow)),
            ("queue_overflow_count", len(overflow)),
            ("admissible_candidate_count", len(candidates)),
            ("current_shared_mask_capacity", len(candidates)),
        ):
            check(
                _is_nonnegative_int(record.get(count_field)),
                f"{prefix}:invalid_{count_field}",
            )
            check(record.get(count_field) == expected_count, f"{prefix}:{count_field}")
        mask_sha = _candidate_mask_sha(candidates)
        check(record.get("candidate_mask_sha256") == mask_sha, f"{prefix}:mask_sha")
        check(record.get("candidate_mask_id") == mask_sha[:16], f"{prefix}:mask_id")
        check(record.get("queue_partition_sha256") == _queue_partition_sha(observed, planned, overflow), f"{prefix}:queue_sha")
        selection_step = record.get("selection_time_step")
        commit_step = record.get("commit_time_step")
        check(_is_nonnegative_int(selection_step), f"{prefix}:invalid_selection_step")
        check(_is_nonnegative_int(commit_step), f"{prefix}:invalid_commit_step")
        check(
            _is_nonnegative_int(selection_step)
            and _is_nonnegative_int(commit_step)
            and commit_step >= selection_step,
            f"{prefix}:commit_precedes_selection",
        )
        check(
            isinstance(record.get("proposal_id"), str)
            and bool(record.get("proposal_id")),
            f"{prefix}:missing_proposal_id",
        )
        check(selector_record.get("candidate_mask_id") == mask_sha[:16], f"{prefix}:selector_mask")
        check(
            _is_nonnegative_int(selector_record.get("candidate_count"))
            and selector_record.get("candidate_count") == len(candidates),
            f"{prefix}:selector_candidate_count",
        )
        check(tuple(selector_record.get("chosen_cell", ())) == chosen, f"{prefix}:selector_chosen")
        check(selector_record.get("block_label") == current, f"{prefix}:selector_block")
        check(
            _is_nonnegative_int(selector_record.get("selection_time_step"))
            and selector_record.get("selection_time_step") == selection_step,
            f"{prefix}:selector_selection_time",
        )
        check(
            _is_nonnegative_int(selector_record.get("commit_time_step"))
            and selector_record.get("commit_time_step") == commit_step,
            f"{prefix}:selector_commit_time",
        )
        expected_seed = base.GA_SEED_BASE + seed + index
        check(_is_nonnegative_int(record.get("optimizer_seed")), f"{prefix}:invalid_optimizer_seed")
        check(record.get("optimizer_seed") == expected_seed, f"{prefix}:optimizer_seed")
        config = record.get("optimizer_config")
        if not isinstance(config, Mapping):
            issues.append(f"{prefix}:missing_optimizer_config")
        else:
            expected_config = {
                "population_size": base.ROLLING_POPULATION,
                "generations": base.ROLLING_GENERATIONS,
                "elite_count": 2,
                "tournament_size": 3,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "seed": expected_seed,
            }
            check(set(config) == _GA_CONFIG_KEYS, f"{prefix}:optimizer_config_schema")
            check(
                all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in config.values()
                ),
                f"{prefix}:optimizer_config_numeric_type",
            )
            check(dict(config) == expected_config, f"{prefix}:optimizer_config")
        expected_mode = (
            "exhaustive_singleton"
            if method == REPAIRED_COMPLETE_METHOD and len(planned) == 1
            else "genetic_search"
        )
        check(record.get("optimizer_mode") == expected_mode, f"{prefix}:optimizer_mode")
        check(tuple(record.get("objective_lexicographic_fields", ())) == expected_lex, f"{prefix}:lex_fields")
        components = record.get("best_cost_components")
        if not isinstance(components, Mapping):
            issues.append(f"{prefix}:missing_cost_components")
        else:
            check(tuple(components) == expected_lex, f"{prefix}:cost_component_order")
            check(
                all(_is_nonnegative_int(value) for value in components.values()),
                f"{prefix}:cost_component_type_or_sign",
            )
            check(
                _is_nonnegative_int(record.get("best_cost_scalar")),
                f"{prefix}:cost_scalar_type_or_sign",
            )
            try:
                check(int(record.get("best_cost_scalar")) == _cost_scalar(components, expected_lex), f"{prefix}:cost_scalar")
            except (KeyError, TypeError, ValueError):
                issues.append(f"{prefix}:invalid_cost_components")
        check(record.get("egress_weight") == expected_egress, f"{prefix}:egress_weight")
        check(
            record.get("optimizer_seed_offset_contract")
            == "base_seed_plus_committed_assignment_index",
            f"{prefix}:optimizer_seed_contract",
        )
        check(record.get("reservation_integrity") is True, f"{prefix}:reservation_integrity")

        observed_total += len(observed)
        planned_total += len(planned)
        overflow_total += len(overflow)
        decisions_with_overflow += int(bool(overflow))
        maximum_observed = max(maximum_observed, len(observed))
        maximum_planned = max(maximum_planned, len(planned))
        maximum_overflow = max(maximum_overflow, len(overflow))

    for field, expected in (
        ("observed_pending_membership_count_total", observed_total),
        ("planned_membership_count_total", planned_total),
        ("deferred_membership_count_total", overflow_total),
        ("decisions_with_deferred_remainder", decisions_with_overflow),
        ("max_observed_arrived_pending_count", maximum_observed),
        ("max_planned_count", maximum_planned),
        ("max_deferred_count", maximum_overflow),
    ):
        check(source.get(field) == expected, f"aggregate:{field}")
    return sorted(set(issues))


_REPAIR_EXTRA_ROW_KEYS = frozenset(
    {
        "assignment_source_version",
        "historical_lineage_method_id",
        "capacity_aware_queue_contract",
        "capacity_aware_source_audit_schema_version",
        "capacity_aware_source_audit",
        "capacity_aware_selector_audit",
        "capacity_aware_reservation_audit",
        "capacity_aware_reservation_integrity",
        "capacity_aware_source_audit_issues",
        "capacity_aware_source_audit_summary",
    }
)
_REPAIR_ROW_KEYS = frozenset(
    {*base._BASELINE_NORMALIZED_ROW_KEYS, *_REPAIR_EXTRA_ROW_KEYS}
)


def _validate_repair_row_schema(row: Mapping) -> None:
    observed = frozenset(row)
    if observed != _REPAIR_ROW_KEYS:
        raise CapacityRepairComparisonError(
            "repaired normalized row schema mismatch: "
            f"missing={sorted(_REPAIR_ROW_KEYS - observed)!r}, "
            f"extra={sorted(observed - _REPAIR_ROW_KEYS)!r}"
        )


def _source_audit_summary(source: Optional[Mapping]) -> Optional[dict]:
    if not isinstance(source, Mapping):
        return None
    return {
        "decision_count": source.get("decision_count"),
        "committed_plan_count": source.get("committed_plan_count"),
        "minted_unreserved_preview_count": source.get("minted_unreserved_preview_count"),
        "reserved_uncommitted_preview_count": source.get("reserved_uncommitted_preview_count"),
        "decisions_with_queue_overflow": source.get("decisions_with_deferred_remainder"),
        "observed_pending_membership_count_total": source.get("observed_pending_membership_count_total"),
        "planned_membership_count_total": source.get("planned_membership_count_total"),
        "queue_overflow_membership_count_total": source.get("deferred_membership_count_total"),
        "max_observed_arrived_pending_count": source.get("max_observed_arrived_pending_count"),
        "max_planned_count": source.get("max_planned_count"),
        "max_queue_overflow_count": source.get("max_deferred_count"),
    }


def _normalize_repaired_run(
    method: str,
    raw: Mapping,
    *,
    instance,
    identities: Mapping[int, Mapping],
) -> dict:
    seed = int(raw["eval_seed"])
    issues = _capacity_audit_issues(raw, method=method, seed=seed)
    if issues:
        raise CapacityRepairComparisonError(
            f"capacity-aware source audit failed for {method}:{seed}: {issues!r}"
        )
    historical = REPAIRED_METHOD_TO_HISTORICAL[method]
    row = base._normalize_deterministic_baseline(
        historical, raw, instance, identities
    )
    source_audit = raw["selector_audit"]["source_audit"]
    row.update(
        {
            "protocol": PROTOCOL,
            "method_id": method,
            "assignment_source": REPAIRED_METHOD_TO_SOURCE[method],
            "assignment_source_version": REPAIRED_METHOD_TO_VERSION[method],
            "historical_lineage_method_id": historical,
            "capacity_aware_queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
            "capacity_aware_source_audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
            "capacity_aware_source_audit": source_audit,
            # Preserve the complete selector and inbound-option evidence used
            # for the one-to-one proposal/reservation/commit authentication.
            # This lets resume validation recompute the proof rather than
            # trusting a normalization-time boolean.
            "capacity_aware_selector_audit": raw["selector_audit"],
            "capacity_aware_reservation_audit": raw["scheduler_audit"],
            "capacity_aware_reservation_integrity": bool(
                raw["reservation_integrity"]
            ),
            "capacity_aware_source_audit_issues": (),
            "capacity_aware_source_audit_summary": _source_audit_summary(source_audit),
            "source_origin": "new_capacity_aware_partial_ga_execution",
        }
    )
    _validate_repair_row_schema(row)
    return row


def _failed_repaired_row(
    method: str,
    seed: int,
    identities: Mapping[int, Mapping],
    error: Exception,
) -> dict:
    historical = REPAIRED_METHOD_TO_HISTORICAL[method]
    row = base._failed_baseline_row(historical, seed, identities, error)
    row.update(
        {
            "protocol": PROTOCOL,
            "method_id": method,
            "assignment_source": REPAIRED_METHOD_TO_SOURCE[method],
            "assignment_source_version": REPAIRED_METHOD_TO_VERSION[method],
            "historical_lineage_method_id": historical,
            "capacity_aware_queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
            "capacity_aware_source_audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
            "capacity_aware_source_audit": None,
            "capacity_aware_selector_audit": None,
            "capacity_aware_reservation_audit": None,
            "capacity_aware_reservation_integrity": None,
            "capacity_aware_source_audit_issues": (
                f"execution_exception:{type(error).__name__}:{error}",
            ),
            "capacity_aware_source_audit_summary": None,
            "method_failure_reason": (
                f"execution_exception:{type(error).__name__}:{error}"
            ),
            "source_origin": "new_capacity_aware_partial_ga_execution_failure",
        }
    )
    _validate_repair_row_schema(row)
    return row


def _repair_ledger_path(output_dir: Path, method: str, seed: int) -> Path:
    return output_dir / "run-ledger" / method / f"seed-{seed}.json"


def _scan_repair_ledger_tree(
    output_dir: Path, *, require_complete: bool
) -> tuple[Path, ...]:
    """Reject every noncanonical ledger path, including symlinks and temps."""

    root = output_dir / "run-ledger"
    allowed_files = {
        _repair_ledger_path(output_dir, method, seed).resolve()
        for method in REPAIRED_METHODS
        for seed in PANEL_SEEDS
    }
    allowed_directories = {root.resolve()}
    allowed_directories.update(
        (root / method).resolve() for method in REPAIRED_METHODS
    )
    if root.is_symlink():
        raise CapacityRepairComparisonError(
            "repair run-ledger root may not be a symlink"
        )
    if not root.exists():
        if require_complete:
            raise CapacityRepairComparisonError(
                "completed repair run has no run-ledger directory"
            )
        return ()
    if not root.is_dir():
        raise CapacityRepairComparisonError(
            "repair run-ledger root must be a canonical regular directory"
        )

    observed_files = set()
    for directory_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory = Path(directory_text)
        if directory.is_symlink() or directory.resolve() not in allowed_directories:
            raise CapacityRepairComparisonError(
                f"noncanonical repair ledger directory: {directory}"
            )
        for name in tuple(directory_names):
            child = directory / name
            if child.is_symlink() or child.resolve() not in allowed_directories:
                raise CapacityRepairComparisonError(
                    f"extra or symlinked repair ledger directory: {child}"
                )
        for name in file_names:
            child = directory / name
            if child.is_symlink() or not child.is_file():
                raise CapacityRepairComparisonError(
                    f"repair ledger must be a regular non-symlink file: {child}"
                )
            resolved = child.resolve()
            if resolved not in allowed_files:
                raise CapacityRepairComparisonError(
                    f"extra/temp/protected repair ledger path: {child}"
                )
            observed_files.add(resolved)
    if require_complete and observed_files != allowed_files:
        missing = sorted(str(path) for path in allowed_files - observed_files)
        raise CapacityRepairComparisonError(
            f"completed repair ledger tree is missing canonical rows: {missing!r}"
        )
    return tuple(sorted(observed_files))


def _recheck_path_snapshot(
    paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
    *,
    label: str,
) -> dict[str, str]:
    _require_equal(f"{label} path-key set", set(paths), set(expected_sha256))
    observed = {}
    for key, path in sorted(paths.items()):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise CapacityRepairComparisonError(
                f"{label} is missing or noncanonical: {key} -> {path}"
            )
        observed[key] = _sha256_file(path.resolve())
    _require_equal(f"{label} post-run SHA snapshot", observed, expected_sha256)
    return observed


def _validate_repaired_row_semantics(
    row: Mapping,
    *,
    method: str,
    seed: int,
    input_contract: Mapping,
    identities: Mapping[int, Mapping],
) -> None:
    _validate_repair_row_schema(row)
    for field, expected in {
        "protocol": PROTOCOL,
        "method_id": method,
        "method_category": "deterministic_online_primary",
        "information_regime": "online_arrived_only",
        "instance_seed": seed,
        "assignment_source": REPAIRED_METHOD_TO_SOURCE[method],
        "assignment_source_version": REPAIRED_METHOD_TO_VERSION[method],
        "historical_lineage_method_id": REPAIRED_METHOD_TO_HISTORICAL[method],
        "capacity_aware_queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
        "capacity_aware_source_audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
        "evaluation_learning": False,
        "evaluation_epsilon": 0.0,
        "source_execution_reused": False,
        "source_checkpoint_sha256": None,
        "source_validation_checkpoint_sha256": None,
        "source_ledger_sha256": None,
    }.items():
        _require_equal(f"repaired row.{field}", row.get(field), expected)
    identity = base._identity_fields(seed, identities)
    for field in identity:
        _require_equal(f"repaired row identity.{field}", row.get(field), identity[field])
    _require_equal(
        "repaired input instance SHA",
        row.get("episode_instance_sha256"),
        input_contract["episode_instance_sha256"],
    )
    if row.get("strict_method_success") is True:
        if row.get("capacity_aware_source_audit_issues") not in ((), []):
            raise CapacityRepairComparisonError(
                "successful repaired row carries capacity source audit issues"
            )
        source = row.get("capacity_aware_source_audit")
        if not isinstance(source, Mapping):
            raise CapacityRepairComparisonError(
                "successful repaired row lacks its full capacity source audit"
            )
        selector_audit = row.get("capacity_aware_selector_audit")
        reservation_audit = row.get("capacity_aware_reservation_audit")
        if not isinstance(selector_audit, Mapping) or not isinstance(
            reservation_audit, Mapping
        ):
            raise CapacityRepairComparisonError(
                "successful repaired row lacks selector/reservation evidence"
            )
        _require_equal(
            "repaired row selector/source audit binding",
            selector_audit.get("source_audit"),
            source,
        )
        recomputed_audit_issues = _capacity_audit_issues(
            {
                "assignment_source": row["assignment_source"],
                "assignment_source_version": row[
                    "assignment_source_version"
                ],
                "delivery_count": row["delivery_count"],
                "reservation_integrity": row[
                    "capacity_aware_reservation_integrity"
                ],
                "selector_audit": selector_audit,
                "scheduler_audit": reservation_audit,
            },
            method=method,
            seed=seed,
        )
        if recomputed_audit_issues:
            raise CapacityRepairComparisonError(
                "successful repaired ledger fails recomputed capacity audit: "
                f"{recomputed_audit_issues!r}"
            )
        legacy = row.get("legacy_environment_return")
        deviations = row.get("delivery_deviations")
        if legacy is None or not math.isfinite(float(legacy)):
            raise CapacityRepairComparisonError(
                "successful repaired row lacks finite legacy return"
            )
        _, expected_dense = _dual_rescore_from_legacy_return(
            float(legacy), deviations, base.FROZEN_OBJECTIVE_SPEC
        )
        if not math.isclose(
            float(row.get("dense_objective_return")),
            float(expected_dense),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise CapacityRepairComparisonError(
                "successful repaired row dense objective fails exact rescore"
            )
        row_issues = _repaired_row_issues(row, identities)
        if row_issues:
            raise CapacityRepairComparisonError(
                "successful repaired ledger fails row safety authentication: "
                f"{row_issues!r}"
            )


def _load_or_execute_repaired(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    input_contract: Mapping,
    identities: Mapping[int, Mapping],
    execute: bool,
    executor: Callable[[], dict],
) -> Optional[dict]:
    path = _repair_ledger_path(output_dir, method, seed)
    fingerprint = _digest_json(input_contract)
    if path.is_file():
        ledger = _load_json(path)
        _require_equal(
            "repair ledger exact schema",
            set(ledger),
            {
                "schema_version",
                "protocol",
                "run_key",
                "input_contract",
                "input_fingerprint",
                "run",
                "ledger_sha256",
            },
        )
        _require_equal("repair ledger schema version", ledger.get("schema_version"), SCHEMA_VERSION)
        _require_equal("repair ledger protocol", ledger.get("protocol"), PROTOCOL)
        _require_equal("repair ledger run key", ledger.get("run_key"), f"{method}:{seed}")
        _require_equal("repair ledger input", ledger.get("input_contract"), input_contract)
        _require_equal("repair ledger fingerprint", ledger.get("input_fingerprint"), fingerprint)
        _verify_self_hash(ledger, "ledger_sha256", name=f"repair ledger {method}:{seed}")
        row = ledger.get("run")
        if not isinstance(row, Mapping):
            raise CapacityRepairComparisonError("repair ledger run is not an object")
    elif not execute:
        return None
    else:
        row = executor()
        _validate_repaired_row_semantics(
            row,
            method=method,
            seed=seed,
            input_contract=input_contract,
            identities=identities,
        )
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "run_key": f"{method}:{seed}",
            "input_contract": dict(input_contract),
            "input_fingerprint": fingerprint,
            "run": dict(row),
        }
        ledger["ledger_sha256"] = _digest_json(ledger)
        _atomic_json(path, ledger)
    _validate_repaired_row_semantics(
        row,
        method=method,
        seed=seed,
        input_contract=input_contract,
        identities=identities,
    )
    return dict(row)


def execute_or_load_repaired_grid(
    *,
    output_dir: Path,
    sources: base.AuthenticatedSources,
    repair_contract: Mapping,
    device,
    execute: bool,
    evaluator: Callable = evaluate_assignment_ablation_one,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    runtime_args = base._baseline_args(
        {"environment": sources.v23_contract["environment"]}, device
    )
    rows = []
    missing = []
    for method in REPAIRED_METHODS:
        for seed in PANEL_SEEDS:
            instance = sources.instances[seed]
            input_contract = _repair_input_contract(
                method,
                seed,
                sources=sources,
                repair_contract=repair_contract,
            )

            def run_one(method=method, seed=seed, instance=instance):
                try:
                    raw = evaluator(
                        runtime_args,
                        seed,
                        None,
                        assignment_source=REPAIRED_METHOD_TO_SOURCE[method],
                        episode_instance=instance,
                    )
                    return _normalize_repaired_run(
                        method,
                        raw,
                        instance=instance,
                        identities=sources.identities,
                    )
                except Exception as error:
                    return _failed_repaired_row(
                        method, seed, sources.identities, error
                    )

            row = _load_or_execute_repaired(
                output_dir=output_dir,
                method=method,
                seed=seed,
                input_contract=input_contract,
                identities=sources.identities,
                execute=execute,
                executor=run_one,
            )
            if row is None:
                missing.append(
                    {
                        "method_id": method,
                        "instance_seed": seed,
                        "ledger_path": str(
                            _repair_ledger_path(output_dir, method, seed)
                        ),
                    }
                )
            else:
                rows.append(row)
                if execute:
                    dense = row.get("dense_objective_return")
                    dense_text = "NA" if dense is None else f"{float(dense):8.2f}"
                    overflow = (row.get("capacity_aware_source_audit_summary") or {}).get(
                        "queue_overflow_membership_count_total"
                    )
                    print(
                        f"[{method}] instance={seed} DenseR={dense_text} "
                        f"strict={int(bool(row['strict_method_success']))} "
                        f"overflow_memberships={overflow} "
                        f"failure={row['method_failure_reason']}",
                        flush=True,
                    )
    return tuple(rows), tuple(missing)


def _proxy_for_base_safety(row: Mapping) -> dict:
    proxy = dict(row)
    historical = REPAIRED_METHOD_TO_HISTORICAL[row["method_id"]]
    proxy["protocol"] = base.PROTOCOL
    proxy["method_id"] = historical
    proxy["assignment_source"] = base.DETERMINISTIC_METHOD_TO_SOURCE[historical]
    proxy.pop("assignment_source_version", None)
    proxy.pop("historical_lineage_method_id", None)
    proxy.pop("capacity_aware_queue_contract", None)
    proxy.pop("capacity_aware_source_audit_schema_version", None)
    proxy.pop("capacity_aware_source_audit", None)
    proxy.pop("capacity_aware_selector_audit", None)
    proxy.pop("capacity_aware_reservation_audit", None)
    proxy.pop("capacity_aware_reservation_integrity", None)
    proxy.pop("capacity_aware_source_audit_issues", None)
    proxy.pop("capacity_aware_source_audit_summary", None)
    return proxy


def _repaired_row_issues(
    row: Mapping, identities: Mapping[int, Mapping]
) -> list[str]:
    method = row.get("method_id")
    if method not in REPAIRED_METHODS:
        return ["unknown_repaired_method"]
    issues = list(base._row_safety_issues(_proxy_for_base_safety(row), identities))
    if row.get("assignment_source") != REPAIRED_METHOD_TO_SOURCE[method]:
        issues.append("repaired_assignment_source_mismatch")
    if row.get("assignment_source_version") != REPAIRED_METHOD_TO_VERSION[method]:
        issues.append("repaired_source_version_mismatch")
    if row.get("capacity_aware_queue_contract") != CAPACITY_AWARE_QUEUE_CONTRACT:
        issues.append("repaired_queue_contract_mismatch")
    if row.get("capacity_aware_source_audit_schema_version") != CAPACITY_AWARE_AUDIT_SCHEMA_VERSION:
        issues.append("repaired_audit_schema_mismatch")
    if row.get("capacity_aware_source_audit_issues") not in ((), []):
        issues.append("repaired_source_audit_failed")
    if row.get("capacity_aware_reservation_integrity") is not True:
        issues.append("repaired_reservation_integrity_false")
    source = row.get("capacity_aware_source_audit")
    if not isinstance(source, Mapping):
        issues.append("repaired_source_audit_missing")
    else:
        if source.get("reserved_uncommitted_preview_count") != 0:
            issues.append("repaired_reservation_left_uncommitted")
        if source.get("half_minted_unbound_preview_present") is not False:
            issues.append("repaired_half_minted_preview_present")
        if source.get("future_schedule_accessed") is not False:
            issues.append("repaired_future_schedule_accessed")
    return sorted(set(issues))


def _summarize_repaired_method(
    method: str,
    rows: Sequence[Mapping],
    identities: Mapping[int, Mapping],
) -> dict:
    selected = tuple(row for row in rows if row.get("method_id") == method)
    observed = [int(row.get("instance_seed", -1)) for row in selected]
    grid_issues = []
    if len(selected) != len(PANEL_SEEDS) or set(observed) != set(PANEL_SEEDS):
        grid_issues.append("deterministic_instance_grid_incomplete_or_duplicate")
    row_issues = []
    for row in selected:
        issues = _repaired_row_issues(row, identities)
        if issues:
            row_issues.append(
                {
                    "instance_seed": row.get("instance_seed"),
                    "model_seed": None,
                    "policy_rng_index": None,
                    "issues": issues,
                }
            )
    eligible = not grid_issues and not row_issues
    summary = {
        "method_id": method,
        "category": "deterministic_online_primary_capacity_aware_repair",
        "historical_lineage_method_id": REPAIRED_METHOD_TO_HISTORICAL[method],
        "assignment_source": REPAIRED_METHOD_TO_SOURCE[method],
        "assignment_source_version": REPAIRED_METHOD_TO_VERSION[method],
        "queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
        "source_row_count": len(selected),
        "expected_source_row_count": len(PANEL_SEEDS),
        "instance_count": len(set(observed)),
        "grid_issues": grid_issues,
        "row_safety_issues": row_issues,
        "whole_method_numeric_eligible": eligible,
        "complete_case_filtering_used": False,
        "performance_metrics_suppressed_if_any_row_unsafe": True,
        "estimand": "equal_12_instances_one_deterministic_run_each",
        "observed_strict_success_rate": (
            float(fmean(bool(row.get("strict_method_success")) for row in selected))
            if selected
            else None
        ),
        "training_seed_replication": {
            "n": 0,
            "variability_estimable": False,
            "status": "deterministic_method_not_applicable",
        },
        "capacity_audit_aggregate": {
            "total_committed_decisions": sum(
                int((row.get("capacity_aware_source_audit_summary") or {}).get("decision_count", 0) or 0)
                for row in selected
            ),
            "total_queue_overflow_memberships": sum(
                int((row.get("capacity_aware_source_audit_summary") or {}).get("queue_overflow_membership_count_total", 0) or 0)
                for row in selected
            ),
            "episodes_with_any_queue_overflow": sum(
                int((row.get("capacity_aware_source_audit_summary") or {}).get("decisions_with_queue_overflow", 0) > 0)
                for row in selected
            ),
            "total_discarded_lookahead_previews": sum(
                int((row.get("capacity_aware_source_audit_summary") or {}).get("minted_unreserved_preview_count", 0) or 0)
                for row in selected
            ),
            "discarded_lookahead_previews_are_not_reservations_or_defer_actions": True,
        },
    }
    if not eligible:
        summary.update(
            {
                "metrics": None,
                "instance_cluster_points": (),
                "instance_cluster_statistics": None,
                "model_seed_summaries": None,
                "delivery_position_metrics": None,
                "descriptive_rehandle_nominal_one_sided_95_ucb": None,
            }
        )
        return summary
    points = base._instance_cluster_points(selected)
    metrics = base._mean_cluster_metrics(points)
    stats = {
        field: base._cluster_stat(points, field)
        for field in (
            "mean_dense_objective_return",
            "mean_signed_deviation",
            "mean_absolute_error",
            "mean_tardiness",
            "mean_earliness",
            "within_target_window_rate",
            "mean_steps",
            "physical_rehandles_per_100_required_deliveries",
        )
    }
    summary.update(
        {
            "metrics": metrics,
            "instance_cluster_points": points,
            "instance_cluster_statistics": stats,
            "model_seed_summaries": None,
            "delivery_position_metrics": base._delivery_position_metrics(selected),
            "descriptive_rehandle_nominal_one_sided_95_ucb": stats[
                "physical_rehandles_per_100_required_deliveries"
            ]["nominal_one_sided_95_upper_bound"],
        }
    )
    return summary


def _deterministic_outcome_equivalence(
    rows: Sequence[Mapping], summaries: Sequence[Mapping]
) -> dict:
    deterministic = set(base.DETERMINISTIC_METHODS) | set(REPAIRED_METHODS)
    eligible = {
        item["method_id"]
        for item in summaries
        if item["method_id"] in deterministic
        and item["whole_method_numeric_eligible"]
    }
    by_method = {
        method: {
            int(row["instance_seed"]): row
            for row in rows
            if row.get("method_id") == method
        }
        for method in eligible
    }
    fields = (
        "dense_objective_return",
        "delivery_deviations",
        "steps",
        "physical_storage_relocations",
        "target_bound_obstruction_clearances",
        "standalone_reconfigurations",
    )
    pairs = []
    from itertools import combinations

    for left, right in combinations(sorted(eligible), 2):
        differences = []
        for seed in PANEL_SEEDS:
            differing = [
                field
                for field in fields
                if base._json_safe(by_method[left][seed].get(field))
                != base._json_safe(by_method[right][seed].get(field))
            ]
            if differing:
                differences.append(
                    {"instance_seed": seed, "differing_fields": differing}
                )
        pairs.append(
            {
                "left": left,
                "right": right,
                "exactly_outcome_equivalent_on_all_12_instances": not differences,
                "per_instance_differences": differences,
            }
        )
    return {
        "compared_fields": fields,
        "pairs": pairs,
        "exact_equivalent_pairs": tuple(
            {"left": item["left"], "right": item["right"]}
            for item in pairs
            if item["exactly_outcome_equivalent_on_all_12_instances"]
        ),
        "lineage_variants_are_not_independent_evidence": True,
    }


def build_expanded_report(
    *,
    matched: AuthenticatedMatchedSource,
    repaired_rows: Sequence[Mapping],
    sources: base.AuthenticatedSources,
    repair_contract: Mapping,
) -> dict:
    old_summaries = tuple(matched.report["method_summaries"])
    repaired_summaries = tuple(
        _summarize_repaired_method(method, repaired_rows, sources.identities)
        for method in REPAIRED_METHODS
    )
    summaries = (*old_summaries, *repaired_summaries)
    rows = (*matched.rows, *repaired_rows)
    screen = base._seed_stability_screen(summaries, sources)
    screen.update(
        {
            "screen_name": "development_method_capacity_or_pareto_screen",
            "frozen_before_capacity_aware_repair_outcomes": True,
            "not_evidence_of_training_seed_stability": True,
        }
    )
    method_registry = dict(matched.report["method_registry"])
    method_registry.update(
        {
            "ranked_primary_online": EXPANDED_PRIMARY_METHODS,
            "capacity_aware_repaired_primary": tuple(
                {
                    "method_id": method,
                    "source_id": REPAIRED_METHOD_TO_SOURCE[method],
                    "source_version": REPAIRED_METHOD_TO_VERSION[method],
                    "historical_lineage": REPAIRED_METHOD_TO_HISTORICAL[method],
                    "same_objective_as_historical_lineage": True,
                    "new_evidence_not_an_independent_literature_method": True,
                }
                for method in REPAIRED_METHODS
            ),
            "historical_overconstrained_ga_adapters_retained": tuple(
                {
                    "method_id": method,
                    "status": "immutable_historical_result_retained;whole_method_safety_suppressed_if_failed",
                    "repaired_successor": repaired,
                }
                for repaired, method in REPAIRED_METHOD_TO_HISTORICAL.items()
            ),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "complete_case_filtering_used": False,
        "repair_contract_sha256": repair_contract["contract_sha256"],
        "immutable_matched_source": {
            "protocol": base.PROTOCOL,
            "report_raw_sha256": MATCHED_REPORT_RAW_SHA256,
            "report_canonical_sha256": MATCHED_REPORT_CANONICAL_SHA256,
            "rows_reused": len(matched.rows),
            "policy_executions": 0,
            "source_time_provenance_preserved": True,
        },
        "mixed_source_time_provenance": {
            "historical_rows": "authenticated_under_pinned_original_source_hashes",
            "new_repaired_rows": "executed_under_repair_contract_current_source_hashes",
            "legacy_optional_hook_behavior": "no_op_for_sources_without_hook_methods",
            "historical_rows_reinterpreted_under_current_code": False,
        },
        "objective_spec": base.FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_operational_return",
        "common_metric_definitions": matched.report["common_metric_definitions"],
        "primary_online_methods": EXPANDED_PRIMARY_METHODS,
        "method_summaries": summaries,
        "whole_method_safety_exclusions": tuple(
            item["method_id"]
            for item in summaries
            if not item["whole_method_numeric_eligible"]
        ),
        "rankings_are_metric_specific_no_composite_winner": True,
        "metric_rankings": base._rank_methods(summaries),
        "pareto_fronts": base._pareto_fronts(summaries),
        "paired_descriptive_v2_3_contrasts": base._paired_descriptive_contrasts(summaries),
        "deterministic_outcome_equivalence": _deterministic_outcome_equivalence(rows, summaries),
        "development_method_capacity_or_pareto_screen": screen,
        "seed_stability_screen": screen,
        "v2_3_authoritative_development_selection": matched.report[
            "v2_3_authoritative_development_selection"
        ],
        "controller_specific_diagnostics_not_cross_family_metrics": matched.report[
            "controller_specific_diagnostics_not_cross_family_metrics"
        ],
        "v2_2_superseded_diagnostic": matched.report["v2_2_superseded_diagnostic"],
        "method_registry": method_registry,
        "statistical_interpretation": matched.report["statistical_interpretation"],
        "queue_overflow_diagnostics": {
            summary["method_id"]: summary["capacity_audit_aggregate"]
            for summary in repaired_summaries
        },
        "run_count": len(rows),
        "historical_reused_run_count": len(matched.rows),
        "new_repaired_run_count": len(repaired_rows),
        "runs": rows,
    }


CSV_FIELDS = tuple(
    dict.fromkeys(
        (
            *base.CSV_FIELDS,
            "assignment_source_version",
            "historical_lineage_method_id",
            "capacity_aware_queue_contract",
            "capacity_aware_source_audit_schema_version",
            "capacity_aware_source_audit_issues",
            "capacity_aware_source_audit_summary",
            "capacity_aware_source_audit",
            "capacity_aware_selector_audit",
            "capacity_aware_reservation_audit",
            "capacity_aware_reservation_integrity",
        )
    )
)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                record = {field: row.get(field) for field in CSV_FIELDS}
                for field in (
                    "delivery_deviations",
                    "capacity_aware_source_audit_issues",
                    "capacity_aware_source_audit_summary",
                    "capacity_aware_source_audit",
                    "capacity_aware_selector_audit",
                    "capacity_aware_reservation_audit",
                ):
                    record[field] = json.dumps(
                        base._json_safe(record.get(field)),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                writer.writerow(record)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _repair_ledger_manifest(output_dir: Path) -> tuple[dict, ...]:
    records = []
    for method in REPAIRED_METHODS:
        for seed in PANEL_SEEDS:
            path = _repair_ledger_path(output_dir, method, seed)
            if not path.is_file():
                raise CapacityRepairComparisonError(
                    f"completed repair grid is missing ledger: {path}"
                )
            ledger = _load_json(path)
            records.append(
                {
                    "run_key": ledger["run_key"],
                    "path": str(path.resolve()),
                    "raw_sha256": _sha256_file(path),
                    "ledger_sha256": _verify_self_hash(
                        ledger, "ledger_sha256", name="repair audit ledger"
                    ),
                    "input_fingerprint": ledger["input_fingerprint"],
                }
            )
    return tuple(records)


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate the completed V2.3 matched development comparison "
            "and optionally execute exactly four capacity-aware GA repairs x12"
        )
    )
    parser.add_argument(
        "--v2-3-source-dir",
        type=Path,
        default=root / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
    )
    parser.add_argument(
        "--v1-1-control-dir",
        type=Path,
        default=root / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
    )
    parser.add_argument(
        "--v2-2-source-dir",
        type=Path,
        default=root / "results/vcg-constrained-v2-2-development-seed10-200ep",
    )
    parser.add_argument(
        "--matched-source-dir",
        type=Path,
        default=root / "results/vcg-v2-3-matched-baselines-85k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/vcg-v2-3-capacity-aware-ga-repair-85k",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--execute-repaired-ga",
        action="store_true",
        help="execute missing repaired rows; absent means authenticate/plan only",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume only ledgers matching the exact saved repair contract",
    )
    return parser


def _assert_disjoint_output(output_dir: Path, source_dirs: Iterable[Path]) -> None:
    for source_dir in source_dirs:
        source_dir = source_dir.resolve()
        if (
            output_dir == source_dir
            or output_dir.is_relative_to(source_dir)
            or source_dir.is_relative_to(output_dir)
        ):
            raise CapacityRepairComparisonError(
                "output directory must be disjoint from every immutable source directory"
            )


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    source_dirs = (
        args.v2_3_source_dir.resolve(),
        args.v1_1_control_dir.resolve(),
        args.v2_2_source_dir.resolve(),
        args.matched_source_dir.resolve(),
    )
    _assert_disjoint_output(output_dir, source_dirs)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing only for this exact contract"
        )

    # This authentication opens only already-used 85xxx development artifacts.
    sources = base.authenticate_sources(
        v23_source_dir=args.v2_3_source_dir,
        v11_control_dir=args.v1_1_control_dir,
        v22_source_dir=args.v2_2_source_dir,
    )
    matched = authenticate_completed_matched_comparison(
        args.matched_source_dir,
        sources=sources,
    )
    learned_source_before = _recheck_path_snapshot(
        sources.source_paths,
        sources.source_sha256,
        label="authenticated learned/source-chain artifacts",
    )
    historical_ledger_before = _recheck_path_snapshot(
        matched.historical_ledger_paths,
        matched.historical_ledger_raw_sha256,
        label="immutable historical baseline ledgers",
    )
    device = resolve_device(args.device)
    repair_contract = build_repair_contract(
        sources=sources,
        matched=matched,
        device=str(device),
    )
    source_paths = _current_source_paths()
    source_before = _source_hashes(source_paths)
    _require_equal(
        "repair contract/current source hashes",
        repair_contract["current_repair_source_sha256"],
        source_before,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / CONTRACT_FILENAME
    if contract_path.is_file():
        _require_equal("resumed repair contract", _load_json(contract_path), repair_contract)
    else:
        if (output_dir / "run-ledger").exists():
            raise CapacityRepairComparisonError(
                "cannot resume repair ledgers without their exact contract"
            )
        _atomic_json(contract_path, repair_contract)

    _scan_repair_ledger_tree(output_dir, require_complete=False)
    repaired_rows, missing = execute_or_load_repaired_grid(
        output_dir=output_dir,
        sources=sources,
        repair_contract=repair_contract,
        device=device,
        execute=bool(args.execute_repaired_ga),
    )
    source_after = _source_hashes(source_paths)
    _require_equal("repair implementation sources unchanged", source_after, source_before)
    _scan_repair_ledger_tree(output_dir, require_complete=not missing)
    learned_source_after = _recheck_path_snapshot(
        sources.source_paths,
        learned_source_before,
        label="authenticated learned/source-chain artifacts",
    )
    historical_ledger_after = _recheck_path_snapshot(
        matched.historical_ledger_paths,
        historical_ledger_before,
        label="immutable historical baseline ledgers",
    )
    for path_name, path in matched.paths.items():
        _require_equal(
            f"immutable matched source unchanged {path_name}",
            _sha256_file(path),
            matched.raw_sha256[path_name],
        )

    preflight = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if not missing else "prepared_missing_repaired_ledgers",
        "execute_repaired_ga_requested": bool(args.execute_repaired_ga),
        "repair_contract": str(contract_path),
        "repair_contract_sha256": repair_contract["contract_sha256"],
        "authenticated_historical_rows_reused": len(matched.rows),
        "historical_policy_executions": 0,
        "loaded_or_executed_repaired_rows": len(repaired_rows),
        "expected_repaired_rows": len(REPAIRED_METHODS) * len(PANEL_SEEDS),
        "missing_repaired_rows": missing,
        "only_four_new_methods_executable": True,
        "final_86xxx_panel_opened": False,
        "performance_claim_authorized": False,
    }
    _atomic_json(output_dir / PREFLIGHT_FILENAME, preflight)
    if missing:
        print(
            f"Prepared/authenticated repair contract; {len(missing)} repaired rows "
            "remain. Re-run with --execute-repaired-ga --resume-existing.",
            flush=True,
        )
        return preflight

    _require_equal(
        "complete repaired grid row count",
        len(repaired_rows),
        len(REPAIRED_METHODS) * len(PANEL_SEEDS),
    )
    report = build_expanded_report(
        matched=matched,
        repaired_rows=repaired_rows,
        sources=sources,
        repair_contract=repair_contract,
    )
    report["report_sha256"] = _digest_json(report)
    all_rows = (*matched.rows, *repaired_rows)
    _write_csv(output_dir / RUNS_FILENAME, all_rows)
    _atomic_json(output_dir / REPORT_FILENAME, report)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "development_only": True,
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "repair_contract_sha256": repair_contract["contract_sha256"],
        "historical_matched_source_raw_sha256": matched.raw_sha256,
        "historical_rows_reused": len(matched.rows),
        "historical_policy_executions": 0,
        "new_repaired_grid_row_count": len(repaired_rows),
        "new_method_execution_grid": "exactly_4_methods_x_12_opened_instances",
        "nearest_dynamic_and_historical_ga_rows_reexecuted": False,
        "repair_source_sha256_before": source_before,
        "repair_source_sha256_after": source_after,
        "repair_sources_unchanged": True,
        "authenticated_learned_source_sha256_before": learned_source_before,
        "authenticated_learned_source_sha256_after": learned_source_after,
        "authenticated_learned_sources_unchanged": True,
        "historical_baseline_ledger_sha256_before": historical_ledger_before,
        "historical_baseline_ledger_sha256_after": historical_ledger_after,
        "historical_baseline_ledgers_unchanged": True,
        "repair_ledger_manifest": _repair_ledger_manifest(output_dir),
        "expanded_runs_raw_sha256": _sha256_file(output_dir / RUNS_FILENAME),
        "expanded_report_raw_sha256": _sha256_file(output_dir / REPORT_FILENAME),
        "whole_method_safety_exclusions": report["whole_method_safety_exclusions"],
    }
    audit["audit_sha256"] = _digest_json(audit)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(f"Runs: {output_dir / RUNS_FILENAME}", flush=True)
    print(f"Report: {output_dir / REPORT_FILENAME}", flush=True)
    print(f"Audit: {output_dir / AUDIT_FILENAME}", flush=True)
    return report


if __name__ == "__main__":
    main()
