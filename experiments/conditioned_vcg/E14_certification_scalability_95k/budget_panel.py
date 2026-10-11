#!/usr/bin/env python3
"""E14: same-controller certificate-budget sensitivity with E14+D12 enabled."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e14_budget_scalability_95k_v1"
SCHEMA_VERSION = 1
PARENT_E13 = ROOT / "results/vcg-conditioned-e13-scalability-95k-v2"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e14-budget-scalability-95k"
CONTRACT_NAME = "e14-budget-contract.json"
REPORT_NAME = "e14-budget-report.json"
TABLE_NAME = "e14-budget-table.md"

INSTANCE_SEED = 95_100
BUDGETS = (2, 4, 8, 16, 20_000)
EXECUTED_BUDGETS = BUDGETS[:-1]
REFERENCE_BUDGET = BUDGETS[-1]
SCENARIOS = e13.MAIN_SCENARIOS


class E14BudgetError(RuntimeError):
    pass


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise E14BudgetError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise E14BudgetError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E14BudgetError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E14BudgetError(f"{label} must contain an object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise E14BudgetError(f"{label} self-hash mismatch")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parent() -> tuple[dict, dict]:
    contract, manifest, _inputs = e13.authenticate_frozen(PARENT_E13)
    if contract["protocol"] != e13.PROTOCOL:
        raise E14BudgetError("unexpected E13 parent protocol")
    if contract["main_scenarios"] != list(SCENARIOS):
        raise E14BudgetError("E13 main panel changed")
    if contract["search_config"]["max_nodes"] != REFERENCE_BUDGET:
        raise E14BudgetError("E13 does not supply the declared reference budget")
    return contract, manifest


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(e13.__file__).resolve(),
        ROOT / "PSLAP/relocation_family_certification.py",
        ROOT / "PSLAP/viability.py",
        ROOT / "PSLAP/viability_candidates.py",
        ROOT / "PSLAP/viability_filter.py",
        ROOT
        / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def expected_contract() -> dict:
    parent_contract, parent_manifest = _parent()
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "question": (
                "how_does_native_certificate_budget_trade_computation_for_"
                "certified_coverage_with_E14_cleanup_and_D12_enabled"
            ),
            "paper_role": "E14_certificate_budget_and_latency_scalability",
            "scenarios": list(SCENARIOS),
            "instance_seed": INSTANCE_SEED,
            "budgets_max_nodes": list(BUDGETS),
            "executed_budgets_max_nodes": list(EXECUTED_BUDGETS),
            "reference_budget_max_nodes": REFERENCE_BUDGET,
            "expected_executed_rows": len(SCENARIOS) * len(EXECUTED_BUDGETS),
            "expected_logical_rows": len(SCENARIOS) * len(BUDGETS),
            "reference_rows_reused_from_e13": True,
            "reference_compute_rerun": False,
            "model_seed": e13.MODEL_SEED,
            "preference_lambda": e13.PREFERENCE_LAMBDA,
            "device": "cpu",
            "training": False,
            "implementation": {
                "path_enumeration_cleanup": "e14_order_preserving_cleanup",
                "timing_invariant_cache": True,
                "relocation_family_strategy": (
                    e13.RELOCATION_FAMILY_CERTIFICATION
                ),
                "family_miss_behavior": (
                    "native_exact_fallback_under_same_budget"
                ),
            },
            "budget_arm_anchor_rule": (
                "fresh_per_episode_cache_and_anchor_acquired_under_arm_budget"
            ),
            "cross_budget_positive_proof_reuse": False,
            "unknown_reused_as_negative_proof": False,
            "unsafe_interpretation": (
                "only_exhaustive_native_search_is_impossibility_evidence"
            ),
            "wall_limit_seconds_per_episode": e13.WALL_LIMIT_SECONDS,
            "deadline_seconds": list(e13.DEADLINE_SECONDS),
            "primary_deadline_seconds": e13.PRIMARY_DEADLINE_SECONDS,
            "one_process_per_episode_required": True,
            "parent_e13_contract_sha256": parent_contract["contract_sha256"],
            "parent_e13_manifest_sha256": parent_manifest["manifest_sha256"],
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    contract = expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        if _load(path, "E14 budget contract") != contract:
            raise E14BudgetError("E14 budget contract, sources, or inputs changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E14BudgetError("nonempty E14 budget output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, contract)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "scenarios": len(SCENARIOS),
        "budgets": list(BUDGETS),
        "new_episode_runs": contract["expected_executed_rows"],
        "reference_rows_reused": len(SCENARIOS),
        "training": False,
    }


def authenticate(output: Path) -> dict:
    contract = _load(output / CONTRACT_NAME, "E14 budget contract")
    if contract != expected_contract():
        raise E14BudgetError("E14 budget contract, sources, or inputs changed")
    return contract


def authenticate_frozen(output: Path) -> dict:
    """Authenticate the completed E14 sweep under its recorded code hashes."""

    contract = _self_hashed(
        output / CONTRACT_NAME,
        "contract_sha256",
        "frozen E14 budget contract",
    )
    parent_contract, parent_manifest = _parent()
    if (
        contract.get("protocol") != PROTOCOL
        or contract.get("scenarios") != list(SCENARIOS)
        or contract.get("budgets_max_nodes") != list(BUDGETS)
        or contract.get("parent_e13_contract_sha256")
        != parent_contract["contract_sha256"]
        or contract.get("parent_e13_manifest_sha256")
        != parent_manifest["manifest_sha256"]
    ):
        raise E14BudgetError("frozen E14 contract binding changed")
    return contract


def _ledger_path(output: Path, budget: int, scenario: str) -> Path:
    return output / "run-ledger" / f"max-nodes-{budget}" / f"{scenario}.json"


def _authenticate_ledger(
    output: Path, contract: Mapping, budget: int, scenario: str
) -> dict:
    ledger = _self_hashed(
        _ledger_path(output, budget, scenario),
        "ledger_sha256",
        "E14 budget run ledger",
    )
    row = ledger.get("row", {})
    if (
        ledger.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("manifest_sha256")
        != contract["parent_e13_manifest_sha256"]
        or row.get("protocol") != PROTOCOL
        or row.get("scenario_id") != scenario
        or int(row.get("instance_seed", -1)) != INSTANCE_SEED
        or int(row.get("search_max_nodes", -1)) != int(budget)
    ):
        raise E14BudgetError("E14 budget ledger coordinate mismatch")
    return ledger


def _reference_row(contract: Mapping, scenario: str) -> dict:
    parent_contract, _manifest = _parent()
    ledger = e13._authenticate_ledger(
        PARENT_E13, parent_contract, scenario, INSTANCE_SEED
    )
    row = dict(ledger["row"])
    if int(row["search_max_nodes"]) != REFERENCE_BUDGET:
        raise E14BudgetError("E13 reference row has wrong budget")
    row["budget_row_source"] = "authenticated_E13_reference"
    row["e13_ledger_sha256"] = ledger["ledger_sha256"]
    return row


def run_one(output: Path, budget: int, scenario: str) -> dict:
    if scenario not in SCENARIOS:
        raise E14BudgetError(f"unsupported E14 budget scenario: {scenario}")
    if int(budget) not in BUDGETS:
        raise E14BudgetError(f"unsupported E14 max-nodes budget: {budget}")
    contract = authenticate(output)
    if int(budget) == REFERENCE_BUDGET:
        row = _reference_row(contract, scenario)
        return {
            "status": "reused_reference",
            "scenario": scenario,
            "max_nodes": int(budget),
            "strict_safe_complete": row["strict_safe_complete"],
            "failure_class": row["failure_class"],
            "e13_ledger_sha256": row["e13_ledger_sha256"],
        }
    path = _ledger_path(output, int(budget), scenario)
    if path.exists():
        ledger = _authenticate_ledger(output, contract, int(budget), scenario)
    else:
        ledger = e13.run_coordinate(
            PARENT_E13,
            scenario,
            INSTANCE_SEED,
            max_nodes=int(budget),
            contract_override=contract,
            ledger_path_override=path,
        )
        ledger = _authenticate_ledger(output, contract, int(budget), scenario)
    row = ledger["row"]
    return {
        "status": "completed" if row["strict_safe_complete"] else "recorded",
        "scenario": scenario,
        "max_nodes": int(budget),
        "strict_safe_complete": row["strict_safe_complete"],
        "failure_class": row["failure_class"],
        "deliveries": f'{row["deliveries_at_stop"]}/{row["required_deliveries"]}',
        "decisions": row["macro_decisions"],
        "episode_wall_seconds": row["episode_wall_seconds"],
        "family_proofs": row["relocation_family_proof_count"],
        "family_misses": row["relocation_family_miss_count"],
        "native_searches": row["native_search_count"],
        "native_exact_search_seconds": row["native_exact_search_seconds"],
        "ledger": str(path.relative_to(ROOT)),
    }


def _rows(output: Path, contract: Mapping) -> list[dict]:
    rows = []
    for budget in BUDGETS:
        for scenario in SCENARIOS:
            if budget == REFERENCE_BUDGET:
                path = e13._ledger_path(PARENT_E13, scenario, INSTANCE_SEED)
                if path.exists():
                    rows.append(_reference_row(contract, scenario))
            else:
                path = _ledger_path(output, budget, scenario)
                if path.exists():
                    row = dict(
                        _authenticate_ledger(
                            output, contract, budget, scenario
                        )["row"]
                    )
                    row["budget_row_source"] = "executed_E14_arm"
                    rows.append(row)
    return rows


def _distribution(values: Sequence[float]) -> dict:
    return e13._distribution(tuple(float(value) for value in values))


def _budget_summary(budget: int, rows: Sequence[Mapping]) -> dict:
    frontiers = [item for row in rows for item in row.get("frontiers", ())]
    decisions = [item for row in rows for item in row.get("decision_costs", ())]
    warm = [item for item in decisions if not item["cold_first_decision"]]
    calls = [
        item
        for frontier in frontiers
        for item in frontier["native_certification_calls"]
    ]
    physical = sum(int(item["physical_candidate_count"]) for item in frontiers)
    certified = sum(
        int(item["certified_candidate_count"]) for item in frontiers
    )
    candidates_without_defer = sum(
        int(item["physical_candidate_count"])
        - int(item["physical_defer_count"])
        for item in frontiers
    )
    unknown = sum(int(item["unknown_candidate_count"]) for item in frontiers)
    family_attempts = sum(
        int(item["relocation_family_attempt_count"]) for item in frontiers
    )
    family_proofs = sum(
        int(item["relocation_family_proof_count"]) for item in frontiers
    )
    strict = [row for row in rows if row["strict_safe_complete"]]
    return {
        "max_nodes": int(budget),
        "expected_rows": len(SCENARIOS),
        "observed_rows": len(rows),
        "complete_budget_arm": len(rows) == len(SCENARIOS),
        "row_sources": dict(
            Counter(row["budget_row_source"] for row in rows)
        ),
        "failure_classes": dict(Counter(row["failure_class"] for row in rows)),
        "strict_safe_complete": len(strict),
        "strict_completion_rate": len(strict) / len(rows) if rows else None,
        "deliveries_at_stop": sum(
            int(row["deliveries_at_stop"] or 0) for row in rows
        ),
        "required_deliveries": sum(
            int(row["required_deliveries"]) for row in rows
        ),
        "candidate_space": {
            "physical_total": physical,
            "certified_total": certified,
            "certified_to_physical_ratio": (
                certified / physical if physical else None
            ),
            "unknown_count": unknown,
            "unknown_rate_excluding_defer": (
                unknown / candidates_without_defer
                if candidates_without_defer
                else None
            ),
            "empty_frontier_count": sum(
                bool(item["no_positively_certified_candidate"])
                for item in frontiers
            ),
            "empty_frontier_rate": (
                sum(
                    bool(item["no_positively_certified_candidate"])
                    for item in frontiers
                )
                / len(frontiers)
                if frontiers
                else None
            ),
            "physical_per_frontier": _distribution(
                [item["physical_candidate_count"] for item in frontiers]
            ),
            "certified_per_frontier": _distribution(
                [item["certified_candidate_count"] for item in frontiers]
            ),
        },
        "family": {
            "anchor_available_rate": (
                sum(item["relocation_family_anchor_available"] for item in frontiers)
                / len(frontiers)
                if frontiers
                else None
            ),
            "attempts": family_attempts,
            "proofs": family_proofs,
            "misses": sum(
                int(item["relocation_family_miss_count"]) for item in frontiers
            ),
            "proof_rate": (
                family_proofs / family_attempts if family_attempts else None
            ),
            "setup_seconds": sum(
                float(item["relocation_family_setup_seconds"])
                for item in frontiers
            ),
            "connection_seconds": sum(
                float(item["relocation_family_connection_seconds"])
                for item in frontiers
            ),
        },
        "native_searches": {
            "count": len(calls),
            "status_counts": dict(Counter(item["status"] for item in calls)),
            "total_seconds": sum(float(item["seconds"]) for item in calls),
            "per_frontier": len(calls) / len(frontiers) if frontiers else None,
            "expanded_nodes": _distribution(
                [item["explored_nodes"] for item in calls]
            ),
            "seconds_per_search": _distribution(
                [item["seconds"] for item in calls]
            ),
            "by_role": {
                role: e13._call_distribution(
                    [item for item in calls if item["check_role"] == role]
                )
                for role in ("current_state", "accept", "deliver", "reconfigure")
            },
        },
        "latency": {
            "episode_wall_seconds": _distribution(
                [row["episode_wall_seconds"] for row in rows]
            ),
            "warm_end_to_end_decision_seconds": _distribution(
                [item["end_to_end_decision_seconds"] for item in warm]
            ),
            "cold_first_decision_seconds": _distribution(
                [
                    item["end_to_end_decision_seconds"]
                    for item in decisions
                    if item["cold_first_decision"]
                ]
            ),
            "warm_deadline_exceedance_rate": {
                str(deadline): (
                    sum(
                        item["end_to_end_decision_seconds"] > deadline
                        for item in warm
                    )
                    / len(warm)
                    if warm
                    else None
                )
                for deadline in e13.DEADLINE_SECONDS
            },
        },
        "memory": {
            "peak_rss_kib": _distribution(
                [row["process_peak_rss_after_kib"] for row in rows]
            )
        },
    }


def _fmt(value, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _table(summaries: Sequence[Mapping]) -> str:
    lines = [
        "# E14 certificate-budget and latency scalability",
        "",
        (
            "Each arm uses its own fresh cache and obtains its anchor under "
            "that arm's budget. The 20,000-node arm reuses authenticated E13 "
            "rows rather than repeating their compute."
        ),
        "",
        "| Max nodes | Strict | Deliveries | Certified/physical | UNKNOWN | Empty frontiers | D12 proofs | D12 misses | Native searches | Expansions p95 | Warm p95 (s) | >1 s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        candidate = item["candidate_space"]
        native = item["native_searches"]
        latency = item["latency"]
        family = item["family"]
        lines.append(
            "| {budget} | {strict}/{expected} | {delivered}/{required} | "
            "{coverage} | {unknown} | {empty} | {proofs} | {misses} | "
            "{searches} | {expansion} | {p95} | {deadline} |".format(
                budget=item["max_nodes"],
                strict=item["strict_safe_complete"],
                expected=item["expected_rows"],
                delivered=item["deliveries_at_stop"],
                required=item["required_deliveries"],
                coverage=_fmt(candidate["certified_to_physical_ratio"], 3),
                unknown=candidate["unknown_count"],
                empty=candidate["empty_frontier_count"],
                proofs=family["proofs"],
                misses=family["misses"],
                searches=native["count"],
                expansion=_fmt(native["expanded_nodes"]["p95"], 1),
                p95=_fmt(latency["warm_end_to_end_decision_seconds"]["p95"]),
                deadline=_fmt(
                    None
                    if latency["warm_deadline_exceedance_rate"]["1.0"] is None
                    else 100.0
                    * latency["warm_deadline_exceedance_rate"]["1.0"],
                    1,
                ),
            )
        )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, allow_partial: bool = False) -> dict:
    contract = authenticate_frozen(output)
    rows = _rows(output, contract)
    expected = len(SCENARIOS) * len(BUDGETS)
    if len(rows) != expected and not allow_partial:
        raise E14BudgetError(
            f"expected {expected} logical E14 rows, found {len(rows)}"
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["search_max_nodes"])].append(row)
    summaries = [
        _budget_summary(budget, grouped.get(budget, ())) for budget in BUDGETS
    ]
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "expected_logical_rows": expected,
            "observed_logical_rows": len(rows),
            "partial": len(rows) != expected,
            "all_budget_arms_complete": len(rows) == expected,
            "reference_compute_reused_not_rerun": True,
            "summaries": summaries,
            "interpretation_scope": (
                "single_frozen_instance_seed_across_nine_scale_by_occupancy_"
                "coordinates_with_fresh_per_arm_anchors"
            ),
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    _atomic_text(output / TABLE_NAME, _table(summaries))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "authenticate", "run-one", "analyze")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--max-nodes", type=int, choices=BUDGETS)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "authenticate":
        contract = authenticate(output)
        result = {
            "status": "authenticated",
            "contract_sha256": contract["contract_sha256"],
        }
    elif args.command == "run-one":
        if args.scenario is None or args.max_nodes is None:
            raise E14BudgetError("run-one needs --scenario and --max-nodes")
        prepare(output)
        result = run_one(output, args.max_nodes, args.scenario)
    else:
        report = analyze(output, allow_partial=args.allow_partial)
        result = {
            "status": "analyzed",
            "observed_logical_rows": report["observed_logical_rows"],
            "expected_logical_rows": report["expected_logical_rows"],
            "partial": report["partial"],
            "report": str((output / REPORT_NAME).relative_to(ROOT)),
            "table": str((output / TABLE_NAME).relative_to(ROOT)),
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
