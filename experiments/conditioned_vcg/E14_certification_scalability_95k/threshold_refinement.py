#!/usr/bin/env python3
"""Post-hoc E14 refinement between the 16-node and reference-budget arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    budget_panel as e14,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e14_budget_threshold_refinement_95k_v1"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e14-threshold-refinement-95k"
CONTRACT_NAME = "threshold-contract.json"
REPORT_NAME = "threshold-report.json"
TABLE_NAME = "threshold-table.md"
REFINEMENT_BUDGETS = (32, 64)
ALL_OBSERVED_BUDGETS = (2, 4, 8, 16, 32, 64, 20_000)


class ThresholdRefinementError(RuntimeError):
    pass


def _parent() -> tuple[dict, dict, dict, dict]:
    e13_contract, e13_manifest, _inputs = e13.authenticate_frozen(
        e13.DEFAULT_OUTPUT
    )
    e14_contract = e14.authenticate_frozen(e14.DEFAULT_OUTPUT)
    e14_report = e14._self_hashed(
        e14.DEFAULT_OUTPUT / e14.REPORT_NAME,
        "report_sha256",
        "complete E14 budget report",
    )
    if (
        e14_report.get("contract_sha256") != e14_contract["contract_sha256"]
        or e14_report.get("all_budget_arms_complete") is not True
        or e14_report.get("observed_logical_rows")
        != e14_report.get("expected_logical_rows")
    ):
        raise ThresholdRefinementError("parent E14 sweep is incomplete")
    return e13_contract, e13_manifest, e14_contract, e14_report


def _reference_search_maximum() -> int:
    e13_contract, _manifest, _inputs = e13.authenticate_frozen(
        e13.DEFAULT_OUTPUT
    )
    maximum = 0
    for scenario in e13.MAIN_SCENARIOS:
        ledger = e13._authenticate_ledger(
            e13.DEFAULT_OUTPUT, e13_contract, scenario, e14.INSTANCE_SEED
        )
        for frontier in ledger["row"]["frontiers"]:
            for call in frontier["native_certification_calls"]:
                maximum = max(maximum, int(call["explored_nodes"]))
    return maximum


def _conservative_power_of_two(maximum: int) -> int:
    if maximum < 1:
        raise ThresholdRefinementError("reference trace has no native searches")
    return 1 << math.ceil(math.log2(maximum + 1))


def expected_contract() -> dict:
    e13_contract, e13_manifest, e14_contract, e14_report = _parent()
    maximum = _reference_search_maximum()
    return e14._with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "paper_role": "post_hoc_E14_threshold_resolution",
            "reason": (
                "observed_initial_anchor_requirements_lie_between_16_and_94_"
                "nodes_so_the_predeclared_jump_from_16_to_20000_is_too_coarse"
            ),
            "post_hoc_declared_after_parent_E14_outcomes": True,
            "refinement_budgets_max_nodes": list(REFINEMENT_BUDGETS),
            "scenarios": list(e13.MAIN_SCENARIOS),
            "instance_seed": e14.INSTANCE_SEED,
            "expected_new_rows": len(REFINEMENT_BUDGETS)
            * len(e13.MAIN_SCENARIOS),
            "training": False,
            "controller_and_E14_D12_stack_unchanged": True,
            "fresh_per_episode_cache_and_same_budget_anchor": True,
            "cross_budget_proof_reuse": False,
            "reference_budget_rerun": False,
            "reference_trace_maximum_explored_nodes": maximum,
            "conservative_reference_equivalent_cap": (
                _conservative_power_of_two(maximum)
            ),
            "reference_equivalence_scope": (
                "deterministic_native_searches_on_the_observed_seed95100_"
                "reference_trajectories_only"
            ),
            "parent_e13_contract_sha256": e13_contract["contract_sha256"],
            "parent_e13_manifest_sha256": e13_manifest["manifest_sha256"],
            "parent_e14_contract_sha256": e14_contract["contract_sha256"],
            "parent_e14_report_sha256": e14_report["report_sha256"],
            "source_sha256": {
                str(Path(__file__).resolve().relative_to(ROOT)): e14._sha(
                    Path(__file__).resolve()
                ),
                str(Path(e13.__file__).resolve().relative_to(ROOT)): e14._sha(
                    Path(e13.__file__).resolve()
                ),
            },
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    contract = expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        if e14._load(path, "threshold contract") != contract:
            raise ThresholdRefinementError(
                "threshold contract, parent artifacts, or sources changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ThresholdRefinementError(
                "nonempty threshold output has no contract"
            )
        output.mkdir(parents=True, exist_ok=True)
        e14._atomic_json(path, contract)
    return {
        "status": "prepared",
        "new_episode_runs": contract["expected_new_rows"],
        "budgets": list(REFINEMENT_BUDGETS),
        "reference_trace_maximum_explored_nodes": contract[
            "reference_trace_maximum_explored_nodes"
        ],
        "conservative_reference_equivalent_cap": contract[
            "conservative_reference_equivalent_cap"
        ],
        "training": False,
        "contract_sha256": contract["contract_sha256"],
    }


def authenticate(output: Path) -> dict:
    contract = e14._load(output / CONTRACT_NAME, "threshold contract")
    if contract != expected_contract():
        raise ThresholdRefinementError(
            "threshold contract, parent artifacts, or sources changed"
        )
    return contract


def authenticate_frozen(output: Path) -> dict:
    contract = e14._self_hashed(
        output / CONTRACT_NAME,
        "contract_sha256",
        "frozen E14 threshold contract",
    )
    e13_contract, e13_manifest, e14_contract, e14_report = _parent()
    if (
        contract.get("protocol") != PROTOCOL
        or contract.get("refinement_budgets_max_nodes")
        != list(REFINEMENT_BUDGETS)
        or contract.get("parent_e13_contract_sha256")
        != e13_contract["contract_sha256"]
        or contract.get("parent_e13_manifest_sha256")
        != e13_manifest["manifest_sha256"]
        or contract.get("parent_e14_contract_sha256")
        != e14_contract["contract_sha256"]
        or contract.get("parent_e14_report_sha256")
        != e14_report["report_sha256"]
    ):
        raise ThresholdRefinementError("frozen threshold contract binding changed")
    return contract


def _ledger_path(output: Path, budget: int, scenario: str) -> Path:
    return output / "run-ledger" / f"max-nodes-{budget}" / f"{scenario}.json"


def _authenticate_ledger(
    output: Path, contract: Mapping, budget: int, scenario: str
) -> dict:
    ledger = e14._self_hashed(
        _ledger_path(output, budget, scenario),
        "ledger_sha256",
        "threshold run ledger",
    )
    row = ledger.get("row", {})
    if (
        ledger.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("manifest_sha256")
        != contract["parent_e13_manifest_sha256"]
        or row.get("protocol") != PROTOCOL
        or row.get("scenario_id") != scenario
        or int(row.get("instance_seed", -1)) != e14.INSTANCE_SEED
        or int(row.get("search_max_nodes", -1)) != int(budget)
    ):
        raise ThresholdRefinementError("threshold ledger binding changed")
    return ledger


def run_one(output: Path, budget: int, scenario: str) -> dict:
    if budget not in REFINEMENT_BUDGETS or scenario not in e13.MAIN_SCENARIOS:
        raise ThresholdRefinementError("unsupported threshold coordinate")
    contract = authenticate(output)
    path = _ledger_path(output, budget, scenario)
    if path.exists():
        ledger = _authenticate_ledger(output, contract, budget, scenario)
    else:
        e13.run_coordinate(
            e13.DEFAULT_OUTPUT,
            scenario,
            e14.INSTANCE_SEED,
            max_nodes=budget,
            contract_override=contract,
            ledger_path_override=path,
        )
        ledger = _authenticate_ledger(output, contract, budget, scenario)
    row = ledger["row"]
    return {
        "status": "completed" if row["strict_safe_complete"] else "recorded",
        "scenario": scenario,
        "max_nodes": budget,
        "strict_safe_complete": row["strict_safe_complete"],
        "failure_class": row["failure_class"],
        "deliveries": f'{row["deliveries_at_stop"]}/{row["required_deliveries"]}',
        "episode_wall_seconds": row["episode_wall_seconds"],
        "family_proofs": row["relocation_family_proof_count"],
        "family_misses": row["relocation_family_miss_count"],
        "native_searches": row["native_search_count"],
        "ledger": str(path.relative_to(ROOT)),
    }


def _rows(output: Path, contract: Mapping) -> list[dict]:
    rows = []
    for budget in REFINEMENT_BUDGETS:
        for scenario in e13.MAIN_SCENARIOS:
            path = _ledger_path(output, budget, scenario)
            if path.exists():
                row = dict(
                    _authenticate_ledger(output, contract, budget, scenario)[
                        "row"
                    ]
                )
                row["budget_row_source"] = "executed_threshold_refinement"
                rows.append(row)
    return rows


def _matched_initial_summary(budget: int, rows: Sequence[Mapping]) -> dict:
    frontiers = [row["frontiers"][0] for row in rows if row.get("frontiers")]
    physical = sum(int(item["physical_candidate_count"]) for item in frontiers)
    certified = sum(
        int(item["certified_candidate_count"]) for item in frontiers
    )
    return {
        "max_nodes": budget,
        "matched_initial_states": len(frontiers),
        "current_state_safe": sum(
            item["current_recovery_status"] == "SAFE" for item in frontiers
        ),
        "physical_candidates": physical,
        "certified_candidates": certified,
        "certified_to_physical_ratio": (
            certified / physical if physical else None
        ),
        "empty_certified_frontiers": sum(
            bool(item["no_positively_certified_candidate"])
            for item in frontiers
        ),
        "unknown_candidates": sum(
            int(item["unknown_candidate_count"]) for item in frontiers
        ),
    }


def _all_rows(output: Path, contract: Mapping) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    parent_contract = e14.authenticate_frozen(e14.DEFAULT_OUTPUT)
    for row in e14._rows(e14.DEFAULT_OUTPUT, parent_contract):
        grouped[int(row["search_max_nodes"])].append(row)
    for row in _rows(output, contract):
        grouped[int(row["search_max_nodes"])].append(row)
    return grouped


def _table(summaries: Sequence[Mapping], matched: Sequence[Mapping]) -> str:
    matched_by_budget = {item["max_nodes"]: item for item in matched}
    lines = [
        "# E14 budget-threshold refinement",
        "",
        "The 32- and 64-node arms are post-hoc refinements of the completed predeclared sweep.",
        "",
        "| Max nodes | Strict | Deliveries | Initial anchors SAFE | Initial certified/physical | Initial empty | D12 proofs | Native searches | Warm p95 (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        matched_item = matched_by_budget[item["max_nodes"]]
        latency = item["latency"]["warm_end_to_end_decision_seconds"]["p95"]
        coverage = matched_item["certified_to_physical_ratio"]
        lines.append(
            "| {budget} | {strict}/9 | {delivered}/{required} | {safe}/9 | "
            "{coverage} | {empty} | {proofs} | {native} | {latency} |".format(
                budget=item["max_nodes"],
                strict=item["strict_safe_complete"],
                delivered=item["deliveries_at_stop"],
                required=item["required_deliveries"],
                safe=matched_item["current_state_safe"],
                coverage="—" if coverage is None else f"{coverage:.3f}",
                empty=matched_item["empty_certified_frontiers"],
                proofs=item["family"]["proofs"],
                native=item["native_searches"]["count"],
                latency="—" if latency is None else f"{latency:.2f}",
            )
        )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, allow_partial: bool = False) -> dict:
    contract = authenticate_frozen(output)
    grouped = _all_rows(output, contract)
    observed_new = sum(len(grouped.get(budget, ())) for budget in REFINEMENT_BUDGETS)
    if observed_new != contract["expected_new_rows"] and not allow_partial:
        raise ThresholdRefinementError(
            f"expected {contract['expected_new_rows']} new rows, found {observed_new}"
        )
    budgets = [
        budget for budget in ALL_OBSERVED_BUDGETS if grouped.get(budget)
    ]
    summaries = [
        e14._budget_summary(budget, grouped[budget]) for budget in budgets
    ]
    matched = [
        _matched_initial_summary(budget, grouped[budget]) for budget in budgets
    ]
    report = e14._with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "expected_new_rows": contract["expected_new_rows"],
            "observed_new_rows": observed_new,
            "partial": observed_new != contract["expected_new_rows"],
            "post_hoc_threshold_refinement": True,
            "summaries": summaries,
            "matched_initial_frontiers": matched,
            "reference_trace_maximum_explored_nodes": contract[
                "reference_trace_maximum_explored_nodes"
            ],
            "conservative_reference_equivalent_cap": contract[
                "conservative_reference_equivalent_cap"
            ],
            "aggregate_trajectory_coverage_not_used_for_cross_budget_claim": True,
        },
        "report_sha256",
    )
    e14._atomic_json(output / REPORT_NAME, report)
    e14._atomic_text(output / TABLE_NAME, _table(summaries, matched))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "authenticate", "run-one", "analyze")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-nodes", type=int, choices=REFINEMENT_BUDGETS)
    parser.add_argument("--scenario", choices=e13.MAIN_SCENARIOS)
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
        if args.max_nodes is None or args.scenario is None:
            raise ThresholdRefinementError(
                "run-one requires --max-nodes and --scenario"
            )
        prepare(output)
        result = run_one(output, args.max_nodes, args.scenario)
    else:
        report = analyze(output, allow_partial=args.allow_partial)
        result = {
            "status": "analyzed",
            "observed_new_rows": report["observed_new_rows"],
            "expected_new_rows": report["expected_new_rows"],
            "partial": report["partial"],
            "report": str((output / REPORT_NAME).relative_to(ROOT)),
            "table": str((output / TABLE_NAME).relative_to(ROOT)),
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
