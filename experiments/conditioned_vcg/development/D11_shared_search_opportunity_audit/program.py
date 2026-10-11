#!/usr/bin/env python3
"""Frozen contract and deterministic sampling for the D11 reuse audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.conditioned_vcg.E14_certification_scalability_95k import capture
from experiments.conditioned_vcg.E14_certification_scalability_95k import program as e14
from experiments.conditioned_vcg.E14_certification_scalability_95k import replay


PROTOCOL = "vcg_d11_shared_search_opportunity_95k_v1"
SCHEMA_VERSION = 1
SCENARIO_ID = "size_10x10_occ_medium"
MAX_ROOT_QUERIES = 1_100
MAX_SEARCH_WALL_SECONDS = 900.0
DISTRIBUTED_QUERIES_PER_NONANCHOR_FRONTIER = 5
E14_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e14-certificate-reuse-95k"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-d11-shared-search-opportunity-95k"
CONTRACT_NAME = "d11-contract.json"
LEDGER_NAME = "intermediate-state-audit.jsonl"
REPORT_NAME = "d11-report.json"


class D11Error(RuntimeError):
    pass


def _source_hashes() -> dict:
    relative = (
        "experiments/conditioned_vcg/development/"
        "D11_shared_search_opportunity_audit/program.py",
        "experiments/conditioned_vcg/development/"
        "D11_shared_search_opportunity_audit/audit.py",
    )
    return {name: e14.sha256(PROJECT_ROOT / name) for name in relative}


def _balanced_subset(queries: Sequence[Mapping], limit: int) -> list[Mapping]:
    selected: list[Mapping] = []
    selected_indices = set()
    for role in ("current_state", "accept", "deliver", "reconfigure"):
        matches = [query for query in queries if query["check_role"] == role]
        if matches:
            query = matches[len(matches) // 2]
            selected.append(query)
            selected_indices.add(int(query["query_index"]))
    for query in queries:
        index = int(query["query_index"])
        if index not in selected_indices:
            selected.append(query)
            selected_indices.add(index)
        if len(selected) >= limit:
            break
    return sorted(selected[:limit], key=lambda item: int(item["query_index"]))


def select_queries(trace: Mapping) -> dict:
    completed = {
        int(item["frontier_index"]) for item in trace["completed_frontiers"]
    }
    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for query in trace["queries"]:
        frontier = int(query["frontier_index"])
        if frontier in completed and query["result"] is not None:
            grouped[frontier].append(query)
    if len(grouped) < 3:
        raise D11Error("D11 requires at least three completed captured frontiers")
    ranked = sorted(grouped, key=lambda value: (len(grouped[value]), value))
    anchors = {
        "smallest": ranked[0],
        "median": ranked[(len(ranked) - 1) // 2],
        "widest": ranked[-1],
    }
    anchor_ids = set(anchors.values())
    selected = []
    for frontier in sorted(grouped):
        queries = grouped[frontier]
        selected.extend(
            queries
            if frontier in anchor_ids
            else _balanced_subset(
                queries, DISTRIBUTED_QUERIES_PER_NONANCHOR_FRONTIER
            )
        )
    selected = sorted(
        {int(item["query_index"]): item for item in selected}.values(),
        key=lambda item: int(item["query_index"]),
    )
    if len(selected) > MAX_ROOT_QUERIES:
        raise D11Error(
            f"deterministic sample has {len(selected)} roots, above "
            f"the frozen cap of {MAX_ROOT_QUERIES}"
        )
    return {
        "query_indices": [int(item["query_index"]) for item in selected],
        "root_query_count": len(selected),
        "completed_frontier_count": len(grouped),
        "anchor_frontiers": {
            name: {
                "frontier_index": frontier,
                "complete_root_query_count": len(grouped[frontier]),
            }
            for name, frontier in anchors.items()
        },
        "nonanchor_sampling": (
            "up_to_five_queries_per_frontier_role_stratified_then_canonical"
        ),
    }


def parent_inputs() -> tuple[dict, dict, dict, dict]:
    contract, _manifest = e14.authenticate(E14_OUTPUT)
    capture_ledger, trace = capture.authenticate_capture(
        E14_OUTPUT, contract, SCENARIO_ID
    )
    path_records = replay._load_records(
        e14.path_replay_path(E14_OUTPUT, SCENARIO_ID), contract, SCENARIO_ID
    )
    sample = select_queries(trace)
    queries = {int(item["query_index"]): item for item in trace["queries"]}
    missing = [
        index for index in sample["query_indices"]
        if replay._query_key(queries[index]) not in path_records
    ]
    if missing:
        raise D11Error("E14 path replay does not cover the frozen D11 sample")
    return contract, capture_ledger, trace, sample


def expected_contract() -> dict:
    parent, capture_ledger, trace, sample = parent_inputs()
    path = e14.path_replay_path(E14_OUTPUT, SCENARIO_ID)
    return e14.with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "question": "how_much_path_cleaned_search_work_repeats_at_canonical_intermediate_states",
        "training": False,
        "scenario_id": SCENARIO_ID,
        "parent_protocol": parent["protocol"],
        "parent_contract_sha256": parent["contract_sha256"],
        "parent_capture_ledger_sha256": capture_ledger["ledger_sha256"],
        "parent_trace_sha256": trace["trace_sha256"],
        "parent_path_replay_file_sha256": e14.sha256(path),
        "source_sha256": _source_hashes(),
        "sample": sample,
        "bounds": {
            "maximum_root_queries": MAX_ROOT_QUERIES,
            "maximum_accumulated_instrumented_search_wall_seconds": (
                MAX_SEARCH_WALL_SECONDS
            ),
            "censoring": "partial_measurement_not_failure_not_infeasibility",
        },
        "instrumented_solver": "E14_path_enumeration_cleanup_exact_search",
        "canonical_state": (
            "fixed_scenario_and_contract_scoped_solver_configuration_key;"
            "block_clocks_excluded"
        ),
        "reuse_scopes": [
            "within_one_root_search",
            "across_candidates_within_one_frontier",
            "across_decision_frontiers",
        ],
        "reuse_semantics": {
            "measured_object": "exact_action_and_deterministic_successor_transition_construction",
            "repeated_state_is_not_a_safe_certificate": True,
            "positive_proof_requires_constructive_witness": True,
            "unsafe_proof_requires_exhaustive_search": True,
            "unknown_is_never_a_proof": True,
        },
    }, "contract_sha256")


def prepare(output: Path) -> dict:
    contract = expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        observed = e14.load_json(path, label="D11 contract")
        if observed != contract:
            raise D11Error("D11 contract, parent artifacts, or sources changed")
    else:
        e14.atomic_json(path, contract)
    return contract


def authenticate(output: Path) -> tuple[dict, dict, dict]:
    contract = e14.load_json(output / CONTRACT_NAME, label="D11 contract")
    if contract != expected_contract():
        raise D11Error("D11 contract, parent artifacts, or sources changed")
    parent, _capture_ledger, trace, sample = parent_inputs()
    if sample != contract["sample"]:
        raise D11Error("D11 deterministic sample changed")
    return contract, trace, parent


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "authenticate", "sample"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "authenticate":
        result = authenticate(output)[0]
    else:
        _parent, _capture, trace, sample = parent_inputs()
        result = sample
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
