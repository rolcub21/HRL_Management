#!/usr/bin/env python3
"""Replay E14 certificate queries through exact, certificate-preserving reuse stages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import PSLAP.viability as viability
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_dataset import recovery_state_from_dict
from experiments.conditioned_vcg.E14_certification_scalability_95k import capture
from experiments.conditioned_vcg.E14_certification_scalability_95k import program
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse


def _query_key(query: Mapping) -> str:
    return reuse.digest({
        "physical_state_id": query["physical_state_id"],
        "search": query["search"],
    })


def _self_hashed(value: Mapping) -> dict:
    return program.with_hash(value, "record_sha256")


def _validate_record(record: Mapping, contract: Mapping, scenario_id: str) -> None:
    if record.get("record_sha256") != program.digest(
        record, hash_field="record_sha256"
    ):
        raise program.E14Error("path-replay record self-hash mismatch")
    if record.get("contract_sha256") != contract["contract_sha256"]:
        raise program.E14Error("path-replay record contract mismatch")
    if record.get("scenario_id") != scenario_id:
        raise program.E14Error("path-replay scenario mismatch")


def _load_records(
    path: Path, contract: Mapping, scenario_id: str
) -> dict[str, dict]:
    if not path.exists():
        return {}
    if not path.is_file() or path.is_symlink():
        raise program.E14Error(f"invalid path-replay ledger: {path}")
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise program.E14Error(
                    f"invalid path-replay JSON at line {line_number}"
                ) from error
            _validate_record(record, contract, scenario_id)
            key = record["query_key"]
            previous = records.get(key)
            if previous is not None and previous != record:
                raise program.E14Error("conflicting duplicate path-replay result")
            records[key] = record
    return records


def _append_record(path: Path, record: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _ordered_unique_queries(trace: Mapping) -> list[Mapping]:
    seen = set()
    result = []
    for query in trace["queries"]:
        key = _query_key(query)
        if key not in seen:
            seen.add(key)
            result.append(query)
    return result


def _matching_baseline_results(trace: Mapping, query_key: str) -> list[dict]:
    return [
        query["result"]
        for query in trace["queries"]
        if _query_key(query) == query_key and query["result"] is not None
    ]


def _run_path_query(
    contract: Mapping,
    scenario_id: str,
    trace: Mapping,
    query: Mapping,
) -> dict:
    query_key = _query_key(query)
    state = recovery_state_from_dict(
        trace["physical_states"][query["physical_state_id"]]
    )
    full_state = recovery_state_from_dict(
        trace["full_states"][query["full_state_id"]]
    )
    config = reuse.config_from_dict(query["search"])

    abstraction_started = perf_counter()
    abstraction = reuse.validate_timing_abstraction(full_state)
    abstraction_seconds = perf_counter() - abstraction_started

    started = perf_counter()
    with reuse.path_cleanup_active():
        certificate = viability.analyze_recoverability(
            state,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            max_primitive_steps=config.max_primitive_steps,
            search_order=config.search_order,
        )
    search_seconds = perf_counter() - started
    result = reuse.certificate_to_dict(certificate)
    baselines = _matching_baseline_results(trace, query_key)
    baseline_digests = {reuse.digest(item) for item in baselines}
    baseline_consistent = len(baseline_digests) <= 1
    exact_match = bool(baselines) and baseline_consistent and all(
        item == result for item in baselines
    )
    if not baseline_consistent:
        raise program.E14Error(
            "clock-equivalent baseline searches produced different certificates"
        )
    if baselines and not exact_match:
        raise program.E14Error(
            "path/enumeration cleanup changed an exact certificate or witness"
        )
    if certificate.status is ViabilityStatus.SAFE:
        reuse.replay_witness(state, certificate)

    return _self_hashed({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "artifact": "path_enumeration_replay_record",
        "contract_sha256": contract["contract_sha256"],
        "scenario_id": scenario_id,
        "query_key": query_key,
        "physical_state_id": query["physical_state_id"],
        "representative_query_index": query["query_index"],
        "search": query["search"],
        "certificate": result,
        "search_seconds": search_seconds,
        "timing_abstraction_validation_seconds": abstraction_seconds,
        "timing_abstraction": abstraction,
        "baseline_completed_result_count": len(baselines),
        "baseline_results_consistent": baseline_consistent,
        "exact_certificate_and_witness_match": exact_match if baselines else None,
        "positive_witness_replayed": certificate.status is ViabilityStatus.SAFE,
    })


def replay_one(
    output: Path,
    contract: Mapping,
    scenario_id: str,
    *,
    chunk_seconds: int,
) -> dict:
    _ledger, trace = capture.authenticate_capture(output, contract, scenario_id)
    path = program.path_replay_path(output, scenario_id)
    records = _load_records(path, contract, scenario_id)
    unique_queries = _ordered_unique_queries(trace)
    started = perf_counter()
    completed_now = 0
    for query in unique_queries:
        key = _query_key(query)
        if key in records:
            continue
        if completed_now and perf_counter() - started >= chunk_seconds:
            break
        record = _run_path_query(contract, scenario_id, trace, query)
        _append_record(path, record)
        records[key] = record
        completed_now += 1
        if completed_now % 25 == 0:
            print(
                f"E14 path replay {scenario_id} | "
                f"{len(records)}/{len(unique_queries)} unique queries",
                flush=True,
            )
    complete = len(records) == len(unique_queries)
    return {
        "scenario_id": scenario_id,
        "status": "complete" if complete else "chunk_complete_resume_required",
        "completed_this_invocation": completed_now,
        "completed_unique_queries": len(records),
        "required_unique_queries": len(unique_queries),
        "remaining_unique_queries": len(unique_queries) - len(records),
        "wall_seconds_this_invocation": perf_counter() - started,
    }


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time_summary(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "total_seconds": sum(data),
        "mean_seconds": statistics.fmean(data) if data else None,
        "median_seconds": statistics.median(data) if data else None,
        "p95_seconds": _percentile(data, 0.95),
        "max_seconds": max(data) if data else None,
    }


def _stage_current(trace: Mapping) -> dict:
    queries = trace["queries"]
    misses = [query for query in queries if not query["original_cache_hit"]]
    completed = [
        query for query in misses if query["original_search_seconds"] is not None
    ]
    return {
        "stage": reuse.STAGE_CURRENT,
        "cold_unique_full_state_checks": len(misses),
        "accumulated": {
            "queries": len(queries),
            "outcome_cache_hits": sum(query["original_cache_hit"] for query in queries),
            "searches_started": len(misses),
            "searches_completed": len(completed),
            "unresolved_searches": len(misses) - len(completed),
            "search_time": _time_summary(
                query["original_search_seconds"] for query in completed
            ),
        },
        "status_counts": dict(Counter(
            query["result"]["status"]
            for query in queries if query["result"] is not None
        )),
    }


def _stage_timing_key(trace: Mapping) -> dict:
    seen = set()
    hits = 0
    searches = []
    unresolved = 0
    conflicting = []
    grouped_results: dict[str, set[str]] = defaultdict(set)
    for query in trace["queries"]:
        key = _query_key(query)
        if query["result"] is not None:
            grouped_results[key].add(reuse.digest(query["result"]))
        if key in seen:
            hits += 1
            continue
        seen.add(key)
        seconds = query["original_search_seconds"]
        if seconds is None:
            unresolved += 1
        else:
            searches.append(float(seconds))
    conflicting = sorted(key for key, values in grouped_results.items() if len(values) > 1)
    if conflicting:
        raise program.E14Error(
            "timing projection merged states with different exact certificates"
        )
    return {
        "stage": reuse.STAGE_TIMING_KEY,
        "cold_unique_physical_state_checks": len(seen),
        "accumulated": {
            "queries": len(trace["queries"]),
            "timing_key_hits": hits,
            "searches_completed": len(searches),
            "unresolved_searches": unresolved,
            "search_time_estimate_from_observed_baseline": _time_summary(searches),
        },
        "clock_equivalence": {
            "physical_keys_with_completed_results": len(grouped_results),
            "conflicting_certificate_keys": 0,
        },
    }


def _path_records_for_trace(
    output: Path, contract: Mapping, scenario_id: str, trace: Mapping
) -> dict[str, dict]:
    records = _load_records(
        program.path_replay_path(output, scenario_id), contract, scenario_id
    )
    required = {_query_key(query) for query in _ordered_unique_queries(trace)}
    if set(records) != required:
        raise program.E14Error(
            f"path replay for {scenario_id} is incomplete; rerun its replay command"
        )
    return records


def _stage_path(trace: Mapping, records: Mapping[str, Mapping]) -> dict:
    seconds = [record["search_seconds"] for record in records.values()]
    abstraction_seconds = [
        record["timing_abstraction_validation_seconds"]
        for record in records.values()
    ]
    comparable = [
        record for record in records.values()
        if record["exact_certificate_and_witness_match"] is not None
    ]
    return {
        "stage": reuse.STAGE_PATH_CLEANUP,
        "cold": {
            "unique_physical_state_checks": len(records),
            "search_time": _time_summary(seconds),
            "abstraction_validation_time": _time_summary(abstraction_seconds),
        },
        "accumulated": {
            "queries": len(trace["queries"]),
            "timing_key_hits": len(trace["queries"]) - len(records),
            "searches": len(records),
            "search_time": _time_summary(seconds),
        },
        "equivalence": {
            "timing_homomorphism_checks": len(records),
            "exact_baseline_comparisons": len(comparable),
            "certificate_or_witness_mismatches": sum(
                not record["exact_certificate_and_witness_match"]
                for record in comparable
            ),
            "positive_witnesses_replayed": sum(
                record["positive_witness_replayed"] for record in records.values()
            ),
        },
    }


def _stage_suffix(trace: Mapping, records: Mapping[str, Mapping]) -> dict:
    outcomes = {}
    store = reuse.PositiveProofStore()
    outcome_hits = 0
    proof_hits = 0
    searches = 0
    unresolved = 0
    search_seconds = []
    status_counts = Counter()
    started = perf_counter()
    for query in trace["queries"]:
        key = _query_key(query)
        state = recovery_state_from_dict(
            trace["physical_states"][query["physical_state_id"]]
        )
        config = reuse.config_from_dict(query["search"])
        outcome = outcomes.get(key)
        certificate = None
        if outcome is not None and outcome.status is not ViabilityStatus.UNKNOWN:
            outcome_hits += 1
            certificate = outcome
        else:
            certificate = store.lookup(state, config)
            if certificate is not None:
                proof_hits += 1
            elif outcome is not None:
                outcome_hits += 1
                certificate = outcome
        if certificate is None:
            record = records.get(key)
            if record is None:
                unresolved += 1
                continue
            certificate = reuse.certificate_from_dict(record["certificate"])
            outcomes[key] = certificate
            searches += 1
            search_seconds.append(float(record["search_seconds"]))
            if certificate.status is ViabilityStatus.SAFE:
                store.insert_certificate(state, certificate)
        status_counts[certificate.status.value] += 1
    overhead_seconds = perf_counter() - started
    if unresolved:
        raise program.E14Error("witness-suffix replay did not cover every query")
    return {
        "stage": reuse.STAGE_WITNESS_SUFFIX,
        "cold_independent_checks": {
            "note": "without accumulated positive proofs this equals path_enumeration_cleanup",
            "unique_physical_state_checks": len(records),
        },
        "accumulated": {
            "queries": len(trace["queries"]),
            "outcome_cache_hits": outcome_hits,
            "constructive_proof_hits": proof_hits,
            "searches": searches,
            "searches_avoided_by_any_reuse": len(trace["queries"]) - searches,
            "search_time_estimate": _time_summary(search_seconds),
            "proof_lookup_insertion_and_validation_seconds": overhead_seconds,
            "status_counts": dict(status_counts),
        },
        "proof_store": {
            "stored_physical_states": len(store),
            "insertions": store.insertions,
            "shorter_witness_replacements": store.replacements,
            "lookups": store.lookups,
            "hits": store.hits,
            "unknown_insertions": 0,
            "witness_semantics": "validated_completion_upper_bound_not_shortest_rank",
        },
        "soundness": {
            "all_inserted_positive_witnesses_replayed_before_storage": True,
            "stored_suffixes_derive_from_validated_witnesses": True,
            "unknown_used_as_positive_proof": False,
            "unsafe_used_as_cross_budget_proof": False,
        },
    }


def analyze_one(
    output: Path, contract: Mapping, scenario_id: str
) -> dict:
    ledger, trace = capture.authenticate_capture(output, contract, scenario_id)
    records = _path_records_for_trace(output, contract, scenario_id, trace)
    stages = [
        _stage_current(trace),
        _stage_timing_key(trace),
        _stage_path(trace, records),
        _stage_suffix(trace, records),
    ]
    return {
        "scenario_id": scenario_id,
        "capture_outcome": ledger["capture_outcome"],
        "query_workload": {
            "queries": trace["query_count"],
            "unique_full_states": trace["unique_full_states"],
            "unique_physical_states": trace["unique_physical_states"],
            "completed_frontiers": trace["completed_frontier_count"],
            "unresolved_original_queries": len(trace["unresolved_query_indices"]),
        },
        "stages": stages,
    }


def analyze(output: Path, scenario_ids: Sequence[str]) -> dict:
    contract, _manifest = program.authenticate(output)
    scenarios = [analyze_one(output, contract, value) for value in scenario_ids]
    report = program.with_hash({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "artifact": "certificate_reuse_report",
        "contract_sha256": contract["contract_sha256"],
        "hardware": program.hardware(),
        "training": False,
        "scenarios": scenarios,
        "interpretation_scope": {
            "timing_key": "exact_only_if_recorded_homomorphism_and_certificate_checks_pass",
            "path_cleanup": "must_preserve_exact_certificate_and_witness",
            "suffix_store": "may_change_witness_and_resolve_unknown_only_via_validated_positive_proof",
            "shared_search_dag": "conditional_on_remaining_measured_cost",
        },
    }, "report_sha256")
    program.atomic_json(output / program.REPORT_NAME, report)
    return report


def _scenario_command(command: str) -> tuple[str, ...]:
    if command.endswith("-medium"):
        return (program.SCENARIO_IDS[0],)
    if command.endswith("-high"):
        return (program.SCENARIO_IDS[1],)
    return program.SCENARIO_IDS


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "replay-medium",
            "replay-high",
            "replay-both",
            "analyze-medium",
            "analyze-high",
            "analyze-both",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-seconds", type=int, default=program.REPLAY_CHUNK_SECONDS
    )
    args = parser.parse_args(argv)
    if args.chunk_seconds <= 0:
        raise program.E14Error("chunk-seconds must be positive")
    output = args.output_dir.resolve()
    program.prepare(output)
    scenarios = _scenario_command(args.command)
    if args.command.startswith("replay-"):
        contract, _manifest = program.authenticate(output)
        result = {
            "rows": [
                replay_one(
                    output,
                    contract,
                    scenario_id,
                    chunk_seconds=args.chunk_seconds,
                )
                for scenario_id in scenarios
            ]
        }
    else:
        result = analyze(output, scenarios)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
