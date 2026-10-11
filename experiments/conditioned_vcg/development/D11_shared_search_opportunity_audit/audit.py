#!/usr/bin/env python3
"""Measure repeated intermediate-state work in path-cleaned exact searches."""

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
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import PSLAP.viability as viability
from PSLAP.viability_dataset import recovery_state_from_dict
from experiments.conditioned_vcg.E14_certification_scalability_95k import program as e14
from experiments.conditioned_vcg.E14_certification_scalability_95k import replay
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D11_shared_search_opportunity_audit import program


def _canonical_state_key(state) -> tuple:
    # The exact solver's own configuration key already excludes block clocks.
    # D11 is fixed to one geometry/contract, so reusing that tuple avoids an
    # artificial full-state serialization cost in the runtime estimate.
    return viability._configuration_key(state)


def _canonical_state_id(key: tuple) -> str:
    return reuse.digest(key)


class ExpansionRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._active_state = None
        self._active_event: Optional[dict] = None
        self._original_apply = viability._apply_legal_action

    def legal_actions(self, state):
        key_started = perf_counter()
        key = _canonical_state_key(state)
        key_seconds = perf_counter() - key_started
        state_id = _canonical_state_id(key)

        enumeration_started = perf_counter()
        actions = reuse.legal_recovery_actions_optimized(state)
        enumeration_seconds = perf_counter() - enumeration_started
        action_payload = [reuse.action_to_dict(action) for action in actions]
        event = {
            "expansion_index": len(self.events),
            "canonical_state_id": state_id,
            "block_count": len(state.blocks),
            "action_count": len(actions),
            "cache_key_construction_seconds": key_seconds,
            "action_enumeration_seconds": enumeration_seconds,
            "successor_application_seconds": 0.0,
            "successor_calls": 0,
            "action_signature": reuse.digest(action_payload),
            "action_payload_bytes": len(reuse.canonical_bytes(action_payload)),
            "estimated_successor_identifier_bytes": 32 * len(actions),
            "on_returned_witness": False,
        }
        self.events.append(event)
        self._active_state = state
        self._active_event = event
        return actions

    def apply(self, state, action):
        if self._active_event is None or state != self._active_state:
            raise program.D11Error(
                "successor construction was not associated with the active expansion"
            )
        started = perf_counter()
        successor = self._original_apply(state, action)
        elapsed = perf_counter() - started
        self._active_event["successor_application_seconds"] += elapsed
        self._active_event["successor_calls"] += 1
        return successor

    def mark_witness(self, root_state, certificate) -> list[str]:
        current = reuse.timing_erased_state(root_state)
        witness_ids = []
        for action in certificate.witness:
            state_id = _canonical_state_id(_canonical_state_key(current))
            witness_ids.append(state_id)
            current = self._original_apply(current, action)
        members = set(witness_ids)
        for event in self.events:
            event["on_returned_witness"] = event["canonical_state_id"] in members
        return witness_ids


def instrumented_search(state, config) -> tuple[object, list[dict], list[str], float]:
    recorder = ExpansionRecorder()
    started = perf_counter()
    with patch.object(
        viability, "legal_recovery_actions", recorder.legal_actions
    ), patch.object(viability, "_apply_legal_action", recorder.apply):
        certificate = viability.analyze_recoverability(
            state,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            max_primitive_steps=config.max_primitive_steps,
            search_order=config.search_order,
        )
    wall_seconds = perf_counter() - started
    witness_ids = recorder.mark_witness(state, certificate)
    return certificate, recorder.events, witness_ids, wall_seconds


def _record_hash(value: Mapping) -> dict:
    return e14.with_hash(value, "record_sha256")


def _validate_record(record: Mapping, contract: Mapping) -> None:
    if record.get("record_sha256") != e14.digest(
        record, hash_field="record_sha256"
    ):
        raise program.D11Error("D11 audit record self-hash mismatch")
    if record.get("contract_sha256") != contract["contract_sha256"]:
        raise program.D11Error("D11 audit record contract mismatch")


def _load_records(path: Path, contract: Mapping) -> dict[int, dict]:
    if not path.exists():
        return {}
    if not path.is_file() or path.is_symlink():
        raise program.D11Error("D11 audit ledger is not a regular file")
    records = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise program.D11Error(
                    f"invalid D11 JSONL record at line {line_number}"
                ) from error
            _validate_record(record, contract)
            index = int(record["query_index"])
            previous = records.get(index)
            if previous is not None and previous != record:
                raise program.D11Error("conflicting duplicate D11 root record")
            records[index] = record
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


def _run_root(
    contract: Mapping,
    trace: Mapping,
    query: Mapping,
    path_record: Mapping,
) -> dict:
    state = recovery_state_from_dict(
        trace["physical_states"][query["physical_state_id"]]
    )
    config = reuse.config_from_dict(query["search"])
    certificate, events, witness_ids, wall_seconds = instrumented_search(
        state, config
    )
    observed = reuse.certificate_to_dict(certificate)
    exact_match = observed == path_record["certificate"]
    if not exact_match:
        raise program.D11Error(
            "D11 instrumentation changed an E14 certificate or witness"
        )
    return _record_hash({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "artifact": "instrumented_intermediate_state_root",
        "contract_sha256": contract["contract_sha256"],
        "query_index": int(query["query_index"]),
        "frontier_index": int(query["frontier_index"]),
        "decision_epoch": int(query["decision_epoch"]),
        "check_role": query["check_role"],
        "query_key": replay._query_key(query),
        "physical_state_id": query["physical_state_id"],
        "reference_path_search_seconds": float(path_record["search_seconds"]),
        "instrumented_search_wall_seconds": wall_seconds,
        "certificate_status": certificate.status.value,
        "certificate_and_witness_exact_match": exact_match,
        "expanded_state_count": len(events),
        "witness_state_ids": witness_ids,
        "events": events,
    })


def run(output: Path) -> dict:
    contract, trace, parent = program.authenticate(output)
    parent_records = replay._load_records(
        e14.path_replay_path(program.E14_OUTPUT, program.SCENARIO_ID),
        parent,
        program.SCENARIO_ID,
    )
    queries = {int(item["query_index"]): item for item in trace["queries"]}
    path = output / program.LEDGER_NAME
    records = _load_records(path, contract)
    accumulated = sum(
        float(item["instrumented_search_wall_seconds"])
        for item in records.values()
    )
    completed_now = 0
    for index in contract["sample"]["query_indices"]:
        if index in records:
            continue
        if accumulated >= program.MAX_SEARCH_WALL_SECONDS:
            break
        query = queries[index]
        parent_record = parent_records[replay._query_key(query)]
        record = _run_root(contract, trace, query, parent_record)
        _append_record(path, record)
        records[index] = record
        accumulated += float(record["instrumented_search_wall_seconds"])
        completed_now += 1
        if completed_now % 25 == 0:
            print(
                f"D11 {len(records)}/{contract['sample']['root_query_count']} roots "
                f"| search wall={accumulated:.1f}/{program.MAX_SEARCH_WALL_SECONDS:.0f}s",
                flush=True,
            )
    required = int(contract["sample"]["root_query_count"])
    status = (
        "complete" if len(records) == required
        else "censored_accumulated_search_wall_budget"
    )
    return {
        "status": status,
        "completed_this_invocation": completed_now,
        "completed_roots": len(records),
        "required_roots": required,
        "accumulated_instrumented_search_wall_seconds": accumulated,
        "censored_is_not_failure_or_infeasibility": status != "complete",
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


def _summary(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    return {
        "n": len(data),
        "total": sum(data),
        "mean": statistics.fmean(data) if data else None,
        "median": statistics.median(data) if data else None,
        "p95": _percentile(data, 0.95),
        "maximum": max(data) if data else None,
    }


def _lookup_pass_seconds(state_ids: Sequence[str]) -> dict:
    if not state_ids:
        return {"rounds": 0, "seconds_per_ordered_pass": 0.0}
    rounds = max(1, min(1_000, math.ceil(1_000_000 / len(state_ids))))
    started = perf_counter()
    for _ in range(rounds):
        cache = {}
        for state_id in state_ids:
            if state_id not in cache:
                cache[state_id] = True
    elapsed = perf_counter() - started
    return {
        "rounds": rounds,
        "operations_per_round": len(state_ids),
        "seconds_per_ordered_pass": elapsed / rounds,
        "note": "precomputed_identifier_dictionary_lower_bound",
    }


def overlap_analysis(records: Sequence[Mapping]) -> dict:
    ordered = sorted(records, key=lambda item: int(item["query_index"]))
    first_seen = {}
    signatures = {}
    unique_payload = {}
    scope_counts = Counter()
    scope_states = defaultdict(set)
    scope_enumeration_seconds = Counter()
    scope_application_seconds = Counter()
    role_counts = Counter()
    role_seconds = Counter()
    all_ids = []
    total_enumeration = 0.0
    total_application = 0.0
    total_key = 0.0
    repeated_root_occurrences = 0
    repeated_witness_occurrences = 0
    event_count = 0

    for root in ordered:
        query_index = int(root["query_index"])
        frontier_index = int(root["frontier_index"])
        local_seen = set()
        for event in root["events"]:
            event_count += 1
            state_id = event["canonical_state_id"]
            all_ids.append(state_id)
            enumeration = float(event["action_enumeration_seconds"])
            application = float(event["successor_application_seconds"])
            key_seconds = float(event["cache_key_construction_seconds"])
            local = enumeration + application
            total_enumeration += enumeration
            total_application += application
            total_key += key_seconds
            signature = event["action_signature"]
            previous_signature = signatures.setdefault(state_id, signature)
            if previous_signature != signature:
                raise program.D11Error(
                    "one canonical state produced conflicting action transitions"
                )
            unique_payload.setdefault(state_id, (
                int(event["action_payload_bytes"]),
                int(event["estimated_successor_identifier_bytes"]),
            ))

            if state_id in local_seen:
                scope = "within_one_root_search"
            elif state_id in first_seen:
                first = first_seen[state_id]
                scope = (
                    "across_candidates_within_one_frontier"
                    if first["frontier_index"] == frontier_index
                    else "across_decision_frontiers"
                )
            else:
                scope = "first_observation"
                first_seen[state_id] = {
                    "query_index": query_index,
                    "frontier_index": frontier_index,
                }
            local_seen.add(state_id)
            scope_counts[scope] += 1
            scope_states[scope].add(state_id)
            if scope != "first_observation":
                scope_enumeration_seconds[scope] += enumeration
                scope_application_seconds[scope] += application
                role_counts[root["check_role"]] += 1
                role_seconds[root["check_role"]] += local
                repeated_root_occurrences += int(event["expansion_index"] == 0)
                repeated_witness_occurrences += int(event["on_returned_witness"])

    lookup = _lookup_pass_seconds(all_ids)
    gross_enumeration = sum(scope_enumeration_seconds.values())
    gross_application = sum(scope_application_seconds.values())
    gross_transition = gross_enumeration + gross_application
    lookup_cost = float(lookup["seconds_per_ordered_pass"])
    conservative_key_cost = total_key
    net_action_optimistic = max(0.0, gross_enumeration - lookup_cost)
    net_action_conservative = max(
        0.0, gross_enumeration - lookup_cost - conservative_key_cost
    )
    net_transition_optimistic = max(0.0, gross_transition - lookup_cost)
    reference_seconds = sum(
        float(record["reference_path_search_seconds"]) for record in ordered
    )
    action_bytes = sum(value[0] for value in unique_payload.values())
    edge_identifier_bytes = sum(value[1] for value in unique_payload.values())
    action_cache_bytes = 32 * len(unique_payload) + action_bytes
    memory_bytes = 32 * len(unique_payload) + action_bytes + edge_identifier_bytes
    conservative_fraction = (
        net_action_conservative / reference_seconds if reference_seconds else 0.0
    )
    memory_acceptable = action_cache_bytes <= 512 * 1024 * 1024
    advance = conservative_fraction >= 0.20 and memory_acceptable

    return {
        "sample": {
            "root_queries": len(ordered),
            "frontiers_represented": len({int(item["frontier_index"]) for item in ordered}),
            "expanded_state_occurrences": event_count,
            "unique_canonical_states": len(first_seen),
            "repeated_occurrences": event_count - len(first_seen),
            "repeated_occurrence_rate": (
                (event_count - len(first_seen)) / event_count if event_count else 0.0
            ),
            "reference_path_search_seconds": reference_seconds,
            "instrumented_search_wall_seconds": sum(
                float(record["instrumented_search_wall_seconds"])
                for record in ordered
            ),
            "exact_certificate_and_witness_mismatches": sum(
                not record["certificate_and_witness_exact_match"]
                for record in ordered
            ),
        },
        "recorded_local_work": {
            "action_enumeration_seconds": total_enumeration,
            "successor_application_seconds": total_application,
            "cache_key_construction_seconds": total_key,
        },
        "overlap_by_scope": {
            scope: {
                "occurrences": scope_counts[scope],
                "distinct_states": len(scope_states[scope]),
                "gross_repeated_action_enumeration_seconds": (
                    scope_enumeration_seconds[scope]
                ),
                "gross_repeated_successor_application_seconds": (
                    scope_application_seconds[scope]
                ),
            }
            for scope in (
                "within_one_root_search",
                "across_candidates_within_one_frontier",
                "across_decision_frontiers",
            )
        },
        "repeated_work_by_root_role": {
            role: {
                "occurrences": role_counts[role],
                "gross_repeated_transition_seconds": role_seconds[role],
            }
            for role in sorted(role_counts)
        },
        "repeated_state_context": {
            "root_state_occurrences": repeated_root_occurrences,
            "returned_witness_state_occurrences": repeated_witness_occurrences,
        },
        "cost_adjusted_estimate": {
            "gross_repeated_action_enumeration_seconds": gross_enumeration,
            "gross_repeated_successor_application_seconds": gross_application,
            "gross_repeated_transition_seconds": gross_transition,
            "lookup_benchmark": lookup,
            "proof_validation_seconds": 0.0,
            "proof_validation_reason": (
                "transition_reuse_does_not_assert_SAFE_or_UNSAFE"
            ),
            "action_cache_optimistic_net_seconds_existing_solver_key_reused": (
                net_action_optimistic
            ),
            "action_cache_conservative_net_seconds_reconstructing_each_key": (
                net_action_conservative
            ),
            "transition_cache_optimistic_net_seconds": net_transition_optimistic,
            "conservative_net_fraction_of_reference_exact_search": (
                conservative_fraction
            ),
            "scope": (
                "gate_uses_action_enumeration_only;transition_estimate_is_separate;"
                "does_not_claim_reuse_of_unproved_search_subtrees"
            ),
        },
        "memory": {
            "unique_entries": len(unique_payload),
            "canonical_identifier_bytes": 32 * len(unique_payload),
            "serialized_action_payload_bytes": action_bytes,
            "successor_identifier_bytes": edge_identifier_bytes,
            "portable_action_cache_bytes": action_cache_bytes,
            "portable_transition_cache_lower_bound_bytes": memory_bytes,
            "successor_state_object_bytes_not_included": True,
            "python_object_overhead_not_included": True,
            "development_threshold_bytes": 512 * 1024 * 1024,
            "acceptable_on_sample": memory_acceptable,
        },
        "development_gate": {
            "threshold_net_fraction": 0.20,
            "advance_to_bounded_shared_action_cache": advance,
            "decision": "advance" if advance else "stop",
            "shared_SAFE_or_UNSAFE_proof_cache_authorized": False,
        },
    }


def analyze(output: Path, *, allow_partial: bool) -> dict:
    contract, _trace, _parent = program.authenticate(output)
    records = _load_records(output / program.LEDGER_NAME, contract)
    required = int(contract["sample"]["root_query_count"])
    if not records:
        raise program.D11Error("D11 has no completed audit roots")
    if len(records) != required and not allow_partial:
        raise program.D11Error(
            f"D11 has {len(records)}/{required} roots; use --allow-partial"
        )
    analysis = overlap_analysis(list(records.values()))
    report = e14.with_hash({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "artifact": "shared_search_opportunity_report",
        "contract_sha256": contract["contract_sha256"],
        "hardware": e14.hardware(),
        "status": "complete" if len(records) == required else "censored_partial",
        "completed_roots": len(records),
        "required_roots": required,
        "censored_is_not_failure_or_infeasibility": len(records) != required,
        "analysis": analysis,
        "interpretation_scope": {
            "frequency_alone_is_not_a_speedup": True,
            "net_estimate_subtracts_lookup_and_key_construction": True,
            "memory_is_portable_lower_bound_not_process_RSS": True,
            "repeated_state_is_not_a_recoverability_proof": True,
        },
    }, "report_sha256")
    e14.atomic_json(output / program.REPORT_NAME, report)
    return report


def authenticate_report(output: Path) -> dict:
    contract, _trace, _parent = program.authenticate(output)
    report = e14.load_json(output / program.REPORT_NAME, label="D11 report")
    if report.get("report_sha256") != e14.digest(
        report, hash_field="report_sha256"
    ):
        raise program.D11Error("D11 report self-hash mismatch")
    if report.get("contract_sha256") != contract["contract_sha256"]:
        raise program.D11Error("D11 report contract mismatch")
    _load_records(output / program.LEDGER_NAME, contract)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze", "authenticate"))
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    program.prepare(output)
    if args.command == "run":
        result = run(output)
    elif args.command == "analyze":
        result = analyze(output, allow_partial=args.allow_partial)
    else:
        result = authenticate_report(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
