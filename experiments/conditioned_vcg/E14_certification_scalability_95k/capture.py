#!/usr/bin/env python3
"""Capture the ordered certificate-query workload from the frozen verifier."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import signal
import sys
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E14_certification_scalability_95k import program
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
from experiments.conditioned_vcg.development.D10_scalability_support_screen import run as d10
import PSLAP.viability_candidates as candidate_module
import PSLAP.viability_filter as filter_module
from PSLAP.viability import RecoverabilityCertificate, RecoveryActionKind, RecoveryState
from PSLAP.viability_candidates import ViabilityCertificateCache
from PSLAP.viability_dataset import recovery_state_to_dict
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


class CaptureLimit(RuntimeError):
    pass


@contextmanager
def wall_limit(seconds: int):
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise CaptureLimit(f"predeclared {seconds}s capture limit reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


class QueryTracer:
    def __init__(self, *, reserve_queue_cells: bool) -> None:
        self.reserve_queue_cells = bool(reserve_queue_cells)
        self.full_states: dict[str, dict] = {}
        self.physical_states: dict[str, dict] = {}
        self.queries: list[dict] = []
        self.completed_frontiers: list[dict] = []
        self.pending_by_full_state: dict[str, deque[int]] = defaultdict(deque)
        self.recovery_roles: dict[RecoveryState, str] = {}
        self.frontier_index = -1
        self.decision_epoch: Optional[int] = None
        self._role: Optional[str] = None
        self._suppress_lookup = False
        self._cached_helper_calls = 0
        self.active_query_index: Optional[int] = None

    @contextmanager
    def role(self, value: str, *, suppress_lookup: bool = False):
        previous_role = self._role
        previous_suppress = self._suppress_lookup
        self._role = value
        self._suppress_lookup = suppress_lookup
        try:
            yield
        finally:
            self._role = previous_role
            self._suppress_lookup = previous_suppress

    def begin_frontier(self, env) -> None:
        self.frontier_index += 1
        self.decision_epoch = int(env.time_steps)
        self._cached_helper_calls = 0
        self.recovery_roles.clear()

    def finish_frontier(self, record: Mapping) -> None:
        if int(record["decision_epoch"]) != self.decision_epoch:
            raise program.E14Error("captured frontier epoch changed")
        self.completed_frontiers.append({
            "frontier_index": self.frontier_index,
            "decision_epoch": self.decision_epoch,
            "candidate_frontier_digest": record["candidate_frontier_digest"],
            "candidate_keys": list(record["candidate_keys"]),
            "cache_hits": int(record["cache_hits"]),
            "cache_misses": int(record["cache_misses"]),
            "safe_candidate_count": int(record["safe_candidate_count"]),
            "unsafe_candidates_rejected": int(record["unsafe_candidates_rejected"]),
            "unknown_candidates_rejected": int(record["unknown_candidates_rejected"]),
        })

    def register_recovery_successor(self, state: RecoveryState, kind) -> None:
        role = (
            "deliver"
            if kind is RecoveryActionKind.DELIVERY
            else "reconfigure"
        )
        previous = self.recovery_roles.get(state)
        if previous is not None and previous != role:
            raise program.E14Error("one recovery successor received conflicting roles")
        self.recovery_roles[state] = role

    def cached_helper_role(self, state: RecoveryState) -> tuple[str, bool]:
        self._cached_helper_calls += 1
        if self._cached_helper_calls == 1:
            return "current_state", False
        role = self.recovery_roles.get(state)
        if role is None:
            raise program.E14Error("could not attribute recovery cache query")
        # Misses in the explicit recovery precheck are recorded when the
        # ordered exact-check loop actually requests them. This also handles
        # duplicate successors that become late cache hits correctly.
        return role, False

    def _store_state(self, state: RecoveryState) -> tuple[str, str]:
        full_id = reuse.full_state_id(state)
        physical = reuse.timing_erased_state(state)
        physical_id = reuse.physical_state_id(physical)
        self.full_states.setdefault(full_id, recovery_state_to_dict(state))
        self.physical_states.setdefault(
            physical_id, recovery_state_to_dict(physical)
        )
        return full_id, physical_id

    def _role_for(self, state: RecoveryState) -> str:
        if self._role is not None:
            return self._role
        role = self.recovery_roles.get(state)
        if role is None:
            raise program.E14Error("certificate query has no attributed role")
        return role

    def cache_lookup(self, key, certificate: Optional[RecoverabilityCertificate]) -> None:
        if self._suppress_lookup:
            return
        if not isinstance(key, tuple) or len(key) != 5 or not isinstance(key[0], RecoveryState):
            raise program.E14Error("unexpected certificate cache key during capture")
        state = key[0]
        role = self._role_for(state)
        if self._role is None and role in {"deliver", "reconfigure"} and certificate is None:
            # This is the candidate module's preliminary recovery-cache scan,
            # not yet an executed certificate query.
            return
        full_id, physical_id = self._store_state(state)
        query = {
            "query_index": len(self.queries),
            "frontier_index": self.frontier_index,
            "decision_epoch": self.decision_epoch,
            "check_role": role,
            "full_state_id": full_id,
            "physical_state_id": physical_id,
            "search": {
                "max_depth": key[1],
                "max_nodes": key[2],
                "max_primitive_steps": key[3],
                "reserve_queue_cells": self.reserve_queue_cells,
                "search_order": key[4],
            },
            "original_cache_hit": certificate is not None,
            "original_search_seconds": 0.0 if certificate is not None else None,
            "result": (
                reuse.certificate_to_dict(certificate)
                if certificate is not None
                else None
            ),
        }
        self.queries.append(query)
        if certificate is None:
            self.pending_by_full_state[full_id].append(query["query_index"])

    def search_started(self, state: RecoveryState) -> int:
        full_id, _physical_id = self._store_state(state)
        pending = self.pending_by_full_state.get(full_id)
        if not pending:
            raise program.E14Error("search began without a preceding cache miss")
        index = pending.popleft()
        self.active_query_index = index
        return index

    def search_finished(
        self,
        index: int,
        certificate: RecoverabilityCertificate,
        seconds: float,
    ) -> None:
        if index != self.active_query_index:
            raise program.E14Error("certificate search completion order changed")
        query = self.queries[index]
        if query["result"] is not None:
            raise program.E14Error("certificate query was completed twice")
        query["result"] = reuse.certificate_to_dict(certificate)
        query["original_search_seconds"] = float(seconds)
        self.active_query_index = None

    def payload(self, *, scenario_id: str, capture_outcome: Mapping) -> dict:
        unresolved = [
            query["query_index"] for query in self.queries
            if query["result"] is None
        ]
        payload = {
            "schema_version": program.SCHEMA_VERSION,
            "protocol": program.PROTOCOL,
            "artifact": "ordered_certificate_query_workload",
            "scenario_id": scenario_id,
            "instance_seed": program.INSTANCE_SEED,
            "capture_outcome": dict(capture_outcome),
            "full_states": self.full_states,
            "physical_states": self.physical_states,
            "queries": self.queries,
            "completed_frontiers": self.completed_frontiers,
            "completed_frontier_count": len(self.completed_frontiers),
            "query_count": len(self.queries),
            "unique_full_states": len(self.full_states),
            "unique_physical_states": len(self.physical_states),
            "unresolved_query_indices": unresolved,
            "ordered_duplicates_retained": True,
        }
        return program.with_hash(payload, "trace_sha256")


class TracingCertificateCache(ViabilityCertificateCache):
    def __init__(self, tracer: QueryTracer) -> None:
        super().__init__()
        self.tracer = tracer

    def get(self, key, default=None):
        certificate = self._entries.get(key)
        self.tracer.cache_lookup(key, certificate)
        return default if certificate is None else certificate


@contextmanager
def capture_active(tracer: QueryTracer, cache: TracingCertificateCache):
    original_enumerate = benchmark._enumerate_frontier
    original_cached = candidate_module._analyze_cached
    original_apply = candidate_module.apply_recovery_action
    original_accept = candidate_module.certify_post_accept_candidates
    original_candidate_analyze = candidate_module.analyze_recoverability
    original_filter_analyze = filter_module.analyze_recoverability

    def traced_enumerate(env, *args, **kwargs):
        tracer.begin_frontier(env)
        result = original_enumerate(env, *args, **kwargs)
        tracer.finish_frontier(result[1])
        return result

    def traced_apply(state, action):
        successor = original_apply(state, action)
        tracer.register_recovery_successor(successor, action.kind)
        return successor

    def traced_cached(state, config, active_cache):
        role, suppress = tracer.cached_helper_role(state)
        with tracer.role(role, suppress_lookup=suppress):
            return original_cached(state, config, active_cache)

    def traced_accept(*args, **kwargs):
        with tracer.role("accept"):
            return original_accept(*args, **kwargs)

    def timed(function, state, *args, **kwargs):
        index = tracer.search_started(state)
        started = perf_counter()
        certificate = function(state, *args, **kwargs)
        tracer.search_finished(index, certificate, perf_counter() - started)
        return certificate

    def traced_candidate_analyze(state, *args, **kwargs):
        return timed(original_candidate_analyze, state, *args, **kwargs)

    def traced_filter_analyze(state, *args, **kwargs):
        return timed(original_filter_analyze, state, *args, **kwargs)

    cache_uses = 0

    def cache_factory():
        nonlocal cache_uses
        cache_uses += 1
        if cache_uses != 1:
            raise program.E14Error("capture expected one episode-local cache")
        return cache

    with ExitStack() as stack:
        stack.enter_context(patch.object(benchmark, "_enumerate_frontier", traced_enumerate))
        stack.enter_context(patch.object(candidate_module, "apply_recovery_action", traced_apply))
        stack.enter_context(patch.object(candidate_module, "_analyze_cached", traced_cached))
        stack.enter_context(patch.object(candidate_module, "certify_post_accept_candidates", traced_accept))
        stack.enter_context(patch.object(candidate_module, "analyze_recoverability", traced_candidate_analyze))
        stack.enter_context(patch.object(filter_module, "analyze_recoverability", traced_filter_analyze))
        stack.enter_context(patch.object(benchmark, "ViabilityCertificateCache", cache_factory))
        yield


def _parent_record(manifest: Mapping, scenario_id: str) -> Mapping:
    records = [
        record for record in manifest["records"]
        if record["scenario_id"] == scenario_id
        and int(record["seed"]) == program.INSTANCE_SEED
    ]
    if len(records) != 1:
        raise program.E14Error(f"missing unique parent record for {scenario_id}")
    return records[0]


def capture_one(output: Path, contract: Mapping, manifest: Mapping, scenario_id: str) -> dict:
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    record = _parent_record(manifest, scenario_id)
    instance = occupancy._load_instance(program.PARENT_OUTPUT, record)
    conditioned = occupancy._conditioned_auth()
    arm = conditioned["inputs"]["arms"][program.MODEL_SEED]
    search_config = benchmark._search_config(arm.payload)
    tracer = QueryTracer(reserve_queue_cells=search_config.reserve_queue_cells)
    cache = TracingCertificateCache(tracer)
    env_holder = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            PROJECT_ROOT,
            {"conditioned": conditioned},
            model_seed=program.MODEL_SEED,
            base=base,
            device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        return conditioned_seed0._FixedLambdaAgent(
            agent, program.PREFERENCE_LAMBDA
        )

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        env_holder.append(env)
        return env

    started = perf_counter()
    raw = None
    error = None
    failure_class = None
    try:
        with wall_limit(program.CAPTURE_LIMIT_SECONDS), capture_active(tracer, cache), patch.object(benchmark, "_make_env", env_factory), pilot._agent_factory(agent_factory):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=search_config,
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
            )
    except CaptureLimit as caught:
        failure_class = "censored_wall_clock"
        error = str(caught)
    except Exception as caught:  # retained as an authenticated diagnostic row
        failure_class = "unsupported_or_implementation_error"
        error = f"{type(caught).__name__}: {caught}"
    elapsed = perf_counter() - started

    if raw is not None:
        strict = bool(
            raw["strict_method_success"] and raw["terminal"]
            and raw["method_failure_reason"] is None
            and raw["complete_frontier_exactly_verified"]
            and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
            and len(raw["delivery_deviations"]) == scenario.total_jobs
        )
        failure_class = "completed" if strict else "valid_but_operationally_incomplete"
        capture_outcome = {
            "failure_class": failure_class,
            "strict_safe_complete": strict,
            "episode_wall_seconds": float(raw["episode_wall_seconds"]),
            "steps": int(raw["steps"]),
            "deliveries_at_stop": len(raw["delivery_deviations"]),
            "required_deliveries": scenario.total_jobs,
            "method_failure_reason": raw["method_failure_reason"],
            "behavior_digest": raw["behavior_digest"],
            "frontier_digests": [
                item["candidate_frontier_digest"] for item in raw["frontiers"]
            ],
        }
    else:
        capture_outcome = {
            "failure_class": failure_class,
            "strict_safe_complete": False,
            "wall_seconds_at_stop": elapsed,
            "reason": error,
            "infeasibility_claimed": False,
            "occupancy_at_stop": env_holder[0]._measurement() if env_holder else None,
        }
    trace = tracer.payload(
        scenario_id=scenario_id, capture_outcome=capture_outcome
    )
    trace_file = program.trace_path(output, scenario_id)
    program.atomic_gzip_json(trace_file, trace)
    ledger = program.with_hash({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "scenario_id": scenario_id,
        "instance_seed": program.INSTANCE_SEED,
        "capture_outcome": capture_outcome,
        "query_count": trace["query_count"],
        "unique_full_states": trace["unique_full_states"],
        "unique_physical_states": trace["unique_physical_states"],
        "completed_frontiers": trace["completed_frontier_count"],
        "unresolved_queries": len(trace["unresolved_query_indices"]),
        "trace_relative_path": str(trace_file.relative_to(output)),
        "trace_file_sha256": program.sha256(trace_file),
        "trace_sha256": trace["trace_sha256"],
    }, "ledger_sha256")
    program.atomic_json(program.capture_ledger_path(output, scenario_id), ledger)
    return ledger


def authenticate_capture(output: Path, contract: Mapping, scenario_id: str) -> tuple[dict, dict]:
    ledger = program.load_json(
        program.capture_ledger_path(output, scenario_id),
        label=f"E14 {scenario_id} capture ledger",
    )
    if ledger.get("ledger_sha256") != program.digest(ledger, hash_field="ledger_sha256"):
        raise program.E14Error("capture ledger self-hash mismatch")
    if ledger.get("contract_sha256") != contract["contract_sha256"]:
        raise program.E14Error("capture ledger contract mismatch")
    path = output / ledger["trace_relative_path"]
    if program.sha256(path) != ledger["trace_file_sha256"]:
        raise program.E14Error("query workload file hash mismatch")
    trace = program.load_gzip_json(path, label="E14 query workload")
    if trace.get("trace_sha256") != program.digest(trace, hash_field="trace_sha256"):
        raise program.E14Error("query workload semantic hash mismatch")
    if trace.get("trace_sha256") != ledger["trace_sha256"]:
        raise program.E14Error("capture ledger/query workload mismatch")
    if trace.get("scenario_id") != scenario_id:
        raise program.E14Error("query workload scenario mismatch")
    return ledger, trace


def run(output: Path, scenario_ids: Sequence[str]) -> dict:
    contract, manifest = program.authenticate(output)
    rows = []
    for scenario_id in scenario_ids:
        path = program.capture_ledger_path(output, scenario_id)
        if path.exists():
            ledger, _trace = authenticate_capture(output, contract, scenario_id)
        else:
            ledger = capture_one(output, contract, manifest, scenario_id)
        rows.append({
            "scenario_id": scenario_id,
            "failure_class": ledger["capture_outcome"]["failure_class"],
            "queries": ledger["query_count"],
            "unique_physical_states": ledger["unique_physical_states"],
            "unresolved_queries": ledger["unresolved_queries"],
        })
        print(
            f"E14 capture {scenario_id} | {rows[-1]['failure_class']} "
            f"| queries={rows[-1]['queries']}",
            flush=True,
        )
    return {"status": "complete", "rows": rows}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("capture-medium", "capture-high", "capture-both", "authenticate"),
    )
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    program.prepare(output)
    if args.command == "authenticate":
        contract, _manifest = program.authenticate(output)
        result = {
            scenario_id: authenticate_capture(output, contract, scenario_id)[0]
            for scenario_id in program.SCENARIO_IDS
            if program.capture_ledger_path(output, scenario_id).exists()
        }
    else:
        scenario_ids = {
            "capture-medium": (program.SCENARIO_IDS[0],),
            "capture-high": (program.SCENARIO_IDS[1],),
            "capture-both": program.SCENARIO_IDS,
        }[args.command]
        result = run(output, scenario_ids)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
