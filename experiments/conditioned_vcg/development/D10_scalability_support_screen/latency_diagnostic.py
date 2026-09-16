#!/usr/bin/env python3
"""Per-candidate latency diagnosis for the 10x10 medium/high D10 pilots.

This is deliberately separate from the frozen occupancy-extension protocol.
It reuses the same frozen controller and EpisodeInstances, but adds the two
pieces of telemetry needed to distinguish candidate breadth from difficult
individual certificates: decision-boundary occupancy and a macro-family label
for every exact cache miss.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
from statistics import fmean
import sys
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
from experiments.conditioned_vcg.development.D10_scalability_support_screen import run as d10
import PSLAP.viability_candidates as candidate_module
import PSLAP.viability_filter as filter_module
from PSLAP.viability import RecoveryActionKind, apply_recovery_action
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_d10_10x10_latency_diagnostic_95k_v1"
SCHEMA_VERSION = 1
SCENARIO_IDS = ("size_10x10_occ_medium", "size_10x10_occ_high")
INSTANCE_SEED = 95_100
WALL_CLOCK_LIMIT_SECONDS = 3_600
PARENT_OUTPUT = PROJECT_ROOT / "results/vcg-d10-occupancy-extension-95k"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-d10-10x10-latency-diagnostic-95k"
CONTRACT_NAME = "latency-diagnostic-contract.json"
REPORT_NAME = "latency-diagnostic-report.json"


class LatencyDiagnosticError(RuntimeError):
    pass


class EpisodeWallClockLimit(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Mapping, *, field: Optional[str] = None) -> str:
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
        raise LatencyDiagnosticError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise LatencyDiagnosticError(f"missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise LatencyDiagnosticError(f"{label} must contain an object")
    return value


def _source_path() -> Path:
    return Path(__file__).resolve()


def _expected_contract() -> dict:
    parent_contract, parent_manifest = occupancy.authenticate(PARENT_OUTPUT)
    records = [
        record for record in parent_manifest["records"]
        if record["scenario_id"] in SCENARIO_IDS
        and int(record["seed"]) == INSTANCE_SEED
    ]
    if len(records) != len(SCENARIO_IDS):
        raise LatencyDiagnosticError("parent manifest lacks the two pilot instances")
    return _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "role": "development_candidate_breadth_vs_individual_check_difficulty",
        "parent_protocol": parent_contract["protocol"],
        "parent_contract_sha256": parent_contract["contract_sha256"],
        "parent_manifest_sha256": parent_manifest["manifest_sha256"],
        "parent_runner_sha256": _sha(Path(occupancy.__file__).resolve()),
        "diagnostic_source_sha256": _sha(_source_path()),
        "scenario_ids": list(SCENARIO_IDS),
        "instance_seed": INSTANCE_SEED,
        "model_seed": occupancy.MODEL_SEED,
        "preference_lambda": occupancy.PREFERENCE_LAMBDA,
        "device": "cpu",
        "sequential_isolated_execution_required": True,
        "wall_clock_limit_seconds_per_episode": WALL_CLOCK_LIMIT_SECONDS,
        "timeout_interpretation": "censored_not_infeasible_not_failed",
        "frozen_controller_no_training": True,
        "telemetry": [
            "decision_boundary_occupancy_and_admitted_workload",
            "candidate_counts_by_macro_family",
            "cache_hits_and_misses_by_certificate_subject",
            "certificate_latency_and_expansions_per_cache_miss",
            "slowest_individual_certificate_checks",
        ],
    }, "contract_sha256")


def prepare(output: Path) -> dict:
    contract = _expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        existing = _load_json(path, "latency diagnostic contract")
        if existing != contract:
            raise LatencyDiagnosticError("latency diagnostic contract, parent, or source changed")
    else:
        _atomic_json(path, contract)
    return contract


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = _load_json(output / CONTRACT_NAME, "latency diagnostic contract")
    if contract != _expected_contract():
        raise LatencyDiagnosticError("latency diagnostic contract, parent, or source changed")
    _parent_contract, manifest = occupancy.authenticate(PARENT_OUTPUT)
    return contract, manifest


@contextmanager
def _wall_clock_limit(seconds: int):
    if not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise EpisodeWallClockLimit(f"predeclared {seconds}s episode limit reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _call_distribution(calls: Sequence[Mapping]) -> dict:
    return {
        "count": len(calls),
        "seconds": d10._distribution([float(call["seconds"]) for call in calls]),
        "explored_nodes": d10._distribution([float(call["explored_nodes"]) for call in calls]),
        "total_seconds": sum(float(call["seconds"]) for call in calls),
        "status_counts": dict(Counter(str(call["status"]) for call in calls)),
    }


class RichFrontierInstrumentation:
    """Record exact-check type, per-check cost, and decision-state load."""

    def __init__(self) -> None:
        self.certificate_calls: list[dict] = []
        self.legal_action_batches: list[tuple] = []
        self.recovery_roles: dict[object, deque[dict]] = defaultdict(deque)
        self.completed_frontiers: list[dict] = []
        self.in_progress: Optional[dict] = None
        self.active_check: Optional[dict] = None

    def _timed(self, function, state, role: str, *args, **kwargs):
        started = perf_counter()
        self.active_check = {"check_role": role, "started": started}
        certificate = function(state, *args, **kwargs)
        self.certificate_calls.append({
            "check_role": role,
            "status": certificate.status.value,
            "explored_nodes": int(certificate.explored_nodes),
            "generated_states": int(certificate.generated_states),
            "max_depth_reached": int(certificate.max_depth_reached),
            "max_primitive_steps_reached": int(certificate.max_primitive_steps_reached),
            "exhaustive": bool(certificate.exhaustive),
            "seconds": perf_counter() - started,
        })
        self.active_check = None
        return certificate

    def partial_frontier(self) -> Optional[dict]:
        if self.in_progress is None:
            return None
        start = int(self.in_progress["call_start"])
        calls = self.certificate_calls[start:]
        partial = {
            **self.in_progress,
            "call_start": None,
            "elapsed_seconds_at_censoring": perf_counter() - float(self.in_progress["started"]),
            "certification_calls_completed": calls,
            "checks_by_role": {
                role: _call_distribution([call for call in calls if call["check_role"] == role])
                for role in sorted({call["check_role"] for call in calls})
            },
        }
        if self.active_check is not None:
            partial["interrupted_check"] = {
                "check_role": self.active_check["check_role"],
                "elapsed_seconds_at_censoring": (
                    perf_counter() - float(self.active_check["started"])
                ),
                "completed": False,
            }
        return partial

    @contextmanager
    def active(self):
        core_candidate_analyze = candidate_module.analyze_recoverability
        core_filter_analyze = filter_module.analyze_recoverability
        core_legal = candidate_module.legal_recovery_actions
        original_enumerate = benchmark._enumerate_frontier

        def timed_candidate(state, *args, **kwargs):
            queued = self.recovery_roles.get(state)
            metadata = queued.popleft() if queued else None
            role = metadata["check_role"] if metadata else "current_state"
            return self._timed(core_candidate_analyze, state, role, *args, **kwargs)

        def timed_accept(state, *args, **kwargs):
            return self._timed(core_filter_analyze, state, "accept", *args, **kwargs)

        def recorded_legal(state):
            actions = tuple(core_legal(state))
            self.legal_action_batches.append(actions)
            for action in actions:
                successor = apply_recovery_action(state, action)
                role = (
                    "deliver"
                    if action.kind is RecoveryActionKind.DELIVERY
                    else "reconfigure"
                )
                self.recovery_roles[successor].append({
                    "check_role": role,
                    "target_label": action.block_label,
                    "destination": list(action.destination) if action.destination is not None else None,
                })
            return actions

        def enriched_enumerate(env, *args, **kwargs):
            call_start = len(self.certificate_calls)
            legal_start = len(self.legal_action_batches)
            started = perf_counter()
            occupancy_now = env._measurement()
            self.in_progress = {
                "decision_boundary": occupancy_now,
                "call_start": call_start,
                "started": started,
            }
            snapshot, record = original_enumerate(env, *args, **kwargs)
            audit = snapshot.audit.audit_dict()
            calls = self.certificate_calls[call_start:]
            batches = self.legal_action_batches[legal_start:]
            legal = batches[-1] if batches else ()
            kinds = Counter(action.kind.value for action in legal)
            subjects = {
                "current_state": 1,
                "accept": int(audit["executable_accept_count"]),
                "deliver": int(kinds[RecoveryActionKind.DELIVERY.value]),
                "reconfigure": int(kinds[RecoveryActionKind.RELOCATION.value]),
            }
            miss_counts = Counter(call["check_role"] for call in calls)
            cache_by_role = {
                role: {
                    "subjects": count,
                    "cache_misses": int(miss_counts[role]),
                    "cache_hits": int(count - miss_counts[role]),
                    "certificate_seconds": sum(
                        float(call["seconds"]) for call in calls
                        if call["check_role"] == role
                    ),
                }
                for role, count in subjects.items()
            }
            if sum(item["cache_misses"] for item in cache_by_role.values()) != int(audit["cache_misses"]):
                raise LatencyDiagnosticError("cache-miss role attribution mismatch")
            if sum(item["cache_hits"] for item in cache_by_role.values()) != int(audit["cache_hits"]):
                raise LatencyDiagnosticError("cache-hit role attribution mismatch")
            record.update({
                "decision_boundary": occupancy_now,
                "physical_candidate_count": int(audit["physical_accept_count"]) + len(legal) + int(bool(audit["defer_allowed"])),
                "physical_accept_count": int(audit["physical_accept_count"]),
                "physical_deliver_count": int(kinds[RecoveryActionKind.DELIVERY.value]),
                "physical_reconfigure_count": int(kinds[RecoveryActionKind.RELOCATION.value]),
                "physical_defer_count": int(bool(audit["defer_allowed"])),
                "certified_candidate_count": int(audit["candidate_count"]),
                "certified_accept_count": int(audit["accept_candidate_count"]),
                "certified_deliver_count": int(audit["deliver_candidate_count"]),
                "certified_reconfigure_count": int(audit["reconfigure_candidate_count"]),
                "certified_defer_count": int(audit["defer_candidate_count"]),
                "unknown_candidate_count": int(audit["unknown_accept_count"]) + int(audit["unknown_recovery_count"]),
                "unsafe_candidate_count": int(audit["unsafe_accept_count"]) + int(audit["unsafe_recovery_count"]),
                "cache_by_check_role": cache_by_role,
                "candidate_generation_and_overhead_seconds": max(float(record["total_frontier_seconds"]) - float(record["exact_search_seconds"]), 0.0),
                "certification_calls": calls,
                "checks_by_role": {
                    role: _call_distribution([call for call in calls if call["check_role"] == role])
                    for role in subjects
                },
            })
            self.completed_frontiers.append(record)
            self.in_progress = None
            return snapshot, record

        with ExitStack() as stack:
            stack.enter_context(patch.object(candidate_module, "analyze_recoverability", timed_candidate))
            stack.enter_context(patch.object(filter_module, "analyze_recoverability", timed_accept))
            stack.enter_context(patch.object(candidate_module, "legal_recovery_actions", recorded_legal))
            stack.enter_context(patch.object(benchmark, "_enumerate_frontier", enriched_enumerate))
            yield


def _parent_record(manifest: Mapping, scenario_id: str) -> Mapping:
    matches = [
        record for record in manifest["records"]
        if record["scenario_id"] == scenario_id and int(record["seed"]) == INSTANCE_SEED
    ]
    if len(matches) != 1:
        raise LatencyDiagnosticError(f"expected one parent record for {scenario_id}")
    return matches[0]


def _run_one(contract: Mapping, manifest: Mapping, scenario_id: str) -> dict:
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    record = _parent_record(manifest, scenario_id)
    instance = occupancy._load_instance(PARENT_OUTPUT, record)
    conditioned = occupancy._conditioned_auth()
    arm = conditioned["inputs"]["arms"][occupancy.MODEL_SEED]
    agent_holder = []
    env_holder = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            PROJECT_ROOT, {"conditioned": conditioned},
            model_seed=occupancy.MODEL_SEED, base=base, device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(agent, occupancy.PREFERENCE_LAMBDA)
        timed = d10.SelectionTimer(fixed)
        agent_holder.append(timed)
        return timed

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        env_holder.append(env)
        return env

    recorder = RichFrontierInstrumentation()
    started = perf_counter()
    try:
        with _wall_clock_limit(WALL_CLOCK_LIMIT_SECONDS), recorder.active(), patch.object(benchmark, "_make_env", env_factory), pilot._agent_factory(agent_factory):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=benchmark._search_config(arm.payload),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
            )
    except EpisodeWallClockLimit as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scenario": scenario.public_dict(),
            "instance_seed": INSTANCE_SEED,
            "failure_class": "censored_wall_clock",
            "strict_safe_complete": False,
            "censoring_reason": str(error),
            "wall_seconds_at_censoring": perf_counter() - started,
            "completed_frontiers": recorder.completed_frontiers,
            "in_progress_frontier": recorder.partial_frontier(),
            "occupancy_at_censoring": env_holder[0]._measurement() if env_holder else None,
            "occupancy_summary_at_censoring": env_holder[0].occupancy_summary() if env_holder else None,
            "infeasibility_claimed": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scenario": scenario.public_dict(),
            "instance_seed": INSTANCE_SEED,
            "failure_class": "unsupported_or_implementation_error",
            "strict_safe_complete": False,
            "implementation_error": f"{type(error).__name__}: {error}",
            "episode_wall_seconds": perf_counter() - started,
            "completed_frontiers": recorder.completed_frontiers,
            "in_progress_frontier": recorder.partial_frontier(),
        }

    if len(agent_holder) != 1 or len(env_holder) != 1 or len(agent_holder[0].records) != len(raw["decisions"]):
        raise LatencyDiagnosticError("runtime instrumentation did not align")
    selected_by_frontier = {int(item["frontier_index"]): item for item in raw["decisions"]}
    frontiers = []
    for index, frontier in enumerate(raw["frontiers"]):
        selected = selected_by_frontier.get(index)
        frontiers.append({
            **frontier,
            "selected_action_type": selected["selected_action_type"] if selected else None,
            "selected_key": selected["selected_key"] if selected else None,
        })
    strict = bool(
        raw["strict_method_success"] and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == scenario.total_jobs
    )
    timing = d10._timing(raw["delivery_deviations"], expected_deliveries=scenario.total_jobs) if strict else {}
    rehandles = int(raw.get("physical_storage_relocations", raw.get("relocations", 0)))
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scenario": scenario.public_dict(),
        "instance_seed": INSTANCE_SEED,
        "model_seed": occupancy.MODEL_SEED,
        "preference_lambda": occupancy.PREFERENCE_LAMBDA,
        "failure_class": "completed" if strict else "valid_but_operationally_incomplete",
        "strict_safe_complete": strict,
        "method_failure_reason": raw["method_failure_reason"],
        "deliveries_at_stop": len(raw["delivery_deviations"]),
        "required_deliveries": scenario.total_jobs,
        "steps": int(raw["steps"]),
        "physical_rehandles": rehandles,
        "physical_rehandles_per_100_required_deliveries": 100.0 * rehandles / scenario.total_jobs,
        **timing,
        "episode_wall_seconds": float(raw["episode_wall_seconds"]),
        "frontiers": frontiers,
        "occupancy": env_holder[0].occupancy_summary(),
        "all_selected_candidates_exact_safe": bool(raw["complete_frontier_exactly_verified"]),
        "behavior_digest": raw["behavior_digest"],
    }


def _ledger_path(output: Path, scenario_id: str) -> Path:
    return output / "run-ledger" / scenario_id / f"seed-{INSTANCE_SEED}.json"


def run(output: Path, scenario_ids: Sequence[str]) -> dict:
    contract, manifest = authenticate(output)
    completed = []
    for scenario_id in scenario_ids:
        path = _ledger_path(output, scenario_id)
        if path.exists():
            ledger = _load_json(path, "latency diagnostic ledger")
            if ledger.get("ledger_sha256") != _digest(ledger, field="ledger_sha256"):
                raise LatencyDiagnosticError("latency diagnostic ledger hash mismatch")
        else:
            row = _run_one(contract, manifest, scenario_id)
            ledger = _with_hash({
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "contract_sha256": contract["contract_sha256"],
                "row": row,
            }, "ledger_sha256")
            _atomic_json(path, ledger)
        completed.append(ledger["row"]["failure_class"])
        print(f"D10-latency {scenario_id} | {ledger['row']['failure_class']}", flush=True)
    return {"status": "complete", "scenario_ids": list(scenario_ids), "outcomes": completed}


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))[0, 1])


def _summarize_row(row: Mapping) -> dict:
    frontiers = list(row.get("frontiers", row.get("completed_frontiers", ())))
    partial = row.get("in_progress_frontier")
    all_calls = [call for frontier in frontiers for call in frontier.get("certification_calls", ())]
    if partial:
        all_calls.extend(partial.get("certification_calls_completed", ()))
    complete_with_calls = [frontier for frontier in frontiers if frontier.get("certification_calls")]
    breadth = [float(frontier["physical_candidate_count"]) for frontier in complete_with_calls]
    total_time = [float(frontier["exact_search_seconds"]) for frontier in complete_with_calls]
    mean_check = [
        fmean(float(call["seconds"]) for call in frontier["certification_calls"])
        for frontier in complete_with_calls
    ]
    max_check = [
        max(float(call["seconds"]) for call in frontier["certification_calls"])
        for frontier in complete_with_calls
    ]
    slowest = sorted(all_calls, key=lambda call: float(call["seconds"]), reverse=True)[:20]
    role_names = ("current_state", "accept", "deliver", "reconfigure")
    return {
        "scenario": row["scenario"],
        "failure_class": row["failure_class"],
        "strict_safe_complete": row["strict_safe_complete"],
        "episode_wall_seconds": row.get("episode_wall_seconds", row.get("wall_seconds_at_censoring")),
        "completed_frontiers": len(frontiers),
        "occupancy": row.get("occupancy", row.get("occupancy_summary_at_censoring")),
        "candidate_breadth": d10._distribution(breadth),
        "individual_check_seconds": d10._distribution([float(call["seconds"]) for call in all_calls]),
        "individual_check_expansions": d10._distribution([float(call["explored_nodes"]) for call in all_calls]),
        "checks_by_role": {
            role: _call_distribution([call for call in all_calls if call["check_role"] == role])
            for role in role_names
        },
        "frontier_seconds": d10._distribution([float(frontier["total_frontier_seconds"]) for frontier in frontiers]),
        "certification_seconds": sum(float(call["seconds"]) for call in all_calls),
        "certification_share_of_observed_wall": (
            sum(float(call["seconds"]) for call in all_calls)
            / float(row.get("episode_wall_seconds", row.get("wall_seconds_at_censoring")))
            if row.get("episode_wall_seconds", row.get("wall_seconds_at_censoring"))
            else None
        ),
        "breadth_vs_frontier_time_pearson": _correlation(breadth, total_time),
        "mean_check_time_vs_frontier_time_pearson": _correlation(mean_check, total_time),
        "max_check_time_vs_frontier_time_pearson": _correlation(max_check, total_time),
        "slowest_checks": slowest,
        "censored_is_not_infeasible": row["failure_class"] == "censored_wall_clock",
    }


def analyze(output: Path, allow_partial: bool) -> dict:
    contract, _manifest = authenticate(output)
    rows = []
    for scenario_id in SCENARIO_IDS:
        path = _ledger_path(output, scenario_id)
        if not path.exists():
            continue
        ledger = _load_json(path, "latency diagnostic ledger")
        if ledger.get("ledger_sha256") != _digest(ledger, field="ledger_sha256"):
            raise LatencyDiagnosticError("latency diagnostic ledger hash mismatch")
        if ledger.get("contract_sha256") != contract["contract_sha256"]:
            raise LatencyDiagnosticError("latency diagnostic ledger contract mismatch")
        rows.append(ledger["row"])
    if not allow_partial and len(rows) != len(SCENARIO_IDS):
        raise LatencyDiagnosticError(f"expected {len(SCENARIO_IDS)} ledgers, found {len(rows)}")
    report = _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "partial": len(rows) != len(SCENARIO_IDS),
        "expected_rows": len(SCENARIO_IDS),
        "observed_rows": len(rows),
        "summaries": [_summarize_row(row) for row in rows],
        "interpretation": {
            "purpose": "distinguish frontier breadth from individual exact-check difficulty",
            "timeout": "right-censored development observation; no infeasibility or failure inference",
            "latency": "isolated CPU timing required; no training occurs",
        },
    }, "report_sha256")
    _atomic_json(output / REPORT_NAME, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-medium", "run-high", "run-both", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-medium":
        prepare(output)
        result = run(output, (SCENARIO_IDS[0],))
    elif args.command == "run-high":
        prepare(output)
        result = run(output, (SCENARIO_IDS[1],))
    elif args.command == "run-both":
        prepare(output)
        result = run(output, SCENARIO_IDS)
    else:
        result = analyze(output, args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
