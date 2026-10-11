#!/usr/bin/env python3
"""D10: zero-shot scale support and instrumentation screen for Block 6."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
from statistics import fmean, median
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
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
import PSLAP.viability_candidates as candidate_module
import PSLAP.viability_filter as filter_module
from PSLAP.viability import RecoveryActionKind
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_d10_zero_shot_scalability_support_screen_95k_v2"
SCHEMA_VERSION = 2
MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
INSTANCE_SEEDS = (95_000, 95_001, 95_002)
PILOT_INSTANCE_LIMIT = 1
SEARCH_MAX_NODES = 20_000
MAX_STEPS = 2_000
PRIMARY_DEADLINE_SECONDS = 1.0
DEADLINE_SECONDS = (0.1, 1.0, 5.0)
CONTRACT_NAME = "d10-contract.json"
MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "d10-report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-d10-scalability-support-screen-95k-v2"
_CONDITIONED_AUTH_CACHE = None


class D10Error(RuntimeError):
    pass


def _conditioned_auth():
    global _CONDITIONED_AUTH_CACHE
    if _CONDITIONED_AUTH_CACHE is None:
        _CONDITIONED_AUTH_CACHE = final90._authenticate_conditioned(PROJECT_ROOT)
    return _CONDITIONED_AUTH_CACHE


@dataclass(frozen=True)
class Scale:
    scale_id: str
    axis: str
    rows: int
    cols: int
    blocks: int
    comparison_group: str

    @property
    def door(self) -> tuple[int, int]:
        return (0, self.cols - 2)

    @property
    def exits(self) -> tuple[tuple[int, int], ...]:
        right = self.cols - 2
        return tuple((self.rows - 1, col) for col in range(right - 2, right + 1))

    def make_env(self) -> SmallRoomsEnv:
        return SmallRoomsEnv(
            grid_rows=self.rows,
            grid_cols=self.cols,
            start_state=(1, 1),
            door_cell=self.door,
            pickup_cells=(1, self.cols - 2),
            exit_cells=list(self.exits),
            number_blocks=self.blocks,
            choose_storage=False,
            arrival_rate=10.0,
            proc_mean=80,
        )

    def public_dict(self) -> dict:
        env = self.make_env()
        return {
            **asdict(self),
            "yard_cells": self.rows * self.cols,
            "storage_capacity": len(env.storage_positions),
            "nominal_workload_to_capacity": self.blocks / len(env.storage_positions),
            "door_cell": list(env.door_cell),
            "pickup_cell": list(env.pickup_cell),
            "exit_cells": [list(cell) for cell in env.exit_cells],
        }


SCALES = (
    Scale("reference_5x5_n8", "reference", 5, 5, 8, "reference"),
    Scale("geometry_6x6_n8", "geometry_only", 6, 6, 8, "geometry"),
    Scale("geometry_7x7_n8", "geometry_only", 7, 7, 8, "geometry"),
    Scale("geometry_8x8_n8", "geometry_only", 8, 8, 8, "geometry"),
    Scale("workload_6x6_n12", "workload_at_6x6", 6, 6, 12, "workload"),
    Scale("workload_6x6_n15", "workload_at_6x6", 6, 6, 15, "workload"),
    Scale("coupled_7x7_n18", "coupled_size_workload", 7, 7, 18, "coupled"),
    Scale("coupled_8x8_n26", "coupled_size_workload", 8, 8, 26, "coupled"),
)
SCALE_BY_ID = {scale.scale_id: scale for scale in SCALES}


def _canonical(value: Mapping) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Mapping, *, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise D10Error(f"missing regular file: {path}")
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
        raise D10Error(f"missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise D10Error(f"{label} must contain an object")
    return value


def _contract() -> dict:
    # D10 uses only the final conditioned controller. Authenticating GA,
    # Kim2020, PSLAP, and historical VCG inputs here would add unrelated work.
    auth = _conditioned_auth()
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "PSLAP/viability.py",
        PROJECT_ROOT / "PSLAP/viability_candidates.py",
        PROJECT_ROOT / "PSLAP/viability_filter.py",
        PROJECT_ROOT / "example/small_rooms_env.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "vcg_v11_conditioned_handling.py",
    )
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "role": "development_support_and_runtime_screen_not_paper_confirmation",
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "instance_seeds": list(INSTANCE_SEEDS),
        "pilot_instance_limit": PILOT_INSTANCE_LIMIT,
        "scales": [scale.public_dict() for scale in SCALES],
        "search_max_nodes": SEARCH_MAX_NODES,
        "max_steps": MAX_STEPS,
        "primary_deadline_seconds": PRIMARY_DEADLINE_SECONDS,
        "deadline_sensitivity_seconds": list(DEADLINE_SECONDS),
        "zero_shot_no_training": True,
        "weights_and_normalization_frozen": True,
        "exact_checker_and_candidate_generator_frozen": True,
        "same_arrival_rate_and_proc_mean": True,
        "three_exit_count_fixed_across_geometry": True,
        "simulated_steps_separate_from_computation_time": True,
        "quality_metrics_suppressed_on_incomplete_runs": True,
        "timing_instrumentation": {
            "candidate_plus_overhead": "frontier_wall_minus_exact_search",
            "certification": "sum_exact_recoverability_calls",
            "representation": "graph_encoder_plus_action_encoder_forward_hooks",
            "qop": "operational_q_head_forward_hook",
            "qn": "conditioned_handling_network_forward_hook",
            "selection_residual": "selection_wall_minus_instrumented_neural_components",
            "end_to_end_decision": "frontier_wall_plus_selection_wall",
            "no_double_counting": True,
        },
        "certification_call_distribution_is_cache_misses_only": True,
        "candidate_unknown_rate_denominator": "all_certified_candidate_subjects_with_SAFE_UNSAFE_or_UNKNOWN_status;defer_excluded",
        "conditioned_terminal_sha256": auth["terminal_sha256"][MODEL_SEED],
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in sources
        },
    }
    return _with_hash(semantic, "contract_sha256")


def _instance_path(output: Path, scale: Scale, seed: int) -> Path:
    return output / "episode-instances" / scale.scale_id / f"seed-{seed}.json"


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = _contract()
    contract_path = output / CONTRACT_NAME
    if contract_path.is_file():
        observed = _load_json(contract_path, "D10 contract")
        if observed != expected or observed.get("contract_sha256") != _digest(observed, field="contract_sha256"):
            raise D10Error("D10 contract, checkpoints, or sources changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise D10Error("nonempty D10 output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(contract_path, expected)
    records = []
    for scale in SCALES:
        env = scale.make_env()
        for index, seed in enumerate(INSTANCE_SEEDS):
            expected_instance = env.sample_episode_instance(seed)
            expected_instance.validate_for(env)
            path = _instance_path(output, scale, seed)
            if path.is_file():
                instance = EpisodeInstance.from_json(path.read_text())
                if instance != expected_instance:
                    raise D10Error(f"D10 instance changed: {scale.scale_id}/{seed}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected_instance.to_json())
                instance = EpisodeInstance.from_json(path.read_text())
            records.append({
                "scale_id": scale.scale_id,
                "axis": scale.axis,
                "seed": seed,
                "instance_index": index,
                "relative_path": str(path.relative_to(output)),
                "raw_sha256": _sha(path),
                "episode_instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
            })
    manifest = _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": expected["contract_sha256"],
        "records": records,
    }, "manifest_sha256")
    manifest_path = output / MANIFEST_NAME
    if manifest_path.is_file():
        if _load_json(manifest_path, "D10 manifest") != manifest:
            raise D10Error("D10 manifest changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {"status": "prepared", "scales": len(SCALES), "instances": len(records), "pilot_rollouts": len(SCALES)}


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = _load_json(output / CONTRACT_NAME, "D10 contract")
    if contract != _contract() or contract.get("contract_sha256") != _digest(contract, field="contract_sha256"):
        raise D10Error("D10 contract authentication failed")
    manifest = _load_json(output / MANIFEST_NAME, "D10 manifest")
    if manifest.get("manifest_sha256") != _digest(manifest, field="manifest_sha256") or manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise D10Error("D10 manifest authentication failed")
    return contract, manifest


def _rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


class SelectionTimer:
    """Measure frozen neural components through hooks without changing choice."""

    def __init__(self, fixed_agent) -> None:
        self.fixed_agent = fixed_agent
        self.config = fixed_agent.config
        self.records: list[dict] = []
        self.active: Optional[dict] = None
        conditioned = fixed_agent.agent
        self.handles = []
        for module, name in (
            (conditioned.base_agent.Q_local.graph_encoder, "graph_encoding_seconds"),
            (conditioned.base_agent.Q_local.action_encoder, "action_encoding_seconds"),
            (conditioned.base_agent.Q_local.q_head, "qop_seconds"),
            (conditioned.handling_network, "qn_seconds"),
        ):
            starts = []

            def before(_module, _inputs, starts=starts):
                starts.append(perf_counter())

            def after(_module, _inputs, _output, starts=starts, name=name):
                started = starts.pop()
                if self.active is not None:
                    self.active[name] += perf_counter() - started

            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))

    def reset_episode_state(self):
        return self.fixed_agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        record = {
            "graph_encoding_seconds": 0.0,
            "action_encoding_seconds": 0.0,
            "qop_seconds": 0.0,
            "qn_seconds": 0.0,
        }
        self.active = record
        started = perf_counter()
        try:
            decision = self.fixed_agent.select(snapshot, training=training, epsilon=epsilon)
        finally:
            record["selection_wall_seconds"] = perf_counter() - started
            self.active = None
        record["representation_seconds"] = record["graph_encoding_seconds"] + record["action_encoding_seconds"]
        record["selection_residual_seconds"] = max(
            record["selection_wall_seconds"]
            - record["representation_seconds"]
            - record["qop_seconds"]
            - record["qn_seconds"],
            0.0,
        )
        self.records.append(record)
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.fixed_agent.observe_outcome(decision, next_snapshot=next_snapshot, done=done)


class FrontierInstrumentation:
    def __init__(self) -> None:
        self.certificate_calls: list[dict] = []
        self.legal_action_batches: list[tuple] = []

    @contextmanager
    def active(self):
        core_analyze = candidate_module.analyze_recoverability
        core_legal = candidate_module.legal_recovery_actions

        def timed_analyze(state, *args, **kwargs):
            started = perf_counter()
            certificate = core_analyze(state, *args, **kwargs)
            self.certificate_calls.append({
                "status": certificate.status.value,
                "explored_nodes": int(certificate.explored_nodes),
                "generated_states": int(certificate.generated_states),
                "exhaustive": bool(certificate.exhaustive),
                "seconds": perf_counter() - started,
                "stored_blocks": len(state.blocks),
                "storage_cells": len(state.storage_cells),
            })
            return certificate

        def recorded_legal(state):
            actions = tuple(core_legal(state))
            self.legal_action_batches.append(actions)
            return actions

        original_enumerate = benchmark._enumerate_frontier

        def enriched_enumerate(*args, **kwargs):
            call_start = len(self.certificate_calls)
            legal_start = len(self.legal_action_batches)
            snapshot, record = original_enumerate(*args, **kwargs)
            audit = snapshot.audit.audit_dict()
            calls = self.certificate_calls[call_start:]
            batches = self.legal_action_batches[legal_start:]
            legal = batches[-1] if batches else ()
            kinds = Counter(action.kind.value for action in legal)
            record.update({
                "physical_candidate_count": int(audit["physical_accept_count"]) + int(audit["legal_recovery_count"]) + int(bool(audit["defer_allowed"])),
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
                "state_contains_unknown": bool(audit["unknown_accept_count"] or audit["unknown_recovery_count"]),
                "no_positively_certified_candidate": int(audit["candidate_count"]) == 0,
                "candidate_generation_and_overhead_seconds": max(float(record["total_frontier_seconds"]) - float(record["exact_search_seconds"]), 0.0),
                "certification_calls": calls,
            })
            return snapshot, record

        with ExitStack() as stack:
            stack.enter_context(patch.object(candidate_module, "analyze_recoverability", timed_analyze))
            stack.enter_context(patch.object(filter_module, "analyze_recoverability", timed_analyze))
            stack.enter_context(patch.object(candidate_module, "legal_recovery_actions", recorded_legal))
            stack.enter_context(patch.object(benchmark, "_enumerate_frontier", enriched_enumerate))
            yield


@contextmanager
def _runtime_environment(scale: Scale):
    with patch.object(benchmark, "_make_env", lambda _payload: scale.make_env()):
        yield


def _load_instance(output: Path, record: Mapping) -> EpisodeInstance:
    path = output / record["relative_path"]
    if _sha(path) != record["raw_sha256"]:
        raise D10Error("D10 instance bytes changed")
    instance = EpisodeInstance.from_json(path.read_text())
    instance.validate_for(SCALE_BY_ID[record["scale_id"]].make_env())
    return instance


def _run_one(output: Path, contract: Mapping, manifest: Mapping, record: Mapping, *, device: torch.device) -> dict:
    conditioned = _conditioned_auth()
    auth = {"conditioned": conditioned}
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    instance = _load_instance(output, record)
    scale = SCALE_BY_ID[record["scale_id"]]
    holder = []

    def factory(base):
        agent = final90._load_conditioned_agent(PROJECT_ROOT, auth, model_seed=MODEL_SEED, base=base, device=device)
        agent.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(agent, PREFERENCE_LAMBDA)
        timed = SelectionTimer(fixed)
        holder.append(timed)
        return timed

    recorder = FrontierInstrumentation()
    rss_before = _rss_bytes()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with recorder.active(), _runtime_environment(scale):
        raw = pilot._run_raw(arm, instance, device=device, wrapper_factory=factory)
    rss_after = _rss_bytes()
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if len(holder) != 1 or len(holder[0].records) != len(raw["decisions"]):
        raise D10Error("selection instrumentation did not align with decisions")
    selection_records = holder[0].records
    for decision, timing in zip(raw["decisions"], selection_records):
        timing["executor_selection_seconds"] = float(decision["selection_seconds"])
    decision_costs = []
    for decision, timing in zip(raw["decisions"], selection_records):
        frontier = raw["frontiers"][int(decision["frontier_index"])]
        decision_costs.append({
            **timing,
            "frontier_seconds": float(frontier["total_frontier_seconds"]),
            "candidate_generation_and_overhead_seconds": float(frontier["candidate_generation_and_overhead_seconds"]),
            "certification_seconds": float(frontier["exact_search_seconds"]),
            "end_to_end_decision_seconds": float(frontier["total_frontier_seconds"]) + float(timing["selection_wall_seconds"]),
            "cold_first_decision": len(decision_costs) == 0,
            "liveness_forced": bool(decision["liveness_forced"]),
        })
    strict = bool(
        raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and raw["illegal_drops"] == 0
        and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == scale.blocks
    )
    timing = _timing(
        raw["delivery_deviations"], expected_deliveries=scale.blocks
    ) if strict else {
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
    }
    rehandles = int(raw.get("physical_storage_relocations", raw.get("relocations", 0)))
    row = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scale": scale.public_dict(),
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "strict_safe_complete": strict,
        "method_failure_reason": raw["method_failure_reason"],
        "dense_return": float(raw["return"]) if strict else None,
        **timing,
        "steps": int(raw["steps"]) if strict else None,
        "steps_per_delivery": float(raw["steps"] / scale.blocks) if strict else None,
        "physical_rehandles": rehandles if strict else None,
        "physical_rehandles_per_100_required_deliveries": 100.0 * rehandles / scale.blocks if strict else None,
        "observed_steps_to_stop": int(raw["steps"]),
        "deliveries_at_stop": len(raw["delivery_deviations"]),
        "macro_decisions": int(raw["macro_decisions"]),
        "all_selected_candidates_exact_safe": bool(raw["complete_frontier_exactly_verified"]),
        "frontiers": raw["frontiers"],
        "decision_costs": decision_costs,
        "episode_wall_seconds": float(raw["episode_wall_seconds"]),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        "process_peak_rss_before_kib": peak_before,
        "process_peak_rss_after_kib": peak_after,
        "behavior_digest": raw["behavior_digest"],
    }
    return _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "conditioned_terminal_sha256": contract["conditioned_terminal_sha256"],
        "row": row,
    }, "ledger_sha256")


def _ledger_path(output: Path, record: Mapping) -> Path:
    return output / "run-ledger" / record["scale_id"] / f"seed-{record['seed']}.json"


def run(output: Path, *, scale_ids: Sequence[str], instance_limit: int, device_name: str) -> dict:
    contract, manifest = authenticate(output)
    unknown = set(scale_ids) - set(SCALE_BY_ID)
    if unknown:
        raise D10Error(f"unknown scales: {sorted(unknown)}")
    if not 1 <= instance_limit <= len(INSTANCE_SEEDS):
        raise D10Error("invalid instance limit")
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise D10Error("D10 hardware timings are frozen to CPU")
    records = [record for record in manifest["records"] if record["scale_id"] in scale_ids and int(record["instance_index"]) < instance_limit]
    completed = safe = 0
    for record in records:
        path = _ledger_path(output, record)
        if path.is_file():
            ledger = _load_json(path, "D10 ledger")
            if ledger.get("ledger_sha256") != _digest(ledger, field="ledger_sha256") or ledger.get("contract_sha256") != contract["contract_sha256"]:
                raise D10Error("D10 ledger authentication failed")
        else:
            ledger = _run_one(output, contract, manifest, record, device=device)
            _atomic_json(path, ledger)
        completed += 1
        safe += int(ledger["row"]["strict_safe_complete"])
        print(f"D10 {completed}/{len(records)} | {record['scale_id']} seed={record['seed']} | strict={int(ledger['row']['strict_safe_complete'])} | wall={ledger['row']['episode_wall_seconds']:.2f}s", flush=True)
    return {"status": "complete", "rows": completed, "strict_safe_complete": safe}


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return None if not values else float(np.percentile(np.asarray(values, dtype=float), q))


def _distribution(values: Sequence[float]) -> dict:
    values = tuple(float(value) for value in values)
    return {
        "n": len(values),
        "mean": fmean(values) if values else None,
        "median": median(values) if values else None,
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "maximum": max(values) if values else None,
    }


def _timing(deviations: Sequence[float], *, expected_deliveries: int) -> dict:
    """Compute E1-compatible timing metrics for any declared workload size."""
    values = tuple(float(value) for value in deviations)
    if (
        expected_deliveries <= 0
        or len(values) != expected_deliveries
        or not all(math.isfinite(value) for value in values)
    ):
        raise D10Error(
            f"complete D10 row needs {expected_deliveries} finite timing deviations"
        )
    return {
        "mean_signed_deviation": float(fmean(values)),
        "mean_absolute_error": float(fmean(abs(value) for value in values)),
        "mean_tardiness": float(fmean(max(value, 0.0) for value in values)),
        "mean_earliness": float(fmean(max(-value, 0.0) for value in values)),
        "within_target_window_rate": float(
            fmean(abs(value) <= final90.TARGET_WINDOW for value in values)
        ),
    }


def analyze(output: Path, *, allow_partial: bool) -> dict:
    contract, _manifest = authenticate(output)
    ledgers = []
    for path in sorted((output / "run-ledger").glob("**/seed-*.json")):
        value = _load_json(path, "D10 ledger")
        if value.get("ledger_sha256") != _digest(value, field="ledger_sha256"):
            raise D10Error("D10 ledger self hash mismatch")
        ledgers.append(value["row"])
    expected = len(SCALES) * len(INSTANCE_SEEDS)
    if not allow_partial and len(ledgers) != expected:
        raise D10Error(f"expected {expected} rows, found {len(ledgers)}")
    groups = defaultdict(list)
    for row in ledgers:
        groups[row["scale"]["scale_id"]].append(row)
    summaries = []
    for scale_id, rows in sorted(groups.items()):
        frontiers = [frontier for row in rows for frontier in row["frontiers"]]
        calls = [call for frontier in frontiers for call in frontier["certification_calls"]]
        decisions = [decision for row in rows for decision in row["decision_costs"]]
        strict = [row for row in rows if row["strict_safe_complete"]]
        candidate_subjects = sum(
            int(f["certified_accept_count"])
            + int(f["certified_deliver_count"])
            + int(f["certified_reconfigure_count"])
            + int(f["unknown_candidate_count"])
            + int(f["unsafe_candidate_count"])
            for f in frontiers
        )
        unknown = sum(int(f["unknown_candidate_count"]) for f in frontiers)
        summary = {
            "scale": SCALE_BY_ID[scale_id].public_dict(),
            "rows": len(rows),
            "strict_safe_complete": len(strict),
            "strict_completion_rate": len(strict) / len(rows),
            "quality_metrics_suppressed": len(strict) != len(rows),
            "mean_absolute_error": fmean(row["mean_absolute_error"] for row in strict) if len(strict) == len(rows) else None,
            "physical_rehandles_per_100_required_deliveries": fmean(row["physical_rehandles_per_100_required_deliveries"] for row in strict) if len(strict) == len(rows) else None,
            "steps_per_delivery": fmean(row["steps_per_delivery"] for row in strict) if len(strict) == len(rows) else None,
            "within_target_window_rate": fmean(row["within_target_window_rate"] for row in strict) if len(strict) == len(rows) else None,
            "dense_return": fmean(row["dense_return"] for row in strict) if len(strict) == len(rows) else None,
            "physical_candidates_per_state": _distribution([f["physical_candidate_count"] for f in frontiers]),
            "certified_candidates_per_state": _distribution([f["certified_candidate_count"] for f in frontiers]),
            "candidate_unknown_rate": unknown / candidate_subjects if candidate_subjects else 0.0,
            "state_contains_unknown_rate": sum(bool(f["state_contains_unknown"]) for f in frontiers) / len(frontiers) if frontiers else 0.0,
            "empty_certified_frontier_rate": sum(bool(f["no_positively_certified_candidate"]) for f in frontiers) / len(frontiers) if frontiers else 0.0,
            "certification_expanded_nodes_per_cache_miss": _distribution([call["explored_nodes"] for call in calls]),
            "certification_seconds_per_cache_miss": _distribution([call["seconds"] for call in calls]),
            "candidate_generation_and_overhead_seconds": _distribution([f["candidate_generation_and_overhead_seconds"] for f in frontiers]),
            "frontier_seconds": _distribution([f["total_frontier_seconds"] for f in frontiers]),
            "representation_seconds": _distribution([d["representation_seconds"] for d in decisions]),
            "qop_seconds": _distribution([d["qop_seconds"] for d in decisions]),
            "qn_seconds": _distribution([d["qn_seconds"] for d in decisions]),
            "selection_residual_seconds": _distribution([d["selection_residual_seconds"] for d in decisions]),
            "end_to_end_decision_seconds": _distribution([d["end_to_end_decision_seconds"] for d in decisions]),
            "deadline_exceedance_rate": {
                str(deadline): sum(d["end_to_end_decision_seconds"] > deadline for d in decisions) / len(decisions) if decisions else None
                for deadline in DEADLINE_SECONDS
            },
            "liveness_intervention_rate": sum(bool(d["liveness_forced"]) for d in decisions) / len(decisions) if decisions else None,
            "fallback_installed": False,
            "rss_after_bytes": _distribution([row["rss_after_bytes"] for row in rows]),
        }
        summaries.append(summary)
    report = _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "partial": len(ledgers) != expected,
        "expected_rows": expected,
        "observed_rows": len(ledgers),
        "summaries": summaries,
        "interpretation_scope": "development support and runtime screen only",
    }, "report_sha256")
    _atomic_json(output / REPORT_NAME, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "run-pilot", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale", action="append", choices=tuple(SCALE_BY_ID))
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command in ("run", "run-pilot"):
        prepare(output)
        result = run(
            output,
            scale_ids=tuple(args.scale or SCALE_BY_ID),
            instance_limit=(args.instance_limit or (PILOT_INSTANCE_LIMIT if args.command == "run-pilot" else len(INSTANCE_SEEDS))),
            device_name=args.device,
        )
    else:
        result = analyze(output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
