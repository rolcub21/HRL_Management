#!/usr/bin/env python3
"""D10 occupancy/10x10 extension for the Block 6 scalability design."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
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
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    run as d10,
)
import PSLAP.viability_candidates as candidate_module
import PSLAP.viability_filter as filter_module
from PSLAP.viability import RecoveryActionKind
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_d10_occupancy_10x10_support_95k_v1"
SCHEMA_VERSION = 1
MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
INSTANCE_SEEDS = (95_100, 95_101, 95_102)
PILOT_INSTANCE_LIMIT = 1
PROC_MEAN = 80
EXTRA_ARRIVAL_FIRST = 20
EXTRA_ARRIVAL_GAP = 10
MIN_MAX_STEPS = 2_000
MAX_STEPS_PER_JOB = 100
PRIMARY_DEADLINE_SECONDS = 1.0
DEADLINES = (0.1, 1.0, 5.0)
CONTRACT_NAME = "d10-occupancy-contract.json"
MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "d10-occupancy-report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-d10-occupancy-extension-95k"
_AUTH_CACHE = None


class OccupancyD10Error(RuntimeError):
    pass


def _conditioned_auth():
    global _AUTH_CACHE
    if _AUTH_CACHE is None:
        _AUTH_CACHE = final90._authenticate_conditioned(PROJECT_ROOT)
    return _AUTH_CACHE


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    comparison: str
    occupancy_level: str
    rows: int
    cols: int
    initial_occupied_slots: int
    total_jobs: int

    @property
    def door(self) -> tuple[int, int]:
        return (0, self.cols - 2)

    @property
    def exits(self) -> tuple[tuple[int, int], ...]:
        right = self.cols - 2
        return tuple((self.rows - 1, col) for col in range(right - 2, right + 1))

    def make_base_env(self) -> SmallRoomsEnv:
        return SmallRoomsEnv(
            grid_rows=self.rows,
            grid_cols=self.cols,
            start_state=(1, 1),
            door_cell=self.door,
            pickup_cells=(1, self.cols - 2),
            exit_cells=list(self.exits),
            number_blocks=self.total_jobs,
            choose_storage=False,
            arrival_rate=1.0 / EXTRA_ARRIVAL_GAP,
            proc_mean=PROC_MEAN,
        )

    @property
    def storage_capacity(self) -> int:
        return len(self.make_base_env().storage_positions)

    @property
    def placement_cells(self) -> tuple[tuple[int, int], ...]:
        env = self.make_base_env()
        # Fill columns from left to right while keeping the right-hand access
        # corridor open as long as possible. The robot start is never occupied.
        eligible = sorted(
            (cell for cell in env.storage_positions if cell != env.start_state),
            key=lambda cell: (cell[1], cell[0]),
        )
        return tuple(eligible[: self.initial_occupied_slots])

    @property
    def max_steps(self) -> int:
        return max(MIN_MAX_STEPS, MAX_STEPS_PER_JOB * self.total_jobs)

    def public_dict(self) -> dict:
        env = self.make_base_env()
        capacity = len(env.storage_positions)
        return {
            **asdict(self),
            "yard_cells": self.rows * self.cols,
            "storage_capacity": capacity,
            "initial_occupancy_ratio": self.initial_occupied_slots / capacity,
            "turnover_jobs": self.total_jobs - self.initial_occupied_slots,
            "nominal_total_jobs_to_capacity": self.total_jobs / capacity,
            "door_cell": list(env.door_cell),
            "pickup_cell": list(env.pickup_cell),
            "exit_cells": [list(cell) for cell in env.exit_cells],
            "initial_placement_cells": [list(cell) for cell in self.placement_cells],
            "max_steps": self.max_steps,
        }


SCENARIOS = (
    Scenario("size_5x5_occ_low", "size_by_occupancy", "low", 5, 5, 2, 4),
    Scenario("size_5x5_occ_medium", "size_by_occupancy", "medium", 5, 5, 4, 6),
    Scenario("size_5x5_occ_high", "size_by_occupancy", "high", 5, 5, 6, 8),
    Scenario("size_8x8_occ_low", "size_by_occupancy", "low", 8, 8, 11, 13),
    Scenario("size_8x8_occ_medium", "size_by_occupancy", "medium", 8, 8, 18, 20),
    Scenario("size_8x8_occ_high", "size_by_occupancy", "high", 8, 8, 25, 27),
    Scenario("size_10x10_occ_low", "size_by_occupancy", "low", 10, 10, 19, 21),
    Scenario("size_10x10_occ_medium", "size_by_occupancy", "medium", 10, 10, 32, 34),
    Scenario("size_10x10_occ_high", "size_by_occupancy", "high", 10, 10, 44, 46),
    Scenario("fixed_workload_8x8_k6_n8", "geometry_fixed_workload", "not_applicable", 8, 8, 6, 8),
    Scenario("fixed_workload_10x10_k6_n8", "geometry_fixed_workload", "not_applicable", 10, 10, 6, 8),
    Scenario("episode_8x8_medium_n36", "episode_length", "medium", 8, 8, 18, 36),
    Scenario("episode_8x8_medium_n54", "episode_length", "medium", 8, 8, 18, 54),
    Scenario("rectangle_6x10_medium", "aspect_ratio", "medium", 6, 10, 16, 18),
    Scenario("rectangle_10x6_medium", "aspect_ratio", "medium", 10, 6, 16, 18),
)
SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
MAIN_GRID_IDS = tuple(
    scenario.scenario_id
    for scenario in SCENARIOS
    if scenario.comparison == "size_by_occupancy"
)
CORE_10X10_IDS = tuple(
    scenario.scenario_id
    for scenario in SCENARIOS
    if scenario.comparison == "size_by_occupancy" and scenario.rows == 10
)


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
        raise OccupancyD10Error(f"missing regular file: {path}")
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
        raise OccupancyD10Error(f"missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise OccupancyD10Error(f"{label} must contain an object")
    return value


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
    }


def _validate_scenario(scenario: Scenario) -> None:
    capacity = scenario.storage_capacity
    if not 0 < scenario.initial_occupied_slots < capacity:
        raise OccupancyD10Error(f"invalid initial occupancy: {scenario.scenario_id}")
    if scenario.total_jobs < scenario.initial_occupied_slots:
        raise OccupancyD10Error(f"total jobs below initial workload: {scenario.scenario_id}")
    placements = scenario.placement_cells
    if len(placements) != scenario.initial_occupied_slots or len(set(placements)) != len(placements):
        raise OccupancyD10Error(f"invalid placement set: {scenario.scenario_id}")
    env = scenario.make_base_env()
    if env.start_state in placements or not set(placements).issubset(env.storage_positions):
        raise OccupancyD10Error(f"placement outside usable storage: {scenario.scenario_id}")


def _instance(scenario: Scenario, seed: int) -> EpisodeInstance:
    env = scenario.make_base_env()
    rng = np.random.default_rng(seed)
    durations = np.maximum(rng.poisson(PROC_MEAN, size=scenario.total_jobs), 1)
    extra = scenario.total_jobs - scenario.initial_occupied_slots
    arrivals = [0] * scenario.initial_occupied_slots + [
        EXTRA_ARRIVAL_FIRST + EXTRA_ARRIVAL_GAP * index for index in range(extra)
    ]
    instance = EpisodeInstance(
        schema_version=EpisodeInstance.SCHEMA_VERSION,
        seed=seed,
        arrival_rate=float(env.arrival_rate),
        proc_mean=float(env.proc_mean),
        arrival_steps=tuple(arrivals),
        storage_steps_needed=tuple(int(value) for value in durations),
        grid_rows=env.grid_rows,
        grid_cols=env.grid_cols,
        start_state=tuple(env.start_state),
        door_cell=tuple(env.door_cell),
        pickup_cell=tuple(env.pickup_cell),
        waiting_cell=tuple(env.waiting_cell),
        exit_cells=tuple(env.exit_cells),
        storage_positions=tuple(env.storage_positions),
        room_rows=tuple("".join(row) for row in env.rooms),
    )
    instance.validate_for(env)
    return instance


def _contract() -> dict:
    auth = _conditioned_auth()
    sources = (
        Path(__file__).resolve(),
        Path(d10.__file__).resolve(),
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "PSLAP/viability.py",
        PROJECT_ROOT / "PSLAP/viability_candidates.py",
        PROJECT_ROOT / "PSLAP/viability_filter.py",
        PROJECT_ROOT / "example/episode_instance.py",
        PROJECT_ROOT / "example/small_rooms_env.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "vcg_v11_conditioned_handling.py",
    )
    for scenario in SCENARIOS:
        _validate_scenario(scenario)
    return _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "role": "development_occupancy_and_10x10_support_not_paper_confirmation",
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "instance_seeds": list(INSTANCE_SEEDS),
        "pilot_instance_limit": PILOT_INSTANCE_LIMIT,
        "scenarios": [scenario.public_dict() for scenario in SCENARIOS],
        "occupancy_denominator": "usable_storage_capacity=len(env.storage_positions)",
        "occupancy_measurement": "initial_and_every_primitive_simulator_step",
        "time_weighted_mean": "left_continuous_unit_step_integral",
        "warm_start": "first_K_blocks_stored_at_t0_in_declared_unique_storage_cells",
        "workload_fields_separate": True,
        "extra_arrival_first": EXTRA_ARRIVAL_FIRST,
        "extra_arrival_gap": EXTRA_ARRIVAL_GAP,
        "proc_mean": PROC_MEAN,
        "max_steps_rule": f"max({MIN_MAX_STEPS},{MAX_STEPS_PER_JOB}*total_jobs)",
        "strict_unknown_fail_closed": True,
        "zero_shot_no_training": True,
        "latency_development_only": True,
        "primary_deadline_seconds": PRIMARY_DEADLINE_SECONDS,
        "deadline_sensitivity_seconds": list(DEADLINES),
        "conditioned_terminal_sha256": auth["terminal_sha256"][MODEL_SEED],
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in sources
        },
    }, "contract_sha256")


def _instance_path(output: Path, scenario_id: str, seed: int) -> Path:
    return output / "episode-instances" / scenario_id / f"seed-{seed}.json"


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected_contract = _contract()
    contract_path = output / CONTRACT_NAME
    if contract_path.is_file():
        observed = _load_json(contract_path, "occupancy contract")
        if observed != expected_contract or observed.get("contract_sha256") != _digest(observed, field="contract_sha256"):
            raise OccupancyD10Error("occupancy contract, checkpoint, or sources changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise OccupancyD10Error("nonempty occupancy output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(contract_path, expected_contract)
    records = []
    for scenario in SCENARIOS:
        for index, seed in enumerate(INSTANCE_SEEDS):
            expected = _instance(scenario, seed)
            path = _instance_path(output, scenario.scenario_id, seed)
            if path.is_file():
                observed = EpisodeInstance.from_json(path.read_text())
                if observed != expected:
                    raise OccupancyD10Error(f"instance changed: {scenario.scenario_id}/{seed}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected.to_json() + "\n")
            records.append({
                "scenario_id": scenario.scenario_id,
                "comparison": scenario.comparison,
                "occupancy_level": scenario.occupancy_level,
                "seed": seed,
                "instance_index": index,
                "relative_path": str(path.relative_to(output)),
                "raw_sha256": _sha(path),
                "episode_instance_id": expected.instance_id,
                "schedule_id": expected.schedule_id,
            })
    manifest = _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": expected_contract["contract_sha256"],
        "records": records,
    }, "manifest_sha256")
    manifest_path = output / MANIFEST_NAME
    if manifest_path.is_file():
        if _load_json(manifest_path, "occupancy manifest") != manifest:
            raise OccupancyD10Error("occupancy manifest changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "scenarios": len(SCENARIOS),
        "instances": len(records),
        "core_10x10_pilot_rollouts": len(CORE_10X10_IDS),
        "all_coordinate_pilot_rollouts": len(SCENARIOS),
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = _load_json(output / CONTRACT_NAME, "occupancy contract")
    if contract != _contract() or contract.get("contract_sha256") != _digest(contract, field="contract_sha256"):
        raise OccupancyD10Error("occupancy contract authentication failed")
    manifest = _load_json(output / MANIFEST_NAME, "occupancy manifest")
    if manifest.get("manifest_sha256") != _digest(manifest, field="manifest_sha256") or manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise OccupancyD10Error("occupancy manifest authentication failed")
    return contract, manifest


class OccupancyTrackingEnv(SmallRoomsEnv):
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.occupancy_trace: list[dict] = []
        base = scenario.make_base_env()
        super().__init__(
            grid_rows=base.grid_rows,
            grid_cols=base.grid_cols,
            start_state=base.start_state,
            door_cell=base.door_cell,
            pickup_cells=base.pickup_cell,
            exit_cells=list(base.exit_cells),
            number_blocks=base.number_blocks,
            choose_storage=False,
            arrival_rate=base.arrival_rate,
            proc_mean=base.proc_mean,
        )

    def _measurement(self) -> dict:
        storage = set(self.storage_positions)
        occupied = sum(
            not block.delivered
            and not block.carrying
            and block.position in storage
            for block in self.blocks
        )
        admitted = sum(
            block.stored_time_step is not None and not block.delivered
            for block in self.blocks
        )
        present = sum(block.position is not None and not block.delivered for block in self.blocks)
        arrived_unadmitted = sum(
            block.arrival_step <= self.time_steps
            and block.stored_time_step is None
            and not block.delivered
            for block in self.blocks
        )
        return {
            "time_step": int(self.time_steps),
            "occupied_storage_slots": int(occupied),
            "storage_occupancy_ratio": occupied / len(storage),
            "concurrent_admitted_workload": int(admitted),
            "physical_blocks_present": int(present),
            "arrived_unadmitted": int(arrived_unadmitted),
        }

    def reset(self, instance: EpisodeInstance | None = None):
        super().reset(instance=instance)
        placements = self.scenario.placement_cells
        for index, block in enumerate(self.blocks):
            if index < len(placements):
                cell = placements[index]
                block.position = cell
                block.storage_location = cell
                block.carrying = False
                block.stored = True
                block.delivered = False
                block.picked = True
                block.stored_time_step = 0
                block.storage_steps_elapsed = 0
                self.storage_counts[cell] = 1
            else:
                block.position = None
                block.storage_location = None
                block.carrying = False
                block.stored = False
                block.delivered = False
                block.picked = False
                block.stored_time_step = None
                block.storage_steps_elapsed = 0
        self.update_pickup_queue()
        self.occupancy_trace = [self._measurement()]
        initial = self.occupancy_trace[0]
        if initial["occupied_storage_slots"] != self.scenario.initial_occupied_slots:
            raise OccupancyD10Error("warm-start occupancy mismatch")
        return self.get_current_state()

    def step(self, action):
        result = super().step(action)
        self.occupancy_trace.append(self._measurement())
        return result

    def occupancy_summary(self) -> dict:
        if not self.occupancy_trace:
            raise OccupancyD10Error("occupancy tracker has no samples")
        interval_samples = self.occupancy_trace[:-1] or self.occupancy_trace
        capacity = len(self.storage_positions)
        return {
            "usable_storage_capacity": capacity,
            "initial_occupied_storage_slots": self.occupancy_trace[0]["occupied_storage_slots"],
            "initial_storage_occupancy_ratio": self.occupancy_trace[0]["storage_occupancy_ratio"],
            "time_weighted_mean_occupied_storage_slots": fmean(item["occupied_storage_slots"] for item in interval_samples),
            "time_weighted_mean_storage_occupancy_ratio": fmean(item["storage_occupancy_ratio"] for item in interval_samples),
            "peak_occupied_storage_slots": max(item["occupied_storage_slots"] for item in self.occupancy_trace),
            "peak_storage_occupancy_ratio": max(item["storage_occupancy_ratio"] for item in self.occupancy_trace),
            "time_weighted_mean_admitted_workload": fmean(item["concurrent_admitted_workload"] for item in interval_samples),
            "peak_concurrent_admitted_workload": max(item["concurrent_admitted_workload"] for item in self.occupancy_trace),
            "peak_physical_blocks_present": max(item["physical_blocks_present"] for item in self.occupancy_trace),
            "peak_arrived_unadmitted": max(item["arrived_unadmitted"] for item in self.occupancy_trace),
            "primitive_step_intervals": len(interval_samples),
        }


class FrontierInstrumentation:
    """Capture complete frontier partitions and cache-miss verifier costs."""

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
                "current_recovery_status": audit["current_recovery_status"],
                "physical_candidate_count": int(audit["physical_accept_count"]) + int(audit["legal_recovery_count"]) + int(bool(audit["defer_allowed"])),
                "physical_accept_count": int(audit["physical_accept_count"]),
                "physical_deliver_count": int(kinds[RecoveryActionKind.DELIVERY.value]),
                "physical_reconfigure_count": int(kinds[RecoveryActionKind.RELOCATION.value]),
                "physical_defer_count": int(bool(audit["defer_allowed"])),
                "executable_accept_count": int(audit["executable_accept_count"]),
                "accept_executor_rejections": int(audit["accept_executor_rejections"]),
                "recovery_executor_rejections": int(audit["recovery_executor_rejections"]),
                "certified_candidate_count": int(audit["candidate_count"]),
                "certified_accept_count": int(audit["accept_candidate_count"]),
                "certified_deliver_count": int(audit["deliver_candidate_count"]),
                "certified_reconfigure_count": int(audit["reconfigure_candidate_count"]),
                "certified_defer_count": int(audit["defer_candidate_count"]),
                "safe_accept_count": int(audit["safe_accept_count"]),
                "unsafe_accept_count": int(audit["unsafe_accept_count"]),
                "unknown_accept_count": int(audit["unknown_accept_count"]),
                "safe_recovery_count": int(audit["safe_recovery_count"]),
                "unsafe_recovery_count": int(audit["unsafe_recovery_count"]),
                "unknown_recovery_count": int(audit["unknown_recovery_count"]),
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


def _load_instance(output: Path, record: Mapping) -> EpisodeInstance:
    path = output / record["relative_path"]
    if _sha(path) != record["raw_sha256"]:
        raise OccupancyD10Error("instance bytes changed")
    instance = EpisodeInstance.from_json(path.read_text())
    instance.validate_for(SCENARIO_BY_ID[record["scenario_id"]].make_base_env())
    return instance


def _rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _compact_frontier(frontier: Mapping) -> dict:
    keys = (
        "decision_epoch", "candidate_frontier_digest", "cache_hits", "cache_misses",
        "complete_frontier_exactly_verified", "current_recovery_status",
        "physical_candidate_count", "physical_accept_count", "physical_deliver_count",
        "physical_reconfigure_count", "physical_defer_count", "executable_accept_count",
        "accept_executor_rejections", "recovery_executor_rejections",
        "certified_candidate_count", "certified_accept_count", "certified_deliver_count",
        "certified_reconfigure_count", "certified_defer_count", "safe_accept_count",
        "unsafe_accept_count", "unknown_accept_count", "safe_recovery_count",
        "unsafe_recovery_count", "unknown_recovery_count", "unknown_candidate_count",
        "unsafe_candidate_count", "state_contains_unknown",
        "no_positively_certified_candidate", "candidate_generation_and_overhead_seconds",
        "exact_search_seconds", "total_frontier_seconds", "certification_calls",
    )
    return {key: frontier[key] for key in keys}


def _failure_class(row: Mapping) -> str:
    if row.get("implementation_error"):
        return "unsupported_or_implementation_error"
    if row.get("state_contains_unknown"):
        return "certificate_budget_limited"
    if row.get("strict_safe_complete"):
        return "completed"
    return "valid_but_operationally_incomplete"


def _run_one(output: Path, contract: Mapping, manifest: Mapping, record: Mapping, *, device: torch.device) -> dict:
    scenario = SCENARIO_BY_ID[record["scenario_id"]]
    instance = _load_instance(output, record)
    conditioned = _conditioned_auth()
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    agent_holder = []
    env_holder = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            PROJECT_ROOT, {"conditioned": conditioned}, model_seed=MODEL_SEED,
            base=base, device=device,
        )
        agent.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(agent, PREFERENCE_LAMBDA)
        timed = d10.SelectionTimer(fixed)
        agent_holder.append(timed)
        return timed

    def env_factory(_payload):
        env = OccupancyTrackingEnv(scenario)
        env_holder.append(env)
        return env

    recorder = FrontierInstrumentation()
    rss_before = _rss_bytes()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        with recorder.active(), patch.object(benchmark, "_make_env", env_factory), pilot._agent_factory(agent_factory):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=benchmark._search_config(arm.payload),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=device,
            )
    except Exception as error:
        row = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scenario": scenario.public_dict(),
            "instance_seed": int(record["seed"]),
            "instance_index": int(record["instance_index"]),
            "episode_instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "model_seed": MODEL_SEED,
            "preference_lambda": PREFERENCE_LAMBDA,
            "strict_safe_complete": False,
            "implementation_error": f"{type(error).__name__}: {error}",
            "method_failure_reason": None,
            "state_contains_unknown": None,
            "failure_class": "unsupported_or_implementation_error",
        }
    else:
        if len(agent_holder) != 1 or len(env_holder) != 1 or len(agent_holder[0].records) != len(raw["decisions"]):
            raise OccupancyD10Error("runtime instrumentation did not align")
        frontiers = [_compact_frontier(frontier) for frontier in raw["frontiers"]]
        decision_costs = []
        for decision, timing in zip(raw["decisions"], agent_holder[0].records):
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
            raw["strict_method_success"] and raw["terminal"]
            and raw["method_failure_reason"] is None
            and raw["complete_frontier_exactly_verified"]
            and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
            and len(raw["delivery_deviations"]) == scenario.total_jobs
        )
        timing_metrics = d10._timing(raw["delivery_deviations"], expected_deliveries=scenario.total_jobs) if strict else {
            "mean_signed_deviation": None, "mean_absolute_error": None,
            "mean_tardiness": None, "mean_earliness": None,
            "within_target_window_rate": None,
        }
        rehandles = int(raw.get("physical_storage_relocations", raw.get("relocations", 0)))
        state_unknown = any(frontier["state_contains_unknown"] for frontier in frontiers)
        row = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scenario": scenario.public_dict(),
            "instance_seed": int(record["seed"]),
            "instance_index": int(record["instance_index"]),
            "episode_instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "model_seed": MODEL_SEED,
            "preference_lambda": PREFERENCE_LAMBDA,
            "strict_safe_complete": strict,
            "implementation_error": None,
            "method_failure_reason": raw["method_failure_reason"],
            "state_contains_unknown": state_unknown,
            "dense_return": float(raw["return"]) if strict else None,
            **timing_metrics,
            "steps": int(raw["steps"]) if strict else None,
            "steps_per_delivery": float(raw["steps"] / scenario.total_jobs) if strict else None,
            "physical_rehandles": rehandles if strict else None,
            "physical_rehandles_per_100_required_deliveries": 100.0 * rehandles / scenario.total_jobs if strict else None,
            "observed_steps_to_stop": int(raw["steps"]),
            "deliveries_at_stop": len(raw["delivery_deviations"]),
            "required_deliveries": scenario.total_jobs,
            "macro_decisions": int(raw["macro_decisions"]),
            "all_selected_candidates_exact_safe": bool(raw["complete_frontier_exactly_verified"]),
            "initial_current_recovery_status": frontiers[0]["current_recovery_status"] if frontiers else None,
            "frontiers": frontiers,
            "decision_costs": decision_costs,
            "episode_wall_seconds": float(raw["episode_wall_seconds"]),
            "occupancy": env_holder[0].occupancy_summary(),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": _rss_bytes(),
            "process_peak_rss_before_kib": peak_before,
            "process_peak_rss_after_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "behavior_digest": raw["behavior_digest"],
        }
        row["failure_class"] = _failure_class(row)
    return _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "conditioned_terminal_sha256": contract["conditioned_terminal_sha256"],
        "row": row,
    }, "ledger_sha256")


def _ledger_path(output: Path, record: Mapping) -> Path:
    return output / "run-ledger" / record["scenario_id"] / f"seed-{record['seed']}.json"


def run(output: Path, *, scenario_ids: Sequence[str], instance_limit: int, device_name: str) -> dict:
    contract, manifest = authenticate(output)
    unknown = set(scenario_ids) - set(SCENARIO_BY_ID)
    if unknown:
        raise OccupancyD10Error(f"unknown scenarios: {sorted(unknown)}")
    if not 1 <= instance_limit <= len(INSTANCE_SEEDS):
        raise OccupancyD10Error("invalid instance limit")
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise OccupancyD10Error("occupancy extension timings are frozen to CPU")
    records = [
        record for record in manifest["records"]
        if record["scenario_id"] in scenario_ids
        and int(record["instance_index"]) < instance_limit
    ]
    completed = strict = errors = 0
    for record in records:
        path = _ledger_path(output, record)
        if path.is_file():
            ledger = _load_json(path, "occupancy ledger")
            if ledger.get("ledger_sha256") != _digest(ledger, field="ledger_sha256") or ledger.get("contract_sha256") != contract["contract_sha256"]:
                raise OccupancyD10Error("occupancy ledger authentication failed")
        else:
            ledger = _run_one(output, contract, manifest, record, device=device)
            _atomic_json(path, ledger)
        row = ledger["row"]
        completed += 1
        strict += int(row["strict_safe_complete"])
        errors += int(bool(row.get("implementation_error")))
        print(
            f"D10-occ {completed}/{len(records)} | {record['scenario_id']} seed={record['seed']} "
            f"| class={row['failure_class']} | strict={int(row['strict_safe_complete'])}",
            flush=True,
        )
    return {"status": "complete", "rows": completed, "strict_safe_complete": strict, "implementation_errors": errors}


def _distribution(values: Sequence[float]) -> dict:
    return d10._distribution(values)


def analyze(output: Path, *, allow_partial: bool) -> dict:
    contract, _manifest = authenticate(output)
    rows = []
    for path in sorted((output / "run-ledger").glob("**/seed-*.json")):
        ledger = _load_json(path, "occupancy ledger")
        if ledger.get("ledger_sha256") != _digest(ledger, field="ledger_sha256"):
            raise OccupancyD10Error("occupancy ledger self hash mismatch")
        rows.append(ledger["row"])
    expected = len(SCENARIOS) * len(INSTANCE_SEEDS)
    if not allow_partial and len(rows) != expected:
        raise OccupancyD10Error(f"expected {expected} rows, found {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"]["scenario_id"]].append(row)
    summaries = []
    for scenario_id, group in sorted(groups.items()):
        successful_runtime = [row for row in group if not row.get("implementation_error") and "frontiers" in row]
        strict = [row for row in group if row["strict_safe_complete"]]
        frontiers = [frontier for row in successful_runtime for frontier in row["frontiers"]]
        calls = [call for frontier in frontiers for call in frontier["certification_calls"]]
        decisions = [decision for row in successful_runtime for decision in row["decision_costs"]]
        warm_decisions = [decision for decision in decisions if not decision["cold_first_decision"]]
        candidate_subjects = sum(
            frontier["safe_accept_count"] + frontier["unsafe_accept_count"] + frontier["unknown_accept_count"]
            + frontier["safe_recovery_count"] + frontier["unsafe_recovery_count"] + frontier["unknown_recovery_count"]
            for frontier in frontiers
        )
        unknown_candidates = sum(frontier["unknown_candidate_count"] for frontier in frontiers)
        all_observed_strict = bool(group) and len(strict) == len(group)
        all_complete = len(group) == len(INSTANCE_SEEDS) and all_observed_strict
        observed_metrics = None
        if all_observed_strict:
            observed_metrics = {
                "mean_absolute_error": fmean(row["mean_absolute_error"] for row in strict),
                "physical_rehandles_per_100_required_deliveries": fmean(row["physical_rehandles_per_100_required_deliveries"] for row in strict),
                "steps_per_delivery": fmean(row["steps_per_delivery"] for row in strict),
                "within_target_window_rate": fmean(row["within_target_window_rate"] for row in strict),
            }
        summary = {
            "scenario": SCENARIO_BY_ID[scenario_id].public_dict(),
            "rows": len(group),
            "failure_classes": dict(Counter(row["failure_class"] for row in group)),
            "strict_safe_complete": len(strict),
            "strict_completion_rate": len(strict) / len(group),
            "quality_metrics_suppressed": not all_complete,
            "observed_strict_metrics": observed_metrics,
            "observed_strict_metrics_scope": (
                "complete_three-instance_coordinate"
                if all_complete
                else "descriptive_partial_development_coordinate"
                if observed_metrics is not None
                else "suppressed_due_to_observed_failure"
            ),
            "mean_absolute_error": fmean(row["mean_absolute_error"] for row in strict) if all_complete else None,
            "physical_rehandles_per_100_required_deliveries": fmean(row["physical_rehandles_per_100_required_deliveries"] for row in strict) if all_complete else None,
            "steps_per_delivery": fmean(row["steps_per_delivery"] for row in strict) if all_complete else None,
            "within_target_window_rate": fmean(row["within_target_window_rate"] for row in strict) if all_complete else None,
            "occupancy": {
                key: _distribution([row["occupancy"][key] for row in successful_runtime])
                for key in (
                    "initial_storage_occupancy_ratio",
                    "time_weighted_mean_storage_occupancy_ratio",
                    "peak_storage_occupancy_ratio",
                    "time_weighted_mean_admitted_workload",
                    "peak_concurrent_admitted_workload",
                    "peak_arrived_unadmitted",
                )
            },
            "frontier_by_action": {
                key: _distribution([frontier[key] for frontier in frontiers])
                for key in (
                    "physical_accept_count", "physical_deliver_count", "physical_reconfigure_count", "physical_defer_count",
                    "certified_accept_count", "certified_deliver_count", "certified_reconfigure_count", "certified_defer_count",
                    "physical_candidate_count", "certified_candidate_count",
                )
            },
            "candidate_unknown_rate": unknown_candidates / candidate_subjects if candidate_subjects else 0.0,
            "state_contains_unknown_rate": sum(frontier["state_contains_unknown"] for frontier in frontiers) / len(frontiers) if frontiers else None,
            "empty_certified_frontier_rate": sum(frontier["no_positively_certified_candidate"] for frontier in frontiers) / len(frontiers) if frontiers else None,
            "certification_expanded_nodes_per_cache_miss": _distribution([call["explored_nodes"] for call in calls]),
            "certification_seconds_per_cache_miss": _distribution([call["seconds"] for call in calls]),
            "candidate_generation_and_overhead_seconds": _distribution([frontier["candidate_generation_and_overhead_seconds"] for frontier in frontiers]),
            "frontier_seconds": _distribution([frontier["total_frontier_seconds"] for frontier in frontiers]),
            "representation_seconds": _distribution([decision["representation_seconds"] for decision in decisions]),
            "qop_seconds": _distribution([decision["qop_seconds"] for decision in decisions]),
            "qn_seconds": _distribution([decision["qn_seconds"] for decision in decisions]),
            "warm_end_to_end_decision_seconds": _distribution([decision["end_to_end_decision_seconds"] for decision in warm_decisions]),
            "cold_first_decision_seconds": _distribution([decision["end_to_end_decision_seconds"] for decision in decisions if decision["cold_first_decision"]]),
            "warm_deadline_exceedance_rate": {
                str(deadline): sum(decision["end_to_end_decision_seconds"] > deadline for decision in warm_decisions) / len(warm_decisions) if warm_decisions else None
                for deadline in DEADLINES
            },
            "liveness_intervention_rate": sum(decision["liveness_forced"] for decision in decisions) / len(decisions) if decisions else None,
        }
        summaries.append(summary)
    report = _with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "partial": len(rows) != expected,
        "expected_rows": expected,
        "observed_rows": len(rows),
        "hardware_at_analysis": _hardware(),
        "summaries": summaries,
        "interpretation_scope": "development occupancy and 10x10 support only",
    }, "report_sha256")
    _atomic_json(output / REPORT_NAME, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "run-core-pilot", "run-grid-pilot", "run-pilot", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", action="append", choices=tuple(SCENARIO_BY_ID))
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command in ("run", "run-core-pilot", "run-grid-pilot", "run-pilot"):
        prepare(output)
        default_ids = {
            "run-core-pilot": CORE_10X10_IDS,
            "run-grid-pilot": MAIN_GRID_IDS,
            "run-pilot": tuple(SCENARIO_BY_ID),
            "run": tuple(SCENARIO_BY_ID),
        }[args.command]
        result = run(
            output,
            scenario_ids=tuple(args.scenario or default_ids),
            instance_limit=(args.instance_limit or (PILOT_INSTANCE_LIMIT if args.command != "run" else len(INSTANCE_SEEDS))),
            device_name=args.device,
        )
    else:
        result = analyze(output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
