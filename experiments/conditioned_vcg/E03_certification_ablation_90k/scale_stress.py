#!/usr/bin/env python3
"""Prospective E3 scale-stress extension on frozen 8x8 and 10x10 episodes.

This protocol keeps the frozen seed-0 conditioned controller, hierarchical
selector, liveness guard, E14 cleanup/cache, and D12 relocation-family
certification fixed.  Within each serialized EpisodeInstance, the paired
factor is candidate eligibility only: all executable physical candidates or
only exact-SAFE candidates.

The instances deliberately impose a synchronized arrival burst whose total
workload equals usable storage capacity.  They are a designed saturation
stress test, not a representative prevalence sample.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E03_certification_ablation_90k import run as e3
from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
    run as d10,
)
from PSLAP.relocation_family_certification import (
    RELOCATION_FAMILY_CERTIFICATION,
)
from PSLAP.viability import ViabilityStatus
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e03_scale_stress_97k_v1"
SCHEMA_VERSION = 1
MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
SEARCH_MAX_NODES = 20_000
WALL_LIMIT_SECONDS = 3_600
INSTANCE_SEEDS = tuple(range(97_100, 97_110))
PILOT_SEED = INSTANCE_SEEDS[0]

PHYSICAL = e3.PHYSICAL
CERTIFIED = e3.CERTIFIED
CANDIDATE_SOURCES = (PHYSICAL, CERTIFIED)

SCENARIOS = (
    occupancy.Scenario(
        "stress_8x8_saturation_burst",
        "e3_scale_stress",
        "high_to_saturation",
        8,
        8,
        25,
        35,
    ),
    occupancy.Scenario(
        "stress_10x10_saturation_burst",
        "e3_scale_stress",
        "high_to_saturation",
        10,
        10,
        44,
        63,
    ),
)
SCENARIO_BY_ID = {item.scenario_id: item for item in SCENARIOS}
SCENARIO_IDS = tuple(SCENARIO_BY_ID)

CONTRACT_NAME = "e03-scale-stress-contract.json"
MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "e03-scale-stress-report.json"
TABLE_NAME = "e03-scale-stress-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e03-scale-stress-97k"


class ScaleStressError(RuntimeError):
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
        raise ScaleStressError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ScaleStressError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScaleStressError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ScaleStressError(f"{label} must contain an object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise ScaleStressError(f"{label} self-hash mismatch")
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


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(e3.__file__).resolve(),
        Path(e13.__file__).resolve(),
        Path(reuse.__file__).resolve(),
        Path(occupancy.__file__).resolve(),
        PROJECT_ROOT / "PSLAP/relocation_family_certification.py",
        PROJECT_ROOT / "PSLAP/viability.py",
        PROJECT_ROOT / "PSLAP/viability_candidates.py",
        PROJECT_ROOT / "PSLAP/viability_filter.py",
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "example/episode_instance.py",
        PROJECT_ROOT / "example/small_rooms_env.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _parent_inputs() -> dict:
    inputs = e13._parent_inputs()
    return {
        "d10_contract": inputs["d10_contract"],
        "d12_contract": inputs["d12_contract"],
        "d12_report": inputs["d12_report"],
        "conditioned": inputs["conditioned"],
    }


def _scenario_dict(scenario: occupancy.Scenario) -> dict:
    value = scenario.public_dict()
    value.update(
        {
            "arrival_pattern": "all_jobs_synchronized_at_t0",
            "offered_jobs_to_usable_capacity": (
                scenario.total_jobs / scenario.storage_capacity
            ),
            "stress_interpretation": (
                "initial_high_occupancy_plus_burst_to_nominal_saturation"
            ),
        }
    )
    return value


def expected_contract(output: Path) -> dict:
    parents = _parent_inputs()
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "prepared",
            "scientific_question": (
                "does_recoverability_certification_prevent_self_blocking_"
                "under_larger_high_pressure_yards"
            ),
            "paper_role": "prospective_E3_scale_stress_extension",
            "scope": (
                "designed_saturation_stress_not_representative_failure_"
                "prevalence_estimate"
            ),
            "candidate_sources": list(CANDIDATE_SOURCES),
            "only_paired_factor": "candidate_eligibility_source",
            "scenarios": [_scenario_dict(item) for item in SCENARIOS],
            "instance_seeds": list(INSTANCE_SEEDS),
            "pilot_seed": PILOT_SEED,
            "expected_pairs": len(SCENARIOS) * len(INSTANCE_SEEDS),
            "expected_runs": (
                len(SCENARIOS) * len(INSTANCE_SEEDS) * len(CANDIDATE_SOURCES)
            ),
            "model_seed": MODEL_SEED,
            "preference_lambda": PREFERENCE_LAMBDA,
            "device": "cpu",
            "training": False,
            "instance_selection_after_outcomes": False,
            "all_instances_serialized_before_rollout": True,
            "arrival_steps": "all_zero",
            "storage_duration_distribution": (
                f"Poisson(mean={occupancy.PROC_MEAN}) clipped to >=1"
            ),
            "weights_normalization_selector_and_guard_frozen": True,
            "search_config": {
                "max_depth": None,
                "max_nodes": SEARCH_MAX_NODES,
                "max_primitive_steps": None,
                "reserve_queue_cells": True,
                "search_order": "goal_directed",
            },
            "wall_limit_seconds_per_run": WALL_LIMIT_SECONDS,
            "certification_stack": {
                "path_enumeration_cleanup": "E14_order_preserving_cleanup",
                "timing_invariant_cache": True,
                "relocation_family_certification": (
                    RELOCATION_FAMILY_CERTIFICATION
                ),
                "family_miss_behavior": "native_exact_fallback",
            },
            "physical_arm_shadow_certification": {
                "eligibility_authority": False,
                "diagnostic_label": True,
                "stop_after_first_successfully_executed_non_SAFE_candidate": True,
                "UNSAFE_and_UNKNOWN_reported_separately": True,
            },
            "primary_metrics": [
                "strict_completion",
                "self_blocking",
                "selected_UNSAFE_candidate",
                "selected_UNKNOWN_candidate",
                "unsafe_admission",
                "unsafe_reconfiguration",
                "empty_certified_frontier",
                "unsafe_candidate_exposure_per_frontier",
            ],
            "secondary_metrics": [
                "dense_return",
                "mean_absolute_error",
                "within_target_window_rate",
                "steps",
                "physical_rehandles_per_100_required_deliveries",
            ],
            "secondary_metric_scope": "strict_completers_only",
            "output_dir": str(output.resolve()),
            "d10_contract_sha256": parents["d10_contract"]["contract_sha256"],
            "d12_contract_sha256": parents["d12_contract"]["contract_sha256"],
            "d12_report_sha256": parents["d12_report"]["report_sha256"],
            "conditioned_terminal_sha256": parents["conditioned"][
                "terminal_sha256"
            ][MODEL_SEED],
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def _instance(scenario: occupancy.Scenario, seed: int) -> EpisodeInstance:
    nominal = occupancy._instance(scenario, seed)
    burst = replace(
        nominal,
        arrival_steps=tuple(0 for _ in range(scenario.total_jobs)),
    )
    burst.validate_for(scenario.make_base_env())
    return burst


def _instance_path(output: Path, scenario_id: str, seed: int) -> Path:
    return output / "episode-instances" / scenario_id / f"seed-{seed}.json"


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = expected_contract(output)
    contract_path = output / CONTRACT_NAME
    if contract_path.exists():
        if _load(contract_path, "scale-stress contract") != contract:
            raise ScaleStressError(
                "scale-stress contract, parents, or sources changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ScaleStressError("nonempty output has no scale-stress contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(contract_path, contract)

    records = []
    for scenario in SCENARIOS:
        occupancy._validate_scenario(scenario)
        if scenario.total_jobs != scenario.storage_capacity:
            raise ScaleStressError("stress offered load must equal usable capacity")
        for index, seed in enumerate(INSTANCE_SEEDS):
            expected = _instance(scenario, seed)
            path = _instance_path(output, scenario.scenario_id, seed)
            if path.exists():
                observed = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
                if observed != expected:
                    raise ScaleStressError(
                        f"frozen instance changed: {scenario.scenario_id}/{seed}"
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected.to_json() + "\n", encoding="utf-8")
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "instance_seed": seed,
                    "instance_index": index,
                    "relative_path": str(path.relative_to(output)),
                    "raw_sha256": _sha(path),
                    "episode_instance_id": expected.instance_id,
                    "schedule_id": expected.schedule_id,
                }
            )
    manifest = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "records": records,
        },
        "manifest_sha256",
    )
    manifest_path = output / MANIFEST_NAME
    if manifest_path.exists():
        if _load(manifest_path, "scale-stress manifest") != manifest:
            raise ScaleStressError("scale-stress manifest changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "scenarios": len(SCENARIOS),
        "instances": len(records),
        "paired_runs": len(records) * len(CANDIDATE_SOURCES),
        "training_runs": 0,
        "output_dir": str(output),
    }


def authenticate(output: Path) -> tuple[dict, dict, dict]:
    contract = _self_hashed(
        output / CONTRACT_NAME,
        "contract_sha256",
        "scale-stress contract",
    )
    if contract != expected_contract(output):
        raise ScaleStressError(
            "scale-stress contract, parents, or sources changed"
        )
    manifest = _self_hashed(
        output / MANIFEST_NAME,
        "manifest_sha256",
        "scale-stress manifest",
    )
    if manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise ScaleStressError("manifest contract binding changed")
    records = {
        (item["scenario_id"], int(item["instance_seed"])): item
        for item in manifest["records"]
    }
    expected = {
        (scenario_id, seed)
        for scenario_id in SCENARIO_IDS
        for seed in INSTANCE_SEEDS
    }
    if set(records) != expected or len(records) != len(expected):
        raise ScaleStressError("manifest coordinate grid changed")
    return contract, manifest, _parent_inputs()


def _load_instance(output: Path, record: Mapping) -> EpisodeInstance:
    path = output / record["relative_path"]
    if _sha(path) != record["raw_sha256"]:
        raise ScaleStressError("frozen instance hash changed")
    instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
    scenario = SCENARIO_BY_ID[record["scenario_id"]]
    instance.validate_for(scenario.make_base_env())
    return instance


class ScaleTracker(e3.EpisodeTracker):
    """E3 tracker with per-frontier and action-family exposure counts."""

    def __init__(self) -> None:
        super().__init__()
        self.unsafe_frontiers = 0
        self.unknown_frontiers = 0
        self.unsafe_action_types = Counter()
        self.unknown_action_types = Counter()

    def record_frontier(self, env, physical) -> None:
        statuses = [
            e3._shadow_certificate(candidate).status
            for candidate in physical.candidates
        ]
        self.unsafe_frontiers += int(ViabilityStatus.UNSAFE in statuses)
        self.unknown_frontiers += int(ViabilityStatus.UNKNOWN in statuses)
        for candidate in physical.candidates:
            status = e3._shadow_certificate(candidate).status
            if status is ViabilityStatus.UNSAFE:
                self.unsafe_action_types[candidate.action_type.value] += 1
            elif status is ViabilityStatus.UNKNOWN:
                self.unknown_action_types[candidate.action_type.value] += 1
        super().record_frontier(env, physical)

    def audit_dict(self) -> dict:
        result = super().audit_dict()
        result.update(
            {
                "unsafe_exposure_frontiers": self.unsafe_frontiers,
                "unknown_exposure_frontiers": self.unknown_frontiers,
                "unsafe_exposed_by_action_type": dict(self.unsafe_action_types),
                "unknown_exposed_by_action_type": dict(self.unknown_action_types),
            }
        )
        return result


def _record(
    manifest: Mapping, scenario_id: str, seed: int
) -> dict:
    matches = [
        item
        for item in manifest["records"]
        if item["scenario_id"] == scenario_id
        and int(item["instance_seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ScaleStressError("manifest coordinate is not unique")
    return dict(matches[0])


def _ledger_path(
    output: Path, candidate_source: str, scenario_id: str, seed: int
) -> Path:
    return (
        output
        / "run-ledger"
        / candidate_source
        / scenario_id
        / f"seed-{seed}.json"
    )


def _timing(raw: Optional[Mapping], expected: int, strict: bool) -> dict:
    if raw is None or not strict:
        return {
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "mean_tardiness": None,
            "mean_earliness": None,
            "within_target_window_rate": None,
        }
    return d10._timing(
        raw["delivery_deviations"], expected_deliveries=expected
    )


def _run_raw(
    scenario: occupancy.Scenario,
    instance: EpisodeInstance,
    *,
    candidate_source: str,
    conditioned: Mapping,
) -> tuple[Optional[dict], ScaleTracker, Optional[occupancy.OccupancyTrackingEnv], Optional[str], Optional[str], float]:
    if candidate_source not in CANDIDATE_SOURCES:
        raise ScaleStressError("unknown candidate source")
    tracker = ScaleTracker()
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    base_enumerator = benchmark.enumerate_viability_candidates
    base_execute = benchmark.execute_certified_macro
    env_holder = []

    def enumerator(env, **kwargs):
        cache = kwargs.get("cache")
        if not isinstance(cache, e3.ViabilityCertificateCache):
            raise ScaleStressError("exact certificate cache is required")
        certified = base_enumerator(env, **kwargs)
        physical = e3._physical_snapshot(
            env,
            certified,
            search_config=kwargs["search_config"],
            cache=cache,
            max_replans=int(kwargs.get("max_replans", 8)),
        )
        tracker.record_frontier(env, physical)
        if candidate_source == CERTIFIED:
            return certified
        if tracker.stop_after_non_safe:
            statuses = Counter(
                e3._shadow_certificate(candidate).status.value
                for candidate in physical.candidates
            )
            tracker.post_failure_frontier = {
                "current_recovery_status": (
                    certified.current_certificate.status.value
                ),
                "physical_candidate_count": len(physical.candidates),
                "certified_candidate_count": statuses[
                    ViabilityStatus.SAFE.value
                ],
                "unsafe_candidate_count": statuses[
                    ViabilityStatus.UNSAFE.value
                ],
                "unknown_candidate_count": statuses[
                    ViabilityStatus.UNKNOWN.value
                ],
            }
            return e3._empty_copy(physical)
        return physical

    def traced_execute(env, candidate, **kwargs):
        execution = base_execute(env, candidate, **kwargs)
        if candidate_source == PHYSICAL:
            tracker.record_execution(env, candidate, execution)
        return execution

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            PROJECT_ROOT,
            {"conditioned": conditioned},
            model_seed=MODEL_SEED,
            base=base,
            device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(agent, PREFERENCE_LAMBDA)
        if candidate_source == PHYSICAL:
            return e3.DiagnosticFixedLambdaAgent(fixed, tracker)
        return fixed

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        env_holder.append(env)
        return env

    raw = None
    failure_class = None
    error = None
    started = perf_counter()
    cache = reuse.TimingInvariantCertificateCache()
    try:
        with ExitStack() as stack:
            stack.enter_context(e13._WallClock(WALL_LIMIT_SECONDS))
            stack.enter_context(reuse.path_cleanup_active())
            stack.enter_context(
                patch.object(
                    e3,
                    "legal_recovery_actions",
                    reuse.legal_recovery_actions_optimized,
                )
            )
            stack.enter_context(
                patch.object(
                    e3, "apply_recovery_action", reuse.apply_enumerated_action
                )
            )
            stack.enter_context(patch.object(benchmark, "_make_env", env_factory))
            stack.enter_context(pilot._agent_factory(agent_factory))
            stack.enter_context(
                patch.object(
                    benchmark, "ViabilityCertificateCache", lambda: cache
                )
            )
            stack.enter_context(
                patch.object(
                    benchmark, "enumerate_viability_candidates", enumerator
                )
            )
            if candidate_source == PHYSICAL:
                stack.enter_context(
                    patch.object(
                        benchmark, "execute_certified_macro", traced_execute
                    )
                )
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=e13._search_config(arm, SEARCH_MAX_NODES),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
                recovery_certification_strategy=(
                    RELOCATION_FAMILY_CERTIFICATION
                ),
            )
    except e13.EpisodeWallLimit as caught:
        failure_class = "censored_wall_clock"
        error = str(caught)
    except Exception as caught:  # recorded distinctly; never called infeasible
        failure_class = "unsupported_or_implementation_error"
        error = f"{type(caught).__name__}: {caught}"
    return (
        raw,
        tracker,
        env_holder[0] if env_holder else None,
        failure_class,
        error,
        perf_counter() - started,
    )


def _row(
    scenario: occupancy.Scenario,
    instance: EpisodeInstance,
    record: Mapping,
    *,
    candidate_source: str,
    conditioned: Mapping,
) -> dict:
    raw, tracker, env, failure_class, error, wall_seconds = _run_raw(
        scenario,
        instance,
        candidate_source=candidate_source,
        conditioned=conditioned,
    )
    audit = tracker.audit_dict()
    blocking = audit["blocking_event"]
    selected_unsafe = bool(
        blocking and blocking["selected_shadow_status"] == "UNSAFE"
    )
    selected_unknown = bool(
        blocking and blocking["selected_shadow_status"] == "UNKNOWN"
    )
    required = scenario.total_jobs
    strict = bool(
        raw is not None
        and raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and blocking is None
        and raw["illegal_drops"] == 0
        and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == required
        and (
            candidate_source == PHYSICAL
            or raw["complete_frontier_exactly_verified"]
        )
    )
    if failure_class is None:
        if strict:
            failure_class = "completed"
        elif selected_unsafe:
            failure_class = "self_blocking_nonrecoverable_successor"
        elif selected_unknown:
            failure_class = "selected_unknown_recoverability_successor"
        elif raw is not None and raw["method_failure_reason"] is not None:
            failure_class = str(raw["method_failure_reason"])
        else:
            failure_class = "valid_but_operationally_incomplete"

    action_type = None if blocking is None else blocking["selected"]["action_type"]
    rehandles = None
    if raw is not None:
        rehandles = int(
            raw.get("physical_storage_relocations", raw.get("relocations", 0))
        )
    unfinished = [] if blocking is None else list(
        blocking["unfinished_accepted_workloads"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "candidate_source": candidate_source,
        "scenario_id": scenario.scenario_id,
        "scenario": _scenario_dict(scenario),
        "instance_seed": int(record["instance_seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "strict_completion": strict,
        "failure_class": failure_class,
        "error": error,
        "self_blocking": selected_unsafe,
        "selected_unsafe_candidate": selected_unsafe,
        "selected_unknown_candidate": selected_unknown,
        "unsafe_admission": bool(selected_unsafe and action_type == "accept"),
        "unsafe_reconfiguration": bool(
            selected_unsafe and action_type == "reconfigure"
        ),
        "accepted_workloads_rendered_impossible": bool(
            selected_unsafe and unfinished
        ),
        "empty_certified_frontier_observed": bool(
            audit["safe_empty_frontiers"]
        ),
        "unfinished_accepted_workloads": unfinished,
        "empty_certified_frontier_after_selection": bool(
            blocking
            and int(
                (audit["post_failure_frontier"] or {}).get(
                    "certified_candidate_count", 0
                )
            )
            == 0
        ),
        "completed_deliveries": (
            None if raw is None else len(raw["delivery_deviations"])
        ),
        "required_deliveries": required,
        "dense_return": float(raw["return"]) if strict else None,
        **_timing(raw, required, strict),
        "steps": int(raw["steps"]) if strict else None,
        "physical_rehandles": rehandles if strict else None,
        "physical_rehandles_per_100_required_deliveries": (
            None if not strict else 100.0 * rehandles / required
        ),
        "episode_wall_seconds": wall_seconds,
        "occupancy": None if env is None else env.occupancy_summary(),
        "shadow_audit": audit,
        "training_or_learning": False,
    }


def _validate_row(
    row: Mapping,
    *,
    candidate_source: str,
    scenario_id: str,
    seed: int,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "candidate_source": candidate_source,
        "scenario_id": scenario_id,
        "instance_seed": int(seed),
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ScaleStressError(f"ledger binding changed: {key}")
    for key in (
        "strict_completion",
        "self_blocking",
        "selected_unsafe_candidate",
        "selected_unknown_candidate",
        "unsafe_admission",
        "unsafe_reconfiguration",
        "accepted_workloads_rendered_impossible",
        "empty_certified_frontier_observed",
        "empty_certified_frontier_after_selection",
    ):
        if type(row.get(key)) is not bool:
            raise ScaleStressError(f"invalid Boolean metric: {key}")
    if candidate_source == CERTIFIED and (
        row["selected_unsafe_candidate"]
        or row["selected_unknown_candidate"]
    ):
        raise ScaleStressError("certified source selected a non-SAFE candidate")
    if row["strict_completion"] and row["failure_class"] != "completed":
        raise ScaleStressError("strict completion has a failure label")


def _authenticate_ledger(
    path: Path,
    contract: Mapping,
    *,
    candidate_source: str,
    scenario_id: str,
    seed: int,
) -> dict:
    ledger = _self_hashed(path, "ledger_sha256", "scale-stress ledger")
    if ledger.get("contract_sha256") != contract["contract_sha256"]:
        raise ScaleStressError("ledger contract binding changed")
    row = ledger.get("row", {})
    _validate_row(
        row,
        candidate_source=candidate_source,
        scenario_id=scenario_id,
        seed=seed,
    )
    return ledger


def run_one(
    output: Path,
    candidate_source: str,
    scenario_id: str,
    seed: int,
) -> dict:
    if candidate_source not in CANDIDATE_SOURCES:
        raise ScaleStressError("unsupported candidate source")
    if scenario_id not in SCENARIO_BY_ID:
        raise ScaleStressError("unsupported stress scenario")
    if int(seed) not in INSTANCE_SEEDS:
        raise ScaleStressError("unsupported stress instance seed")
    contract, manifest, parents = authenticate(output)
    path = _ledger_path(output, candidate_source, scenario_id, seed)
    if path.exists():
        return _authenticate_ledger(
            path,
            contract,
            candidate_source=candidate_source,
            scenario_id=scenario_id,
            seed=seed,
        )
    record = _record(manifest, scenario_id, seed)
    instance = _load_instance(output, record)
    row = _row(
        SCENARIO_BY_ID[scenario_id],
        instance,
        record,
        candidate_source=candidate_source,
        conditioned=parents["conditioned"],
    )
    _validate_row(
        row,
        candidate_source=candidate_source,
        scenario_id=scenario_id,
        seed=seed,
    )
    ledger = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "candidate_source": candidate_source,
            "scenario_id": scenario_id,
            "instance_seed": int(seed),
            "row": row,
        },
        "ledger_sha256",
    )
    _atomic_json(path, ledger)
    return ledger


def run_pairs(
    output: Path,
    *,
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> dict:
    prepare(output)
    coordinates = [(scenario, int(seed)) for scenario in scenarios for seed in seeds]
    completed = 0
    strict = Counter()
    failures = Counter()
    for scenario_id, seed in coordinates:
        for source in (CERTIFIED, PHYSICAL):
            ledger = run_one(output, source, scenario_id, seed)
            row = ledger["row"]
            completed += 1
            strict[source] += int(row["strict_completion"])
            failures[row["failure_class"]] += int(not row["strict_completion"])
            print(
                f"E3-scale {completed}/{2 * len(coordinates)} | "
                f"{scenario_id} seed={seed} | {source} | "
                f"strict={int(row['strict_completion'])} | "
                f"failure={row['failure_class']} | "
                f"wall={row['episode_wall_seconds']:.2f}s",
                flush=True,
            )
    return {
        "status": "complete",
        "pairs": len(coordinates),
        "runs": completed,
        "strict_by_source": dict(strict),
        "failure_counts": dict(failures),
        "training_runs": 0,
    }


def _rows(output: Path, contract: Mapping) -> tuple[list[dict], list[tuple]]:
    rows = []
    missing = []
    for source in CANDIDATE_SOURCES:
        for scenario_id in SCENARIO_IDS:
            for seed in INSTANCE_SEEDS:
                path = _ledger_path(output, source, scenario_id, seed)
                if not path.exists():
                    missing.append((source, scenario_id, seed))
                    continue
                ledger = _authenticate_ledger(
                    path,
                    contract,
                    candidate_source=source,
                    scenario_id=scenario_id,
                    seed=seed,
                )
                rows.append(dict(ledger["row"]))
    return rows, missing


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(fmean(values))


def _source_summary(rows: Sequence[Mapping]) -> dict:
    rows = tuple(rows)
    complete = tuple(item for item in rows if item["strict_completion"])
    audits = tuple(item["shadow_audit"] for item in rows)
    frontiers = sum(int(item["frontier_count"]) for item in audits)
    unsafe = sum(int(item["unsafe_candidates_exposed"]) for item in audits)
    unknown = sum(int(item["unknown_candidates_exposed"]) for item in audits)
    unsafe_frontiers = sum(
        int(item["unsafe_exposure_frontiers"]) for item in audits
    )
    secondary_names = (
        "dense_return",
        "mean_absolute_error",
        "within_target_window_rate",
        "steps",
        "physical_rehandles_per_100_required_deliveries",
    )
    return {
        "runs": len(rows),
        "strict_completion_count": sum(
            int(item["strict_completion"]) for item in rows
        ),
        "self_blocking_count": sum(int(item["self_blocking"]) for item in rows),
        "selected_unsafe_count": sum(
            int(item["selected_unsafe_candidate"]) for item in rows
        ),
        "selected_unknown_count": sum(
            int(item["selected_unknown_candidate"]) for item in rows
        ),
        "unsafe_admission_count": sum(
            int(item["unsafe_admission"]) for item in rows
        ),
        "unsafe_reconfiguration_count": sum(
            int(item["unsafe_reconfiguration"]) for item in rows
        ),
        "empty_certified_frontier_count": sum(
            int(item["empty_certified_frontier_observed"]) for item in rows
        ),
        "frontiers_observed": frontiers,
        "unsafe_candidates_exposed": unsafe,
        "unknown_candidates_exposed": unknown,
        "unsafe_exposure_frontiers": unsafe_frontiers,
        "unsafe_candidates_per_frontier": (
            None if not frontiers else unsafe / frontiers
        ),
        "unsafe_frontier_rate": (
            None if not frontiers else unsafe_frontiers / frontiers
        ),
        "strict_completer_secondary_metrics": {
            "denominator": len(complete),
            **{
                name: _mean_or_none([float(item[name]) for item in complete])
                for name in secondary_names
            },
        },
        "failure_class_counts": dict(
            Counter(item["failure_class"] for item in rows)
        ),
    }


def _paired_summary(rows: Sequence[Mapping]) -> dict:
    by_coordinate = defaultdict(dict)
    for row in rows:
        by_coordinate[(row["scenario_id"], row["instance_seed"])][
            row["candidate_source"]
        ] = row
    result = Counter()
    complete_pairs = []
    for pair in by_coordinate.values():
        if set(pair) != set(CANDIDATE_SOURCES):
            continue
        physical = pair[PHYSICAL]
        certified = pair[CERTIFIED]
        p_ok = bool(physical["strict_completion"])
        c_ok = bool(certified["strict_completion"])
        if p_ok and c_ok:
            result["both_complete"] += 1
            complete_pairs.append((physical, certified))
        elif c_ok:
            result["certified_only_complete"] += 1
        elif p_ok:
            result["physical_only_complete"] += 1
        else:
            result["neither_complete"] += 1
    paired_differences = {}
    for metric in (
        "dense_return",
        "mean_absolute_error",
        "steps",
        "physical_rehandles_per_100_required_deliveries",
    ):
        values = [
            float(certified[metric]) - float(physical[metric])
            for physical, certified in complete_pairs
        ]
        paired_differences[f"certified_minus_physical_{metric}"] = {
            "n": len(values),
            "mean": _mean_or_none(values),
        }
    return {
        "complete_pairs": sum(result.values()),
        **dict(result),
        "both_complete_paired_differences": paired_differences,
    }


def _table(summaries: Sequence[Mapping], pairs: Sequence[Mapping]) -> str:
    lines = [
        "# E3 scale-stress extension",
        "",
        (
            "Designed saturation stress; these rows do not estimate failure "
            "prevalence in representative industrial workloads."
        ),
        "",
        "| Scenario | Candidate source | Strict completion | Self-block | "
        "Selected UNSAFE | Selected UNKNOWN | Unsafe/frontier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        summary = item["summary"]
        rate = summary["unsafe_candidates_per_frontier"]
        lines.append(
            f"| {item['scenario_id']} | {item['candidate_source']} | "
            f"{summary['strict_completion_count']}/{summary['runs']} | "
            f"{summary['self_blocking_count']} | "
            f"{summary['selected_unsafe_count']} | "
            f"{summary['selected_unknown_count']} | "
            f"{'--' if rate is None else f'{rate:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "| Scenario | Both complete | Certified only | Physical only | Neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in pairs:
        summary = item["paired"]
        lines.append(
            f"| {item['scenario_id']} | {summary.get('both_complete', 0)} | "
            f"{summary.get('certified_only_complete', 0)} | "
            f"{summary.get('physical_only_complete', 0)} | "
            f"{summary.get('neither_complete', 0)} |"
        )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, allow_partial: bool = False) -> dict:
    contract, manifest, _parents = authenticate(output)
    rows, missing = _rows(output, contract)
    if missing and not allow_partial:
        raise ScaleStressError(
            f"scale-stress grid incomplete: {len(missing)} runs missing"
        )
    invalid = [
        item
        for item in rows
        if item["failure_class"] == "unsupported_or_implementation_error"
    ]
    summaries = []
    pair_summaries = []
    for scenario_id in SCENARIO_IDS:
        scenario_rows = [item for item in rows if item["scenario_id"] == scenario_id]
        for source in CANDIDATE_SOURCES:
            source_rows = [
                item for item in scenario_rows if item["candidate_source"] == source
            ]
            summaries.append(
                {
                    "scenario_id": scenario_id,
                    "candidate_source": source,
                    "summary": _source_summary(source_rows),
                }
            )
        pair_summaries.append(
            {"scenario_id": scenario_id, "paired": _paired_summary(scenario_rows)}
        )
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "expected_runs": len(SCENARIOS)
            * len(INSTANCE_SEEDS)
            * len(CANDIDATE_SOURCES),
            "observed_runs": len(rows),
            "missing_runs": [list(item) for item in missing],
            "partial": bool(missing),
            "implementation_error_runs": len(invalid),
            "summaries": summaries,
            "paired_summaries": pair_summaries,
            "pooled_paired_summary": _paired_summary(rows),
            "interpretation": {
                "causal_factor": "candidate eligibility source within pair",
                "scope": (
                    "prospective seed0 designed saturation stress on two "
                    "larger supported geometries"
                ),
                "not_supported": (
                    "representative real-yard self-blocking prevalence or "
                    "multi-model-seed generalization"
                ),
                "UNKNOWN": "budget-limited unresolved, never relabeled UNSAFE",
                "secondary_metrics": "strict completers only",
            },
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    _atomic_text(output / TABLE_NAME, _table(summaries, pair_summaries))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "authenticate",
            "run-one",
            "run-pair",
            "run-scenario",
            "run-pilot",
            "run-all",
            "analyze",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-source", choices=CANDIDATE_SOURCES)
    parser.add_argument("--scenario", choices=SCENARIO_IDS)
    parser.add_argument("--seed", type=int, choices=INSTANCE_SEEDS)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()

    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "authenticate":
        contract, manifest, _parents = authenticate(output)
        result = {
            "status": "authenticated",
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        }
    elif args.command == "run-one":
        if args.candidate_source is None or args.scenario is None or args.seed is None:
            raise ScaleStressError(
                "run-one requires --candidate-source, --scenario, and --seed"
            )
        prepare(output)
        result = run_one(
            output, args.candidate_source, args.scenario, args.seed
        )["row"]
    elif args.command == "run-pair":
        if args.scenario is None or args.seed is None:
            raise ScaleStressError("run-pair requires --scenario and --seed")
        result = run_pairs(
            output, scenarios=(args.scenario,), seeds=(args.seed,)
        )
    elif args.command == "run-scenario":
        if args.scenario is None:
            raise ScaleStressError("run-scenario requires --scenario")
        result = run_pairs(
            output, scenarios=(args.scenario,), seeds=INSTANCE_SEEDS
        )
    elif args.command == "run-pilot":
        result = run_pairs(
            output, scenarios=SCENARIO_IDS, seeds=(PILOT_SEED,)
        )
        result["analysis"] = analyze(output, allow_partial=True)
    elif args.command == "run-all":
        result = run_pairs(
            output, scenarios=SCENARIO_IDS, seeds=INSTANCE_SEEDS
        )
        result["analysis"] = analyze(output)
    else:
        result = analyze(output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
