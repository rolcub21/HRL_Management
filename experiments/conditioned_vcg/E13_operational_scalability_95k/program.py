#!/usr/bin/env python3
"""E13: frozen-policy operational and decision-space scalability panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
from statistics import fmean
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
    run as d10,
)
import PSLAP.viability_candidates as candidate_module
import PSLAP.viability_filter as filter_module
from PSLAP.relocation_family_certification import (
    RELOCATION_FAMILY_CERTIFICATION,
)
from PSLAP.viability import RecoveryActionKind
from PSLAP.viability_filter import ViabilitySearchConfig
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_final86_four_method as final86
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e13_operational_scalability_95k_v2"
SCHEMA_VERSION = 1
PARENT_D10 = ROOT / "results/vcg-d10-occupancy-extension-95k"
PARENT_D12 = (
    ROOT
    / "results/vcg-d12-relocation-family-integrated-confirmation-95k-v2"
)
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e13-scalability-95k-v2"
CONDITIONED_TERMINAL = (
    ROOT
    / "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/"
    "terminal.pth"
)
CONTRACT_NAME = "e13-contract.json"
MANIFEST_NAME = "e13-instance-manifest.json"
REPORT_NAME = "e13-report.json"
TABLE_NAME = "e13-companion-table.md"

MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
SEARCH_MAX_NODES = 20_000
WALL_LIMIT_SECONDS = 1_800
INSTANCE_SEEDS = (95_100, 95_101, 95_102)
DEADLINE_SECONDS = (0.1, 1.0, 5.0)
PRIMARY_DEADLINE_SECONDS = 1.0

MAIN_SCENARIOS = (
    "size_5x5_occ_low",
    "size_5x5_occ_medium",
    "size_5x5_occ_high",
    "size_8x8_occ_low",
    "size_8x8_occ_medium",
    "size_8x8_occ_high",
    "size_10x10_occ_low",
    "size_10x10_occ_medium",
    "size_10x10_occ_high",
)
COMPANION_SCENARIOS = (
    "fixed_workload_8x8_k6_n8",
    "fixed_workload_10x10_k6_n8",
    "episode_8x8_medium_n36",
    "episode_8x8_medium_n54",
    "rectangle_6x10_medium",
    "rectangle_10x6_medium",
)
SCENARIOS = MAIN_SCENARIOS + COMPANION_SCENARIOS


class E13Error(RuntimeError):
    pass


class EpisodeWallLimit(RuntimeError):
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
        raise E13Error(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise E13Error(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E13Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E13Error(f"{label} must contain an object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise E13Error(f"{label} self-hash mismatch")
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


def _seed0_inputs(d10_contract: Mapping) -> dict:
    arms, _records = final86._authenticate_v11()
    selected = [arm for arm in arms if int(arm.model_seed) == MODEL_SEED]
    if len(selected) != 1:
        raise E13Error("could not resolve frozen VCG seed-0 base")
    terminal_sha = _sha(CONDITIONED_TERMINAL)
    if terminal_sha != d10_contract["conditioned_terminal_sha256"]:
        raise E13Error("conditioned checkpoint differs from D10 parent")
    terminal = torch.load(
        CONDITIONED_TERMINAL, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("fixed_terminal_checkpoint") is not True
        or not isinstance(terminal.get("agent_checkpoint"), Mapping)
    ):
        raise E13Error("conditioned terminal checkpoint is invalid")
    return {
        "inputs": {"arms": {MODEL_SEED: selected[0]}},
        "terminals": {MODEL_SEED: terminal},
        "terminal_sha256": {MODEL_SEED: terminal_sha},
    }


def _parent_inputs() -> dict:
    d10_contract = _self_hashed(
        PARENT_D10 / "d10-occupancy-contract.json",
        "contract_sha256",
        "D10 occupancy contract",
    )
    d10_manifest = _self_hashed(
        PARENT_D10 / "episode-instance-manifest.json",
        "manifest_sha256",
        "D10 occupancy manifest",
    )
    if d10_manifest.get("contract_sha256") != d10_contract["contract_sha256"]:
        raise E13Error("D10 manifest contract mismatch")
    records = []
    for item in d10_manifest["records"]:
        if (
            item["scenario_id"] in SCENARIOS
            and int(item["seed"]) in INSTANCE_SEEDS
        ):
            path = PARENT_D10 / item["relative_path"]
            if _sha(path) != item["raw_sha256"]:
                raise E13Error("D10 instance hash mismatch")
            records.append(dict(item))
    expected = {(scenario, seed) for scenario in SCENARIOS for seed in INSTANCE_SEEDS}
    observed = {(item["scenario_id"], int(item["seed"])) for item in records}
    if observed != expected or len(records) != len(expected):
        raise E13Error("D10 manifest does not contain the frozen E13 panel")

    d12_contract = _self_hashed(
        PARENT_D12 / "contract.json",
        "contract_sha256",
        "D12 integrated contract",
    )
    d12_report = _self_hashed(
        PARENT_D12 / "report.json",
        "report_sha256",
        "D12 integrated report",
    )
    if (
        d12_report.get("contract_sha256") != d12_contract["contract_sha256"]
        or d12_report.get("missing_scenarios") != []
        or d12_report.get("all_observed_strict_safe_complete") is not True
        or d12_report.get("all_reference_prefixes_identical") is not True
    ):
        raise E13Error("D12 acceptance evidence is incomplete")
    conditioned = _seed0_inputs(d10_contract)
    return {
        "d10_contract": d10_contract,
        "d10_manifest": d10_manifest,
        "d12_contract": d12_contract,
        "d12_report": d12_report,
        "records": records,
        "conditioned": conditioned,
    }


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        ROOT / "PSLAP/relocation_family_certification.py",
        ROOT / "PSLAP/viability.py",
        ROOT / "PSLAP/viability_candidates.py",
        ROOT / "PSLAP/viability_filter.py",
        ROOT / "benchmark_viability_critic_priority.py",
        ROOT / "run_vcg_conditioned_final_comparison_90k.py",
        ROOT / "run_vcg_final86_four_method.py",
        ROOT / "run_vcg_v11_conditioned_handling_seed0_85k.py",
        ROOT / "run_vcg_v11_nested_handling_pilot.py",
        ROOT
        / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT
        / "experiments/conditioned_vcg/development/"
        "D10_scalability_support_screen/occupancy_extension.py",
        ROOT
        / "experiments/conditioned_vcg/development/"
        "D10_scalability_support_screen/run.py",
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def expected_contract() -> dict:
    inputs = _parent_inputs()
    scenarios = [occupancy.SCENARIO_BY_ID[item].public_dict() for item in SCENARIOS]
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "question": (
                "how_do_operational_quality_candidate_breadth_and_decision_"
                "latency_scale_with_geometry_occupancy_and_episode_length"
            ),
            "paper_role": "E13_operational_and_decision_space_scalability",
            "main_scenarios": list(MAIN_SCENARIOS),
            "companion_scenarios": list(COMPANION_SCENARIOS),
            "scenarios": scenarios,
            "instance_seeds": list(INSTANCE_SEEDS),
            "expected_rows": len(SCENARIOS) * len(INSTANCE_SEEDS),
            "model_seed": MODEL_SEED,
            "preference_lambda": PREFERENCE_LAMBDA,
            "device": "cpu",
            "training": False,
            "weights_and_normalization_frozen": True,
            "search_config": {
                "max_depth": None,
                "max_nodes": SEARCH_MAX_NODES,
                "max_primitive_steps": None,
                "reserve_queue_cells": True,
                "search_order": "goal_directed",
            },
            "wall_limit_seconds_per_episode": WALL_LIMIT_SECONDS,
            "recovery_certification_strategy": RELOCATION_FAMILY_CERTIFICATION,
            "path_enumeration_cleanup": "e14_order_preserving_cleanup",
            "timing_invariant_cache": True,
            "family_proofs_stored_in_native_outcome_cache": False,
            "family_miss_behavior": "native_exact_fallback_under_same_budget",
            "deadline_seconds": list(DEADLINE_SECONDS),
            "primary_deadline_seconds": PRIMARY_DEADLINE_SECONDS,
            "deadline_scope": (
                "predeclared_computational_threshold_not_industrial_SLA"
            ),
            "one_process_per_episode_required_for_latency_and_memory": True,
            "quality_metrics_not_used_to_hide_incomplete_runs": True,
            "d10_contract_sha256": inputs["d10_contract"]["contract_sha256"],
            "d10_manifest_sha256": inputs["d10_manifest"]["manifest_sha256"],
            "d12_contract_sha256": inputs["d12_contract"]["contract_sha256"],
            "d12_report_sha256": inputs["d12_report"]["report_sha256"],
            "conditioned_terminal_sha256": inputs["conditioned"][
                "terminal_sha256"
            ][MODEL_SEED],
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    inputs = _parent_inputs()
    contract = expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        if _load(path, "E13 contract") != contract:
            raise E13Error("E13 contract, sources, or inputs changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E13Error("nonempty E13 output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, contract)
    manifest = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "parent_manifest_sha256": inputs["d10_manifest"]["manifest_sha256"],
            "parent_root": str(PARENT_D10.relative_to(ROOT)),
            "records": inputs["records"],
        },
        "manifest_sha256",
    )
    manifest_path = output / MANIFEST_NAME
    if manifest_path.exists():
        if _load(manifest_path, "E13 manifest") != manifest:
            raise E13Error("E13 instance manifest changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "scenarios": len(SCENARIOS),
        "rows": len(inputs["records"]),
        "training": False,
    }


def authenticate(output: Path) -> tuple[dict, dict, dict]:
    contract = _load(output / CONTRACT_NAME, "E13 contract")
    if contract != expected_contract():
        raise E13Error("E13 contract, sources, or inputs changed")
    manifest = _self_hashed(
        output / MANIFEST_NAME, "manifest_sha256", "E13 manifest"
    )
    if manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise E13Error("E13 manifest contract mismatch")
    inputs = _parent_inputs()
    if manifest.get("parent_manifest_sha256") != inputs["d10_manifest"][
        "manifest_sha256"
    ]:
        raise E13Error("E13 parent manifest mismatch")
    return contract, manifest, inputs


def authenticate_frozen(output: Path) -> tuple[dict, dict, dict]:
    """Authenticate completed E13 artifacts without rebinding live sources.

    Downstream reanalysis must preserve the source hashes recorded when E13
    ran.  A later executor correction should not silently relabel those rows
    as current-code results, but it also should not make their self-hashed
    contract, manifest, and ledgers unreadable.
    """

    contract = _self_hashed(
        output / CONTRACT_NAME, "contract_sha256", "frozen E13 contract"
    )
    if (
        contract.get("protocol") != PROTOCOL
        or contract.get("main_scenarios") != list(MAIN_SCENARIOS)
        or contract.get("instance_seeds") != list(INSTANCE_SEEDS)
    ):
        raise E13Error("frozen E13 contract coordinates changed")
    manifest = _self_hashed(
        output / MANIFEST_NAME, "manifest_sha256", "frozen E13 manifest"
    )
    if manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise E13Error("frozen E13 manifest contract mismatch")
    return contract, manifest, {"authentication_scope": "frozen_artifacts"}


class _WallClock:
    def __init__(self, seconds: int):
        self.seconds = int(seconds)

    def __enter__(self):
        if not hasattr(signal, "setitimer"):
            return self
        self.previous = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise EpisodeWallLimit(
                f"predeclared {self.seconds}s E13 episode limit reached"
            )

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *_args):
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous)


def _call_distribution(calls: Sequence[Mapping]) -> dict:
    return {
        "count": len(calls),
        "seconds": d10._distribution([item["seconds"] for item in calls]),
        "explored_nodes": d10._distribution(
            [item["explored_nodes"] for item in calls]
        ),
        "generated_states": d10._distribution(
            [item["generated_states"] for item in calls]
        ),
        "status_counts": dict(Counter(item["status"] for item in calls)),
        "total_seconds": sum(float(item["seconds"]) for item in calls),
    }


class FrontierInstrumentation:
    """Role-aware native-search and D12 accounting for one episode."""

    def __init__(self, scenario_id: str, seed: int):
        self.scenario_id = scenario_id
        self.seed = int(seed)
        self.certificate_calls: list[dict] = []
        self.legal_action_batches: list[tuple] = []
        self.recovery_roles: dict[object, deque[str]] = defaultdict(deque)
        self.completed_frontiers: list[dict] = []
        self.in_progress: Optional[dict] = None
        self.active_check: Optional[dict] = None
        self.interrupted_check: Optional[dict] = None

    def _timed(self, function, state, role: str, *args, **kwargs):
        started = perf_counter()
        self.active_check = {"check_role": role, "started": started}
        try:
            certificate = function(state, *args, **kwargs)
        except BaseException:
            self.interrupted_check = {
                "check_role": role,
                "elapsed_seconds": perf_counter() - started,
            }
            raise
        finally:
            self.active_check = None
        self.certificate_calls.append(
            {
                "check_role": role,
                "status": certificate.status.value,
                "explored_nodes": int(certificate.explored_nodes),
                "generated_states": int(certificate.generated_states),
                "exhaustive": bool(certificate.exhaustive),
                "seconds": perf_counter() - started,
            }
        )
        return certificate

    def partial_frontier(self) -> Optional[dict]:
        if self.in_progress is None:
            return None
        calls = self.certificate_calls[self.in_progress["call_start"] :]
        result = {
            "decision_boundary": self.in_progress["decision_boundary"],
            "elapsed_seconds": perf_counter() - self.in_progress["started"],
            "certification_calls_completed": calls,
            "checks_by_role": {
                role: _call_distribution(
                    [item for item in calls if item["check_role"] == role]
                )
                for role in sorted({item["check_role"] for item in calls})
            },
        }
        if self.interrupted_check is not None:
            result["interrupted_check"] = self.interrupted_check
        return result

    @contextmanager
    def active(self):
        core_candidate_analyze = candidate_module.analyze_recoverability
        core_filter_analyze = filter_module.analyze_recoverability
        core_legal = candidate_module.legal_recovery_actions
        original_enumerate = benchmark._enumerate_frontier

        def timed_candidate(state, *args, **kwargs):
            queued = self.recovery_roles.get(state)
            role = queued.popleft() if queued else "current_state"
            return self._timed(
                core_candidate_analyze, state, role, *args, **kwargs
            )

        def timed_accept(state, *args, **kwargs):
            return self._timed(
                core_filter_analyze, state, "accept", *args, **kwargs
            )

        def recorded_legal(state):
            actions = tuple(core_legal(state))
            self.legal_action_batches.append(actions)
            for action in actions:
                successor = candidate_module.apply_recovery_action(state, action)
                role = (
                    "deliver"
                    if action.kind is RecoveryActionKind.DELIVERY
                    else "reconfigure"
                )
                self.recovery_roles[successor].append(role)
            return actions

        def enriched_enumerate(env, *args, **kwargs):
            self.recovery_roles.clear()
            self.interrupted_check = None
            call_start = len(self.certificate_calls)
            legal_start = len(self.legal_action_batches)
            started = perf_counter()
            self.in_progress = {
                "decision_boundary": env._measurement(),
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
            native_by_role = Counter(item["check_role"] for item in calls)
            family_by_role = Counter(
                {"reconfigure": int(record["relocation_family_proof_count"])}
            )
            cache_by_role = {}
            for role, count in subjects.items():
                hits = count - native_by_role[role] - family_by_role[role]
                if hits < 0:
                    raise E13Error("negative cache-hit attribution")
                role_calls = [
                    item for item in calls if item["check_role"] == role
                ]
                cache_by_role[role] = {
                    "subjects": count,
                    "native_searches": native_by_role[role],
                    "family_proofs": family_by_role[role],
                    "cache_hits": hits,
                    "native_search_seconds": sum(
                        float(item["seconds"]) for item in role_calls
                    ),
                }
            legacy_misses = int(audit["cache_misses"])
            if legacy_misses != len(calls) + int(
                record["relocation_family_proof_count"]
            ):
                raise E13Error("native/family accounting does not match audit")
            if int(audit["cache_hits"]) != sum(
                item["cache_hits"] for item in cache_by_role.values()
            ):
                raise E13Error("role cache-hit accounting does not match audit")
            record.update(
                {
                    "decision_boundary": self.in_progress["decision_boundary"],
                    "current_recovery_status": audit[
                        "current_recovery_status"
                    ],
                    "physical_candidate_count": int(
                        audit["physical_accept_count"]
                    )
                    + len(legal)
                    + int(bool(audit["defer_allowed"])),
                    "physical_accept_count": int(audit["physical_accept_count"]),
                    "physical_deliver_count": int(
                        kinds[RecoveryActionKind.DELIVERY.value]
                    ),
                    "physical_reconfigure_count": int(
                        kinds[RecoveryActionKind.RELOCATION.value]
                    ),
                    "physical_defer_count": int(bool(audit["defer_allowed"])),
                    "certified_candidate_count": int(audit["candidate_count"]),
                    "certified_accept_count": int(
                        audit["accept_candidate_count"]
                    ),
                    "certified_deliver_count": int(
                        audit["deliver_candidate_count"]
                    ),
                    "certified_reconfigure_count": int(
                        audit["reconfigure_candidate_count"]
                    ),
                    "certified_defer_count": int(
                        audit["defer_candidate_count"]
                    ),
                    "unknown_candidate_count": int(audit["unknown_accept_count"])
                    + int(audit["unknown_recovery_count"]),
                    "unsafe_candidate_count": int(audit["unsafe_accept_count"])
                    + int(audit["unsafe_recovery_count"]),
                    "no_positively_certified_candidate": int(
                        audit["candidate_count"]
                    )
                    == 0,
                    "cache_by_check_role": cache_by_role,
                    "native_certification_calls": calls,
                    "checks_by_role": {
                        role: _call_distribution(
                            [
                                item
                                for item in calls
                                if item["check_role"] == role
                            ]
                        )
                        for role in subjects
                    },
                    "candidate_generation_and_overhead_seconds": max(
                        float(record["total_frontier_seconds"])
                        - float(record["exact_search_seconds"]),
                        0.0,
                    ),
                }
            )
            self.completed_frontiers.append(record)
            self.in_progress = None
            print(
                json.dumps(
                    {
                        "scenario": self.scenario_id,
                        "seed": self.seed,
                        "frontier": len(self.completed_frontiers),
                        "physical": record["physical_candidate_count"],
                        "certified": record["certified_candidate_count"],
                        "family_proofs": record[
                            "relocation_family_proof_count"
                        ],
                        "native_searches": len(calls),
                        "seconds": record["total_frontier_seconds"],
                    }
                ),
                flush=True,
            )
            return snapshot, record

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    candidate_module, "analyze_recoverability", timed_candidate
                )
            )
            stack.enter_context(
                patch.object(
                    filter_module, "analyze_recoverability", timed_accept
                )
            )
            stack.enter_context(
                patch.object(
                    candidate_module, "legal_recovery_actions", recorded_legal
                )
            )
            stack.enter_context(
                patch.object(
                    benchmark, "_enumerate_frontier", enriched_enumerate
                )
            )
            yield


FRONTIER_FIELDS = (
    "decision_epoch",
    "candidate_frontier_digest",
    "decision_boundary",
    "current_recovery_status",
    "physical_candidate_count",
    "physical_accept_count",
    "physical_deliver_count",
    "physical_reconfigure_count",
    "physical_defer_count",
    "certified_candidate_count",
    "certified_accept_count",
    "certified_deliver_count",
    "certified_reconfigure_count",
    "certified_defer_count",
    "unknown_candidate_count",
    "unsafe_candidate_count",
    "no_positively_certified_candidate",
    "cache_hits",
    "cache_misses",
    "cache_by_check_role",
    "native_certification_calls",
    "checks_by_role",
    "relocation_family_anchor_available",
    "relocation_family_attempt_count",
    "relocation_family_proof_count",
    "relocation_family_miss_count",
    "relocation_family_setup_seconds",
    "relocation_family_connection_seconds",
    "native_recovery_search_count",
    "exact_search_seconds",
    "candidate_generation_and_overhead_seconds",
    "total_frontier_seconds",
    "complete_frontier_exactly_verified",
)


def _compact_frontier(record: Mapping) -> dict:
    missing = [name for name in FRONTIER_FIELDS if name not in record]
    if missing:
        raise E13Error(f"frontier record lacks fields: {missing}")
    return {name: record[name] for name in FRONTIER_FIELDS}


def _record(manifest: Mapping, scenario_id: str, seed: int) -> dict:
    matches = [
        item
        for item in manifest["records"]
        if item["scenario_id"] == scenario_id and int(item["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise E13Error("E13 manifest coordinate is not unique")
    return dict(matches[0])


def _load_instance(record: Mapping) -> EpisodeInstance:
    path = PARENT_D10 / record["relative_path"]
    if _sha(path) != record["raw_sha256"]:
        raise E13Error("E13 parent instance hash mismatch")
    instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
    scenario = occupancy.SCENARIO_BY_ID[record["scenario_id"]]
    instance.validate_for(scenario.make_base_env())
    return instance


def _ledger_path(output: Path, scenario_id: str, seed: int) -> Path:
    return output / "run-ledger" / scenario_id / f"seed-{seed}.json"


def _authenticate_ledger(
    output: Path, contract: Mapping, scenario_id: str, seed: int
) -> dict:
    ledger = _self_hashed(
        _ledger_path(output, scenario_id, seed),
        "ledger_sha256",
        "E13 run ledger",
    )
    row = ledger.get("row", {})
    if (
        ledger.get("contract_sha256") != contract["contract_sha256"]
        or row.get("scenario_id") != scenario_id
        or int(row.get("instance_seed", -1)) != int(seed)
    ):
        raise E13Error("E13 ledger coordinate mismatch")
    return ledger


def _search_config(arm, max_nodes: int) -> ViabilitySearchConfig:
    base = benchmark._search_config(arm.payload)
    config = replace(base, max_nodes=int(max_nodes))
    expected = ViabilitySearchConfig(
        max_depth=None,
        max_nodes=int(max_nodes),
        max_primitive_steps=None,
        reserve_queue_cells=True,
        search_order="goal_directed",
    )
    if config != expected:
        raise E13Error("unexpected frozen search configuration")
    return config


def run_coordinate(
    output: Path,
    scenario_id: str,
    seed: int,
    *,
    max_nodes: int = SEARCH_MAX_NODES,
    contract_override: Optional[Mapping] = None,
    ledger_path_override: Optional[Path] = None,
) -> dict:
    if scenario_id not in SCENARIOS:
        raise E13Error(f"unsupported E13 scenario: {scenario_id}")
    if int(seed) not in INSTANCE_SEEDS:
        raise E13Error(f"unsupported E13 instance seed: {seed}")
    contract, manifest, inputs = authenticate(output)
    active_contract = contract if contract_override is None else contract_override
    path = (
        _ledger_path(output, scenario_id, seed)
        if ledger_path_override is None
        else ledger_path_override
    )
    if path.exists() and contract_override is None:
        return _authenticate_ledger(output, contract, scenario_id, seed)

    record = _record(manifest, scenario_id, seed)
    instance = _load_instance(record)
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    conditioned = inputs["conditioned"]
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    agent_holder = []
    env_holder = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            ROOT,
            {"conditioned": conditioned},
            model_seed=MODEL_SEED,
            base=base,
            device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        fixed = conditioned_seed0._FixedLambdaAgent(agent, PREFERENCE_LAMBDA)
        timed = d10.SelectionTimer(fixed)
        agent_holder.append(timed)
        return timed

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        env_holder.append(env)
        return env

    recorder = FrontierInstrumentation(scenario_id, seed)
    rss_before = occupancy._rss_bytes()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = perf_counter()
    raw = None
    failure_class = None
    error = None
    try:
        cache = reuse.TimingInvariantCertificateCache()
        with (
            _WallClock(WALL_LIMIT_SECONDS),
            reuse.path_cleanup_active(),
            recorder.active(),
            patch.object(benchmark, "_make_env", env_factory),
            pilot._agent_factory(agent_factory),
            patch.object(
                benchmark, "ViabilityCertificateCache", lambda: cache
            ),
        ):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=_search_config(arm, max_nodes),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
                recovery_certification_strategy=(
                    RELOCATION_FAMILY_CERTIFICATION
                ),
            )
    except EpisodeWallLimit as caught:
        failure_class = "censored_wall_clock"
        error = str(caught)
    except Exception as caught:
        failure_class = "unsupported_or_implementation_error"
        error = f"{type(caught).__name__}: {caught}"
    wall_seconds = perf_counter() - started

    frontiers = [_compact_frontier(item) for item in recorder.completed_frontiers]
    decisions = [] if raw is None else list(raw["decisions"])
    selection_records = agent_holder[0].records if agent_holder else []
    decision_costs = []
    if raw is not None and len(selection_records) != len(decisions):
        raise E13Error("selection instrumentation did not align")
    for decision, timing in zip(decisions, selection_records):
        frontier = frontiers[int(decision["frontier_index"])]
        decision_costs.append(
            {
                **timing,
                "frontier_seconds": float(frontier["total_frontier_seconds"]),
                "candidate_generation_and_overhead_seconds": float(
                    frontier["candidate_generation_and_overhead_seconds"]
                ),
                "native_exact_search_seconds": float(
                    frontier["exact_search_seconds"]
                ),
                "family_setup_seconds": float(
                    frontier["relocation_family_setup_seconds"]
                ),
                "family_connection_seconds": float(
                    frontier["relocation_family_connection_seconds"]
                ),
                "end_to_end_decision_seconds": float(
                    frontier["total_frontier_seconds"]
                )
                + float(timing["selection_wall_seconds"]),
                "cold_first_decision": len(decision_costs) == 0,
                "liveness_forced": bool(decision["liveness_forced"]),
                "selected_action_type": decision["selected_action_type"],
            }
        )

    strict = bool(
        raw is not None
        and raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and raw["illegal_drops"] == 0
        and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == scenario.total_jobs
    )
    state_contains_unknown = any(
        int(item["unknown_candidate_count"]) > 0 for item in frontiers
    )
    empty_frontier = any(
        bool(item["no_positively_certified_candidate"]) for item in frontiers
    )
    if failure_class is None:
        if strict:
            failure_class = "completed"
        elif state_contains_unknown or empty_frontier:
            failure_class = "certificate_budget_limited"
        else:
            failure_class = "valid_but_operationally_incomplete"
    timing = (
        d10._timing(
            raw["delivery_deviations"],
            expected_deliveries=scenario.total_jobs,
        )
        if strict
        else {
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "mean_tardiness": None,
            "mean_earliness": None,
            "within_target_window_rate": None,
        }
    )
    rehandles = (
        None
        if raw is None
        else int(
            raw.get("physical_storage_relocations", raw.get("relocations", 0))
        )
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "protocol": str(active_contract["protocol"]),
        "scenario_id": scenario_id,
        "scenario": scenario.public_dict(),
        "instance_seed": int(seed),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "search_max_nodes": int(max_nodes),
        "recovery_certification_strategy": RELOCATION_FAMILY_CERTIFICATION,
        "failure_class": failure_class,
        "error": error,
        "strict_safe_complete": strict,
        "state_contains_unknown": state_contains_unknown,
        "empty_certified_frontier_observed": empty_frontier,
        "quality_metrics_scope": (
            "strict_complete_episode" if strict else "suppressed"
        ),
        "dense_return": float(raw["return"]) if strict else None,
        **timing,
        "steps": int(raw["steps"]) if strict else None,
        "steps_per_delivery": (
            float(raw["steps"] / scenario.total_jobs) if strict else None
        ),
        "physical_rehandles": rehandles if strict else None,
        "physical_rehandles_per_100_required_deliveries": (
            100.0 * rehandles / scenario.total_jobs if strict else None
        ),
        "deliveries_at_stop": (
            len(raw["delivery_deviations"]) if raw is not None else None
        ),
        "required_deliveries": scenario.total_jobs,
        "observed_steps_to_stop": int(raw["steps"]) if raw is not None else None,
        "macro_decisions": len(decisions),
        "completed_frontiers": len(frontiers),
        "episode_wall_seconds": wall_seconds,
        "frontiers": frontiers,
        "in_progress_frontier": recorder.partial_frontier(),
        "decision_costs": decision_costs,
        "occupancy": (
            env_holder[0].occupancy_summary() if env_holder else None
        ),
        "relocation_family_attempt_count": sum(
            int(item["relocation_family_attempt_count"]) for item in frontiers
        ),
        "relocation_family_proof_count": sum(
            int(item["relocation_family_proof_count"]) for item in frontiers
        ),
        "relocation_family_miss_count": sum(
            int(item["relocation_family_miss_count"]) for item in frontiers
        ),
        "relocation_family_setup_seconds": sum(
            float(item["relocation_family_setup_seconds"]) for item in frontiers
        ),
        "relocation_family_connection_seconds": sum(
            float(item["relocation_family_connection_seconds"])
            for item in frontiers
        ),
        "native_search_count": sum(
            len(item["native_certification_calls"]) for item in frontiers
        ),
        "native_exact_search_seconds": sum(
            float(item["exact_search_seconds"]) for item in frontiers
        ),
        "frontier_seconds": sum(
            float(item["total_frontier_seconds"]) for item in frontiers
        ),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": occupancy._rss_bytes(),
        "process_peak_rss_before_kib": peak_before,
        "process_peak_rss_after_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "hardware": occupancy._hardware(),
        "behavior_digest": raw.get("behavior_digest") if raw is not None else None,
    }
    ledger = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": str(active_contract["protocol"]),
            "contract_sha256": active_contract["contract_sha256"],
            "manifest_sha256": (
                manifest["manifest_sha256"]
                if contract_override is None
                else active_contract["parent_e13_manifest_sha256"]
            ),
            "conditioned_terminal_sha256": inputs["conditioned"][
                "terminal_sha256"
            ][MODEL_SEED],
            "row": row,
        },
        "ledger_sha256",
    )
    _atomic_json(path, ledger)
    return ledger


def _rows(output: Path, contract: Mapping) -> list[dict]:
    rows = []
    for scenario_id in SCENARIOS:
        for seed in INSTANCE_SEEDS:
            path = _ledger_path(output, scenario_id, seed)
            if path.exists():
                rows.append(
                    _authenticate_ledger(
                        output, contract, scenario_id, int(seed)
                    )["row"]
                )
    return rows


def _distribution(values: Sequence[float]) -> dict:
    return d10._distribution(tuple(float(value) for value in values))


def _scenario_summary(scenario_id: str, rows: Sequence[Mapping]) -> dict:
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    strict = [item for item in rows if item["strict_safe_complete"]]
    frontiers = [item for row in rows for item in row.get("frontiers", ())]
    decisions = [item for row in rows for item in row.get("decision_costs", ())]
    warm = [item for item in decisions if not item["cold_first_decision"]]
    calls = [
        item for frontier in frontiers for item in frontier["native_certification_calls"]
    ]
    complete_coordinate = len(rows) == len(INSTANCE_SEEDS)
    all_strict = complete_coordinate and len(strict) == len(rows)
    complete_case_metrics = None
    if strict:
        complete_case_metrics = {
            "n": len(strict),
            "dense_return": fmean(item["dense_return"] for item in strict),
            "mean_absolute_error": fmean(
                item["mean_absolute_error"] for item in strict
            ),
            "physical_rehandles_per_100_required_deliveries": fmean(
                item["physical_rehandles_per_100_required_deliveries"]
                for item in strict
            ),
            "steps_per_delivery": fmean(
                item["steps_per_delivery"] for item in strict
            ),
            "within_target_window_rate": fmean(
                item["within_target_window_rate"] for item in strict
            ),
        }
    candidate_subjects = sum(
        int(item["physical_candidate_count"])
        - int(item["physical_defer_count"])
        for item in frontiers
    )
    unknown = sum(int(item["unknown_candidate_count"]) for item in frontiers)
    role_names = ("current_state", "accept", "deliver", "reconfigure")
    return {
        "scenario": scenario.public_dict(),
        "observed_rows": len(rows),
        "expected_rows": len(INSTANCE_SEEDS),
        "failure_classes": dict(Counter(item["failure_class"] for item in rows)),
        "strict_safe_complete": len(strict),
        "strict_completion_rate": (
            len(strict) / len(rows) if rows else None
        ),
        "complete_coordinate": complete_coordinate,
        "all_required_rows_strict": all_strict,
        "complete_case_metrics": complete_case_metrics,
        "all_required_rows_metrics": complete_case_metrics if all_strict else None,
        "occupancy": {
            key: _distribution(
                [item["occupancy"][key] for item in rows if item.get("occupancy")]
            )
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
            key: _distribution([item[key] for item in frontiers])
            for key in (
                "physical_accept_count",
                "physical_deliver_count",
                "physical_reconfigure_count",
                "physical_defer_count",
                "certified_accept_count",
                "certified_deliver_count",
                "certified_reconfigure_count",
                "certified_defer_count",
                "physical_candidate_count",
                "certified_candidate_count",
            )
        },
        "candidate_unknown_rate": (
            unknown / candidate_subjects if candidate_subjects else 0.0
        ),
        "empty_certified_frontier_rate": (
            sum(item["no_positively_certified_candidate"] for item in frontiers)
            / len(frontiers)
            if frontiers
            else None
        ),
        "family": {
            "attempts": sum(
                int(item["relocation_family_attempt_count"]) for item in frontiers
            ),
            "proofs": sum(
                int(item["relocation_family_proof_count"]) for item in frontiers
            ),
            "misses": sum(
                int(item["relocation_family_miss_count"]) for item in frontiers
            ),
            "setup_seconds": sum(
                float(item["relocation_family_setup_seconds"])
                for item in frontiers
            ),
            "connection_seconds": sum(
                float(item["relocation_family_connection_seconds"])
                for item in frontiers
            ),
            "anchor_available_rate": (
                sum(item["relocation_family_anchor_available"] for item in frontiers)
                / len(frontiers)
                if frontiers
                else None
            ),
        },
        "native_searches": {
            "count": len(calls),
            "seconds": sum(float(item["seconds"]) for item in calls),
            "per_decision": len(calls) / len(frontiers) if frontiers else None,
            "expanded_nodes": _distribution(
                [item["explored_nodes"] for item in calls]
            ),
            "seconds_per_search": _distribution(
                [item["seconds"] for item in calls]
            ),
            "by_role": {
                role: _call_distribution(
                    [item for item in calls if item["check_role"] == role]
                )
                for role in role_names
            },
        },
        "latency": {
            "frontier_seconds": _distribution(
                [item["total_frontier_seconds"] for item in frontiers]
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
                for deadline in DEADLINE_SECONDS
            },
            "episode_wall_seconds": _distribution(
                [item["episode_wall_seconds"] for item in rows]
            ),
        },
        "memory": {
            "peak_rss_kib": _distribution(
                [item["process_peak_rss_after_kib"] for item in rows]
            ),
            "rss_after_bytes": _distribution(
                [item["rss_after_bytes"] for item in rows]
            ),
        },
        "liveness_intervention_rate": (
            sum(item["liveness_forced"] for item in decisions) / len(decisions)
            if decisions
            else None
        ),
    }


def _fmt(value, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _table(summaries: Sequence[Mapping]) -> str:
    lines = [
        "# E13 operational and decision-space scalability",
        "",
        (
            "Quality values are reported only when every required row for a "
            "coordinate strictly completes. Latency is descriptive CPU timing."
        ),
        "",
        "| Scenario | Strict | Initial occ. | Mean occ. | Peak occ. | MAE | Rehandles/100 | Steps/delivery | Within ±20 | Physical cand. | Certified cand. | Warm p95 (s) | >1 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        scenario = item["scenario"]
        metrics = item["all_required_rows_metrics"]
        occupancy_summary = item["occupancy"]
        physical = item["frontier_by_action"]["physical_candidate_count"]
        certified = item["frontier_by_action"]["certified_candidate_count"]
        latency = item["latency"]
        lines.append(
            "| {scenario} | {strict}/{expected} | {initial} | {mean_occ} | "
            "{peak} | {mae} | {rehandles} | {steps} | {window} | {physical} | "
            "{certified} | {p95} | {deadline} |".format(
                scenario=scenario["scenario_id"],
                strict=item["strict_safe_complete"],
                expected=item["expected_rows"],
                initial=_fmt(
                    occupancy_summary["initial_storage_occupancy_ratio"]["mean"],
                    3,
                ),
                mean_occ=_fmt(
                    occupancy_summary[
                        "time_weighted_mean_storage_occupancy_ratio"
                    ]["mean"],
                    3,
                ),
                peak=_fmt(
                    occupancy_summary["peak_storage_occupancy_ratio"]["mean"],
                    3,
                ),
                mae=_fmt(None if metrics is None else metrics["mean_absolute_error"]),
                rehandles=_fmt(
                    None
                    if metrics is None
                    else metrics[
                        "physical_rehandles_per_100_required_deliveries"
                    ]
                ),
                steps=_fmt(None if metrics is None else metrics["steps_per_delivery"]),
                window=_fmt(
                    None
                    if metrics is None
                    else 100.0 * metrics["within_target_window_rate"],
                    1,
                ),
                physical=_fmt(physical["mean"], 1),
                certified=_fmt(certified["mean"], 1),
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
    contract, _manifest, _inputs = authenticate_frozen(output)
    rows = _rows(output, contract)
    expected = len(SCENARIOS) * len(INSTANCE_SEEDS)
    if len(rows) != expected and not allow_partial:
        raise E13Error(f"expected {expected} E13 rows, found {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    summaries = [
        _scenario_summary(scenario_id, grouped.get(scenario_id, ()))
        for scenario_id in SCENARIOS
    ]
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "expected_rows": expected,
            "observed_rows": len(rows),
            "partial": len(rows) != expected,
            "main_panel_complete": all(
                len(grouped.get(item, ())) == len(INSTANCE_SEEDS)
                for item in MAIN_SCENARIOS
            ),
            "all_coordinates_complete": len(rows) == expected,
            "summaries": summaries,
            "interpretation_scope": (
                "frozen_seed0_lambda_0.10_zero_shot_scalability_with_E14_"
                "cleanup_and_D12_enabled"
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
        "command", choices=("prepare", "run-one", "analyze", "authenticate")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int, choices=INSTANCE_SEEDS)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "authenticate":
        contract, manifest, _inputs = authenticate(output)
        result = {
            "status": "authenticated",
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        }
    elif args.command == "run-one":
        if args.scenario is None or args.seed is None:
            raise E13Error("run-one requires --scenario and --seed")
        prepare(output)
        ledger = run_coordinate(output, args.scenario, args.seed)
        row = ledger["row"]
        result = {
            "status": "completed" if row["strict_safe_complete"] else "recorded",
            "scenario": row["scenario_id"],
            "instance_seed": row["instance_seed"],
            "strict_safe_complete": row["strict_safe_complete"],
            "failure_class": row["failure_class"],
            "deliveries": f'{row["deliveries_at_stop"]}/{row["required_deliveries"]}',
            "decisions": row["macro_decisions"],
            "episode_wall_seconds": row["episode_wall_seconds"],
            "family_proofs": row["relocation_family_proof_count"],
            "family_misses": row["relocation_family_miss_count"],
            "native_searches": row["native_search_count"],
            "native_exact_search_seconds": row["native_exact_search_seconds"],
            "ledger": str(
                _ledger_path(output, args.scenario, args.seed).relative_to(ROOT)
            ),
        }
    else:
        result = analyze(output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
