#!/usr/bin/env python3
"""E11: frozen conditioned-VCG generalization under declared shifts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e11_distribution_shift_93k_v1"
SCHEMA_VERSION = 1
INSTANCE_SEEDS = tuple(range(93_000, 93_030))
MODEL_SEEDS = final90.MODEL_SEEDS
DEPLOYMENT_LAMBDAS = (0.0, 0.05, 0.10, 0.20)
PILOT_INSTANCE_COUNT = 3
PILOT_REGIMES = (
    "reference",
    "arrival_spread",
    "dwell_long",
    "mirrored_entry",
)
EXPECTED_BLOCKS = 8
MAX_STEPS = final90.MAX_STEPS

QOP = "frozen_qop_safe"
CONDITIONED = "conditioned_vcg"
HEURISTIC = "heuristic_safe"
METHODS = (QOP, CONDITIONED, HEURISTIC)

CONTRACT_NAME = "e11-contract.json"
MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "e11-report.json"
TABLE_NAME = "e11-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e11-distribution-shift-93k"

DEFAULT_DOOR = (0, 3)
MIRRORED_DOOR = (0, 1)
DEFAULT_EXITS = ((4, 1), (4, 2), (4, 3))


class E11Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Regime:
    regime_id: str
    shift_axis: str
    arrival_rate: float = 10.0
    proc_mean: int = 80
    duration_profile: str = "poisson"
    mirrored_entry: bool = False
    combined: bool = False

    @property
    def mean_interarrival(self) -> float:
        return 1.0 / self.arrival_rate

    def make_env(self) -> SmallRoomsEnv:
        door = MIRRORED_DOOR if self.mirrored_entry else DEFAULT_DOOR
        start = (1, 3) if self.mirrored_entry else (1, 1)
        return SmallRoomsEnv(
            grid_rows=5,
            grid_cols=5,
            start_state=start,
            door_cell=door,
            pickup_cells=(1, door[1]),
            exit_cells=list(DEFAULT_EXITS),
            number_blocks=EXPECTED_BLOCKS,
            choose_storage=False,
            arrival_rate=self.arrival_rate,
            proc_mean=self.proc_mean,
        )

    def public_dict(self) -> dict:
        env = self.make_env()
        return {
            **asdict(self),
            "mean_interarrival": self.mean_interarrival,
            "grid_rows": env.grid_rows,
            "grid_cols": env.grid_cols,
            "number_blocks": env.number_blocks,
            "door_cell": list(env.door_cell),
            "pickup_cell": list(env.pickup_cell),
            "exit_cells": [list(cell) for cell in env.exit_cells],
            "storage_positions": [list(cell) for cell in env.storage_positions],
            "storage_capacity": len(env.storage_positions),
        }


REGIMES = (
    Regime("reference", "in_distribution_reference"),
    Regime("arrival_spread", "arrival_intensity", arrival_rate=0.10),
    Regime("dwell_short", "storage_duration", proc_mean=40),
    Regime("dwell_long", "storage_duration", proc_mean=120),
    Regime("dwell_bimodal", "workload_composition", duration_profile="bimodal_40_120"),
    Regime("mirrored_entry", "entrance_geometry", mirrored_entry=True),
    Regime(
        "combined_shift",
        "combined",
        arrival_rate=0.10,
        duration_profile="bimodal_40_120",
        mirrored_entry=True,
        combined=True,
    ),
)
REGIME_BY_ID = {regime.regime_id: regime for regime in REGIMES}


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha256(path: Path) -> str:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E11Error(f"missing canonical file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E11Error(f"missing canonical {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E11Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E11Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, hash_field=field):
        raise E11Error(f"{label} self hash mismatch")


def _source_contract(project_root: Path) -> dict:
    auth = final90._authenticate_inputs(project_root)["conditioned"]
    source_paths = (
        Path(__file__).resolve(),
        project_root / "benchmark_viability_critic_priority.py",
        project_root / "example/small_rooms_env.py",
        project_root / "example/episode_instance.py",
        project_root / "viability_graph_hierarchy.py",
        project_root / "vcg_v11_conditioned_handling.py",
        project_root / "run_vcg_conditioned_final_comparison_90k.py",
        Path(e04.__file__).resolve(),
    )
    return {
        "conditioned_terminal_sha256": {
            str(seed): auth["terminal_sha256"][seed] for seed in MODEL_SEEDS
        },
        "source_sha256": {
            str(path.relative_to(project_root)): _sha256(path)
            for path in source_paths
        },
    }


def _contract(project_root: Path) -> dict:
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": (
            "does_frozen_VCG_preserve_completion_and_useful_timing_handling_"
            "operating_points_under_declared_distribution_shifts"
        ),
        "instance_seeds": list(INSTANCE_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "deployment_lambdas": list(DEPLOYMENT_LAMBDAS),
        "methods": list(METHODS),
        "regimes": [regime.public_dict() for regime in REGIMES],
        "reference_regime": "reference",
        "paired_within_regime_on_exact_EpisodeInstance": True,
        "cross_regime_common_random_seed": True,
        "one_factor_regimes_precede_combined_shift": True,
        "training_or_checkpoint_selection": False,
        "deployment_specific_lambda_selection": False,
        "weights_normalization_verifier_and_candidate_generation_frozen": True,
        "complete_case_filtering_allowed": False,
        "incomplete_coordinate_metrics_suppressed": True,
        "job_classes_supported_by_simulator": False,
        "workload_composition_proxy": "mean_matched_bimodal_dwell_40_120",
        "geometry_qualification": (
            "mirrored_entry_preserves_5x5_topology_exit_count_and_eight_storage_"
            "cells_but_reflects_the_excluded_pickup_storage_cell"
        ),
        "difficulty_diagnostics": [
            "nominal_peak_concurrency",
            "simultaneous_arrival_peak",
            "arrival_span",
            "duration_mean_and_cv",
            "actual_peak_active_jobs",
            "actual_peak_storage_occupancy",
            "actual_peak_arrived_unstored",
        ],
        "primary_metrics": [
            "strict_completion_rate",
            "mean_absolute_error",
            "physical_rehandles_per_100_required_deliveries",
        ],
        "secondary_metrics": [
            "within_target_window_rate",
            "dense_return",
            "steps",
        ],
        "pilot": {
            "model_seed": 0,
            "instance_count": PILOT_INSTANCE_COUNT,
            "regimes": list(PILOT_REGIMES),
            "mechanism_check_only": True,
        },
        "expected_full_rows": len(REGIMES)
        * len(INSTANCE_SEEDS)
        * (1 + len(MODEL_SEEDS) * len(DEPLOYMENT_LAMBDAS)),
        **_source_contract(project_root),
    }
    return _with_hash(semantic, "contract_sha256")


def _instance_path(output_dir: Path, regime_id: str, seed: int) -> Path:
    return output_dir / "episode-instances" / regime_id / f"seed-{seed}.json"


def _bimodal_durations(seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(11_930_000 + int(seed))
    values = np.concatenate((rng.poisson(40.0, 4), rng.poisson(120.0, 4)))
    values = np.maximum(values, 1)
    return tuple(int(value) for value in values[rng.permutation(len(values))])


def _instance(regime: Regime, seed: int) -> EpisodeInstance:
    reference = REGIME_BY_ID["reference"].make_env().sample_episode_instance(seed)
    env = regime.make_env()
    sampled = env.sample_episode_instance(seed)
    if regime.duration_profile == "bimodal_40_120":
        durations = _bimodal_durations(seed)
    elif regime.regime_id in ("dwell_short", "dwell_long"):
        durations = sampled.storage_steps_needed
    else:
        durations = reference.storage_steps_needed

    arrivals = (
        sampled.arrival_steps
        if regime.arrival_rate != REGIME_BY_ID["reference"].arrival_rate
        else reference.arrival_steps
    )
    instance = EpisodeInstance(
        schema_version=EpisodeInstance.SCHEMA_VERSION,
        seed=seed,
        arrival_rate=regime.arrival_rate,
        proc_mean=regime.proc_mean,
        arrival_steps=arrivals,
        storage_steps_needed=durations,
        grid_rows=env.grid_rows,
        grid_cols=env.grid_cols,
        start_state=env.start_state,
        door_cell=env.door_cell,
        pickup_cell=env.pickup_cell,
        waiting_cell=env.waiting_cell,
        exit_cells=tuple(env.exit_cells),
        storage_positions=tuple(env.storage_positions),
        room_rows=tuple("".join(row) for row in env.rooms),
    )
    instance.validate_for(env)
    return instance


def _workload_diagnostics(instance: EpisodeInstance) -> dict:
    arrivals = tuple(int(value) for value in instance.arrival_steps)
    durations = tuple(int(value) for value in instance.storage_steps_needed)
    event_times = sorted(
        set(arrivals) | {arrival + duration for arrival, duration in zip(arrivals, durations)}
    )
    nominal_peak = max(
        sum(
            arrival <= time < arrival + duration
            for arrival, duration in zip(arrivals, durations)
        )
        for time in event_times
    )
    mean_duration = fmean(durations)
    return {
        "arrival_span": max(arrivals) - min(arrivals),
        "simultaneous_arrival_peak": max(Counter(arrivals).values()),
        "mean_storage_duration": mean_duration,
        "storage_duration_cv": (
            pstdev(durations) / mean_duration if mean_duration else None
        ),
        "nominal_peak_concurrency": nominal_peak,
    }


def prepare(project_root: Path, output_dir: Path) -> dict:
    output_dir = output_dir.absolute()
    expected = _contract(project_root)
    contract_path = output_dir / CONTRACT_NAME
    if contract_path.is_file():
        observed = _load_json(contract_path, label="E11 contract")
        _verify_hash(observed, "contract_sha256", label="E11 contract")
        if observed != expected:
            raise E11Error("E11 contract, checkpoints, or sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise E11Error("nonempty E11 output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        final90._atomic_json(contract_path, expected)

    records = []
    for regime in REGIMES:
        for index, seed in enumerate(INSTANCE_SEEDS):
            path = _instance_path(output_dir, regime.regime_id, seed)
            sampled = _instance(regime, seed)
            if path.is_file():
                observed = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
                if observed != sampled:
                    raise E11Error(f"serialized E11 instance changed: {regime.regime_id}/{seed}")
            else:
                final90._atomic_text(path, sampled.to_json())
                observed = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "regime_id": regime.regime_id,
                    "shift_axis": regime.shift_axis,
                    "seed": seed,
                    "instance_index": index,
                    "relative_path": str(path.relative_to(output_dir)),
                    "raw_sha256": _sha256(path),
                    "canonical_sha256": hashlib.sha256(
                        observed.to_json().encode("utf-8")
                    ).hexdigest(),
                    "episode_instance_id": observed.instance_id,
                    "schedule_id": observed.schedule_id,
                    "workload": _workload_diagnostics(observed),
                }
            )
    manifest = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": expected["contract_sha256"],
            "instances_sampled_after_contract_write": True,
            "records": records,
        },
        "manifest_sha256",
    )
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        observed = _load_json(manifest_path, label="E11 manifest")
        _verify_hash(observed, "manifest_sha256", label="E11 manifest")
        if observed != manifest:
            raise E11Error("E11 instance manifest changed")
    else:
        final90._atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "regimes": len(REGIMES),
        "instances_per_regime": len(INSTANCE_SEEDS),
        "expected_full_rows": expected["expected_full_rows"],
        "pilot_rows": len(PILOT_REGIMES) * PILOT_INSTANCE_COUNT * 5,
        "contract": str(contract_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }


def authenticate(project_root: Path, output_dir: Path) -> tuple[dict, dict]:
    contract = _load_json(output_dir / CONTRACT_NAME, label="E11 contract")
    _verify_hash(contract, "contract_sha256", label="E11 contract")
    if contract != _contract(project_root):
        raise E11Error("E11 contract, checkpoints, or sources changed")
    manifest = _load_json(output_dir / MANIFEST_NAME, label="E11 manifest")
    _verify_hash(manifest, "manifest_sha256", label="E11 manifest")
    if (
        manifest.get("contract_sha256") != contract["contract_sha256"]
        or len(manifest.get("records", ())) != len(REGIMES) * len(INSTANCE_SEEDS)
    ):
        raise E11Error("E11 manifest binding or size changed")
    return contract, manifest


def _load_instance(output_dir: Path, record: Mapping) -> EpisodeInstance:
    path = output_dir / record["relative_path"]
    if _sha256(path) != record["raw_sha256"]:
        raise E11Error("E11 instance bytes changed")
    instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
    if (
        instance.instance_id != record["episode_instance_id"]
        or instance.schedule_id != record["schedule_id"]
    ):
        raise E11Error("E11 instance identity changed")
    instance.validate_for(REGIME_BY_ID[record["regime_id"]].make_env())
    return instance


def _specs(selected_seed: Optional[int]) -> tuple[dict, ...]:
    seeds = MODEL_SEEDS if selected_seed is None else (selected_seed,)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise E11Error("model seed must be 0, 1, or 2")
    specs = []
    if selected_seed in (None, 0):
        specs.append({"method": HEURISTIC, "model_seed": None, "lambda": None})
    for seed in seeds:
        specs.append({"method": QOP, "model_seed": seed, "lambda": 0.0})
        specs.extend(
            {"method": CONDITIONED, "model_seed": seed, "lambda": value}
            for value in DEPLOYMENT_LAMBDAS[1:]
        )
    return tuple(specs)


def _spec_token(spec: Mapping) -> str:
    if spec["method"] == HEURISTIC:
        return HEURISTIC
    value = str(float(spec["lambda"])).replace(".", "p")
    return f"{spec['method']}__seed-{spec['model_seed']}__lambda-{value}"


def _ledger_path(output_dir: Path, regime_id: str, spec: Mapping, seed: int) -> Path:
    return (
        output_dir
        / "run-ledger"
        / regime_id
        / _spec_token(spec)
        / f"instance-{seed}.json"
    )


def _e4_spec(spec: Mapping) -> dict:
    if spec["method"] == HEURISTIC:
        return {
            "ranking_signal": e04.HEURISTIC_SAFE,
            "model_seed": None,
            "preference_lambda": None,
            "ranking_seed": None,
        }
    if spec["method"] == QOP:
        return {
            "ranking_signal": e04.QOP_SAFE,
            "model_seed": int(spec["model_seed"]),
            "preference_lambda": 0.0,
            "ranking_seed": None,
        }
    return {
        "ranking_signal": e04.CONDITIONED_SAFE,
        "model_seed": int(spec["model_seed"]),
        "preference_lambda": float(spec["lambda"]),
        "ranking_seed": None,
    }


@contextmanager
def _runtime_environment(regime: Regime):
    with patch.object(benchmark, "_make_env", lambda _payload: regime.make_env()):
        yield


def _execution_diagnostics(raw: Mapping, instance: EpisodeInstance) -> dict:
    snapshots = [decision["post_macro_state"] for decision in raw.get("decisions", ())]
    peak_active = sum(value <= 0 for value in instance.arrival_steps)
    peak_storage = 0
    peak_unstored = peak_active
    final_blocks = []
    for snapshot in snapshots:
        time_step = int(snapshot["time_steps"])
        blocks = snapshot["blocks"]
        active = sum(
            instance.arrival_steps[int(block["label"][1:]) - 1] <= time_step
            and not block["delivered"]
            for block in blocks
        )
        storage = sum(block["stored"] and not block["delivered"] for block in blocks)
        unstored = sum(
            instance.arrival_steps[int(block["label"][1:]) - 1] <= time_step
            and block["stored_time_step"] is None
            and not block["delivered"]
            for block in blocks
        )
        peak_active = max(peak_active, active)
        peak_storage = max(peak_storage, storage)
        peak_unstored = max(peak_unstored, unstored)
        final_blocks = blocks
    admitted = sum(block["stored_time_step"] is not None for block in final_blocks)
    return {
        "nominal_peak_concurrency": _workload_diagnostics(instance)[
            "nominal_peak_concurrency"
        ],
        "actual_peak_active_jobs": peak_active,
        "actual_peak_storage_occupancy": peak_storage,
        "actual_peak_arrived_unstored": peak_unstored,
        "jobs_ever_admitted": admitted,
        "jobs_unadmitted_at_stop": EXPECTED_BLOCKS - admitted,
        "jobs_delivered": len(raw.get("delivery_deviations", ())),
        "explicit_admission_rejections": 0,
        "explicit_rejection_action_supported": False,
    }


def _row(raw: Mapping, instance: EpisodeInstance, record: Mapping, spec: Mapping) -> dict:
    strict = bool(
        raw.get("strict_method_success")
        and raw.get("terminal")
        and raw.get("method_failure_reason") is None
        and raw.get("complete_frontier_exactly_verified")
        and int(raw.get("illegal_drops", 0)) == 0
        and int(raw.get("macro_failures", 0)) == 0
        and len(raw.get("delivery_deviations", ())) == EXPECTED_BLOCKS
    )
    compact = pilot._compact_row(raw, instance) if strict else None
    timing = final90._timing(raw["delivery_deviations"]) if strict else {
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
    }
    observed_steps = int(raw.get("steps", 0))
    observed_rehandles = int(raw.get("physical_storage_relocations", raw.get("relocations", 0)))
    reason = raw.get("method_failure_reason")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "regime_id": record["regime_id"],
        "shift_axis": record["shift_axis"],
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "episode_instance_sha256": record["canonical_sha256"],
        **dict(spec),
        "strict_safe_complete": strict,
        "all_selected_candidates_exact_safe": bool(
            raw.get("complete_frontier_exactly_verified", False)
        ),
        "method_failure_reason": reason,
        "deadlock_or_empty_frontier": reason == "no_exact_safe_candidate",
        "episode_step_limit": reason == "episode_step_limit",
        "dense_return": float(compact["dense_return"]) if strict else None,
        **timing,
        "steps": int(compact["steps"]) if strict else None,
        "physical_storage_relocations": (
            int(compact["physical_rehandles"]) if strict else None
        ),
        "physical_rehandles_per_100_required_deliveries": (
            float(compact["physical_rehandles_per_100"]) if strict else None
        ),
        "observed_steps_to_stop": observed_steps,
        "observed_rehandles_to_stop": observed_rehandles,
        "required_deliveries": EXPECTED_BLOCKS,
        "behavior_digest": raw.get("behavior_digest"),
        "evaluation_learning": False,
        **_execution_diagnostics(raw, instance),
    }


def _failed_row(error: Exception, record: Mapping, spec: Mapping) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "regime_id": record["regime_id"],
        "shift_axis": record["shift_axis"],
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["canonical_sha256"],
        **dict(spec),
        "strict_safe_complete": False,
        "all_selected_candidates_exact_safe": None,
        "method_failure_reason": f"{type(error).__name__}: {error}",
        "deadlock_or_empty_frontier": False,
        "episode_step_limit": False,
        "dense_return": None,
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
        "steps": None,
        "physical_storage_relocations": None,
        "physical_rehandles_per_100_required_deliveries": None,
        "observed_steps_to_stop": None,
        "observed_rehandles_to_stop": None,
        "required_deliveries": EXPECTED_BLOCKS,
        "behavior_digest": None,
        "evaluation_learning": False,
        "nominal_peak_concurrency": record["workload"]["nominal_peak_concurrency"],
        "actual_peak_active_jobs": None,
        "actual_peak_storage_occupancy": None,
        "actual_peak_arrived_unstored": None,
        "jobs_ever_admitted": None,
        "jobs_unadmitted_at_stop": None,
        "jobs_delivered": None,
        "explicit_admission_rejections": 0,
        "explicit_rejection_action_supported": False,
    }


def run(
    project_root: Path,
    output_dir: Path,
    *,
    selected_seed: Optional[int],
    regime_ids: Sequence[str],
    instance_limit: Optional[int],
    device_name: str,
) -> dict:
    contract, manifest = authenticate(project_root, output_dir)
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E11Error("E11 is frozen to CPU execution")
    unknown = set(regime_ids) - set(REGIME_BY_ID)
    if unknown:
        raise E11Error(f"unknown E11 regimes: {sorted(unknown)}")
    if instance_limit is not None and not 1 <= instance_limit <= len(INSTANCE_SEEDS):
        raise E11Error("invalid E11 instance limit")
    auth = final90._authenticate_inputs(project_root)
    conditioned = auth["conditioned"]
    specs = _specs(selected_seed)
    records = [
        record
        for record in manifest["records"]
        if record["regime_id"] in regime_ids
        and (instance_limit is None or int(record["instance_index"]) < instance_limit)
    ]
    complete = safe = 0
    for record in records:
        regime = REGIME_BY_ID[record["regime_id"]]
        instance = _load_instance(output_dir, record)
        for spec in specs:
            path = _ledger_path(
                output_dir, regime.regime_id, spec, int(record["seed"])
            )
            if path.is_file():
                ledger = _load_json(path, label="E11 rollout ledger")
                _verify_hash(ledger, "ledger_sha256", label="E11 rollout ledger")
                if (
                    ledger.get("contract_sha256") != contract["contract_sha256"]
                    or ledger.get("manifest_sha256") != manifest["manifest_sha256"]
                    or ledger.get("spec") != spec
                    or ledger.get("instance_sha256") != record["canonical_sha256"]
                ):
                    raise E11Error("E11 rollout ledger binding changed")
                row = ledger["run"]
            else:
                model_seed = 0 if spec["method"] == HEURISTIC else int(spec["model_seed"])
                arm = conditioned["inputs"]["arms"][model_seed]
                factory = e04._factory(
                    project_root,
                    auth,
                    _e4_spec(spec),
                    arm,
                    device,
                )
                try:
                    with _runtime_environment(regime):
                        raw = pilot._run_raw(
                            arm,
                            instance,
                            device=device,
                            wrapper_factory=factory,
                        )
                    row = _row(raw, instance, record, spec)
                except Exception as error:
                    row = _failed_row(error, record, spec)
                ledger = _with_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "instance_sha256": record["canonical_sha256"],
                        "spec": spec,
                        "run": row,
                    },
                    "ledger_sha256",
                )
                final90._atomic_json(path, ledger)
            complete += 1
            safe += int(row["strict_safe_complete"])
            if complete % 10 == 0:
                print(f"E11 {complete}/{len(records) * len(specs)} | safe={safe}", flush=True)
    return {
        "status": "complete",
        "regimes": list(regime_ids),
        "model_seeds": list(MODEL_SEEDS if selected_seed is None else (selected_seed,)),
        "instances_per_regime": len(INSTANCE_SEEDS) if instance_limit is None else instance_limit,
        "rows_seen_or_written": complete,
        "strict_safe_complete": safe,
    }


METRICS = (
    "dense_return",
    "mean_absolute_error",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


def _all_rows(output_dir: Path) -> list[dict]:
    rows = []
    root = output_dir / "run-ledger"
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("**/instance-*.json")):
        ledger = _load_json(path, label="E11 rollout ledger")
        _verify_hash(ledger, "ledger_sha256", label="E11 rollout ledger")
        rows.append(ledger["run"])
    return rows


def _aggregate(rows: Sequence[Mapping], *, expected: int) -> dict:
    safe = [row for row in rows if row["strict_safe_complete"]]
    complete = len(rows) == expected and len(safe) == expected
    result = {
        "rows": len(rows),
        "expected_rows": expected,
        "strict_safe_complete": len(safe),
        "strict_completion_rate": len(safe) / expected if expected else None,
        "available_row_completion_rate": len(safe) / len(rows) if rows else None,
        "deadlock_or_empty_frontier": sum(
            bool(row["deadlock_or_empty_frontier"]) for row in rows
        ),
        "episode_step_limit": sum(bool(row["episode_step_limit"]) for row in rows),
        "complete_case_metrics_suppressed": not complete,
        "failure_reasons": dict(
            Counter(
                row["method_failure_reason"]
                for row in rows
                if row["method_failure_reason"] is not None
            )
        ),
    }
    for metric in METRICS:
        result[metric] = fmean(float(row[metric]) for row in rows) if complete else None
    for metric in (
        "nominal_peak_concurrency",
        "actual_peak_active_jobs",
        "actual_peak_storage_occupancy",
        "actual_peak_arrived_unstored",
        "jobs_ever_admitted",
        "jobs_unadmitted_at_stop",
        "jobs_delivered",
    ):
        usable = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = fmean(float(value) for value in usable) if usable else None
    return result


def _coordinate(method: str, value: Optional[float]) -> str:
    return method if value is None else f"{method}_lambda_{value:.2f}"


def _selected(rows, regime_id, method, value=None, model_seed=None):
    return [
        row
        for row in rows
        if row["regime_id"] == regime_id
        and row["method"] == method
        and (value is None or float(row["lambda"]) == value)
        and (model_seed is None or int(row["model_seed"]) == model_seed)
    ]


def _paired_delta(shift_rows, reference_rows) -> dict:
    def key(row):
        return (row.get("model_seed"), int(row["instance_seed"]))

    shift = {key(row): row for row in shift_rows}
    reference = {key(row): row for row in reference_rows}
    keys = sorted(set(shift).intersection(reference), key=str)
    all_complete = bool(keys) and all(
        shift[item]["strict_safe_complete"] and reference[item]["strict_safe_complete"]
        for item in keys
    )
    return {
        "pairs": len(keys),
        "strict_completion_shift_minus_reference": (
            fmean(
                int(shift[item]["strict_safe_complete"])
                - int(reference[item]["strict_safe_complete"])
                for item in keys
            )
            if keys
            else None
        ),
        **{
            f"{metric}_shift_minus_reference": (
                fmean(float(shift[item][metric]) - float(reference[item][metric]) for item in keys)
                if all_complete
                else None
            )
            for metric in METRICS
        },
    }


def _table(report: Mapping) -> str:
    lines = [
        "# E11 frozen-policy distribution-shift results",
        "",
        "| Regime | Method | Strict complete | MAE | Rehandles/100 | Within +/-20 | Return | Steps | Nominal peak concurrency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ((QOP, 0.0), (CONDITIONED, 0.05), (CONDITIONED, 0.10), (CONDITIONED, 0.20), (HEURISTIC, None))
    for regime in REGIMES:
        for method, value in order:
            item = report["aggregate"][regime.regime_id][_coordinate(method, value)]
            def metric(name, digits=2):
                observed = item[name]
                return "--" if observed is None else f"{observed:.{digits}f}"
            label = method if value is None else f"{method} (lambda={value:.2f})"
            lines.append(
                f"| {regime.regime_id} | {label} | "
                f"{item['strict_safe_complete']}/{item['expected_rows']} | "
                f"{metric('mean_absolute_error')} | "
                f"{metric('physical_rehandles_per_100_required_deliveries')} | "
                f"{metric('within_target_window_rate', 3)} | "
                f"{metric('dense_return')} | {metric('steps')} | "
                f"{metric('nominal_peak_concurrency')} |"
            )
    lines.extend(
        [
            "",
            "Whole-coordinate operational metrics are suppressed when any required row is missing or fails strict completion. Completion, failure, admission, and workload-size diagnostics remain reported for every available row.",
            "",
            "No preference or checkpoint is selected separately for a deployment regime.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(project_root: Path, output_dir: Path, *, allow_partial: bool) -> dict:
    contract, manifest = authenticate(project_root, output_dir)
    rows = _all_rows(output_dir)
    expected_total = int(contract["expected_full_rows"])
    if not allow_partial and len(rows) != expected_total:
        raise E11Error(f"E11 grid is incomplete: {len(rows)}/{expected_total}")
    aggregate = {}
    per_model_seed = {}
    deltas = {}
    order = ((QOP, 0.0), (CONDITIONED, 0.05), (CONDITIONED, 0.10), (CONDITIONED, 0.20), (HEURISTIC, None))
    for regime in REGIMES:
        aggregate[regime.regime_id] = {}
        per_model_seed[regime.regime_id] = {}
        if regime.regime_id != "reference":
            deltas[regime.regime_id] = {}
        for method, value in order:
            selected = _selected(rows, regime.regime_id, method, value)
            expected = len(INSTANCE_SEEDS) if method == HEURISTIC else len(MODEL_SEEDS) * len(INSTANCE_SEEDS)
            key = _coordinate(method, value)
            aggregate[regime.regime_id][key] = _aggregate(selected, expected=expected)
            if method != HEURISTIC:
                per_model_seed[regime.regime_id][key] = {
                    str(seed): _aggregate(
                        _selected(rows, regime.regime_id, method, value, seed),
                        expected=len(INSTANCE_SEEDS),
                    )
                    for seed in MODEL_SEEDS
                }
            if regime.regime_id != "reference":
                reference = _selected(rows, "reference", method, value)
                deltas[regime.regime_id][key] = _paired_delta(selected, reference)

    complete = len(rows) == expected_total
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete" if complete else "partial",
            "paper_evidence": complete,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "rows": len(rows),
            "expected_rows": expected_total,
            "aggregate": aggregate,
            "per_model_seed": per_model_seed,
            "paired_shift_minus_reference": deltas,
            "claim_boundary": (
                "fixed_5x5_eight_job_zero_shot_distribution_shifts_not_scaling_"
                "to_larger_problem_sizes"
            ),
        },
        "report_sha256",
    )
    final90._atomic_json(output_dir / REPORT_NAME, report)
    final90._atomic_text(output_dir / TABLE_NAME, _table(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=MODEL_SEEDS)
    parser.add_argument("--regime", action="append", choices=tuple(REGIME_BY_ID))
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(project_root, output_dir)
    elif args.command == "run":
        result = run(
            project_root,
            output_dir,
            selected_seed=args.model_seed,
            regime_ids=tuple(args.regime or REGIME_BY_ID),
            instance_limit=args.instance_limit,
            device_name=args.device,
        )
    else:
        result = analyze(project_root, output_dir, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
