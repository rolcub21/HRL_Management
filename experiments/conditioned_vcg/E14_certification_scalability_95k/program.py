#!/usr/bin/env python3
"""Immutable contract and artifact helpers for E14."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
import benchmark_viability_critic_priority as benchmark


PROTOCOL = "vcg_conditioned_e14_certificate_reuse_95k_v1"
SCHEMA_VERSION = 1
SCENARIO_IDS = ("size_10x10_occ_medium", "size_10x10_occ_high")
INSTANCE_SEED = 95_100
MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
CAPTURE_LIMIT_SECONDS = 3_600
REPLAY_CHUNK_SECONDS = 3_600
PARENT_OUTPUT = PROJECT_ROOT / "results/vcg-d10-occupancy-extension-95k"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e14-certificate-reuse-95k"
CONTRACT_NAME = "e14-contract.json"
REPORT_NAME = "e14-report.json"
ONLINE_REPORT_NAME = "e14-online-report.json"


class E14Error(RuntimeError):
    pass


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = digest(result)
    return result


def sha256(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E14Error(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Mapping) -> None:
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


def load_json(path: Path, *, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise E14Error(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E14Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E14Error(f"{label} must contain an object")
    return value


def atomic_gzip_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(canonical_bytes(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_gzip_json(path: Path, *, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise E14Error(f"missing {label}: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E14Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E14Error(f"{label} must contain an object")
    return value


def hardware() -> dict:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_model = platform.processor() or "unknown"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
    }


def trace_path(output: Path, scenario_id: str) -> Path:
    return output / "query-workloads" / f"{scenario_id}.json.gz"


def capture_ledger_path(output: Path, scenario_id: str) -> Path:
    return output / "capture-ledgers" / f"{scenario_id}.json"


def path_replay_path(output: Path, scenario_id: str) -> Path:
    return output / "path-replay" / f"{scenario_id}.jsonl"


def online_ledger_path(output: Path, scenario_id: str, stage: str) -> Path:
    return output / "online-ledgers" / scenario_id / f"{stage}.json"


def _source_contract() -> dict:
    relative = (
        "experiments/conditioned_vcg/E14_certification_scalability_95k/program.py",
        "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        "experiments/conditioned_vcg/E14_certification_scalability_95k/capture.py",
        "experiments/conditioned_vcg/E14_certification_scalability_95k/replay.py",
        "experiments/conditioned_vcg/E14_certification_scalability_95k/confirm.py",
        "PSLAP/viability.py",
        "PSLAP/viability_candidates.py",
        "PSLAP/viability_filter.py",
        "PSLAP/viability_dataset.py",
        "benchmark_viability_critic_priority.py",
    )
    return {name: sha256(PROJECT_ROOT / name) for name in relative}


def _parent_records(manifest: Mapping) -> list[dict]:
    records = [
        dict(record) for record in manifest["records"]
        if record["scenario_id"] in SCENARIO_IDS
        and int(record["seed"]) == INSTANCE_SEED
    ]
    if {record["scenario_id"] for record in records} != set(SCENARIO_IDS):
        raise E14Error("parent manifest lacks an E14 capture instance")
    return sorted(records, key=lambda item: SCENARIO_IDS.index(item["scenario_id"]))


def expected_contract() -> dict:
    parent_contract, parent_manifest = occupancy.authenticate(PARENT_OUTPUT)
    conditioned = occupancy._conditioned_auth()
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    search_config = benchmark._search_config(arm.payload)
    if search_config.search_order != "goal_directed":
        raise E14Error("E14 suffix proofs are scoped to goal-directed search")
    return with_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "question": "how_much_exact_certification_cost_is_removable_duplication",
        "parent_protocol": parent_contract["protocol"],
        "parent_contract_sha256": parent_contract["contract_sha256"],
        "parent_manifest_sha256": parent_manifest["manifest_sha256"],
        "parent_records": _parent_records(parent_manifest),
        "source_sha256": _source_contract(),
        "scenario_ids": list(SCENARIO_IDS),
        "instance_seed": INSTANCE_SEED,
        "model_seed": MODEL_SEED,
        "preference_lambda": PREFERENCE_LAMBDA,
        "search_config": {
            "max_depth": search_config.max_depth,
            "max_nodes": search_config.max_nodes,
            "max_primitive_steps": search_config.max_primitive_steps,
            "reserve_queue_cells": search_config.reserve_queue_cells,
            "search_order": search_config.search_order,
        },
        "capture_wall_limit_seconds": CAPTURE_LIMIT_SECONDS,
        "replay_chunk_seconds": REPLAY_CHUNK_SECONDS,
        "capture_timeout_semantics": "right_censored_not_failure_not_infeasibility",
        "execution": "sequential_isolated_cpu",
        "training": False,
        "stages": [
            {
                "id": "current",
                "mechanism": "complete_state_and_budget_outcome_cache",
            },
            {
                "id": "timing_invariant_key",
                "mechanism": "project_only_irrelevant_block_clocks_from_cache_identity",
            },
            {
                "id": "path_enumeration_cleanup",
                "mechanism": "one_canonical_transport_BFS_per_block_and_no_revalidation_of_already_enumerated_actions",
            },
            {
                "id": "witness_suffix_store",
                "mechanism": "separate_constructive_positive_proofs_from_budgeted_search_outcomes",
            },
        ],
        "shared_search_structure": "conditional_not_implemented_until_residual_duplication_is_measured",
        "proof_rules": {
            "safe": "must_replay_to_completion_under_identical_physical_contract",
            "witness_length": "completion_upper_bound_not_shortest_rank",
            "unknown": "never_inserted_into_positive_proof_store",
            "unsafe": "ordinary_outcome_cache_only_and_only_if_exhaustive",
        },
    }, "contract_sha256")


def prepare(output: Path) -> dict:
    contract = expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        observed = load_json(path, label="E14 contract")
        if observed != contract:
            raise E14Error("E14 contract, parent artifacts, or sources changed")
    else:
        atomic_json(path, contract)
    return contract


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = load_json(output / CONTRACT_NAME, label="E14 contract")
    if contract != expected_contract():
        raise E14Error("E14 contract, parent artifacts, or sources changed")
    _parent, manifest = occupancy.authenticate(PARENT_OUTPUT)
    return contract, manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "authenticate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    result = prepare(output) if args.command == "prepare" else authenticate(output)[0]
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
