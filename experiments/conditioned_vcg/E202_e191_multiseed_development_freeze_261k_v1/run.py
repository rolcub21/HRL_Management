#!/usr/bin/env python3
"""Authenticate and freeze all E191 through-120 development selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import torch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    run as e191_run,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e202_e191_multiseed_development_freeze_261k_v1"
SOURCE_OUTPUT = e191.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e202-e191-multiseed-development-freeze-261k-v1"
)
SCENARIO = e191.SCENARIO
MODEL_SEEDS = e191.MODEL_SEEDS
SELECTION_HORIZON = 120
EXPECTED_VALIDATION_EPISODES = tuple(
    range(e191.VALIDATION_EVERY, SELECTION_HORIZON + 1, e191.VALIDATION_EVERY)
)
FROZEN_NAME = "selected-development-checkpoint.pth"
MANIFEST_NAME = "freeze.json"
CANDIDATE_NAMES = ("best-candidate-slot-a.pth", "best-candidate-slot-b.pth")


class FreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _record_identity(record: Mapping) -> tuple:
    return (
        int(record["checkpoint_episode"]),
        tuple(float(value) for value in record["selection_score"]),
        tuple(str(value) for value in record["selection_score_fields"]),
    )


def _score(record: Mapping) -> tuple[float, ...]:
    return tuple(float(value) for value in record["selection_score"])


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(
            sha256(destination) == sha256(source),
            f"refusing to overwrite another frozen checkpoint: {destination}",
        )
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    require(not temporary.exists(), f"temporary path exists: {temporary}")
    try:
        with source.open("rb") as source_stream, temporary.open("xb") as target:
            shutil.copyfileobj(source_stream, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == encoded, f"freeze manifest changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inspect_seed(source_output: Path, model_seed: int) -> dict:
    require(model_seed in MODEL_SEEDS, f"unsupported E191 model seed: {model_seed}")
    authenticated = e191_run.authenticate(source_output.resolve())
    directory = (
        source_output.resolve()
        / "training/operational"
        / SCENARIO
        / f"seed-{model_seed}"
    )
    summary_path = directory / "training-summary.json"
    summary = read_json(summary_path)
    require(summary.get("status") == "paused", "E191 seed is not paused")
    require(
        int(summary.get("completed_training_episodes", -1)) == SELECTION_HORIZON,
        "E191 seed did not complete the through-120 selection horizon",
    )
    require(int(summary.get("model_seed", -1)) == model_seed, "summary seed differs")
    require(not summary.get("test_panels_opened"), "E191 test panels were opened")
    require(not summary.get("performance_claim_authorized"), "unexpected claim state")

    history = tuple(summary.get("validation_history", ()))
    require(
        tuple(int(row["checkpoint_episode"]) for row in history)
        == EXPECTED_VALIDATION_EPISODES,
        "E191 validation history is incomplete",
    )
    selected = summary.get("best_validation_record")
    require(isinstance(selected, Mapping), "E191 summary lacks its best record")
    authoritative = max(history, key=_score)
    require(
        _record_identity(authoritative) == _record_identity(selected),
        "E191 stored selection differs from its authoritative score",
    )

    candidate_name = summary.get("best_candidate_checkpoint_filename")
    require(candidate_name in CANDIDATE_NAMES, "invalid E191 candidate filename")
    source = (directory / str(candidate_name)).resolve()
    require(source.parent == directory, "E191 candidate escaped seed directory")
    require(source.is_file(), f"missing E191 candidate: {source}")
    source_hash = sha256(source)
    require(
        source_hash == summary.get("best_candidate_checkpoint_sha256"),
        "E191 candidate SHA-256 differs from its summary",
    )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping), "E191 candidate is not a mapping")
    expected = {
        "training_protocol": e191.PROTOCOL,
        "method_version": e191.METHOD_VERSION,
        "checkpoint_role": "best_candidate_unfinalized",
        "model_seed": model_seed,
        "completed_training_episodes": int(selected["checkpoint_episode"]),
        "trainer_resumable": False,
        "protocol_training_complete": False,
        "selection_finalized_after_total_episodes": False,
        "deployment_checkpoint_eligible": False,
        "performance_claim_authorized": False,
        "test_panels_opened": False,
        "e187_contract_sha256": authenticated["contract_sha256"],
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    require(not mismatches, f"E191 candidate metadata differs: {mismatches}")
    payload_record = payload.get("best_validation_record")
    require(isinstance(payload_record, Mapping), "candidate lacks its best record")
    require(
        _record_identity(payload_record) == _record_identity(selected),
        "candidate and summary select different records",
    )

    metrics = selected["summary"]
    return {
        "status": "ready",
        "protocol": PROTOCOL,
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": authenticated["contract_sha256"],
        "source_summary": str(summary_path),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "model_seed": model_seed,
        "selection_horizon": SELECTION_HORIZON,
        "selected_episode": int(selected["checkpoint_episode"]),
        "selection_score_fields": list(selected["selection_score_fields"]),
        "selection_score": [float(value) for value in selected["selection_score"]],
        "selected_metrics": {
            "J": float(metrics["mean_e187_J"]),
            "storage_clock_mae": float(metrics["mean_storage_clock_mae"]),
            "within_20": float(metrics["mean_storage_within_20_rate"]),
            "admission_wait": float(metrics["mean_admission_wait"]),
            "rehandles_per_100": float(metrics["relocations_per_100_deliveries"]),
            "peak_stored": float(metrics["mean_peak_stored"]),
            "strict_completion_rate": float(metrics["strict_method_success_rate"]),
        },
        "checkpoint_role": "best_candidate_unfinalized",
        "protocol_training_complete": False,
        "deployment_checkpoint_eligible": False,
        "development_only": True,
        "test_panels_opened": False,
        "paper_evidence": False,
    }


def freeze_seed(source_output: Path, output: Path, model_seed: int) -> dict:
    inspected = inspect_seed(source_output.resolve(), model_seed)
    directory = output.resolve() / f"seed-{model_seed}"
    checkpoint = directory / FROZEN_NAME
    manifest_path = directory / MANIFEST_NAME
    source = Path(inspected["source_checkpoint"])
    _atomic_copy(source, checkpoint)
    frozen_hash = sha256(checkpoint)
    require(
        frozen_hash == inspected["source_checkpoint_sha256"],
        "frozen checkpoint is not byte-identical to its source",
    )
    core = {
        **inspected,
        "status": "complete",
        "frozen_checkpoint": str(checkpoint),
        "frozen_checkpoint_sha256": frozen_hash,
        "byte_identical_to_source_checkpoint": True,
        "source_mutated": False,
    }
    manifest = {**core, "manifest_core_sha256": digest(core)}
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def freeze_all(source_output: Path, output: Path) -> dict:
    rows = [freeze_seed(source_output, output, seed) for seed in MODEL_SEEDS]
    require(
        len({row["source_contract_sha256"] for row in rows}) == 1,
        "E191 seeds authenticate against different contracts",
    )
    core = {
        "status": "complete",
        "protocol": PROTOCOL,
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": rows[0]["source_contract_sha256"],
        "selection_horizon": SELECTION_HORIZON,
        "model_seeds": list(MODEL_SEEDS),
        "selected_episodes": {
            str(row["model_seed"]): row["selected_episode"] for row in rows
        },
        "frozen_checkpoint_sha256": {
            str(row["model_seed"]): row["frozen_checkpoint_sha256"] for row in rows
        },
        "protocol_training_complete": False,
        "development_only": True,
        "test_panels_opened": False,
        "paper_evidence": False,
    }
    manifest = {**core, "manifest_core_sha256": digest(core)}
    _atomic_json(output.resolve() / MANIFEST_NAME, manifest)
    return {**manifest, "seed_manifests": [row["manifest"] for row in rows]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "freeze", "freeze-all"))
    parser.add_argument("--source-output", type=Path, default=SOURCE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=MODEL_SEEDS)
    args = parser.parse_args()
    if args.command == "inspect":
        require(args.model_seed is not None, "inspect requires --model-seed")
        result = inspect_seed(args.source_output, args.model_seed)
    elif args.command == "freeze":
        require(args.model_seed is not None, "freeze requires --model-seed")
        result = freeze_seed(args.source_output, args.output, args.model_seed)
    else:
        result = freeze_all(args.source_output, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
