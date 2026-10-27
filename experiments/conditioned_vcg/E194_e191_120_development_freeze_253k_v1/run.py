#!/usr/bin/env python3
"""Inspect and freeze E191's selected episode-100/120 development checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import torch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191_protocol,
    run as e191_run,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e194_e191_120_development_freeze_253k_v1"
SOURCE_OUTPUT = (
    ROOT / "results/vcg-conditioned-e191-fresh-long-stay-crowding-250k-v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e194-e191-120-development-freeze-253k-v1"
)
SCENARIO = e191_protocol.SCENARIO
MODEL_SEED = 0
SELECTION_HORIZON = 120
CANDIDATE_EPISODES = (100, 120)
EXPECTED_VALIDATION_EPISODES = tuple(range(20, SELECTION_HORIZON + 1, 20))
SCENARIO_RELATIVE = Path(
    f"training/operational/{SCENARIO}/seed-{MODEL_SEED}"
)
SOURCE_SUMMARY_RELATIVE = SCENARIO_RELATIVE / "training-summary.json"
FROZEN_NAME = "selected-development-checkpoint.pth"
MANIFEST_NAME = "freeze.json"
CANDIDATE_NAMES = (
    "best-candidate-slot-a.pth",
    "best-candidate-slot-b.pth",
)


class FreezeError(RuntimeError):
    """Raised when the through-120 checkpoint cannot be frozen safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def record_identity(record: Mapping) -> tuple:
    return (
        int(record["checkpoint_episode"]),
        tuple(float(value) for value in record["selection_score"]),
        tuple(str(value) for value in record["selection_score_fields"]),
    )


def score(record: Mapping) -> tuple[float, ...]:
    return tuple(float(value) for value in record["selection_score"])


def select_record(records: Sequence[Mapping]) -> Mapping:
    """Select between episodes 100 and 120; an exact tie keeps episode 100."""

    by_episode = {int(record["checkpoint_episode"]): record for record in records}
    require(
        tuple(sorted(by_episode)) == EXPECTED_VALIDATION_EPISODES,
        "E191 must contain exactly the through-120 validation history",
    )
    first = by_episode[CANDIDATE_EPISODES[0]]
    second = by_episode[CANDIDATE_EPISODES[1]]
    chosen = second if score(second) > score(first) else first
    authoritative = max(records, key=score)
    require(
        record_identity(authoritative) == record_identity(chosen),
        "an earlier checkpoint outranks the requested episode-100/120 pair",
    )
    return chosen


def _candidate_summary(record: Mapping) -> dict:
    summary = record["summary"]
    groups = summary["e191_by_workload"]
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "selection_score_fields": list(record["selection_score_fields"]),
        "selection_score": [float(value) for value in record["selection_score"]],
        "overall": {
            "J": float(summary["mean_e187_J"]),
            "storage_clock_mae": float(summary["mean_storage_clock_mae"]),
            "admission_wait": float(summary["mean_admission_wait"]),
            "rehandles_per_100": float(
                summary["relocations_per_100_deliveries"]
            ),
            "peak_stored": float(summary["mean_peak_stored"]),
            "strict_completion_rate": float(
                summary["strict_method_success_rate"]
            ),
        },
        "by_workload_J": {
            level: float(groups[level]["means"]["J"])
            for level in ("20%", "50%", "80%")
        },
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(
            sha256(destination) == sha256(source),
            f"refusing to overwrite a different frozen checkpoint: {destination}",
        )
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    require(not temporary.exists(), f"temporary path already exists: {temporary}")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
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
        require(
            path.read_bytes() == encoded,
            f"refusing to overwrite a different freeze manifest: {path}",
        )
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    require(not temporary.exists(), f"temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inspect(source_output: Path) -> dict:
    source_output = source_output.resolve()
    authenticated = e191_run.authenticate(source_output)
    summary_path = source_output / SOURCE_SUMMARY_RELATIVE
    require(summary_path.is_file(), f"missing E191 training summary: {summary_path}")
    summary = read_json(summary_path)
    require(
        int(summary["completed_training_episodes"]) == SELECTION_HORIZON,
        "E191 seed 0 must be paused exactly after episode 120",
    )
    require(summary.get("status") == "paused", "E191 summary must be paused")
    require(not summary.get("test_panels_opened"), "E191 test panels were opened")
    require(not summary.get("performance_claim_authorized"), "unexpected claim state")

    records = list(summary["validation_history"])
    selected = select_record(records)
    stored_best = summary.get("best_validation_record")
    require(isinstance(stored_best, Mapping), "E191 summary lacks its best record")
    require(
        record_identity(stored_best) == record_identity(selected),
        "E191 trainer selection differs from the through-120 selector",
    )

    seed_directory = (source_output / SCENARIO_RELATIVE).resolve()
    candidate_name = summary.get("best_candidate_checkpoint_filename")
    require(candidate_name in CANDIDATE_NAMES, "invalid E191 candidate filename")
    candidate_path = (seed_directory / str(candidate_name)).resolve()
    require(candidate_path.parent == seed_directory, "candidate escaped seed directory")
    require(candidate_path.is_file(), f"missing E191 candidate: {candidate_path}")
    candidate_hash = sha256(candidate_path)
    require(
        candidate_hash == summary.get("best_candidate_checkpoint_sha256"),
        "E191 candidate SHA-256 differs from the training summary",
    )

    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping), "E191 candidate is not a mapping")
    expected = {
        "training_protocol": e191_protocol.PROTOCOL,
        "method_version": e191_protocol.METHOD_VERSION,
        "checkpoint_role": "best_candidate_unfinalized",
        "model_seed": MODEL_SEED,
        "completed_training_episodes": int(selected["checkpoint_episode"]),
        "trainer_resumable": False,
        "protocol_training_complete": False,
        "selection_finalized_after_total_episodes": False,
        "deployment_checkpoint_eligible": False,
        "performance_claim_authorized": False,
        "test_panels_opened": False,
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    require(not mismatches, f"E191 candidate metadata differs: {mismatches}")
    payload_best = payload.get("best_validation_record")
    require(isinstance(payload_best, Mapping), "candidate lacks its best record")
    require(
        record_identity(payload_best) == record_identity(selected),
        "candidate best record differs from the selected record",
    )
    source_contract_sha = authenticated["contract_sha256"]
    require(
        payload.get("e187_contract_sha256") == source_contract_sha,
        "candidate E191 contract SHA-256 differs",
    )

    return {
        "status": "ready",
        "protocol": PROTOCOL,
        "source_protocol": e191_protocol.PROTOCOL,
        "source_contract_sha256": source_contract_sha,
        "source_summary": str(summary_path),
        "source_candidate": str(candidate_path),
        "source_candidate_sha256": candidate_hash,
        "selection_horizon": SELECTION_HORIZON,
        "candidate_episodes": list(CANDIDATE_EPISODES),
        "candidates": {
            str(episode): _candidate_summary(
                next(
                    record
                    for record in records
                    if int(record["checkpoint_episode"]) == episode
                )
            )
            for episode in CANDIDATE_EPISODES
        },
        "selected_episode": int(selected["checkpoint_episode"]),
        "selection_score_fields": list(selected["selection_score_fields"]),
        "selection_score": [float(value) for value in selected["selection_score"]],
        "selection_rule": (
            "authoritative E191 lexicographic score; compare episodes 100 and "
            "120; exact tie retains episode 100; require no earlier winner"
        ),
        "checkpoint_role": "best_candidate_unfinalized",
        "deployment_checkpoint_eligible": False,
        "development_only": True,
        "test_panels_opened": False,
        "paper_evidence": False,
    }


def freeze(source_output: Path, output: Path) -> dict:
    inspected = inspect(source_output)
    output = output.resolve() / f"seed-{MODEL_SEED}"
    checkpoint_path = output / FROZEN_NAME
    manifest_path = output / MANIFEST_NAME
    source_candidate = Path(inspected["source_candidate"])
    _atomic_copy(source_candidate, checkpoint_path)
    frozen_hash = sha256(checkpoint_path)
    require(
        frozen_hash == inspected["source_candidate_sha256"],
        "frozen checkpoint is not byte-identical to its source",
    )
    manifest_core = {
        **inspected,
        "status": "complete",
        "frozen_checkpoint": str(checkpoint_path),
        "frozen_checkpoint_sha256": frozen_hash,
        "byte_identical_to_source_candidate": True,
        "source_mutated": False,
    }
    manifest = {
        **manifest_core,
        "manifest_core_sha256": digest(manifest_core),
    }
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "freeze"))
    parser.add_argument("--source-output", type=Path, default=SOURCE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = (
        inspect(args.source_output)
        if args.command == "inspect"
        else freeze(args.source_output, args.output)
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
