#!/usr/bin/env python3
"""Freeze the requested-distance law from the completed reward-free R47 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "current_behavior_alignment_r47"
)
DEFAULT_OUTPUT = HERE / "artifacts/command_distance_bank_r47.npz"
EXPECTED_SUMMARY_SHA256 = "0a007cbb8f9af97b9e02cde9aad35723336176b56fcba3075cdef70840219bfe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    summary_path = source / "summary.json"
    if sha256(summary_path) != EXPECTED_SUMMARY_SHA256:
        raise ValueError("The frozen R47 command source changed")
    summary = json.loads(summary_path.read_text())
    if summary.get("rows") != 2000 or summary.get("task_reward_used") is not False:
        raise ValueError("R47 is not the completed reward-free 2,000-command source")

    shards = sorted((source / "shards").glob("task_*.npz"))
    if len(shards) != 40:
        raise ValueError(f"Expected 40 R47 shards, found {len(shards)}")
    distances = []
    shard_hashes = []
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as shard:
            starts = np.asarray(shard["states"][:, 0, :2], dtype=np.float64)
            commands = np.asarray(shard["commands"], dtype=np.float64)
            reported = np.asarray(shard["selected_command_distance"], dtype=np.float64)
        computed = np.linalg.norm(commands - starts, axis=1)
        if not np.allclose(computed, reported, atol=2e-5, rtol=2e-5):
            raise ValueError(f"R47 requested distances disagree in {shard_path}")
        distances.append(computed)
        shard_hashes.append(sha256(shard_path))
    distances = np.concatenate(distances).astype(np.float32)
    if distances.shape != (2000,) or not np.isfinite(distances).all():
        raise ValueError("R47 distance bank is incomplete")
    # Exact zero cannot define a direction; R47 has none, so this is an audit,
    # not a filtering rule.
    if np.any(distances <= 0.0):
        raise ValueError("R47 unexpectedly contains a zero-distance command")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        requested_distance_m=distances,
        source_shard_sha256=np.asarray(shard_hashes, dtype="<U64"),
    )
    quantiles = {
        str(q): float(np.quantile(distances, q))
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    audit = {
        "schema_version": 1,
        "artifact": str(args.output.resolve()),
        "artifact_sha256": sha256(args.output.resolve()),
        "source": str(source),
        "source_summary_sha256": EXPECTED_SUMMARY_SHA256,
        "rows": int(len(distances)),
        "requested_distance_mean_m": float(distances.mean()),
        "requested_distance_quantiles_m": quantiles,
        "task_reward_used": False,
        "hidden_goals_used": False,
    }
    audit_path = args.output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

