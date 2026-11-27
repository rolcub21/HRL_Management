#!/usr/bin/env python3
"""Bind a completed official CRL seed-0 run to the development protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    epoch = int(protocol["executor"]["training_steps"])
    candidates = sorted(args.run_root.rglob(f"params_{epoch}.pkl"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one completed CRL checkpoint, found {len(candidates)}"
        )
    checkpoint = candidates[0]
    flags_path = checkpoint.parent / "flags.json"
    flags = json.loads(flags_path.read_text())
    agent = flags["agent"]
    checks = {
        "environment_matches": flags["env_name"] == protocol["benchmark"]["dataset"],
        "seed_matches": flags["seed"] == protocol["executor"]["training_seed"],
        "training_steps_match": flags["train_steps"] == epoch,
        "from_scratch": flags["restore_path"] is None,
        "reference_crl": agent["agent_name"] == "crl",
        "official_alpha": agent["alpha"] == protocol["executor"]["alpha"],
        "official_discount": agent["discount"] == protocol["executor"]["discount"],
        "task_evaluation_disabled": flags["eval_tasks"] == 0
        and flags["eval_episodes"] == 0,
        "checkpoint_nonempty": checkpoint.stat().st_size > 0,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "executor": "official_OGBench_CRL",
        "environment": flags["env_name"],
        "training_seed": flags["seed"],
        "training_steps": flags["train_steps"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "flags": str(flags_path.resolve()),
        "flags_sha256": sha256(flags_path),
        "checks": checks,
        "native_evaluation_panel_accessed": False,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"executor record failed: {checks}")


if __name__ == "__main__":
    main()
