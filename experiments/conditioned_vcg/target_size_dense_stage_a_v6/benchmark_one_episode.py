"""Run one isolated exact-v6 episode without changing the source checkpoint."""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from experiments.conditioned_vcg.E79_direct_legal_overdue_progress_full_10x10_v1 import (
    run as continuation,
)


def _load_v6():
    path = Path(__file__).resolve().with_name("acceleration.py")
    spec = importlib.util.spec_from_file_location(
        "e79_isolated_v6_acceleration", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact-v6 acceleration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-through-episode", type=int)
    args = parser.parse_args()

    source = args.source_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_checkpoint = source / "latest.pth"
    source_sha_before = audit._sha256(source_checkpoint)
    device = torch.device(args.device)
    v6 = _load_v6()
    _, _, payload, agent = continuation._load(source, device)
    completed = int(payload["through_episode"])
    if (
        args.expected_through_episode is not None
        and completed != args.expected_through_episode
    ):
        raise RuntimeError(
            f"source is through episode {completed}, expected "
            f"{args.expected_through_episode}"
        )

    output.mkdir(parents=True)
    (output / "heartbeats").mkdir()
    with (
        patch.object(continuation, "acceleration", v6),
        patch.object(builtins, "print"),
    ):
        result = continuation._run_episode(
            agent,
            payload["resume_contract"],
            output,
            completed + 1,
            device,
        )

    source_sha_after = audit._sha256(source_checkpoint)
    if source_sha_before != source_sha_after:
        raise RuntimeError("source checkpoint changed during isolated episode")
    report = {
        "status": "passed" if result["success"] else "episode_failed",
        "scope": "one_isolated_training_episode_no_checkpoint_commit",
        "source_output": str(source),
        "source_through_episode": completed,
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": source_sha_after,
        "source_checkpoint_unchanged": True,
        "episode_number": completed + 1,
        "device": str(device),
        "td_backup_implementation": v6.IMPLEMENTATION,
        "td_backup_audit": v6.audit_payload(agent),
        "result": result,
    }
    report_path = output / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(
            audit._json_safe(report),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    line = continuation._episode_console_line(result, agent)
    compact_path = output / "episode-console.log"
    with compact_path.open("x", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
