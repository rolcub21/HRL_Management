#!/usr/bin/env python3
"""Run host-independent seed-1/2 compute shards for the frozen E222 panel."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191_representation,
)
from experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1 import (
    run as e203,
)
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import (
    scenario,
)

from . import protocol as p


class E224Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E224Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E224 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E224 JSON object: {path}")
    return value


def _anchor_contract() -> dict:
    # E203 needs only the frozen identity when writing a compute shard.  The
    # full E222 contract remains at the panel owner and authenticates imports.
    return {
        "protocol": p.ROW_PROTOCOL,
        "contract_sha256": p.ANCHOR_CONTRACT_SHA256,
    }


PROXY = SimpleNamespace(
    PROTOCOL=p.ROW_PROTOCOL,
    MODEL_SEEDS=p.MODEL_SEEDS,
    METHODS=("dynamic_pslap", "ga", "e191"),
    EXPECTED_ROWS=240,
    ROOT=p.ROOT,
    contract=_anchor_contract,
    digest=p.digest,
    sha256=p.sha256,
)


def _validate_sources() -> None:
    mismatches = {}
    for relative, expected in p.SOURCE_SHA256.items():
        path = p.ROOT / relative
        actual = p.sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = (actual, expected)
    require(
        not mismatches,
        "E224 live sources differ from the opened E222 panel: " + repr(mismatches),
    )


def _freeze(model_seed: int) -> dict:
    seed = int(model_seed)
    require(seed in p.MODEL_SEEDS, "E224 model seed must be 1 or 2")
    expected = p.CHECKPOINTS[seed]
    summary_path, checkpoint_path = p.checkpoint_paths(seed)
    summary = _read(summary_path)
    require(
        p.sha256(summary_path) == expected["summary_sha256"],
        f"E224 seed {seed} summary differs from E222",
    )
    require(
        checkpoint_path.is_file()
        and p.sha256(checkpoint_path) == expected["checkpoint_sha256"],
        f"E224 seed {seed} best checkpoint differs from E222",
    )
    selected = summary.get("best_validation_record", {})
    require(
        summary.get("status") == "complete"
        and int(summary.get("completed_training_episodes", -1)) == 300
        and int(selected.get("checkpoint_episode", -1))
        == int(expected["selected_episode"]),
        f"E224 seed {seed} is not its finalized E218 selection",
    )
    return {
        "status": "complete",
        "model_seed": seed,
        "selected_episode": int(expected["selected_episode"]),
        "frozen_checkpoint": str(checkpoint_path.resolve()),
        "frozen_checkpoint_sha256": expected["checkpoint_sha256"],
    }


def _load_agent(model_seed: int, device: str):
    frozen = _freeze(model_seed)
    payload = torch.load(
        frozen["frozen_checkpoint"], map_location="cpu", weights_only=False
    )
    expected = {
        "model_seed": int(model_seed),
        "completed_training_episodes": 300,
        "checkpoint_role": "best_deployment_finalized",
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "test_panels_opened": False,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    require(not mismatches, f"E224 checkpoint metadata differs: {mismatches}")
    agent = e191_representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=int(model_seed)
    )
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    return agent, frozen


@contextmanager
def _engine():
    with (
        patch.object(e203, "p", PROXY),
        patch.object(e203, "scenario", scenario),
        patch.object(e203, "_freeze", _freeze),
        patch.object(e203, "_load_agent", _load_agent),
    ):
        yield


def gate(model_seed: int) -> dict:
    _validate_sources()
    frozen = _freeze(model_seed)
    with _engine():
        manifest = e203.instance_manifest(_anchor_contract())
    require(
        manifest["manifest_sha256"] == p.ANCHOR_INSTANCE_MANIFEST_SHA256,
        "E224 reconstructed another E222 instance panel",
    )
    return {
        "status": "pass",
        "protocol": p.PROTOCOL,
        "row_protocol": p.ROW_PROTOCOL,
        "anchor_contract_sha256": p.ANCHOR_CONTRACT_SHA256,
        "anchor_instance_manifest_sha256": p.ANCHOR_INSTANCE_MANIFEST_SHA256,
        "model_seed": int(model_seed),
        "selected_episode": frozen["selected_episode"],
        "checkpoint_sha256": frozen["frozen_checkpoint_sha256"],
        "test_panel_opened": True,
        "training_runs": 0,
        "episode_runs": 0,
    }


def run_seed(output: Path, model_seed: int, device: str) -> dict:
    seed = int(model_seed)
    gate(seed)
    output = output.resolve()
    with _engine():
        e203.prepare(output)
        rows = [
            e203.run_vcg_one(output, seed, instance_seed, device)
            for instance_seed in scenario.TEST_SEEDS
        ]
    return {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "row_protocol": p.ROW_PROTOCOL,
        "model_seed": seed,
        "rows": len(rows),
        "fresh_episode_runs": sum(int(row.get("episode_runs", 0)) for row in rows),
        "row_directory": str(
            (output / "rows/e191" / f"model-seed-{seed}").resolve()
        ),
        "anchor_contract_sha256": p.ANCHOR_CONTRACT_SHA256,
        "test_panel_opened": True,
        "training_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate", "run-seed"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "gate":
        result = gate(args.model_seed)
    else:
        result = run_seed(args.output, args.model_seed, args.device)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

