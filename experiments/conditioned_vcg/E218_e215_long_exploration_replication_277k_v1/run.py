#!/usr/bin/env python3
"""Prepare and train the seed-1/2 replication of E215."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    run as e191_run,
    stage as e191_stage,
)

from . import protocol as p


class E218Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_e215_long_exploration_replication"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E218Error(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"required E218 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E218 JSON object: {path}")
    return value


def _recorded_declaration(contract: dict) -> dict:
    keys = (
        "method_reference", "method_revision", "claim_level",
        "declared_deviations", "oracle_inputs", "nonconforming_shortcuts",
        "conformance_audit",
    )
    value = {key: deepcopy(contract[key]) for key in keys}
    render_validated_conformance_audit(value)
    return value


def _conformance(output: Path | None = None) -> dict:
    # Once an expensive run has started, its launch-validated declaration is
    # authoritative even if the living method reference later advances.
    if output is not None and (output.resolve() / "contract.json").is_file():
        return _recorded_declaration(_read(output.resolve() / "contract.json"))
    return validate_protocol(p.contract(), source=p.PROTOCOL)


@contextmanager
def implementation():
    with (
        patch.object(e191_run, "p", p),
        patch.object(e191_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e191_stage, "p", p),
    ):
        yield


def prepare(output: Path) -> dict:
    _conformance(output)
    with implementation():
        result = e191_run.prepare(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "model_seeds": list(p.MODEL_SEEDS),
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "target_floor_episode_approximate": p.EPSILON_TARGET_FLOOR_EPISODE,
        "matched_to_e215": True,
    }


def authenticate(output: Path) -> dict:
    expected = p.contract()
    observed = _read(output.resolve() / "contract.json")
    require(observed == expected, "E218 contract or behavioral sources changed")
    _conformance(output)
    return observed


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract = authenticate(output)
    require(p.MODEL_SEEDS == (1, 2), "E218 seeds must be exactly 1 and 2")
    require(p.EPSILON_DECAY_DECISIONS == 57_300,
            "E218 epsilon schedule differs from E215")
    require(contract["replication"]["replay_capacity"] == 20_000,
            "E218 replay capacity differs from E215")
    require(contract["replication"]["acceleration_unchanged"] is True,
            "E218 acceleration differs from E215")
    return {
        **prepared,
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "fresh_model_seeds": list(p.MODEL_SEEDS),
        "test_panel_opened": False,
    }


def train_seed(output: Path, model_seed: int, through_episode: int,
               device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    require(model_seed in p.MODEL_SEEDS, "E218 model seed must be 1 or 2")
    require(1 <= through_episode <= p.TOTAL_EPISODES,
            "E218 through-episode must be between 1 and 300")
    require(device == "cuda" and torch.cuda.is_available(),
            "E218 training requires CUDA")
    with implementation():
        result = e191_run.train_seed(
            output, model_seed, through_episode, device
        )
    result.pop("e191_protocol", None)
    result.pop("e191_contract_sha256", None)
    return {
        **result,
        "e218_protocol": p.PROTOCOL,
        "e218_contract_sha256": contract["contract_sha256"],
        "matched_to_e215": True,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "paper_evidence": False,
    }


def status(output: Path) -> dict:
    contract = authenticate(output)
    seeds = {}
    for seed in p.MODEL_SEEDS:
        path = (
            output.resolve() / "training/operational" / p.SCENARIO
            / f"seed-{seed}/training-summary.json"
        )
        if path.is_file():
            summary = _read(path)
            seeds[str(seed)] = {
                "completed_training_episodes": int(
                    summary.get("completed_training_episodes", 0)
                ),
                "latest_checkpoint": summary.get("latest_checkpoint"),
            }
        else:
            seeds[str(seed)] = {"completed_training_episodes": 0}
    return {
        "status": "available",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "seeds": seeds,
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "train-seed", "status")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=220)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "status":
        result = status(args.output)
    else:
        require(args.model_seed is not None,
                "train-seed requires --model-seed 1 or 2")
        result = train_seed(
            args.output, args.model_seed, args.through_episode, args.device
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
