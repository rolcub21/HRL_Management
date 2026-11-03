#!/usr/bin/env python3
"""Authenticate E218 and train Stage B independently for seeds 1 and 2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from PSLAP.viability_candidates import BoundedEventDeferRule
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191_representation,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    run as e204,
)
from experiments.conditioned_vcg.E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1 import (
    run as e214,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)
from train_viability_graph_smdp import resolve_device

from . import protocol as p


class E221Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E221Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E221 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E221 JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _declaration(contract: dict) -> dict:
    return {
        key: deepcopy(contract[key])
        for key in (
            "method_reference", "method_revision", "claim_level",
            "declared_deviations", "oracle_inputs", "nonconforming_shortcuts",
            "conformance_audit",
        )
    }


def prepare(output: Path) -> dict:
    output = output.resolve()
    path = output / "contract.json"
    if path.is_file():
        contract = _read(path)
        payload = {key: value for key, value in contract.items()
                   if key != "contract_sha256"}
        require(contract.get("protocol") == p.PROTOCOL,
                "E221 protocol identity differs")
        require(p.digest(payload) == contract.get("contract_sha256"),
                "E221 contract digest differs")
        render_validated_conformance_audit(_declaration(contract))
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E221 refuses a nonempty uncontracted output")
        contract = p.contract()
        validate_protocol(contract, source=p.PROTOCOL)
        _write(path, contract)
        _write(output / "schedule.json", {
            "protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "training_schedule": contract["training_schedule"],
        })
        _write(output / "import-manifest.json", {
            "status": "authenticated",
            "protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "frozen_stage_a": contract["frozen_stage_a"],
            "test_panel_opened": False,
        })
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "model_seeds": list(p.MODEL_SEEDS),
        "rounds_per_seed": p.ROUNDS,
        "training_episodes_per_seed": p.TRAIN_EPISODES_PER_SEED,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    prepare(output)
    contract = _read(output.resolve() / "contract.json")
    for relative, expected in contract["source_sha256"].items():
        path = p.ROOT / relative
        require(path.is_file() and p.sha256(path) == expected,
                f"E221 behavioral source changed: {relative}")
    for frozen in contract["frozen_stage_a"]:
        for key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("summary", "summary_sha256"),
            ("training_contract", "training_contract_sha256"),
        ):
            require(p.sha256(Path(frozen[key])) == frozen[hash_key],
                    f"E221 imported seed {frozen['model_seed']} {key} changed")
    return contract


def _search(payload: dict) -> ViabilitySearchConfig:
    value = payload["viability_search"]
    return ViabilitySearchConfig(
        max_depth=value["max_depth"],
        max_nodes=int(value["max_nodes"]),
        max_primitive_steps=value["max_primitive_steps"],
        reserve_queue_cells=bool(value["reserve_queue_cells"]),
        search_order=str(value["search_order"]),
    )


def _liveness(payload: dict) -> BoundedEventDeferRule:
    value = payload["liveness_rule"]
    return BoundedEventDeferRule(
        max_option_steps=int(value["max_option_steps"]),
        max_consecutive_defer_decisions=int(value["max_consecutive_defer_decisions"]),
    )


def _frozen(contract: dict, model_seed: int) -> dict:
    rows = [row for row in contract["frozen_stage_a"]
            if int(row["model_seed"]) == int(model_seed)]
    require(len(rows) == 1, "E221 frozen Stage A is not unique")
    return rows[0]


def load_base(contract: dict, model_seed: int, device: torch.device):
    frozen = _frozen(contract, model_seed)
    checkpoint = Path(frozen["checkpoint"])
    require(checkpoint.is_file() and p.sha256(checkpoint) == frozen["checkpoint_sha256"],
            "E221 frozen E218 checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expected = {
        "training_protocol": frozen["source_protocol"],
        "model_seed": int(model_seed),
        "checkpoint_role": "best_deployment_finalized",
        "completed_training_episodes": 300,
        "selected_checkpoint_episode": int(frozen["selected_episode"]),
        "trainer_resumable": False,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "selection_finalized_at_training_episode": 300,
        "test_panels_opened": False,
        "e187_contract_sha256": frozen["source_contract_sha256"],
        "resume_contract_sha256": frozen["resume_contract_sha256"],
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items()
                if payload.get(key) != value}
    require(not mismatch, f"E221 frozen E218 identity differs: {mismatch}")
    selected = payload.get("best_validation_record", {})
    require(tuple(selected.get("selection_score", ()))
            == tuple(frozen["selection_score"]),
            "E221 checkpoint selection score differs")
    search = _search(payload)
    liveness = _liveness(payload)
    base = e191_representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=int(model_seed)
    )
    base.set_epsilon(0.0)
    for network in (base.Q_local, base.Q_target):
        network.requires_grad_(False).eval()
    return base, search, liveness, e204.state_digest(base.Q_local.state_dict()), frozen


def _seed_contract(contract: dict, model_seed: int) -> dict:
    value = deepcopy(contract)
    value["frozen_stage_a"] = [deepcopy(_frozen(contract, model_seed))]
    return value


def _engine_protocol(model_seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        ROOT=p.ROOT,
        PROTOCOL=p.PROTOCOL,
        DEFAULT_OUTPUT=p.DEFAULT_OUTPUT,
        MODEL_SEED=int(model_seed),
        SCENARIO=p.SCENARIO,
        ROUNDS=p.ROUNDS,
        EPISODES_PER_ROUND=p.EPISODES_PER_ROUND,
        TRAIN_EPISODES=p.TRAIN_EPISODES_PER_SEED,
        TRACE_ETA=p.TRACE_ETA,
        OPTIMIZER_LR=p.OPTIMIZER_LR,
        FIT_EPOCHS_PER_ROUND=p.FIT_EPOCHS_PER_ROUND,
        FIT_BATCH_SIZE=p.FIT_BATCH_SIZE,
        HEAD_SEED=p.head_seed(model_seed),
        sha256=p.sha256,
        digest=p.digest,
    )


@contextmanager
def _engine(contract: dict, model_seed: int):
    view = _seed_contract(contract, model_seed)
    namespace = _engine_protocol(model_seed)
    with (
        patch.object(e214, "p", namespace),
        patch.object(e214, "authenticate", lambda _output: view),
        patch.object(e204, "load_base", load_base),
    ):
        yield namespace


def self_check(output: Path) -> dict:
    contract = authenticate(output)
    rows = {}
    digests = set()
    for seed in p.MODEL_SEEDS:
        base, search, liveness, base_digest, frozen = load_base(
            contract, seed, torch.device("cpu")
        )
        with _engine(contract, seed):
            head = e214.new_agent(base)
        require(head.config.lambda_center == 10.0
                and head.config.lambda_max == 40.0,
                "E221 preference range differs")
        digests.add(base_digest)
        rows[str(seed)] = {
            "selected_episode": frozen["selected_episode"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "base_policy_digest": base_digest,
            "head_seed": p.head_seed(seed),
            "search_max_nodes": search.max_nodes,
            "liveness_max_option_steps": liveness.max_option_steps,
        }
    require(len(digests) == len(p.MODEL_SEEDS),
            "E221 imported base policies are not distinct")
    return {
        "status": "self_check_passed",
        "seeds": rows,
        "training_episodes_per_seed": p.TRAIN_EPISODES_PER_SEED,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def train_seed(output: Path, model_seed: int, through_round: int,
               device_name: str) -> dict:
    contract = authenticate(output)
    require(int(model_seed) in p.MODEL_SEEDS, "E221 model seed must be 1 or 2")
    require(1 <= int(through_round) <= p.ROUNDS,
            "E221 through-round must be 1..4")
    with _engine(contract, model_seed):
        result = e214.train(output.resolve(), int(through_round), device_name)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "source_stage_a_protocol": contract["source_stage_a_protocol"],
        "source_stage_a_episode": _frozen(contract, model_seed)["selected_episode"],
        "stage_b_procedure": p.STAGE_B_PROCEDURE["name"],
        "paper_evidence": False,
    }


def status(output: Path) -> dict:
    contract = authenticate(output)
    rows = {}
    for seed in p.MODEL_SEEDS:
        summary_path = (
            output.resolve() / "training" / f"seed-{seed}"
            / "training-summary.json"
        )
        summary = _read(summary_path) if summary_path.is_file() else {}
        endpoint = (
            output.resolve() / "training" / f"seed-{seed}"
            / "checkpoints" / f"round-{p.ROUNDS}.pth"
        )
        rows[str(seed)] = {
            "status": summary.get("status", "not_started"),
            "completed_rounds": int(summary.get("completed_rounds", 0)),
            "completed_training_episodes": int(
                summary.get("completed_training_episodes", 0)
            ),
            "endpoint_ready": endpoint.is_file(),
        }
    return {
        "status": "available",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "seeds": rows,
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "train-seed", "status")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-round", type=int, default=p.ROUNDS)
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
            args.output, args.model_seed, args.through_round, args.device
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
