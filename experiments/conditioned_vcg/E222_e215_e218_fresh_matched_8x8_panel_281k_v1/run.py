#!/usr/bin/env python3
"""Gate, prepare, and run the E222 fresh matched 8x8 panel."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    representation as e191_representation,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1 import (
    run as e203,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p, scenario


class E222Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E222Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E222 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E222 JSON object: {path}")
    return value


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _identity(seed: int) -> dict:
    value = p.checkpoint_identity(seed)
    payload = torch.load(
        value["source_checkpoint"], map_location="cpu", weights_only=False
    )
    expected = {
        "model_seed": seed,
        "completed_training_episodes": 300,
        "checkpoint_role": "best_deployment_finalized",
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "test_panels_opened": False,
    }
    mismatches = {
        key: (payload.get(key), expected_value)
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    require(not mismatches, f"E222 checkpoint seed {seed} metadata differs: {mismatches}")
    require(
        int(payload.get("best_validation_record", {}).get("checkpoint_episode", -1))
        == int(value["selected_episode"]),
        f"E222 checkpoint seed {seed} selected episode differs",
    )
    return value


def _freeze(seed: int) -> dict:
    value = _identity(seed)
    return {
        "status": "complete",
        "model_seed": seed,
        "selected_episode": int(value["selected_episode"]),
        "frozen_checkpoint": value["source_checkpoint"],
        "frozen_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_contract_sha256": value["source_contract_sha256"],
    }


def _load_agent(seed: int, device: str):
    frozen = _freeze(seed)
    payload = torch.load(
        frozen["frozen_checkpoint"], map_location="cpu", weights_only=False
    )
    agent = e191_representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=seed
    )
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    return agent, frozen


@contextmanager
def _implementation(protocol, scenario_module):
    with (
        patch.object(e203, "p", protocol),
        patch.object(e203, "scenario", scenario_module),
        patch.object(e203, "_freeze", _freeze),
        patch.object(e203, "_load_agent", _load_agent),
    ):
        yield


class _GateEnv(e191_scenario.LongStayCrowdingEnv):
    pass


GATE_SCENARIO = SimpleNamespace(
    SCENARIO=e191.SCENARIO,
    LEVELS=(0.8,),
    GAPS=e191_scenario.GAPS,
    TOTAL_JOBS=e191_scenario.TOTAL_JOBS,
    CAPACITY=e191_scenario.CAPACITY,
    MEAN_DURATION=e191_scenario.MEAN_DURATION,
    MAX_STEPS=e191_scenario.MAX_STEPS,
    CASES_PER_LEVEL=1,
    TEST_SEEDS=(p.GATE_SEED,),
    TestPanel8x8Env=_GateEnv,
    level_for_seed=lambda seed: 0.8 if int(seed) == p.GATE_SEED else (_ for _ in ()).throw(ValueError("outside E222 gate")),
    replicate_for_seed=lambda seed: 0 if int(seed) == p.GATE_SEED else (_ for _ in ()).throw(ValueError("outside E222 gate")),
    level_label=lambda level: f"{100.0 * float(level):.0f}%",
)


def _gate_contract() -> dict:
    return p.gate_contract()


def _gate_protocol():
    contract = _gate_contract()
    return SimpleNamespace(
        PROTOCOL=p.GATE_PROTOCOL,
        MODEL_SEEDS=(0,),
        METHODS=("dynamic_pslap", "ga", "e191"),
        DYNAMIC_REPLICATES=(0,),
        GA_REPLICATES=(0,),
        VCG_REPLICATES=(0,),
        EXPECTED_ROWS=3,
        REPORT_NAME="gate-report.json",
        TABLE_NAME="gate-table.md",
        ROOT=p.ROOT,
        contract=lambda: contract,
        digest=p.digest,
        sha256=p.sha256,
        replicates=lambda method: (0,),
    )


def gate() -> dict:
    output = p.DEFAULT_GATE_OUTPUT.resolve()
    final = output / p.GATE_NAME
    if final.is_file():
        value = p.gate_manifest()
        return {**value, "reused": True, "episode_runs": 0}
    protocol = _gate_protocol()
    with _implementation(protocol, GATE_SCENARIO):
        e203.prepare(output)
        vcg = e203.run_vcg_one(output, 0, p.GATE_SEED, "cuda")
        dynamic = e203.run_baseline_one(output, "dynamic_pslap", p.GATE_SEED, 0)
        ga = e203.run_baseline_one(output, "ga", p.GATE_SEED, 0)
    rows = (vcg, dynamic, ga)
    require(all(row.get("strict_complete") for row in rows), "E222 live gate did not complete every route")
    require(vcg["run"].get("horizon_cap_hit") is False, "E222 VCG gate hit its horizon")
    require(vcg["run"].get("objective_audit_integrity") is True, "E222 VCG objective audit failed")
    require(vcg["metrics"].get("objective_accounting_exact") is True, "E222 VCG objective accounting failed")
    row_paths = sorted((output / "rows").glob("**/*.json"))
    require(len(row_paths) == 3, "E222 gate retained an unexpected row count")
    core = {
        "status": "pass",
        "protocol": p.GATE_PROTOCOL,
        "development_seed": p.GATE_SEED,
        "routes_completed": ["e191", "dynamic_pslap", "ga"],
        "strict_completion": {row["method"]: bool(row["strict_complete"]) for row in rows},
        "effective_compute_limits": {
            "max_steps": e191_scenario.MAX_STEPS,
            "vcg_viability_search": torch.load(
                _freeze(0)["frozen_checkpoint"], map_location="cpu", weights_only=False
            )["viability_search"],
            "ga_replicates_in_panel": len(p.GA_REPLICATES),
        },
        "required_audit_signals": {
            "objective_audit_integrity": True,
            "objective_accounting_exact": True,
            "unknown_candidates_rejected": int(vcg["run"]["audit"].get("unknown_candidates_rejected", 0)),
        },
        "checkpoint_sha256": {
            str(seed): _freeze(seed)["frozen_checkpoint_sha256"]
            for seed in p.MODEL_SEEDS
        },
        "row_sha256": {str(path.relative_to(output)): p.sha256(path) for path in row_paths},
        "gate_contract_sha256": _gate_contract()["contract_sha256"],
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    manifest = {**core, "manifest_sha256": p.digest(core)}
    _write(final, manifest)
    return {**manifest, "episode_runs": sum(int(row.get("episode_runs", 0)) for row in rows)}


def _recorded_declaration(contract: dict) -> dict:
    keys = (
        "method_reference", "method_revision", "claim_level",
        "declared_deviations", "oracle_inputs", "nonconforming_shortcuts",
        "conformance_audit",
    )
    value = {key: deepcopy(contract[key]) for key in keys}
    render_validated_conformance_audit(value)
    return value


def prepare(output: Path) -> dict:
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    with _implementation(p, scenario):
        result = e203.prepare(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "frozen_checkpoints": p.checkpoint_identities(),
        "development_gate_sha256": contract["development_gate"]["manifest_sha256"],
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    with _implementation(p, scenario):
        return e203.authenticate(output)


def self_check(output: Path) -> dict:
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    with _implementation(p, scenario):
        result = e203.self_check(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "conformance": _recorded_declaration(contract),
        "selected_episodes": {
            str(seed): p.checkpoint_identity(seed)["selected_episode"]
            for seed in p.MODEL_SEEDS
        },
        "test_panel_opened": False,
    }


def run_vcg_one(output: Path, model_seed: int, seed: int, device: str) -> dict:
    with _implementation(p, scenario):
        return e203.run_vcg_one(output, model_seed, seed, device)


def run_baseline_one(output: Path, method: str, seed: int, replicate: int) -> dict:
    with _implementation(p, scenario):
        return e203.run_baseline_one(output, method, seed, replicate)


def analyze(output: Path) -> dict:
    with _implementation(p, scenario):
        result = e203.analyze(output)
    report_path = output.resolve() / p.REPORT_NAME
    report = _read(report_path)
    report["method_label"] = "VCG long-exploration (E215 seed 0; E218 seeds 1/2)"
    report["selected_episodes"] = {
        str(seed): p.checkpoint_identity(seed)["selected_episode"]
        for seed in p.MODEL_SEEDS
    }
    report["fresh_panel"] = True
    _write(report_path, report)
    table = output.resolve() / p.TABLE_NAME
    if table.is_file():
        text = table.read_text(encoding="utf-8")
        text = text.replace("# E203 — E191", "# E222 — finalized long-exploration VCG")
        text = text.replace("E191 seed", "VCG seed").replace("E191 pooled", "VCG pooled")
        text = text.replace("All E191 checkpoints", "All VCG checkpoints")
        table.write_text(text, encoding="utf-8")
    return {**result, "protocol": p.PROTOCOL, "paper_evidence": report["paper_evidence"]}


def _coordinates(command: str, method: str | None, model_seed: int | None):
    if command == "run-method":
        require(method is not None, "run-method requires --method")
        if method == "e191":
            seeds = p.MODEL_SEEDS if model_seed is None else (model_seed,)
            return tuple(("e191", seed, instance, 0) for seed in seeds for instance in scenario.TEST_SEEDS)
        return tuple((method, None, instance, replicate) for instance in scenario.TEST_SEEDS for replicate in p.replicates(method))
    return tuple(
        (method, None, instance, replicate)
        for method in ("dynamic_pslap", "ga")
        for instance in scenario.TEST_SEEDS
        for replicate in p.replicates(method)
    ) + tuple(
        ("e191", seed, instance, 0)
        for seed in p.MODEL_SEEDS for instance in scenario.TEST_SEEDS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "gate", "prepare", "self-check", "run-one", "run-method", "run-all",
        "analyze", "status",
    ))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=p.METHODS)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--seed", type=int, choices=scenario.TEST_SEEDS)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.command == "gate":
        result = gate()
    elif args.command == "prepare":
        result = prepare(output)
    elif args.command == "self-check":
        result = self_check(output)
    elif args.command in ("analyze", "status"):
        result = analyze(output)
    else:
        prepare(output)
        if args.command == "run-one":
            require(args.method is not None and args.seed is not None, "run-one requires --method and --seed")
            coordinates = ((args.method, args.model_seed if args.method == "e191" else None, args.seed, args.replicate),)
            if args.method == "e191":
                require(args.model_seed is not None, "VCG requires --model-seed")
        else:
            coordinates = _coordinates(args.command, args.method, args.model_seed)
        rows = []
        for method, model_seed, seed, replicate in coordinates:
            if method == "e191":
                rows.append(run_vcg_one(output, model_seed, seed, args.device))
            else:
                rows.append(run_baseline_one(output, method, seed, replicate))
        result = {
            "status": "complete",
            "rows": len(rows),
            "fresh_episode_runs": sum(int(row.get("episode_runs", 0)) for row in rows),
            "training_runs": 0,
            "analysis": analyze(output),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
