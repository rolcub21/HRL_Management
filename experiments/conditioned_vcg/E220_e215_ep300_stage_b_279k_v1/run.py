#!/usr/bin/env python3
"""Train and evaluate frozen E214/E216 Stage B on E215 episode 300."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

import torch

import train_vcg_preference_conditioned as common
from PSLAP.viability_candidates import BoundedEventDeferRule
from PSLAP.viability_filter import ViabilitySearchConfig
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    representation as e191_representation,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    run as e204,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.centered import (
    CenteredHandlingAgent,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    run as e205_run,
)
from experiments.conditioned_vcg.E207_e204_immediate_only_ablation_266k_v1.immediate import (
    from_stage_b_agent,
)
from experiments.conditioned_vcg.E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1 import (
    run as e214,
)
from experiments.conditioned_vcg.E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1.returns import (
    diagnostics,
    examples_from_episode,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)
from train_viability_graph_smdp import resolve_device

from . import protocol as p


class E220Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E220Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E220 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E220 JSON object: {path}")
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


def load_base(contract: dict, model_seed: int, device: torch.device):
    frozen = contract["frozen_stage_a"][0]
    require(int(model_seed) == p.MODEL_SEED == int(frozen["model_seed"]),
            "E220 model seed differs")
    checkpoint = Path(frozen["checkpoint"])
    require(
        checkpoint.is_file() and p.sha256(checkpoint) == frozen["checkpoint_sha256"],
        "E220 frozen E215 checkpoint changed",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expected = {
        "training_protocol": frozen["source_protocol"],
        "model_seed": p.MODEL_SEED,
        "checkpoint_role": "best_deployment_finalized",
        "completed_training_episodes": 300,
        "trainer_resumable": False,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "test_panels_opened": False,
        "e187_contract_sha256": frozen["source_contract_sha256"],
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items() if payload.get(key) != value
    }
    require(not mismatch, f"E220 frozen E215 identity differs: {mismatch}")
    require(
        int(payload.get("best_validation_record", {}).get("checkpoint_episode", -1)) == 300,
        "E220 frozen checkpoint selection is not episode 300",
    )
    search = _search(payload)
    liveness = _liveness(payload)
    base = e191_representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=p.MODEL_SEED
    )
    base.set_epsilon(0.0)
    for network in (base.Q_local, base.Q_target):
        network.requires_grad_(False).eval()
    return base, search, liveness, e204.state_digest(base.Q_local.state_dict()), frozen


def _make_instance(seed: int) -> EpisodeInstance:
    p.workload_for_seed(seed)
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    instance = env.sample_episode_instance(seed=int(seed))
    instance.validate_for(env)
    return instance


def _instance_manifest(contract: dict) -> dict:
    rows = []
    for seed in p.EVAL_INSTANCE_SEEDS:
        instance = _make_instance(seed)
        rows.append({
            "seed": int(seed),
            "workload": p.workload_for_seed(seed),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "instance_json": instance.to_json(),
        })
    core = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "records": rows,
    }
    return {**core, "manifest_sha256": p.digest(core)}


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    if contract_path.is_file():
        contract = _read(contract_path)
        payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
        require(contract.get("protocol") == p.PROTOCOL, "E220 protocol identity differs")
        require(p.digest(payload) == contract.get("contract_sha256"),
                "E220 contract digest differs")
        render_validated_conformance_audit(_declaration(contract))
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E220 refuses a nonempty uncontracted output")
        contract = p.contract()
        validate_protocol(contract, source=p.PROTOCOL)
        _write(contract_path, contract)
        _write(output / "schedule.json", {
            "protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "training_schedule": contract["training_schedule"],
        })
    manifest = _instance_manifest(contract)
    if manifest_path.is_file():
        require(_read(manifest_path) == manifest, "E220 instance manifest changed")
    else:
        _write(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "selected_stage_a_episode": 300,
        "stage_b_rounds": p.ROUNDS,
        "training_episodes": p.TRAIN_EPISODES,
        "endpoint_episode_runs": 15,
        "test_panel_opened": False,
        "paper_evidence": False,
    }


def authenticate(output: Path) -> dict:
    prepare(output)
    contract = _read(output.resolve() / "contract.json")
    for relative, expected in contract["source_sha256"].items():
        path = p.ROOT / relative
        require(path.is_file() and p.sha256(path) == expected,
                f"E220 behavioral source changed: {relative}")
    frozen = contract["frozen_stage_a"][0]
    require(
        p.sha256(Path(frozen["checkpoint"])) == frozen["checkpoint_sha256"],
        "E220 frozen checkpoint changed",
    )
    require(
        p.sha256(Path(frozen["summary"])) == frozen["summary_sha256"],
        "E220 frozen Stage-A summary changed",
    )
    return contract


@contextmanager
def _e214_engine():
    """Install E220 identities around the immutable E214 training engine."""
    with (
        patch.object(e214, "p", p),
        patch.object(e214, "authenticate", authenticate),
        patch.object(e204, "load_base", load_base),
    ):
        yield


def self_check(output: Path, device_name: str = "cpu") -> dict:
    contract = authenticate(output)
    device = resolve_device(device_name)
    base, search, liveness, base_digest, frozen = load_base(
        contract, p.MODEL_SEED, device
    )
    with _e214_engine():
        stage_b = e214.new_agent(base)
    require(stage_b.config.lambda_center == 10.0, "E220 lambda center differs")
    require(stage_b.config.lambda_max == 40.0, "E220 lambda maximum differs")
    require(len(contract["training_schedule"]) == 36, "E220 schedule size differs")
    return {
        "status": "self_check_passed",
        "base_checkpoint_sha256": frozen["checkpoint_sha256"],
        "base_policy_digest": base_digest,
        "search_max_nodes": search.max_nodes,
        "liveness_max_option_steps": liveness.max_option_steps,
        "lambda_10_exact_stage_a_delegation": True,
        "training_episodes": 36,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def train(output: Path, through_round: int, device_name: str) -> dict:
    authenticate(output)
    with _e214_engine():
        result = e214.train(output.resolve(), int(through_round), device_name)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "source_stage_a_protocol": e215.PROTOCOL,
        "source_stage_a_episode": 300,
        "stage_b_procedure": p.STAGE_B_PROCEDURE["name"],
        "paper_evidence": False,
    }


def _load_endpoint(output: Path, contract: dict, device_name: str):
    with _e214_engine():
        return e214._load_endpoint(output.resolve(), contract, device_name)


def _frozen_instance(manifest: dict, seed: int) -> EpisodeInstance:
    matches = [row for row in manifest["records"] if int(row["seed"]) == int(seed)]
    require(len(matches) == 1, "E220 instance is not uniquely frozen")
    row = matches[0]
    instance = EpisodeInstance.from_json(row["instance_json"])
    require(
        instance.instance_id == row["instance_id"]
        and instance.schedule_id == row["schedule_id"],
        "E220 frozen instance identity differs",
    )
    return instance


def _row_path(output: Path, method: str, seed: int, value: float) -> Path:
    return (
        output.resolve() / "evaluation" / method / f"instance-{int(seed)}"
        / f"lambda-{float(value):g}.json"
    )


def _run_one(
    output: Path,
    contract: dict,
    manifest: dict,
    agent,
    search,
    liveness,
    endpoint: Path,
    base_digest: str,
    *,
    method: str,
    seed: int,
    value: float,
) -> tuple[dict, bool]:
    path = _row_path(output, method, seed, value)
    instance = _frozen_instance(manifest, seed)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "endpoint_sha256": p.sha256(endpoint),
        "model_seed": p.MODEL_SEED,
        "instance_seed": int(seed),
        "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "workload": p.workload_for_seed(seed),
        "method": method,
        "lambda": float(value),
    }
    if path.is_file():
        row = _read(path)
        require(all(row.get(key) == item for key, item in identity.items()),
                f"E220 saved row differs: {path}")
        return row, False
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    agent.base_agent.bind_context_environment(
        env,
        required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    head_before = (
        e204.state_digest(agent.handling_network.state_dict())
        if method == "learned_trace" else None
    )
    label = f"{method} rho={100*p.workload_for_seed(seed):.0f}% lambda={value:g}"
    print(f"E220 eval start {label}", flush=True)
    started = perf_counter()
    with e205_run.heartbeat(f"E220 {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent,
            env,
            instance_seed=int(seed),
            preference_lambda=float(value),
            preference_schedule_record={
                "behavior_alpha": float(value),
                "method": method,
                "endpoint_round": p.ROUNDS,
                "fixed_for_complete_episode": True,
            },
            max_steps=e191.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
        )
    samples = tuple(agent.pop_completed_samples())
    agent.base_agent.decision_log.clear()
    require(bool(result["strict_method_success"]), f"E220 episode failed: {result}")
    require(e204.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
            "E220 evaluation mutated frozen Stage A")
    if head_before is not None:
        require(e204.state_digest(agent.handling_network.state_dict()) == head_before,
                "E220 evaluation mutated the learned head")
    metrics = env.e187_metrics()
    components = metrics["components"]
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    calibration = None
    if method == "learned_trace":
        with _e214_engine():
            target = e214._snapshot_network(agent)
        examples = examples_from_episode(
            samples, target, trace_eta=p.TRACE_ETA, device=agent.device
        )
        calibration = diagnostics(
            agent.handling_network, examples, device=agent.device
        )
    row = {
        **identity,
        "strict_complete": True,
        "J10": j10,
        "J_lambda": j10 + (float(value) - 10.0) * rehandles,
        "physical_rehandles": rehandles,
        "timing_cost_2T_plus_E": (
            2.0 * float(components["storage_tardiness_accrual"])
            + float(components["storage_earliness_at_delivery"])
        ),
        "admission_wait_cost_0p5W": 0.5 * float(components["admission_wait_accrual"]),
        "equipment_cost_0p1L": 0.1 * float(components["equipment_work"]),
        "steps": int(result["steps"]),
        "peak_stored": int(metrics["peak_stored"]),
        "mean_stored": float(metrics["mean_stored"]),
        "calibration": calibration,
        "wall_seconds": perf_counter() - started,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    require(
        env.current_episode_instance.instance_id == instance.instance_id,
        "E220 executed another EpisodeInstance",
    )
    _write(path, row)
    print(
        f"E220 eval done {label} Jalpha={row['J_lambda']:.1f} "
        f"R={rehandles} timing={row['timing_cost_2T_plus_E']:.1f}",
        flush=True,
    )
    return row, True


def evaluate(output: Path, device_name: str) -> dict:
    contract = authenticate(output)
    manifest = _read(output.resolve() / "instances.json")
    learned, search, liveness, base_digest, endpoint = _load_endpoint(
        output, contract, device_name
    )
    immediate = from_stage_b_agent(learned, seed=p.HEAD_SEED + 100)
    with torch.no_grad():
        probe = torch.zeros((3, immediate.config.feature_dim), device=learned.device)
        values = torch.tensor([10.0, 20.0, 40.0], device=learned.device)
        require(bool((immediate.handling_network(probe, values) == 0).all()),
                "E220 immediate-only continuation is not exactly zero")
    rows = []
    new_runs = 0
    for seed in p.EVAL_INSTANCE_SEEDS:
        for value in p.EVAL_LAMBDAS:
            row, executed = _run_one(
                output, contract, manifest, learned, search, liveness, endpoint,
                base_digest, method="learned_trace", seed=seed, value=value,
            )
            rows.append(row)
            new_runs += int(executed)
        for value in p.IMMEDIATE_ONLY_LAMBDAS:
            row, executed = _run_one(
                output, contract, manifest, immediate, search, liveness, endpoint,
                base_digest, method="immediate_only", seed=seed, value=value,
            )
            rows.append(row)
            new_runs += int(executed)
    analysis = analyze(output)
    return {
        "status": "complete",
        "rows": len(rows),
        "new_episode_runs": new_runs,
        "reused_rows": len(rows) - new_runs,
        "all_gates_passed": analysis["all_gates_passed"],
        "report": analysis["report"],
        "test_panel_opened": False,
    }


def _weighted_calibration(rows: list[dict], key: str) -> float:
    numerator = sum(
        float(row["calibration"][key]) * int(row["calibration"]["sample_count"])
        for row in rows
    )
    denominator = sum(int(row["calibration"]["sample_count"]) for row in rows)
    require(denominator > 0, "E220 calibration has no samples")
    return numerator / denominator


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    learned = [
        _read(_row_path(output, "learned_trace", seed, value))
        for seed in p.EVAL_INSTANCE_SEEDS for value in p.EVAL_LAMBDAS
    ]
    immediate = [
        _read(_row_path(output, "immediate_only", seed, value))
        for seed in p.EVAL_INSTANCE_SEEDS for value in p.IMMEDIATE_ONLY_LAMBDAS
    ]
    comparisons = []
    for seed in p.EVAL_INSTANCE_SEEDS:
        for value in p.IMMEDIATE_ONLY_LAMBDAS:
            left = next(row for row in learned
                        if row["instance_seed"] == seed and row["lambda"] == value)
            right = next(row for row in immediate
                         if row["instance_seed"] == seed and row["lambda"] == value)
            require(left["instance_id"] == right["instance_id"],
                    "E220 paired instance differs")
            relative = (
                (float(left["J_lambda"]) - float(right["J_lambda"]))
                / max(abs(float(right["J_lambda"])), 1.0)
            )
            comparisons.append({
                "instance_seed": seed,
                "workload": left["workload"],
                "lambda": value,
                "learned_rehandles": left["physical_rehandles"],
                "immediate_rehandles": right["physical_rehandles"],
                "learned_timing": left["timing_cost_2T_plus_E"],
                "immediate_timing": right["timing_cost_2T_plus_E"],
                "learned_J_lambda": left["J_lambda"],
                "immediate_J_lambda": right["J_lambda"],
                "delta_learned_minus_immediate_J_lambda": (
                    left["J_lambda"] - right["J_lambda"]
                ),
                "relative_learned_minus_immediate_J_lambda": relative,
            })
    mc = _weighted_calibration(learned, "mc_mae")
    zero = _weighted_calibration(learned, "zero_mc_mae")
    gates = {
        "calibration_beats_zero": mc < zero,
        "frontier_adds_value_somewhere": any(
            row["delta_learned_minus_immediate_J_lambda"] < 0
            for row in comparisons
        ),
        "no_catastrophic_regression": all(
            row["relative_learned_minus_immediate_J_lambda"]
            <= p.MAX_RELATIVE_J_REGRESSION
            for row in comparisons
        ),
        "strict_completion": all(
            bool(row["strict_complete"]) for row in learned + immediate
        ),
    }
    report = {
        "status": "complete",
        "verdict": (
            "seed0_stage_b_candidate" if all(gates.values())
            else "stopped_at_failed_development_gate"
        ),
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "source_stage_a_episode": 300,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "sample_weighted_heldout_mc_mae": mc,
        "sample_weighted_zero_head_mc_mae": zero,
        "lower_J_than_immediate_rate": fmean(
            float(row["delta_learned_minus_immediate_J_lambda"] < 0)
            for row in comparisons
        ),
        "maximum_relative_J_regression": max(
            row["relative_learned_minus_immediate_J_lambda"]
            for row in comparisons
        ),
        "learned_rows": learned,
        "immediate_only_rows": immediate,
        "paired_comparisons": comparisons,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    report_path = output.resolve() / "e220-report.json"
    _write(report_path, report)
    lines = [
        render_validated_conformance_audit(_declaration(contract)),
        "RESULTS\n" + "=" * 72,
        "",
        f"Verdict: {report['verdict']}",
        f"Held-out MC MAE: {mc:.3f} (zero head {zero:.3f})",
        f"Lower J_alpha than immediate-only: "
        f"{100*report['lower_J_than_immediate_rate']:.1f}%",
        f"Maximum relative J_alpha regression: "
        f"{100*report['maximum_relative_J_regression']:.1f}%",
        "",
        "Predeclared gates:",
    ]
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}"
                 for name, value in gates.items())
    lines += [
        "",
        "Development-only seed-0 expansion; no test panel was opened.",
        "Alpha remains an operating-point coordinate, not an exact price.",
        "",
    ]
    report_text = output.resolve() / "REPORT.txt"
    report_text.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "complete",
        "verdict": report["verdict"],
        "all_gates_passed": report["all_gates_passed"],
        "report": str(report_path),
        "report_text": str(report_text),
        "test_panel_opened": False,
    }


def status(output: Path) -> dict:
    contract = authenticate(output)
    summary_path = (
        output.resolve() / "training" / f"seed-{p.MODEL_SEED}"
        / "training-summary.json"
    )
    summary = _read(summary_path) if summary_path.is_file() else {}
    endpoint = (
        output.resolve() / "training" / f"seed-{p.MODEL_SEED}"
        / "checkpoints" / f"round-{p.ROUNDS}.pth"
    )
    return {
        "status": summary.get("status", "not_started"),
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "completed_rounds": int(summary.get("completed_rounds", 0)),
        "completed_training_episodes": int(summary.get("completed_training_episodes", 0)),
        "endpoint_ready": endpoint.is_file(),
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "self-check", "train", "evaluate", "analyze", "status"),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--through-round", type=int, default=p.ROUNDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output, args.device)
    elif args.command == "train":
        result = train(args.output, args.through_round, args.device)
    elif args.command == "evaluate":
        result = evaluate(args.output, args.device)
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        result = status(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
