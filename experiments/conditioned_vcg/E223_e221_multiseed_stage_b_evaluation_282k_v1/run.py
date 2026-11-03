#!/usr/bin/env python3
"""Evaluate finalized E221 Stage-B heads on the paired E220 panel."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

import train_vcg_preference_conditioned as common
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    run as e204,
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
from experiments.conditioned_vcg.E221_e218_multiseed_stage_b_280k_v1 import (
    run as e221_run,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p


class E223Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E223Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E223 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E223 JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
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
        payload = {
            key: value for key, value in contract.items()
            if key != "contract_sha256"
        }
        require(contract.get("protocol") == p.PROTOCOL,
                "E223 protocol identity differs")
        require(p.digest(payload) == contract.get("contract_sha256"),
                "E223 contract digest differs")
        render_validated_conformance_audit(_declaration(contract))
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E223 refuses a nonempty uncontracted output")
        contract = p.contract()
        validate_protocol(contract, source=p.PROTOCOL)
        _write(contract_path, contract)
    manifest = _instance_manifest(contract)
    if manifest_path.is_file():
        require(_read(manifest_path) == manifest,
                "E223 instance manifest changed")
    else:
        _write(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "model_seeds": list(p.MODEL_SEEDS),
        "episode_runs_per_seed": 15,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    prepare(output)
    contract = _read(output.resolve() / "contract.json")
    for relative, expected in contract["source_sha256"].items():
        path = p.ROOT / relative
        require(path.is_file() and p.sha256(path) == expected,
                f"E223 behavioral source changed: {relative}")
    source_contract = _read(p.SOURCE_CONTRACT)
    require(
        source_contract.get("contract_sha256")
        == contract["frozen_stage_b"][0]["source_contract_sha256"],
        "E223 E221 source contract changed",
    )
    for frozen in contract["frozen_stage_b"]:
        for key, hash_key in (
            ("endpoint", "endpoint_sha256"),
            ("summary", "summary_sha256"),
        ):
            path = Path(frozen[key])
            require(path.is_file() and p.sha256(path) == frozen[hash_key],
                    f"E223 seed {frozen['model_seed']} {key} changed")
    return contract


def _frozen(contract: dict, model_seed: int) -> dict:
    rows = [
        row for row in contract["frozen_stage_b"]
        if int(row["model_seed"]) == int(model_seed)
    ]
    require(len(rows) == 1, "E223 frozen Stage B is not unique")
    return rows[0]


def _frozen_instance(manifest: dict, seed: int) -> EpisodeInstance:
    rows = [row for row in manifest["records"] if int(row["seed"]) == int(seed)]
    require(len(rows) == 1, "E223 frozen instance is not unique")
    row = rows[0]
    instance = EpisodeInstance.from_json(row["instance_json"])
    require(
        instance.instance_id == row["instance_id"]
        and instance.schedule_id == row["schedule_id"],
        "E223 frozen instance identity differs",
    )
    return instance


def _load_endpoint(contract: dict, model_seed: int, device_name: str):
    source_contract = _read(p.SOURCE_CONTRACT)
    seed_contract = e221_run._seed_contract(source_contract, model_seed)
    with e221_run._engine(source_contract, model_seed):
        loaded = e214._load_endpoint(
            p.SOURCE_OUTPUT, seed_contract, device_name
        )
    agent, search, liveness, base_digest, endpoint = loaded
    frozen = _frozen(contract, model_seed)
    require(p.sha256(endpoint) == frozen["endpoint_sha256"],
            "E223 loaded another Stage-B endpoint")
    return agent, search, liveness, base_digest, endpoint


def _row_path(
    output: Path, model_seed: int, method: str, instance_seed: int, value: float,
) -> Path:
    return (
        output.resolve() / "evaluation" / f"model-seed-{int(model_seed)}"
        / method / f"instance-{int(instance_seed)}"
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
    model_seed: int,
    method: str,
    instance_seed: int,
    value: float,
) -> tuple[dict, bool]:
    path = _row_path(output, model_seed, method, instance_seed, value)
    instance = _frozen_instance(manifest, instance_seed)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "endpoint_sha256": p.sha256(endpoint),
        "model_seed": int(model_seed),
        "instance_seed": int(instance_seed),
        "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "workload": p.workload_for_seed(instance_seed),
        "method": method,
        "lambda": float(value),
    }
    if path.is_file():
        row = _read(path)
        require(all(row.get(key) == item for key, item in identity.items()),
                f"E223 saved row differs: {path}")
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
    label = (
        f"seed={model_seed} {method} "
        f"rho={100*p.workload_for_seed(instance_seed):.0f}% lambda={value:g}"
    )
    print(f"E223 eval start {label}", flush=True)
    started = perf_counter()
    with e205_run.heartbeat(f"E223 {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent,
            env,
            instance_seed=int(instance_seed),
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
    require(bool(result["strict_method_success"]),
            f"E223 episode failed: {result}")
    require(e204.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
            "E223 evaluation mutated frozen Stage A")
    if head_before is not None:
        require(e204.state_digest(agent.handling_network.state_dict()) == head_before,
                "E223 evaluation mutated the learned head")

    metrics = env.e187_metrics()
    components = metrics["components"]
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    calibration = None
    if method == "learned_trace":
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
        "admission_wait_cost_0p5W": (
            0.5 * float(components["admission_wait_accrual"])
        ),
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
        "E223 executed another EpisodeInstance",
    )
    _write(path, row)
    print(
        f"E223 eval done {label} Jalpha={row['J_lambda']:.1f} "
        f"R={rehandles} timing={row['timing_cost_2T_plus_E']:.1f}",
        flush=True,
    )
    return row, True


def evaluate_seed(
    output: Path, model_seed: int, device_name: str,
) -> dict:
    contract = authenticate(output)
    seed = int(model_seed)
    require(seed in p.MODEL_SEEDS, "E223 model seed must be 1 or 2")
    manifest = _read(output.resolve() / "instances.json")
    source_contract = _read(p.SOURCE_CONTRACT)
    with e221_run._engine(source_contract, seed):
        learned, search, liveness, base_digest, endpoint = _load_endpoint(
            contract, seed, device_name
        )
        immediate = from_stage_b_agent(
            learned, seed=p.head_seed(seed) + 100
        )
        with torch.no_grad():
            probe = torch.zeros((3, immediate.config.feature_dim), device=learned.device)
            values = torch.tensor([10.0, 20.0, 40.0], device=learned.device)
            require(bool((immediate.handling_network(probe, values) == 0).all()),
                    "E223 immediate-only continuation is not exactly zero")
        rows = []
        new_runs = 0
        for instance_seed in p.EVAL_INSTANCE_SEEDS:
            for value in p.EVAL_LAMBDAS:
                row, executed = _run_one(
                    output, contract, manifest, learned, search, liveness,
                    endpoint, base_digest, model_seed=seed,
                    method="learned_trace", instance_seed=instance_seed,
                    value=value,
                )
                rows.append(row)
                new_runs += int(executed)
            for value in p.IMMEDIATE_ONLY_LAMBDAS:
                row, executed = _run_one(
                    output, contract, manifest, immediate, search, liveness,
                    endpoint, base_digest, model_seed=seed,
                    method="immediate_only", instance_seed=instance_seed,
                    value=value,
                )
                rows.append(row)
                new_runs += int(executed)
    return {
        "status": "complete",
        "model_seed": seed,
        "rows": len(rows),
        "new_episode_runs": new_runs,
        "reused_rows": len(rows) - new_runs,
        "test_panel_opened": False,
    }


def _weighted_calibration(rows: list[dict], key: str) -> float:
    denominator = sum(int(row["calibration"]["sample_count"]) for row in rows)
    require(denominator > 0, "E223 calibration has no samples")
    return sum(
        float(row["calibration"][key])
        * int(row["calibration"]["sample_count"])
        for row in rows
    ) / denominator


def _seed_analysis(output: Path, model_seed: int) -> dict:
    learned = [
        _read(_row_path(
            output, model_seed, "learned_trace", instance_seed, value
        ))
        for instance_seed in p.EVAL_INSTANCE_SEEDS
        for value in p.EVAL_LAMBDAS
    ]
    immediate = [
        _read(_row_path(
            output, model_seed, "immediate_only", instance_seed, value
        ))
        for instance_seed in p.EVAL_INSTANCE_SEEDS
        for value in p.IMMEDIATE_ONLY_LAMBDAS
    ]
    comparisons = []
    for instance_seed in p.EVAL_INSTANCE_SEEDS:
        for value in p.IMMEDIATE_ONLY_LAMBDAS:
            left = next(
                row for row in learned
                if row["instance_seed"] == instance_seed
                and row["lambda"] == value
            )
            right = next(
                row for row in immediate
                if row["instance_seed"] == instance_seed
                and row["lambda"] == value
            )
            require(left["instance_id"] == right["instance_id"],
                    "E223 paired instance differs")
            relative = (
                (float(left["J_lambda"]) - float(right["J_lambda"]))
                / max(abs(float(right["J_lambda"])), 1.0)
            )
            comparisons.append({
                "model_seed": int(model_seed),
                "instance_seed": int(instance_seed),
                "workload": left["workload"],
                "lambda": float(value),
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
    return {
        "model_seed": int(model_seed),
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
    }


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    seeds = {
        str(seed): _seed_analysis(output, seed) for seed in p.MODEL_SEEDS
    }
    all_passed = all(row["all_gates_passed"] for row in seeds.values())
    report = {
        "status": "complete",
        "verdict": (
            "multiseed_stage_b_candidate" if all_passed
            else "stopped_at_failed_multiseed_development_gate"
        ),
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "all_gates_passed": all_passed,
        "seeds": seeds,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    report_json = output.resolve() / "e223-report.json"
    _write(report_json, report)
    lines = [
        render_validated_conformance_audit(_declaration(contract)),
        "RESULTS\n" + "=" * 72,
        "",
        f"Verdict: {report['verdict']}",
    ]
    for seed in p.MODEL_SEEDS:
        row = seeds[str(seed)]
        lines += [
            "",
            f"Seed {seed}",
            f"Held-out MC MAE: {row['sample_weighted_heldout_mc_mae']:.3f} "
            f"(zero head {row['sample_weighted_zero_head_mc_mae']:.3f})",
            f"Lower J_alpha than immediate-only: "
            f"{100*row['lower_J_than_immediate_rate']:.1f}%",
            f"Maximum relative J_alpha regression: "
            f"{100*row['maximum_relative_J_regression']:.1f}%",
        ]
        lines.extend(
            f"- {name}: {'PASS' if value else 'FAIL'}"
            for name, value in row["gates"].items()
        )
    lines += [
        "",
        "Development-only seeds 1 and 2; no test panel was opened.",
        "Alpha is an operating-point coordinate, not an exact price.",
        "",
    ]
    report_text = output.resolve() / "REPORT.txt"
    report_text.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "complete",
        "verdict": report["verdict"],
        "all_gates_passed": all_passed,
        "report": str(report_json),
        "report_text": str(report_text),
        "test_panel_opened": False,
    }


def status(output: Path) -> dict:
    authenticate(output)
    rows = {}
    for seed in p.MODEL_SEEDS:
        expected = (
            len(p.EVAL_INSTANCE_SEEDS)
            * (len(p.EVAL_LAMBDAS) + len(p.IMMEDIATE_ONLY_LAMBDAS))
        )
        observed = sum(
            _row_path(output, seed, method, instance_seed, value).is_file()
            for method, values in (
                ("learned_trace", p.EVAL_LAMBDAS),
                ("immediate_only", p.IMMEDIATE_ONLY_LAMBDAS),
            )
            for instance_seed in p.EVAL_INSTANCE_SEEDS
            for value in values
        )
        rows[str(seed)] = {"observed_rows": observed, "expected_rows": expected}
    return {
        "status": "available",
        "seeds": rows,
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "evaluate-seed", "analyze", "status"),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "evaluate-seed":
        require(args.model_seed is not None,
                "evaluate-seed requires --model-seed 1 or 2")
        result = evaluate_seed(
            args.output, args.model_seed, args.device
        )
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        result = status(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
