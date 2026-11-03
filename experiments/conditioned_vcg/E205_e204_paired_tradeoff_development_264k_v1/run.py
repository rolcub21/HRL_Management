#!/usr/bin/env python3
"""Run E204 seed 0 on a small exactly paired development panel."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean
from threading import Event, Thread
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
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.centered import (
    CenteredHandlingAgent,
)
from train_viability_graph_smdp import resolve_device

from . import protocol as p


class E205Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E205Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E205 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E205 JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _level_label(value: float) -> str:
    return f"{100 * float(value):.0f}%"


def _lambda_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def make_instance(seed: int) -> EpisodeInstance:
    p.workload_for_seed(seed)
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    instance = env.sample_episode_instance(seed=int(seed))
    instance.validate_for(env)
    return instance


def instance_manifest(contract: dict) -> dict:
    records = []
    for seed in p.INSTANCE_SEEDS:
        instance = make_instance(seed)
        records.append({
            "seed": seed,
            "workload": p.workload_for_seed(seed),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "instance_json": instance.to_json(),
        })
    core = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "records": records,
    }
    return {**core, "manifest_sha256": p.digest(core)}


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    manifest = instance_manifest(contract)
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    if contract_path.is_file():
        require(_read(contract_path) == contract, "E205 contract changed")
        require(_read(manifest_path) == manifest, "E205 instance manifest changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E205 refuses a nonempty uncontracted output",
        )
        _write(contract_path, contract)
        _write(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "instances": len(p.INSTANCE_SEEDS),
        "lambda_values": list(p.LAMBDAS),
        "expected_rows": len(p.INSTANCE_SEEDS) * len(p.LAMBDAS),
        "episode_runs": 0,
        "training_runs": 0,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    prepared = prepare(output)
    contract = _read(output.resolve() / "contract.json")
    manifest = _read(output.resolve() / "instances.json")
    require(
        prepared["contract_sha256"] == contract["contract_sha256"],
        "E205 authentication failed",
    )
    return contract, manifest


def frozen_instance(manifest: dict, seed: int) -> EpisodeInstance:
    rows = [row for row in manifest["records"] if int(row["seed"]) == int(seed)]
    require(len(rows) == 1, "E205 instance is not uniquely frozen")
    row = rows[0]
    instance = EpisodeInstance.from_json(row["instance_json"])
    require(
        instance.instance_id == row["instance_id"]
        and instance.schedule_id == row["schedule_id"],
        "E205 frozen instance identity differs",
    )
    return instance


def load_agent(device_name: str):
    device = resolve_device(device_name)
    stage_b = p.stage_b_identity()
    e204_contract = e204.authenticate(p.STAGE_B_OUTPUT)
    base, search, liveness, base_digest, source = e204.load_base(
        e204_contract, p.MODEL_SEED, device
    )
    require(
        base_digest == stage_b["base_policy_digest"]
        and source["checkpoint_sha256"] == stage_b["base_checkpoint_sha256"],
        "E205 frozen E191 identity differs from E204",
    )
    terminal = torch.load(
        p.STAGE_B_TERMINAL, map_location="cpu", weights_only=False
    )
    expected = {
        "training_protocol": e204.p.PROTOCOL,
        "role": "terminal",
        "scenario": p.SCENARIO,
        "model_seed": p.MODEL_SEED,
        "completed_rounds": e204.p.ROUNDS,
        "base_checkpoint_sha256": stage_b["base_checkpoint_sha256"],
        "base_policy_digest": stage_b["base_policy_digest"],
        "development_only": True,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    mismatch = {
        key: (terminal.get(key), value)
        for key, value in expected.items()
        if terminal.get(key) != value
    }
    require(not mismatch, f"E205 E204 terminal identity differs: {mismatch}")
    agent = CenteredHandlingAgent.from_checkpoint(
        terminal["agent_checkpoint"],
        base_agent=base,
        expected_base_checkpoint_sha256=stage_b["base_checkpoint_sha256"],
        expected_base_policy_digest=stage_b["base_policy_digest"],
        seed=264_205_700,
    )
    require(
        agent.config.lambda_center == p.LAMBDA_CENTER,
        "E205 loaded another lambda center",
    )
    return agent, search, liveness, base_digest, stage_b, device


@contextmanager
def heartbeat(label: str, seconds: float = 120.0):
    stop = Event()
    started = perf_counter()

    def report() -> None:
        while not stop.wait(seconds):
            print(
                f"E205 running {label} elapsed={(perf_counter() - started) / 60:.1f}m",
                flush=True,
            )

    thread = Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def row_path(output: Path, seed: int, value: float) -> Path:
    workload = p.workload_for_seed(seed)
    return (
        output.resolve() / "rows" / f"rho-{_level_label(workload).rstrip('%')}"
        / f"lambda-{_lambda_label(value)}.json"
    )


def _run_loaded(
    output: Path,
    contract: dict,
    manifest: dict,
    agent,
    search,
    liveness,
    base_digest: str,
    stage_b: dict,
    *,
    seed: int,
    value: float,
) -> dict:
    value = float(value)
    require(value in p.LAMBDAS, "lambda is outside the E205 panel")
    expected_instance = frozen_instance(manifest, seed)
    workload = p.workload_for_seed(seed)
    path = row_path(output, seed, value)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "instance_seed": int(seed),
        "instance_id": expected_instance.instance_id,
        "schedule_id": expected_instance.schedule_id,
        "workload": workload,
        "lambda": value,
        "stage_b_terminal_sha256": stage_b["terminal_checkpoint_sha256"],
    }
    if path.is_file():
        row = _read(path)
        require(
            all(row.get(key) == expected for key, expected in identity.items()),
            f"saved E205 row differs: {path}",
        )
        return {**row, "reused": True, "episode_runs": 0}

    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    agent.base_agent.bind_context_environment(
        env, required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    label = f"rho={_level_label(workload)} lambda={value:g}"
    print(f"E205 start {label}", flush=True)
    network_before = e204.state_digest(agent.handling_network.state_dict())
    started = perf_counter()
    with heartbeat(label), ExitStack() as stack:
        e204._runtime_stack(stack)
        run = common.run_preference_episode(
            agent,
            env,
            instance_seed=int(seed),
            preference_lambda=value,
            preference_schedule_record={
                "behavior_lambda": value,
                "lambda_center": p.LAMBDA_CENTER,
                "paired_panel": p.PROTOCOL,
                "workload": workload,
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
    samples = agent.pop_completed_samples()
    require(
        len(samples) == int(run["macro_decisions"]),
        "E205 complete episode did not yield one sample per decision",
    )
    require(
        e204.state_digest(agent.handling_network.state_dict()) == network_before,
        "E205 evaluation mutated the handling head",
    )
    require(
        e204.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
        "E205 evaluation mutated E191",
    )
    metrics = env.e187_metrics()
    components = {key: float(item) for key, item in metrics["components"].items()}
    rehandles = int(run["physical_rehandles"])
    require(
        math.isclose(components["physical_rehandles"], rehandles, abs_tol=0.0),
        "E205 component and physical rehandle counts differ",
    )
    j10 = float(metrics["J"])
    j_lambda = j10 + (value - p.LAMBDA_CENTER) * rehandles
    row = {
        **identity,
        "status": "complete" if run["strict_method_success"] else "failed",
        "strict_complete": bool(run["strict_method_success"]),
        "method_failure_reason": run["method_failure_reason"],
        "J10": j10,
        "J_lambda": float(j_lambda),
        "components": components,
        "timing_cost_2T_plus_E": (
            2.0 * components["storage_tardiness_accrual"]
            + components["storage_earliness_at_delivery"]
        ),
        "admission_wait_cost_0p5W": 0.5 * components["admission_wait_accrual"],
        "equipment_cost_0p1L": 0.1 * components["equipment_work"],
        "physical_rehandles": rehandles,
        "steps": int(run["steps"]),
        "macro_decisions": int(run["macro_decisions"]),
        "storage_clock_mae": metrics["storage_clock_mae"],
        "storage_clock_signed_error": metrics["storage_clock_signed_error"],
        "storage_within_20_rate": metrics["storage_within_20_rate"],
        "release_clock_mae": metrics["release_clock_mae"],
        "release_clock_signed_error": metrics["release_clock_signed_error"],
        "release_within_20_rate": metrics["release_within_20_rate"],
        "mean_admission_wait": metrics["mean_admission_wait"],
        "total_admission_wait": metrics["total_admission_wait"],
        "mean_stored": float(metrics["mean_stored"]),
        "mean_stored_fraction": float(metrics["mean_stored_fraction"]),
        "peak_stored": int(metrics["peak_stored"]),
        "peak_stored_fraction": float(metrics["peak_stored_fraction"]),
        "mean_external_queue": float(metrics["mean_external_queue"]),
        "peak_external_queue": int(metrics["peak_external_queue"]),
        "unknown_candidates_rejected": int(
            run["audit"].get("unknown_candidates_rejected", 0)
        ),
        "wall_seconds": perf_counter() - started,
        "training_runs": 0,
        "episode_runs": 1,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    require(row["strict_complete"], f"E205 episode failed: {row}")
    require(
        env.current_episode_instance.instance_id == expected_instance.instance_id,
        "E205 executed another EpisodeInstance",
    )
    _write(path, row)
    print(
        f"E205 done {label} J10={j10:.1f} Jlambda={j_lambda:.1f} "
        f"R={rehandles} timing={row['timing_cost_2T_plus_E']:.1f} "
        f"wait={row['admission_wait_cost_0p5W']:.1f} "
        f"steps={row['steps']} peak={row['peak_stored']}/35 "
        f"wall={row['wall_seconds']:.1f}s",
        flush=True,
    )
    return row


def run_one(
    output: Path, seed: int, value: float, device_name: str
) -> dict:
    contract, manifest = authenticate(output)
    agent, search, liveness, base_digest, stage_b, _device = load_agent(device_name)
    return _run_loaded(
        output, contract, manifest, agent, search, liveness, base_digest, stage_b,
        seed=seed, value=value,
    )


def run_all(output: Path, device_name: str) -> dict:
    contract, manifest = authenticate(output)
    agent, search, liveness, base_digest, stage_b, _device = load_agent(device_name)
    ordered_lambdas = (p.LAMBDA_CENTER,) + tuple(
        value for value in p.LAMBDAS if value != p.LAMBDA_CENTER
    )
    rows = []
    episode_runs = 0
    for seed in p.INSTANCE_SEEDS:
        for value in ordered_lambdas:
            row = _run_loaded(
                output, contract, manifest, agent, search, liveness,
                base_digest, stage_b, seed=seed, value=value,
            )
            rows.append(row)
            episode_runs += int(row["episode_runs"])
    return {
        "status": "complete",
        "rows": len(rows),
        "new_episode_runs": episode_runs,
        "reused_rows": len(rows) - episode_runs,
        "test_panel_opened": False,
        "paper_evidence": False,
    }


OUTCOMES = (
    "physical_rehandles", "J10", "J_lambda", "timing_cost_2T_plus_E",
    "admission_wait_cost_0p5W", "equipment_cost_0p1L", "steps",
    "storage_clock_mae", "storage_within_20_rate", "release_clock_mae",
    "mean_stored", "peak_stored", "mean_external_queue",
)


def _mean(rows: list[dict], key: str) -> float:
    return float(fmean(float(row[key]) for row in rows))


def analyze(output: Path) -> dict:
    contract, manifest = authenticate(output)
    rows = []
    for seed in p.INSTANCE_SEEDS:
        for value in p.LAMBDAS:
            path = row_path(output, seed, value)
            require(path.is_file(), f"E205 row is missing: {path}")
            row = _read(path)
            rows.append(row)
    require(len(rows) == 15 and all(row["strict_complete"] for row in rows),
            "E205 panel is incomplete")
    for seed in p.INSTANCE_SEEDS:
        paired = [row for row in rows if int(row["instance_seed"]) == seed]
        require(
            len({row["instance_id"] for row in paired}) == 1
            and len({row["schedule_id"] for row in paired}) == 1,
            "E205 lambda runs are not instance paired",
        )

    center_by_seed = {
        int(row["instance_seed"]): row
        for row in rows if float(row["lambda"]) == p.LAMBDA_CENTER
    }
    paired_rows = []
    for row in sorted(rows, key=lambda item: (item["workload"], item["lambda"])):
        center = center_by_seed[int(row["instance_seed"])]
        deltas = {
            key: float(row[key]) - float(center[key]) for key in OUTCOMES
        }
        paired_rows.append({**row, "delta_from_lambda10": deltas})

    by_lambda = []
    for value in p.LAMBDAS:
        selected = [row for row in rows if float(row["lambda"]) == value]
        by_lambda.append({
            "lambda": value,
            "cases": len(selected),
            **{f"mean_{key}": _mean(selected, key) for key in OUTCOMES},
            **{
                f"mean_delta_{key}_from_lambda10": float(fmean(
                    float(row[key])
                    - float(center_by_seed[int(row["instance_seed"])][key])
                    for row in selected
                ))
                for key in OUTCOMES
            },
        })

    monotone_by_workload = {}
    for seed in p.INSTANCE_SEEDS:
        selected = sorted(
            (row for row in rows if int(row["instance_seed"]) == seed),
            key=lambda row: float(row["lambda"]),
        )
        counts = [int(row["physical_rehandles"]) for row in selected]
        monotone_by_workload[_level_label(p.workload_for_seed(seed))] = {
            "counts": counts,
            "nonincreasing": all(
                left >= right for left, right in zip(counts, counts[1:])
            ),
        }
    non_rehandle_changes = {
        key: any(
            not math.isclose(
                float(row[key]),
                float(center_by_seed[int(row["instance_seed"])][key]),
                abs_tol=1.0e-12,
            )
            for row in rows
            if float(row["lambda"]) != p.LAMBDA_CENTER
        )
        for key in OUTCOMES
        if key != "physical_rehandles"
    }
    result = {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "stage_b_terminal_sha256": p.stage_b_identity()["terminal_checkpoint_sha256"],
        "pairing_verified": True,
        "strict_completion_rate": 1.0,
        "lambda_center": p.LAMBDA_CENTER,
        "by_lambda": by_lambda,
        "paired_rows": paired_rows,
        "rehandles_nonincreasing_by_workload": monotone_by_workload,
        "all_workloads_rehandles_nonincreasing": all(
            value["nonincreasing"] for value in monotone_by_workload.values()
        ),
        "non_rehandle_outcomes_changed_from_lambda10": non_rehandle_changes,
        "scope": "small paired development mechanism panel",
        "final_inference_authorized": False,
        "fresh_larger_panel_required_for_paper": True,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(output.resolve() / p.REPORT_NAME, result)
    _write_table(output.resolve() / p.TABLE_NAME, result)
    _write_csv(output.resolve() / p.CSV_NAME, paired_rows)
    from .plot import create_plot
    create_plot(result, output.resolve() / p.PLOT_NAME)
    return {
        "status": "complete",
        "report": str(output.resolve() / p.REPORT_NAME),
        "table": str(output.resolve() / p.TABLE_NAME),
        "csv": str(output.resolve() / p.CSV_NAME),
        "plot": str(output.resolve() / p.PLOT_NAME),
        "all_workloads_rehandles_nonincreasing": result[
            "all_workloads_rehandles_nonincreasing"
        ],
        "test_panel_opened": False,
        "paper_evidence": False,
    }


def _write_table(path: Path, result: dict) -> None:
    lines = [
        "# E205 paired development trade-off",
        "",
        "Values are means over the same three workload instances. Deltas are paired to lambda=10.",
        "",
        "| lambda | rehandles | delta R | J10 | delta J10 | timing | wait cost | equipment cost | steps | peak |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_lambda"]:
        lines.append(
            f"| {row['lambda']:g} | {row['mean_physical_rehandles']:.2f} | "
            f"{row['mean_delta_physical_rehandles_from_lambda10']:+.2f} | "
            f"{row['mean_J10']:.1f} | {row['mean_delta_J10_from_lambda10']:+.1f} | "
            f"{row['mean_timing_cost_2T_plus_E']:.1f} | "
            f"{row['mean_admission_wait_cost_0p5W']:.1f} | "
            f"{row['mean_equipment_cost_0p1L']:.1f} | "
            f"{row['mean_steps']:.1f} | {row['mean_peak_stored']:.1f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "instance_seed", "workload", "lambda", "physical_rehandles", "J10",
        "J_lambda", "timing_cost_2T_plus_E", "admission_wait_cost_0p5W",
        "equipment_cost_0p1L", "steps", "storage_clock_mae",
        "storage_within_20_rate", "release_clock_mae", "mean_stored",
        "peak_stored", "mean_external_queue", "wall_seconds",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run-one", "run-all", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, choices=p.INSTANCE_SEEDS)
    parser.add_argument("--lambda-value", type=float, choices=p.LAMBDAS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-all":
        result = run_all(output, args.device)
    elif args.command == "analyze":
        result = analyze(output)
    else:
        require(args.seed is not None, "run-one requires --seed")
        require(args.lambda_value is not None, "run-one requires --lambda-value")
        result = run_one(output, args.seed, args.lambda_value, args.device)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
