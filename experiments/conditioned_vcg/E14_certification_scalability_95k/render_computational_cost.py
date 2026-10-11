#!/usr/bin/env python3
"""Recover paper-facing computational-cost evidence from existing artifacts.

No policy is trained or evaluated here.  The script audits the timestamps that
were actually persisted, labels unavailable quantities explicitly, and reduces
the authenticated E13 decision records into component and latency tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-block6-scalability-95k"
E13_ROOT = PROJECT_ROOT / "results/vcg-conditioned-e13-scalability-95k-v2"
E1_ROOT = PROJECT_ROOT / "results/vcg-conditioned-final-comparison-90k-cpu-v3"

STAGE_A_SUMMARIES = {
    seed: PROJECT_ROOT / f"results/vcg-dense-v1-1-seed{seed}-500ep/training-summary.json"
    for seed in (0, 1, 2)
}
STAGE_B_SUMMARIES = {
    0: PROJECT_ROOT / "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/damped-convergence-summary.json",
    1: PROJECT_ROOT / "results/vcg-v1-1-conditioned-handling-seed1-convergence-continuation/seed1-continuation-summary.json",
    2: PROJECT_ROOT / "results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development/seed-2/training-summary.json",
}


class ComputationalCostError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComputationalCostError(f"expected JSON object: {path}")
    return value


def _canonical_hash(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit is not None:
        payload.pop(omit, None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hashed(path: Path, field: str) -> dict[str, Any]:
    value = _load(path)
    if value.get(field) != _canonical_hash(value, omit=field):
        raise ComputationalCostError(f"self-hash mismatch: {path}")
    return value


def _dist(values: Iterable[float]) -> dict[str, float | int | None]:
    data = tuple(float(item) for item in values)
    array = np.asarray(data, dtype=float)
    return {
        "n": len(data),
        "mean": fmean(data) if data else None,
        "median": median(data) if data else None,
        "p95": float(np.percentile(array, 95)) if data else None,
        "maximum": max(data) if data else None,
    }


def _timing_named_scalars(value: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_timing_named_scalars(item, (*prefix, str(key))))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_timing_named_scalars(item, (*prefix, str(index))))
    elif isinstance(value, (int, float)):
        joined = ".".join(prefix).lower()
        if any(token in joined for token in ("wall", "elapsed", "second", "duration")):
            found.append(".".join(prefix))
    return found


def _stage_a() -> list[dict[str, Any]]:
    rows = []
    for seed, path in STAGE_A_SUMMARIES.items():
        summary = _load(path)
        if int(summary["model_seed"]) != seed:
            raise ComputationalCostError(f"Stage A model-seed mismatch: {path}")
        training_runs = summary["training"]["runs"]
        validation_runs = [
            run
            for record in summary["validation_history"]
            for run in record["runs"]
        ]
        if len(training_runs) != 500 or len(validation_runs) != 400:
            raise ComputationalCostError(f"unexpected Stage A workload: {path}")
        train_seconds = sum(float(run["wall_seconds"]) for run in training_runs)
        validation_seconds = sum(
            float(run["wall_seconds"]) for run in validation_runs
        )
        rows.append(
            {
                "model_seed": seed,
                "training_episodes": len(training_runs),
                "validation_rollouts": len(validation_runs),
                "recorded_training_episode_seconds": train_seconds,
                "recorded_validation_seconds": validation_seconds,
                "recorded_workload_seconds": train_seconds + validation_seconds,
                "recorded_workload_hours": (train_seconds + validation_seconds) / 3600.0,
                "complete_process_elapsed_seconds": None,
                "training_hardware": None,
                "interpretation": (
                    "sum of persisted per-episode and validation wall clocks; "
                    "not a process-wide elapsed training time"
                ),
                "source": str(path.relative_to(PROJECT_ROOT)),
                "source_sha256": _sha(path),
            }
        )
    return rows


def _stage_b(parameter_registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    handling = parameter_registry["methods"]["conditioned_vcg"][
        "handling_predictor"
    ]
    episodes = handling["collection_episodes_by_seed"]
    rounds = handling["rounds_by_seed"]
    rows = []
    for seed, path in STAGE_B_SUMMARIES.items():
        summary = _load(path)
        timing_fields = _timing_named_scalars(summary)
        if timing_fields:
            raise ComputationalCostError(
                f"unexpected persisted Stage B timing fields in {path}: {timing_fields}"
            )
        rows.append(
            {
                "model_seed": seed,
                "collection_episodes": int(episodes[str(seed)]),
                "training_rounds": int(rounds[str(seed)]),
                "episodes_per_round": int(handling["episodes_per_round"]),
                "fit_epochs_per_round": int(handling["fit_epochs_per_round"]),
                "fit_batch_size": int(handling["fit_batch_size"]),
                "collection_elapsed_seconds": None,
                "fitting_elapsed_seconds": None,
                "complete_process_elapsed_seconds": None,
                "training_hardware": None,
                "interpretation": (
                    "rollout collection and fitting wall clocks were not persisted"
                ),
                "source": str(path.relative_to(PROJECT_ROOT)),
                "source_sha256": _sha(path),
            }
        )
    return rows


def _hardware_key(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decision_components(decision: Mapping[str, Any]) -> dict[str, float]:
    encoding = float(decision["representation_seconds"])
    qop = float(decision["qop_seconds"])
    qn = float(decision["qn_seconds"])
    predictions = qop + qn
    neural = encoding + predictions
    certification = float(decision["frontier_seconds"])
    residual = float(decision["selection_residual_seconds"])
    complete = float(decision["end_to_end_decision_seconds"])
    reconstructed = certification + neural + residual
    if abs(reconstructed - complete) > max(1e-8, 1e-8 * complete):
        raise ComputationalCostError("decision timing components do not close")
    return {
        "encoding": encoding,
        "qop": qop,
        "qn": qn,
        "prediction_heads_total": predictions,
        "neural_total": neural,
        "certification": certification,
        "selection_guard_residual": residual,
        "complete_decision": complete,
    }


def _inference() -> dict[str, Any]:
    contract = _self_hashed(E13_ROOT / "e13-contract.json", "contract_sha256")
    report = _self_hashed(E13_ROOT / "e13-report.json", "report_sha256")
    if report["contract_sha256"] != contract["contract_sha256"]:
        raise ComputationalCostError("E13 report/contract mismatch")
    by_summary = {
        item["scenario"]["scenario_id"]: item for item in report["summaries"]
    }
    hardware: dict[str, Mapping[str, Any]] = {}
    scenario_rows = []
    pooled: list[dict[str, float]] = []
    ledger_sources = []
    for scenario_id in contract["main_scenarios"]:
        decisions: list[dict[str, float]] = []
        for seed in contract["instance_seeds"]:
            path = E13_ROOT / "run-ledger" / scenario_id / f"seed-{seed}.json"
            ledger = _self_hashed(path, "ledger_sha256")
            if ledger["contract_sha256"] != contract["contract_sha256"]:
                raise ComputationalCostError(f"E13 ledger contract mismatch: {path}")
            row = ledger["row"]
            if row["scenario_id"] != scenario_id or int(row["instance_seed"]) != int(seed):
                raise ComputationalCostError(f"E13 ledger coordinate mismatch: {path}")
            hardware[_hardware_key(row["hardware"])] = row["hardware"]
            for decision in row["decision_costs"]:
                if not decision["cold_first_decision"]:
                    decisions.append(_decision_components(decision))
            ledger_sources.append(
                {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha(path)}
            )
        if not decisions:
            raise ComputationalCostError(f"no warm decisions: {scenario_id}")
        complete = _dist(item["complete_decision"] for item in decisions)
        expected = by_summary[scenario_id]["latency"][
            "warm_end_to_end_decision_seconds"
        ]
        for key in ("n", "median", "p95", "maximum"):
            if abs(float(complete[key]) - float(expected[key])) > 1e-10:
                raise ComputationalCostError(
                    f"E13 warm latency mismatch for {scenario_id}/{key}"
                )
        scenario = by_summary[scenario_id]["scenario"]
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "grid": [scenario["rows"], scenario["cols"]],
                "occupancy_level": scenario["occupancy_level"],
                "initial_occupancy_ratio": scenario["initial_occupancy_ratio"],
                "warm_decisions": len(decisions),
                "neural_total_seconds": _dist(
                    item["neural_total"] for item in decisions
                ),
                "certification_seconds": _dist(
                    item["certification"] for item in decisions
                ),
                "selection_guard_residual_seconds": _dist(
                    item["selection_guard_residual"] for item in decisions
                ),
                "complete_decision_seconds": complete,
            }
        )
        pooled.extend(decisions)
    if len(hardware) != 1:
        raise ComputationalCostError("E13 hardware differs across main-panel ledgers")
    components = {
        name: _dist(item[name] for item in pooled)
        for name in (
            "encoding",
            "qop",
            "qn",
            "prediction_heads_total",
            "neural_total",
            "certification",
            "selection_guard_residual",
            "complete_decision",
        )
    }
    return {
        "scope": (
            "E13 main 5x5/8x8/10x10 low-medium-high panel; all three "
            "instance seeds; warm decisions only"
        ),
        "cold_first_decisions_excluded": True,
        "hardware": next(iter(hardware.values())),
        "process_isolation": contract["one_process_per_episode_required_for_latency_and_memory"],
        "model_seed": contract["model_seed"],
        "preference_lambda": contract["preference_lambda"],
        "component_definition": {
            "encoding": "graph encoder plus action encoder forward-hook time",
            "qop": "operational-head forward-hook time",
            "qn": "handling-head forward-hook time",
            "prediction_heads_total": "Qop + QN",
            "neural_total": "encoding + Qop + QN",
            "certification": "complete exact-SAFE frontier construction, including candidate-generation overhead and D12 proof construction",
            "selection_guard_residual": "selection wall time not captured by neural hooks; includes mode aggregation, ranking, liveness guard, and bookkeeping",
            "complete_decision": "certification + full selection wall time",
        },
        "pooled_warm_decisions": len(pooled),
        "pooled_components_seconds": components,
        "scenarios": scenario_rows,
        "sources": {
            "contract": {
                "path": str((E13_ROOT / "e13-contract.json").relative_to(PROJECT_ROOT)),
                "sha256": _sha(E13_ROOT / "e13-contract.json"),
            },
            "report": {
                "path": str((E13_ROOT / "e13-report.json").relative_to(PROJECT_ROOT)),
                "sha256": _sha(E13_ROOT / "e13-report.json"),
            },
            "ledgers": ledger_sources,
        },
    }


def build() -> dict[str, Any]:
    registry_path = E1_ROOT / "e1-parameter-register.json"
    parameter_registry = _load(registry_path)
    if parameter_registry.get("registry_sha256") != _canonical_hash(
        parameter_registry, omit="registry_sha256"
    ):
        raise ComputationalCostError("parameter registry self-hash mismatch")
    inference = _inference()
    e1_runtime = parameter_registry["shared"]["evaluation"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "title": "Computational-cost evidence recovered from frozen artifacts",
        "no_new_training_or_evaluation": True,
        "training": {
            "stage_A_operational": _stage_a(),
            "stage_B_handling": _stage_b(parameter_registry),
            "claim_boundary": (
                "Complete elapsed training time and training hardware are unavailable. "
                "Stage A recorded-workload sums are not substitutes for process-wide elapsed time."
            ),
        },
        "runtime_contexts": {
            "stage_A_training": {
                "hardware": None,
                "thread_settings": None,
                "status": "unavailable_in_training_artifacts",
            },
            "stage_B_training": {
                "hardware": None,
                "thread_settings": None,
                "status": "unavailable_in_training_artifacts",
            },
            "E1_common_evaluation": {
                "device": e1_runtime["execution_device"],
                "torch_threads": e1_runtime["torch_threads"],
                "software_versions": e1_runtime["software_versions"],
                "cpu_model": None,
                "platform": None,
                "status": "runtime_and_software_recorded_but_machine_model_unavailable",
            },
            "E13_latency_measurement": {
                **inference["hardware"],
                "process_isolation": inference["process_isolation"],
                "status": "recorded_in_every_main_panel_ledger",
            },
        },
        "inference": inference,
        "parameter_registry": {
            "path": str(registry_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha(registry_path),
            "registry_sha256": parameter_registry["registry_sha256"],
        },
    }
    result["report_sha256"] = _canonical_hash(result, omit="report_sha256")
    return result


def _number(value: float | None, digits: int = 3) -> str:
    return "unavailable" if value is None else f"{float(value):.{digits}f}"


def render(report: Mapping[str, Any]) -> str:
    stage_a = report["training"]["stage_A_operational"]
    stage_b = report["training"]["stage_B_handling"]
    inference = report["inference"]
    hardware = inference["hardware"]
    contexts = report["runtime_contexts"]
    components = inference["pooled_components_seconds"]
    lines = [
        "# Computational-cost reporting for Section 6",
        "",
        "This report uses only existing frozen artifacts; it performs no training or evaluation. It distinguishes persisted measurements from unavailable quantities.",
        "",
        "## Training-time audit",
        "",
        "| Stage | Model seed | Recorded workload | Recorded timed workload | Complete elapsed training time | Training hardware |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in stage_a:
        lines.append(
            f"| A: encoder + operational critic | {row['model_seed']} | "
            f"{row['training_episodes']} training episodes + "
            f"{row['validation_rollouts']} validation rollouts | "
            f"{row['recorded_workload_hours']:.2f} h* | unavailable | unavailable |"
        )
    for row in stage_b:
        lines.append(
            f"| B: handling predictor | {row['model_seed']} | "
            f"{row['collection_episodes']} collection episodes; "
            f"{row['training_rounds']} rounds x {row['fit_epochs_per_round']} fit epochs | "
            "unavailable | unavailable | unavailable |"
        )
    lines.extend(
        [
            "",
            "*Stage A's value is the sum of persisted in-episode wall clocks and the 400 scheduled validation-rollout clocks. It includes the agent updates executed inside those episodes, but excludes unmeasured process startup, checkpoint I/O, orchestration gaps, and any other untimed work. It is therefore an instrumented workload total, not complete elapsed training time. Stage B persisted neither collection nor fitting wall clocks. Neither stage persisted training-machine details. These unavailable values must not be inferred from episode or round counts.*",
            "",
            "## Hardware and runtime contexts",
            "",
            "| Measurement context | Device and machine | Thread settings | Software / platform | Availability |",
            "|---|---|---|---|---|",
            "| Stage A training | unavailable | unavailable | unavailable | Training artifacts did not persist machine metadata |",
            "| Stage B training | unavailable | unavailable | unavailable | Training artifacts did not persist machine metadata |",
            f"| E1 common evaluation | {contexts['E1_common_evaluation']['device']}; CPU model unavailable | PyTorch threads={contexts['E1_common_evaluation']['torch_threads']} | Python {contexts['E1_common_evaluation']['software_versions']['python']}; NumPy {contexts['E1_common_evaluation']['software_versions']['numpy']}; PyTorch {contexts['E1_common_evaluation']['software_versions']['torch']}; platform unavailable | Runtime/software recorded; machine model unavailable |",
            f"| E13 latency measurement | {hardware['cpu_model']}; {hardware['logical_cpu_count']} logical CPUs; device={hardware['device']} | intra-op={hardware['torch_intraop_threads']}; inter-op={hardware['torch_interop_threads']}; isolated process per episode | Python {hardware['python']}; PyTorch {hardware['torch']}; {hardware['platform']} | Fully recorded in every main-panel ledger |",
            "",
            "Hardware contexts are not interchangeable. E13 decision-time measurements must be reported with the E13 machine and thread settings; the E1 one-thread contract describes benchmark execution but does not identify its processor. Current-machine specifications must not be substituted retrospectively for missing training or E1 hardware metadata.",
            "",
            "## Frozen-controller inference and decision timing",
            "",
            f"Timing hardware: **{hardware['cpu_model']}**, {hardware['logical_cpu_count']} logical CPUs, CPU inference, PyTorch {hardware['torch']}, {hardware['torch_intraop_threads']} intra-op and {hardware['torch_interop_threads']} inter-op threads, {hardware['platform']}. Every episode ran in an isolated process. These settings apply to E13 latency only.",
            "",
            f"The following component distribution pools {inference['pooled_warm_decisions']:,} warm decisions from the nine E13 size-by-occupancy coordinates. The first decision of every episode is excluded.",
            "",
            "| Component | Median (ms) | p95 (ms) | Maximum (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    component_labels = (
        ("encoding", "Graph + candidate encoding"),
        ("qop", "Operational prediction Qop"),
        ("qn", "Handling prediction QN"),
        ("prediction_heads_total", "Prediction heads total (Qop + QN)"),
        ("neural_total", "Neural inference total"),
        ("certification", "Certified-frontier construction"),
        ("selection_guard_residual", "Selector/guard residual"),
        ("complete_decision", "Complete controller decision"),
    )
    for key, label in component_labels:
        item = components[key]
        lines.append(
            f"| {label} | {1000.0 * item['median']:.2f} | "
            f"{1000.0 * item['p95']:.2f} | {1000.0 * item['maximum']:.2f} |"
        )
    lines.extend(
        [
            "",
            "`Neural inference total` is encoding + Qop + QN. `Selector/guard residual` is the measured selection time not captured by the neural hooks; it includes mode aggregation, ranking, the liveness guard, and bookkeeping. `Complete controller decision` is certified-frontier construction plus the full selection wall time.",
            "",
            "### Warm end-to-end decision latency by scale and occupancy",
            "",
            "| Coordinate | Warm decisions | Neural median (s) | Neural p95 (s) | Decision median (s) | Decision p95 (s) | Decision maximum (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inference["scenarios"]:
        item = row["complete_decision_seconds"]
        neural = row["neural_total_seconds"]
        level = row["occupancy_level"][0].upper()
        occupancy = 100.0 * row["initial_occupancy_ratio"]
        lines.append(
            f"| {row['grid'][0]}x{row['grid'][1]} {level} ({occupancy:.1f}%) | "
            f"{row['warm_decisions']} | {neural['median']:.3f} | "
            f"{neural['p95']:.3f} | {item['median']:.3f} | "
            f"{item['p95']:.3f} | {item['maximum']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These latency distributions describe the decisions actually reached at each coordinate. They are not simulation steps, delivery-time errors, or matched counterfactual timings across different scale/occupancy regimes. Certification includes candidate generation, exact search, cache handling, and D12 family-proof construction; the neural columns measure only the hooked forward-pass components.",
            "",
            "## Recommended manuscript wording",
            "",
            "Complete elapsed training time was not captured. Stage A logs retain per-episode and scheduled-validation wall clocks, whose sums range from "
            f"{min(row['recorded_workload_hours'] for row in stage_a):.2f} to "
            f"{max(row['recorded_workload_hours'] for row in stage_a):.2f} h per seed, but these are instrumented workload totals rather than process-wide training times. Stage B collection and fitting time was not persisted. Frozen-controller latency was measured separately on the stated CPU platform; neural forward-pass time is reported independently from certified-frontier construction and complete controller decision time.",
            "",
            f"Machine-readable report SHA-256: `{report['report_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paper-copy",
        type=Path,
        default=PROJECT_ROOT / "docs/PAPER_COMPUTATIONAL_COST_REPORT.md",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build()
    json_path = output / "computational-cost-report.json"
    markdown_path = output / "computational-cost-table.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    paper_copy = args.paper_copy.resolve()
    paper_copy.parent.mkdir(parents=True, exist_ok=True)
    paper_copy.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(json_path),
                "table": str(markdown_path),
                "paper_copy": str(paper_copy),
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
