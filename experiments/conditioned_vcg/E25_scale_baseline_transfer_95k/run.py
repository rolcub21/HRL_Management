#!/usr/bin/env python3
"""E25: non-certified benchmark baselines on E13's scale/occupancy grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import fmean, stdev
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import compare_vcg_v2_3_matched_baselines as matched
import compare_vcg_v2_3_capacity_aware_ga_repair_v2 as ga_repair_v2
from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
import run_kim2020_final86_comparison as kim_final
import run_vcg_conditioned_final_comparison_90k as e1
import track_b_urgency_evaluate as baseline
import vcg_v2_3_kim2020_supplement_evaluation as kim85


PROTOCOL = "vcg_conditioned_e25_scale_baseline_transfer_95k_v1"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e25-scale-baselines-95k"
E13_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e13-scalability-95k-v2"
CONTRACT_NAME = "e25-contract.json"
REPORT_NAME = "e25-report.json"
TABLE_NAME = "e25-scale-baseline-table.md"
FIGURE_NAME = "e25-scale-baseline-comparison"

SCENARIOS = tuple(e13.MAIN_SCENARIOS)
INSTANCE_SEEDS = tuple(e13.INSTANCE_SEEDS)
METHOD_VCG = "vcg_lambda_010"
METHOD_DYNAMIC = e1.DYNAMIC_METHOD
METHOD_GA = e1.GA_METHOD
METHOD_A3C = e1.KIM_METHOD
NEW_METHODS = (METHOD_DYNAMIC, METHOD_GA, METHOD_A3C)
METHODS = (METHOD_VCG, *NEW_METHODS)
METHOD_LABELS = {
    METHOD_VCG: r"VCG, $\lambda=.10$",
    METHOD_GA: "Capacity-aware GA",
    METHOD_DYNAMIC: "Dynamic PSLAP",
    METHOD_A3C: "A3C adaptation",
}
REPLICATES = {
    METHOD_DYNAMIC: tuple(range(1)),
    METHOD_GA: tuple(range(4)),
    METHOD_A3C: tuple(range(15)),
}
EXPECTED_NEW_ROWS = sum(
    len(REPLICATES[method]) * len(SCENARIOS) * len(INSTANCE_SEEDS)
    for method in NEW_METHODS
)
EXPECTED_REUSED_VCG_ROWS = len(SCENARIOS) * len(INSTANCE_SEEDS)

GA_RNG_BASE = 645_000_000
GA_RNG_STRIDE = 2_001
A3C_POLICY_RNG_BASE = 646_000_000
TARGET_WINDOW = 20.0
MAX_DEFER_STEPS = 10
LOOKAHEAD_MARGIN_STEPS = 2.0
EXIT_WIDTH = 3
ROLLING_POPULATION = 16
ROLLING_GENERATIONS = 10
ROLLING_GA_EGRESS_WEIGHT = 4


class E25Error(RuntimeError):
    pass


def _json_form(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        _json_form(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _hashed(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E25Error(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E25Error(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E25Error(f"{label} is not a JSON object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise E25Error(f"{label} self-hash mismatch")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(_json_form(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "device": "cpu",
    }


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "compare_vcg_v2_3_matched_baselines.py",
        PROJECT_ROOT / "track_b_urgency_evaluate.py",
        PROJECT_ROOT / "example/Options/selector_v5.py",
        PROJECT_ROOT / "PSLAP/kim2020_a3c_spatial.py",
        PROJECT_ROOT / "PSLAP/ga_policy.py",
        PROJECT_ROOT / "PSLAP/track_a.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _parent() -> tuple[dict, dict, dict]:
    contract, manifest, _ = e13.authenticate_frozen(E13_OUTPUT)
    report = _self_hashed(E13_OUTPUT / e13.REPORT_NAME, "report_sha256", "E13 report")
    if (
        report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("observed_rows") != report.get("expected_rows")
        or report.get("main_panel_complete") is not True
    ):
        raise E25Error("frozen E13 parent is not complete")
    for scenario_id in SCENARIOS:
        for seed in INSTANCE_SEEDS:
            e13._authenticate_ledger(E13_OUTPUT, contract, scenario_id, seed)
    return contract, manifest, report


def _a3c_inputs() -> tuple[dict[int, dict], dict[int, dict]]:
    completions = e1._kim_completions(PROJECT_ROOT)
    checkpoints = {
        seed: kim85._load_checkpoint_bound(
            kim_final.TRAINING_ROOT / "training" / f"seed-{seed}" / "best.pth",
            completions[seed]["selected_checkpoint_raw_sha256"],
        )
        for seed in kim_final.MODEL_SEEDS
    }
    return completions, checkpoints


def _contract_payload(parent_contract, parent_manifest, parent_report) -> dict:
    completions, _checkpoints = _a3c_inputs()
    scenarios = [occupancy.SCENARIO_BY_ID[item].public_dict() for item in SCENARIOS]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "post_hoc": True,
        "question": (
            "does VCG's operational degradation with yard size differ from "
            "non-certified Table-3 baselines on identical E13 workloads"
        ),
        "parent": {
            "e13_contract_sha256": parent_contract["contract_sha256"],
            "e13_manifest_sha256": parent_manifest["manifest_sha256"],
            "e13_report_sha256": parent_report["report_sha256"],
            "vcg_rows_reused": EXPECTED_REUSED_VCG_ROWS,
            "reused_controls_are_independent_replications": False,
        },
        "coordinates": {
            "scenarios": scenarios,
            "instance_seeds": list(INSTANCE_SEEDS),
            "methods": list(METHODS),
            "new_rows": EXPECTED_NEW_ROWS,
            "reused_vcg_rows": EXPECTED_REUSED_VCG_ROWS,
            "replicates_per_instance": {
                METHOD_VCG: 1,
                **{method: len(REPLICATES[method]) for method in NEW_METHODS},
            },
        },
        "method_contract": {
            METHOD_VCG: {
                "source": "completed_E13_seed0_lambda_.10_rows",
                "certified": True,
            },
            METHOD_DYNAMIC: {
                "source": "E1_duration-aware_Dynamic_PSLAP",
                "certified": False,
                "training": False,
            },
            METHOD_GA: {
                "source": "E1_capacity-aware_rolling_GA",
                "certified": False,
                "optimizer_replicates": 4,
            },
            METHOD_A3C: {
                "source": "E1_A3C_adaptation",
                "certified": False,
                "model_seeds": list(kim_final.MODEL_SEEDS),
                "policy_rollouts_per_model_instance": len(kim_final.ROLLOUTS),
                "fully_convolutional_geometry_transfer": True,
                "geometry_shift_explicitly_enabled": True,
                "arrival_regime_metadata_shift_explicitly_enabled": True,
                "checkpoint_sha256": {
                    str(seed): completions[seed]["selected_checkpoint_raw_sha256"]
                    for seed in kim_final.MODEL_SEEDS
                },
            },
        },
        "shared_baseline_stack": {
            "scheduler": "duration_aware_source_neutral",
            "assignment_commitment": baseline.DECISION_EPOCH_RESERVED,
            "information_regime": "online_arrived_only",
            "exit_width": EXIT_WIDTH,
            "max_defer_steps": MAX_DEFER_STEPS,
            "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
            "target_window": TARGET_WINDOW,
        },
        "aggregation": (
            "average nuisance/model replications within EpisodeInstance, then "
            "give the three EpisodeInstances equal weight within configuration"
        ),
        "quality_scope": "strict-complete runs only; completion always shown separately",
        "composition_warning": (
            "every main-grid workload has two arriving jobs; the remaining jobs "
            "are initially stored, so absolute size trends also change composition"
        ),
        "training_runs": 0,
        "rng": {
            "ga_base": GA_RNG_BASE,
            "ga_stride": GA_RNG_STRIDE,
            "a3c_policy_base": A3C_POLICY_RNG_BASE,
        },
        "source_sha256": _source_hashes(),
    }


def prepare(output: Path) -> dict:
    parent_contract, parent_manifest, parent_report = _parent()
    expected = _hashed(
        _contract_payload(parent_contract, parent_manifest, parent_report),
        "contract_sha256",
    )
    path = output.resolve() / CONTRACT_NAME
    if path.is_file():
        observed = _self_hashed(path, "contract_sha256", "E25 contract")
        if observed != expected:
            raise E25Error("existing E25 contract differs from current sources")
    else:
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "output": str(output.resolve()),
        "new_rows": EXPECTED_NEW_ROWS,
        "reused_vcg_rows": EXPECTED_REUSED_VCG_ROWS,
        "contract_sha256": expected["contract_sha256"],
        "training_runs": 0,
    }


def authenticate(output: Path) -> tuple[dict, dict, dict, dict]:
    parent_contract, parent_manifest, parent_report = _parent()
    expected = _hashed(
        _contract_payload(parent_contract, parent_manifest, parent_report),
        "contract_sha256",
    )
    observed = _self_hashed(output.resolve() / CONTRACT_NAME, "contract_sha256", "E25 contract")
    if observed != expected:
        raise E25Error("E25 contract, sources, or parents changed")
    return observed, parent_contract, parent_manifest, parent_report


def _coordinate_index(scenario_id: str, seed: int) -> int:
    return SCENARIOS.index(scenario_id) * len(INSTANCE_SEEDS) + INSTANCE_SEEDS.index(seed)


def _replicate_identity(method: str, scenario_id: str, seed: int, replicate: int) -> dict:
    coordinate = _coordinate_index(scenario_id, seed)
    if method == METHOD_DYNAMIC:
        if replicate != 0:
            raise E25Error("Dynamic PSLAP has only replicate 0")
        return {"replicate": 0, "model_seed": None, "rng_index": None, "rng_seed": None}
    if method == METHOD_GA:
        if replicate not in REPLICATES[METHOD_GA]:
            raise E25Error("GA replicate must be 0..3")
        rng_seed = GA_RNG_BASE + GA_RNG_STRIDE * (4 * coordinate + replicate)
        return {"replicate": replicate, "model_seed": None, "rng_index": replicate, "rng_seed": rng_seed}
    if method == METHOD_A3C:
        if replicate not in REPLICATES[METHOD_A3C]:
            raise E25Error("A3C replicate must be 0..14")
        model_seed, rollout = divmod(replicate, len(kim_final.ROLLOUTS))
        rng_seed = A3C_POLICY_RNG_BASE + 1_000 * model_seed + 10 * coordinate + rollout
        return {"replicate": replicate, "model_seed": model_seed, "rng_index": rollout, "rng_seed": rng_seed}
    raise E25Error(f"unsupported new method: {method}")


def _ledger_path(output: Path, method: str, scenario_id: str, seed: int, replicate: int) -> Path:
    return output / "run-ledger" / method / scenario_id / f"seed-{seed}-rep-{replicate}.json"


def _runtime_args(scenario, *, ga_seed_base: int) -> SimpleNamespace:
    return SimpleNamespace(
        lam=float(1.0 / occupancy.EXTRA_ARRIVAL_GAP),
        mu=float(occupancy.PROC_MEAN),
        grid_rows=scenario.rows,
        grid_cols=scenario.cols,
        exit_width=EXIT_WIDTH,
        number_blocks=scenario.total_jobs,
        max_steps=scenario.max_steps,
        max_defer_steps=MAX_DEFER_STEPS,
        lookahead_margin_steps=LOOKAHEAD_MARGIN_STEPS,
        target_window=TARGET_WINDOW,
        assignment_commitment=baseline.DECISION_EPOCH_RESERVED,
        allow_selector_regime_shift=True,
        allow_assignment_geometry_shift=True,
        instance=None,
        save_instances_dir=None,
        device="cpu",
        ga_seed_base=int(ga_seed_base),
        rolling_population=ROLLING_POPULATION,
        rolling_generations=ROLLING_GENERATIONS,
        rolling_ga_egress_weight=ROLLING_GA_EGRESS_WEIGHT,
    )


def _failure_class(raw: Mapping, strict: bool) -> str:
    if strict:
        return "completed"
    if raw.get("method_failure_reason"):
        return str(raw["method_failure_reason"]).split(":", 1)[0]
    if bool(raw.get("truncated")):
        return "step_limit"
    return "strict_completion_failed"


def _normalize(raw: Mapping, *, method: str, scenario, identity: Mapping, elapsed: float) -> dict:
    strict = bool(raw.get("strict_method_success")) and int(raw.get("delivery_count", -1)) == scenario.total_jobs
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.public_dict(),
        "instance_seed": int(raw["eval_seed"]),
        "episode_instance_id": raw["instance_id"],
        "schedule_id": raw["schedule_id"],
        **identity,
        "strict_safe_complete": strict,
        "failure_class": _failure_class(raw, strict),
        "method_failure_reason": raw.get("method_failure_reason"),
        "deliveries_at_stop": int(raw.get("delivery_count", 0)),
        "required_deliveries": scenario.total_jobs,
        "steps_at_stop": int(raw.get("steps", 0)),
        "physical_rehandles_at_stop": int(raw.get("physical_storage_relocations", 0)),
        "mean_absolute_error": float(raw["mean_absolute_error"]) if strict else None,
        "within_target_window_rate": float(raw["within_target_window_rate"]) if strict else None,
        "physical_rehandles_per_100_required_deliveries": (
            100.0 * float(raw["physical_storage_relocations"]) / scenario.total_jobs
            if strict else None
        ),
        "steps_per_delivery": float(raw["steps"]) / scenario.total_jobs if strict else None,
        "initially_stored_jobs": scenario.initial_occupied_slots,
        "arriving_jobs": scenario.total_jobs - scenario.initial_occupied_slots,
        "episode_wall_seconds": float(elapsed),
        "assignment_source": raw.get("assignment_source"),
        "assignment_source_version": raw.get("assignment_source_version"),
        "assignment_policy_realization": raw.get("assignment_policy_realization"),
        "selector_deployment_digest": raw.get("selector_deployment_digest"),
        "invalid_assignment_count": int(raw["selector_audit"]["invalid_assignment_count"]),
        "fallback_count": int(raw["selector_audit"].get("fallback_count", 0)),
        "illegal_drops": int(raw.get("illegal_drops", 0)),
        "geometry_signature": raw["geometry"]["geometry_signature"],
        "geometry_shift_explicitly_enabled": method == METHOD_A3C,
        "training_runs": 0,
    }


def run_one(output: Path, method: str, scenario_id: str, seed: int, replicate: int) -> dict:
    output = output.resolve()
    contract, _parent_contract, parent_manifest, _parent_report = authenticate(output)
    if method not in NEW_METHODS or scenario_id not in SCENARIOS or seed not in INSTANCE_SEEDS:
        raise E25Error("unsupported E25 coordinate")
    identity = _replicate_identity(method, scenario_id, seed, replicate)
    path = _ledger_path(output, method, scenario_id, seed, replicate)
    if path.is_file():
        ledger = _self_hashed(path, "ledger_sha256", "E25 ledger")
        if ledger.get("contract_sha256") != contract["contract_sha256"]:
            raise E25Error("E25 ledger contract changed")
        return dict(ledger["row"])

    record = e13._record(parent_manifest, scenario_id, seed)
    instance = e13._load_instance(record)
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    args = _runtime_args(
        scenario,
        ga_seed_base=(
            int(identity["rng_seed"]) - seed
            if method == METHOD_GA
            else GA_RNG_BASE
        ),
    )
    if method == METHOD_DYNAMIC:
        payload = None
        assignment_source = matched.DETERMINISTIC_METHOD_TO_SOURCE[matched.DYNAMIC_METHOD]
        policy_seed = None
    elif method == METHOD_GA:
        implementation = ga_repair_v2._impl
        payload = None
        assignment_source = implementation.REPAIRED_METHOD_TO_SOURCE[
            implementation.REPAIRED_2009_METHOD
        ]
        policy_seed = None
    else:
        _completions, checkpoints = _a3c_inputs()
        payload = checkpoints[int(identity["model_seed"])]
        assignment_source = kim85.TRACK_A_KIM2020_A3C_SPATIAL
        policy_seed = int(identity["rng_seed"])

    started = perf_counter()
    raw = baseline._evaluate_one(
        args,
        seed,
        payload,
        episode_instance=instance,
        scheduler_variant=baseline.DURATION_AWARE_VARIANT,
        assignment_source=assignment_source,
        source_neutral_scheduler=True,
        assignment_commitment=baseline.DECISION_EPOCH_RESERVED,
        deployment_mode=kim85.DEPLOYMENT_STOCHASTIC,
        policy_seed=policy_seed,
    )
    elapsed = perf_counter() - started
    row = _normalize(raw, method=method, scenario=scenario, identity=identity, elapsed=elapsed)
    if row["episode_instance_id"] != record["episode_instance_id"] or row["schedule_id"] != record["schedule_id"]:
        raise E25Error("baseline execution lost its E13 instance binding")
    ledger = _hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "row": row,
        },
        "ledger_sha256",
    )
    _atomic_json(path, ledger)
    return row


def run_method(output: Path, method: str, *, pilot: bool = False) -> dict:
    if method not in NEW_METHODS:
        raise E25Error("run-method requires Dynamic PSLAP, GA, or A3C")
    if pilot:
        coordinates = (("size_8x8_occ_high", INSTANCE_SEEDS[0], REPLICATES[method][0]),)
    else:
        coordinates = tuple(
            (scenario, seed, replicate)
            for scenario in SCENARIOS
            for seed in INSTANCE_SEEDS
            for replicate in REPLICATES[method]
        )
    strict = 0
    for index, (scenario, seed, replicate) in enumerate(coordinates, start=1):
        row = run_one(output, method, scenario, seed, replicate)
        strict += int(row["strict_safe_complete"])
        print(
            f"E25 {METHOD_LABELS[method]} {index}/{len(coordinates)} | "
            f"{scenario} seed={seed} rep={replicate} strict={int(row['strict_safe_complete'])}",
            flush=True,
        )
    return {
        "status": "complete",
        "scope": "pilot" if pilot else "full_method_grid",
        "method": method,
        "rows": len(coordinates),
        "strict_safe_complete_rows": strict,
        "training_runs": 0,
    }


def run_pilot(output: Path) -> dict:
    results = [run_method(output, method, pilot=True) for method in NEW_METHODS]
    return {
        "status": "complete",
        "scope": "three-run geometry/integration pilot; not paper evidence",
        "scenario_id": "size_8x8_occ_high",
        "instance_seed": INSTANCE_SEEDS[0],
        "methods": results,
        "paper_evidence": False,
        "training_runs": 0,
    }


def _new_rows(output: Path) -> list[dict]:
    rows = []
    root = output / "run-ledger"
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*.json")):
        ledger = _self_hashed(path, "ledger_sha256", "E25 ledger")
        rows.append(dict(ledger["row"]))
    return rows


def _vcg_rows(parent_contract: Mapping) -> list[dict]:
    rows = []
    for scenario_id in SCENARIOS:
        scenario = occupancy.SCENARIO_BY_ID[scenario_id]
        for seed in INSTANCE_SEEDS:
            source = e13._authenticate_ledger(E13_OUTPUT, parent_contract, scenario_id, seed)["row"]
            rows.append(
                {
                    "method": METHOD_VCG,
                    "method_label": METHOD_LABELS[METHOD_VCG],
                    "scenario_id": scenario_id,
                    "scenario": scenario.public_dict(),
                    "instance_seed": seed,
                    "replicate": 0,
                    "model_seed": 0,
                    "rng_index": None,
                    "strict_safe_complete": bool(source["strict_safe_complete"]),
                    "failure_class": source["failure_class"],
                    "mean_absolute_error": source["mean_absolute_error"],
                    "within_target_window_rate": source["within_target_window_rate"],
                    "physical_rehandles_per_100_required_deliveries": source[
                        "physical_rehandles_per_100_required_deliveries"
                    ],
                    "steps_per_delivery": source["steps_per_delivery"],
                    "initially_stored_jobs": scenario.initial_occupied_slots,
                    "arriving_jobs": scenario.total_jobs - scenario.initial_occupied_slots,
                    "source_ledger": str(
                        e13._ledger_path(E13_OUTPUT, scenario_id, seed).relative_to(PROJECT_ROOT)
                    ),
                    "reused_control": True,
                }
            )
    return rows


QUALITY_METRICS = (
    "mean_absolute_error",
    "within_target_window_rate",
    "physical_rehandles_per_100_required_deliveries",
    "steps_per_delivery",
)


def _instance_means(rows: Sequence[Mapping]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["scenario_id"], int(row["instance_seed"]))].append(row)
    output = []
    for (method, scenario_id, seed), selected in sorted(groups.items()):
        strict = [row for row in selected if row["strict_safe_complete"]]
        record = {
            "method": method,
            "scenario_id": scenario_id,
            "instance_seed": seed,
            "replicate_rows": len(selected),
            "strict_rows": len(strict),
            "strict_completion_rate": len(strict) / len(selected),
        }
        for metric in QUALITY_METRICS:
            values = [float(row[metric]) for row in strict if row.get(metric) is not None]
            record[metric] = fmean(values) if values else None
        output.append(record)
    return output


def _summary(rows: Sequence[Mapping]) -> list[dict]:
    instance_means = _instance_means(rows)
    groups = defaultdict(list)
    for row in instance_means:
        groups[(row["method"], row["scenario_id"])].append(row)
    output = []
    for (method, scenario_id), selected in sorted(groups.items()):
        scenario = occupancy.SCENARIO_BY_ID[scenario_id]
        record = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "scenario_id": scenario_id,
            "yard_size": scenario.rows,
            "occupancy_level": scenario.occupancy_level,
            "initial_occupancy_ratio": scenario.initial_occupied_slots / scenario.storage_capacity,
            "initially_stored_jobs": scenario.initial_occupied_slots,
            "arriving_jobs": scenario.total_jobs - scenario.initial_occupied_slots,
            "episode_instances": len(selected),
            "strict_completion_rate": fmean(item["strict_completion_rate"] for item in selected),
            "strict_completion_min": min(item["strict_completion_rate"] for item in selected),
            "strict_completion_max": max(item["strict_completion_rate"] for item in selected),
        }
        for metric in QUALITY_METRICS:
            values = [float(item[metric]) for item in selected if item.get(metric) is not None]
            record[metric] = fmean(values) if values else None
            record[f"{metric}_instance_n"] = len(values)
            record[f"{metric}_instance_min"] = min(values) if values else None
            record[f"{metric}_instance_max"] = max(values) if values else None
            record[f"{metric}_instance_sd"] = stdev(values) if len(values) > 1 else None
        output.append(record)
    return output


def _render(summary: Sequence[Mapping], output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        METHOD_VCG: "#0072B2",
        METHOD_GA: "#D55E00",
        METHOD_DYNAMIC: "#009E73",
        METHOD_A3C: "#CC79A7",
    }
    markers = {METHOD_VCG: "o", METHOD_GA: "s", METHOD_DYNAMIC: "^", METHOD_A3C: "D"}
    metrics = (
        ("strict_completion_rate", "Strict completion (%) ↑", 100.0),
        ("mean_absolute_error", "Placement MAE (steps) ↓", 1.0),
        ("physical_rehandles_per_100_required_deliveries", "Rehandles per 100 ↓", 1.0),
        ("steps_per_delivery", "Steps per delivery ↓", 1.0),
    )
    occupancy_levels = ("low", "medium", "high")
    sizes = (5, 8, 10)
    lookup = {(item["method"], item["occupancy_level"], item["yard_size"]): item for item in summary}
    fig, axes = plt.subplots(4, 3, figsize=(8.2, 8.7), sharex=True, sharey="row")
    for column, level in enumerate(occupancy_levels):
        axes[0, column].set_title(f"{level.capitalize()} initial occupancy", fontsize=10)
        for row_index, (metric, label, scale) in enumerate(metrics):
            axis = axes[row_index, column]
            for method in METHODS:
                selected = [lookup.get((method, level, size)) for size in sizes]
                values = [None if item is None or item.get(metric) is None else scale * float(item[metric]) for item in selected]
                lows = []
                highs = []
                for item, value in zip(selected, values):
                    if value is None:
                        lows.append(np.nan)
                        highs.append(np.nan)
                    elif metric == "strict_completion_rate":
                        lows.append(100.0 * float(item["strict_completion_min"]))
                        highs.append(100.0 * float(item["strict_completion_max"]))
                    else:
                        lows.append(float(item[f"{metric}_instance_min"]))
                        highs.append(float(item[f"{metric}_instance_max"]))
                numeric = np.asarray([np.nan if value is None else value for value in values], dtype=float)
                axis.plot(
                    sizes, numeric, color=colors[method], marker=markers[method],
                    linewidth=1.7, markersize=4.6, label=METHOD_LABELS[method], zorder=3,
                )
                axis.fill_between(sizes, lows, highs, color=colors[method], alpha=0.09, linewidth=0)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks(sizes, ["5×5", "8×8", "10×10"])
            if column == 0:
                axis.set_ylabel(label, fontsize=9)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Yard size", fontsize=9)
            if metric == "strict_completion_rate":
                axis.set_ylim(-3, 105)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.96, bottom=0.09, hspace=0.22, wspace=0.12)
    pdf = output / f"{FIGURE_NAME}.pdf"
    png = output / f"{FIGURE_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    data = _hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "summary": list(summary),
            "visual_encoding": {
                "columns": list(occupancy_levels),
                "rows": [item[0] for item in metrics],
                "x": list(sizes),
                "bands": "minimum_to_maximum_across_three_EpisodeInstance_means",
                "colors": colors,
            },
        },
        "data_sha256",
    )
    data_path = output / f"{FIGURE_NAME}-data.json"
    _atomic_json(data_path, data)
    return {"pdf": str(pdf), "png": str(png), "data": str(data_path)}


def analyze(output: Path, *, partial: bool = False) -> dict:
    output = output.resolve()
    contract, parent_contract, _manifest, _report = authenticate(output)
    new_rows = _new_rows(output)
    if len(new_rows) > EXPECTED_NEW_ROWS:
        raise E25Error("E25 contains more rows than the frozen grid")
    if not partial and len(new_rows) != EXPECTED_NEW_ROWS:
        raise E25Error(f"incomplete E25 grid: {len(new_rows)}/{EXPECTED_NEW_ROWS}")
    rows = [*_vcg_rows(parent_contract), *new_rows]
    summary = _summary(rows)
    expected_summaries = len(METHODS) * len(SCENARIOS)
    if not partial and len(summary) != expected_summaries:
        raise E25Error("complete E25 analysis lacks a method/scenario summary")
    figures = _render(summary, output)
    report = _hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if partial else "complete",
            "paper_evidence": not partial,
            "post_hoc": True,
            "expected_new_rows": EXPECTED_NEW_ROWS,
            "observed_new_rows": len(new_rows),
            "reused_vcg_rows": EXPECTED_REUSED_VCG_ROWS,
            "reused_controls_are_independent_replications": False,
            "summaries": summary,
            "failure_counts": {
                method: {
                    name: sum(row["failure_class"] == name for row in new_rows if row["method"] == method)
                    for name in sorted({row["failure_class"] for row in new_rows if row["method"] == method})
                }
                for method in NEW_METHODS
            },
            "quality_scope": "strict-complete runs only",
            "statistical_unit": "EpisodeInstance after within-instance nuisance averaging",
            "figures": figures,
            "runtime": _runtime(),
            "training_runs": 0,
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    lines = [
        "# E25 scale-baseline comparison",
        "",
        "| Method | Yard | Occupancy | Initial + arriving | Strict | MAE | Rehandles/100 | Steps/delivery |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        def fmt(name, multiplier=1.0):
            value = item.get(name)
            return "—" if value is None else f"{multiplier * float(value):.2f}"
        lines.append(
            f"| {item['method_label']} | {item['yard_size']}×{item['yard_size']} | "
            f"{item['occupancy_level']} ({item['initial_occupancy_ratio']:.1%}) | "
            f"{item['initially_stored_jobs']} + {item['arriving_jobs']} | "
            f"{fmt('strict_completion_rate', 100.0)}% | {fmt('mean_absolute_error')} | "
            f"{fmt('physical_rehandles_per_100_required_deliveries')} | "
            f"{fmt('steps_per_delivery')} |"
        )
    lines.extend(
        [
            "",
            "Quality is conditional on strict completion; completion is reported separately. VCG rows are reused E13 controls, not independent replications. Each main-grid workload contains exactly two arriving jobs, so the proportion initially stored rises with scale.",
            "",
        ]
    )
    (output / TABLE_NAME).write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": report["status"],
        "new_rows": len(new_rows),
        "reused_vcg_rows": EXPECTED_REUSED_VCG_ROWS,
        "report": str((output / REPORT_NAME).resolve()),
        "table": str((output / TABLE_NAME).resolve()),
        "figures": figures,
        "training_runs": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run-pilot", "run-one", "run-method", "analyze", "analyze-partial"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=NEW_METHODS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int, choices=INSTANCE_SEEDS)
    parser.add_argument("--replicate", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-pilot":
        result = run_pilot(output)
    elif args.command == "run-one":
        if args.method is None or args.scenario is None or args.seed is None or args.replicate is None:
            raise E25Error("run-one requires --method, --scenario, --seed, and --replicate")
        row = run_one(output, args.method, args.scenario, args.seed, args.replicate)
        result = {"status": "complete", "row": row, "training_runs": 0}
    elif args.command == "run-method":
        if args.method is None:
            raise E25Error("run-method requires --method")
        result = run_method(output, args.method)
    else:
        result = analyze(output, partial=args.command == "analyze-partial")
    print(json.dumps(_json_form(result), indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
