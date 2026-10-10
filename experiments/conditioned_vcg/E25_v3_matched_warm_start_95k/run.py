#!/usr/bin/env python3
"""E25-v3: correct E25's baseline reset and compare matched warm-start yards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E25_scale_baseline_transfer_95k import run as e25
from experiments.conditioned_vcg.E25_v2_full_vcg_family_scale_95k import run as e25v2
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e25_v3_matched_warm_start_95k_v1"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e25-v3-matched-warm-start-95k"
VCG_OUTPUT = e25v2.DEFAULT_OUTPUT
CONTRACT_NAME = "e25-v3-contract.json"
REPORT_NAME = "e25-v3-report.json"
TABLE_NAME = "e25-v3-full-family-table.md"
FIGURE_NAME = "e25-v3-full-family-scale-comparison"

SCENARIOS = e25v2.SCENARIOS
INSTANCE_SEEDS = e25v2.INSTANCE_SEEDS
METHODS = e25v2.BASELINE_METHODS
EXPECTED_BASELINE_ROWS = e25.EXPECTED_NEW_ROWS
EXPECTED_VCG_ROWS = e25v2.EXPECTED_NEW_VCG_ROWS


class E25V3Error(RuntimeError):
    pass


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(e25.__file__).resolve(),
        Path(e25v2.__file__).resolve(),
        Path(occupancy.__file__).resolve(),
        PROJECT_ROOT / "track_b_urgency_evaluate.py",
        PROJECT_ROOT / "example/episode_instance.py",
        PROJECT_ROOT / "example/small_rooms_env.py",
        PROJECT_ROOT / "example/yard_geometry.py",
        PROJECT_ROOT / "example/urgency_scheduler.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): e25v2._sha(path) for path in paths}


def _parents() -> tuple[dict, dict, dict]:
    vcg_contract, parent_bundle = e25v2.authenticate(VCG_OUTPUT)
    vcg_report = e25v2._self_hashed(
        VCG_OUTPUT / e25v2.REPORT_NAME, "report_sha256", "E25-v2 report"
    )
    if (
        vcg_report.get("contract_sha256") != vcg_contract["contract_sha256"]
        or vcg_report.get("status") != "complete"
        or vcg_report.get("observed_new_vcg_rows") != EXPECTED_VCG_ROWS
    ):
        raise E25V3Error("frozen VCG preference parent is incomplete")
    return vcg_contract, vcg_report, parent_bundle[3]


def _contract_payload(vcg_contract: Mapping, vcg_report: Mapping, manifest: Mapping) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "post_hoc": True,
        "question": "how_do_baselines_compare_with_VCG_from_the_same_warm_start_state",
        "correction_of": [e25.PROTOCOL, e25v2.PROTOCOL],
        "historical_e25_baseline_rows_reused": False,
        "vcg_preference_rows_reused": EXPECTED_VCG_ROWS,
        "new_baseline_rows": EXPECTED_BASELINE_ROWS,
        "baseline_replicates_per_instance": {
            method: len(e25.REPLICATES[method]) for method in METHODS
        },
        "scenarios": list(SCENARIOS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "preference_lambdas": list(e25v2.LAMBDAS),
        "reset_contract": (
            "the first K blocks occupy the E13 scenario's declared cells with "
            "stored_time_step=0; all remaining jobs retain their declared arrivals"
        ),
        "per_run_reset_assertion": True,
        "initial_state_is_not_encoded_by_episode_instance_id": True,
        "parent": {
            "vcg_contract_sha256": vcg_contract["contract_sha256"],
            "vcg_report_sha256": vcg_report["report_sha256"],
            "e13_manifest_sha256": manifest["manifest_sha256"],
        },
        "aggregation": (
            "average baseline replications within EpisodeInstance, then average "
            "the three EpisodeInstances equally within configuration"
        ),
        "quality_scope": "strict-complete runs only; completion shown separately",
        "training_runs": 0,
        "source_sha256": _source_hashes(),
    }


def prepare(output: Path) -> dict:
    parents = _parents()
    expected = e25v2._hashed(_contract_payload(*parents), "contract_sha256")
    path = output.resolve() / CONTRACT_NAME
    if path.is_file():
        if e25v2._self_hashed(path, "contract_sha256", "E25-v3 contract") != expected:
            raise E25V3Error("existing E25-v3 contract differs from current sources")
    else:
        e25v2._atomic_json(path, expected)
    return {
        "status": "prepared",
        "output": str(output.resolve()),
        "new_baseline_rows": EXPECTED_BASELINE_ROWS,
        "reused_vcg_rows": EXPECTED_VCG_ROWS,
        "training_runs": 0,
    }


def authenticate(output: Path) -> tuple[dict, tuple[dict, dict, dict]]:
    parents = _parents()
    expected = e25v2._hashed(_contract_payload(*parents), "contract_sha256")
    observed = e25v2._self_hashed(
        output.resolve() / CONTRACT_NAME, "contract_sha256", "E25-v3 contract"
    )
    if observed != expected:
        raise E25V3Error("E25-v3 contract, parent, or sources changed")
    return observed, parents


def _state_signature(blocks, transporter_position) -> str:
    payload = {
        "transporter_position": list(transporter_position),
        "blocks": [
            {
                "label": block.label,
                "position": None if block.position is None else list(block.position),
                "storage_location": (
                    None if block.storage_location is None else list(block.storage_location)
                ),
                "stored": bool(block.stored),
                "stored_time_step": block.stored_time_step,
                "arrival_step": int(block.arrival_step),
                "storage_steps_needed": int(block.storage_steps_needed),
            }
            for block in blocks
        ],
    }
    return hashlib.sha256(e25v2._canonical(payload)).hexdigest()


def _warm_start_audit(env, scenario, instance) -> dict:
    if env.time_steps != 0 or tuple(env.current_state) != tuple(scenario.make_base_env().start_state):
        raise E25V3Error("baseline did not start at the E13 transporter state")
    if env.current_episode_instance.instance_id != instance.instance_id:
        raise E25V3Error("baseline reset changed EpisodeInstance")
    placements = scenario.placement_cells
    for index, block in enumerate(env.blocks):
        if int(block.arrival_step) != instance.arrival_steps[index]:
            raise E25V3Error("baseline reset changed arrival time")
        if int(block.storage_steps_needed) != instance.storage_steps_needed[index]:
            raise E25V3Error("baseline reset changed storage duration")
        if index < len(placements):
            cell = placements[index]
            if (
                not block.stored
                or tuple(block.position or ()) != cell
                or tuple(block.storage_location or ()) != cell
                or block.stored_time_step != 0
                or block.carrying
                or block.delivered
            ):
                raise E25V3Error(f"initial stored block {index} differs from E13")
        elif (
            block.stored
            or block.stored_time_step is not None
            or block.position is not None
            or block.carrying
            or block.delivered
        ):
            raise E25V3Error(f"future arrival {index} was present at reset")
    occupied = env.occupancy_trace[0]["occupied_storage_slots"]
    if occupied != scenario.initial_occupied_slots:
        raise E25V3Error("baseline initial usable-storage occupancy differs from E13")
    if len(env.blocks) != scenario.total_jobs:
        raise E25V3Error("baseline block count differs from E13")
    initial_targets = [
        int(block.stored_time_step + block.storage_steps_needed)
        for block in env.blocks[: scenario.initial_occupied_slots]
    ]
    return {
        "passed": True,
        "episode_instance_id": instance.instance_id,
        "time_step": 0,
        "initially_stored_jobs": scenario.initial_occupied_slots,
        "occupied_storage_slots": occupied,
        "future_arrivals": scenario.total_jobs - scenario.initial_occupied_slots,
        "initial_target_min": min(initial_targets),
        "initial_target_max": max(initial_targets),
        "state_signature_sha256": _state_signature(env.blocks, env.current_state),
    }


class AuditedWarmStartEnv(occupancy.OccupancyTrackingEnv):
    """E13's warm start plus an independently checked, saved reset signature."""

    def __init__(self, scenario):
        super().__init__(scenario)
        self.warm_start_audit = None

    def reset(self, instance=None):
        if instance is None:
            raise E25V3Error("matched warm-start baseline requires a frozen instance")
        state = super().reset(instance=instance)
        self.warm_start_audit = _warm_start_audit(self, self.scenario, instance)
        return state


def _warm_start_factory(scenario, *, arrival_rate, proc_mean, grid_rows, grid_cols,
                        exit_width, number_blocks):
    if (
        grid_rows != scenario.rows
        or grid_cols != scenario.cols
        or exit_width != e25.EXIT_WIDTH
        or number_blocks != scenario.total_jobs
        or arrival_rate != 1.0 / occupancy.EXTRA_ARRIVAL_GAP
        or proc_mean != occupancy.PROC_MEAN
    ):
        raise E25V3Error("baseline environment request differs from E13 scenario")
    return AuditedWarmStartEnv(scenario)


def _ledger_path(output: Path, method: str, scenario_id: str, seed: int, replicate: int) -> Path:
    return output / "run-ledger" / method / scenario_id / f"seed-{seed}-rep-{replicate}.json"


def _authenticate_ledger(path: Path, contract: Mapping, method: str,
                         scenario_id: str, seed: int, replicate: int) -> dict:
    ledger = e25v2._self_hashed(path, "ledger_sha256", "E25-v3 ledger")
    row = ledger.get("row", {})
    if (
        ledger.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("protocol") != PROTOCOL
        or row.get("method") != method
        or row.get("scenario_id") != scenario_id
        or row.get("instance_seed") != seed
        or row.get("replicate") != replicate
        or row.get("initial_state_audit", {}).get("passed") is not True
    ):
        raise E25V3Error(f"E25-v3 ledger coordinate or reset mismatch: {path}")
    return ledger


def run_one(output: Path, method: str, scenario_id: str, seed: int, replicate: int) -> dict:
    if (
        method not in METHODS
        or scenario_id not in SCENARIOS
        or seed not in INSTANCE_SEEDS
        or replicate not in e25.REPLICATES[method]
    ):
        raise E25V3Error("unsupported E25-v3 coordinate")
    contract, (_vcg_contract, _vcg_report, manifest) = authenticate(output)
    path = _ledger_path(output.resolve(), method, scenario_id, seed, replicate)
    if path.is_file():
        return dict(_authenticate_ledger(path, contract, method, scenario_id, seed, replicate)["row"])
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    record = e13._record(manifest, scenario_id, seed)
    instance = e13._load_instance(record)
    identity = e25._replicate_identity(method, scenario_id, seed, replicate)
    args = e25._runtime_args(
        scenario,
        ga_seed_base=(
            int(identity["rng_seed"]) - seed if method == e25.METHOD_GA else e25.GA_RNG_BASE
        ),
    )
    if method == e25.METHOD_DYNAMIC:
        payload = None
        assignment_source = e25.matched.DETERMINISTIC_METHOD_TO_SOURCE[e25.matched.DYNAMIC_METHOD]
        policy_seed = None
    elif method == e25.METHOD_GA:
        implementation = e25.ga_repair_v2._impl
        payload = None
        assignment_source = implementation.REPAIRED_METHOD_TO_SOURCE[implementation.REPAIRED_2009_METHOD]
        policy_seed = None
    else:
        _completions, checkpoints = e25._a3c_inputs()
        payload = checkpoints[int(identity["model_seed"])]
        assignment_source = e25.kim85.TRACK_A_KIM2020_A3C_SPATIAL
        policy_seed = int(identity["rng_seed"])

    created = []

    def factory(**kwargs):
        env = _warm_start_factory(scenario, **kwargs)
        created.append(env)
        return env

    started = perf_counter()
    with patch.object(e25.baseline, "make_shipyard_env", side_effect=factory):
        raw = e25.baseline._evaluate_one(
            args,
            seed,
            payload,
            episode_instance=instance,
            scheduler_variant=e25.baseline.DURATION_AWARE_VARIANT,
            assignment_source=assignment_source,
            source_neutral_scheduler=True,
            assignment_commitment=e25.baseline.DECISION_EPOCH_RESERVED,
            deployment_mode=e25.kim85.DEPLOYMENT_STOCHASTIC,
            policy_seed=policy_seed,
        )
    elapsed = perf_counter() - started
    if len(created) != 1 or created[0].warm_start_audit is None:
        raise E25V3Error("baseline did not use exactly one checked warm-start environment")
    row = e25._normalize(raw, method=method, scenario=scenario, identity=identity, elapsed=elapsed)
    row.update(
        schema_version=SCHEMA_VERSION,
        protocol=PROTOCOL,
        initial_state_audit=created[0].warm_start_audit,
    )
    if (
        row["episode_instance_id"] != record["episode_instance_id"]
        or row["schedule_id"] != record["schedule_id"]
    ):
        raise E25V3Error("baseline execution lost its E13 instance binding")
    ledger = e25v2._hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "row": row,
        },
        "ledger_sha256",
    )
    e25v2._atomic_json(path, ledger)
    return row


def _coordinates(methods: Sequence[str]):
    return (
        (method, scenario_id, seed, replicate)
        for method in methods
        for scenario_id in SCENARIOS
        for seed in INSTANCE_SEEDS
        for replicate in e25.REPLICATES[method]
    )


def run_method(output: Path, method: str) -> dict:
    if method not in METHODS:
        raise E25V3Error("run-method requires a baseline method")
    coordinates = tuple(_coordinates((method,)))
    strict = 0
    for index, (_, scenario_id, seed, replicate) in enumerate(coordinates, start=1):
        row = run_one(output, method, scenario_id, seed, replicate)
        strict += int(row["strict_safe_complete"])
        print(
            f"E25-v3 {e25.METHOD_LABELS[method]} {index}/{len(coordinates)} | "
            f"{scenario_id} seed={seed} rep={replicate} "
            f"strict={int(row['strict_safe_complete'])}",
            flush=True,
        )
    return {"status": "complete", "method": method, "rows": len(coordinates),
            "strict_safe_complete_rows": strict, "training_runs": 0}


def run_pilot(output: Path) -> dict:
    scenarios = ("size_5x5_occ_low", "size_10x10_occ_high")
    rows = []
    for scenario_id in scenarios:
        for method in METHODS:
            row = run_one(output, method, scenario_id, INSTANCE_SEEDS[0], 0)
            rows.append({
                "method": method,
                "scenario_id": scenario_id,
                "strict_safe_complete": row["strict_safe_complete"],
                "initial_state_audit": row["initial_state_audit"],
            })
            print(
                f"E25-v3 pilot {method} {scenario_id} "
                f"strict={int(row['strict_safe_complete'])}", flush=True
            )
    return {"status": "passed", "scope": "integration_only", "paper_evidence": False,
            "rows": rows, "training_runs": 0}


def _baseline_rows(output: Path, contract: Mapping) -> list[dict]:
    rows = []
    for method, scenario_id, seed, replicate in _coordinates(METHODS):
        path = _ledger_path(output, method, scenario_id, seed, replicate)
        if path.is_file():
            rows.append(dict(_authenticate_ledger(
                path, contract, method, scenario_id, seed, replicate
            )["row"]))
    return rows


def _render(summary: Sequence[Mapping], output: Path) -> dict:
    # The visual encoding is intentionally identical to E25-v2, but its figure
    # and provenance paths are versioned and bound to the corrected protocol.
    with patch.multiple(
        e25v2, PROTOCOL=PROTOCOL, SCHEMA_VERSION=SCHEMA_VERSION, FIGURE_NAME=FIGURE_NAME
    ):
        return e25v2._render(summary, output)


def analyze(output: Path, *, partial: bool = False) -> dict:
    output = output.resolve()
    contract, (vcg_contract, vcg_report, _manifest) = authenticate(output)
    baselines = _baseline_rows(output, contract)
    if not partial and len(baselines) != EXPECTED_BASELINE_ROWS:
        raise E25V3Error(f"incomplete matched baseline grid: {len(baselines)}/{EXPECTED_BASELINE_ROWS}")
    vcg = e25v2._new_vcg_rows(VCG_OUTPUT, vcg_contract)
    if len(vcg) != EXPECTED_VCG_ROWS:
        raise E25V3Error("frozen VCG preference grid is incomplete")
    summary = e25v2._summary([*vcg, *baselines])
    expected_summaries = (len(e25v2.LAMBDAS) + len(METHODS)) * len(SCENARIOS)
    if not partial and len(summary) != expected_summaries:
        raise E25V3Error("complete analysis lacks method/scenario summaries")
    figures = _render(summary, output)
    report = e25v2._hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if partial else "complete",
            "paper_evidence": not partial,
            "post_hoc": True,
            "historical_e25_cross_method_comparison_valid": False,
            "corrected_initial_state_parity_rows": len(baselines),
            "expected_initial_state_parity_rows": EXPECTED_BASELINE_ROWS,
            "observed_baseline_rows": len(baselines),
            "reused_vcg_rows": len(vcg),
            "reused_vcg_rows_are_independent_replications": False,
            "vcg_parent_report_sha256": vcg_report["report_sha256"],
            "preference_lambdas": list(e25v2.LAMBDAS),
            "summaries": summary,
            "baseline_failure_counts": {
                method: dict(Counter(
                    row["failure_class"] for row in baselines if row["method"] == method
                )) for method in METHODS
            },
            "quality_scope": "strict-complete runs only",
            "statistical_unit": "EpisodeInstance after within-instance nuisance averaging",
            "figures": figures,
            "runtime": e25v2._runtime(),
            "training_runs": 0,
        },
        "report_sha256",
    )
    e25v2._atomic_json(output / REPORT_NAME, report)
    lines = [
        "# E25-v3 matched warm-start full-family comparison",
        "",
        "| Method | λ | Yard | Occupancy | Initial + arriving | Strict | MAE | Within ±20 | Rehandles/100 | Steps/delivery |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(item: Mapping, name: str, multiplier: float = 1.0) -> str:
        value = item.get(name)
        return "—" if value is None else f"{multiplier * float(value):.2f}"

    for item in summary:
        value = item.get("preference_lambda")
        lines.append(
            f"| {e25v2.METHOD_LABELS[item['method']]} | "
            f"{'—' if value is None else f'{float(value):.4g}'} | "
            f"{item['yard_size']}×{item['yard_size']} | "
            f"{item['occupancy_level']} ({item['initial_occupancy_ratio']:.1%}) | "
            f"{item['initially_stored_jobs']} + {item['arriving_jobs']} | "
            f"{fmt(item, 'strict_completion_rate', 100.0)}% | "
            f"{fmt(item, 'mean_absolute_error')} | "
            f"{fmt(item, 'within_target_window_rate', 100.0)}% | "
            f"{fmt(item, 'physical_rehandles_per_100_required_deliveries')} | "
            f"{fmt(item, 'steps_per_delivery')} |"
        )
    lines.extend([
        "",
        "All methods now start from the same E13 warm-start yard and placement-clock origins. "
        "The baseline rows are new E25-v3 rollouts; VCG rows are reused E25-v2 rollouts, "
        "not independent replications. Quality is conditional on strict completion.",
        "",
    ])
    (output / TABLE_NAME).write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": report["status"],
        "baseline_rows": len(baselines),
        "reused_vcg_rows": len(vcg),
        "report": str(output / REPORT_NAME),
        "table": str(output / TABLE_NAME),
        "figures": figures,
        "training_runs": 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run-pilot", "run-one", "run-method", "run-all", "analyze", "analyze-partial"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int, choices=INSTANCE_SEEDS)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-pilot":
        result = run_pilot(output)
    elif args.command == "run-one":
        if args.method is None or args.scenario is None or args.seed is None or args.replicate is None:
            parser.error("run-one requires --method, --scenario, --seed, and --replicate")
        row = run_one(output, args.method, args.scenario, args.seed, args.replicate)
        result = {"status": "complete", "row": row, "training_runs": 0}
    elif args.command == "run-method":
        if args.method is None:
            parser.error("run-method requires --method")
        result = run_method(output, args.method)
    elif args.command == "run-all":
        results = [run_method(output, method) for method in METHODS]
        result = {"status": "complete", "methods": results, "training_runs": 0}
    else:
        result = analyze(output, partial=args.command == "analyze-partial")
    print(json.dumps(e25v2._json_form(result), indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
