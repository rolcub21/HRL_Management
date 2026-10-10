#!/usr/bin/env python3
"""E25-v4: matched scale baselines with a legal standalone reconfiguration macro."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import torch

from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E25_scale_baseline_transfer_95k import run as e25
from experiments.conditioned_vcg.E25_v2_full_vcg_family_scale_95k import run as e25v2
from experiments.conditioned_vcg.E25_v3_matched_warm_start_95k import run as v3
from experiments.conditioned_vcg.E25_v4_reconfiguration_baselines_95k import controller
from experiments.conditioned_vcg.development.D10_scalability_support_screen import occupancy_extension as occupancy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e25-v4-reconfiguration-baselines-95k"
PROTOCOL = "vcg_conditioned_e25_v4_reconfiguration_baselines_95k_v1"
SCHEMA_VERSION = 1
CONTRACT_NAME = "e25-v4-contract.json"
REPORT_NAME = "e25-v4-report.json"
TABLE_NAME = "e25-v4-full-family-table.md"
FIGURE_NAME = "e25-v4-full-family-scale-comparison"
SCENARIOS = e25v2.SCENARIOS
SEEDS = e25v2.INSTANCE_SEEDS
METHODS = e25v2.BASELINE_METHODS
EXPECTED_BASELINE_ROWS = e25.EXPECTED_NEW_ROWS
EXPECTED_VCG_ROWS = e25v2.EXPECTED_NEW_VCG_ROWS


class E25V4Error(RuntimeError):
    pass


def _contract_payload(parents):
    vcg_contract, vcg_report, manifest = parents
    sources = (
        Path(__file__).resolve(), Path(controller.__file__).resolve(),
        Path(v3.__file__).resolve(), Path(e25.__file__).resolve(),
        PROJECT_ROOT / "track_b_urgency_evaluate.py",
        PROJECT_ROOT / "example/urgency_scheduler.py",
        PROJECT_ROOT / "example/Options/ReconfigureOption.py",
        PROJECT_ROOT / "PSLAP/viability.py",
        PROJECT_ROOT / "PSLAP/viability_filter.py",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "post_hoc": True,
        "question": "do placement and dispatch baselines remain competitive when a physically legal standalone reconfiguration is available",
        "historical_e25_v3_rows_reused": False,
        "vcg_rows_reused": EXPECTED_VCG_ROWS,
        "expected_new_baseline_rows": EXPECTED_BASELINE_ROWS,
        "scenarios": list(SCENARIOS),
        "seeds": list(SEEDS),
        "replicates": {method: list(e25.REPLICATES[method]) for method in METHODS},
        "same_assignment_sources_as_e25_v3": True,
        "same_warm_start_as_e25_v3": True,
        "additional_macro": controller.ReconfigurationScheduler.POLICY,
        "trigger": "neither inbound admission nor strict retrieval executable",
        "resolution": "first relocation of least-slack baseline retrieval plan if legal; otherwise shortest physically legal relocation; no repeated physical state",
        "certification": "none for these baseline controllers; physical action legality only",
        "quality_scope": "strict-complete runs only; completion separately",
        "training_runs": 0,
        "parent": {
            "vcg_contract_sha256": vcg_contract["contract_sha256"],
            "vcg_report_sha256": vcg_report["report_sha256"],
            "e13_manifest_sha256": manifest["manifest_sha256"],
        },
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): e25v2._sha(path) for path in sources},
    }


def prepare(output: Path):
    parents = v3._parents()
    expected = e25v2._hashed(_contract_payload(parents), "contract_sha256")
    path = output.resolve() / CONTRACT_NAME
    if path.is_file():
        if e25v2._self_hashed(path, "contract_sha256", "E25-v4 contract") != expected:
            raise E25V4Error("existing contract differs from frozen sources or parent")
    else:
        e25v2._atomic_json(path, expected)
    return {"status": "prepared", "new_baseline_rows": EXPECTED_BASELINE_ROWS,
            "reused_vcg_rows": EXPECTED_VCG_ROWS, "training_runs": 0,
            "output": str(output.resolve())}


def authenticate(output: Path):
    parents = v3._parents()
    expected = e25v2._hashed(_contract_payload(parents), "contract_sha256")
    observed = e25v2._self_hashed(output.resolve() / CONTRACT_NAME,
                                  "contract_sha256", "E25-v4 contract")
    if observed != expected:
        raise E25V4Error("E25-v4 contract, parent, or sources changed")
    return observed, parents


def _ledger_path(output: Path, method: str, scenario_id: str, seed: int, replicate: int):
    return output / "run-ledger" / method / scenario_id / f"seed-{seed}-rep-{replicate}.json"


def _read_ledger(path: Path, contract: Mapping, method: str, scenario_id: str,
                 seed: int, replicate: int):
    ledger = e25v2._self_hashed(path, "ledger_sha256", "E25-v4 ledger")
    row = ledger.get("row", {})
    if (ledger.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("protocol") != PROTOCOL
        or (row.get("method"), row.get("scenario_id"), row.get("instance_seed"), row.get("replicate"))
        != (method, scenario_id, seed, replicate)
        or row.get("initial_state_audit", {}).get("passed") is not True):
        raise E25V4Error(f"ledger coordinate or reset mismatch: {path}")
    return ledger


def _coordinate(method: str, scenario_id: str, seed: int, replicate: int):
    if (method not in METHODS or scenario_id not in SCENARIOS
        or seed not in SEEDS or replicate not in e25.REPLICATES[method]):
        raise E25V4Error("unsupported E25-v4 coordinate")


def run_one(output: Path, method: str, scenario_id: str, seed: int, replicate: int):
    _coordinate(method, scenario_id, seed, replicate)
    output = output.resolve()
    contract, (_vcg_contract, _vcg_report, manifest) = authenticate(output)
    path = _ledger_path(output, method, scenario_id, seed, replicate)
    if path.is_file():
        return dict(_read_ledger(path, contract, method, scenario_id, seed, replicate)["row"])
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    record = e13._record(manifest, scenario_id, seed)
    instance = e13._load_instance(record)
    identity = e25._replicate_identity(method, scenario_id, seed, replicate)
    args = e25._runtime_args(
        scenario,
        ga_seed_base=(int(identity["rng_seed"]) - seed if method == e25.METHOD_GA else e25.GA_RNG_BASE),
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
    started = perf_counter()
    raw = controller.evaluate_one(
        args, seed, payload, instance=instance, scenario=scenario,
        assignment_source=assignment_source, policy_seed=policy_seed,
        deployment_mode=e25.kim85.DEPLOYMENT_STOCHASTIC,
    )
    elapsed = perf_counter() - started
    row = e25._normalize(raw, method=method, scenario=scenario,
                         identity=identity, elapsed=elapsed)
    row.update({
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "initial_state_audit": raw["initial_state_audit"],
        "standalone_reconfiguration_attempts": raw["urgency_scheduler_audit"]["standalone_reconfiguration_attempts"],
        "standalone_reconfiguration_successes": raw["urgency_scheduler_audit"]["standalone_reconfiguration_successes"],
        "standalone_reconfiguration_failures": raw["urgency_scheduler_audit"]["standalone_reconfiguration_failures"],
        "standalone_reconfiguration_events": raw["standalone_reconfigurations"],
        "target_bound_relocation_events": raw["target_bound_obstruction_clearances"],
        "scheduler_decisions": raw["urgency_scheduler_audit"]["decisions"],
    })
    if row["episode_instance_id"] != record["episode_instance_id"] or row["schedule_id"] != record["schedule_id"]:
        raise E25V4Error("execution lost E13 instance binding")
    ledger = e25v2._hashed({
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"], "row": row,
    }, "ledger_sha256")
    e25v2._atomic_json(path, ledger)
    return row


def _coordinates(methods: Sequence[str]):
    return ((method, scenario_id, seed, replicate)
            for method in methods for scenario_id in SCENARIOS
            for seed in SEEDS for replicate in e25.REPLICATES[method])


def run_method(output: Path, method: str):
    if method not in METHODS:
        raise E25V4Error("unsupported method")
    coordinates = tuple(_coordinates((method,)))
    strict = 0
    for index, (_, scenario_id, seed, replicate) in enumerate(coordinates, 1):
        row = run_one(output, method, scenario_id, seed, replicate)
        strict += int(row["strict_safe_complete"])
        print(f"E25-v4 {e25.METHOD_LABELS[method]} {index}/{len(coordinates)} | "
              f"{scenario_id} seed={seed} rep={replicate} "
              f"strict={int(row['strict_safe_complete'])} "
              f"standalone={row['standalone_reconfiguration_events']}", flush=True)
    return {"status": "complete", "method": method, "rows": len(coordinates),
            "strict_safe_complete_rows": strict, "training_runs": 0}


def run_pilot(output: Path):
    rows = []
    for scenario_id in ("size_5x5_occ_low", "size_5x5_occ_medium", "size_8x8_occ_high"):
        row = run_one(output, METHODS[0], scenario_id, SEEDS[0], 0)
        rows.append({"scenario_id": scenario_id, "strict_safe_complete": row["strict_safe_complete"],
                     "standalone_reconfiguration_events": row["standalone_reconfiguration_events"],
                     "failure_class": row["failure_class"]})
        print(f"E25-v4 pilot {scenario_id} | strict={int(row['strict_safe_complete'])} "
              f"standalone={row['standalone_reconfiguration_events']}", flush=True)
    return {"status": "complete", "scope": "integration_only_no_quality_selection",
            "paper_evidence": False, "rows": rows, "training_runs": 0}


def analyze(output: Path, *, partial: bool = False):
    output = output.resolve()
    contract, (vcg_contract, vcg_report, _manifest) = authenticate(output)
    baselines = []
    for method, scenario_id, seed, replicate in _coordinates(METHODS):
        path = _ledger_path(output, method, scenario_id, seed, replicate)
        if path.is_file():
            baselines.append(dict(_read_ledger(path, contract, method, scenario_id, seed, replicate)["row"]))
    if not partial and len(baselines) != EXPECTED_BASELINE_ROWS:
        raise E25V4Error(f"incomplete baseline grid: {len(baselines)}/{EXPECTED_BASELINE_ROWS}")
    vcg = e25v2._new_vcg_rows(e25v2.DEFAULT_OUTPUT, vcg_contract)
    if len(vcg) != EXPECTED_VCG_ROWS:
        raise E25V4Error("frozen VCG family is incomplete")
    summary = e25v2._summary([*vcg, *baselines])
    from unittest.mock import patch
    with patch.multiple(e25v2, PROTOCOL=PROTOCOL, SCHEMA_VERSION=SCHEMA_VERSION, FIGURE_NAME=FIGURE_NAME):
        figures = e25v2._render(summary, output)
    report = e25v2._hashed({
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "partial" if partial else "complete", "paper_evidence": not partial,
        "post_hoc": True, "observed_baseline_rows": len(baselines),
        "expected_baseline_rows": EXPECTED_BASELINE_ROWS,
        "reused_vcg_rows": len(vcg), "reused_controls_are_independent_replications": False,
        "vcg_parent_report_sha256": vcg_report["report_sha256"],
        "baseline_failure_counts": {method: dict(Counter(
            row["failure_class"] for row in baselines if row["method"] == method
        )) for method in METHODS},
        "standalone_reconfiguration_events": {method: sum(
            row["standalone_reconfiguration_events"] for row in baselines if row["method"] == method
        ) for method in METHODS},
        "quality_scope": "strict-complete runs only",
        "statistical_unit": "EpisodeInstance after within-instance nuisance averaging",
        "summaries": summary, "figures": figures,
        "runtime": e25v2._runtime(), "training_runs": 0,
    }, "report_sha256")
    e25v2._atomic_json(output / REPORT_NAME, report)
    lines = [
        "# E25-v4 matched scale comparison with standalone reconfiguration",
        "",
        "| Method | λ | Yard | Occupancy | Initial + arriving | Strict | Placement MAE | Within ±20 | Rehandles/100 | Steps/delivery |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(item, key, multiplier=1.0):
        value = item.get(key)
        return "—" if value is None else f"{float(value)*multiplier:.2f}"
    for item in summary:
        value = item.get("preference_lambda")
        lines.append(
            f"| {e25v2.METHOD_LABELS[item['method']]} | "
            f"{'—' if value is None else f'{float(value):.4g}'} | "
            f"{item['yard_size']}×{item['yard_size']} | "
            f"{item['occupancy_level']} ({item['initial_occupancy_ratio']:.1%}) | "
            f"{item['initially_stored_jobs']} + {item['arriving_jobs']} | "
            f"{fmt(item, 'strict_completion_rate', 100)}% | "
            f"{fmt(item, 'mean_absolute_error')} | "
            f"{fmt(item, 'within_target_window_rate', 100)}% | "
            f"{fmt(item, 'physical_rehandles_per_100_required_deliveries')} | "
            f"{fmt(item, 'steps_per_delivery')} |"
        )
    lines.extend(["", "Baseline assignments and warm starts match E25-v3; all new baseline runs have "
                  "a legal, non-certified standalone Reconfigure option. VCG rows are reused, not independent replications. "
                  "Quality is conditional on strict completion.", ""])
    (output / TABLE_NAME).write_text("\n".join(lines), encoding="utf-8")
    return {"status": report["status"], "baseline_rows": len(baselines),
            "reused_vcg_rows": len(vcg), "report": str(output / REPORT_NAME),
            "table": str(output / TABLE_NAME), "figures": figures, "training_runs": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-pilot", "run-one", "run-method", "run-all", "analyze", "analyze-partial"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
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
        if None in (args.method, args.scenario, args.seed, args.replicate):
            parser.error("run-one requires --method, --scenario, --seed, and --replicate")
        result = {"status": "complete", "row": run_one(output, args.method, args.scenario, args.seed, args.replicate)}
    elif args.command == "run-method":
        if args.method is None:
            parser.error("run-method requires --method")
        result = run_method(output, args.method)
    elif args.command == "run-all":
        result = {"status": "complete", "methods": [run_method(output, method) for method in METHODS]}
    else:
        result = analyze(output, partial=args.command == "analyze-partial")
    print(json.dumps(e25v2._json_form(result), indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
