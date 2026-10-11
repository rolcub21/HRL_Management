#!/usr/bin/env python3
"""Optional end-to-end confirmation of E14's exact reuse stages."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E14_certification_scalability_95k import capture
from experiments.conditioned_vcg.E14_certification_scalability_95k import program
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


ONLINE_STAGES = (
    reuse.STAGE_TIMING_KEY,
    reuse.STAGE_PATH_CLEANUP,
    reuse.STAGE_WITNESS_SUFFIX,
)


def _cache_for(stage: str):
    if stage == reuse.STAGE_TIMING_KEY:
        return reuse.TimingInvariantCertificateCache()
    if stage == reuse.STAGE_PATH_CLEANUP:
        return reuse.TimingInvariantCertificateCache()
    if stage == reuse.STAGE_WITNESS_SUFFIX:
        return reuse.WitnessSuffixCertificateCache()
    raise program.E14Error(f"unsupported online stage: {stage}")


def _frontier_projection(record: Mapping, index: int) -> dict:
    return {
        "frontier_index": index,
        "decision_epoch": int(record["decision_epoch"]),
        "candidate_frontier_digest": record["candidate_frontier_digest"],
        "candidate_keys": list(record["candidate_keys"]),
        "cache_hits": int(record["cache_hits"]),
        "cache_misses": int(record["cache_misses"]),
        "safe_candidate_count": int(record["safe_candidate_count"]),
        "unsafe_candidates_rejected": int(record["unsafe_candidates_rejected"]),
        "unknown_candidates_rejected": int(record["unknown_candidates_rejected"]),
        "exact_search_seconds": float(record["exact_search_seconds"]),
        "total_frontier_seconds": float(record["total_frontier_seconds"]),
    }


def _authenticate_ledger(
    output: Path, contract: Mapping, scenario_id: str, stage: str
) -> dict:
    ledger = program.load_json(
        program.online_ledger_path(output, scenario_id, stage),
        label=f"E14 {scenario_id}/{stage} online ledger",
    )
    if ledger.get("ledger_sha256") != program.digest(
        ledger, hash_field="ledger_sha256"
    ):
        raise program.E14Error("online ledger self-hash mismatch")
    if ledger.get("contract_sha256") != contract["contract_sha256"]:
        raise program.E14Error("online ledger contract mismatch")
    if ledger.get("scenario_id") != scenario_id or ledger.get("stage") != stage:
        raise program.E14Error("online ledger coordinate mismatch")
    return ledger


def _baseline_prefix_comparison(
    trace: Mapping, frontiers: Sequence[Mapping]
) -> dict:
    baseline = trace["completed_frontiers"]
    length = min(len(baseline), len(frontiers))
    mismatches = [
        index for index in range(length)
        if baseline[index]["candidate_frontier_digest"]
        != frontiers[index]["candidate_frontier_digest"]
    ]
    return {
        "common_completed_frontiers": length,
        "frontier_digest_mismatches": mismatches,
        "identical_common_frontier_prefix": not mismatches,
        "baseline_completed_frontiers": len(baseline),
        "online_completed_frontiers": len(frontiers),
    }


def run_one(
    output: Path,
    contract: Mapping,
    manifest: Mapping,
    scenario_id: str,
    stage: str,
) -> dict:
    existing = program.online_ledger_path(output, scenario_id, stage)
    if existing.exists():
        return _authenticate_ledger(output, contract, scenario_id, stage)

    _capture_ledger, trace = capture.authenticate_capture(
        output, contract, scenario_id
    )
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    parent_record = capture._parent_record(manifest, scenario_id)
    instance = occupancy._load_instance(program.PARENT_OUTPUT, parent_record)
    conditioned = occupancy._conditioned_auth()
    arm = conditioned["inputs"]["arms"][program.MODEL_SEED]
    search_config = benchmark._search_config(arm.payload)
    active_cache = _cache_for(stage)
    env_holder = []
    frontiers = []
    original_enumerate = benchmark._enumerate_frontier

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            PROJECT_ROOT,
            {"conditioned": conditioned},
            model_seed=program.MODEL_SEED,
            base=base,
            device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        return conditioned_seed0._FixedLambdaAgent(
            agent, program.PREFERENCE_LAMBDA
        )

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        env_holder.append(env)
        return env

    def cache_factory():
        return active_cache

    def record_enumerate(*args, **kwargs):
        result = original_enumerate(*args, **kwargs)
        frontiers.append(_frontier_projection(result[1], len(frontiers)))
        return result

    raw = None
    error = None
    failure_class = None
    started = perf_counter()
    cleanup = (
        reuse.path_cleanup_active()
        if stage in (reuse.STAGE_PATH_CLEANUP, reuse.STAGE_WITNESS_SUFFIX)
        else nullcontext()
    )
    try:
        with capture.wall_limit(program.CAPTURE_LIMIT_SECONDS), cleanup, ExitStack() as stack:
            stack.enter_context(patch.object(benchmark, "_make_env", env_factory))
            stack.enter_context(pilot._agent_factory(agent_factory))
            stack.enter_context(
                patch.object(benchmark, "ViabilityCertificateCache", cache_factory)
            )
            stack.enter_context(
                patch.object(benchmark, "_enumerate_frontier", record_enumerate)
            )
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=search_config,
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
            )
    except capture.CaptureLimit as caught:
        failure_class = "censored_wall_clock"
        error = str(caught)
    except Exception as caught:  # authenticated diagnostic, not silently dropped
        failure_class = "unsupported_or_implementation_error"
        error = f"{type(caught).__name__}: {caught}"
    elapsed = perf_counter() - started

    strict = False
    behavior = None
    decisions = []
    if raw is not None:
        strict = bool(
            raw["strict_method_success"] and raw["terminal"]
            and raw["method_failure_reason"] is None
            and raw["complete_frontier_exactly_verified"]
            and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
            and len(raw["delivery_deviations"]) == scenario.total_jobs
        )
        failure_class = "completed" if strict else "valid_but_operationally_incomplete"
        behavior = raw["behavior_digest"]
        decisions = [
            {
                "decision_index": int(item["decision_index"]),
                "decision_epoch": int(item["decision_epoch"]),
                "selected_key": item["selected_key"],
                "selected_action_type": item["selected_action_type"],
            }
            for item in raw["decisions"]
        ]

    cache_stats = {"entries": len(active_cache)}
    if isinstance(active_cache, reuse.WitnessSuffixCertificateCache):
        cache_stats.update({
            "outcome_hits": active_cache.outcome_hits,
            "constructive_proof_hits": active_cache.proof_hits,
            "stored_positive_states": len(active_cache.proofs),
            "positive_insertions": active_cache.proofs.insertions,
            "positive_replacements": active_cache.proofs.replacements,
            "unknown_insertions": 0,
        })
    comparison = _baseline_prefix_comparison(trace, frontiers)
    if stage in (reuse.STAGE_TIMING_KEY, reuse.STAGE_PATH_CLEANUP) and not comparison[
        "identical_common_frontier_prefix"
    ]:
        raise program.E14Error(
            f"{stage} changed a baseline frontier and failed exact preservation"
        )
    ledger = program.with_hash({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "artifact": "online_reuse_confirmation",
        "contract_sha256": contract["contract_sha256"],
        "scenario_id": scenario_id,
        "stage": stage,
        "failure_class": failure_class,
        "strict_safe_complete": strict,
        "wall_seconds": elapsed,
        "error": error,
        "infeasibility_claimed": False if raw is None else None,
        "frontiers": frontiers,
        "decisions": decisions,
        "behavior_digest": behavior,
        "steps": None if raw is None else int(raw["steps"]),
        "deliveries_at_stop": (
            None if raw is None else len(raw["delivery_deviations"])
        ),
        "occupancy_at_stop": (
            env_holder[0]._measurement() if env_holder else None
        ),
        "cache": cache_stats,
        "baseline_prefix_comparison": comparison,
        "policy_behavior_scope": (
            "exact_stages_must_match_baseline_prefix;"
            "suffix_stage_may_change_frontier_and_trajectory_via_validated_proofs"
        ),
    }, "ledger_sha256")
    program.atomic_json(existing, ledger)
    return ledger


def analyze(output: Path) -> dict:
    contract, _manifest = program.authenticate(output)
    rows = []
    for scenario_id in program.SCENARIO_IDS:
        for stage in ONLINE_STAGES:
            path = program.online_ledger_path(output, scenario_id, stage)
            if path.exists():
                ledger = _authenticate_ledger(output, contract, scenario_id, stage)
                rows.append({
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "failure_class": ledger["failure_class"],
                    "strict_safe_complete": ledger["strict_safe_complete"],
                    "wall_seconds": ledger["wall_seconds"],
                    "completed_frontiers": len(ledger["frontiers"]),
                    "completed_decisions": len(ledger["decisions"]),
                    "behavior_digest": ledger["behavior_digest"],
                    "cache": ledger["cache"],
                    "baseline_prefix_comparison": ledger[
                        "baseline_prefix_comparison"
                    ],
                })
    by_coordinate = {(row["scenario_id"], row["stage"]): row for row in rows}
    pairwise = []
    for scenario_id in program.SCENARIO_IDS:
        timing = by_coordinate.get((scenario_id, reuse.STAGE_TIMING_KEY))
        path = by_coordinate.get((scenario_id, reuse.STAGE_PATH_CLEANUP))
        if timing is not None and path is not None:
            pairwise.append({
                "scenario_id": scenario_id,
                "comparison": "timing_key_vs_path_cleanup",
                "behavior_digest_equal_when_both_complete": (
                    timing["behavior_digest"] == path["behavior_digest"]
                    if timing["behavior_digest"] is not None
                    and path["behavior_digest"] is not None
                    else None
                ),
            })
    report = program.with_hash({
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "artifact": "online_reuse_confirmation_report",
        "contract_sha256": contract["contract_sha256"],
        "rows": rows,
        "pairwise": pairwise,
        "training": False,
    }, "report_sha256")
    program.atomic_json(output / program.ONLINE_REPORT_NAME, report)
    return report


def _coordinates(command: str) -> tuple[tuple[str, str], ...]:
    scenario = (
        program.SCENARIO_IDS[0] if command.endswith("-medium")
        else program.SCENARIO_IDS[1]
    )
    if command.startswith("confirm-timing-"):
        stages = (reuse.STAGE_TIMING_KEY,)
    elif command.startswith("confirm-path-"):
        stages = (reuse.STAGE_PATH_CLEANUP,)
    elif command.startswith("confirm-suffix-"):
        stages = (reuse.STAGE_WITNESS_SUFFIX,)
    else:
        stages = ONLINE_STAGES
    return tuple((scenario, stage) for stage in stages)


def main(argv: Optional[Sequence[str]] = None) -> None:
    commands = tuple(
        f"confirm-{kind}-{level}"
        for kind in ("timing", "path", "suffix")
        for level in ("medium", "high")
    ) + ("confirm-medium", "confirm-high", "analyze")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    program.prepare(output)
    if args.command == "analyze":
        result = analyze(output)
    else:
        contract, manifest = program.authenticate(output)
        result = {
            "rows": [
                run_one(output, contract, manifest, scenario_id, stage)
                for scenario_id, stage in _coordinates(args.command)
            ]
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
