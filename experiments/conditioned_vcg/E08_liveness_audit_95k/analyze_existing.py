#!/usr/bin/env python3
"""E8 existing-log audit of recovery-witness liveness intervention."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "vcg_conditioned_e8_existing_log_liveness_audit_v1"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e08-liveness-audit-95k"
E13_ROOT = ROOT / "results/vcg-conditioned-e13-scalability-95k-v2"

D12_EPISODES = (
    (
        "size_10x10_occ_medium",
        ROOT
        / "results/vcg-d12-relocation-family-integrated-confirmation-95k/"
        "episodes/size_10x10_occ_medium.json",
    ),
    (
        "size_10x10_occ_high",
        ROOT
        / "results/vcg-d12-relocation-family-integrated-confirmation-95k-v2/"
        "episodes/size_10x10_occ_high.json",
    ),
)

PANEL_COVERAGE = (
    {
        "panel": "E1",
        "evaluations": 1860,
        "decision_liveness": False,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "final episode-level comparison ledgers",
    },
    {
        "panel": "E4",
        "evaluations": 540,
        "decision_liveness": False,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "episode-level ranking-ablation ledgers",
    },
    {
        "panel": "E5(b)",
        "evaluations": 180,
        "decision_liveness": False,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "episode-level conditioning-ablation ledgers",
    },
    {
        "panel": "E5(c)",
        "evaluations": 540,
        "decision_liveness": False,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "episode-level future-consequence ledgers",
    },
    {
        "panel": "E11",
        "evaluations": 2730,
        "decision_liveness": False,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "episode-level distribution-shift ledgers",
    },
    {
        "panel": "E12",
        "evaluations": 7560,
        "decision_liveness": False,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "episode-level representation-ablation ledgers",
    },
    {
        "panel": "E13",
        "evaluations": 45,
        "decision_liveness": True,
        "proposal": False,
        "duration_and_rehandles_by_decision": False,
        "basis": "decision-cost records retain forced flag and executed type",
    },
    {
        "panel": "D12 traces",
        "evaluations": 2,
        "decision_liveness": True,
        "proposal": False,
        "duration_and_rehandles_by_decision": True,
        "basis": "complete raw medium/high confirmation episodes",
    },
)


class E8Error(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise E8Error(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E8Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E8Error(f"{label} must contain an object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise E8Error(f"{label} self-hash mismatch")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _counts_by_type(decisions, *, forced: Optional[bool] = None) -> dict:
    counts = Counter()
    for decision in decisions:
        if forced is not None and bool(decision["liveness_forced"]) != forced:
            continue
        counts[str(decision["selected_action_type"])] += 1
    return dict(sorted(counts.items()))


def _forced_runs(decisions) -> tuple[list[dict], Counter]:
    runs = []
    preceding = Counter()
    index = 0
    while index < len(decisions):
        if not decisions[index]["liveness_forced"]:
            index += 1
            continue
        start = index
        while index < len(decisions) and decisions[index]["liveness_forced"]:
            index += 1
        previous_type = (
            "episode_start"
            if start == 0
            else str(decisions[start - 1]["selected_action_type"])
        )
        preceding[previous_type] += 1
        runs.append(
            {
                "start_decision": start,
                "end_decision_exclusive": index,
                "length": index - start,
                "preceding_executed_action_type": previous_type,
            }
        )
    return runs, preceding


def _e13_rows() -> tuple[list[dict], dict]:
    contract = _self_hashed(
        E13_ROOT / "e13-contract.json", "contract_sha256", "E13 contract"
    )
    manifest = _self_hashed(
        E13_ROOT / "e13-instance-manifest.json",
        "manifest_sha256",
        "E13 manifest",
    )
    if manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise E8Error("E13 manifest is not bound to the E13 contract")
    rows = []
    for record in manifest["records"]:
        path = (
            E13_ROOT
            / "run-ledger"
            / record["scenario_id"]
            / f"seed-{record['seed']}.json"
        )
        ledger = _self_hashed(path, "ledger_sha256", "E13 ledger")
        row = ledger.get("row", {})
        if (
            ledger.get("contract_sha256") != contract["contract_sha256"]
            or row.get("scenario_id") != record["scenario_id"]
            or int(row.get("instance_seed", -1)) != int(record["seed"])
            or not row.get("strict_safe_complete")
        ):
            raise E8Error("E13 ledger binding/completion check failed")
        decisions = row.get("decision_costs")
        if not isinstance(decisions, list) or len(decisions) != int(
            row["macro_decisions"]
        ):
            raise E8Error("E13 decision records are incomplete")
        if any(
            "liveness_forced" not in item or "selected_action_type" not in item
            for item in decisions
        ):
            raise E8Error("E13 decision record lacks E8 fields")
        runs, preceding = _forced_runs(decisions)
        forced_count = sum(bool(item["liveness_forced"]) for item in decisions)
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "instance_seed": int(row["instance_seed"]),
                "rows": int(row["scenario"]["rows"]),
                "cols": int(row["scenario"]["cols"]),
                "occupancy_level": row["scenario"].get("occupancy_level"),
                "initial_storage_occupancy_ratio": row["occupancy"][
                    "initial_storage_occupancy_ratio"
                ],
                "time_weighted_mean_storage_occupancy_ratio": row["occupancy"][
                    "time_weighted_mean_storage_occupancy_ratio"
                ],
                "peak_storage_occupancy_ratio": row["occupancy"][
                    "peak_storage_occupancy_ratio"
                ],
                "macro_decisions": len(decisions),
                "liveness_forced_decisions": forced_count,
                "liveness_forced_rate": forced_count / len(decisions),
                "contiguous_forced_runs": len(runs),
                "forced_run_lengths": [item["length"] for item in runs],
                "preceding_executed_action_types": dict(sorted(preceding.items())),
                "executed_action_types": _counts_by_type(decisions),
                "forced_executed_action_types": _counts_by_type(
                    decisions, forced=True
                ),
                "unforced_executed_action_types": _counts_by_type(
                    decisions, forced=False
                ),
            }
        )
    if len(rows) != 45:
        raise E8Error(f"expected 45 E13 ledgers, found {len(rows)}")
    return rows, {
        "e13_contract_file_sha256": _sha(E13_ROOT / "e13-contract.json"),
        "e13_manifest_file_sha256": _sha(
            E13_ROOT / "e13-instance-manifest.json"
        ),
        "e13_ledger_file_sha256": {
            f"{row['scenario_id']}/{row['instance_seed']}": _sha(
                E13_ROOT
                / "run-ledger"
                / row["scenario_id"]
                / f"seed-{row['instance_seed']}.json"
            )
            for row in rows
        },
    }


def _merge_type_counts(rows, field: str) -> dict:
    counts = Counter()
    for row in rows:
        counts.update(row[field])
    return dict(sorted(counts.items()))


def _aggregate_e13(rows: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    total = sum(int(row["macro_decisions"]) for row in rows)
    forced = sum(int(row["liveness_forced_decisions"]) for row in rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    by_scenario = []
    for scenario_id, group in sorted(grouped.items()):
        decisions = sum(int(row["macro_decisions"]) for row in group)
        interventions = sum(
            int(row["liveness_forced_decisions"]) for row in group
        )
        by_scenario.append(
            {
                "scenario_id": scenario_id,
                "episodes": len(group),
                "rows": group[0]["rows"],
                "cols": group[0]["cols"],
                "occupancy_level": group[0]["occupancy_level"],
                "initial_storage_occupancy_ratio": group[0][
                    "initial_storage_occupancy_ratio"
                ],
                "mean_time_weighted_storage_occupancy_ratio": fmean(
                    float(row["time_weighted_mean_storage_occupancy_ratio"])
                    for row in group
                ),
                "mean_peak_storage_occupancy_ratio": fmean(
                    float(row["peak_storage_occupancy_ratio"]) for row in group
                ),
                "macro_decisions": decisions,
                "liveness_forced_decisions": interventions,
                "liveness_forced_rate": interventions / decisions,
                "contiguous_forced_runs": sum(
                    int(row["contiguous_forced_runs"]) for row in group
                ),
                "executed_action_types": _merge_type_counts(
                    group, "executed_action_types"
                ),
                "forced_executed_action_types": _merge_type_counts(
                    group, "forced_executed_action_types"
                ),
                "unforced_executed_action_types": _merge_type_counts(
                    group, "unforced_executed_action_types"
                ),
            }
        )
    global_summary = {
        "episodes": len(rows),
        "strict_safe_complete_episodes": len(rows),
        "macro_decisions": total,
        "liveness_forced_decisions": forced,
        "liveness_forced_rate": forced / total,
        "contiguous_forced_runs": sum(
            int(row["contiguous_forced_runs"]) for row in rows
        ),
        "executed_action_types": _merge_type_counts(rows, "executed_action_types"),
        "forced_executed_action_types": _merge_type_counts(
            rows, "forced_executed_action_types"
        ),
        "unforced_executed_action_types": _merge_type_counts(
            rows, "unforced_executed_action_types"
        ),
        "proposal_observed_decisions": 0,
        "override_status": "not_identifiable_from_E13_ledgers",
    }
    return global_summary, by_scenario


def _d12_trace(scenario_id: str, path: Path) -> tuple[dict, str]:
    episode = _load(path, f"D12 {scenario_id} episode")
    decisions = episode.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != int(
        episode.get("macro_decisions", -1)
    ):
        raise E8Error(f"D12 decision trace is incomplete: {scenario_id}")
    forced = [item for item in decisions if item["liveness_forced"]]
    unforced = [item for item in decisions if not item["liveness_forced"]]
    runs, preceding = _forced_runs(decisions)

    def outcome(group) -> dict:
        return {
            "decisions": len(group),
            "primitive_steps": sum(int(item["duration"]) for item in group),
            "physical_relocations": sum(
                int(item["physical_storage_relocations"]) for item in group
            ),
            "executed_action_types": _counts_by_type(group),
        }

    return {
        "scenario_id": scenario_id,
        "instance_seed": int(episode["instance_seed"]),
        "strict_safe_complete": bool(
            episode["strict_method_success"]
            and episode["terminal"]
            and episode["method_failure_reason"] is None
        ),
        "all_macro_decisions": len(decisions),
        "liveness_forced_rate": len(forced) / len(decisions),
        "forced": outcome(forced),
        "unforced": outcome(unforced),
        "contiguous_forced_runs": len(runs),
        "forced_run_lengths": [item["length"] for item in runs],
        "preceding_executed_action_types": dict(sorted(preceding.items())),
        "proposal_observed_decisions": 0,
        "activation_reason_available": False,
        "scope": (
            "descriptive trace; overlaps E13 seed 95100 and is not added to "
            "the E13 aggregate denominator"
        ),
    }, _sha(path)


def _coverage_markdown() -> str:
    lines = [
        "# E8 source-field coverage",
        "",
        "| Source | Evaluations/episodes | Forced flag | Learned proposal | Per-decision duration/rehandles |",
        "|---|---:|:---:|:---:|:---:|",
    ]
    for item in PANEL_COVERAGE:
        mark = lambda value: "yes" if value else "no"  # noqa: E731
        lines.append(
            f"| {item['panel']} | {item['evaluations']} | "
            f"{mark(item['decision_liveness'])} | {mark(item['proposal'])} | "
            f"{mark(item['duration_and_rehandles_by_decision'])} |"
        )
    lines.extend(
        [
            "",
            "Missing proposal fields are missing data, not learned/guard agreement.",
            "E13 is the aggregate intervention source. D12 traces only supply the",
            "forced/unforced duration and relocation decomposition.",
        ]
    )
    return "\n".join(lines) + "\n"


def _intervention_markdown(report: Mapping) -> str:
    lines = [
        "# E8 existing-log liveness intervention summary",
        "",
        "| Scenario | Episodes | Decisions | Forced | Forced rate | Forced action types |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in report["e13_by_scenario"]:
        types = ", ".join(
            f"{key}={value}"
            for key, value in item["forced_executed_action_types"].items()
        )
        lines.append(
            f"| {item['scenario_id']} | {item['episodes']} | "
            f"{item['macro_decisions']} | {item['liveness_forced_decisions']} | "
            f"{100.0 * item['liveness_forced_rate']:.1f}% | {types or 'none'} |"
        )
    total = report["e13_global"]
    types = ", ".join(
        f"{key}={value}"
        for key, value in total["forced_executed_action_types"].items()
    )
    lines.extend(
        [
            f"| **All E13** | **{total['episodes']}** | "
            f"**{total['macro_decisions']}** | "
            f"**{total['liveness_forced_decisions']}** | "
            f"**{100.0 * total['liveness_forced_rate']:.1f}%** | {types} |",
            "",
            "These are activation/execution counts, not override counts. E13 did not",
            "store the contemporaneous unrestricted learned proposal.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(output: Path) -> dict:
    rows, e13_sources = _e13_rows()
    e13_global, e13_by_scenario = _aggregate_e13(rows)
    traces = []
    d12_hashes = {}
    for scenario_id, path in D12_EPISODES:
        trace, source_hash = _d12_trace(scenario_id, path)
        traces.append(trace)
        d12_hashes[str(path.relative_to(ROOT))] = source_hash
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete_existing_log_reanalysis",
            "training_runs": 0,
            "new_episode_runs": 0,
            "e13_global": e13_global,
            "e13_by_scenario": e13_by_scenario,
            "e13_episode_rows": rows,
            "d12_detailed_traces": traces,
            "source_field_coverage": list(PANEL_COVERAGE),
            "missing_fields": {
                "e13": [
                    "contemporaneous_unrestricted_learned_proposal",
                    "guard_activation_reason",
                    "forced_vs_unforced_macro_duration_and_relocations",
                    "candidate_score_margins",
                    "handling_predictions_at_override",
                ],
                "d12_traces": [
                    "contemporaneous_unrestricted_learned_proposal",
                    "guard_activation_reason",
                    "candidate_score_margins",
                    "handling_predictions_at_override",
                ],
            },
            "interpretation": {
                "forced_rate": (
                    "fraction of executed macros selected from the retained exact "
                    "recovery witness"
                ),
                "not_causal": (
                    "forced and unforced macros occur in different state populations"
                ),
                "override_not_inferred": (
                    "a forced choice is not counted as an override without the "
                    "unrestricted learned proposal"
                ),
            },
            "source_sha256": {**e13_sources, **d12_hashes},
        },
        "report_sha256",
    )
    _atomic_json(output / "e8-existing-log-report.json", report)
    (output / "e8-source-coverage.md").write_text(
        _coverage_markdown(), encoding="utf-8"
    )
    (output / "e8-intervention-table.md").write_text(
        _intervention_markdown(report), encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = analyze(args.output.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "e13_episodes": report["e13_global"]["episodes"],
                "macro_decisions": report["e13_global"]["macro_decisions"],
                "liveness_forced_decisions": report["e13_global"][
                    "liveness_forced_decisions"
                ],
                "liveness_forced_rate": report["e13_global"][
                    "liveness_forced_rate"
                ],
                "proposal_observed_decisions": report["e13_global"][
                    "proposal_observed_decisions"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
