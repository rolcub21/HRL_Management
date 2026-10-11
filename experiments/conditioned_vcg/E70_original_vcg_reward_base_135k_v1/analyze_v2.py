"""Analysis-only repair for E70's already-normalized base-40 control rows.

This file is deliberately outside E70's frozen training-source manifest. It
does not modify the training contract, checkpoints, or episode execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.conditioned_vcg.E70_original_vcg_reward_base_135k_v1 import protocol as p
from experiments.conditioned_vcg.E70_original_vcg_reward_base_135k_v1 import run as e70


def summarize(record: dict, *, delivery_base: float) -> dict:
    """Accept either a frozen control row or a raw validation record."""
    rows = record["runs"]
    if not rows:
        raise e70.E70Error("E70 validation record has no runs")
    seeds = [int(row["instance_seed"]) for row in rows]
    if seeds != list(p.VALIDATION_SEEDS[p.SCENARIO]):
        raise e70.E70Error("E70 validation instances are not paired")
    strict = all(bool(row["strict_method_success"])
                 and int(row["delivery_count"]) == 36 for row in rows)
    relocations = [int(row["relocations"] if "relocations" in row
                       else row["audit"]["relocations"]) for row in rows]
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "strict_complete": sum(bool(row["strict_method_success"]) for row in rows),
        "episodes": len(rows),
        "base40_equivalent_return": (
            sum(float(row["dense_rescored_return"]) - 36 * (delivery_base - 40.0)
                for row in rows) / len(rows) if strict else None
        ),
        "placement_mae": (
            sum(float(row["mean_absolute_error"]) for row in rows) / len(rows)
            if strict else None
        ),
        "within_20_rate": (
            sum(float(row["within_target_window_rate"]) for row in rows)
            / len(rows) if strict else None
        ),
        "steps": sum(int(row["steps"]) for row in rows) / len(rows),
        "rehandles_per_100": (
            100 * sum(relocations) / sum(int(row["delivery_count"]) for row in rows)
            if strict else None
        ),
        "instances": [row["episode_instance_id"] for row in rows],
    }


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = e70.authenticate(output)
    summary = e70._read(output / "training/operational" / p.SCENARIO
                        / "seed-0/training-summary.json")
    if (summary.get("model_seed") != 0
            or summary.get("scenario") != p.SCENARIO
            or summary.get("training_protocol") != p.PROTOCOL
            or summary.get("objective_spec") != p.SPEC.to_dict()
            or summary.get("target_contract_sha256") != contract["contract_sha256"]):
        raise e70.E70Error("E70 training summary identity differs")
    baseline = {int(row["checkpoint_episode"]): row
                for row in contract["base40_control"]["rows"]}
    matched = []
    for record in summary["validation_history"]:
        episode = int(record["checkpoint_episode"])
        if episode not in baseline:
            continue
        control = summarize(baseline[episode], delivery_base=40.0)
        pilot = summarize(record, delivery_base=80.0)
        if control["instances"] != pilot["instances"]:
            raise e70.E70Error(f"E70 episode {episode} workload mismatch")
        comparable = control["strict_complete"] == pilot["strict_complete"] == 5
        matched.append({
            "checkpoint_episode": episode,
            "control_base40": control,
            "pilot_base80": pilot,
            "base80_minus_base40": (
                {key: pilot[key] - control[key] for key in (
                    "base40_equivalent_return", "placement_mae", "within_20_rate",
                    "steps", "rehandles_per_100")}
                if comparable else None
            ),
        })
    if not matched:
        raise e70.E70Error("E70 has no matched validation checkpoint")
    report = {
        "status": "complete" if summary["completed_training_episodes"] >= 30 else "partial",
        "analysis_version": "e70_matched_control_analysis_v2",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "scope": "development-only matched reward-base test, original VCG architecture",
        "completed_training_episodes": summary["completed_training_episodes"],
        "matched_checkpoints": matched,
        "comparison_basis": "base-80 undiscounted return minus 40 per completed delivery; strict-complete rows only",
        "caveat": "single training seed and reused development instances; not a sealed test or paper result",
        "training_runs": 0,
        "episode_runs": 0,
        "paper_evidence": False,
    }
    target = output / "e70-report-v2.json"
    e70.v2run.e13._atomic_json(target, report)
    return {"status": report["status"], "report": str(target),
            "matched_checkpoints": [row["checkpoint_episode"] for row in matched],
            "training_runs": 0, "episode_runs": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
