"""Per-episode guard-free liveness telemetry for E226."""

from __future__ import annotations

from statistics import fmean

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    instrumentation as e191,
)


single_objective_rescore = e191.single_objective_rescore


def _episode_telemetry(records: tuple[dict, ...], run: dict) -> dict:
    streak = 0
    longest = 0
    reconfigure_only = 0
    for record in records:
        action = str(record["selected_action_type"])
        if action == "reconfigure":
            streak += 1
            longest = max(longest, streak)
        elif action in ("accept", "deliver"):
            streak = 0
        # Defer intentionally does not reset the admission/delivery progress run.
        reconfigure_only += int(
            int(record["safe_reconfigure_candidates"]) > 0
            and int(record["safe_nonreconfigure_candidates"]) == 0
        )
    frontier_count = len(records)
    return {
        "contract": "e226_guard_free_episode_telemetry_v1",
        "longest_reconfigure_run_without_accept_or_deliver": int(longest),
        "reconfigure_only_safe_frontiers": int(reconfigure_only),
        "safe_frontiers": int(frontier_count),
        "reconfigure_only_safe_frontier_rate": (
            0.0 if frontier_count == 0 else reconfigure_only / frontier_count
        ),
        "step_limit_hit": bool(run.get("horizon_cap_hit")),
        "forced_decisions": int(
            run.get("audit", {}).get("liveness_forced_decisions", 0)
        ),
        "masked_frontiers": sum(
            bool(record.get("liveness_restricted")) for record in records
        ),
        "critic_decisions": int(frontier_count),
    }


def run_episode(*args, **kwargs) -> dict:
    agent = args[0] if args else kwargs["agent"]
    before = len(agent.guard_free_decision_records)
    run = e191.run_episode(*args, **kwargs)
    records = tuple(agent.guard_free_decision_records[before:])
    if len(records) != int(run["macro_decisions"]):
        raise RuntimeError("E226 decision telemetry does not match macro count")
    if any(
        bool(record["liveness_forced"])
        or bool(record["liveness_restricted"])
        for record in records
    ):
        raise RuntimeError("E226 observed a guard intervention")
    telemetry = _episode_telemetry(records, run)
    if telemetry["forced_decisions"] or telemetry["masked_frontiers"]:
        raise RuntimeError("E226 guard-free contract was violated")
    run["e226_guard_free"] = telemetry
    return run


def summarize_runs(runs) -> dict:
    runs = tuple(runs)
    summary = e191.summarize_runs(runs)
    if not runs:
        return summary
    telemetry = tuple(run["e226_guard_free"] for run in runs)
    summary["e226_guard_free"] = {
        "cases": len(runs),
        "maximum_longest_reconfigure_run_without_accept_or_deliver": max(
            int(row["longest_reconfigure_run_without_accept_or_deliver"])
            for row in telemetry
        ),
        "mean_longest_reconfigure_run_without_accept_or_deliver": fmean(
            int(row["longest_reconfigure_run_without_accept_or_deliver"])
            for row in telemetry
        ),
        "reconfigure_only_safe_frontiers": sum(
            int(row["reconfigure_only_safe_frontiers"]) for row in telemetry
        ),
        "safe_frontiers": sum(int(row["safe_frontiers"]) for row in telemetry),
        "step_limit_hits": sum(bool(row["step_limit_hit"]) for row in telemetry),
        "forced_decisions": sum(int(row["forced_decisions"]) for row in telemetry),
        "masked_frontiers": sum(int(row["masked_frontiers"]) for row in telemetry),
    }
    frontiers = summary["e226_guard_free"]["safe_frontiers"]
    summary["e226_guard_free"]["reconfigure_only_safe_frontier_rate"] = (
        0.0 if frontiers == 0 else
        summary["e226_guard_free"]["reconfigure_only_safe_frontiers"] / frontiers
    )
    return summary


__all__ = ["run_episode", "single_objective_rescore", "summarize_runs"]

