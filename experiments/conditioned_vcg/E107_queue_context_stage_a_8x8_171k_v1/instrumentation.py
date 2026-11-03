"""Service, exploration, and queue-context instrumentation for E107."""

from __future__ import annotations

from statistics import fmean

import run_vcg_objective_gamma_audit as audit
from train_viability_graph_smdp import epsilon_at

from experiments.conditioned_vcg.E105_wait_aware_stage_a_8x8_169k_v1.rewards import (
    episode_service_metrics,
)
from .representation import QueueContextAgent


BASE_RUN_EPISODE = audit.run_objective_audit_episode
BASE_SUMMARIZE_RUNS = audit.summarize_objective_runs


def run_episode(*args, **kwargs) -> dict:
    agent = args[0] if args else kwargs["agent"]
    env = args[1] if len(args) >= 2 else kwargs["env"]
    if not isinstance(agent, QueueContextAgent):
        raise TypeError("E107 episode received another agent architecture")
    agent.bind_context_environment(
        env,
        wait_cap=80,
        required_jobs=36,
        max_steps=int(kwargs["max_steps"]),
    )
    decisions_before = int(agent.decision_count)
    run = BASE_RUN_EPISODE(*args, **kwargs)
    decisions_after = int(agent.decision_count)
    service = episode_service_metrics(env)
    training = bool(kwargs.get("training"))
    scheduled_epsilon = (
        epsilon_at(
            decisions_after,
            start=float(kwargs["epsilon_start"]),
            end=float(kwargs["epsilon_end"]),
            decay_decisions=int(kwargs["epsilon_decay_decisions"]),
            warmup_decisions=int(kwargs.get("epsilon_warmup_decisions", 0)),
        )
        if training
        else 0.0
    )
    run.update({
        "service_metrics": service,
        "base_e78_return_without_wait_penalty": (
            float(run["return"]) - float(service["outside_wait_penalty_total"])
        ),
        "training_decision_count_before": decisions_before,
        "training_decision_count_after": decisions_after,
        "scheduled_epsilon_after": float(scheduled_epsilon),
    })

    def shown(value) -> str:
        return "n/a" if value is None else f"{float(value):.1f}"

    phase = "train" if training else "validation"
    episode = getattr(env, "current_episode", "?")
    print(
        f"E107 {phase} ep={episode} | return={run['return']:.2f} "
        f"strict={int(run['strict_method_success'])} steps={run['steps']} "
        f"wait={shown(service['mean_wait_before_storage'])} "
        f"overcap={service['jobs_exceeding_wait_cap']}/{service['stored_jobs']} "
        f"releaseMAE={shown(service['release_clock_mae'])} "
        f"storageMAE={shown(service['storage_clock_mae'])} "
        f"decisions={decisions_after} eps={scheduled_epsilon:.3f}",
        flush=True,
    )
    return run


def _mean(runs, key: str):
    values = [run["service_metrics"].get(key) for run in runs]
    values = [float(value) for value in values if value is not None]
    return float(fmean(values)) if values else None


def summarize_runs(runs) -> dict:
    runs = tuple(runs)
    summary = BASE_SUMMARIZE_RUNS(runs)
    summary.update({
        "mean_wait_before_storage": _mean(runs, "mean_wait_before_storage"),
        "mean_max_wait_before_storage": _mean(runs, "max_wait_before_storage"),
        "mean_jobs_exceeding_wait_cap": _mean(runs, "jobs_exceeding_wait_cap"),
        "mean_wait_cap_compliance_rate": _mean(runs, "wait_cap_compliance_rate"),
        "mean_wait_cap_excess_job_steps": _mean(runs, "wait_cap_excess_job_steps"),
        "mean_release_clock_mae": _mean(runs, "release_clock_mae"),
        "mean_release_clock_signed_error": _mean(
            runs, "release_clock_mean_signed_error"
        ),
        "mean_storage_clock_mae": _mean(runs, "storage_clock_mae"),
        "mean_storage_clock_signed_error": _mean(
            runs, "storage_clock_mean_signed_error"
        ),
        "mean_outside_wait_penalty": _mean(runs, "outside_wait_penalty_total"),
        "wait_accounting_exact_for_all_complete_runs": all(
            run["service_metrics"]["completed_wait_excess_matches_accrual"]
            for run in runs
            if run.get("success")
        ),
    })
    return summary
