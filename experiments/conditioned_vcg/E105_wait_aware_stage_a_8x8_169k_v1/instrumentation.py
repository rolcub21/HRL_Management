"""Non-invasive episode and validation instrumentation for E105."""

from __future__ import annotations

from statistics import fmean

import run_vcg_objective_gamma_audit as audit

from .rewards import episode_service_metrics


BASE_RUN_EPISODE = audit.run_objective_audit_episode
BASE_SUMMARIZE_RUNS = audit.summarize_objective_runs


def run_episode_with_service_metrics(*args, **kwargs) -> dict:
    run = BASE_RUN_EPISODE(*args, **kwargs)
    env = args[1] if len(args) >= 2 else kwargs["env"]
    service = episode_service_metrics(env)
    run["service_metrics"] = service
    run["base_e78_return_without_wait_penalty"] = (
        float(run["return"]) - float(service["outside_wait_penalty_total"])
    )
    phase = "train" if bool(kwargs.get("training")) else "validation"
    episode = getattr(env, "current_episode", "?")
    def shown(value) -> str:
        return "n/a" if value is None else f"{float(value):.1f}"

    print(
        f"E105 {phase} ep={episode} | return={run['return']:.2f} "
        f"strict={int(run['strict_method_success'])} steps={run['steps']} "
        f"wait={shown(service['mean_wait_before_storage'])} "
        f"overcap={service['jobs_exceeding_wait_cap']}/{service['stored_jobs']} "
        f"releaseMAE={shown(service['release_clock_mae'])} "
        f"storageMAE={shown(service['storage_clock_mae'])}",
        flush=True,
    )
    return run


def _mean(runs, key: str):
    values = [run["service_metrics"].get(key) for run in runs]
    values = [float(value) for value in values if value is not None]
    return float(fmean(values)) if values else None


def summarize_runs_with_service_metrics(runs) -> dict:
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
