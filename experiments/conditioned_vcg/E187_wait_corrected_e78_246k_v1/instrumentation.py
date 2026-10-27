"""Episode and validation instrumentation for E187."""

from __future__ import annotations

import math
from statistics import fmean

import run_vcg_objective_gamma_audit as audit
from train_viability_graph_smdp import epsilon_at

from .objective import COMPONENT_NAMES, WaitCorrectedE78Env
from .representation import WaitCorrectedQueueAgent


BASE_RUN_EPISODE = audit.run_objective_audit_episode
BASE_SUMMARIZE_RUNS = audit.summarize_objective_runs


def single_objective_rescore(run: dict, spec) -> None:
    """Adapt the legacy dual-stream audit to E187's single exact objective.

    E187 replaces every primitive reward, so decomposing its return into an
    E78 delivery term and a residual would be false.  Both inherited audit
    stream names therefore authenticate the same realized E187 return.
    """

    realized = float(run["return"])
    run.update({
        "selected_objective": spec.objective,
        "selected_objective_return": realized,
        "non_delivery_return": realized,
        "legacy_rescored_return": realized,
        "dense_rescored_return": realized,
        "dual_rescore_uses_fixed_realized_trajectory": False,
        "e187_single_objective_audit": True,
    })


def run_episode(*args, **kwargs) -> dict:
    agent = args[0] if args else kwargs["agent"]
    env = args[1] if len(args) >= 2 else kwargs["env"]
    if not isinstance(agent, WaitCorrectedQueueAgent):
        raise TypeError("E187 episode received another agent architecture")
    if not isinstance(env, WaitCorrectedE78Env):
        raise TypeError("E187 episode received another objective environment")
    agent.bind_context_environment(
        env,
        required_jobs=len(env.blocks),
        max_steps=int(kwargs["max_steps"]),
    )
    decisions_before = int(agent.decision_count)
    run = BASE_RUN_EPISODE(*args, **kwargs)
    decisions_after = int(agent.decision_count)
    metrics = env.e187_metrics()
    if not math.isclose(
        float(run["return"]), -float(metrics["J"]), abs_tol=1.0e-8
    ):
        raise RuntimeError("E187 episode return does not equal negative J")
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
        "e187_metrics": metrics,
        "training_decision_count_before": decisions_before,
        "training_decision_count_after": decisions_after,
        "scheduled_epsilon_after": float(scheduled_epsilon),
    })

    def shown(value) -> str:
        return "n/a" if value is None else f"{float(value):.1f}"

    phase = "train" if training else "validation"
    episode = getattr(env, "current_episode", "?")
    print(
        f"E187 {phase} ep={episode} J={metrics['J']:.1f} "
        f"strict={int(run['strict_method_success'])} steps={run['steps']} "
        f"wait={shown(metrics['mean_admission_wait'])} "
        f"storageMAE={shown(metrics['storage_clock_mae'])} "
        f"releaseMAE={shown(metrics['release_clock_mae'])} "
        f"peak={metrics['peak_stored']}/{metrics['storage_capacity']} "
        f"unknown={run['audit'].get('unknown_candidates_rejected', 0)} "
        f"decisions={decisions_after} eps={scheduled_epsilon:.3f}",
        flush=True,
    )
    return run


def _mean(runs, key: str):
    values = [run["e187_metrics"].get(key) for run in runs]
    values = [float(value) for value in values if value is not None]
    return float(fmean(values)) if values else None


def summarize_runs(runs) -> dict:
    runs = tuple(runs)
    strict_runs = tuple(run for run in runs if run.get("strict_method_success"))
    summary = BASE_SUMMARIZE_RUNS(runs)
    summary.update({
        "mean_e187_J": _mean(runs, "J"),
        "mean_admission_wait": _mean(runs, "mean_admission_wait"),
        "mean_median_admission_wait": _mean(runs, "median_admission_wait"),
        "mean_p90_admission_wait": _mean(runs, "p90_admission_wait"),
        "mean_max_admission_wait": _mean(runs, "max_admission_wait"),
        "mean_total_admission_wait": _mean(runs, "total_admission_wait"),
        "mean_storage_clock_mae": _mean(runs, "storage_clock_mae"),
        "mean_storage_clock_signed_error": _mean(
            runs, "storage_clock_signed_error"
        ),
        "mean_storage_within_20_rate": _mean(
            runs, "storage_within_20_rate"
        ),
        "mean_release_clock_mae": _mean(runs, "release_clock_mae"),
        "mean_release_clock_signed_error": _mean(
            runs, "release_clock_signed_error"
        ),
        "mean_release_within_20_rate": _mean(
            runs, "release_within_20_rate"
        ),
        "mean_peak_stored": _mean(runs, "peak_stored"),
        "max_peak_stored": max(
            (int(run["e187_metrics"]["peak_stored"]) for run in runs),
            default=0,
        ),
        "objective_accounting_exact_for_all_runs": bool(strict_runs) and all(
            bool(run["e187_metrics"]["objective_accounting_exact"])
            for run in strict_runs
        ),
        "reward_sum_exact_for_all_runs": all(
            bool(run["e187_metrics"]["reward_sum_equals_negative_J"])
            for run in runs
        ),
        "total_unknown_candidates_rejected": sum(
            int(run.get("audit", {}).get("unknown_candidates_rejected", 0))
            for run in runs
        ),
        "total_fail_closed_rejections": sum(
            int(run.get("audit", {}).get("fail_closed_rejections", 0))
            for run in runs
        ),
    })
    for name in COMPONENT_NAMES:
        values = [
            float(run["e187_metrics"]["components"][name])
            for run in runs
        ]
        summary[f"mean_component_{name}"] = (
            float(fmean(values)) if values else None
        )
    return summary


__all__ = ["run_episode", "single_objective_rescore", "summarize_runs"]
