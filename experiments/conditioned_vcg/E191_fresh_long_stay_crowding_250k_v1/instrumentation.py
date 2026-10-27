"""Workload and occupancy telemetry layered on E188 instrumentation."""

from __future__ import annotations

import builtins
from statistics import fmean
from unittest.mock import patch

from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    instrumentation as e188,
)


single_objective_rescore = e188.single_objective_rescore


def run_episode(*args, **kwargs) -> dict:
    env = args[1] if len(args) >= 2 else kwargs["env"]
    inherited_print = builtins.print

    def e191_print(*values, **keywords):
        if values and isinstance(values[0], str):
            first = values[0]
            if first.startswith("E188 "):
                values = ("E191 " + first[len("E188 "):], *values[1:])
        return inherited_print(*values, **keywords)

    with patch.object(builtins, "print", e191_print):
        run = e188.run_episode(*args, **kwargs)
    metrics = run["e187_metrics"]
    workload = dict(metrics["e191_workload"])
    run["e191_workload"] = workload
    run["e191_occupancy"] = {
        name: metrics[name]
        for name in (
            "mean_stored", "mean_stored_fraction",
            "arrival_window_mean_stored",
            "nominal_steady_window_mean_stored", "peak_stored",
            "peak_stored_fraction", "mean_external_queue",
            "peak_external_queue", "transporter_utilization",
        )
    }
    inherited_print(
        f"E191 workload ep={getattr(env, 'current_episode', '?')} "
        f"target={workload['target_label']} gap={workload['arrival_gap_steps']} "
        f"meanStored={metrics['mean_stored']:.2f} "
        f"arrivalMean={metrics['arrival_window_mean_stored']:.2f} "
        f"queueMean={metrics['mean_external_queue']:.2f} "
        f"util={metrics['transporter_utilization']:.3f}",
        flush=True,
    )
    return run


def _mean(runs, path: tuple[str, ...]):
    values = []
    for run in runs:
        value = run
        for name in path:
            value = value[name]
        if value is not None:
            values.append(float(value))
    return float(fmean(values)) if values else None


def _group_summary(runs) -> dict:
    runs = tuple(runs)
    metrics = (
        "J", "mean_admission_wait", "storage_clock_mae",
        "storage_within_20_rate", "release_clock_mae",
        "release_within_20_rate", "peak_stored", "mean_stored",
        "mean_stored_fraction", "arrival_window_mean_stored",
        "nominal_steady_window_mean_stored", "mean_external_queue",
        "peak_external_queue", "transporter_utilization",
    )
    return {
        "cases": len(runs),
        "strict_completion_rate": (
            sum(bool(run.get("strict_method_success")) for run in runs)
            / len(runs)
        ),
        "means": {
            name: _mean(runs, ("e187_metrics", name)) for name in metrics
        },
        "max_peak_stored": max(
            int(run["e187_metrics"]["peak_stored"]) for run in runs
        ),
        "horizon_cap_hits": sum(bool(run.get("horizon_cap_hit")) for run in runs),
        "unknown_candidates_rejected": sum(
            int(run.get("audit", {}).get("unknown_candidates_rejected", 0))
            for run in runs
        ),
    }


def summarize_runs(runs) -> dict:
    runs = tuple(runs)
    summary = e188.summarize_runs(runs)
    groups = {}
    for run in runs:
        label = str(run["e191_workload"]["target_label"])
        groups.setdefault(label, []).append(run)
    summary.update({
        "e191_by_workload": {
            label: _group_summary(group)
            for label, group in sorted(groups.items())
        },
        "mean_e191_stored": _mean(runs, ("e187_metrics", "mean_stored")),
        "mean_e191_stored_fraction": _mean(
            runs, ("e187_metrics", "mean_stored_fraction")
        ),
        "mean_e191_external_queue": _mean(
            runs, ("e187_metrics", "mean_external_queue")
        ),
        "mean_e191_transporter_utilization": _mean(
            runs, ("e187_metrics", "transporter_utilization")
        ),
    })
    return summary


__all__ = ["run_episode", "single_objective_rescore", "summarize_runs"]
