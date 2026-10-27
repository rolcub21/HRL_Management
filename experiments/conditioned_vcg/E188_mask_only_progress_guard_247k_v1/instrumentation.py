"""Guard-specific telemetry layered on E187's exact objective audit."""

from __future__ import annotations

import builtins
from statistics import fmean
from unittest.mock import patch

from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    instrumentation as e187,
)


def _guard_audit(agent) -> dict:
    guard = getattr(agent, "recovery_witness_guard", None)
    if guard is None:
        return {}
    for name in ("audit", "audit_dict"):
        function = getattr(guard, name, None)
        if callable(function):
            value = function()
            if isinstance(value, dict):
                return dict(value)
    return {}


def _delta(before: dict, after: dict) -> dict:
    result = {}
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if isinstance(new, (int, float)):
            baseline = old if isinstance(old, (int, float)) else 0
            result[key] = new - baseline
        elif isinstance(new, dict):
            baseline = old if isinstance(old, dict) else {}
            keys = set(baseline) | set(new)
            if all(isinstance(baseline.get(item, 0), (int, float))
                   and isinstance(new.get(item, 0), (int, float))
                   for item in keys):
                result[key] = {
                    item: new.get(item, 0) - baseline.get(item, 0)
                    for item in sorted(keys)
                }
    return result


def run_episode(*args, **kwargs) -> dict:
    agent = args[0] if args else kwargs["agent"]
    before = _guard_audit(agent)
    inherited_print = builtins.print

    def e188_print(*values, **keywords):
        if values and isinstance(values[0], str) and values[0].startswith("E187 "):
            values = ("E188 " + values[0][len("E187 "):], *values[1:])
        return inherited_print(*values, **keywords)

    with patch.object(builtins, "print", e188_print):
        run = e187.run_episode(*args, **kwargs)
    after = _guard_audit(agent)
    guard = _delta(before, after)
    forced_deliveries = int(
        run.get("audit", {}).get("liveness_forced_delivery_decisions", 0)
    )
    deliveries = int(run.get("delivery_count", 0))
    guard.update({
        "forced_deliveries": forced_deliveries,
        "voluntary_deliveries": max(deliveries - forced_deliveries, 0),
        "forced_delivery_share": (
            None if deliveries == 0 else forced_deliveries / deliveries
        ),
    })
    run["e188_guard"] = guard
    return run


def _sum_numeric(runs, key: str) -> int | float:
    return sum(
        run.get("e188_guard", {}).get(key, 0)
        for run in runs
        if isinstance(run.get("e188_guard", {}).get(key, 0), (int, float))
    )


def summarize_runs(runs) -> dict:
    runs = tuple(runs)
    summary = e187.summarize_runs(runs)
    forced = sum(
        int(run.get("e188_guard", {}).get("forced_deliveries", 0))
        for run in runs
    )
    voluntary = sum(
        int(run.get("e188_guard", {}).get("voluntary_deliveries", 0))
        for run in runs
    )
    total = forced + voluntary
    numeric_keys = sorted({
        key
        for run in runs
        for key, value in run.get("e188_guard", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    summary.update({
        "e188_guard_totals": {
            key: _sum_numeric(runs, key) for key in numeric_keys
        },
        "e188_forced_deliveries": forced,
        "e188_voluntary_deliveries": voluntary,
        "e188_guard_forced_delivery_share": (
            None if total == 0 else forced / total
        ),
        "e188_mean_case_guard_forced_delivery_share": (
            fmean(
                run["e188_guard"]["forced_delivery_share"]
                for run in runs
                if run.get("e188_guard", {}).get("forced_delivery_share")
                is not None
            ) if any(
                run.get("e188_guard", {}).get("forced_delivery_share")
                is not None for run in runs
            ) else None
        ),
    })
    return summary


single_objective_rescore = e187.single_objective_rescore


__all__ = ["run_episode", "single_objective_rescore", "summarize_runs"]

