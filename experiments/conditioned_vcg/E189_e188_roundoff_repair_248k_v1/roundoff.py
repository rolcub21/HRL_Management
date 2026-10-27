"""Numerical-only repair for long E188 return-accounting comparisons.

The inherited audit compares one primitive-sequential sum with the same
rewards grouped by macro at a fixed absolute tolerance of 1e-9.  This module
changes only those end-of-episode comparisons.  It derives a conservative
episode-specific floating-point bound from the recorded primitive rewards.
Per-step reward checks, rewards, actions, learning and certification are
untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import sys
from unittest.mock import patch

import run_vcg_objective_gamma_audit as audit
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    instrumentation as e187_instrumentation,
)

from . import protocol as p


class _MathProxy:
    def __init__(self) -> None:
        self.bound = 1.0e-9
        self.max_difference = 0.0
        self.relaxed_comparisons = 0
        self.primitive_steps = 0

    def __getattr__(self, name: str):
        return getattr(math, name)

    def isclose(self, a, b, *, rel_tol=1.0e-9, abs_tol=0.0):
        # These exact arguments identify only the inherited episode-total
        # comparisons.  Every other comparison keeps its original tolerance.
        if rel_tol != 0.0 or abs_tol != 1.0e-9:
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        difference = abs(float(a) - float(b))
        self.max_difference = max(self.max_difference, difference)
        if difference > self.bound:
            raise RuntimeError(
                "E189 return-accounting discrepancy exceeds the recorded "
                f"floating-point bound: difference={difference:.12g}, "
                f"bound={self.bound:.12g}, steps={self.primitive_steps}"
            )
        if difference > abs_tol:
            self.relaxed_comparisons += 1
        return math.isclose(a, b, rel_tol=0.0, abs_tol=self.bound)


def _bound_from_records(records: tuple[dict, ...]) -> float:
    count = len(records)
    if not count:
        return 1.0e-9
    # Standard gamma_n bound for sequential floating summation.  Factor four
    # covers both primitive-sequential and macro-grouped paths plus the small
    # per-step scalar decomposition roundoff.  This is the validated E79-v2
    # construction and is derived from this episode, not yard size.
    u = sys.float_info.epsilon / 2.0
    gamma = (count * u) / (1.0 - count * u)
    magnitude = math.fsum(
        max(
            abs(float(value))
            for value in record["step_reward_by_objective"].values()
        )
        for record in records
    )
    return max(1.0e-9, 4.0 * gamma * magnitude)


@contextmanager
def repaired_audit():
    proxy = _MathProxy()
    original_summary = p.TargetDenseEnv.objective_audit_summary
    original_episode = e187_instrumentation.BASE_RUN_EPISODE

    def objective_audit_summary(env):
        records = env.objective_audit_records
        proxy.primitive_steps = len(records)
        proxy.bound = _bound_from_records(records)
        return original_summary(env)

    def run_episode(*args, **kwargs):
        proxy.bound = 1.0e-9
        proxy.max_difference = 0.0
        proxy.relaxed_comparisons = 0
        proxy.primitive_steps = 0
        result = original_episode(*args, **kwargs)
        result["e189_roundoff_audit"] = {
            "primitive_steps": proxy.primitive_steps,
            "absolute_bound": proxy.bound,
            "maximum_absolute_difference": proxy.max_difference,
            "relaxed_comparisons": proxy.relaxed_comparisons,
        }
        return result

    with (
        patch.object(
            p.TargetDenseEnv,
            "objective_audit_summary",
            objective_audit_summary,
        ),
        patch.object(audit, "math", proxy),
        patch.object(
            e187_instrumentation,
            "BASE_RUN_EPISODE",
            run_episode,
        ),
    ):
        yield


__all__ = ["_MathProxy", "_bound_from_records", "repaired_audit"]
