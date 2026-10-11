"""Numerical-only E79 repair for long-episode return accounting.

The original audit compares sequentially summed primitive rewards with
macro-grouped sums at absolute tolerance 1e-9. Both paths see identical
environment steps, but long trajectories can exceed that fixed threshold.
This wrapper changes only those end-of-episode comparisons. It derives
an upper bound from the number and absolute size of the *recorded* primitive
rewards. The environment's independent per-step component check is untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import sys
from unittest.mock import patch

import run_vcg_objective_gamma_audit as audit

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
        # The original module uses exactly this signature for its
        # end-of-episode return checks. All other comparisons stay unchanged.
        if rel_tol != 0.0 or abs_tol != 1.0e-9:
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        difference = abs(float(a) - float(b))
        self.max_difference = max(self.max_difference, difference)
        if difference > self.bound:
            raise RuntimeError(
                "E79-v2 return-accounting discrepancy exceeds the recorded "
                f"floating-point bound: difference={difference:.12g}, "
                f"bound={self.bound:.12g}, steps={self.primitive_steps}"
            )
        if difference > abs_tol and difference <= self.bound:
            self.relaxed_comparisons += 1
        return math.isclose(a, b, rel_tol=0.0, abs_tol=self.bound)


def _bound_from_records(records: tuple[dict, ...]) -> float:
    count = len(records)
    if not count:
        return 1.0e-9
    # A sequential n-term floating sum has error at most gamma_(n-1) times
    # the absolute sum, with unit roundoff u=epsilon/2. A macro-grouped sum
    # introduces a second such path. Factor 4 covers both and per-step
    # decomposition rounding, without a fixed yard-size-dependent allowance.
    u = sys.float_info.epsilon / 2.0
    gamma = (count * u) / (1.0 - count * u)
    magnitude = math.fsum(
        max(abs(float(value)) for value in record["step_reward_by_objective"].values())
        for record in records
    )
    return max(1.0e-9, 4.0 * gamma * magnitude)


@contextmanager
def repaired_audit():
    proxy = _MathProxy()
    original_summary = p.TargetDenseEnv.objective_audit_summary
    original_episode = audit.run_objective_audit_episode

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
        result["e79_v2_roundoff_audit"] = {
            "primitive_steps": proxy.primitive_steps,
            "absolute_bound": proxy.bound,
            "maximum_absolute_difference": proxy.max_difference,
            "relaxed_comparisons": proxy.relaxed_comparisons,
        }
        return result

    with patch.object(p.TargetDenseEnv, "objective_audit_summary", objective_audit_summary), \
         patch.object(audit, "math", proxy), \
         patch.object(audit, "run_objective_audit_episode", run_episode):
        yield
