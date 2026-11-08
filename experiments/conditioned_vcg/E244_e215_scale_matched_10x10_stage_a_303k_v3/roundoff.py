"""Bind the numerical return audit to E244's 10x10 environment exactly once."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1 import (
    roundoff as e189,
)

from . import protocol as p


@contextmanager
def repaired_audit():
    """Install E189's validated bound on the environment E244 actually uses."""

    with patch.object(e189.p, "TargetDenseEnv", p.TargetDenseEnv):
        with e189.repaired_audit():
            yield


_MathProxy = e189._MathProxy
_bound_from_records = e189._bound_from_records


__all__ = ["_MathProxy", "_bound_from_records", "repaired_audit"]
