"""Bind E189's numerical-only long-episode audit repair to E191."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1 import (
    roundoff as e189,
)

from . import protocol as p


@contextmanager
def repaired_audit():
    # E189's repair intentionally closes over its protocol environment class.
    # Rebind only that class while using the already-validated implementation.
    with patch.object(e189.p, "TargetDenseEnv", p.TargetDenseEnv):
        with e189.repaired_audit():
            yield


_MathProxy = e189._MathProxy
_bound_from_records = e189._bound_from_records


__all__ = ["_MathProxy", "_bound_from_records", "repaired_audit"]
