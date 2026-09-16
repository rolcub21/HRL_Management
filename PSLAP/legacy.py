"""Frozen adapter for the pre-audit PSLAP-style controller.

This module intentionally preserves the historical implementation for internal
ablation. Correctness fixes belong in :mod:`PSLAP.dynamic_policy`.
"""

from PSLAP.PSLAPPolicy import PSLaPPolicy


LegacyAdaptedPSLAPPolicy = PSLaPPolicy

__all__ = ["LegacyAdaptedPSLAPPolicy"]
