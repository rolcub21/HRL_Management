"""Repaired online dynamic PSLAP controller."""

from PSLAP.dynamic_yard import select_storage_location
from PSLAP.online_policy import OnlinePSLAPPolicy


def _dynamic_assignment(yard, block, source):
    return select_storage_location(yard, block, source=source)


class DynamicPSLAPPolicy(OnlinePSLAPPolicy):
    """Dynamic storage assignment with the common retrieval dispatcher."""

    def __init__(self, env):
        super().__init__(env, _dynamic_assignment)


__all__ = ["DynamicPSLAPPolicy"]
