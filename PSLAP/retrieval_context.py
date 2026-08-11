"""Shared read-only retrieval plans for one observable decision state."""

from __future__ import annotations

from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.retrieval_dispatch import plan_retrieval


def _selector_identity(selector):
    if selector is None:
        return "default_dynamic_relocation"
    return (
        getattr(selector, "VERSION", None)
        or getattr(selector, "__module__", "")
        + "."
        + getattr(selector, "__qualname__", repr(selector))
    )


def _observable_key(env, selector):
    active = tuple(
        (
            block.label,
            block.position,
            block.storage_location,
            bool(block.carrying),
            bool(block.stored),
            bool(block.delivered),
            block.stored_time_step,
            block.storage_steps_needed,
        )
        for block in env.blocks
        if (
            block.position is not None
            or block.carrying
            or block.stored
            or block.delivered
        )
    )
    return (
        int(env.time_steps),
        tuple(env.current_state),
        active,
        env.rooms.tobytes(),
        _selector_identity(selector),
    )


class RetrievalPlanningContext:
    """Lazily memoize named plans without changing environment semantics."""

    def __init__(self, env, relocation_selector):
        self.env = env
        self.relocation_selector = relocation_selector
        self.yard = YardSnapshot.from_env(env)
        self._plans = {}

    def plan(self, block_label):
        label = str(block_label)
        if label not in self._plans:
            self._plans[label] = plan_retrieval(
                self.yard,
                self.env.current_state,
                label,
                relocation_selector=self.relocation_selector,
            )
        return self._plans[label]


def retrieval_planning_context(env, *, relocation_selector=None):
    key = _observable_key(env, relocation_selector)
    cached = getattr(env, "_retrieval_planning_context_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    context = RetrievalPlanningContext(env, relocation_selector)
    env._retrieval_planning_context_cache = (key, context)
    return context


__all__ = ["RetrievalPlanningContext", "retrieval_planning_context"]
