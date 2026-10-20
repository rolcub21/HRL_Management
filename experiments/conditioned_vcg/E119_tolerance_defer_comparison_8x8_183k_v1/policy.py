"""Shift only the liveness rule's absolute waiting deadline."""

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from PSLAP.viability_candidates import BoundedEventDeferRule


@dataclass(frozen=True)
class ToleranceDeferRule(BoundedEventDeferRule):
    tolerance_steps: int = 20
    version: str = field(default="e119_absolute_due_plus_tolerance_defer_v1", init=False)

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.tolerance_steps, bool)
                or not isinstance(self.tolerance_steps, int) or self.tolerance_steps < 0):
            raise ValueError("tolerance_steps must be a nonnegative integer")

    def evaluate(self, env, yard, **kwargs):
        # This view is used only by the waiting rule. Never change the actual
        # yard, recovery state, graph due slack, environment clock, or reward.
        deadlines = SimpleNamespace(blocks=tuple(
            SimpleNamespace(remaining_time=float(block.remaining_time) + self.tolerance_steps)
            for block in yard.blocks
        ))
        result = super().evaluate(env, deadlines, **kwargs)
        if self.tolerance_steps:
            names = {
                "admitted_work_due_requires_recovery_progress": "latest_tolerated_due_requires_progress",
                "bounded_wait_toward_known_positive_deadline": "bounded_wait_toward_latest_tolerated_due",
            }
            result = replace(result, reason=names.get(result.reason, result.reason))
        return result
