"""Online-safe event-driven defer option for temporal scheduling."""

from collections import Counter

from option import BaseOption
from PSLAP.retrieval_context import retrieval_planning_context


class StrategicDeferOption(BaseOption):
    VERSION = "strategic_defer_until_event_v1"
    controller_identifier = "option:DeferUntilEvent:v1"

    def __init__(self, env, max_defer_steps=10, *, relocation_selector=None):
        super().__init__(is_primitive=False)
        self.env = env
        self.max_defer_steps = int(max_defer_steps)
        if self.max_defer_steps <= 0:
            raise ValueError("max_defer_steps must be positive")
        self.relocation_selector = relocation_selector
        self.last_outcome = None
        self.episode_outcome_counts = Counter()
        self._reset_active()

    def _reset_active(self):
        self.started = False
        self.steps = 0
        self.initial_observed_signature = None
        self.initial_due_labels = frozenset()

    def _observed_signature(self):
        # Do not inspect attributes of unarrived blocks.  The signature changes
        # only when an observable status/position event occurs.
        return tuple(
            (
                index,
                block.position,
                bool(block.carrying),
                bool(block.stored),
                bool(block.delivered),
            )
            for index, block in enumerate(self.env.blocks)
            if (
                block.position is not None
                or block.carrying
                or block.stored
                or block.delivered
            )
        )

    def _due_labels(self):
        if any(block.carrying for block in self.env.blocks):
            return frozenset()
        context = retrieval_planning_context(
            self.env, relocation_selector=self.relocation_selector
        )
        due = []
        for block in context.yard.blocks:
            plan = context.plan(block.label)
            if plan is not None and plan.slack <= 0:
                due.append(block.label)
        return frozenset(due)

    def initiation(self, state):
        return bool(
            not any(block.carrying for block in self.env.blocks)
            and not self.env.is_state_terminal(self.env.current_state)
        )

    def policy(self, state, test=False):
        if not self.started:
            self.started = True
            self.steps = 0
            self.initial_observed_signature = self._observed_signature()
            self.initial_due_labels = self._due_labels()
        self.steps += 1
        return self.env.ACTION_IDS["WAIT"]

    def termination(self, state):
        if not self.started:
            return False
        reason = None
        if self.env.is_state_terminal(self.env.current_state):
            reason = "environment_terminal"
        elif any(block.carrying for block in self.env.blocks):
            reason = "carry_state_changed"
        elif self.initial_due_labels and self.steps >= 1:
            reason = "already_due_single_step"
        elif self._observed_signature() != self.initial_observed_signature:
            reason = "observed_event"
        elif self._due_labels().difference(self.initial_due_labels):
            reason = "retrieval_became_due"
        elif self.steps >= self.max_defer_steps:
            reason = "defer_cap"
        if reason is None:
            return False
        self.last_outcome = {"reason": reason, "steps": self.steps}
        self.episode_outcome_counts[reason] += 1
        self._reset_active()
        return True

    def on_env_reset(self):
        self._reset_active()
        self.last_outcome = None
        self.episode_outcome_counts.clear()

    def __str__(self):
        return "StrategicDeferOption"

    __repr__ = __str__

    def __hash__(self):
        return hash(type(self))


__all__ = ["StrategicDeferOption"]
