"""Atomic block-specific retrieval option for the learned scheduler."""

from collections import Counter

from option import BaseOption
from PSLAP.retrieval_executor import (
    CompleteLiveRetrievalExecutor,
    RetrievalExecutor,
    StrictStartRetrievalExecutor,
)


class RetrieveDeliverOption(BaseOption):
    VERSION = "retrieve_deliver_option_v1"

    def __init__(self, env, block_index, *, relocation_selector=None):
        super().__init__(is_primitive=False)
        self.env = env
        self.block_index = int(block_index)
        self.target_label = env.blocks[self.block_index].label
        self.relocation_selector = relocation_selector
        self.controller_identifier = (
            f"option:RetrieveDeliver:block-{self.block_index:04d}"
        )
        self.executor = None
        self.success_count = 0
        self.failure_count = 0
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_relocations = 0
        self.episode_replans = 0
        self.episode_failure_outcomes = Counter()
        self.episode_failure_events = []
        self.last_outcome = None
        self._completion_recorded = False

    def _new_executor(self):
        return RetrievalExecutor(
            self.env,
            self.target_label,
            relocation_selector=self.relocation_selector,
        )

    def initiation(self, state):
        # Feasibility is deliberately independent of timing slack.  The critic
        # receives signed slack and learns whether to retrieve or defer.
        if any(block.carrying for block in self.env.blocks):
            return False
        return self._new_executor().can_start()

    def policy(self, state, test=False):
        if self.executor is None:
            executor = self._new_executor()
            if not executor.start():
                self.executor = executor
                return self.env.ACTION_IDS["WAIT"]
            self.executor = executor
            self._completion_recorded = False
        return self.executor.next_action()

    def termination(self, state):
        target = self.env.blocks[self.block_index]
        if target.delivered:
            if not self._completion_recorded:
                self.success_count += 1
                self.episode_success_count += 1
                self.episode_relocations += (
                    self.executor.relocation_count if self.executor else 0
                )
                self.episode_replans += (
                    self.executor.replan_count if self.executor else 0
                )
                self.last_outcome = {
                    "success": True,
                    "reason": "target_delivered",
                    "actual_steps": (
                        self.executor.actual_steps if self.executor else None
                    ),
                    "relocations": (
                        self.executor.relocation_count if self.executor else None
                    ),
                    "replans": (
                        self.executor.replan_count if self.executor else None
                    ),
                    "initial_estimated_steps": (
                        self.executor.initial_estimated_steps
                        if self.executor
                        else None
                    ),
                }
                if (
                    self.last_outcome["actual_steps"] is not None
                    and self.last_outcome["initial_estimated_steps"] is not None
                ):
                    self.last_outcome["path_expansion_steps"] = (
                        self.last_outcome["actual_steps"]
                        - self.last_outcome["initial_estimated_steps"]
                    )
                else:
                    self.last_outcome["path_expansion_steps"] = None
                self._completion_recorded = True
            self.executor = None
            return True
        if self.executor is not None and self.executor.failed:
            carrying = any(block.carrying for block in self.env.blocks)
            if not carrying:
                self.failure_count += 1
                self.episode_failure_count += 1
                self.episode_relocations += self.executor.relocation_count
                self.episode_replans += self.executor.replan_count
                self.last_outcome = {
                    "success": False,
                    "reason": self.executor.failure_reason,
                    "actual_steps": self.executor.actual_steps,
                    "relocations": self.executor.relocation_count,
                    "replans": self.executor.replan_count,
                    "initial_estimated_steps": (
                        self.executor.initial_estimated_steps
                    ),
                    "path_expansion_steps": (
                        None
                        if self.executor.initial_estimated_steps is None
                        else self.executor.actual_steps
                        - self.executor.initial_estimated_steps
                    ),
                }
                self.episode_failure_outcomes[
                    self.executor.failure_reason
                ] += 1
                self.episode_failure_events.append(
                    {
                        "block_label": self.target_label,
                        **self.last_outcome,
                    }
                )
                self.executor = None
                return True
        return False

    def on_env_reset(self):
        self.executor = None
        self.last_outcome = None
        self._completion_recorded = False
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_relocations = 0
        self.episode_replans = 0
        self.episode_failure_outcomes.clear()
        self.episode_failure_events.clear()

    def __str__(self):
        return f"RetrieveDeliverOption({self.target_label})"

    __repr__ = __str__

    def __hash__(self):
        return hash((type(self), self.block_index))

    def __eq__(self, other):
        return (
            isinstance(other, RetrieveDeliverOption)
            and self.block_index == other.block_index
        )


class FirstLegRetrieveDeliverOption(RetrieveDeliverOption):
    """Historical v2 option retained for protocol-reproduction runs."""

    VERSION = "retrieve_deliver_option_v2"

    def _new_executor(self):
        return StrictStartRetrievalExecutor(
            self.env,
            self.target_label,
            relocation_selector=self.relocation_selector,
        )

    def __eq__(self, other):
        return (
            isinstance(other, FirstLegRetrieveDeliverOption)
            and self.block_index == other.block_index
        )

    __hash__ = RetrieveDeliverOption.__hash__


class StrictRetrieveDeliverOption(RetrieveDeliverOption):
    """v3 named retrieval validated through delivery before initiation."""

    VERSION = "retrieve_deliver_option_v3"

    def _new_executor(self):
        return CompleteLiveRetrievalExecutor(
            self.env,
            self.target_label,
            relocation_selector=self.relocation_selector,
        )

    def __eq__(self, other):
        return (
            isinstance(other, StrictRetrieveDeliverOption)
            and self.block_index == other.block_index
        )

    __hash__ = RetrieveDeliverOption.__hash__


__all__ = [
    "RetrieveDeliverOption",
    "FirstLegRetrieveDeliverOption",
    "StrictRetrieveDeliverOption",
]
