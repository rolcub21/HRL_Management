"""One policy-independent, explicitly bounded defer macro."""

from __future__ import annotations

from collections import Counter

from option import BaseOption


class ExplicitDeferOption(BaseOption):
    """Advance time until an observable event or an exact WAIT cap.

    The frontier decides whether waiting is live and supplies its horizon.
    This execution object deliberately performs no retrieval planning, due
    calculation, relocation proposal, or action ranking.  It merely executes
    WAIT and terminates when the online-observable inventory signature changes,
    the environment terminates, carrying state appears, or the cap is reached.
    """

    VERSION = "bound_observable_event_defer_option_v1"
    BINDING_CONTRACT = "episode_epoch_horizon_bound_once_v1"
    EVENT_CONTRACT = "observable_inventory_status_or_position_change_v1"
    POLICY_CONTRACT = "fixed_wait_no_planner_or_selector_v1"

    def __init__(self, env, horizon_steps: int):
        super().__init__(is_primitive=False)
        if (
            isinstance(horizon_steps, bool)
            or not isinstance(horizon_steps, int)
            or horizon_steps <= 0
        ):
            raise ValueError("horizon_steps must be a positive integer")
        self.env = env
        self.horizon_steps = int(horizon_steps)
        self.bound_time_step = int(env.time_steps)
        instance = getattr(env, "current_episode_instance", None)
        self.bound_instance_id = (
            None if instance is None else instance.instance_id
        )
        self.controller_identifier = (
            f"option:ExplicitDefer:horizon-{self.horizon_steps:04d}"
        )

        self.success_count = 0
        self.failure_count = 0
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_outcome_counts = Counter()
        self.episode_outcomes = []
        self.last_outcome = None
        self._consumed = False
        self._reset_active()

    def _reset_active(self):
        self.started = False
        self.failed = False
        self.failure_reason = None
        self.steps = 0
        self.initial_observed_signature = None
        self._completion_recorded = False

    def _observed_signature(self):
        # Unarrived attributes are deliberately absent.  Advancing storage
        # clocks alone is not an event: the frontier already chose the exact
        # deadline-bounded horizon at the decision epoch.
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

    def _binding_failure(self):
        if self._consumed:
            return "bound_action_already_consumed"
        if int(self.env.time_steps) != self.bound_time_step:
            return "bound_action_not_started_at_decision_epoch"
        instance = getattr(self.env, "current_episode_instance", None)
        instance_id = None if instance is None else instance.instance_id
        if instance_id != self.bound_instance_id:
            return "bound_action_episode_changed"
        if any(block.carrying for block in self.env.blocks):
            return "inventory_already_carried"
        if self.env.is_state_terminal(self.env.current_state):
            return "environment_terminal"
        return None

    def initiation(self, state):
        return bool(
            not self.started
            and not self.failed
            and self._binding_failure() is None
        )

    def _start(self):
        reason = self._binding_failure()
        self.started = True
        self.steps = 0
        if reason is not None:
            self.failed = True
            self.failure_reason = reason
            return False
        self.initial_observed_signature = self._observed_signature()
        return True

    def policy(self, state, test=False):
        if not self.started and not self.failed:
            self._start()
        if not self.failed:
            self.steps += 1
        return self.env.ACTION_IDS["WAIT"]

    def _record(self, *, success, reason):
        if self._completion_recorded:
            return
        self.last_outcome = {
            "success": bool(success),
            "reason": str(reason),
            "steps": int(self.steps),
            "horizon_steps": self.horizon_steps,
        }
        self.episode_outcomes.append(dict(self.last_outcome))
        self.episode_outcome_counts[str(reason)] += 1
        if success:
            self.success_count += 1
            self.episode_success_count += 1
        else:
            self.failure_count += 1
            self.episode_failure_count += 1
        self._completion_recorded = True
        self._consumed = True

    def termination(self, state):
        if not self.started:
            return False
        if self.failed:
            self._record(success=False, reason=self.failure_reason)
            return True

        reason = None
        if self.env.is_state_terminal(self.env.current_state):
            reason = "environment_terminal"
        elif any(block.carrying for block in self.env.blocks):
            # WAIT cannot create carrying inventory under the environment
            # contract, so this indicates external interference.
            self.failed = True
            self.failure_reason = "carry_state_changed"
            self._record(success=False, reason=self.failure_reason)
            return True
        elif self._observed_signature() != self.initial_observed_signature:
            reason = "observed_event"
        elif self.steps >= self.horizon_steps:
            reason = "defer_cap"
        if reason is None:
            return False
        self._record(success=True, reason=reason)
        return True

    def on_env_reset(self):
        # Bound instances are intentionally not reusable across episodes.
        self._reset_active()
        self._consumed = True
        self.last_outcome = None
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_outcome_counts.clear()
        self.episode_outcomes.clear()

    def __str__(self):
        return f"ExplicitDeferOption(horizon={self.horizon_steps})"

    __repr__ = __str__

    def __hash__(self):
        return hash(
            (
                type(self),
                self.bound_instance_id,
                self.bound_time_step,
                self.horizon_steps,
            )
        )

    def __eq__(self, other):
        return bool(
            isinstance(other, ExplicitDeferOption)
            and self.bound_instance_id == other.bound_instance_id
            and self.bound_time_step == other.bound_time_step
            and self.horizon_steps == other.horizon_steps
        )


__all__ = ["ExplicitDeferOption"]
