"""One policy-independent, certified, event-interruptible Hold macro.

This is a V2 execution object.  It is intentionally separate from the frozen
V1 :mod:`ExplicitDeferOption` so prospective V1/V1.2 training cannot change
semantics while it is running.
"""

from __future__ import annotations

from collections import Counter

from option import BaseOption


class ExplicitHoldOption(BaseOption):
    """Execute WAIT until an observable event or a certified primitive cap.

    The option never reads an unarrived block's arrival time, processing time,
    or any future schedule.  Its event detector uses only presently observable
    inventory positions/statuses.  A changed signature ends the option so the
    exact verifier can rebuild the frontier under the new physical state.

    ``idle_budget_steps`` is the primitive idle budget remaining when the
    option is bound.  The constructor enforces ``horizon_steps`` against it;
    the controller owns the cross-macro budget transition.
    """

    VERSION = "certified_event_interruptible_hold_option_v2"
    BINDING_CONTRACT = "episode_epoch_horizon_and_idle_budget_bound_once_v2"
    EVENT_CONTRACT = "online_observable_inventory_stutter_or_change_v2"
    POLICY_CONTRACT = "fixed_wait_no_planner_selector_or_future_schedule_v2"
    ROBUST_STUTTER_CONTRACT = (
        "wait_preserves_certified_physical_configuration_until_observed_event_v2"
    )

    def __init__(
        self,
        env,
        *,
        horizon_steps: int,
        idle_budget_steps: int,
    ) -> None:
        super().__init__(is_primitive=False)
        for name, value in (
            ("horizon_steps", horizon_steps),
            ("idle_budget_steps", idle_budget_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if horizon_steps > idle_budget_steps:
            raise ValueError("Hold horizon cannot exceed remaining idle budget")

        self.env = env
        self.horizon_steps = int(horizon_steps)
        self.idle_budget_steps = int(idle_budget_steps)
        self.bound_time_step = int(env.time_steps)
        self.bound_agent_position = tuple(env.current_state)
        instance = getattr(env, "current_episode_instance", None)
        self.bound_instance_id = (
            None if instance is None else instance.instance_id
        )
        self.controller_identifier = (
            f"option:CertifiedHoldV2:horizon-{self.horizon_steps:04d}"
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

    def _reset_active(self) -> None:
        self.started = False
        self.failed = False
        self.failure_reason = None
        self.steps = 0
        self.initial_observed_signature = None
        self._completion_recorded = False

    def _observed_signature(self):
        """Return only state that is observable at the current time.

        In particular, this method deliberately does not access
        ``arrival_step`` or ``storage_steps_needed``.  An outside block is
        represented only by its currently observable absence/status.
        """

        return (
            tuple(self.env.current_state),
            tuple(
                (
                    index,
                    block.position,
                    bool(block.carrying),
                    bool(block.stored),
                    bool(block.delivered),
                )
                for index, block in enumerate(self.env.blocks)
            ),
        )

    def _binding_failure(self):
        if self._consumed:
            return "bound_action_already_consumed"
        if int(self.env.time_steps) != self.bound_time_step:
            return "bound_action_not_started_at_decision_epoch"
        if tuple(self.env.current_state) != self.bound_agent_position:
            return "bound_agent_position_changed"
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
        del state
        return bool(
            not self.started
            and not self.failed
            and self._binding_failure() is None
        )

    def _start(self) -> bool:
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
        del state, test
        if not self.started and not self.failed:
            self._start()
        if not self.failed:
            self.steps += 1
        return self.env.ACTION_IDS["WAIT"]

    def _record(self, *, success: bool, reason: str) -> None:
        if self._completion_recorded:
            return
        self.last_outcome = {
            "success": bool(success),
            "reason": str(reason),
            "steps": int(self.steps),
            "horizon_steps": self.horizon_steps,
            "idle_budget_steps_at_bind": self.idle_budget_steps,
            "robust_stutter_contract": self.ROBUST_STUTTER_CONTRACT,
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
        del state
        if not self.started:
            return False
        if self.failed:
            self._record(success=False, reason=self.failure_reason)
            return True

        if self.env.is_state_terminal(self.env.current_state):
            reason = "environment_terminal"
        elif any(block.carrying for block in self.env.blocks):
            self.failed = True
            self.failure_reason = "carry_state_changed"
            self._record(success=False, reason=self.failure_reason)
            return True
        elif self._observed_signature() != self.initial_observed_signature:
            reason = "observed_event"
        elif self.steps >= self.horizon_steps:
            reason = "hold_cap"
        else:
            return False
        self._record(success=True, reason=reason)
        return True

    def on_env_reset(self) -> None:
        self._reset_active()
        self._consumed = True
        self.last_outcome = None
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_outcome_counts.clear()
        self.episode_outcomes.clear()

    def __str__(self) -> str:
        return (
            "ExplicitHoldOption("
            f"horizon={self.horizon_steps}, budget={self.idle_budget_steps})"
        )

    __repr__ = __str__

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.bound_instance_id,
                self.bound_time_step,
                self.horizon_steps,
                self.idle_budget_steps,
            )
        )

    def __eq__(self, other) -> bool:
        return bool(
            isinstance(other, ExplicitHoldOption)
            and self.bound_instance_id == other.bound_instance_id
            and self.bound_time_step == other.bound_time_step
            and self.horizon_steps == other.horizon_steps
            and self.idle_budget_steps == other.idle_budget_steps
        )


__all__ = ["ExplicitHoldOption"]
