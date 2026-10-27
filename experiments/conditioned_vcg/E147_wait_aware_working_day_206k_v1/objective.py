"""Five-component release-clock objective and working-day environment."""

from copy import deepcopy
import math

from vcg_objective_audit import TIMING_OBJECTIVE_INFO_KEY
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.rewards import (
    COMPONENT_NAMES as OLD_COMPONENT_NAMES,
)
from experiments.conditioned_vcg.E130_exact_macro_cost_diagnostic_194k_v1 import (
    costs as old_costs,
)
from experiments.conditioned_vcg.E134_regime_grid_protocol_198k_v1.objective import (
    Job,
    rescore,
)
from experiments.conditioned_vcg.E146_empty_start_working_day import run as day


WAIT_WEIGHT = 0.5
COMPONENT_NAMES = (
    "release_tardiness_accrual",
    "release_earliness_at_delivery",
    "admission_wait_accrual",
    "equipment_work",
    "physical_rehandles",
)
OBJECTIVE_WEIGHTS = (2.0, 1.0, WAIT_WEIGHT, 0.1, 10.0)
SCALAR_WEIGHTS = tuple(-0.01 * value for value in OBJECTIVE_WEIGHTS)


def scalarized_components(values):
    values = tuple(float(value) for value in values)
    if len(values) != len(COMPONENT_NAMES):
        raise ValueError("E147 cost vector has the wrong width")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("E147 cost components must be finite and nonnegative")
    return float(sum(weight * value for weight, value in zip(SCALAR_WEIGHTS, values)))


def objective_cost(values):
    return -scalarized_components(values) / 0.01


def waiting_jobs(env, at_time=None):
    now = int(env.time_steps if at_time is None else at_time)
    return tuple(
        block
        for block in env.blocks
        if int(block.arrival_step) <= now
        and getattr(block, "stored_time_step", None) is None
        and not bool(block.delivered)
    )


def nominal_components(env, candidate, gamma=1.0):
    """Known current cost only; hidden arrivals during a macro are excluded."""

    prediction = old_costs.predict(env, candidate, gamma)
    duration = int(prediction["duration"])
    known_waiting = len(waiting_jobs(env))
    wait = known_waiting * sum(float(gamma) ** k for k in range(duration))
    old = tuple(float(value) for value in prediction["components"])
    values = (old[0], old[1], float(wait), old[2], old[3])
    return {
        "duration": duration,
        "components": values,
        "assumption": (
            "released unadmitted jobs and nominal path only; future arrivals "
            "and execution repairs excluded"
        ),
        "event_sensitive": bool(prediction.get("event_sensitive")),
    }


class WaitDayEnv(day.DayEnv):
    """E146 physical day with admission waiting in the learned objective."""

    def step(self, action):
        wait = float(len(waiting_jobs(self)))
        state, old_reward, done, info = super().step(action)
        old = tuple(float(value) for value in self.factorized_component_records[-1])
        if len(old) != len(OLD_COMPONENT_NAMES):
            raise RuntimeError("E147 parent component width differs")
        values = (old[0], old[1], wait, old[2], old[3])
        reward = -objective_cost(values)
        self.factorized_component_records[-1] = values
        if self.primitive_records:
            self.primitive_records[-1]["components"] = values

        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        record["step_reward_by_objective"] = {
            key: float(reward) for key in record["step_reward_by_objective"]
        }
        record["selected_step_reward"] = float(reward)
        record["e147_wait_aware_components"] = dict(zip(COMPONENT_NAMES, values))
        record["e147_wait_definition"] = (
            "released through first completed storage admission"
        )
        info = dict(info)
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records[-1] = record
        if not math.isclose(float(old_reward) - WAIT_WEIGHT * wait, reward, abs_tol=1e-9):
            raise RuntimeError("E147 incremental objective differs from parent plus wait")
        return state, float(reward), done, info

    def trajectory_metrics(self):
        jobs = [
            Job(
                label=block.label,
                release=block.arrival_step,
                due=self._external_dates[block.label],
                first_storage=block.stored_time_step,
                delivery=block.delivered_time_step,
                initially_stored=False,
            )
            for block in self.blocks
        ]
        scored = rescore(
            jobs,
            observation_end=self.time_steps,
            equipment_work=self.equipment_work_total,
            rehandles=self.physical_rehandles_total,
            strict_complete=all(block.delivered for block in self.blocks),
        )
        dense = sum(objective_cost(values) for values in self.factorized_component_records)
        if not math.isclose(scored["J_W"], dense, abs_tol=1e-7):
            raise RuntimeError("E147 dense and trajectory wait-aware costs disagree")
        old_j = float(scored["J"])
        scored["J_without_wait"] = old_j
        scored["J"] = float(scored["J_W"])
        scored["primary_objective"] = "J_wait_aware"
        scored["sensitivity_only"] = False
        scored["retraining"] = True
        scored["components"] = {
            **scored["components"],
            "admission_wait": float(scored["total_admission_wait"]),
        }
        return scored

