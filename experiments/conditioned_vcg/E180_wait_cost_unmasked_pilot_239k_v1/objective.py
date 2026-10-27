"""Storage-clock timing with a release-to-admission charge for every job."""

from copy import deepcopy
import math

from vcg_objective_audit import TIMING_OBJECTIVE_INFO_KEY
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1 import model as wait_model
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1.objective import (
    objective_cost,
    waiting_jobs,
)
from experiments.conditioned_vcg.E164_fixed_admission_storage_clock_223k_v1 import objective as storage


WAIT_WEIGHT = 0.5


def nominal_components(env, candidate, gamma=1.0):
    """Same macro prediction as E164, plus currently observable outside wait."""
    prediction = storage.nominal_components(env, candidate, gamma)
    duration = int(prediction["duration"])
    wait = len(waiting_jobs(env)) * sum(float(gamma) ** k for k in range(duration))
    values = list(prediction["components"])
    values[2] = float(wait)
    return {
        **prediction,
        "components": tuple(values),
        "assumption": "storage-clock stored jobs; released unadmitted jobs; nominal macro path",
    }


class WaitPricedStorageEnv(storage.FixedAdmissionStorageEnv):
    """Keep FIFO admission physical; price its discretionary timing in J."""

    def step(self, action):
        waiting = float(len(waiting_jobs(self)))
        state, base_reward, done, info = super().step(action)
        components = list(self.factorized_component_records[-1])
        if components[2] != 0.0:
            raise RuntimeError("E164 parent already charges admission wait")
        components[2] = waiting
        self.factorized_component_records[-1] = tuple(components)
        reward = wait_model.scalarized_components(components) / 0.01
        if not math.isclose(reward, base_reward - WAIT_WEIGHT * waiting, abs_tol=1e-8):
            raise RuntimeError("wait-priced step differs from parent plus wait")
        if getattr(self, "primitive_records", None):
            self.primitive_records[-1]["components"] = tuple(components)
        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        record["selected_step_reward"] = float(reward)
        record["step_reward_by_objective"] = {
            key: float(reward) for key in record["step_reward_by_objective"]
        }
        record["e180_wait_priced_components"] = dict(
            zip(wait_model.COMPONENT_NAMES, components)
        )
        record["e180_wait_definition"] = "time from release at zero to first storage admission"
        self._objective_audit_records[-1] = record
        info = dict(info)
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        return state, float(reward), done, info

    def trajectory_metrics(self):
        metrics = super().trajectory_metrics()
        wait = float(sum(
            block.stored_time_step - block.arrival_step
            for block in self.blocks
            if block.stored_time_step is not None
        ))
        dense = float(metrics["components"]["admission_wait_accrual"])
        if metrics["strict_complete"] and not math.isclose(wait, dense, abs_tol=1e-8):
            raise RuntimeError("dense and trajectory admission wait differ")
        metrics["admission_wait"] = dense
        metrics["primary_objective"] = "J_storage_clock_plus_0.5_admission_wait"
        return metrics
