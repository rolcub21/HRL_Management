"""Storage-clock objective for a static, fixed-order inbound batch."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from example.episode_instance import EpisodeInstance
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TIMING_OBJECTIVE_INFO_KEY
from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1.protocol import SPEC
from experiments.conditioned_vcg.E130_exact_macro_cost_diagnostic_194k_v1 import costs
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.environment import GridEnv
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1 import model as wait_model
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1.objective import objective_cost


TOLERANCE = 20


def storage_due(block):
    stored = getattr(block, "stored_time_step", None)
    if stored is None:
        raise ValueError("storage-clock due date is undefined before admission")
    return int(stored) + int(block.storage_steps_needed)


def storage_error(block):
    if block.delivered_time_step is None:
        raise ValueError("storage error requires a delivery")
    return float(int(block.delivered_time_step) - storage_due(block))


def split_error(error):
    value = float(error)
    return max(value - TOLERANCE, 0.0), max(-value - TOLERANCE, 0.0)


class FixedAdmissionStorageEnv(GridEnv):
    """All jobs are released at zero; their deadlines start on admission."""

    def __init__(self, value):
        instance = EpisodeInstance.from_dict(value["instance"])
        self._episode_instance = instance
        self._initial_positions = ()
        self._external_dates = dict(value["external_due_dates"])
        self.initial_labels = set()
        self.expected_admission_labels = tuple(
            f"B{index + 1}" for index in range(int(value["incoming_count"]))
        )
        self.observed_admission_labels = []
        self.peak_stored = 0
        ObjectiveAuditSmallRoomsEnv.__init__(
            self,
            timing_objective=SPEC,
            grid_rows=8,
            grid_cols=8,
            start_state=(1, 1),
            door_cell=(0, 6),
            pickup_cells=(1, 6),
            exit_cells=[(7, 4), (7, 5), (7, 6)],
            number_blocks=instance.number_blocks,
            choose_storage=False,
            arrival_rate=instance.arrival_rate,
            proc_mean=instance.proc_mean,
        )
        self.reset(instance)

    def reset(self, instance=None):
        result = super().reset(instance)
        self.observed_admission_labels = []
        self.peak_stored = 0
        self.action_counts = Counter()
        return result

    def public_due_date(self, block):
        if block.delivered or block.stored_time_step is not None:
            return storage_due(block)
        # Only a hypothetical immediate-admission date is public before the
        # fixed admission.  The liveness event code is separately restricted
        # to actually stored blocks.
        return int(self.time_steps) + int(block.storage_steps_needed)

    def signed_remaining_storage_time(self, block):
        if block.delivered:
            return None
        if block.stored_time_step is None:
            return float(block.storage_steps_needed)
        return float(storage_due(block) - int(self.time_steps))

    def compute_block_features(self, block, agent_pos, goal_state):
        values = list(super().compute_block_features(block, agent_pos, goal_state))
        remaining = self.signed_remaining_storage_time(block)
        if remaining is not None:
            values[5] = remaining
        return tuple(values)

    def _overdue_stored_count(self, at_time):
        return sum(
            block.stored_time_step is not None
            and not block.delivered
            and int(at_time) >= storage_due(block) + TOLERANCE
            for block in self.blocks
        )

    def step(self, action):
        self.action_counts[int(action)] += 1
        tardiness = float(self._overdue_stored_count(self.time_steps))
        equipment = float(int(action) != int(self.ACTION_IDS["WAIT"]))
        state, _old_reward, done, info = ObjectiveAuditSmallRoomsEnv.step(self, action)
        rehandle = float(bool(info.get("relocated_block")))
        early = 0.0
        if info.get("stored_block"):
            label = str(info["stored_block"])
            expected = self.expected_admission_labels[len(self.observed_admission_labels)]
            if label != expected:
                raise RuntimeError(
                    f"fixed admission order changed: expected {expected}, observed {label}"
                )
            self.observed_admission_labels.append(label)
        if info.get("delivered_block"):
            block = next(
                item for item in self.blocks
                if str(item.label) == str(info["delivered_block"])
            )
            error = storage_error(block)
            _late, early = split_error(error)
            info["storage_delivery_error_time"] = error
            info["delivery_error_time"] = error
            block.delivery_error_time = error
            self.delivery_error_times[-1] = (block.label, error, self.current_episode)

        components = (tardiness, early, 0.0, equipment, rehandle)
        reward = wait_model.scalarized_components(components) / 0.01
        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        record["step_reward_by_objective"] = {
            key: float(reward) for key in record["step_reward_by_objective"]
        }
        record["selected_step_reward"] = float(reward)
        record["e164_storage_clock_components"] = dict(
            zip(wait_model.COMPONENT_NAMES, components)
        )
        record["e164_due_rule"] = "stored_time_step + storage_steps_needed"
        info = dict(info)
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records[-1] = record
        self.factorized_component_records.append(components)
        self.equipment_work_total += equipment
        self.physical_rehandles_total += rehandle
        stored = sum(
            block.stored_time_step is not None and not block.delivered
            for block in self.blocks
        )
        self.peak_stored = max(self.peak_stored, stored)
        self.progress_clock.observe(
            self.time_steps,
            sum(block.stored_time_step is not None for block in self.blocks),
            sum(block.delivered for block in self.blocks),
        )
        return state, float(reward), done, info

    def trajectory_metrics(self):
        delivered = tuple(block for block in self.blocks if block.delivered)
        errors = tuple(storage_error(block) for block in delivered)
        totals = tuple(
            float(sum(row[index] for row in self.factorized_component_records))
            for index in range(5)
        )
        complete = len(delivered) == len(self.blocks)
        cost = objective_cost(totals)
        terminal = sum(split_error(error)[0] for error in errors)
        early = sum(split_error(error)[1] for error in errors)
        if complete and (
            not math.isclose(totals[0], terminal, abs_tol=1e-8)
            or not math.isclose(totals[1], early, abs_tol=1e-8)
        ):
            raise RuntimeError("storage-clock dense and terminal costs differ")
        return {
            "J": cost,
            "strict_complete": complete,
            "cost_ranking_eligible": complete,
            "storage_clock_mae": (
                sum(abs(value) for value in errors) / len(errors) if errors else None
            ),
            "within_tolerance_rate": (
                sum(abs(value) <= TOLERANCE for value in errors) / len(errors)
                if errors else None
            ),
            "equipment_work": float(self.equipment_work_total),
            "physical_rehandles": float(self.physical_rehandles_total),
            "admission_wait": 0.0,
            "components": {
                name: value for name, value in zip(wait_model.COMPONENT_NAMES, totals)
            },
        }

    def day_metrics(self):
        return {
            "static_batch": True,
            "incoming_count": len(self.blocks),
            "peak_stored": int(self.peak_stored),
            "peak_stored_fraction": self.peak_stored / len(self.storage_positions),
            "fixed_admission_order_preserved": tuple(self.observed_admission_labels)
            == self.expected_admission_labels[: len(self.observed_admission_labels)],
            "observed_steps": int(self.time_steps),
        }


def stored_window_open(env, at_time=None):
    now = int(env.time_steps if at_time is None else at_time)
    openings = [
        storage_due(block) - TOLERANCE
        for block in env.blocks
        if block.stored_time_step is not None
        and not block.delivered
        and storage_due(block) - TOLERANCE > now
    ]
    return min(openings) if openings else None


def nominal_components(env, candidate, gamma=1.0):
    prediction = costs.predict(env, candidate, gamma)
    duration = int(prediction["duration"])
    jobs = tuple(
        costs.ReleasedJob(str(block.label), storage_due(block))
        for block in env.blocks
        if block.stored_time_step is not None and not block.delivered
    )
    four = costs.components(
        int(env.time_steps),
        duration,
        jobs,
        candidate.action_type.value,
        candidate.target_label,
        gamma,
    )
    return {
        "duration": duration,
        "components": (four[0], four[1], 0.0, four[2], four[3]),
        "assumption": "storage-clock stored jobs and nominal macro path",
        "event_sensitive": candidate.action_type.value == "defer",
    }
