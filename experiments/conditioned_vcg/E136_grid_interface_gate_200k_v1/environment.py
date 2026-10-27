"""External release-fixed dates without changing nominal storage requirements."""
from copy import deepcopy

from example.episode_instance import EpisodeInstance
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TIMING_OBJECTIVE_INFO_KEY
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.rewards import (
    ReleaseETCostEnv, COMPONENT_NAMES, objective_cost)
from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1.protocol import SPEC
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.liveness import ProgressClock
from experiments.conditioned_vcg.E134_regime_grid_protocol_198k_v1.objective import Job, rescore


def due(env, block, *, after_steps=0):
    """No future due-date lookup, even when projecting a current candidate."""
    if block.arrival_step > env.time_steps or block.delivered:
        raise ValueError('external date unavailable for unreleased/finished job')
    return env.public_due_date(block) - env.time_steps - after_steps


class GridEnv(ReleaseETCostEnv):
    def __init__(self, case):
        instance = EpisodeInstance.from_dict(case['instance'])
        # Do not attach the full calibration case/reference trace to a policy's
        # live environment. The simulator retains only its exogenous schedule,
        # initial packing and private dates, exposed after release via the hook.
        self._episode_instance = instance
        self._initial_positions = tuple(tuple(c) for c in case['initial_positions'])
        self._external_dates = dict(case['external_due_dates'])
        self.initial_labels = set()
        ObjectiveAuditSmallRoomsEnv.__init__(self, timing_objective=SPEC,
            grid_rows=8, grid_cols=8, start_state=(1, 1), door_cell=(0, 6),
            pickup_cells=(1, 6), exit_cells=[(7, 4), (7, 5), (7, 6)],
            number_blocks=instance.number_blocks, choose_storage=False,
            arrival_rate=instance.arrival_rate, proc_mean=80)
        self.reset(instance)

    def sample_episode_instance(self, seed=None):
        # The shared simulator constructor calls reset without an instance.
        return self._episode_instance

    def reset(self, instance=None):
        expected = self._episode_instance
        if instance is not None and instance != expected:
            raise ValueError('grid environment cannot silently resample workload')
        ObjectiveAuditSmallRoomsEnv.reset(self, expected)
        initial = self._initial_positions
        self.initial_labels = {b.label for b in self.blocks[:len(initial)]}
        if set(self._external_dates) != {b.label for b in self.blocks}:
            raise ValueError('missing or extra external due date')
        for index, block in enumerate(self.blocks):
            date = self._external_dates[block.label]
            if type(date) is not int or date < block.arrival_step:
                raise ValueError('invalid external release-fixed date')
            cell = tuple(initial[index]) if index < len(initial) else None
            block.position = block.storage_location = cell
            block.stored = block.picked = cell is not None
            block.carrying = block.delivered = False
            block.stored_time_step = 0 if cell is not None else None
            block.storage_steps_elapsed = 0
        self.storage_counts = {c: sum(tuple(x) == c for x in initial)
                               for c in self.storage_positions}
        self.update_pickup_queue()
        self.factorized_component_records = []
        self.equipment_work_total = self.physical_rehandles_total = 0.
        # Pre-existing inventory is not a newly observed admission event.
        self.progress_clock = ProgressClock(admissions=len(initial))
        return self.get_current_state()

    def public_due_date(self, block):
        if block.arrival_step > self.time_steps:
            raise ValueError('unreleased due date is private')
        return self._external_dates[block.label]

    def signed_remaining_storage_time(self, block):
        if block.arrival_step > self.time_steps or block.delivered:
            return None
        return float(due(self, block))

    def _overdue_undelivered_count(self, time_step):
        return sum(b.arrival_step <= time_step and not b.delivered
                   and time_step >= self.public_due_date(b)+20 for b in self.blocks)

    def step(self, action):
        late = float(self._overdue_undelivered_count(self.time_steps))
        state, _, done, info = ObjectiveAuditSmallRoomsEnv.step(self, action)
        early = 0.
        if info.get('delivered_block'):
            block = next(b for b in self.blocks if b.label == info['delivered_block'])
            error = self.time_steps-self.public_due_date(block)
            early = float(max(-error-20, 0))
            info['storage_delivery_error_time'] = info.get('delivery_error_time')
            info['delivery_error_time'] = block.delivery_error_time = error
            self.delivery_error_times[-1] = (block.label, error, self.current_episode)
        components = (late, early, float(action != 6), float(bool(info.get('relocated_block'))))
        reward = -objective_cost(components)
        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        record['step_reward_by_objective'] = {k: reward for k in record['step_reward_by_objective']}
        record['selected_step_reward'] = reward
        record['e115_release_et_components'] = dict(zip(COMPONENT_NAMES, components))
        record['e115_release_clock_due_rule'] = 'explicit_external_release_fixed_due_date'
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records[-1] = record
        self.factorized_component_records.append(components)
        self.equipment_work_total += components[2]
        self.physical_rehandles_total += components[3]
        self.progress_clock.observe(self.time_steps,
            sum(b.stored_time_step is not None for b in self.blocks), sum(b.delivered for b in self.blocks))
        return state, reward, done, info

    def trajectory_metrics(self):
        jobs = [Job(label=b.label, release=b.arrival_step, due=self._external_dates[b.label],
                    first_storage=b.stored_time_step, delivery=b.delivered_time_step,
                    initially_stored=b.label in self.initial_labels) for b in self.blocks]
        result = rescore(jobs, observation_end=self.time_steps,
            equipment_work=self.equipment_work_total, rehandles=self.physical_rehandles_total,
            strict_complete=all(b.delivered for b in self.blocks))
        if abs(result['J']-sum(objective_cost(x) for x in self.factorized_component_records)) > 1e-7:
            raise RuntimeError('dense external-date costs disagree with offline accounting')
        return result
