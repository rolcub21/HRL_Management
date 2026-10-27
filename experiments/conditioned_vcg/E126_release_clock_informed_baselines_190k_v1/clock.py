"""Scoped planning-only clock substitution; no shared source file is changed."""
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import importlib

from PSLAP.dynamic_yard import BlockView
import PSLAP.ga_policy as ga
import PSLAP.ga_capacity_aware as capacity_ga
import example.urgency_scheduler as scheduler
import example.Options.selector_v5 as selector
from experiments.conditioned_vcg.E116_e115_matched_operator_baselines_180k_v1 import run as baseline

accept = importlib.import_module('example.Options.AcceptStoreOption')


def remaining(env, block, *, after_steps=0):
    """Use only arrived-job metadata; a projection may age already-known jobs."""
    if block.arrival_step > env.time_steps or block.delivered:
        raise ValueError('Release clock requested for an unobserved or finished job')
    value = float(block.arrival_step + block.storage_steps_needed - env.time_steps - after_steps)
    if hasattr(env, 'e126_clock_calls'):
        env.e126_clock_calls['release_remaining'] += 1
    return value


def projected_plan(self, inbound, estimate, target_label):
    """Original one-admission lookahead, with the newborn block's release slack."""
    current = scheduler.retrieval_planning_context(
        self.env, relocation_selector=self.relocation_selector).yard
    aged = tuple(replace(b, remaining_time=float(b.remaining_time)-estimate.total_steps)
                 for b in current.blocks)
    newborn = BlockView(label=inbound.label, position=estimate.preview_storage_cell,
                        remaining_time=remaining(self.env, inbound, after_steps=estimate.total_steps))
    if hasattr(self.env, 'e126_clock_calls'):
        self.env.e126_clock_calls['projected_admission'] += 1
    projected = replace(current, blocks=aged+(newborn,))
    return scheduler.plan_retrieval(projected, estimate.preview_storage_cell, target_label,
                                    relocation_selector=self.relocation_selector)


def planning_view(env, kind, *args, **kwargs):
    """Replace only the time field of an otherwise identical immutable view."""
    view = BlockView(*args, **kwargs)
    block = next(b for b in env.blocks if b.label == view.label)
    value = remaining(env, block)
    if hasattr(env, 'e126_clock_calls'):
        env.e126_clock_calls[kind] += 1
        if len(env.e126_clock_samples) < 12:
            env.e126_clock_samples.append(dict(kind=kind, label=block.label,
                now=env.time_steps, release=block.arrival_step, duration=block.storage_steps_needed,
                old_remaining=view.remaining_time, release_remaining=value,
                stored_time_step=block.stored_time_step))
    return replace(view, remaining_time=value)


@contextmanager
def enabled():
    native_env = baseline.TracedEmptyStartEnv
    active = []

    class ReleaseClockEnv(native_env):
        def __init__(self, scenario):
            super().__init__(scenario)
            active[:] = [self]

        def reset(self, instance=None):
            self.e126_clock_calls = Counter()
            self.e126_clock_samples = []
            state = super().reset(instance=instance)
            self.e126_initial_schedule = [(b.label,b.arrival_step,b.storage_steps_needed) for b in self.blocks]
            return state

        def signed_remaining_storage_time(self, block):
            if block.arrival_step > self.time_steps or block.delivered:
                return None
            value = remaining(self, block)
            self.e126_clock_calls['stored_snapshot'] += 1
            return value

    def factory(kind):
        def make(*args, **kwargs):
            if len(active) != 1:
                raise RuntimeError('No unique active baseline environment')
            return planning_view(active[0], kind, *args, **kwargs)
        return make

    with ExitStack() as stack:
        from unittest.mock import patch
        stack.enter_context(patch.object(baseline,'TracedEmptyStartEnv',ReleaseClockEnv))
        for module,kind in ((selector,'assignment_view'),(accept,'accept_mask_view'),
                            (ga,'ga_pending_view'),(capacity_ga,'ga_pending_view')):
            stack.enter_context(patch.object(module,'BlockView',factory(kind)))
        stack.enter_context(patch.object(scheduler.DurationAwareAtomicScheduler,'_projected_plan',projected_plan))
        yield ReleaseClockEnv
