"""Value-preserving state sharing and bounded, arm-lifetime input reuse.

Only immutable graph inputs are cached. Online embeddings/scores are never
cached; frozen-target scores retain the original target-generation lifecycle.
"""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import fast_inputs
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import ProgressAgent, ProgressObservedState

IMPLEMENTATION = "e123_v2_shared_progress_state_persistent_bounded_inputs_v1"
INPUT_CAPACITY = 1_000_000


def share_records(records, pool=None):
    """Intern complete observations, including queue, progress and block clocks."""
    pool = {} if pool is None else pool
    result = []
    for record in records:
        state = record.current_state
        if not isinstance(state, ProgressObservedState):
            raise TypeError("E123-v2 cannot share a clock-unobserved replay state")
        shared = pool.setdefault(state, state)
        result.append(record if shared is state else replace(record, current_state=shared))
    return tuple(result)


def compact_transitions(transitions):
    pool = {}
    result = []
    original_ids = set()
    for transition in transitions:
        records = (transition.chosen, *transition.next_candidates)
        original_ids.update(id(record.current_state) for record in records)
        records = share_records(records, pool)
        replacement = replace(transition, chosen=records[0], next_candidates=records[1:])
        if replacement != transition:
            raise RuntimeError("E123-v2 state sharing changed a replay value")
        result.append(replacement)
    return result, {"transitions": len(result), "original_current_state_objects": len(original_ids),
                    "shared_current_state_objects": len(pool), "all_transition_values_equal": True}


class SharedProgressAgent(ProgressAgent):
    def _admissible_prepared(self, snapshot):
        prepared, forced = super()._admissible_prepared(snapshot)
        return replace(prepared, records=share_records(prepared.records)), forced

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(), "e123_execution_implementation": IMPLEMENTATION,
                "graph_input_cache_capacity": INPUT_CAPACITY}


@contextmanager
def cache_session():
    """One bounded cache across episodes/validation, released at arm exit."""
    original = fast_inputs._cache_for
    fast_inputs._REGISTRY.clear()

    def bounded(network):
        cache = original(network)
        cache.capacity = INPUT_CAPACITY
        return cache

    try:
        with patch.object(fast_inputs, "_cache_for", bounded):
            yield
    finally:
        fast_inputs._REGISTRY.clear()
