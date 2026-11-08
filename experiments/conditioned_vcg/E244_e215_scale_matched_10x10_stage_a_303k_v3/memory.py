"""Exact graph-input reuse with a bounded 10x10 device allocation."""

from __future__ import annotations

from contextlib import contextmanager
import gc
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)


IMPLEMENTATION = "e244_10x10_exact_bounded_device_graph_input_cache_v1"
INPUT_CAPACITY = 500_000
INPUT_CHUNK = 16_384


def cache_stats() -> dict:
    """Return allocation telemetry without retaining cache objects."""

    entries = {}
    for key, cache in fast_inputs._REGISTRY.items():
        dynamic = cache.dynamic
        entries[f"{key[0]}|{key[1]}|{key[2]}"] = {
            **dict(cache.stats),
            "entries": int(cache.size),
            "capacity": int(cache.capacity),
            "chunk": int(cache.chunk),
            "allocated_slots": int(dynamic.shape[0]),
            "allocated_bytes": int(dynamic.numel() * dynamic.element_size()),
        }
    return {
        "implementation": IMPLEMENTATION,
        "exact_inputs": True,
        "capacity": INPUT_CAPACITY,
        "chunk": INPUT_CHUNK,
        "caches": entries,
    }


@contextmanager
def bounded_cache_session(
    *, capacity: int = INPUT_CAPACITY, chunk: int = INPUT_CHUNK
):
    """Bound E100's exact cache while preserving every encoded tensor value."""

    capacity = int(capacity)
    chunk = int(chunk)
    if capacity < 1 or chunk < 1 or chunk > capacity:
        raise ValueError("invalid bounded graph-input cache geometry")
    original = fast_inputs._cache_for
    fast_inputs._REGISTRY.clear()

    def bounded(network):
        cache = original(network)
        cache.capacity = capacity
        cache.chunk = chunk
        return cache

    try:
        with patch.object(fast_inputs, "_cache_for", bounded):
            yield
    finally:
        fast_inputs._REGISTRY.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "IMPLEMENTATION", "INPUT_CAPACITY", "INPUT_CHUNK",
    "bounded_cache_session", "cache_stats",
]
