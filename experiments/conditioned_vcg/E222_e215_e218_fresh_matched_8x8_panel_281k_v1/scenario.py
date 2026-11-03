"""Previously unused, paired 8x8 test instances for E222."""

from __future__ import annotations

from dataclasses import replace

from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    scenario as e191,
)


SCENARIO = e191.SCENARIO
LEVELS = e191.LEVELS
GAPS = e191.GAPS
TOTAL_JOBS = e191.TOTAL_JOBS
CAPACITY = e191.CAPACITY
MEAN_DURATION = e191.MEAN_DURATION
MAX_STEPS = e191.MAX_STEPS
CASES_PER_LEVEL = 10
TEST_BASE = 281_222_000  # divisible by three; seed % 3 identifies workload.
DURATION_BASE = 281_222_500

LEVEL_BY_SEED = {
    TEST_BASE + level_index + 3 * replicate: level
    for level_index, level in enumerate(LEVELS)
    for replicate in range(CASES_PER_LEVEL)
}
TEST_SEEDS = tuple(LEVEL_BY_SEED)


def level_for_seed(seed: int) -> float:
    try:
        return float(LEVEL_BY_SEED[int(seed)])
    except KeyError as error:
        raise ValueError(f"seed is outside the E222 test panel: {seed}") from error


def replicate_for_seed(seed: int) -> int:
    value = int(seed)
    level_for_seed(value)
    return (value - TEST_BASE) // 3


def duration_seed(seed: int) -> int:
    return DURATION_BASE + replicate_for_seed(seed)


def level_label(level: float) -> str:
    return f"{100.0 * float(level):.0f}%"


class TestPanel8x8Env(e191.LongStayCrowdingEnv):
    """E191 environment with a new immutable test split."""

    def __init__(self, name, objective):
        if name != SCENARIO:
            raise ValueError(f"unknown E222 scenario: {name}")
        super().__init__(SCENARIO, objective)

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("E222 requires an explicit test seed")
        seed = int(seed)
        level = level_for_seed(seed)
        gap = int(GAPS[level])
        self.arrival_rate = 1.0 / float(gap)
        self.proc_mean = float(MEAN_DURATION)
        native = SmallRoomsEnv.sample_episode_instance(
            self, seed=duration_seed(seed)
        )
        instance = replace(
            native,
            seed=seed,
            arrival_rate=float(self.arrival_rate),
            proc_mean=float(MEAN_DURATION),
            arrival_steps=tuple(gap * index for index in range(TOTAL_JOBS)),
        )
        instance.validate_for(self)
        return instance


__all__ = [
    "CAPACITY", "CASES_PER_LEVEL", "DURATION_BASE", "GAPS", "LEVELS",
    "LEVEL_BY_SEED", "MAX_STEPS", "MEAN_DURATION", "SCENARIO",
    "TEST_BASE", "TEST_SEEDS", "TOTAL_JOBS", "TestPanel8x8Env",
    "duration_seed", "level_for_seed", "level_label", "replicate_for_seed",
]
