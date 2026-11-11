from experiments.conditioned_vcg.E270_e222_multiseed_prefilled_certification_329k_v1 import (
    protocol,
    scenario,
)


def test_e270_panel_shape_and_seed_mapping():
    assert scenario.INITIAL_FILLS == (18, 25, 32)
    assert scenario.CASES_PER_FILL == 10
    assert len(scenario.SEEDS) == 30
    assert [scenario.initial_fill(seed) for seed in scenario.SEEDS].count(32) == 10
    assert protocol.EXPECTED_VCG_ROWS == 180
    assert protocol.EXPECTED_DYNAMIC_ROWS == 60
    assert protocol.EXPECTED_ROWS == 240


def test_e270_layouts_are_deterministic_unique_and_duration_paired():
    for fill_index, fill in enumerate(scenario.INITIAL_FILLS):
        seed = scenario.SEED_BASE + 100 * fill_index
        first = scenario.candidate_layout(seed, fill, 0)
        assert first == scenario.candidate_layout(seed, fill, 0)
        assert len(first) == len(set(first)) == fill
    assert len({scenario.duration_seed(seed) for seed in scenario.SEEDS[::10]}) == 1


def test_e270_candidate_instance_materializes_declared_fill_and_gap():
    seed = scenario.SEEDS[0]
    record = scenario.candidate_record(seed, scenario.initial_fill(seed), 0)
    env = scenario.Prefilled8x8Env(scenario.SCENARIO, protocol.e191.SPEC, record)
    instance = env.sample_episode_instance(seed=seed)
    env.reset(instance=instance)
    assert sum(bool(block.stored) for block in env.blocks) == 18
    future = instance.arrival_steps[18:]
    assert future[0] == 1
    assert set(b - a for a, b in zip(future, future[1:])) == {40}
