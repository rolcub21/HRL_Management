from experiments.conditioned_vcg.E270_e222_multiseed_prefilled_certification_329k_v1 import (
    protocol,
    scenario,
    tracker,
)
import inspect


def test_e270_panel_shape_and_seed_mapping():
    assert scenario.INITIAL_FILLS == (18, 25, 32)
    assert scenario.CASES_PER_FILL == 10
    assert len(scenario.SEEDS) == 30
    assert [scenario.initial_fill(seed) for seed in scenario.SEEDS].count(32) == 10
    assert protocol.EXPECTED_VCG_ROWS == 180
    assert protocol.EXPECTED_DYNAMIC_ROWS == 60
    assert protocol.EXPECTED_A3C_ROWS == 780
    assert protocol.EXPECTED_ROWS == 1020


def test_e270_includes_paired_a3c_at_both_sizes():
    assert protocol.A3C_SIZES == ("5x5", "8x8")
    assert protocol.A3C_ARMS == (
        "a3c_certified", "a3c_physical_unfiltered",
    )
    assert "a3c_physical_unfiltered" in protocol.UNFILTERED_ARMS
    for size in protocol.A3C_SIZES:
        for seed in protocol.MODEL_SEEDS:
            identity = protocol.a3c_checkpoint_identity(size, seed)
            assert identity["protocol_training_complete"] is True
            assert identity["size"] == size
            assert identity["model_seed"] == seed


def test_e270_tracker_is_self_contained_not_an_e233_import():
    source = inspect.getsource(tracker)
    assert "E233_certification_causal_ablation_292k_v1" not in source
    assert tracker.TrackedE191Agent.__module__ == tracker.__name__


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
