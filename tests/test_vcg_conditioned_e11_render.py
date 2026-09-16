import numpy as np

from experiments.conditioned_vcg.E11_distribution_shift_93k import (
    render_results as render,
)


def test_e11_hierarchical_interval_preserves_constant_paired_effect():
    matrix = np.full((3, 30), 2.75)
    seed_draws, instance_draws = render.bootstrap_indices(replicates=100, seed=7)
    mean, lower, upper = render.hierarchical_interval(
        matrix, seed_draws, instance_draws
    )
    assert mean == 2.75
    assert lower == 2.75
    assert upper == 2.75


def test_e11_renderer_has_seven_regimes_and_four_fixed_coordinates():
    assert len(render.REGIME_ORDER) == 7
    assert len(render.COORDINATES) == 4
    assert tuple(item[1] for item in render.COORDINATES) == (0.0, 0.05, 0.10, 0.20)
