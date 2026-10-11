from __future__ import annotations

import run_kim2020_final86_comparison as subject


def test_exact_450_row_grid_and_fresh_rng_namespace():
    rows = subject.expected_grid()
    assert len(rows) == 450
    assert len({row["policy_rng_seed"] for row in rows}) == 450
    assert {row["instance_seed"] for row in rows} == set(range(86_000, 86_030))
    assert rows[0]["policy_rng_seed"] == 633_000_000
    assert rows[-1]["policy_rng_seed"] == 633_002_294
    assert not any(622_000_000 <= row["policy_rng_seed"] < 623_000_000 for row in rows)


def test_primary_dominance_is_two_dimensional():
    assert subject._point_dominates(
        {
            "mean_absolute_error": 10.0,
            "physical_rehandles_per_100_required_deliveries": 5.0,
        },
        {
            "mean_absolute_error": 11.0,
            "physical_rehandles_per_100_required_deliveries": 5.0,
        },
    )
    assert not subject._point_dominates(
        {
            "mean_absolute_error": 10.0,
            "physical_rehandles_per_100_required_deliveries": 6.0,
        },
        {
            "mean_absolute_error": 11.0,
            "physical_rehandles_per_100_required_deliveries": 5.0,
        },
    )

