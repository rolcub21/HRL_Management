import diagnose_vcg_v11_conditioned_handling_fixed_bank as diagnostic


def test_direction_reversals_ignores_flat_segments():
    assert diagnostic._direction_reversals((3.0, 2.0, 2.0, 1.0)) == 0
    assert diagnostic._direction_reversals((3.0, 2.0, 2.5, 2.0)) == 2


def test_sequence_summary_separates_smooth_change_and_policy_crossing():
    sequences = {
        ("state-000", 0): {
            "qn": [3.0, 2.0, 1.0],
            "penalty": [0.0, 0.2, 0.2],
        },
        ("state-000", 1): {
            "qn": [1.0, 1.1, 1.0],
            "penalty": [0.0, 0.11, 0.2],
        },
    }
    result = diagnostic.summarize_sequences(
        sequences,
        {"state-000": ["candidate-a", "candidate-b", "candidate-b"]},
    )
    assert result["candidate_count"] == 2
    assert result["qn_endpoint_decrease_fraction"] == 0.5
    assert result["qn_direction_reversal_fraction"] == 0.5
    assert result["states_with_policy_switch_fraction"] == 1.0
