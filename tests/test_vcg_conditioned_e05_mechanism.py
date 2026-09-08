from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    diagnose_conditioning_mechanism as mechanism,
)


def _candidate(index, mode, full, *, future, qop=1.0):
    return {
        "candidate_index": str(index),
        "candidate_key": f"candidate-{index}",
        "mode": mode,
        "merit": str(full),
        "qn_future": str(future),
        "qn_immediate": "0",
        "qop": str(qop),
    }


def test_common_prediction_shift_does_not_change_relative_order():
    rows = [
        _candidate(0, "accept", 1.0, future=0.8),
        _candidate(1, "accept", 0.9, future=0.7),
    ]
    full = [1.0, 0.9]
    clamped = [0.8, 0.7]
    inversions, pairs = mechanism._within_mode_inversions(full, clamped, rows)
    assert pairs == 1
    assert inversions == 0
    assert mechanism._selector(full, rows).selected_index == 0
    assert mechanism._selector(clamped, rows).selected_index == 0


def test_candidate_relative_response_can_cross_selection_margin():
    rows = [
        _candidate(0, "accept", 1.0, future=0.8),
        _candidate(1, "accept", 0.9, future=0.7),
    ]
    full = [1.0, 0.9]
    clamped = [0.8, 1.1]
    inversions, pairs = mechanism._within_mode_inversions(full, clamped, rows)
    assert pairs == 1
    assert inversions == 1
    assert mechanism._selector(full, rows).selected_index == 0
    assert mechanism._selector(clamped, rows).selected_index == 1
