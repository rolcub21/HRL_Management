import pytest
import torch

from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    analyze_offline,
)
from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    continuation_bank,
)
from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    decompose_signal,
)
from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    component_rollout_ablation,
)


def _row(index, key, qop, future):
    return {
        "candidate_index": index,
        "candidate_key": key,
        "action_type": "deliver",
        "mode": "recover",
        "qop": qop,
        "qn_immediate": 0,
        "qn_future": future,
    }


def test_future_signal_can_cross_a_candidate_relative_margin():
    rows = (
        _row(0, "deliver:A", 1.0, 3.0),
        _row(1, "deliver:B", 0.9, 0.0),
    )

    result = analyze_offline._frontier_record(
        rows, seed=0, state_id="fixture", value=0.1
    )

    assert result["immediate_selected_key"] == "deliver:A"
    assert result["future_selected_key"] == "deliver:B"
    assert result["selected_candidate_changed"] is True
    assert result["selected_action_type_changed"] is False
    assert result["selected_mode_changed"] is False
    assert result["selected_predicted_future_saving"] == 3.0
    assert result["same_mode_crossing_surplus"] > 0.0


def test_candidate_common_future_shift_does_not_change_ranking():
    rows = (
        _row(0, "deliver:A", 1.0, 2.0),
        _row(1, "deliver:B", 0.9, 2.0),
    )

    result = analyze_offline._frontier_record(
        rows, seed=0, state_id="fixture", value=0.2
    )

    assert result["selected_candidate_changed"] is False
    assert result["predicted_future_range"] == 0.0
    assert result["within_mode_order_inversions"] == 0


def _branch(predicted, future, total, *, complete=True, mae=10.0, steps=100):
    return {
        "predicted_future_rehandles": predicted,
        "realized_future_rehandles_after_forced_macro": future,
        "realized_total_rehandles_from_state": total,
        "strict_safe_complete": complete,
        "mean_absolute_error": mae if complete else None,
        "steps_after_capture": steps,
    }


def test_continuation_pair_uses_post_current_macro_target():
    result = continuation_bank._pair_summary(
        {
            "immediate_only_winner": _branch(4.0, 3, 4),
            "fixed_future_winner": _branch(1.5, 1, 1),
        }
    )
    assert result["predicted_immediate_minus_future_candidate"] == 2.5
    assert result["realized_future_rehandles_immediate_minus_future_candidate"] == 2
    assert result["future_prediction_order_correct"] is True
    assert result["future_minus_immediate_total_rehandles"] == -3


def test_continuation_pair_does_not_call_realized_tie_correct():
    result = continuation_bank._pair_summary(
        {
            "immediate_only_winner": _branch(3.0, 2, 2),
            "fixed_future_winner": _branch(1.0, 2, 2),
        }
    )
    assert result["realized_future_rehandles_tied"] is True
    assert result["future_prediction_order_correct"] is False


def test_continuation_pair_suppresses_quality_delta_after_failure():
    result = continuation_bank._pair_summary(
        {
            "immediate_only_winner": _branch(3.0, 2, 2),
            "fixed_future_winner": _branch(
                1.0, 1, 1, complete=False, mae=None, steps=50
            ),
        }
    )
    assert result["both_branches_strict_safe_complete"] is False
    assert result["future_minus_immediate_mae"] is None
    assert result["future_minus_immediate_steps"] is None


def test_signed_timing_decomposition_reconstructs_mae():
    summary = decompose_signal._timing_summary((-10.0, 0.0, 6.0))
    assert summary["mean_earliness"] == 10.0 / 3.0
    assert summary["mean_tardiness"] == 2.0
    assert summary["mean_absolute_error"] == 16.0 / 3.0
    assert summary["mean_earliness"] + summary["mean_tardiness"] == pytest.approx(
        summary["mean_absolute_error"]
    )


def test_rollout_component_decomposition_reconstructs_full_prediction():
    future = torch.tensor([1.0, 3.0, 10.0])
    action_types = ("accept", "accept", "deliver")
    action_type = component_rollout_ablation.component_future_values(
        future, action_types, component_rollout_ablation.ACTION_TYPE_ONLY
    )
    residual = component_rollout_ablation.component_future_values(
        future, action_types, component_rollout_ablation.CANDIDATE_RESIDUAL_ONLY
    )
    state_mean = future.mean()

    assert action_type.tolist() == pytest.approx([2.0, 2.0, 10.0])
    assert residual.tolist() == pytest.approx(
        [state_mean.item() - 1.0, state_mean.item() + 1.0, state_mean.item()]
    )
    assert action_type + residual - state_mean == pytest.approx(future)


def test_rollout_immediate_and_full_component_endpoints():
    future = torch.tensor([0.5, 4.0])
    kinds = ("accept", "reconfigure")

    assert component_rollout_ablation.component_future_values(
        future, kinds, component_rollout_ablation.IMMEDIATE_ONLY
    ).tolist() == [0.0, 0.0]
    assert component_rollout_ablation.component_future_values(
        future, kinds, component_rollout_ablation.FULL_FUTURE
    ) is future
