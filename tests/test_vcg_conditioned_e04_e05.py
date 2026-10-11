from types import SimpleNamespace

import torch
from torch import nn

from PSLAP.viability_candidates import ViabilityActionType
from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import run as e05
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    run_future_consequence as e05c,
)


def test_e04_hash_ranking_is_reproducible_and_keyed():
    first = e04._hash_score(7, "episode", 3, "candidate-a")
    assert first == e04._hash_score(7, "episode", 3, "candidate-a")
    assert 0.0 <= first < 1.0
    assert first != e04._hash_score(8, "episode", 3, "candidate-a")
    assert first != e04._hash_score(7, "episode", 3, "candidate-b")


def test_e04_pilot_and_full_specs_are_predeclared():
    pilot = e04._run_specs(selected_seed=0, pilot=True)
    full = e04._run_specs(selected_seed=None, pilot=False)
    assert len(pilot) == 6
    assert len(full) == 18
    assert sum(spec["ranking_signal"] == e04.RANDOM_SAFE for spec in pilot) == 1
    assert sum(spec["ranking_signal"] == e04.RANDOM_SAFE for spec in full) == 5
    assert {
        spec["preference_lambda"]
        for spec in pilot
        if spec["ranking_signal"] == e04.CONDITIONED_SAFE
    } == set(e04.CONDITIONED_LAMBDAS)


def test_e04_heuristic_uses_overdue_urgency_without_a_critic():
    recovery_state = SimpleNamespace(
        block=lambda label: SimpleNamespace(remaining_time=-10.0 if label == "late" else 10.0)
    )
    snapshot = SimpleNamespace(recovery_state=recovery_state)
    common = dict(
        rank_delta=0,
        recovery_action=None,
        horizon_steps=None,
        source=(1, 1),
        destination=(1, 2),
        action_type=ViabilityActionType.DELIVER,
    )
    late = SimpleNamespace(target_label="late", **common)
    early = SimpleNamespace(target_label="early", **common)
    assert e04._heuristic_score(snapshot, late) > e04._heuristic_score(snapshot, early)


class _PreferenceEcho(nn.Module):
    def forward(self, features, preference_lambda):
        values = torch.as_tensor(
            preference_lambda, dtype=features.dtype, device=features.device
        )
        if values.ndim == 0:
            values = values.expand(features.shape[0])
        return values


def test_e05_clamp_changes_only_network_input_and_has_exact_midpoint_identity():
    features = torch.zeros((3, 2))
    original = _PreferenceEcho()
    clamped = e05.ClampedPreferenceNetwork(original, 0.10)
    assert torch.equal(clamped(features, 0.05), torch.full((3,), 0.10))
    assert torch.equal(clamped(features, 0.20), torch.full((3,), 0.10))
    assert torch.equal(clamped(features, 0.10), original(features, 0.10))


def test_e05a_uses_final_three_seed_diagnostics():
    validation = e05._assemble_validation()
    assert [validation["seeds"][str(seed)]["mc_validation_sample_count"] for seed in (0, 1, 2)] == [218, 199, 199]
    assert validation["seeds"]["1"]["source"] == "seed_1_final_continuation_round_10"
    assert all(
        item["qn_direction_reversal_fraction"] == 0.0
        and item["lambda_qn_nondecreasing_fraction"] == 1.0
        for item in validation["seeds"].values()
    )


def test_e05c_zero_future_network_preserves_only_immediate_agent_term():
    features = torch.randn(4, 7)
    network = e05c.ZeroFutureHandlingNetwork()
    assert torch.equal(network(features, 0.05), torch.zeros(4))
    assert torch.equal(network(features, 0.20), torch.zeros(4))


def test_e05c_full_grid_reuses_one_control_per_new_row():
    expected = (
        len(e05c.MODEL_SEEDS)
        * len(e05c.DEPLOYMENT_LAMBDAS)
        * len(e05c.INSTANCE_SEEDS)
    )
    assert expected == 270
    assert e05c.FUTURE_INPUT_LAMBDA == 0.10
