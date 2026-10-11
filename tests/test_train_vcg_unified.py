from __future__ import annotations

import argparse
from copy import deepcopy

import pytest

import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import compare_unified_vcg as comparison


def _args(tmp_path, variant: str):
    parser = unified.build_parser()
    return parser.parse_args(
        ["--output-dir", str(tmp_path / variant), "--variant", variant, "--contract-only"]
    )


def _block(rehandles: int = 3, deliveries: int = 10):
    return [
        {"physical_rehandles": rehandles, "required_deliveries": deliveries}
        for _ in range(v23.BLOCK_EPISODES)
    ]


def test_two_contracts_are_identical_except_lambda_control(tmp_path):
    unconstrained, base_u = unified.build_unified_contract(
        _args(tmp_path, unified.VCG)
    )
    constrained, base_c = unified.build_unified_contract(
        _args(tmp_path, unified.VCG_HANDLING_CONSTRAINT)
    )
    assert base_u == base_c
    assert (
        unconstrained["shared_configuration_sha256"]
        == constrained["shared_configuration_sha256"]
    )
    assert unconstrained["shared_configuration"] == constrained["shared_configuration"]
    assert unconstrained["variant"]["lambda_control"] == "fixed_zero"
    assert constrained["variant"]["lambda_control"] == "projected_complete_block_dual_update"
    for contract in (unconstrained, constrained):
        assert contract["shared_configuration"]["q_physical_rehandle_trained"] is True
        assert contract["shared_configuration"]["agent_config"]["cost_loss_weight"] == 1.0
        assert contract["terminal_checkpoint_only"] is True


def test_fresh_seed_profile_is_exact_and_restored(tmp_path):
    before = {
        name: getattr(v23, name)
        for name in (
            "DEFAULT_MODEL_SEED",
            "FRESH_TRAIN_SEED_BASE",
            "TRAINING_POLICY_RNG_BASE",
            "REPLAY_RNG_SEED",
            "VALIDATION_POLICY_RNG_BASE",
        )
    }
    contract, _ = unified.build_unified_contract(_args(tmp_path, unified.VCG))
    shared = contract["shared_configuration"]
    assert shared["model_seed"] == 14
    assert shared["train_seed_base"] == 61_004_000
    assert shared["training_policy_rng_formula"] == "610004000 + episode_number - 1"
    assert shared["replay_rng_seed"] == 610_104_010
    assert shared["validation_policy_rng_grid"][0][0] == 621_000_000
    assert shared["validation_policy_rng_grid"][-1][-1] == 621_000_047
    assert {name: getattr(v23, name) for name in before} == before


def test_unconstrained_controller_monitors_residual_but_never_moves_lambda():
    controller = unified.UnifiedLambdaController(
        unified.UnifiedVCGVariant(unified.VCG)
    )
    update = controller.process_complete_block(
        _block(), block_number=v23.WARMUP_BLOCKS + 1, terminal=False
    )
    assert update["mean_budget_residual"] == 1.0
    assert update["lambda_proposed"] == pytest.approx(0.01)
    assert update["lambda_after"] == 0.0
    assert update["applied"] is False
    assert controller.lambda_value == 0.0
    assert controller.applied_update_count == 0


def test_constrained_controller_applies_same_projected_update():
    unified_controller = unified.UnifiedLambdaController(
        unified.UnifiedVCGVariant(unified.VCG_HANDLING_CONSTRAINT)
    )
    reference = v23.CompleteBlockProjectedDual(deepcopy(unified_controller.config))
    rows = _block()
    observed = unified_controller.process_complete_block(
        rows, block_number=v23.WARMUP_BLOCKS + 1, terminal=False
    )
    expected = reference.update_complete_block(
        rows, block_number=v23.WARMUP_BLOCKS + 1
    )
    assert observed["lambda_after"] == expected["lambda_after"]
    assert observed["mean_budget_residual"] == expected["mean_budget_residual"]
    assert observed["episode_residuals"] == expected["episode_residuals"]


def test_terminal_proposal_is_not_applied_to_either_arm():
    controller = unified.UnifiedLambdaController(
        unified.UnifiedVCGVariant(unified.VCG_HANDLING_CONSTRAINT)
    )
    controller.last_completed_block = v23.TOTAL_BLOCKS - 1
    controller.lambda_value = 0.5
    update = controller.process_complete_block(
        _block(), block_number=v23.TOTAL_BLOCKS, terminal=True
    )
    assert update["applied"] is False
    assert update["lambda_after"] == 0.5
    assert update["reason_not_applied"] == "no_subsequent_primal_block"


def test_contract_only_executes_no_runtime(tmp_path):
    args = _args(tmp_path, unified.VCG)
    summary = unified.run_unified_training(args)
    assert summary["status"] == "contract_only"
    assert summary["episodes_executed"] == 0
    assert summary["final_86xxx_panel_opened"] is False


def test_comparison_reduces_four_rng_rows_within_instance_first():
    rows = []
    for seed in range(85_000, 85_012):
        for rng_index in range(4):
            rows.append(
                {
                    "instance_seed": seed,
                    "policy_rng_index": rng_index,
                    "dense_return": float(seed - 85_000),
                    "mean_absolute_error": 15.0,
                    "physical_rehandles": 2,
                    "required_deliveries": 8,
                    "steps": 100,
                    "delivery_deviations": [-5.0] * 4 + [25.0] * 4,
                }
            )
    points = comparison._instance_points(rows)
    assert len(points) == 12
    assert points[85_000]["physical_rehandles_per_100"] == 25.0
    assert points[85_000]["within_window_percentage"] == 50.0
    assert points[85_000]["mean_earliness"] == 2.5
    assert points[85_000]["mean_tardiness"] == 12.5
