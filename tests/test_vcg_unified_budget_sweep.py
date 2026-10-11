from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path

import pytest

import compare_vcg_unified_budget_sweep as comparison
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import train_vcg_unified_budget_sweep as sweep


def _args(tmp_path, arm: str, *, contract_only: bool = True):
    values = ["--arm", arm, "--output-dir", str(tmp_path / arm), "--device", "cpu"]
    if contract_only:
        values.append("--contract-only")
    return sweep.build_parser().parse_args(values)


def test_budget_grid_and_clock_are_exact_and_context_restores():
    sweep.validate_sweep()
    before = (v23.TOTAL_EPISODES, v23.TOTAL_BLOCKS, unified.METHOD_VERSION)
    with sweep.activated_sweep_protocol():
        assert (v23.TOTAL_EPISODES, v23.TOTAL_BLOCKS) == (300, 30)
        assert unified.METHOD_VERSION == sweep.METHOD_VERSION
        assert v23.schedule_for_episode(300).block_number == 30
        assert v23.schedule_for_episode(300).within_group_temperatures == (0.01,) * 4
    assert (v23.TOTAL_EPISODES, v23.TOTAL_BLOCKS, unified.METHOD_VERSION) == before


def test_custom_budget_changes_only_dual_contract(tmp_path):
    common = []
    for arm_name in sweep.ARM_ORDER:
        arm = sweep.ARMS[arm_name]
        with sweep.activated_sweep_protocol():
            args = sweep._unified_args(_args(tmp_path, arm_name), arm)
            contract, _ = unified.build_unified_contract(args)
        base = comparison._common_base_configuration(contract)
        common.append(base)
        assert contract["base_v2_3_runtime_contract"]["episodes"] == 300
        assert contract["base_v2_3_runtime_contract"]["dual"][
            "budget_per_100_required_deliveries"
        ] == arm.budget_per_100
    assert all(item == common[0] for item in common[1:])


def test_custom_budget_controller_uses_declared_residual():
    variant = unified.UnifiedVCGVariant(
        unified.VCG_HANDLING_CONSTRAINT, handling_budget_per_100=5.0
    )
    controller = unified.UnifiedLambdaController(variant)
    rows = [
        {"physical_rehandles": 1, "required_deliveries": 8}
        for _ in range(v23.BLOCK_EPISODES)
    ]
    update = controller.process_complete_block(
        rows, block_number=v23.WARMUP_BLOCKS + 1, terminal=False
    )
    assert update["mean_budget_residual"] == pytest.approx(0.6)
    assert update["lambda_after"] == pytest.approx(0.006)


def test_contract_only_runs_no_episode_and_records_arm(tmp_path):
    result = sweep.run(_args(tmp_path, sweep.BUDGET_8))
    assert result["status"] == "contract_only"
    assert result["episodes_executed"] == 0
    assert result["sweep_arm"] == sweep.ARMS[sweep.BUDGET_8].to_dict()
    assert result["fixed_training_horizon"] == 300
    assert result["final_86xxx_panel_opened"] is False


def test_fake_300_episode_lifecycle_uses_terminal_checkpoint_only(tmp_path):
    source = Path(__file__).with_name("test_train_vcg_constrained_v2_3_protocol.py")
    spec = importlib.util.spec_from_file_location("_v23_protocol_test_support", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_runtime = module._FakeRuntime

    arm = sweep.ARMS[sweep.BUDGET_8]
    wrapper_args = _args(tmp_path, sweep.BUDGET_8, contract_only=False)
    with sweep.activated_sweep_protocol():
        args = sweep._unified_args(wrapper_args, arm)
        runtime = fake_runtime(args)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = unified.run_unified_training(args, runtime=runtime)
    assert summary["completed_training_episodes"] == 300
    assert summary["lambda_state"]["observed_block_count"] == 28
    assert summary["lambda_state"]["applied_update_count"] == 27
    assert summary["final_validation"]["checkpoint_episode"] == 300
    assert len(list((wrapper_args.output_dir / "validation-ledger").glob("*.json"))) == 1


def test_kkt_classification_is_predeclared_and_finite():
    assert comparison._kkt_classification(0.0, -4.0) == "inactive"
    assert comparison._kkt_classification(0.1, 1.0) == "active"
    assert comparison._kkt_classification(0.1, -8.0) == "unconverged"
    assert math.isfinite(comparison.COMPLEMENTARITY_TOLERANCE)


def test_fake_four_arm_report_builds_without_new_panel(tmp_path, monkeypatch):
    source = Path(__file__).with_name("test_train_vcg_constrained_v2_3_protocol.py")
    spec = importlib.util.spec_from_file_location("_v23_sweep_report_support", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeRuntimeWithDeviations(module._FakeRuntime):
        def run_episode(self, **kwargs):
            row = super().run_episode(**kwargs)
            row["delivery_deviations"] = [-3.0] * 8
            row["steps"] = 100
            return row

    monkeypatch.setattr(
        v23,
        "load_default_runtime",
        lambda args, contract: FakeRuntimeWithDeviations(args),
    )

    root = tmp_path / "sweep"
    with contextlib.redirect_stdout(io.StringIO()):
        for name in sweep.ARM_ORDER:
            arm = sweep.ARMS[name]
            args = sweep.build_parser().parse_args(
                ["--arm", name, "--output-dir", str(root / arm.output_name), "--device", "cpu"]
            )
            sweep.run(args)
    report = comparison.analyze(root)
    assert report["status"] == "complete"
    assert len(report["table"]) == 4
    assert report["all_terminal_rows_strict_safe_complete"] is True
    assert report["new_final_panel_opened"] is False
