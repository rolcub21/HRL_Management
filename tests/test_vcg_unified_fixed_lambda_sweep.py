from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

import compare_vcg_unified_fixed_lambda_sweep as comparison
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import train_vcg_unified_budget_sweep as budget_sweep
import train_vcg_unified_fixed_lambda_sweep as sweep


def _fake_runtime_class():
    source = Path(__file__).with_name("test_train_vcg_constrained_v2_3_protocol.py")
    spec = importlib.util.spec_from_file_location("_v23_fixed_sweep_support", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeRuntime(module._FakeRuntime):
        def run_episode(self, **kwargs):
            row = super().run_episode(**kwargs)
            row["delivery_deviations"] = [-3.0] * 8
            row["steps"] = 100
            return row

    return FakeRuntime


def _args(tmp_path, fixed_lambda: float, *, contract_only: bool = False):
    values = [
        "--fixed-lambda",
        str(fixed_lambda),
        "--output-dir",
        str(tmp_path / sweep.lambda_key(fixed_lambda)),
        "--device",
        "cpu",
    ]
    if contract_only:
        values.append("--contract-only")
    return sweep.build_parser().parse_args(values)


def test_fixed_variant_requires_a_positive_declared_lambda():
    with pytest.raises(unified.UnifiedVCGError):
        unified.UnifiedVCGVariant(unified.VCG_FIXED_HANDLING_WEIGHT)
    variant = unified.UnifiedVCGVariant(
        unified.VCG_FIXED_HANDLING_WEIGHT, fixed_lambda=0.1
    )
    assert variant.fixed_weight is True
    assert variant.to_dict()["lambda_control"] == "fixed_positive_after_warmup"


def test_fixed_controller_never_applies_dual_update():
    variant = unified.UnifiedVCGVariant(
        unified.VCG_FIXED_HANDLING_WEIGHT,
        handling_budget_per_100=0.0,
        fixed_lambda=0.2,
    )
    controller = unified.UnifiedLambdaController(variant)
    rows = [
        {"physical_rehandles": 2, "required_deliveries": 8}
        for _ in range(v23.BLOCK_EPISODES)
    ]
    update = controller.process_complete_block(
        rows, block_number=v23.WARMUP_BLOCKS + 1, terminal=False
    )
    assert update["lambda_before"] == 0.2
    assert update["lambda_after"] == 0.2
    assert update["applied"] is False
    assert update["reason_not_applied"] == "fixed_handling_weight"


def test_fixed_protocol_context_restores_modules():
    before = (v23.TOTAL_EPISODES, v23.TOTAL_BLOCKS, unified.METHOD_VERSION)
    with sweep.activated_fixed_lambda_protocol():
        assert (v23.TOTAL_EPISODES, v23.TOTAL_BLOCKS) == (300, 30)
        assert unified.METHOD_VERSION == sweep.METHOD_VERSION
    assert (v23.TOTAL_EPISODES, v23.TOTAL_BLOCKS, unified.METHOD_VERSION) == before


def test_fake_fixed_arm_uses_zero_warmup_then_constant_lambda(tmp_path, monkeypatch):
    fake_runtime = _fake_runtime_class()
    observed = []

    def factory(args, contract):
        runtime = fake_runtime(args)
        observed.append(runtime)
        return runtime

    monkeypatch.setattr(v23, "load_default_runtime", factory)
    with contextlib.redirect_stdout(io.StringIO()):
        result = sweep.run(_args(tmp_path, 0.1))
    events = [event for event in observed[0].events if event.get("training") is True]
    assert len(events) == 300
    assert all(event["lambda"] == 0.0 for event in events[:20])
    assert all(event["lambda"] == 0.1 for event in events[20:])
    assert result["lambda_state"]["lambda_value"] == 0.1
    assert result["lambda_state"]["applied_update_count"] == 0


def test_fake_five_point_report_builds(tmp_path, monkeypatch):
    fake_runtime = _fake_runtime_class()
    monkeypatch.setattr(
        v23, "load_default_runtime", lambda args, contract: fake_runtime(args)
    )
    parent = tmp_path / "parent-lambda0"
    parent_args = budget_sweep.build_parser().parse_args(
        ["--arm", "lambda0", "--output-dir", str(parent), "--device", "cpu"]
    )
    root = tmp_path / "fixed"
    with contextlib.redirect_stdout(io.StringIO()):
        budget_sweep.run(parent_args)
        for arm in sweep.ARMS:
            sweep.run(
                sweep.build_parser().parse_args(
                    [
                        "--fixed-lambda",
                        str(arm.fixed_lambda),
                        "--output-dir",
                        str(root / arm.name),
                        "--device",
                        "cpu",
                    ]
                )
            )
    report = comparison.analyze(root, parent)
    assert report["status"] == "complete"
    assert len(report["table"]) == 5
    assert report["lambda_values"] == (0.0, 0.05, 0.1, 0.2, 0.3)
    assert report["all_terminal_rows_strict_safe_complete"] is True
    assert report["new_final_panel_opened"] is False

