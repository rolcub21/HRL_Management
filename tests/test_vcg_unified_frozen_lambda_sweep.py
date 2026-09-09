from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path

import evaluate_vcg_unified_frozen_lambda_sweep as evaluation
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified_budget_sweep as budget_sweep


def _fake_runtime_class():
    source = Path(__file__).with_name("test_train_vcg_constrained_v2_3_protocol.py")
    spec = importlib.util.spec_from_file_location("_v23_frozen_sweep_support", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeRuntime(module._FakeRuntime):
        def __init__(self, args):
            super().__init__(args)
            self.agent = self._checkpoint_agent

        def run_episode(self, **kwargs):
            row = super().run_episode(**kwargs)
            row["delivery_deviations"] = [-3.0] * 8
            row["steps"] = 100
            return row

        @staticmethod
        def _module_signature(module):
            digest = hashlib.sha256()
            for name, tensor in module.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
            return digest.hexdigest()

    return FakeRuntime


def _parent(tmp_path, monkeypatch):
    fake_runtime = _fake_runtime_class()
    captured = []

    def factory(args, contract):
        captured.append(args)
        return fake_runtime(args)

    monkeypatch.setattr(v23, "load_default_runtime", factory)
    root = tmp_path / "parent"
    args = budget_sweep.build_parser().parse_args(
        ["--arm", "lambda0", "--output-dir", str(root), "--device", "cpu"]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        budget_sweep.run(args)
    return root, fake_runtime, captured[-1]


def test_prepare_pins_parent_without_evaluation(tmp_path, monkeypatch):
    parent, _, _ = _parent(tmp_path, monkeypatch)
    output = tmp_path / "evaluation"
    contract = evaluation.prepare(output, parent, device="cpu")
    assert contract["expected_total_rows"] == 240
    assert contract["training_or_learning"] is False
    assert contract["final_86xxx_panel_opened"] is False
    assert not (output / "validation-ledger").exists()


def test_fake_frozen_weight_sweep_builds_all_five_points(tmp_path, monkeypatch):
    parent, fake_runtime, parent_args = _parent(tmp_path, monkeypatch)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        evaluation, "_runtime", lambda parent, device: fake_runtime(parent_args)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluation.evaluate(output, parent, device="cpu")
        report = evaluation.analyze(output, parent, device="cpu")
    assert result["weights_unchanged"] is True
    assert sum(result["completed_rows"].values()) == 240
    assert len(report["table"]) == 5
    assert report["same_frozen_weights_for_all_rows"] is True
    assert report["all_240_rows_strict_safe_complete"] is True
    assert report["final_86xxx_panel_opened"] is False

