import importlib.util
from pathlib import Path

import render_vcg_unified_policy_filmstrip as renderer


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = (
    ROOT
    / "experiments/conditioned_vcg/E01_benchmark_90k"
    / "trace_current_method_illustrations.py"
)


def _trace_module():
    spec = importlib.util.spec_from_file_location("e1_current_trace", TRACE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_renderer_accepts_legacy_and_current_trace_arm_names():
    assert renderer._arm_keys({"protocol": renderer.LEGACY_TRACE_PROTOCOL}) == (
        "lambda0",
        "lambda005",
    )
    assert renderer._arm_keys({"protocol": renderer.CONDITIONED_TRACE_PROTOCOL}) == (
        "lambda0",
        "lambda_positive",
    )


def test_current_illustrations_bind_the_verified_final_method_forks():
    trace = _trace_module()
    assert trace.MODEL_SEED == 0
    assert trace.INSTANCE_SEED == 90_001
    assert trace.MACRO_LIMIT == 5
    storage, rehandle = trace.SCENARIOS
    assert storage["fork_decision_index"] == 0
    assert storage["expected"] == {
        "lambda0": "accept:B1:3:1",
        "lambda_positive": "accept:B1:1:1",
    }
    assert storage["selected_macro_indices_1_based"] == {
        "lambda0": (1, 11, 12, 14, 22),
        "lambda_positive": (1, 8, 11, 16, 19),
    }
    assert storage["aligned_events"] == (
        ("accept", "B1"),
        ("accept", "B8"),
        ("deliver", "B5"),
        ("deliver", "B6"),
        ("deliver", "B8"),
    )
    assert all(
        len(indices) == trace.MACRO_LIMIT
        for indices in storage["selected_macro_indices_1_based"].values()
    )
    assert rehandle["fork_decision_index"] == 12
    assert rehandle["expected"] == {
        "lambda0": "reconfigure:B6:3:2",
        "lambda_positive": "deliver:B6:4:2",
    }
