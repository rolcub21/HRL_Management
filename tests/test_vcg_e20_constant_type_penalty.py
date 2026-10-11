import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/conditioned_vcg/E20_constant_type_penalty_92k/run.py"
)
SPEC = importlib.util.spec_from_file_location("e20_constant_type_penalty", MODULE_PATH)
e20 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(e20)


def test_equivalence_margins_have_operational_units():
    assert e20.EQUIVALENCE_MARGINS == {
        "mean_absolute_error": 1.0,
        "physical_rehandles_per_100_required_deliveries": 2.5,
        "steps": 5.0,
        "within_target_window_rate": 0.025,
    }


def test_calibration_requires_all_four_finite_constants(tmp_path, monkeypatch):
    artifact = e20._with_hash(
        {
            "evaluation_panel_opened": False,
            "provenance": {"frozen": True},
            "constants": {
                str(seed): {name: float(seed) for name in e20.ACTION_TYPES}
                for seed in e20.MODEL_SEEDS
            },
        },
        "calibration_sha256",
    )
    e20._atomic_json(tmp_path / e20.CALIBRATION_NAME, artifact)
    monkeypatch.setattr(e20, "_calibration_provenance", lambda: {"frozen": True})
    loaded = e20._load_calibration(tmp_path)
    assert loaded["constants"]["2"]["defer"] == 2.0

    artifact["constants"]["0"].pop("accept")
    artifact.pop("calibration_sha256")
    artifact = e20._with_hash(artifact, "calibration_sha256")
    e20._atomic_json(tmp_path / e20.CALIBRATION_NAME, artifact)
    try:
        e20._load_calibration(tmp_path)
    except e20.E20Error as error:
        assert "incomplete or invalid" in str(error)
    else:
        raise AssertionError("non-finite constant was accepted")


def test_constant_type_values_are_invariant_within_type():
    constants = {
        "accept": 1.0,
        "deliver": 2.0,
        "reconfigure": 3.0,
        "defer": 4.0,
    }
    names = ("accept", "accept", "deliver", "reconfigure", "defer")
    values = torch.as_tensor([constants[name] for name in names])
    assert values.tolist() == [1.0, 1.0, 2.0, 3.0, 4.0]


def test_paired_equivalence_requires_interval_inside_margin(monkeypatch):
    metrics = e20.e16d.METRICS

    def row(seed, instance, offset):
        result = {
            "model_seed": seed,
            "instance_seed": instance,
            "strict_safe_complete": True,
            "behavior_digest": f"b-{offset}",
        }
        for metric in metrics:
            result[metric] = 10.0 + offset
        return result

    constants = [row(0, index, 0.2) for index in range(5)]
    controls = {(0, index): row(0, index, 0.0) for index in range(5)}
    report = e20._paired_summary(constants, controls)
    assert report["metrics"]["mean_absolute_error"][
        "practical_equivalence_supported"
    ]
    assert report["metrics"]["within_target_window_rate"][
        "practical_equivalence_supported"
    ] is False
