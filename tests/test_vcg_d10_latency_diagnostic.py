import importlib
from pathlib import Path


MODULE = (
    "experiments.conditioned_vcg.development."
    "D10_scalability_support_screen.latency_diagnostic"
)


def test_scope_is_exactly_two_10x10_occupancy_pilots():
    diagnostic = importlib.import_module(MODULE)
    assert diagnostic.SCENARIO_IDS == (
        "size_10x10_occ_medium",
        "size_10x10_occ_high",
    )
    assert diagnostic.INSTANCE_SEED == 95100
    assert diagnostic.WALL_CLOCK_LIMIT_SECONDS == 3600


def test_contract_declares_censoring_and_no_training(tmp_path: Path, monkeypatch):
    diagnostic = importlib.import_module(MODULE)
    parent_contract = {
        "protocol": "parent",
        "contract_sha256": "c" * 64,
    }
    parent_manifest = {
        "manifest_sha256": "m" * 64,
        "records": [
            {"scenario_id": scenario, "seed": diagnostic.INSTANCE_SEED}
            for scenario in diagnostic.SCENARIO_IDS
        ],
    }
    monkeypatch.setattr(
        diagnostic.occupancy,
        "authenticate",
        lambda _path: (parent_contract, parent_manifest),
    )
    contract = diagnostic._expected_contract()
    assert contract["frozen_controller_no_training"] is True
    assert contract["timeout_interpretation"] == "censored_not_infeasible_not_failed"
    assert contract["sequential_isolated_execution_required"] is True


def test_call_summary_keeps_role_and_tail():
    diagnostic = importlib.import_module(MODULE)
    calls = [
        {"seconds": 0.1, "explored_nodes": 2, "status": "SAFE"},
        {"seconds": 4.0, "explored_nodes": 9, "status": "SAFE"},
    ]
    result = diagnostic._call_distribution(calls)
    assert result["count"] == 2
    assert result["total_seconds"] == 4.1
    assert result["seconds"]["maximum"] == 4.0
    assert result["explored_nodes"]["maximum"] == 9.0
