import importlib.util
from pathlib import Path
import sys

from example.yard_geometry import make_shipyard_env
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from viability_graph_hierarchy import candidate_action_features

from tests.test_viability_candidates import make_stored


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "experiments/conditioned_vcg/E03_certification_ablation_90k/run.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("e03_certification", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unsafe_accept_environment():
    env = make_shipyard_env(
        arrival_rate=10.0,
        proc_mean=80,
        grid_rows=5,
        grid_cols=5,
        exit_width=1,
        number_blocks=6,
    )
    env.reset(instance=env.sample_episode_instance(7240))
    occupied = ((1, 1), (2, 1), (2, 2), (2, 3), (3, 2))
    for block, cell in zip(env.blocks[:5], occupied):
        make_stored(block, cell)
    inbound = env.blocks[5]
    inbound.position = tuple(env.pickup_cell)
    inbound.storage_location = None
    inbound.stored = False
    inbound.delivered = False
    inbound.carrying = False
    env.current_state = (1, 1)
    env.time_steps = 1
    return env


def test_physical_source_restores_shadow_unsafe_candidate_without_filtering():
    e03 = _module()
    env = _unsafe_accept_environment()
    cache = ViabilityCertificateCache()
    config = ViabilitySearchConfig(max_nodes=20_000, search_order="goal_directed")
    certified = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=config,
        cache=cache,
    )
    physical = e03._physical_snapshot(
        env, certified, search_config=config, cache=cache
    )

    unsafe = [
        candidate
        for candidate in physical.candidates
        if candidate.shadow_certificate.status is ViabilityStatus.UNSAFE
    ]
    assert [candidate.key for candidate in unsafe] == ["accept:B6:1:2"]
    assert unsafe[0].certificate.status is ViabilityStatus.SAFE
    assert e03.ELIGIBILITY_PROXY_REASON in unsafe[0].certificate.reason
    assert len(physical.candidates) == len(certified.candidates) + 1
    assert physical.audit.certification_contract == e03.SHADOW_CONTRACT


def test_certified_candidates_keep_identical_network_features_and_order():
    e03 = _module()
    env = _unsafe_accept_environment()
    cache = ViabilityCertificateCache()
    config = ViabilitySearchConfig(max_nodes=20_000, search_order="goal_directed")
    certified = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=config,
        cache=cache,
    )
    physical = e03._physical_snapshot(
        env, certified, search_config=config, cache=cache
    )
    physical_safe = [
        candidate
        for candidate in physical.candidates
        if candidate.shadow_certificate.status is ViabilityStatus.SAFE
    ]
    assert [candidate.key for candidate in physical_safe] == [
        candidate.key for candidate in certified.candidates
    ]
    assert [candidate_action_features(candidate) for candidate in physical_safe] == [
        candidate_action_features(candidate) for candidate in certified.candidates
    ]


def test_secondary_metrics_are_explicitly_conditioned_on_strict_completion():
    e03 = _module()
    rows = [
        {
            "strict_completion": True,
            "recoverability_deadlock": False,
            "self_blocking": False,
            "unresolved_successor": False,
            "empty_certified_frontier_after_selection": False,
            "empty_physical_frontier": False,
            "unsafe_admission": False,
            "accepted_workloads_rendered_impossible": False,
            "dense_return": 10.0,
            "mean_absolute_error": 2.0,
            "within_target_window_rate": 1.0,
            "steps": 5,
            "physical_rehandles_per_100_required_deliveries": 12.5,
        },
        {
            "strict_completion": False,
            "recoverability_deadlock": True,
            "self_blocking": True,
            "unresolved_successor": False,
            "empty_certified_frontier_after_selection": True,
            "empty_physical_frontier": False,
            "unsafe_admission": True,
            "accepted_workloads_rendered_impossible": True,
            "dense_return": -100.0,
            "mean_absolute_error": None,
            "within_target_window_rate": None,
            "steps": 3,
            "physical_rehandles_per_100_required_deliveries": 0.0,
        },
    ]
    summary = e03._source_summary(rows)
    assert summary["strict_completion"] == {"count": 1, "rate": 0.5}
    assert summary["self_blocking"] == {"count": 1, "rate": 0.5}
    assert summary["strict_completer_secondary_metrics"]["denominator"] == 1
    assert summary["strict_completer_secondary_metrics"]["dense_return"] == 10.0
