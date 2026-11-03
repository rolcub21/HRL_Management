from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E225_e215_seed0_guard_free_development_284k_v1 import (
    guard_free,
    protocol as p,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)


def test_e225_is_one_seed_on_exact_e191_development_coordinates():
    assert p.MODEL_SEED == 0
    assert p.INSTANCE_SEEDS == e191_scenario.VALIDATION_SEEDS
    assert [p.workload_for_seed(seed) for seed in p.INSTANCE_SEEDS] == [
        "20%", "20%", "20%", "50%", "50%", "50%", "80%", "80%", "80%",
    ]


def test_e225_guard_free_agent_overrides_frontier_admissibility():
    assert "_admissible_prepared" in guard_free.FrozenGuardFreeQueueAgent.__dict__
    assert "select" in guard_free.FrozenGuardFreeQueueAgent.__dict__
    assert (
        guard_free.FrozenGuardFreeQueueAgent.__doc__
        and "every exact-SAFE action" in guard_free.FrozenGuardFreeQueueAgent.__doc__
    )


def test_e225_declaration_is_development_only_and_conforming():
    declaration = p.method_declaration()
    validated = validate_protocol(declaration, source=p.PROTOCOL)
    assert validated["claim_level"] == "expressibility"
    assert declaration["oracle_inputs"] == []
    assert declaration["conformance_audit"][
        "reproducibility.preprotected_integration_gate"
    ]["status"] == "not_applicable"
