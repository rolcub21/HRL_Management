from experiments.conditioned_vcg.E220_e215_ep300_stage_b_279k_v1 import (
    protocol as e220,
)
from experiments.conditioned_vcg.E223_e221_multiseed_stage_b_evaluation_282k_v1 import (
    protocol as p,
    run,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)


def test_e223_uses_exact_e220_paired_development_coordinates():
    assert p.EVAL_INSTANCE_SEEDS == e220.EVAL_INSTANCE_SEEDS
    assert p.EVAL_LAMBDAS == e220.EVAL_LAMBDAS == (10.0, 20.0, 40.0)
    assert p.IMMEDIATE_ONLY_LAMBDAS == e220.IMMEDIATE_ONLY_LAMBDAS == (20.0, 40.0)
    assert [p.workload_for_seed(seed) for seed in p.EVAL_INSTANCE_SEEDS] == [
        0.20, 0.50, 0.80
    ]


def test_e223_seed_routes_and_rows_are_disjoint(tmp_path):
    paths = {
        run._row_path(tmp_path, model_seed, method, instance_seed, value)
        for model_seed in p.MODEL_SEEDS
        for method, values in (
            ("learned_trace", p.EVAL_LAMBDAS),
            ("immediate_only", p.IMMEDIATE_ONLY_LAMBDAS),
        )
        for instance_seed in p.EVAL_INSTANCE_SEEDS
        for value in values
    }
    assert len(paths) == 30
    assert p.head_seed(1) != p.head_seed(2)


def test_e223_declaration_is_conforming_and_development_only():
    declaration = p.method_declaration()
    validated = validate_protocol(declaration, source=p.PROTOCOL)
    assert validated["claim_level"] == "expressibility"
    assert declaration["oracle_inputs"] == []
    assert declaration["conformance_audit"][
        "reproducibility.preprotected_integration_gate"
    ]["status"] == "not_applicable"
