from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    frozen_parent_evaluate,
    program,
)


def test_e12_frozen_e11_parent_is_bound_to_original_manifest_bytes():
    contract, manifest = frozen_parent_evaluate.authenticate_frozen_e11_parent(
        program.DEFAULT_OUTPUT,
        program.PROJECT_ROOT,
        frozen_parent_evaluate.e11.DEFAULT_OUTPUT,
    )
    e12_contract, _ = program.authenticate(program.DEFAULT_OUTPUT)
    assert manifest["contract_sha256"] == contract["contract_sha256"]
    assert (
        program.sha256(
            frozen_parent_evaluate.e11.DEFAULT_OUTPUT
            / frozen_parent_evaluate.e11.MANIFEST_NAME
        )
        == e12_contract["e11_manifest_sha256"]
    )
    assert len(manifest["records"]) == 210
