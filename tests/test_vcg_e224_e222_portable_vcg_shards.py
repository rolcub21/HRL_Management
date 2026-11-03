from pathlib import Path

from experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1 import (
    run as e203,
)
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import (
    scenario,
)
from experiments.conditioned_vcg.E224_e222_portable_vcg_shards_283k_v1 import (
    protocol as p,
    run,
)


def test_anchor_reconstructs_exact_e222_instance_manifest():
    with run._engine():
        manifest = e203.instance_manifest(run._anchor_contract())
    assert manifest["manifest_sha256"] == p.ANCHOR_INSTANCE_MANIFEST_SHA256
    assert manifest["contract_sha256"] == p.ANCHOR_CONTRACT_SHA256
    assert len(manifest["records"]) == len(scenario.TEST_SEEDS) == 30


def test_only_finalized_seed_one_and_two_are_supported():
    assert p.MODEL_SEEDS == (1, 2)
    assert p.CHECKPOINTS[1]["selected_episode"] == 280
    assert p.CHECKPOINTS[2]["selected_episode"] == 220
    assert len(p.CHECKPOINTS[1]["checkpoint_sha256"]) == 64
    assert len(p.CHECKPOINTS[2]["checkpoint_sha256"]) == 64


def test_output_is_separate_from_e222_owner_panel():
    assert "e224" in str(p.DEFAULT_OUTPUT).lower()
    assert "e222" in str(p.DEFAULT_OUTPUT).lower()
    assert Path(p.DEFAULT_OUTPUT).name != (
        "vcg-conditioned-e222-e215-e218-fresh-matched-8x8-panel-281k-v1"
    )

