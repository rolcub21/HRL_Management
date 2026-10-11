from __future__ import annotations

from pathlib import Path

import train_vcg_unified as unified
import train_vcg_unified_seed_stability as stability


def _args(tmp_path: Path, seed: int, variant: str):
    return stability.build_parser().parse_args(
        [
            "--model-seed",
            str(seed),
            "--variant",
            variant,
            "--output-dir",
            str(tmp_path / f"seed-{seed}" / variant),
            "--contract-only",
        ]
    )


def test_profiles_are_exact_and_disjoint():
    stability._validate_profiles()
    assert stability.SEED_PROFILES[15].train_seed_base == 61_005_000
    assert stability.SEED_PROFILES[16].behavior_rng_base == 610_006_000
    assert stability.SEED_PROFILES[17].replay_rng_seed == 610_107_010


def test_profile_activation_restores_seed14_globals():
    names = (
        "MODEL_SEED",
        "TRAIN_SEED_BASE",
        "TRAINING_POLICY_RNG_BASE",
        "REPLAY_RNG_SEED",
    )
    before = {name: getattr(unified, name) for name in names}
    with stability.activated_paired_profile(stability.SEED_PROFILES[16]):
        assert unified.MODEL_SEED == 16
        assert unified.TRAIN_SEED_BASE == 61_006_000
        assert unified.TRAINING_POLICY_RNG_BASE == 610_006_000
        assert unified.REPLAY_RNG_SEED == 610_106_010
    assert {name: getattr(unified, name) for name in names} == before


def test_each_seed_pair_has_identical_shared_contract(tmp_path):
    for seed in stability.MODEL_SEEDS:
        profile = stability.SEED_PROFILES[seed]
        with stability.activated_paired_profile(profile):
            u_args = stability._unified_args(_args(tmp_path, seed, unified.VCG))
            c_args = stability._unified_args(
                _args(tmp_path, seed, unified.VCG_HANDLING_CONSTRAINT)
            )
            unconstrained, _ = unified.build_unified_contract(u_args)
            constrained, _ = unified.build_unified_contract(c_args)
        assert (
            unconstrained["shared_configuration_sha256"]
            == constrained["shared_configuration_sha256"]
        )
        shared = unconstrained["shared_configuration"]
        assert shared["model_seed"] == seed
        assert shared["train_seed_base"] == profile.train_seed_base
        assert shared["replay_rng_seed"] == profile.replay_rng_seed


def test_contract_only_runs_no_episode(tmp_path):
    result = stability.run(_args(tmp_path, 15, unified.VCG))
    assert result["status"] == "contract_only"
    assert result["episodes_executed"] == 0
    assert result["stability_model_seed"] == 15
    assert result["final_86xxx_panel_opened"] is False

