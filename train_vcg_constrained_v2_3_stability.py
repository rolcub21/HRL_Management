#!/usr/bin/env python3
"""Isolated seed-profile facade for the frozen VCG V2.3 trainer.

The audited seed-10 module remains byte-unchanged.  This facade installs one
of three predeclared profiles only inside the current Python process, invokes
the same V2.3 lifecycle, and restores every touched module global on exit.
The shared controller, objective, schedule, gates, and selection rule are not
modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Mapping, Optional

import train_vcg_constrained_v2_3 as v23
from vcg_v2_3_seed_stability_protocol import (
    EPISODES,
    MODEL_SEEDS,
    PANEL_SEEDS,
    SEED_PROFILES,
    STABILITY_CHECKPOINT_FAMILY,
    STABILITY_PROTOCOL,
    FrozenSeedProfile,
    StabilityProtocolError,
    digest_json,
    validate_no_final_panel_values,
)


_ORIGINAL_BUILD_TRAINING_CONTRACT = v23.build_training_contract
_ORIGINAL_AUTHENTICATE_CONTRACT = v23._authenticate_frozen_v23_contract
_ACTIVE_BINDING: Optional["StabilityRunBinding"] = None


def _frozen_profile(model_seed: int) -> FrozenSeedProfile:
    if type(model_seed) is not int or model_seed not in MODEL_SEEDS:
        raise StabilityProtocolError(f"model seed must be one of {MODEL_SEEDS}")
    return SEED_PROFILES[model_seed]


@dataclass(frozen=True)
class StabilityRunBinding:
    freeze_spec_sha256: str
    run_manifest_sha256: str
    repair_v2_contract_sha256: str
    repair_v2_report_sha256: str
    implementation_source_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "freeze_spec_sha256", "run_manifest_sha256",
            "repair_v2_contract_sha256", "repair_v2_report_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise StabilityProtocolError(f"{name} must be a lowercase SHA-256")
        if not isinstance(self.implementation_source_sha256, Mapping):
            raise StabilityProtocolError("implementation source binding must be a mapping")
        for name, value in self.implementation_source_sha256.items():
            if not isinstance(name, str) or not name:
                raise StabilityProtocolError("implementation source name is invalid")
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise StabilityProtocolError(f"implementation SHA is invalid: {name}")

    def to_contract(self) -> dict:
        return {
            "freeze_spec_sha256": self.freeze_spec_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
            "repair_v2_contract_sha256": self.repair_v2_contract_sha256,
            "repair_v2_report_sha256": self.repair_v2_report_sha256,
            "implementation_source_sha256": dict(sorted(self.implementation_source_sha256.items())),
        }


def _profile_args(profile: FrozenSeedProfile, *, output_dir: Path, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=Path(output_dir),
        model_seed=profile.model_seed,
        episodes=EPISODES,
        train_seed_base=profile.train_seed_base,
        validation_seeds=PANEL_SEEDS,
        validation_every=20,
        max_steps=2_000,
        grid_rows=5,
        grid_cols=5,
        number_blocks=8,
        arrival_rate=10.0,
        proc_mean=80,
        gamma_operational=1.0,
        reward_scale=0.01,
        rehandle_budget_per_100=20.0,
        dual_lr=0.01,
        lambda_initial=0.0,
        lambda_max=20.0,
        max_hold_steps=10,
        max_idle_steps=20,
        device=str(device),
        contract_only=False,
    )


def _active_profile() -> tuple[FrozenSeedProfile, StabilityRunBinding]:
    binding = _ACTIVE_BINDING
    if binding is None:
        raise StabilityProtocolError("no seed-stability profile is active")
    profile = _frozen_profile(v23.DEFAULT_MODEL_SEED)
    return profile, binding


def _build_stability_training_contract(args: argparse.Namespace) -> dict:
    profile, binding = _active_profile()
    contract = _ORIGINAL_BUILD_TRAINING_CONTRACT(args)
    contract.update(
        {
            "training_protocol": STABILITY_PROTOCOL,
            "checkpoint_family": STABILITY_CHECKPOINT_FAMILY,
            "experiment_role": "frozen_v2_3_training_seed_stability_replication",
            "parent_method_version": "vcg_constrained_v2_3_gamma1_seed10_development",
            "isolated_algorithmic_change": {
                "parameter": "none",
                "algorithmic_change_from_seed10": False,
                "same_v2_3_objective_controller_schedule_gates_and_selection": True,
            },
            "development_panel_reused_after_v2_2": True,
            "training_instances_reused_after_v2_2": False,
            "fresh_prospective_experiment": False,
            "development_only": True,
            "performance_claim_authorized": False,
            "training_policy_rng_formula": (
                f"{profile.behavior_rng_base} + episode_number - 1"
            ),
            "stability_replication": {
                "profile": profile.to_manifest(),
                "seed10_label": "development_only_not_in_stability_aggregate",
                "fresh_training_instances": True,
                "fresh_behavior_action_rngs": True,
                "fresh_replay_rng": True,
                "opened_validation_panel_reused": True,
                "no_tuning": True,
                "no_cross_seed_checkpoint_selection": True,
                "final_86xxx_panel_opened": False,
                "binding": binding.to_contract(),
            },
        }
    )
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = digest_json(contract)
    return contract


def _authenticate_stability_training_contract(contract):
    payload, resolved_path = v23._authenticated_json_mapping(
        contract, name="V2.3 seed-stability training contract"
    )
    contract_sha = v23._verify_self_hash(
        payload,
        hash_field="contract_sha256",
        name="V2.3 seed-stability training contract",
    )
    profile, _ = _active_profile()
    device = payload.get("device")
    if device not in ("cpu", "cuda"):
        raise StabilityProtocolError("seed-stability contract device must be cpu or cuda")
    expected = _build_stability_training_contract(
        _profile_args(
            profile,
            output_dir=Path("__v2_3_stability_contract_authentication_only__"),
            device=str(device),
        )
    )
    if v23._json_safe(payload) != v23._json_safe(expected):
        differences = sorted(
            key for key in set(payload) | set(expected)
            if v23._json_safe(payload.get(key)) != v23._json_safe(expected.get(key))
        )
        raise StabilityProtocolError(
            f"seed-stability training contract provenance mismatch: {differences!r}"
        )
    return payload, resolved_path, contract_sha


@contextmanager
def activated_seed_profile(profile: FrozenSeedProfile, binding: StabilityRunBinding):
    """Install and then fully restore one isolated seed profile."""

    global _ACTIVE_BINDING
    if (
        type(profile.model_seed) is not int
        or profile != SEED_PROFILES.get(profile.model_seed)
    ):
        raise StabilityProtocolError("seed profile is not one of the frozen profiles")
    validate_no_final_panel_values(
        instance_seeds=(*profile.train_seeds, *PANEL_SEEDS),
        action_rngs=(
            *profile.behavior_rng_seeds,
            *range(620_000_000, 620_000_048),
            profile.replay_rng_seed,
        ),
    )
    if _ACTIVE_BINDING is not None:
        raise StabilityProtocolError("nested seed-profile activation is forbidden")

    base_names = (
        "TRAINING_PROTOCOL", "CHECKPOINT_FAMILY", "DEFAULT_MODEL_SEED",
        "FRESH_TRAIN_SEED_BASE", "TRAINING_POLICY_RNG_BASE", "REPLAY_RNG_SEED",
        "build_training_contract", "_authenticate_frozen_v23_contract",
    )
    base_snapshot = {name: getattr(v23, name) for name in base_names}
    # Import and snapshot the core while the canonical seed-10 globals are
    # still installed.  Importing it after patching ``v23`` would make a fresh
    # process copy the active profile values and then incorrectly "restore"
    # those values on exit.
    core = importlib.import_module("viability_graph_constrained_v2_3")
    core_names = ("CHECKPOINT_FAMILY", "REPLAY_RNG_SEED")
    core_snapshot = {name: getattr(core, name) for name in core_names}
    try:
        _ACTIVE_BINDING = binding
        v23.TRAINING_PROTOCOL = STABILITY_PROTOCOL
        v23.CHECKPOINT_FAMILY = STABILITY_CHECKPOINT_FAMILY
        v23.DEFAULT_MODEL_SEED = profile.model_seed
        v23.FRESH_TRAIN_SEED_BASE = profile.train_seed_base
        v23.TRAINING_POLICY_RNG_BASE = profile.behavior_rng_base
        v23.REPLAY_RNG_SEED = profile.replay_rng_seed
        v23.build_training_contract = _build_stability_training_contract
        v23._authenticate_frozen_v23_contract = _authenticate_stability_training_contract

        core.CHECKPOINT_FAMILY = STABILITY_CHECKPOINT_FAMILY
        core.REPLAY_RNG_SEED = profile.replay_rng_seed
        yield
    finally:
        for name, value in core_snapshot.items():
            setattr(core, name, value)
        for name, value in base_snapshot.items():
            setattr(v23, name, value)
        _ACTIVE_BINDING = None


def build_stability_training_contract(
    model_seed: int,
    *,
    binding: StabilityRunBinding,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    profile = _frozen_profile(model_seed)
    with activated_seed_profile(profile, binding):
        return _build_stability_training_contract(
            _profile_args(profile, output_dir=output_dir, device=device)
        )


def authenticate_seed_training_contract(
    model_seed: int,
    contract,
    *,
    binding: StabilityRunBinding,
) -> tuple[dict, Optional[Path], str]:
    """Independently reconstruct and authenticate one profile contract."""

    profile = _frozen_profile(model_seed)
    with activated_seed_profile(profile, binding):
        return _authenticate_stability_training_contract(contract)


def run_frozen_stability_seed(
    model_seed: int,
    *,
    binding: StabilityRunBinding,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """Run exactly one frozen long job; callers own when to invoke it."""

    profile = _frozen_profile(model_seed)
    if str(device) != "cuda":
        raise StabilityProtocolError("the reviewed stability execution device is frozen to cuda")
    with activated_seed_profile(profile, binding):
        args = _profile_args(profile, output_dir=output_dir, device=device)
        return v23.run_development_calibration(args)


def load_seed_candidate_look(
    model_seed: int,
    checkpoint_path,
    *,
    binding: StabilityRunBinding,
    manifest_path,
    validation_instance_manifest_path,
    contract,
    device: str = "cpu",
) -> dict:
    profile = _frozen_profile(model_seed)
    with activated_seed_profile(profile, binding):
        result = v23.load_candidate_look_diagnostic(
            checkpoint_path,
            manifest_path=manifest_path,
            contract=contract,
            device=device,
        )
        if result["checkpoint"].get("training_contract_sha256") != result["contract_sha256"]:
            raise StabilityProtocolError("candidate look lost its profile contract binding")
        contract_payload, resolved_contract_path, _ = (
            _authenticate_stability_training_contract(contract)
        )
        if resolved_contract_path is None:
            raise StabilityProtocolError(
                "candidate-ledger authentication requires a contract file"
            )
        root = resolved_contract_path.parent.resolve()
        validation_manifest, validation_manifest_sha = (
            v23._authenticate_validation_instance_manifest(
                validation_instance_manifest_path,
                root=root,
            )
        )
        checkpoint = result["checkpoint"]
        trusted_core = v23._trusted_runtime_core_contract(contract_payload)
        if v23._json_safe(checkpoint.get("core_contract")) != v23._json_safe(
            trusted_core
        ):
            raise StabilityProtocolError(
                "candidate diagnostic runtime core contract mismatch"
            )
        episode = int(checkpoint["completed_episodes"])
        v23._authenticate_candidate_dual_lifecycle(
            checkpoint,
            contract=contract_payload,
            episode=episode,
        )
        validation = v23._authenticate_candidate_validation_ledger(
            root=root,
            diagnostic=result,
            contract=contract_payload,
            validation_instance_manifest=validation_manifest,
        )
        result["validation_instance_manifest_sha256"] = validation_manifest_sha
        result["validation_ledger"] = validation["ledger"]
        result["validation_ledger_sha256"] = validation["ledger_sha256"]
        result["validation_summary"] = validation["summary"]
        return result


def load_seed_best_candidate(
    model_seed: int,
    checkpoint_path,
    *,
    binding: StabilityRunBinding,
    expected_best_sha256: str,
    manifest_path,
    expected_manifest_sha256: str,
    validation_instance_manifest_path,
    contract,
    device: str = "cpu",
) -> dict:
    """Fail closed on the exact selected-best artifact for one training seed."""

    profile = _frozen_profile(model_seed)
    with activated_seed_profile(profile, binding):
        result = v23.load_best_development_candidate(
            checkpoint_path,
            expected_best_sha256=expected_best_sha256,
            manifest_path=manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            validation_instance_manifest_path=validation_instance_manifest_path,
            contract=contract,
            device=device,
        )
        checkpoint = result.get("checkpoint", {})
        if checkpoint.get("training_contract_sha256") != result.get("contract_sha256"):
            raise StabilityProtocolError("selected best lost its profile contract binding")
        if checkpoint.get("method_version") != "vcg_constrained_v2_3_gamma1":
            raise StabilityProtocolError("selected best is not exact V2.3 gamma1")
        return result


__all__ = [
    "StabilityRunBinding", "activated_seed_profile",
    "authenticate_seed_training_contract",
    "build_stability_training_contract", "load_seed_best_candidate",
    "load_seed_candidate_look", "run_frozen_stability_seed",
]
