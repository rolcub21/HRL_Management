#!/usr/bin/env python3
"""Freeze and authenticate the matched Kim-2020/V2.3 supplement.

This module prepares training only.  It never trains a model, evaluates an
EpisodeInstance, or opens the sealed 86xxx panel.  The eventual evaluator is a
separate, source-bound phase so that its implementation can be audited before
any Kim checkpoint is observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL = "vcg_v2_3_kim2020_matched_supplement_85xxx_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
TRAINING_INSTANCE_SEED_BASE = 70_000_000
VALIDATION_INSTANCE_SEEDS = tuple(range(73_000_000, 73_000_005))
VALIDATION_POLICY_SEED_BASE = 74_000_000
FINAL_EVAL_INSTANCE_SEEDS = tuple(range(85_000, 85_012))
FINAL_EVAL_POLICY_SEED_BASE = 632_000_000
STOCHASTIC_ROLLOUTS = 5
EXPECTED_PRIMARY_EVAL_ROWS = 180
SEALED_INSTANCE_SEEDS = tuple(range(86_000, 86_030))
SEALED_POLICY_SEED_START = 622_000_000
SEALED_POLICY_SEED_STOP = 622_999_999

STABILITY_MANIFEST = Path(
    "results/vcg-v2-3-seed-stability-85k/stability-run-manifest.json"
)
STABILITY_MANIFEST_RAW_SHA256 = (
    "905167e28eff5da3d078b63fa505ac961e9ac80af449d0ef964496be20b6b8b9"
)
STABILITY_MANIFEST_SELF_SHA256 = (
    "9db0ef6b22879a82a16eefeb3e68f5e6662a507d8ac248b375a8bd2eeb550d5b"
)
REPAIR_REPORT = Path(
    "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k/expanded-report.json"
)
REPAIR_REPORT_RAW_SHA256 = (
    "7223fbeca579f78f06f7dbb68c9a65147201e7d9e66422f9ae73ec8e23129ac5"
)
REPAIR_REPORT_SELF_SHA256 = (
    "8da7fe1bc6f2356ae05e4d518bd07c850f766dd23ae11e36c5f600a8434e2b89"
)
REPAIR_CONTRACT = Path(
    "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k/repair-contract.json"
)
REPAIR_CONTRACT_RAW_SHA256 = (
    "d0856f12cecd00b6607b8402372f79913bf80ae8cc888714c0a0ca785c4d8035"
)
REPAIR_CONTRACT_SELF_SHA256 = (
    "77857f993cac93a87d09d40be61598fab76c619ec811fc6a3653e2505b6c0214"
)

DYNAMIC_PSLAP_METHOD_ID = "duration_aware_dynamic_pslap"
DYNAMIC_PSLAP_DISPLAY_LABEL = "Duration-aware dynamic PSLAP"
DYNAMIC_PSLAP_EXPECTED = {
    "mean_dense_objective_return": 24.23749999999958,
    "mean_absolute_error": 22.666666666666668,
    "mean_steps": 169.58333333333334,
    "physical_rehandles_per_100_required_deliveries": 9.375,
    "source_row_count": 12,
}

# Local import closure observed for Kim-v7 training, locked comparison, and
# Track-B evaluation.  The future evaluator must add its own source file and
# vcg_objective_audit.py if it imports the dense rescoring helper directly.
EXISTING_SOURCE_PATHS = (
    "GA/helper_functions.py",
    "PSLAP/checkpoint_identity.py",
    "PSLAP/dynamic_yard.py",
    "PSLAP/ga_capacity_aware.py",
    "PSLAP/ga_optimizer.py",
    "PSLAP/ga_policy.py",
    "PSLAP/kim2020_a3c_spatial.py",
    "PSLAP/neutral_protocol.py",
    "PSLAP/online_policy.py",
    "PSLAP/reg_selector_v4.py",
    "PSLAP/reg_selector_v5.py",
    "PSLAP/retrieval_context.py",
    "PSLAP/retrieval_dispatch.py",
    "PSLAP/retrieval_executor.py",
    "PSLAP/track_a.py",
    "PSLAP/viability.py",
    "compare_kim2020_a3c_spatial.py",
    "contention_metrics.py",
    "environment.py",
    "example/Options/AcceptStoreOption.py",
    "example/Options/DeliverOption.py",
    "example/Options/PickupRipeOption.py",
    "example/Options/RetrieveDeliverOption.py",
    "example/Options/StrategicDeferOption.py",
    "example/Options/pickupOption.py",
    "example/Options/selector.py",
    "example/Options/selector_v5.py",
    "example/Options/storeOption.py",
    "example/Options/waitOption.py",
    "example/__init__.py",
    "example/block_instance.py",
    "example/controller_observation.py",
    "example/controller_options.py",
    "example/episode_instance.py",
    "example/helper/occupancy_pressure.py",
    "example/helper/timing_metrics.py",
    "example/helper/tools.py",
    "example/small_rooms_env.py",
    "example/urgency_scheduler.py",
    "example/yard_geometry.py",
    "gated_agent.py",
    "option.py",
    "options_agent.py",
    "primitive_option.py",
    "relational_scheduler.py",
    "track_b_urgency_evaluate.py",
    "train_kim2020_a3c_spatial.py",
)
NEW_RUNTIME_SOURCE_PATHS = (
    "vcg_v2_3_kim2020_supplement_protocol.py",
    "experiments/vcg_v2_3_kim2020_supplement_85k/run.sh",
)

# Frozen before matched Kim training.  This makes the protocol code—not a
# newly generated manifest—the trust root for every inherited runtime source.
EXPECTED_EXISTING_SOURCE_SHA256 = {
    "GA/helper_functions.py": "f29f4232cc9b4865dca649cfe5c476406da0dff2fd8a9d9687ddefb62f1f8f5e",
    "PSLAP/checkpoint_identity.py": "2158c86551b8b2cd48f0118d32812a0a38b8438edf755e91f68055425f442db7",
    "PSLAP/dynamic_yard.py": "8d899018ea9594d98417e0f59875a3b9bbc30aff8ec687f3c73338768d49adde",
    "PSLAP/ga_capacity_aware.py": "539f0e06eb5ad4d60b5b8e565830d5123899519cc4e5eaedd9534803669fbdc3",
    "PSLAP/ga_optimizer.py": "693797d426c4a1d5caa0344de7d31d2d14a9b8299b04edfc7250b758d6edb53a",
    "PSLAP/ga_policy.py": "7e34b4fa2dbd0ab668938e10e4bd9693a9cc521fbfd433409c208d4118d63551",
    "PSLAP/kim2020_a3c_spatial.py": "d0e8d18d786b7a2fcf80e9621b36c0a45b7837eb91c20c5e7e3086f9bfa18a9a",
    "PSLAP/neutral_protocol.py": "846e6b991d5f9b6a9e40cdc834793ea0a23586f848f6578a014b3f85c88719b6",
    "PSLAP/online_policy.py": "60fe99c33835e2dc7acc1aba1234a532eaed0e7ed0bd3762f9012fdbb7c03263",
    "PSLAP/reg_selector_v4.py": "a3bd83afa0727ccc29b585ba17ea9f20dd7f81a4694c85cc403a86261d1b9796",
    "PSLAP/reg_selector_v5.py": "82e749eacc5de4ed007653f09bba2a6d57f1b021fcc5c4d2baea58b5fde807bb",
    "PSLAP/retrieval_context.py": "78b6aaab7fc778fb5390efeb286ec93effeb48580351ad7eebb5c04c5fb07df7",
    "PSLAP/retrieval_dispatch.py": "552e5b2c2437f1dd747893b4e1c986d0bddb00b1c4807f4dcfee564bd7d5da9e",
    "PSLAP/retrieval_executor.py": "5e7f348f9c7e724ec305bea6d371ed2142aae03721ef1ebd90c369784d723c32",
    "PSLAP/track_a.py": "9176c1315ae395e4c22e1e62e39e66fda79eb85e9c3eeb1fec0033e5347be5af",
    "PSLAP/viability.py": "7b14e69a85e9daf390d3b0c24bbcf08792917f32d5aa2d4cc6048978e945638e",
    "compare_kim2020_a3c_spatial.py": "60d6bf99902a7d04b403fbe8a232852dd0e53874d996311771f7032244766298",
    "contention_metrics.py": "bdad2546c85c8a0cb026637da718105cc5a2ba3a56e86fc144d8888acd6bc42d",
    "environment.py": "549b2731395d809cfcdf4a3d9ee1a961b0be4e068dcfaad09d99810fdd7c5e64",
    "example/Options/AcceptStoreOption.py": "7e29543340dcf270bab9fa20b055d077bd7d38eca07dc0b024dac413897f78ff",
    "example/Options/DeliverOption.py": "a50e01c722436015934483dcf896d44a3a041776c7ece8e893576607ce0efab4",
    "example/Options/PickupRipeOption.py": "8d538472ad1d4a9e13e46a5ae1b9dc4e6a5b4d449712a38ebc1acf034fd1ec3f",
    "example/Options/RetrieveDeliverOption.py": "1541d0347f404c85f2dbe0e16f21107dda221e4ad098c9e89562e8e73ed0f22e",
    "example/Options/StrategicDeferOption.py": "a25438191447f3c6cbaf1d30857072746e6afe05dc18bea438b4ed625681129f",
    "example/Options/pickupOption.py": "c5d0b1ac1e4d638022e57e6cfada6ebffbadaf557763efa1b09117a623e7f11b",
    "example/Options/selector.py": "6dba5972bcb8cfdc276b443a82daafa2d0dab35fd054f23b0134ae85d9e924fd",
    "example/Options/selector_v5.py": "59bc7be42a980b5002fc9a03699d2c11fee4560a9a4449e4ced201cb2701f4a2",
    "example/Options/storeOption.py": "0289bb11acec9bea6cec4eec78638ea4ecf144f4f8de191e7c380bafc1fb7122",
    "example/Options/waitOption.py": "a59862b4d0347a93f6d1d4c6175c7ff11a7305882b1c09ee97f227c331e155e1",
    "example/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "example/block_instance.py": "73bb92778c82df773417d51caea7335970ff4debd84f4735df3c881a1c63aefc",
    "example/controller_observation.py": "addb22e7ee7145fe3c1016cad7ca1e3de0cf7edb11b99abe654bbfe18ec85345",
    "example/controller_options.py": "2a49b94fd28bcc1d9222ebd51ca9155ee71dc991066d9ef1a488a303949a69f7",
    "example/episode_instance.py": "ff87a0af3f7c15b095009586935cdb9e642497971ceccf80f9da4c383eb7ccba",
    "example/helper/occupancy_pressure.py": "27608e3085bd55cd3d7d78c2831fb56992155ad786b45b7dba9ca248ddbab232",
    "example/helper/timing_metrics.py": "ec21c3b80f5bba69e2e31c409eecf87ddd3d69afc6257bb09d7984584e4db9b6",
    "example/helper/tools.py": "4f762b24af1662f7456306777af3e5c0dfb30d70f1b540dea715a634fa882f2f",
    "example/small_rooms_env.py": "f75d959d395bd316b5c8b8455d0e1377f6f4daf18317cf558222336ade0ce51a",
    "example/urgency_scheduler.py": "20e9a74cc508479b488e353ba2dd1109fc8fddbc1bfa02ec488c91d9c480471a",
    "example/yard_geometry.py": "7ca5a33157b979796bb0a996e83c1cc256f10fddee98a09a0eeeb83d6148c098",
    "gated_agent.py": "9c502034a0806a45d87fad0d9633654c39a3f349c3b11cc6497c26068dd93913",
    "option.py": "16f46ab222427bef7fc520ee48d698750352a4e36141a521665a91f68cd34376",
    "options_agent.py": "bc2170cf9be1efb4494e22a2fd21ea38ab47600e467fe023847af3bc3e7255ec",
    "primitive_option.py": "8a9aa9fe8f06c210384d865f9cd2909b3529cc566834ab476b80492d555e28ef",
    "relational_scheduler.py": "dd0d205312200bb973c4bc810baf17d5f6994f6f860c0f5c1fea3acb0448ee6e",
    "track_b_urgency_evaluate.py": "49fd55a96b24c41392cb00d55e44621c902723a7d5b6ff049e41c7df41e22dc5",
    "train_kim2020_a3c_spatial.py": "bf407ae3e45e715c5152948f5b201b5ed8f34e18ccdd9e849e120042849608d5",
}

TRAINING_CONFIG = {
    "arrival_rate": 10.0,
    "processing_time_mean": 80.0,
    "grid_rows": 5,
    "grid_cols": 5,
    "requested_exit_width": None,
    "expected_default_exit_width": 3,
    "expected_geometry_signature": "f4d984dd3f4c27c7",
    "number_blocks": 8,
    "episodes": 1000,
    "max_steps": 2000,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "gamma": 0.99,
    "reward_scale": 1.0,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "gradient_clip": 5.0,
    "hidden_channels": 32,
    "updates_per_episode": 1,
    "evaluation_every_episodes": 100,
    "stochastic_validation_rollouts": 5,
    "deterministic_algorithms": True,
    "device": "cuda",
    "cublas_workspace_config": ":4096:8",
    "training_objective": "kim2020_v7_obstruction_reduction_unchanged",
    "selection_contract": (
        "primary_stochastic_strict_success_then_mean_obstructive_moves_v2"
    ),
    "resume_supported": False,
}

EVALUATION_CONFIG = {
    "deployment": "stochastic_primary",
    "model_count": 3,
    "instance_count": 12,
    "rollouts_per_model_instance": STOCHASTIC_ROLLOUTS,
    "primary_row_count": EXPECTED_PRIMARY_EVAL_ROWS,
    "aggregation_order": "rollouts_then_instances_then_equal_model_seeds",
    "statistical_unit": "EpisodeInstance_cluster_n12",
    "max_steps": 2000,
    "max_defer_steps": 10,
    "lookahead_margin_steps": 2,
    "target_window": 20,
    "assignment_commitment": "decision_epoch_reserved",
    "scheduler": "source_neutral_duration_aware_reserved_cell",
    "map_deployment_ranked": False,
    "complete_case_filtering_allowed": False,
    "primary_contention_metric": (
        "total_physical_storage_relocations_per_100_required_deliveries"
    ),
}
EXPECTED_RUNTIME = {
    "python": "3.10.12",
    "numpy": "2.2.6",
    "torch": "2.5.1+cu124",
    "torch_cuda": "12.4",
}


class ProtocolError(RuntimeError):
    """Fail-closed protocol error."""


TRAINING_ARTIFACT_NAMES = {
    "best.pth",
    "latest.pth",
    "training-history.json",
    "training-summary.json",
    "validation-instances.json",
    "validation-instances",
}
VALIDATION_INSTANCE_FILENAMES = {
    f"seed-{seed}.json" for seed in VALIDATION_INSTANCE_SEEDS
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _runtime_contract() -> dict[str, Any]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if observed != EXPECTED_RUNTIME:
        raise ProtocolError(f"runtime version drift: {observed!r}")
    return {
        **observed,
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "cublas_workspace_config_during_training": ":4096:8",
    }


def _self_hash(value: dict[str, Any]) -> str:
    clone = dict(value)
    clone.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(clone)).hexdigest()


def training_schedule_range(model_seed: int) -> tuple[int, int]:
    if model_seed not in MODEL_SEEDS:
        raise ProtocolError(f"unsupported model seed: {model_seed}")
    start = TRAINING_INSTANCE_SEED_BASE + model_seed * 1_000_000 + 1
    return start, start + TRAINING_CONFIG["episodes"] - 1


def evaluation_policy_seed(model_seed: int, panel_index: int, rollout: int) -> int:
    if model_seed not in MODEL_SEEDS:
        raise ProtocolError(f"unsupported model seed: {model_seed}")
    if not 0 <= panel_index < len(FINAL_EVAL_INSTANCE_SEEDS):
        raise ProtocolError(f"invalid panel index: {panel_index}")
    if not 0 <= rollout < STOCHASTIC_ROLLOUTS:
        raise ProtocolError(f"invalid rollout: {rollout}")
    return FINAL_EVAL_POLICY_SEED_BASE + 1000 * model_seed + 10 * panel_index + rollout


def expected_evaluation_grid() -> list[dict[str, int]]:
    rows = []
    for model_seed in MODEL_SEEDS:
        for panel_index, instance_seed in enumerate(FINAL_EVAL_INSTANCE_SEEDS):
            for rollout in range(STOCHASTIC_ROLLOUTS):
                rows.append(
                    {
                        "model_seed": model_seed,
                        "instance_seed": instance_seed,
                        "panel_index": panel_index,
                        "rollout": rollout,
                        "policy_seed": evaluation_policy_seed(
                            model_seed, panel_index, rollout
                        ),
                    }
                )
    if len(rows) != EXPECTED_PRIMARY_EVAL_ROWS:
        raise ProtocolError("internal evaluation-grid cardinality mismatch")
    policy_seeds = [row["policy_seed"] for row in rows]
    if len(set(policy_seeds)) != len(policy_seeds):
        raise ProtocolError("evaluation policy seeds are not unique")
    return rows


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_output_tree(output_root: Path, *, require_manifest: bool = True) -> None:
    """Validate the exact prepared/training tree without following symlinks."""

    if output_root.is_symlink():
        raise ProtocolError("supplement output root may not be a symlink")
    if not output_root.exists():
        if require_manifest:
            raise ProtocolError("supplement output root is missing")
        return
    if not output_root.is_dir():
        raise ProtocolError("supplement output root is not a directory")
    entries = {entry.name: entry for entry in output_root.iterdir()}
    allowed_root = {"supplement-run-manifest.json", "training"}
    unexpected = sorted(set(entries) - allowed_root)
    if unexpected:
        raise ProtocolError(f"unexpected supplement-root entries: {unexpected!r}")
    manifest = entries.get("supplement-run-manifest.json")
    if require_manifest and (
        manifest is None or manifest.is_symlink() or not manifest.is_file()
    ):
        raise ProtocolError("manifest must be a regular non-symlink file")
    training = entries.get("training")
    if training is None:
        return
    if training.is_symlink() or not training.is_dir():
        raise ProtocolError("training must be a regular directory")
    allowed_seed_dirs = {f"seed-{seed}" for seed in MODEL_SEEDS}
    seed_entries = {entry.name: entry for entry in training.iterdir()}
    unexpected_seeds = sorted(set(seed_entries) - allowed_seed_dirs)
    if unexpected_seeds:
        raise ProtocolError(f"unexpected training entries: {unexpected_seeds!r}")
    for name, seed_dir in seed_entries.items():
        if seed_dir.is_symlink() or not seed_dir.is_dir():
            raise ProtocolError(f"{name} must be a regular directory")
        artifacts = {entry.name: entry for entry in seed_dir.iterdir()}
        if not artifacts:
            # An empty directory is an atomic launch reservation.
            continue
        if set(artifacts) != TRAINING_ARTIFACT_NAMES:
            raise ProtocolError(f"incomplete or unexpected artifacts in {name}")
        for artifact_name, artifact in artifacts.items():
            if artifact.is_symlink():
                raise ProtocolError(f"symlink artifact in {name}: {artifact_name}")
            if artifact_name == "validation-instances":
                if not artifact.is_dir():
                    raise ProtocolError(f"validation-instances is not a directory in {name}")
                validation_files = {entry.name: entry for entry in artifact.iterdir()}
                if set(validation_files) != VALIDATION_INSTANCE_FILENAMES:
                    raise ProtocolError(f"validation instance file set mismatch in {name}")
                if any(entry.is_symlink() or not entry.is_file() for entry in validation_files.values()):
                    raise ProtocolError(f"invalid validation instance file in {name}")
            elif not artifact.is_file():
                raise ProtocolError(f"non-file training artifact in {name}: {artifact_name}")


def _authenticate_parent_artifacts(project_root: Path) -> dict[str, Any]:
    stability_path = project_root / STABILITY_MANIFEST
    repair_path = project_root / REPAIR_REPORT
    repair_contract_path = project_root / REPAIR_CONTRACT
    observed = {
        "stability_manifest": _sha256(stability_path),
        "repair_report": _sha256(repair_path),
        "repair_contract": _sha256(repair_contract_path),
    }
    expected = {
        "stability_manifest": STABILITY_MANIFEST_RAW_SHA256,
        "repair_report": REPAIR_REPORT_RAW_SHA256,
        "repair_contract": REPAIR_CONTRACT_RAW_SHA256,
    }
    if observed != expected:
        raise ProtocolError(f"parent artifact raw hash mismatch: {observed!r}")
    stability = _load_json(stability_path)
    repair = _load_json(repair_path)
    repair_contract = _load_json(repair_contract_path)
    if stability.get("manifest_sha256") != STABILITY_MANIFEST_SELF_SHA256:
        raise ProtocolError("stability manifest self hash mismatch")
    if repair.get("report_sha256") != REPAIR_REPORT_SELF_SHA256:
        raise ProtocolError("repair report self hash mismatch")
    if repair_contract.get("contract_sha256") != REPAIR_CONTRACT_SELF_SHA256:
        raise ProtocolError("repair contract self hash mismatch")
    if stability.get("final_86xxx_panel_opened") is not False:
        raise ProtocolError("stability artifact says final panel was opened")
    if repair.get("final_86xxx_panel_opened") is not False:
        raise ProtocolError("repair artifact says final panel was opened")
    if repair_contract.get("final_86xxx_panel_opened") is not False:
        raise ProtocolError("repair contract says final panel was opened")

    stability_sources = stability.get("implementation_source_sha256")
    repair_sources = repair_contract.get("current_repair_source_sha256")
    if not isinstance(stability_sources, dict) or not isinstance(repair_sources, dict):
        raise ProtocolError("parent source-hash registry missing")
    for relative, frozen_sha in EXPECTED_EXISTING_SOURCE_SHA256.items():
        if relative in stability_sources and stability_sources[relative] != frozen_sha:
            raise ProtocolError(f"stability source trust mismatch: {relative}")
        if relative in repair_sources and repair_sources[relative] != frozen_sha:
            raise ProtocolError(f"repair source trust mismatch: {relative}")
    summaries = repair.get("method_summaries")
    if not isinstance(summaries, list):
        raise ProtocolError("repair report has no method summaries")
    matches = [
        row
        for row in summaries
        if row.get("method_id") == DYNAMIC_PSLAP_METHOD_ID
    ]
    if len(matches) != 1:
        raise ProtocolError("expected exactly one Dynamic PSLAP summary")
    row = matches[0]
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ProtocolError("Dynamic PSLAP summary has no metrics mapping")
    checks = {
        "mean_dense_objective_return": metrics.get("mean_dense_objective_return"),
        "mean_absolute_error": metrics.get("mean_absolute_error"),
        "mean_steps": metrics.get("mean_steps"),
        "physical_rehandles_per_100_required_deliveries": metrics.get(
            "physical_rehandles_per_100_required_deliveries"
        ),
        "source_row_count": row.get("source_row_count"),
    }
    if row.get("whole_method_numeric_eligible") is not True:
        raise ProtocolError("Dynamic PSLAP is not whole-method numeric eligible")
    if float(row.get("observed_strict_success_rate", -1.0)) != 1.0:
        raise ProtocolError("Dynamic PSLAP strict-success rate drifted")
    for key, expected_value in DYNAMIC_PSLAP_EXPECTED.items():
        observed_value = checks[key]
        if isinstance(expected_value, float):
            if observed_value is None or abs(float(observed_value) - expected_value) > 1e-9:
                raise ProtocolError(f"Dynamic PSLAP metric mismatch: {key}")
        elif observed_value != expected_value:
            raise ProtocolError(f"Dynamic PSLAP field mismatch: {key}")
    return {
        "stability_manifest_path": str(STABILITY_MANIFEST),
        "stability_manifest_raw_sha256": observed["stability_manifest"],
        "stability_manifest_self_sha256": stability["manifest_sha256"],
        "repair_report_path": str(REPAIR_REPORT),
        "repair_report_raw_sha256": observed["repair_report"],
        "repair_report_self_sha256": repair["report_sha256"],
        "repair_contract_path": str(REPAIR_CONTRACT),
        "repair_contract_raw_sha256": observed["repair_contract"],
        "repair_contract_self_sha256": repair_contract["contract_sha256"],
        "dynamic_pslap_reused_summary": checks,
    }


def _source_hashes(project_root: Path) -> dict[str, str]:
    result = {}
    for relative in EXISTING_SOURCE_PATHS + NEW_RUNTIME_SOURCE_PATHS:
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise ProtocolError(f"missing/nonregular source: {relative}")
        result[relative] = _sha256(path)
        if relative in EXPECTED_EXISTING_SOURCE_SHA256:
            expected = EXPECTED_EXISTING_SOURCE_SHA256[relative]
            if result[relative] != expected:
                raise ProtocolError(f"frozen inherited source drift: {relative}")
    if Path(__file__).resolve() != (
        project_root / "vcg_v2_3_kim2020_supplement_protocol.py"
    ).resolve():
        raise ProtocolError("protocol must run from its installed project path")
    return result


def training_command(project_root: Path, output_root: Path, model_seed: int) -> list[str]:
    training_schedule_range(model_seed)
    seed_dir = output_root / "training" / f"seed-{model_seed}"
    return [
        str(project_root / ".venv/bin/python"),
        str(project_root / "train_kim2020_a3c_spatial.py"),
        "--lambda", "10",
        "--mu", "80",
        "--grid-rows", "5",
        "--grid-cols", "5",
        "--number-blocks", "8",
        "--seed", str(model_seed),
        "--episodes", "1000",
        "--max-steps", "2000",
        "--learning-rate", "0.0001",
        "--weight-decay", "0",
        "--gamma", "0.99",
        "--reward-scale", "1",
        "--entropy-coef", "0.01",
        "--value-coef", "0.5",
        "--grad-clip", "5",
        "--hidden-channels", "32",
        "--updates-per-episode", "1",
        "--eval-every", "100",
        "--validation-seeds", *(str(value) for value in VALIDATION_INSTANCE_SEEDS),
        "--stochastic-rollouts", "5",
        "--validation-policy-seed-base", str(VALIDATION_POLICY_SEED_BASE),
        "--training-instance-seed-base", str(TRAINING_INSTANCE_SEED_BASE),
        "--deterministic-algorithms",
        "--device", "cuda",
        "--output-dir", str(seed_dir),
    ]


def build_manifest(project_root: Path, output_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    grid = expected_evaluation_grid()
    used_instance_seeds = set(VALIDATION_INSTANCE_SEEDS)
    used_instance_seeds.update(FINAL_EVAL_INSTANCE_SEEDS)
    for seed in MODEL_SEEDS:
        start, stop = training_schedule_range(seed)
        used_instance_seeds.update(range(start, stop + 1))
    if used_instance_seeds.intersection(SEALED_INSTANCE_SEEDS):
        raise ProtocolError("sealed 86xxx instance seed entered protocol")
    used_policy_seeds = {
        *(VALIDATION_POLICY_SEED_BASE + index for index in range(6)),
        *(row["policy_seed"] for row in grid),
    }
    if any(
        SEALED_POLICY_SEED_START <= value <= SEALED_POLICY_SEED_STOP
        for value in used_policy_seeds
    ):
        raise ProtocolError("sealed 622m policy seed entered protocol")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared_training_not_started",
        "development_only": True,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "original_stability_verdict_mutated": False,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
        "vcg_seed_11_13_outcomes_used_to_design_protocol": False,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "parent_artifacts": _authenticate_parent_artifacts(project_root),
        "display_labels": {
            DYNAMIC_PSLAP_METHOD_ID: DYNAMIC_PSLAP_DISPLAY_LABEL,
            "kim2020_a3c_spatial_adapted__stochastic": (
                "Kim et al. (2020)-inspired spatial A3C (stochastic)"
            ),
        },
        "dynamic_pslap_execution": "reuse_authenticated_rows_never_rerun",
        "legacy_adapted_pslap_ranked": False,
        "offline_pslap_full_schedule_ranked": False,
        "model_seeds": list(MODEL_SEEDS),
        "training_instance_seed_base": TRAINING_INSTANCE_SEED_BASE,
        "training_schedule_ranges": {
            str(seed): list(training_schedule_range(seed)) for seed in MODEL_SEEDS
        },
        "validation_instance_seeds": list(VALIDATION_INSTANCE_SEEDS),
        "validation_policy_seed_base": VALIDATION_POLICY_SEED_BASE,
        "final_evaluation_instance_seeds": list(FINAL_EVAL_INSTANCE_SEEDS),
        "final_evaluation_policy_seed_formula": (
            "632000000 + 1000*model_seed + 10*panel_index + rollout"
        ),
        "final_evaluation_grid": grid,
        "training_config": dict(TRAINING_CONFIG),
        "evaluation_config": dict(EVALUATION_CONFIG),
        "whole_method_gate": {
            "eligible_training_checkpoints_required": 3,
            "strict_primary_rows_required": EXPECTED_PRIMARY_EVAL_ROWS,
            "missing_or_unsafe_result": "unevaluable_not_pass",
        },
        "extended_screen": {
            "separate_from_original_stability": True,
            "coordinates": [
                "mean_absolute_error",
                "total_physical_storage_relocations_per_100_required_deliveries",
            ],
            "weak_both_strict_at_least_one": True,
            "require_equal_seed_vcg_nondominated": True,
            "require_vcg_individual_seed_nondominated_count_at_least": 2,
            "final_panel_remains_sealed_until_evaluable_pass": True,
        },
        "source_sha256": _source_hashes(project_root),
        "runtime": _runtime_contract(),
        "training_commands": {
            str(seed): training_command(project_root, output_root, seed)
            for seed in MODEL_SEEDS
        },
    }
    manifest["manifest_sha256"] = _self_hash(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any], project_root: Path, output_root: Path) -> None:
    if set(manifest) != {
        "schema_version", "protocol", "status", "development_only",
        "performance_claim_authorized", "confirmatory_claim_authorized",
        "original_stability_verdict_mutated", "final_86xxx_panel_opened",
        "sealed_622m_policy_namespace_opened",
        "vcg_seed_11_13_outcomes_used_to_design_protocol", "project_root",
        "output_root", "parent_artifacts", "display_labels",
        "dynamic_pslap_execution", "legacy_adapted_pslap_ranked",
        "offline_pslap_full_schedule_ranked", "model_seeds",
        "training_instance_seed_base", "training_schedule_ranges",
        "validation_instance_seeds", "validation_policy_seed_base",
        "final_evaluation_instance_seeds",
        "final_evaluation_policy_seed_formula", "final_evaluation_grid",
        "training_config", "evaluation_config", "whole_method_gate",
        "extended_screen", "source_sha256", "runtime", "training_commands",
        "manifest_sha256",
    }:
        raise ProtocolError("manifest schema mismatch")
    expected = build_manifest(project_root, output_root)
    if manifest != expected:
        raise ProtocolError("manifest differs from independently reconstructed protocol")


def _write_manifest(project_root: Path, output_root: Path) -> Path:
    output_root = output_root.resolve()
    manifest_path = output_root / "supplement-run-manifest.json"
    manifest = build_manifest(project_root, output_root)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_root.exists():
        _validate_output_tree(output_root)
        if (output_root / "training").exists():
            raise ProtocolError("prepare cannot be repeated after training was reserved")
        if manifest_path.read_text(encoding="utf-8") != payload:
            raise ProtocolError("existing manifest bytes differ")
        return manifest_path
    output_root.mkdir(parents=True, exist_ok=False)
    temporary = output_root / ".supplement-run-manifest.json.tmp"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "print-train-command"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.command == "print-train-command" and args.seed not in MODEL_SEEDS:
        parser.error("--seed must be one of 0, 1, 2")
    if args.command != "print-train-command" and args.seed is not None:
        parser.error("--seed is valid only for print-train-command")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = args.project_root.resolve()
    output_argument = args.output_root.absolute()
    if output_argument.is_symlink():
        raise ProtocolError("output-root argument may not be a symlink")
    output_root = output_argument.resolve()
    if args.command == "prepare":
        path = _write_manifest(project_root, output_root)
        print(path)
        return 0
    manifest_path = output_root / "supplement-run-manifest.json"
    _validate_output_tree(output_root)
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, project_root, output_root)
    if args.command == "validate":
        print(manifest["manifest_sha256"])
        return 0
    command = training_command(project_root, output_root, args.seed)
    print(" ".join(shlex.quote(part) for part in command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
