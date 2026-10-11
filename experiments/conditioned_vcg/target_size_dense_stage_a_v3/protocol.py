"""Versioned identities for no-gradient state-deduplicated Stage A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import protocol as v2


PROTOCOL = "vcg_target_size_dense_stage_a_state_dedup_development_v3"
METHOD_VERSION = "vcg_dense_v1_1_target_size_stage_a_state_dedup_v3"
TD_BACKUP_IMPLEMENTATION = "identity_state_dedup_no_grad_continuation_v1"
SCENARIOS = v2.SCENARIOS
MODEL_SEEDS = v2.MODEL_SEEDS
TOTAL_EPISODES = v2.TOTAL_EPISODES
VALIDATION_EVERY = v2.VALIDATION_EVERY
CHECKPOINT_EVERY = v2.CHECKPOINT_EVERY
VALIDATION_SEEDS = v2.VALIDATION_SEEDS
TRAIN_BASE = v2.TRAIN_BASE
ROOT = v2.ROOT
DEFAULT_OUTPUT = ROOT / "results/vcg-target-size-dense-stage-a-development-v3"
scenario_for = v2.scenario_for
instance_for = v2.instance_for
TargetDenseEnv = v2.TargetDenseEnv


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract_payload() -> dict:
    directory = Path(__file__).resolve().parent
    v2_directory = directory.parent / "target_size_dense_stage_a_v2"
    payload = v2.contract_payload()
    payload.update({
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "target-size Stage A acceleration development; no test panel opened",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "td_backup_semantics": "unchanged; deduplication restricted to no-gradient online/target continuation scoring",
        "selection_scoring_implementation": "native",
        "chosen_action_gradient_path": "native",
        "source_sha256": {
            str(path): _sha256(path)
            for path in (
                directory / "protocol.py",
                directory / "acceleration.py",
                directory / "run.py",
                directory / "run.sh",
                v2_directory / "protocol.py",
                v2_directory / "run.py",
                ROOT / "train_vcg_dense_proper.py",
                ROOT / "run_vcg_objective_gamma_audit.py",
                ROOT / "viability_graph_hierarchy.py",
                ROOT / "viability_graph_episodic_audit.py",
                ROOT / "PSLAP/yard_graph.py",
            )
        },
    })
    return payload


def contract() -> dict:
    payload = contract_payload()
    return {**payload, "contract_sha256": _digest(payload)}
