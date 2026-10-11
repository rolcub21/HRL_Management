"""Versioned E79 repair with an episode-specific floating-point audit bound."""

from __future__ import annotations

from pathlib import Path

from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v1 import protocol as v1

ROOT = v1.ROOT
PROTOCOL = "vcg_conditioned_e79_original_base100_full_10x10_144k_v2"
METHOD_VERSION = "vcg_dense_v1_1_target_size_base100_e79_v2"
SCENARIO = v1.SCENARIO
SCENARIOS = v1.SCENARIOS
MODEL_SEEDS = v1.MODEL_SEEDS
TRAINING_JOB_BUDGET = v1.TRAINING_JOB_BUDGET
EPISODES_BY_SCENARIO = v1.EPISODES_BY_SCENARIO
TOTAL_EPISODES = v1.TOTAL_EPISODES
VALIDATION_EVERY = v1.VALIDATION_EVERY
CHECKPOINT_EVERY = v1.CHECKPOINT_EVERY
VALIDATION_SEEDS = v1.VALIDATION_SEEDS
TRAIN_BASE = v1.TRAIN_BASE
TD_BACKUP_IMPLEMENTATION = v1.TD_BACKUP_IMPLEMENTATION
DELIVERY_BASE = v1.DELIVERY_BASE
SPEC = v1.SPEC
LATE_CHECKPOINTS = v1.LATE_CHECKPOINTS
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e79-original-base100-full-10x10-144k-v2"
scenario_for = v1.scenario_for
TargetDenseEnv = v1.TargetDenseEnv
sha256 = v1.sha256
digest = v1.digest


def contract() -> dict:
    # The inherited runner and learner are frozen. Only the numerical audit
    # wrapper and its provenance differ from v1.
    old = v1.contract()
    directory = Path(__file__).resolve().parent
    sources = dict(old["source_sha256"])
    for name in ("protocol.py", "run.py", "roundoff.py", "run.sh"):
        path = directory / name
        sources[str(path.relative_to(ROOT))] = sha256(path)
    payload = {key: value for key, value in old.items() if key != "contract_sha256"}
    payload.update({
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "numerical_audit_repair": (
            "primitive/macro return agreement bounded by sequential "
            "floating-point summation error from recorded episode rewards; "
            "step-level reward checks and training policy unchanged"
        ),
        "source_sha256": sources,
    })
    return {**payload, "contract_sha256": digest(payload)}
