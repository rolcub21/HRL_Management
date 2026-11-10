"""Frozen protocol for the one-seed E195 temperature-scale diagnostic."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import (
    protocol as e195,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e195.ROOT
PROTOCOL = "vcg_conditioned_e255_e195_temperature_scaled_5x5_314k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e255_e195_temperature_scaled_5x5"
SCENARIO = e195.SCENARIO
SCENARIOS = e195.SCENARIOS
MODEL_SEEDS = (0,)
TOTAL_EPISODES = e195.TOTAL_EPISODES
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = e195.VALIDATION_EVERY
CHECKPOINT_EVERY = e195.CHECKPOINT_EVERY
VALIDATION_SEEDS = e195.VALIDATION_SEEDS
TRAIN_BASE = e195.TRAIN_BASE
TRAINING_JOB_BUDGET = e195.TRAINING_JOB_BUDGET
MAX_STEPS = e195.MAX_STEPS
GAMMA = e195.GAMMA
EPSILON_START = e195.EPSILON_START
EPSILON_END = e195.EPSILON_END
EPSILON_WARMUP_DECISIONS = e195.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = e195.EPSILON_DECAY_DECISIONS
REPORT_CHECKPOINTS = tuple(
    range(VALIDATION_EVERY, TOTAL_EPISODES + 1, VALIDATION_EVERY)
)
LATE_CHECKPOINTS = e195.LATE_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e195.TD_BACKUP_IMPLEMENTATION
UNCHANGED_METHOD_FIELDS = e195.UNCHANGED_METHOD_FIELDS
SPEC = e195.SPEC
TargetDenseEnv = e195.TargetDenseEnv
scenario_for = e195.scenario_for
digest = e195.digest
sha256 = e195.sha256
workload = e195.workload
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e255-e195-temperature-scaled-5x5-314k-v1"
)

E195_OUTPUT = e195.DEFAULT_OUTPUT
E215_OUTPUT = ROOT / "results/vcg-conditioned-e215-e191-long-exploration-drift-pilot-274k-v1"
E218_OUTPUT = ROOT / "results/vcg-conditioned-e218-e215-long-exploration-replication-277k-v1"


def _summary(output: Path, scenario: str, seed: int) -> Path:
    return (
        output / "training/operational" / scenario
        / f"seed-{seed}/training-summary.json"
    )


DEVELOPMENT_SUMMARIES = {
    "5x5_seed0": _summary(E195_OUTPUT, SCENARIO, 0),
    "5x5_seed1": _summary(E195_OUTPUT, SCENARIO, 1),
    "5x5_seed2": _summary(E195_OUTPUT, SCENARIO, 2),
    "8x8_seed0": _summary(
        E215_OUTPUT, "empty_8x8_n36_longstay_mixedrho", 0
    ),
    "8x8_seed1": _summary(
        E218_OUTPUT, "empty_8x8_n36_longstay_mixedrho", 1
    ),
    "8x8_seed2": _summary(
        E218_OUTPUT, "empty_8x8_n36_longstay_mixedrho", 2
    ),
}
EXPECTED_SELECTED_EPISODES = {
    "5x5_seed0": 300,
    "5x5_seed1": 300,
    "5x5_seed2": 160,
    "8x8_seed0": 300,
    "8x8_seed1": 280,
    "8x8_seed2": 220,
}
EXPECTED_SELECTED_J = {
    "5x5_seed0": 56.96666666666667,
    "5x5_seed1": 84.06666666666666,
    "5x5_seed2": 79.53333333333335,
    "8x8_seed0": 2452.633333333333,
    "8x8_seed1": 2555.077777777778,
    "8x8_seed2": 2163.1777777777775,
}

ORIGINAL_WITHIN_MODE_TEMPERATURE = 0.1
ORIGINAL_MODE_TEMPERATURE = 1.0
FIVE_BY_FIVE_DEVELOPMENT_MEAN_J = fmean(
    EXPECTED_SELECTED_J[f"5x5_seed{seed}"] for seed in range(3)
)
EIGHT_BY_EIGHT_DEVELOPMENT_MEAN_J = fmean(
    EXPECTED_SELECTED_J[f"8x8_seed{seed}"] for seed in range(3)
)
TEMPERATURE_DIVISOR = (
    EIGHT_BY_EIGHT_DEVELOPMENT_MEAN_J / FIVE_BY_FIVE_DEVELOPMENT_MEAN_J
)
TAU_ACCEPT = ORIGINAL_WITHIN_MODE_TEMPERATURE / TEMPERATURE_DIVISOR
TAU_RECOVER = TAU_ACCEPT
TAU_DEFER = TAU_ACCEPT
TAU_MODE = ORIGINAL_MODE_TEMPERATURE / TEMPERATURE_DIVISOR

METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
CLAIM_LEVEL = "expressibility"
DECLARED_DEVIATIONS = [
    "E255 is a development-only optimization diagnostic of the historical "
    "conditioned-VCG controller; it changes only the four Boltzmann "
    "temperatures and makes no canonical law-condition claim."
]
ORACLE_INPUTS: list[str] = []


def method_revision() -> str:
    text = (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8")
    match = re.search(r"^\*\*Revision:\*\* `([^`]+)`", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("E255 cannot parse the current method revision")
    return match.group(1)


def development_scale_source() -> dict:
    rows = {}
    for name, path in DEVELOPMENT_SUMMARIES.items():
        if not path.is_file():
            raise RuntimeError(f"E255 requires development summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        record = summary["best_validation_record"]
        episode = int(record["checkpoint_episode"])
        value = float(record["summary"]["mean_e187_J"])
        if episode != EXPECTED_SELECTED_EPISODES[name]:
            raise RuntimeError(f"E255 selected episode changed for {name}")
        if abs(value - EXPECTED_SELECTED_J[name]) > 1.0e-9:
            raise RuntimeError(f"E255 selected development J changed for {name}")
        rows[name] = {
            "summary": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "selected_episode": episode,
            "selected_mean_J": value,
        }
    return {
        "selection_split": "existing open development validation only",
        "rows": rows,
        "five_by_five_three_seed_mean_J": FIVE_BY_FIVE_DEVELOPMENT_MEAN_J,
        "eight_by_eight_three_seed_mean_J": EIGHT_BY_EIGHT_DEVELOPMENT_MEAN_J,
        "temperature_divisor": TEMPERATURE_DIVISOR,
        "test_panel_used": False,
    }


def temperature_defaults() -> dict:
    return {
        "tau_accept": TAU_ACCEPT,
        "tau_recover": TAU_RECOVER,
        "tau_defer": TAU_DEFER,
        "tau_mode": TAU_MODE,
    }


def _conformance_declaration() -> dict:
    audit = {}
    for identifier, _description in CHECKLIST:
        if identifier == "reproducibility.method_revision":
            status = "pass"
            evidence = f"E255 pins {method_revision()} before training."
        elif identifier == "reproducibility.content_hashes":
            status = "pass"
            evidence = (
                "E255 hashes its implementation, inherited E195 seams, and "
                "all six development summaries used to derive the scale."
            )
        elif identifier == "reproducibility.oracle_declarations":
            status = "pass"
            evidence = (
                "No oracle input is used; the fixed temperature divisor is "
                "derived only from already-open development validations."
            )
        elif identifier == "reproducibility.failed_gates_retained":
            status = "pass"
            evidence = (
                "Every checkpoint and unfavorable paired development result "
                "is retained."
            )
        elif identifier == "reproducibility.preprotected_integration_gate":
            status = "not_applicable"
            evidence = (
                "E255 is development-only and opens neither a fresh nor a "
                "protected panel."
            )
        else:
            status = "not_applicable"
            evidence = (
                "E255 diagnoses temperature scaling in a historical VCG "
                "controller and makes no canonical law-condition claim."
            )
        audit[identifier] = {"status": status, "evidence": evidence}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": list(DECLARED_DEVIATIONS),
        "oracle_inputs": list(ORACLE_INPUTS),
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def contract() -> dict:
    parent = e195.contract()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "__init__.py",
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/stage.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
    )
    payload = deepcopy(parent)
    payload.pop("contract_sha256", None)
    payload.update({
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": (
            "one-seed fresh E195 rerun; only Boltzmann temperatures change; "
            "development panel only"
        ),
        "parent_protocol": e195.PROTOCOL,
        "parent_contract_sha256": parent["contract_sha256"],
        "model_seeds": [0],
        "episodes_per_seed": TOTAL_EPISODES,
        "training_jobs_per_seed": TRAINING_JOB_BUDGET,
        "temperature_scale_diagnostic": {
            "derivation": development_scale_source(),
            "original": {
                "tau_accept": ORIGINAL_WITHIN_MODE_TEMPERATURE,
                "tau_recover": ORIGINAL_WITHIN_MODE_TEMPERATURE,
                "tau_defer": ORIGINAL_WITHIN_MODE_TEMPERATURE,
                "tau_mode": ORIGINAL_MODE_TEMPERATURE,
            },
            "corrected": temperature_defaults(),
            "reward_scale_unchanged": True,
            "architecture_unchanged": True,
            "objective_unchanged": True,
            "optimizer_unchanged": True,
            "replay_capacity_unchanged": True,
            "epsilon_schedule_unchanged": True,
            "training_budget_unchanged": True,
            "certification_unchanged": True,
            "guard_unchanged": True,
            "training_instances_unchanged": True,
            "validation_panel_unchanged": True,
            "checkpoint_rule_unchanged": True,
            "only_training_change_from_e195": (
                "tau_accept, tau_recover, tau_defer, and tau_mode divided "
                "by the predeclared 8x8-to-5x5 development-cost ratio"
            ),
            "test_panel_opened": False,
        },
        "checkpoint_selection": parent["checkpoint_selection"],
        "test_panel_opened": False,
        "paper_evidence": False,
        **_conformance_declaration(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    })
    return {**payload, "contract_sha256": digest(payload)}


def trainer_protocol() -> SimpleNamespace:
    value = contract()
    return SimpleNamespace(
        PROTOCOL=PROTOCOL,
        METHOD_VERSION=METHOD_VERSION,
        SCENARIO=SCENARIO,
        SCENARIOS=SCENARIOS,
        MODEL_SEEDS=MODEL_SEEDS,
        EPISODES_BY_SCENARIO=EPISODES_BY_SCENARIO,
        TOTAL_EPISODES=TOTAL_EPISODES,
        VALIDATION_EVERY=VALIDATION_EVERY,
        CHECKPOINT_EVERY=CHECKPOINT_EVERY,
        VALIDATION_SEEDS=VALIDATION_SEEDS,
        TRAIN_BASE=TRAIN_BASE,
        TRAINING_JOB_BUDGET=TRAINING_JOB_BUDGET,
        TD_BACKUP_IMPLEMENTATION=TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=LATE_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=TargetDenseEnv,
        scenario_for=scenario_for,
        contract=lambda: value,
    )


__all__ = [
    "CHECKPOINT_EVERY", "DEFAULT_OUTPUT", "EPISODES_BY_SCENARIO",
    "EPSILON_DECAY_DECISIONS", "EPSILON_END", "EPSILON_START",
    "EPSILON_WARMUP_DECISIONS", "GAMMA", "LATE_CHECKPOINTS", "MAX_STEPS",
    "METHOD_VERSION", "MODEL_SEEDS", "PROTOCOL", "REPORT_CHECKPOINTS",
    "SCENARIO", "SCENARIOS", "SPEC", "TAU_ACCEPT", "TAU_DEFER",
    "TAU_MODE", "TAU_RECOVER", "TD_BACKUP_IMPLEMENTATION",
    "TEMPERATURE_DIVISOR", "TOTAL_EPISODES", "TRAIN_BASE",
    "TRAINING_JOB_BUDGET", "TargetDenseEnv", "VALIDATION_EVERY",
    "VALIDATION_SEEDS", "UNCHANGED_METHOD_FIELDS", "contract",
    "development_scale_source", "digest", "method_revision", "scenario_for",
    "sha256", "temperature_defaults", "trainer_protocol", "workload",
]
