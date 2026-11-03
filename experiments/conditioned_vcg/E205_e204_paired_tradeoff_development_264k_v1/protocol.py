"""Locked identities for E204's small paired development panel."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    protocol as e204,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e205_e204_paired_tradeoff_development_264k_v1"
MODEL_SEED = 0
SCENARIO = e191.SCENARIO
LAMBDAS = e204.LAMBDAS
LAMBDA_CENTER = e204.LAMBDA_CENTER
WORKLOAD_LEVELS = tuple(float(value) for value in e191_scenario.LEVELS)
INSTANCE_BASE = 264_205_200  # divisible by three; modulo selects 20/50/80%
INSTANCE_SEEDS = tuple(INSTANCE_BASE + index for index in range(3))
STAGE_B_OUTPUT = e204.DEFAULT_OUTPUT
STAGE_B_DIR = (
    STAGE_B_OUTPUT / "training/handling" / SCENARIO / f"seed-{MODEL_SEED}"
)
STAGE_B_TERMINAL = STAGE_B_DIR / "terminal.pth"
STAGE_B_SUMMARY = STAGE_B_DIR / "training-summary.json"
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e205-e204-paired-tradeoff-development-264k-v1"
)
REPORT_NAME = "e205-report.json"
TABLE_NAME = "e205-paired-table.md"
PLOT_NAME = "e205-paired-tradeoff.png"
CSV_NAME = "e205-paired-rows.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E205 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E205 source artifact: {path}")
    return value


def stage_b_identity() -> dict:
    summary = _read(STAGE_B_SUMMARY)
    expected = {
        "status": "complete",
        "training_protocol": e204.PROTOCOL,
        "scenario": SCENARIO,
        "model_seed": MODEL_SEED,
        "completed_rounds": e204.ROUNDS,
        "completed_collection_episodes": e204.ROUNDS * e204.EPISODES_PER_ROUND,
        "lambda_center": LAMBDA_CENTER,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    mismatch = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"E205 E204 summary differs: {mismatch}")
    if not STAGE_B_TERMINAL.is_file():
        raise RuntimeError("E205 E204 terminal checkpoint is missing")
    return {
        "protocol": e204.PROTOCOL,
        "contract_sha256": summary["contract_sha256"],
        "model_seed": MODEL_SEED,
        "base_selected_episode": int(summary["base_selected_episode"]),
        "base_checkpoint_sha256": summary["base_checkpoint_sha256"],
        "base_policy_digest": summary["base_policy_digest"],
        "terminal_checkpoint": str(STAGE_B_TERMINAL),
        "terminal_checkpoint_sha256": sha256(STAGE_B_TERMINAL),
        "training_summary_sha256": sha256(STAGE_B_SUMMARY),
    }


def workload_for_seed(seed: int) -> float:
    seed = int(seed)
    if seed not in INSTANCE_SEEDS:
        raise ValueError("seed is outside the E205 development panel")
    value = float(e191_scenario.target_level(seed))
    expected = WORKLOAD_LEVELS[seed - INSTANCE_BASE]
    if value != expected:
        raise RuntimeError("E205 seed no longer selects its declared workload")
    return value


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "plot.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.py",
    )
    stage_b = stage_b_identity()
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "small paired development mechanism panel; not paper evidence",
        "stage_b": stage_b,
        "scenario": SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "model_seed": MODEL_SEED,
        "lambda_values": list(LAMBDAS),
        "lambda_center": LAMBDA_CENTER,
        "instance_seeds": list(INSTANCE_SEEDS),
        "workload_levels": list(WORKLOAD_LEVELS),
        "pairing": "same exact EpisodeInstance at every lambda within workload",
        "episode_runs": len(INSTANCE_SEEDS) * len(LAMBDAS),
        "primary_question": (
            "does increasing the deployment rehandle weight reduce realized "
            "physical rehandles on each matched workload?"
        ),
        "paired_effect_outcomes": [
            "physical_rehandles",
            "J10",
            "J_lambda",
            "timing_cost_2T_plus_E",
            "admission_wait_cost_0.5W",
            "equipment_cost_0.1L",
            "steps",
            "storage_clock_mae",
            "storage_within_20_rate",
            "release_clock_mae",
            "mean_stored",
            "peak_stored",
            "mean_external_queue",
        ],
        "reference": "lambda=10 exact E191 delegation",
        "J_lambda_definition": "J10 + (lambda-10)*physical_rehandles",
        "interpretation": (
            "directions of non-rehandle effects are estimated, not imposed; "
            "three cases diagnose mechanism but do not support final inference"
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "CSV_NAME",
    "DEFAULT_OUTPUT",
    "INSTANCE_BASE",
    "INSTANCE_SEEDS",
    "LAMBDA_CENTER",
    "LAMBDAS",
    "MODEL_SEED",
    "PLOT_NAME",
    "PROTOCOL",
    "REPORT_NAME",
    "SCENARIO",
    "STAGE_B_OUTPUT",
    "STAGE_B_TERMINAL",
    "TABLE_NAME",
    "WORKLOAD_LEVELS",
    "contract",
    "digest",
    "sha256",
    "stage_b_identity",
    "workload_for_seed",
]
