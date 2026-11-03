"""Locked protocol for the E225 one-seed guard-free replay."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import (
    protocol as e222,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e225_e215_seed0_guard_free_development_284k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
MODEL_SEED = 0
INSTANCE_SEEDS = tuple(int(seed) for seed in e191_scenario.VALIDATION_SEEDS)
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e225-e215-seed0-guard-free-development-284k-v1"
)
REPORT_NAME = "REPORT.txt"
REPORT_JSON = "e225-report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E225 cannot parse the current method revision")
    return match.group(1)


def checkpoint_identity() -> dict:
    value = e222.checkpoint_identity(MODEL_SEED)
    if int(value["selected_episode"]) != 300:
        raise RuntimeError("E225 requires the finalized E215 episode-300 selection")
    return value


def workload_for_seed(seed: int) -> str:
    level = e191_scenario.target_level(int(seed))
    return f"{100.0 * level:.0f}%"


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E225 is a development-only ablation of a historical "
                "conditioned-VCG controller, not a canonical law-condition "
                "representation claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    entries = {
        "reproducibility.method_revision": (
            "E225 pins the current method-reference revision at preparation."
        ),
        "reproducibility.content_hashes": (
            "The selected checkpoint, embedded reference trajectories, "
            "implementation, protocol, and development coordinates are hashed."
        ),
        "reproducibility.oracle_declarations": (
            "No oracle input is used; the guard-removal intervention is explicit."
        ),
        "reproducibility.failed_gates_retained": (
            "Incomplete episodes, horizon failures, and every paired row are retained."
        ),
    }
    for identifier, evidence in entries.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    audit["reproducibility.preprotected_integration_gate"] = {
        "status": "not_applicable",
        "evidence": (
            "E225 reuses the already-open E191 development panel and opens no "
            "fresh or protected split."
        ),
    }
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": [
            "E225 removes E188 liveness masking and witness forcing from a "
            "historical frozen conditioned-VCG controller."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "guard_free.py",
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/scenario.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/guard.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/roundoff.py",
        ROOT / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
    )


def contract() -> dict:
    checkpoint = checkpoint_identity()
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "one-seed frozen development ablation of all guard intervention",
        "development_only": True,
        "test_panel_opened": False,
        "model_seed": MODEL_SEED,
        "selected_episode": int(checkpoint["selected_episode"]),
        "source_checkpoint": deepcopy(checkpoint),
        "scenario": e191.SCENARIO,
        "instance_seeds": list(INSTANCE_SEEDS),
        "workloads": {
            str(seed): workload_for_seed(seed) for seed in INSTANCE_SEEDS
        },
        "paired_reference": (
            "exact guard-on validation trajectories embedded in the selected "
            "episode-300 checkpoint"
        ),
        "guard_free_rule": (
            "prepare the complete exact-SAFE frontier; retain the frozen "
            "nonprogress context features; never mask Reconfigure, activate a "
            "witness, restrict the frontier, or bypass critic selection"
        ),
        "evaluation_epsilon": 0.0,
        "evaluation_learning": False,
        "objective": deepcopy(e191.contract()["objective"]),
        "max_steps": e191_scenario.MAX_STEPS,
        "accelerations_retained": [
            "relocation-family certification",
            "timing-invariant certificate cache",
            "fast successor inputs",
            "deduplicated queue scoring",
            "roundoff-repaired objective audit",
        ],
        "episode_runs": len(INSTANCE_SEEDS),
        "training_runs": 0,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "DEFAULT_OUTPUT", "INSTANCE_SEEDS", "MODEL_SEED", "PROTOCOL",
    "REPORT_JSON", "REPORT_NAME", "ROOT", "checkpoint_identity", "contract",
    "digest", "method_declaration", "sha256", "workload_for_seed",
]

