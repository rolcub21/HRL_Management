"""Frozen protocol for the objective-matched Kim-inspired placement baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import protocol as e191
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import protocol as e195
from projects.temporal_consequence_realization.method_conformance import CHECKLIST, validate_protocol


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e247_kim2020_objective_matched_a3c_306k_v1"
METHOD_NAME = "kim2020_objective_matched_placement_a3c"
DISPLAY_NAME = "Kim et al. (2020)-inspired placement A3C adaptation"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e247-kim2020-objective-matched-a3c-306k-v1"

SIZES = ("5x5", "8x8")
MODEL_SEEDS = (0, 1, 2)
MIN_EPISODES = 6_000
MAX_EPISODES = 11_000
VALIDATION_EVERY = 250
CHECKPOINT_EVERY = 50
PLATEAU_CHECKS = 8
PLATEAU_RELATIVE_IMPROVEMENT = 0.005
STOCHASTIC_TRIALS = 5
MAX_DEFER_STEPS = 10
LOOKAHEAD_MARGIN_STEPS = 2.0
GAMMA = 1.0
REWARD_SCALE = 0.01
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"

SIZE = {
    "5x5": {
        "scenario": e195.SCENARIO,
        "objective": e195.SPEC,
        "training_seed_base": 247_500_000,
        "validation_seeds": tuple(e195.workload.VALIDATION_SEEDS),
        "max_steps": int(e195.MAX_STEPS),
    },
    "8x8": {
        "scenario": e191.SCENARIO,
        "objective": e191.SPEC,
        "training_seed_base": 247_800_000,
        "validation_seeds": tuple(e191.workload.VALIDATION_SEEDS),
        "max_steps": int(e191.MAX_STEPS),
    },
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def method_revision() -> str:
    text = (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8")
    match = re.search(r"^\*\*Revision:\*\* `([^`]+)`", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("E247 cannot parse the method revision")
    return match.group(1)


def _conformance() -> dict:
    audit = {}
    for identifier, _description in CHECKLIST:
        if identifier == "reproducibility.method_revision":
            audit[identifier] = {"status": "pass", "evidence": f"E247 pins {method_revision()}."}
        elif identifier == "reproducibility.content_hashes":
            audit[identifier] = {"status": "pass", "evidence": "Protocol, implementation, scheduler, and checkpoints are content-hashed."}
        elif identifier == "reproducibility.oracle_declarations":
            audit[identifier] = {"status": "pass", "evidence": "No oracle input is used; only arrived blocks and current physical state enter the policy."}
        elif identifier == "reproducibility.failed_gates_retained":
            audit[identifier] = {"status": "pass", "evidence": "Incomplete episodes, skipped censored updates, and validation history are retained."}
        elif identifier == "reproducibility.preprotected_integration_gate":
            audit[identifier] = {"status": "not_applicable", "evidence": "E247 training is development-only; finalized checkpoint loading is verified before its separately commanded sealed evaluation."}
        else:
            audit[identifier] = {"status": "not_applicable", "evidence": "E247 is an external placement-policy comparator, not a canonical TCR law or belief-control claim."}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": ["External Kim-inspired placement comparator; canonical TCR representation and learning requirements do not apply."],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "__init__.py", directory / "source.py",
        directory / "protocol.py", directory / "run.py",
        directory / "run.sh", directory / "README.md",
        ROOT / "PSLAP/kim2020_a3c_spatial.py",
        ROOT / "experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py",
        ROOT / "example/Options/selector_v5.py",
        ROOT / "example/controller_options.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method": METHOD_NAME,
        "display_name": DISPLAY_NAME,
        "paper_doi": "10.1080/00207543.2020.1748247",
        "adaptation_not_reproduction": True,
        "learned_scope": "inbound placement cell only",
        "placement_frontier": "all physically legal cells; no certification",
        "scheduler": "E25 common Dynamic/GA rule-based scheduler",
        "vcg_guard_used": False,
        "certification_used": False,
        "state": "five-channel relative dwell, occupancy, storage mask, traversability, exit proximity",
        "objective": "J=2T+E+0.5W+0.1L+10R",
        "return": "undiscounted Monte Carlo future -0.01J at placement decisions",
        "sizes": list(SIZES),
        "model_seeds": list(MODEL_SEEDS),
        "minimum_episodes": MIN_EPISODES,
        "maximum_episodes": MAX_EPISODES,
        "validation_every": VALIDATION_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
        "plateau_checks": PLATEAU_CHECKS,
        "plateau_relative_improvement": PLATEAU_RELATIVE_IMPROVEMENT,
        "stochastic_trials_per_instance": STOCHASTIC_TRIALS,
        "checkpoint_selection": "strict completion first, then minimum mean J",
        "failure_reporting": "completion unconditional; J only among strict completers",
        **_conformance(),
        "size_protocols": {
            size: {
                **{key: value for key, value in spec.items() if key != "objective"},
                "validation_seeds": list(spec["validation_seeds"]),
            }
            for size, spec in SIZE.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}
