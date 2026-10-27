"""Frozen development protocol for the E110 guarded counterfactual."""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    protocol as e107,
)
from experiments.conditioned_vcg.E108_e107_low_epsilon_continuation_8x8_172k_v1 import (
    protocol as e108,
)


ROOT = e107.ROOT
PROTOCOL = "vcg_conditioned_e110_guarded_deliver_accept_counterfactual_8x8_174k_v1"
SCENARIO = e107.SCENARIO
ANCHOR_EPISODE = 240
LATE_EPISODE = 269
DEVELOPMENT_SEEDS = tuple(e107.e105.e78.VALIDATION_SEEDS[SCENARIO])
MAX_STEPS = e107.e105.e78.SCENARIOS[SCENARIO].max_steps
MIN_MATCHED_PAIRS = 2
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e110-guarded-deliver-accept-counterfactual-8x8-174k-v1"
)
E108_TRAINING = (
    e108.DEFAULT_OUTPUT / "training/operational" / e108.SCENARIO / "seed-0"
)
ANCHOR_SOURCE = E108_TRAINING / "best-candidate-slot-a.pth"
LATE_SOURCE = E108_TRAINING / "latest.pth"
E108_SUMMARY = E108_TRAINING / "training-summary.json"
E109_REPORT = (
    ROOT
    / "results/vcg-conditioned-e109-checkpoint-value-drift-audit-8x8-173k-v1"
    / "e109-report.json"
)
ORIGINAL_CONTRACT_SHA256 = (
    "db6eccd9e8b5603cf14bc0ef86834ce429facee40ce27f206c237cba08a2e735"
)
# E108's resumable ``latest.pth`` advanced after E110 had authenticated and
# compacted episode 269.  Keep the exact preparation-time identity here so all
# later commands bind to the immutable E110 copies rather than re-reading a
# moving training path.
FROZEN_SOURCE_IDENTITY = {
    "anchor": {
        "bytes": 3_225_196,
        "episode": ANCHOR_EPISODE,
        "path": str(ANCHOR_SOURCE.relative_to(ROOT)),
        "sha256": "c867d105ad6266d7fc401fe215f1a6815ff3208b5a1026bcaa5372d9b7e01af8",
    },
    "late": {
        "bytes": 1_021_155_106,
        "episode": LATE_EPISODE,
        "path": str(LATE_SOURCE.relative_to(ROOT)),
        "sha256": "29ba0a66cfe3675aa6d89e426c61ccb6d1371c1d66622fc6809c8cbc069e1f6a",
    },
    "e108_summary_sha256": (
        "f3598eadb469d5e9239742e2304cdf4c1dee7e150d65213fb4db7369fc172d0a"
    ),
    "e109_report_sha256": (
        "9a8a0a72f6fa08512dff3bdff2778ebe62475d4bf76878e98af8cb994eaaefa5"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"E110 missing source artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"E110 source artifact is not an object: {path}")
    return value


@lru_cache(maxsize=1)
def source_identity() -> dict:
    return deepcopy(FROZEN_SOURCE_IDENTITY)


@lru_cache(maxsize=1)
def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT
        / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT
        / "experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/rewards.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "development-only matched one-action guarded continuations",
        "scenario": SCENARIO,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "panel_role": "reused opened development validation panel",
        "source_identity": source_identity(),
        "contract_repair": (
            "repair1: authenticate immutable compact episode-240/269 copies; "
            "do not re-read mutable E108 latest metadata after freezing"
        ),
        "fork_definition": (
            "first unforced boundary per instance where the frozen episode-269 "
            "controller selects Accept and the episode-240 network selects Deliver "
            "from the identical liveness-admissible exact-SAFE frontier"
        ),
        "reference_arm": (
            "execute episode-269 Accept and continue with frozen episode-269 guarded policy"
        ),
        "counterfactual_arm": (
            "execute episode-240-selected Deliver once, then continue with the same "
            "frozen episode-269 guarded policy"
        ),
        "prefix_matching": [
            "same EpisodeInstance",
            "same native selected-key prefix",
            "same environment-state digest",
            "same admissible candidate-bank digest",
            "same recovery-witness guard digest",
            "same prefix reward",
        ],
        "objective": e107.contract()["objective"],
        "max_steps": MAX_STEPS,
        "minimum_matched_pairs_for_directional_summary": MIN_MATCHED_PAIRS,
        "no_learning": True,
        "no_checkpoint_write": True,
        "independent_test_opened": False,
        "causal_scope": "one-action effect under the fixed episode-269 continuation only",
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}
