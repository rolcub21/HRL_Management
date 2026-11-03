"""Protocol for Stage-B replication on finalized E218 seeds 1 and 2."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E216_e214_stage_b_procedure_freeze_275k_v1 import (
    protocol as e216,
)
from experiments.conditioned_vcg.E218_e215_long_exploration_replication_277k_v1 import (
    protocol as e218,
)
from experiments.conditioned_vcg.E220_e215_ep300_stage_b_279k_v1 import (
    protocol as e220,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e221_e218_multiseed_stage_b_280k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"


def _current_method_revision() -> str:
    reference = ROOT / METHOD_REFERENCE
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        reference.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"E221 cannot parse method revision from {reference}")
    return match.group(1)


METHOD_REVISION = _current_method_revision()
MODEL_SEEDS = (1, 2)
SCENARIO = e191.SCENARIO
ROUNDS = 4
EPISODES_PER_ROUND = 9
TRAIN_EPISODES_PER_SEED = ROUNDS * EPISODES_PER_ROUND
TRACE_ETA = 0.99
OPTIMIZER_LR = 2.0e-4
FIT_EPOCHS_PER_ROUND = 40
FIT_BATCH_SIZE = 128
HEAD_SEED_BASE = 280_221_800
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e221-e218-multiseed-stage-b-280k-v1"
)
SOURCE_OUTPUT = e218.DEFAULT_OUTPUT
SOURCE_CONTRACT = SOURCE_OUTPUT / "contract.json"
STAGE_B_PROCEDURE = deepcopy(e216.STAGE_B_PROCEDURE)


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


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E221 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E221 source artifact: {path}")
    return value


def source_directory(model_seed: int) -> Path:
    if int(model_seed) not in MODEL_SEEDS:
        raise ValueError("E221 model seed must be 1 or 2")
    return (
        SOURCE_OUTPUT / "training/operational" / SCENARIO
        / f"seed-{int(model_seed)}"
    )


def frozen_stage_a_identity(model_seed: int) -> dict:
    seed = int(model_seed)
    directory = source_directory(seed)
    source_contract = _read(SOURCE_CONTRACT)
    training_contract_path = directory / "training-contract.json"
    summary_path = directory / "training-summary.json"
    checkpoint_path = directory / "best.pth"
    training_contract = _read(training_contract_path)
    summary = _read(summary_path)
    selected = summary.get("best_validation_record", {})
    expected_episode = {1: 280, 2: 220}[seed]
    if (
        source_contract.get("protocol") != e218.PROTOCOL
        or summary.get("status") != "complete"
        or int(summary.get("model_seed", -1)) != seed
        or int(summary.get("completed_training_episodes", -1)) != 300
        or summary.get("protocol_training_complete") is not True
        or summary.get("selection_finalized_after_total_episodes") is not True
        or int(selected.get("checkpoint_episode", -1)) != expected_episode
        or summary.get("best_checkpoint_sha256") != sha256(checkpoint_path)
    ):
        raise RuntimeError(f"E221 requires finalized E218 seed {seed}")
    return {
        "source_protocol": e218.PROTOCOL,
        "source_contract_sha256": source_contract["contract_sha256"],
        "model_seed": seed,
        "selected_episode": expected_episode,
        "checkpoint_role": "best_deployment_finalized",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "training_contract": str(training_contract_path),
        "training_contract_sha256": sha256(training_contract_path),
        "resume_contract_sha256": training_contract["resume_contract_sha256"],
        "selection_score": list(selected["selection_score"]),
        "selection_score_fields": list(selected["selection_score_fields"]),
    }


def head_seed(model_seed: int) -> int:
    if int(model_seed) not in MODEL_SEEDS:
        raise ValueError("E221 model seed must be 1 or 2")
    return HEAD_SEED_BASE + 1000 * int(model_seed)


def training_schedule() -> tuple[dict, ...]:
    # Exact pairing with E220 seed 0; only the base policy and head seed differ.
    return tuple(deepcopy(row) for row in e220.training_schedule())


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E221 replicates a historical conditioned-VCG controller head "
                "and makes no canonical law-condition representation claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    entries = {
        "reproducibility.method_revision": (
            "E221 pins the current method-reference revision at preparation."
        ),
        "reproducibility.content_hashes": (
            "Both E218 contracts, summaries, selected checkpoints, the frozen "
            "E216 procedure, paired schedule, and behavioral sources are hashed."
        ),
        "reproducibility.oracle_declarations": (
            "No oracle input or forced-prefix intervention is used."
        ),
        "reproducibility.failed_gates_retained": (
            "Every round's calibration and collection rows are retained."
        ),
    }
    for identifier, evidence in entries.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    if "reproducibility.preprotected_integration_gate" in audit:
        audit["reproducibility.preprotected_integration_gate"] = {
            "status": "not_applicable",
            "evidence": "E221 trains on development cases and opens no protected panel.",
        }
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E221 is a historical conditioned-VCG optimization replication, "
            "not evidence for the canonical law-condition method."
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
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/returns.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E220_e215_ep300_stage_b_279k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "train_vcg_preference_conditioned.py",
    )


def contract() -> dict:
    frozen = [frozen_stage_a_identity(seed) for seed in MODEL_SEEDS]
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": (
            "development-only Stage-B replication for finalized E218 model "
            "seeds 1 and 2; no test panel"
        ),
        "source_stage_a_protocol": e218.PROTOCOL,
        "source_stage_b_freeze_protocol": e216.PROTOCOL,
        "source_stage_b_procedure": deepcopy(STAGE_B_PROCEDURE),
        "changes_from_frozen_e216_procedure": {
            "frozen_stage_a": "finalized E218 seeds 1 and 2",
            "head_initialization": "independent deterministic head seed per model seed",
            "collection_instances": "exactly paired with E220 seed 0",
            "endpoint_evaluation": "deferred to a versioned three-seed paired panel",
            "causal_replays": "omitted because E213's forced prefix is specific to old E191",
            "method_revision": METHOD_REVISION,
        },
        "frozen_stage_a": frozen,
        "scenario": SCENARIO,
        "model_seeds": list(MODEL_SEEDS),
        "head_seeds": {str(seed): head_seed(seed) for seed in MODEL_SEEDS},
        "objective_J10": deepcopy(e191.contract()["objective"]),
        "quantity": STAGE_B_PROCEDURE["predicted_quantity"],
        "trace_target": STAGE_B_PROCEDURE["target"],
        "trace_eta": TRACE_ETA,
        "deployment_merit": STAGE_B_PROCEDURE["deployment_merit"],
        "preference_semantics": STAGE_B_PROCEDURE["preference_semantics"],
        "supported_alpha_range": deepcopy(STAGE_B_PROCEDURE["supported_alpha_range"]),
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "training_schedule": list(training_schedule()),
        "schedule_pairing": "byte-equivalent coordinate values to E220 seed 0",
        "fit": deepcopy(STAGE_B_PROCEDURE["fit"]),
        "unchanged_frontier_certification_guard": True,
        "unchanged_acceleration": deepcopy(e191.contract()["acceleration_assertion"]),
        "training_runs": len(MODEL_SEEDS),
        "expected_complete_episode_runs": len(MODEL_SEEDS) * TRAIN_EPISODES_PER_SEED,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "digest", "frozen_stage_a_identity", "head_seed",
    "method_declaration", "sha256", "source_directory", "training_schedule",
]
