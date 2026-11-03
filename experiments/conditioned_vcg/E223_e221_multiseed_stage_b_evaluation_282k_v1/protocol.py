"""Protocol for paired evaluation of finalized E221 Stage-B heads."""

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
from experiments.conditioned_vcg.E216_e214_stage_b_procedure_freeze_275k_v1 import (
    protocol as e216,
)
from experiments.conditioned_vcg.E218_e215_long_exploration_replication_277k_v1 import (
    protocol as e218,
)
from experiments.conditioned_vcg.E220_e215_ep300_stage_b_279k_v1 import (
    protocol as e220,
)
from experiments.conditioned_vcg.E221_e218_multiseed_stage_b_280k_v1 import (
    protocol as e221,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e223_e221_multiseed_stage_b_evaluation_282k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"


def _current_method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E223 cannot parse the current method revision")
    return match.group(1)


METHOD_REVISION = _current_method_revision()
MODEL_SEEDS = e221.MODEL_SEEDS
SCENARIO = e191.SCENARIO
TRACE_ETA = e221.TRACE_ETA
ROUNDS = e221.ROUNDS
EVAL_INSTANCE_SEEDS = e220.EVAL_INSTANCE_SEEDS
EVAL_LAMBDAS = e220.EVAL_LAMBDAS
IMMEDIATE_ONLY_LAMBDAS = e220.IMMEDIATE_ONLY_LAMBDAS
MAX_RELATIVE_J_REGRESSION = e220.MAX_RELATIVE_J_REGRESSION
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e223-e221-multiseed-stage-b-evaluation-282k-v1"
)
SOURCE_OUTPUT = e221.DEFAULT_OUTPUT
SOURCE_CONTRACT = SOURCE_OUTPUT / "contract.json"


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
        raise RuntimeError(f"missing E223 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E223 source artifact: {path}")
    return value


def source_stage_b_identity(model_seed: int) -> dict:
    seed = int(model_seed)
    if seed not in MODEL_SEEDS:
        raise ValueError("E223 model seed must be 1 or 2")
    source_contract = _read(SOURCE_CONTRACT)
    if source_contract.get("protocol") != e221.PROTOCOL:
        raise RuntimeError("E223 E221 source protocol differs")
    frozen = [
        row for row in source_contract["frozen_stage_a"]
        if int(row["model_seed"]) == seed
    ]
    if len(frozen) != 1:
        raise RuntimeError(f"E223 E221 seed {seed} Stage A is not unique")
    training = SOURCE_OUTPUT / "training" / f"seed-{seed}"
    endpoint = training / "checkpoints" / f"round-{ROUNDS}.pth"
    summary = training / "training-summary.json"
    summary_payload = _read(summary)
    if (
        summary_payload.get("status") != "complete"
        or int(summary_payload.get("completed_rounds", -1)) != ROUNDS
        or int(summary_payload.get("completed_training_episodes", -1))
        != e221.TRAIN_EPISODES_PER_SEED
        or not endpoint.is_file()
    ):
        raise RuntimeError(f"E223 requires completed E221 seed {seed}")
    return {
        "source_protocol": e221.PROTOCOL,
        "source_contract_sha256": source_contract["contract_sha256"],
        "model_seed": seed,
        "selected_stage_a_episode": int(frozen[0]["selected_episode"]),
        "stage_a_checkpoint_sha256": frozen[0]["checkpoint_sha256"],
        "endpoint_round": ROUNDS,
        "endpoint": str(endpoint.resolve()),
        "endpoint_sha256": sha256(endpoint),
        "summary": str(summary.resolve()),
        "summary_sha256": sha256(summary),
    }


def workload_for_seed(seed: int) -> float:
    return float(e220.workload_for_seed(int(seed)))


def head_seed(model_seed: int) -> int:
    return int(e221.head_seed(int(model_seed)))


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E223 evaluates a historical conditioned-VCG controller head "
                "and makes no canonical law-condition representation claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    entries = {
        "reproducibility.method_revision": (
            "E223 pins the current method-reference revision at preparation."
        ),
        "reproducibility.content_hashes": (
            "Both E221 endpoints, summaries, frozen Stage-A identities, paired "
            "instances, and every behavioral source are content-hashed."
        ),
        "reproducibility.oracle_declarations": (
            "No oracle input or forced-prefix intervention is used."
        ),
        "reproducibility.failed_gates_retained": (
            "Completion, calibration, and all learned-versus-immediate rows are retained."
        ),
    }
    for identifier, evidence in entries.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    audit["reproducibility.preprotected_integration_gate"] = {
        "status": "not_applicable",
        "evidence": "E223 is development-only and opens no protected panel.",
    }
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E223 is a historical conditioned-VCG optimization evaluation, "
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
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E221_e218_multiseed_stage_b_280k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E221_e218_multiseed_stage_b_280k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E220_e215_ep300_stage_b_279k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/returns.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/immediate.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "train_vcg_preference_conditioned.py",
    )


def contract() -> dict:
    frozen = [source_stage_b_identity(seed) for seed in MODEL_SEEDS]
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": (
            "paired development evaluation of finalized E221 Stage-B model "
            "seeds 1 and 2; no test panel"
        ),
        "source_stage_a_protocol": e218.PROTOCOL,
        "source_stage_b_training_protocol": e221.PROTOCOL,
        "source_stage_b_freeze_protocol": e216.PROTOCOL,
        "source_stage_b_procedure": deepcopy(e221.STAGE_B_PROCEDURE),
        "frozen_stage_b": frozen,
        "scenario": SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "model_seeds": list(MODEL_SEEDS),
        "objective_J10": deepcopy(e191.contract()["objective"]),
        "quantity": e221.STAGE_B_PROCEDURE["predicted_quantity"],
        "trace_target": e221.STAGE_B_PROCEDURE["target"],
        "trace_eta": TRACE_ETA,
        "deployment_merit": e221.STAGE_B_PROCEDURE["deployment_merit"],
        "preference_semantics": e221.STAGE_B_PROCEDURE["preference_semantics"],
        "endpoint_evaluation": {
            "instance_seeds": list(EVAL_INSTANCE_SEEDS),
            "workload_levels": [workload_for_seed(seed) for seed in EVAL_INSTANCE_SEEDS],
            "learned_lambdas": list(EVAL_LAMBDAS),
            "immediate_only_lambdas": list(IMMEDIATE_ONLY_LAMBDAS),
            "pairing": "exact E220 development instances for every seed and controller arm",
            "episode_runs_per_seed": (
                len(EVAL_INSTANCE_SEEDS)
                * (len(EVAL_LAMBDAS) + len(IMMEDIATE_ONLY_LAMBDAS))
            ),
        },
        "predeclared_gates": {
            "calibration": "sample-weighted held-out MC MAE is below exact zero-head MC MAE",
            "frontier_value": "learned future head has lower J_alpha than immediate-only in at least one paired case",
            "no_catastrophic_regression": (
                f"every paired J_alpha is at most "
                f"{1.0 + MAX_RELATIVE_J_REGRESSION:.2f} times immediate-only"
            ),
            "strict_completion": "every learned and immediate-only episode completes strictly",
        },
        "unchanged_frontier_certification_guard": True,
        "unchanged_acceleration": deepcopy(e191.contract()["acceleration_assertion"]),
        "training_runs": 0,
        "expected_complete_episode_runs": (
            len(MODEL_SEEDS) * len(EVAL_INSTANCE_SEEDS)
            * (len(EVAL_LAMBDAS) + len(IMMEDIATE_ONLY_LAMBDAS))
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "digest", "head_seed", "method_declaration", "sha256",
    "source_stage_b_identity", "workload_for_seed",
]
