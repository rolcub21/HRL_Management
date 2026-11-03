"""Locked seed-0 protocol for the 8x8 on-policy trace Stage-B repair."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
)
from experiments.conditioned_vcg.E207_e204_immediate_only_ablation_266k_v1 import (
    protocol as e207,
)
from experiments.conditioned_vcg.E208_on_policy_td_rehandle_stage_b_267k_v1 import (
    protocol as e208,
)
from experiments.conditioned_vcg.E213_post_placement_continuation_audit_272k_v1 import (
    protocol as e213,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e214_on_policy_trace_rehandle_stage_b_8x8_273k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-29.8"
MODEL_SEED = 0
SCENARIO = e191.SCENARIO
ROUNDS = 4
EPISODES_PER_ROUND = 9
TRAIN_EPISODES = ROUNDS * EPISODES_PER_ROUND
ALPHA_MIN = 10.0
ALPHA_MAX = 40.0
ALPHA_BANDS = ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
EVAL_LAMBDAS = (10.0, 20.0, 40.0)
TRACE_ETA = 0.99
TRAIN_BASE = 273_214_101  # divisible by three
SCHEDULE_SEED = 273_214_700
HEAD_SEED = 273_214_800
OPTIMIZER_LR = 2.0e-4
FIT_EPOCHS_PER_ROUND = 40
FIT_BATCH_SIZE = 128
MAX_RELATIVE_J_REGRESSION = 0.25
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e214-on-policy-trace-rehandle-stage-b-8x8-273k-v1"
E213_REPORT = e213.DEFAULT_OUTPUT / "e213-report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _read(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing E214 source artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E214 source artifact: {path}")
    return value


def freeze_identity() -> dict:
    return deepcopy(e208.freeze_identity())


def round_schedule(round_index: int) -> tuple[dict, ...]:
    if not 0 <= int(round_index) < ROUNDS:
        raise ValueError("invalid E214 round")
    rng = random.Random(SCHEDULE_SEED + 100 * int(round_index))
    origin = TRAIN_BASE + 300 * int(round_index)
    rows = []
    for band_index, (lower, upper) in enumerate(ALPHA_BANDS):
        for workload_index, workload in enumerate(e191_scenario.LEVELS):
            seed = origin + band_index * 3 + workload_index
            rows.append({
                "round": int(round_index) + 1,
                "instance_seed": seed,
                "workload": float(workload),
                "alpha": float(lower + (upper - lower) * rng.random()),
                "alpha_band": [lower, upper],
            })
    rng.shuffle(rows)
    for offset, row in enumerate(rows):
        row["round_episode"] = offset + 1
        row["episode"] = int(round_index) * EPISODES_PER_ROUND + offset + 1
        if float(e191_scenario.target_level(row["instance_seed"])) != row["workload"]:
            raise RuntimeError("E214 schedule workload identity differs")
    return tuple(rows)


def training_schedule() -> tuple[dict, ...]:
    return tuple(
        row for round_index in range(ROUNDS) for row in round_schedule(round_index)
    )


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E214 is a VCG controller-head diagnostic and does not train or "
                "evaluate the canonical law-condition representation."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": "The E214 contract pins the current method revision.",
        "reproducibility.content_hashes": "Sources, frozen E191, schedules, paired rows, causal anchors, and checkpoints are hashed.",
        "reproducibility.oracle_declarations": "The E212/E213 prefix intervention is declared as a causal diagnostic deviation, not a deployable input.",
        "reproducibility.failed_gates_retained": "Calibration, causal, frontier, collapse, and price-regret failures are retained.",
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E214 diagnoses preference-conditioned VCG control rather than the canonical law-condition representation.",
            "Two development replays force E212/E213's observed prefix to test the previously isolated failure causally.",
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    frozen = freeze_identity()
    e213_report = _read(E213_REPORT)
    if e213_report.get("status") != "complete":
        raise RuntimeError("E214 requires completed E213 causal anchors")
    immediate_rows = tuple(
        e207.DEFAULT_OUTPUT / "rows" / f"rho-{rho}" / f"lambda-{value:g}.json"
        for rho in (20, 50, 80)
        for value in (20.0, 40.0)
    )
    sources = tuple(directory / name for name in (
        "returns.py", "protocol.py", "run.py", "plot.py", "run.sh", "README.md"
    )) + (
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E205_e204_paired_tradeoff_development_264k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/immediate.py",
        ROOT / "experiments/conditioned_vcg/E213_post_placement_continuation_audit_272k_v1/run.py",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "seed-0 8x8 development repair pilot; no sealed test panel and no multiseed claim",
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": e191.contract()["contract_sha256"],
        "frozen_stage_a": [frozen],
        "scenario": SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "model_seed": MODEL_SEED,
        "objective_J10": deepcopy(e191.contract()["objective"]),
        "quantity": "N_alpha(s,c)=E[sum_{k>0} I(reconfigure_k) | s,c,then follow pi_alpha]",
        "trace_target": "G_t=I(c_next)+(1-eta)N_round_start(s_next,c_next,alpha)+eta*G_next",
        "trace_eta": TRACE_ETA,
        "behavior_target_relation": (
            "one frozen round-start head selects all nine current-round episodes "
            "and supplies their bootstrap terms"
        ),
        "deployment_merit": "Q_E191 - 0.01*(alpha-10)*(I_reconfigure+N_alpha)",
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "round_data_retention": "current round only; discard after fit",
        "alpha_schedule": "continuous alpha per workload x alpha band per round; fixed within episode",
        "alpha_range": [ALPHA_MIN, ALPHA_MAX],
        "alpha_bands": [list(value) for value in ALPHA_BANDS],
        "training_schedule": list(training_schedule()),
        "fit": {
            "epochs_per_round": FIT_EPOCHS_PER_ROUND,
            "batch_size": FIT_BATCH_SIZE,
            "learning_rate": OPTIMIZER_LR,
            "loss": "smooth_l1 on trace return",
            "gradient_clip": 5.0,
            "primary_calibration": "held-out MC future-count MAE against exact zero-head MAE",
            "secondary_calibration": "held-out trace-target MAE",
            "initialization": "fresh near-zero future-count head; immediate-only initial policy",
        },
        "endpoint_evaluation": {
            "lambdas": list(EVAL_LAMBDAS),
            "instances": "exact E205 three-day development panel",
            "instances_sha256": sha256(e205.DEFAULT_OUTPUT / "instances.json"),
            "learned_episode_runs": 9,
            "immediate_only_episode_runs": 0,
            "immediate_only_source": e207.PROTOCOL,
            "immediate_only_rows": {
                str(path.relative_to(ROOT)): sha256(path) for path in immediate_rows
            },
        },
        "causal_replays": {
            "source_report": str(E213_REPORT.relative_to(ROOT)),
            "source_report_sha256": sha256(E213_REPORT),
            "heavy_instance_seed": e213.HEAVY_INSTANCE_SEED,
            "divergence_index": e213.DIVERGENCE_INDEX,
            "old_failed_placement": e213.FORCED_PLACEMENT_KEY,
            "forced_prefix": list(e213_report["forced_prefix"]),
            "post_placement_handoff_signature": deepcopy(
                e213_report["post_placement_handoff_signature"]
            ),
            "anchors": deepcopy(e213_report["natural_baselines"]),
            "episode_runs": 2,
        },
        "predeclared_gates": {
            "calibration": "sample-weighted held-out MC MAE is strictly below the zero-head MC MAE",
            "decision_7": "after the exact six-decision E212 prefix, repaired lambda20 does not choose accept:B3:6:6",
            "forced_continuation": "after the exact seven-decision E213 prefix, repaired lambda20 rehandles are closer to immediate-only than to the 309-rehandle collapse",
            "frontier_value": "at least one lambda20/40 paired row has lower J_lambda than immediate-only",
            "no_catastrophic_regression": f"every paired J_lambda is at most {1.0 + MAX_RELATIVE_J_REGRESSION:.2f} times immediate-only",
            "strict_completion": "all learned and causal episodes complete strictly",
        },
        "unchanged_frontier_certification_guard": True,
        "unchanged_acceleration": deepcopy(e191.contract()["acceleration_assertion"]),
        "training_runs": 1,
        "expected_complete_episode_runs": TRAIN_EPISODES + 11,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "digest", "freeze_identity", "method_declaration",
    "round_schedule", "sha256", "training_schedule",
]
