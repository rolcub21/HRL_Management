"""Locked protocol for the corrected-controller 5x5 simple-dial curve."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import torch

from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import protocol as e195
from experiments.conditioned_vcg.E201_e195_matched_5x5_test_panel_260k_v1 import protocol as e201
from experiments.conditioned_vcg.E201_e195_matched_5x5_test_panel_260k_v1 import scenario as e201_scenario
from experiments.conditioned_vcg.E255_e195_temperature_scaled_5x5_314k_v1 import protocol as e255
from experiments.conditioned_vcg.E256_e255_temperature_scaled_5x5_panel_315k_v1 import protocol as e256
from experiments.conditioned_vcg.E258_e255_temperature_scaled_replication_5x5_317k_v1 import protocol as e258
from experiments.conditioned_vcg.E261_e255_e258_three_seed_5x5_panel_320k_v1 import protocol as e261
from projects.temporal_consequence_realization.method_conformance import CHECKLIST, validate_protocol


ROOT = e195.ROOT
PROTOCOL = "vcg_conditioned_e262_e255_e258_simple_dial_curve_5x5_321k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
MODEL_SEEDS = (0, 1, 2)
LAMBDA_CENTER = 10.0
FULL_LAMBDAS = (10.0, 20.0, 40.0, 80.0, 160.0)
NEW_LAMBDAS = (20.0, 40.0, 80.0, 160.0)
SELECTED_EPISODES = dict(e261.SELECTED_EPISODES)
SOURCE_PROTOCOLS = dict(e261.SOURCE_PROTOCOLS)
SOURCE_SUMMARIES = dict(e261.SOURCE_SUMMARIES)
SOURCE_CHECKPOINTS = dict(e261.SOURCE_CHECKPOINTS)
SOURCE_OUTPUT = e201.DEFAULT_OUTPUT
SOURCE_CONTRACT = SOURCE_OUTPUT / "contract.json"
SOURCE_MANIFEST = SOURCE_OUTPUT / "instances.json"
SOURCE_REPORT = SOURCE_OUTPUT / e201.REPORT_NAME
CURRENT_PANEL_OUTPUTS = {0: e256.DEFAULT_OUTPUT, 1: e261.DEFAULT_OUTPUT, 2: e261.DEFAULT_OUTPUT}
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e262-e255-e258-simple-dial-curve-5x5-321k-v1"
GATE_OUTPUT = Path(str(DEFAULT_OUTPUT) + "-gate")
REPORT_NAME = "e262-report.json"
REPORT_TEXT_NAME = "REPORT.txt"
TABLE_NAME = "e262-table.md"
PLOT_NAME = "e262-dial-curve.png"
EXPECTED_SOURCE_ROWS = e201.EXPECTED_ROWS
EXPECTED_REUSED_LAMBDA10_ROWS = len(MODEL_SEEDS) * len(e201_scenario.TEST_SEEDS)
EXPECTED_NEW_ROWS = len(NEW_LAMBDAS) * EXPECTED_REUSED_LAMBDA10_ROWS
GATE_MODEL_SEED = 0
GATE_INSTANCE_SEED = int(e201_scenario.TEST_SEEDS[0])


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
        raise RuntimeError(f"missing E262 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E262 source artifact: {path}")
    return value


def method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"), re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E262 cannot parse the current method revision")
    return match.group(1)


def current_panel_row(model_seed: int, instance_seed: int) -> Path:
    if int(model_seed) == 0:
        return e256.DEFAULT_OUTPUT / "rows/e255-episode280" / f"instance-{int(instance_seed)}.json"
    return e261.DEFAULT_OUTPUT / f"rows/model-seed-{int(model_seed)}" / f"instance-{int(instance_seed)}.json"


def _source_row_paths() -> tuple[Path, ...]:
    values = tuple(sorted((SOURCE_OUTPUT / "rows").glob("**/*.json")))
    if len(values) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"E262 requires {EXPECTED_SOURCE_ROWS} E201 rows, found {len(values)}")
    return values


def _freeze_identity(model_seed: int) -> dict:
    summary_path = SOURCE_SUMMARIES[model_seed]
    checkpoint = SOURCE_CHECKPOINTS[model_seed]
    summary = _read(summary_path)
    best = summary.get("best_validation_record", {})
    expected = {
        "completed_training_episodes": 300,
        "selection_finalized_after_total_episodes": True,
    }
    mismatch = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatch or int(best.get("checkpoint_episode", -1)) != SELECTED_EPISODES[model_seed]:
        raise RuntimeError(f"E262 corrected seed {model_seed} selection differs: {mismatch}")
    if Path(summary.get("best_checkpoint", "")).resolve() != checkpoint.resolve():
        raise RuntimeError(f"E262 corrected seed {model_seed} selected path differs")
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != summary.get("best_checkpoint_sha256"):
        raise RuntimeError(f"E262 corrected seed {model_seed} checkpoint hash differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    payload_expected = {
        "training_protocol": SOURCE_PROTOCOLS[model_seed],
        "model_seed": model_seed,
        "selected_checkpoint_episode": SELECTED_EPISODES[model_seed],
        "completed_training_episodes": 300,
    }
    payload_mismatch = {key: (payload.get(key), value) for key, value in payload_expected.items() if payload.get(key) != value}
    if payload_mismatch:
        raise RuntimeError(f"E262 corrected seed {model_seed} metadata differs: {payload_mismatch}")
    return {
        "model_seed": model_seed,
        "training_protocol": SOURCE_PROTOCOLS[model_seed],
        "selected_episode": SELECTED_EPISODES[model_seed],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "freeze_manifest": str(summary_path.resolve()),
        "freeze_manifest_sha256": sha256(summary_path),
    }


def source_identity() -> dict:
    contract = _read(SOURCE_CONTRACT)
    core = {key: value for key, value in contract.items() if key != "contract_sha256"}
    manifest = _read(SOURCE_MANIFEST)
    report = _read(SOURCE_REPORT)
    if (
        contract.get("protocol") != e201.PROTOCOL
        or digest(core) != contract.get("contract_sha256")
        or manifest.get("protocol") != e201.PROTOCOL
        or len(manifest.get("records", ())) != len(e201_scenario.TEST_SEEDS)
        or report.get("status") != "complete"
        or int(report.get("observed_rows", -1)) != EXPECTED_SOURCE_ROWS
    ):
        raise RuntimeError("E262 requires the authenticated complete E201 panel")
    e201_rows = {str(path.relative_to(SOURCE_OUTPUT)): sha256(path) for path in _source_row_paths()}
    current_rows = {}
    for model_seed in MODEL_SEEDS:
        for instance_seed in e201_scenario.TEST_SEEDS:
            path = current_panel_row(model_seed, instance_seed)
            row = _read(path)
            if (
                int(row.get("model_seed", -1)) != model_seed
                or int(row.get("instance_seed", -1)) != int(instance_seed)
                or not bool(row.get("strict_complete"))
            ):
                raise RuntimeError(f"E262 corrected lambda-10 row differs: {path}")
            current_rows[str(path.relative_to(ROOT))] = sha256(path)
    if len(current_rows) != EXPECTED_REUSED_LAMBDA10_ROWS:
        raise RuntimeError("E262 requires 90 corrected lambda-10 rows")
    return {
        "e201_protocol": contract["protocol"],
        "e201_contract": str(SOURCE_CONTRACT.resolve()),
        "e201_contract_sha256": sha256(SOURCE_CONTRACT),
        "e201_contract_identity": contract["contract_sha256"],
        "e201_manifest": str(SOURCE_MANIFEST.resolve()),
        "e201_manifest_sha256": sha256(SOURCE_MANIFEST),
        "e201_report": str(SOURCE_REPORT.resolve()),
        "e201_report_sha256": sha256(SOURCE_REPORT),
        "e201_rows": e201_rows,
        "e201_rows_digest": digest(e201_rows),
        "corrected_lambda10_rows": current_rows,
        "corrected_lambda10_rows_digest": digest(current_rows),
        "frozen_stage_a": [_freeze_identity(seed) for seed in MODEL_SEEDS],
    }


def method_declaration() -> dict:
    audit = {
        identifier: {"status": "not_applicable", "evidence": "E262 is a frozen-controller deployment evaluation."}
        for identifier, _description in CHECKLIST
    }
    evidence = {
        "reproducibility.method_revision": f"E262 records {method_revision()}.",
        "reproducibility.content_hashes": "All corrected checkpoints, reused panel rows, sources, and coordinates are hashed.",
        "reproducibility.oracle_declarations": "No oracle input is used.",
        "reproducibility.failed_gates_retained": "Incomplete episodes and failed gates remain visible.",
        "reproducibility.preprotected_integration_gate": "A corrected seed-0 lambda-10 coordinate must reproduce exactly before evaluation.",
    }
    for identifier, text in evidence.items():
        audit[identifier] = {"status": "pass", "evidence": text}
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": ["Post-hoc simple-dial evaluation on the already-open E201 panel."],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "__init__.py", directory / "protocol.py", directory / "run.py",
        directory / "run.sh", directory / "README.md", ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E239_e195_simple_dial_curve_298k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E239_e195_simple_dial_curve_298k_v1/plot.py",
        ROOT / "experiments/conditioned_vcg/E210_on_policy_td_rehandle_stage_b_5x5_269k_v1/immediate.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
    )


def gate_contract() -> dict:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL + "_integration_gate",
        **deepcopy(method_declaration()),
        "scope": "corrected-checkpoint lambda-center equivalence and lambda-160 route gate",
        "source": source_identity(),
        "model_seed": GATE_MODEL_SEED,
        "instance_seed": GATE_INSTANCE_SEED,
        "lambdas": [LAMBDA_CENTER, max(FULL_LAMBDAS)],
        "test_panel_opened": True,
        "paper_evidence": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()},
    }
    validate_protocol(payload, source=payload["protocol"])
    return {**payload, "contract_sha256": digest(payload)}


def gate_identity() -> dict:
    path = GATE_OUTPUT / "gate.json"
    gate = _read(path)
    expected = gate_contract()
    if (
        gate.get("status") != "pass"
        or gate.get("gate_contract_sha256") != expected["contract_sha256"]
        or gate.get("lambda10_exact_source_match") is not True
        or gate.get("lambda160_route_complete") is not True
    ):
        raise RuntimeError("E262 requires its passing live integration gate")
    return {"gate": str(path.resolve()), "gate_sha256": sha256(path), "gate_contract_sha256": expected["contract_sha256"]}


def contract() -> dict:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "full immediate-only dial curve for the three corrected-temperature controllers on E201",
        "training_runs": 0,
        "test_panel_opened": True,
        "paper_evidence": True,
        "source": source_identity(),
        "integration_gate": gate_identity(),
        "model_seeds": list(MODEL_SEEDS),
        "selected_episodes": {str(key): value for key, value in SELECTED_EPISODES.items()},
        "instances": len(e201_scenario.TEST_SEEDS),
        "workloads": list(e201_scenario.LEVELS),
        "cases_per_workload": e201_scenario.CASES_PER_LEVEL,
        "full_lambda_curve": list(FULL_LAMBDAS),
        "new_lambda_settings": list(NEW_LAMBDAS),
        "controller": "immediate_only_centered_rehandle_price",
        "deployment_merit": "Q_corrected - 0.01*(lambda-10)*I_reconfigure",
        "learned_future_rehandle_term": False,
        "reuse": {"lambda10_vcg": "E256/E261 corrected-controller trajectories", "dynamic_ga": "E201 matched baseline trajectories"},
        "expected_reused_lambda10_rows": EXPECTED_REUSED_LAMBDA10_ROWS,
        "expected_new_episode_rows": EXPECTED_NEW_ROWS,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()},
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "current_panel_row", "digest", "gate_contract", "gate_identity",
    "method_declaration", "method_revision", "sha256", "source_identity",
]
