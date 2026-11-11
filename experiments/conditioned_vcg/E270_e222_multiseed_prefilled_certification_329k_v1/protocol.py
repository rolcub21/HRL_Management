"""Authenticated protocol for the 8x8 prefilled certification curve."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re

import torch

from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import (
    CertifiedDynamicAgent,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import (
    protocol as e222,
)
from experiments.conditioned_vcg.E247_kim2020_objective_matched_a3c_306k_v1 import (
    protocol as e247,
)
from experiments.conditioned_vcg.E255_e195_temperature_scaled_5x5_314k_v1 import (
    protocol as e255,
)
from experiments.conditioned_vcg.E263_e255_prefilled_certification_curve_5x5_322k_v1 import (
    protocol as e263,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)

from . import scenario
from .tracker import MatchedDynamicE191Agent


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e270_e222_multiseed_prefilled_certification_329k_v1"
GATE_PROTOCOL = PROTOCOL + "_development_gate"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
MODEL_SEEDS = (0, 1, 2)
ARMS = (
    "vcg_certified",
    "vcg_physical_unfiltered",
    "dynamic_certified",
    "dynamic_physical_unfiltered",
)
VCG_ARMS = ("vcg_certified", "vcg_physical_unfiltered")
DYNAMIC_ARMS = ("dynamic_certified", "dynamic_physical_unfiltered")
A3C_ARMS = ("a3c_certified", "a3c_physical_unfiltered")
A3C_SIZES = ("5x5", "8x8")
UNFILTERED_ARMS = (
    "vcg_physical_unfiltered", "dynamic_physical_unfiltered",
    "a3c_physical_unfiltered",
)
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e270-e222-multiseed-prefilled-certification-329k-v1"
)
DEFAULT_GATE_OUTPUT = Path(str(DEFAULT_OUTPUT) + "-gate")
GATE_NAME = "gate.json"
REPORT_NAME = "REPORT.txt"
REPORT_JSON = "e270-report.json"
TABLE_NAME = "e270-table.md"
EXPECTED_VCG_ROWS = len(MODEL_SEEDS) * len(VCG_ARMS) * len(scenario.SEEDS)
EXPECTED_DYNAMIC_ROWS = len(DYNAMIC_ARMS) * len(scenario.SEEDS)
EXPECTED_A3C_ROWS = (
    len(MODEL_SEEDS) * len(A3C_ARMS)
    * (len(e263.scenario.SEEDS) + len(scenario.SEEDS))
)
EXPECTED_ROWS = EXPECTED_VCG_ROWS + EXPECTED_DYNAMIC_ROWS + EXPECTED_A3C_ROWS

E247_CONTRACT = e247.DEFAULT_OUTPUT / "contract.json"
E255_SHELL_CHECKPOINT = (
    e255.DEFAULT_OUTPUT
    / "training/operational/empty_5x5_n8_longstay_mixedrho/seed-0/best.pth"
)
E263_INSTANCES = e263.DEFAULT_OUTPUT / "instances.json"


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
        raise RuntimeError(f"missing E270 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E270 artifact: {path}")
    return value


def method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"), re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E270 cannot parse the canonical method revision")
    return match.group(1)


def _portable_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise RuntimeError(f"E270 source lies outside the repository: {path}") from error


def source_panel_identity() -> dict:
    """Return E222's three checkpoint selections without panel-row baggage."""

    checkpoints = deepcopy(e222.checkpoint_identities())
    for checkpoint in checkpoints.values():
        checkpoint["source_summary"] = _portable_path(checkpoint["source_summary"])
        checkpoint["source_checkpoint"] = _portable_path(
            checkpoint["source_checkpoint"]
        )
    return {
        "protocol": e222.PROTOCOL,
        "selection_scope": (
            "the three frozen development-selected checkpoints evaluated by E222; "
            "E270 does not reuse or require E222 test-panel rows"
        ),
        "checkpoints": checkpoints,
    }


def checkpoint_identity(model_seed: int) -> dict:
    if int(model_seed) not in MODEL_SEEDS:
        raise ValueError(f"invalid E270 model seed: {model_seed}")
    value = deepcopy(
        source_panel_identity()["checkpoints"][str(int(model_seed))]
    )
    value["source_summary"] = str((ROOT / value["source_summary"]).resolve())
    value["source_checkpoint"] = str((ROOT / value["source_checkpoint"]).resolve())
    return value


def a3c_checkpoint_identity(size: str, model_seed: int) -> dict:
    """Authenticate one finalized E247 deployment checkpoint without summaries."""

    if size not in A3C_SIZES or int(model_seed) not in MODEL_SEEDS:
        raise ValueError(f"invalid E270 A3C checkpoint: {size}/{model_seed}")
    contract = _read(E247_CONTRACT)
    path = e247.DEFAULT_OUTPUT / f"training/{size}/seed-{int(model_seed)}/best.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    expected = {
        "protocol": e247.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "size": size,
        "model_seed": int(model_seed),
        "protocol_training_complete": True,
    }
    mismatch = {
        key: (payload.get(key), value) for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"E270 E247 checkpoint metadata differs: {mismatch}")
    return {
        "source_protocol": e247.PROTOCOL,
        "source_checkpoint": str(path.resolve()),
        "source_checkpoint_sha256": sha256(path),
        "source_contract": str(E247_CONTRACT.resolve()),
        "source_contract_sha256": sha256(E247_CONTRACT),
        "size": size,
        "model_seed": int(model_seed),
        "selected_episode": int(payload["selected_episode"]),
        "final_training_episode": int(payload["final_training_episode"]),
        "protocol_training_complete": True,
    }


def five_shell_identity() -> dict:
    """Authenticate the 5x5 graph/frontier shell; its critic is never scored."""

    payload = torch.load(
        E255_SHELL_CHECKPOINT, map_location="cpu", weights_only=False, mmap=True
    )
    expected = {
        "training_protocol": e255.PROTOCOL,
        "model_seed": 0,
        "selected_checkpoint_episode": 280,
        "completed_training_episodes": 300,
    }
    mismatch = {
        key: (payload.get(key), value) for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"E270 5x5 frontier shell differs: {mismatch}")
    return {
        "role": "frontier_record_builder_only; Q values are not consulted",
        "source_checkpoint": str(E255_SHELL_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": sha256(E255_SHELL_CHECKPOINT),
        **expected,
    }


def five_panel_identity() -> dict:
    value = _read(E263_INSTANCES)
    if (
        value.get("all_initial_states_have_completion_witness") is not True
        or len(value.get("records", ())) != len(e263.scenario.SEEDS)
    ):
        raise RuntimeError("E270 requires E263's complete witnessed-safe panel")
    return {
        "source_protocol": e263.PROTOCOL,
        "source_instances": str(E263_INSTANCES.resolve()),
        "source_instances_sha256": sha256(E263_INSTANCES),
        "records": len(value["records"]),
    }


def dynamic_ranker_sha256() -> str:
    source = "\n".join((
        inspect.getsource(CertifiedDynamicAgent._choice),
        inspect.getsource(CertifiedDynamicAgent.select),
        inspect.getsource(MatchedDynamicE191Agent.select),
    ))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "protocol.py", directory / "scenario.py",
        directory / "tracker.py", directory / "a3c_ranker.py", directory / "run.py",
        directory / "run.sh", directory / "run_overnight.sh",
        directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "PSLAP/viability.py",
        ROOT / "PSLAP/viability_candidates.py",
        ROOT / "experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/selector.py",
        ROOT / "experiments/conditioned_vcg/E198_e195_shadow_certification_ablation_257k_v1/frontier.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/scenario.py",
        ROOT / "experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/tracker.py",
        ROOT / "experiments/conditioned_vcg/E247_kim2020_objective_matched_a3c_306k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E247_kim2020_objective_matched_a3c_306k_v1/source.py",
        ROOT / "PSLAP/kim2020_a3c_spatial.py",
    )


def gate_contract() -> dict:
    source = source_panel_identity()
    payload = {
        "schema_version": 1,
        "protocol": GATE_PROTOCOL,
        "scope": "development-only eight-route dual-size prefilled integration gate",
        "development_seed": scenario.GATE_SEED,
        "development_fill": scenario.GATE_FILL,
        "routes": [*ARMS, "5x5/a3c_certified", "5x5/a3c_physical_unfiltered",
                   "8x8/a3c_certified", "8x8/a3c_physical_unfiltered"],
        "checkpoint_sha256": {
            key: item["source_checkpoint_sha256"]
            for key, item in source["checkpoints"].items()
        },
        "dynamic_ranker_sha256": dynamic_ranker_sha256(),
        "a3c_checkpoint_sha256": {
            f"{size}/seed-{seed}": a3c_checkpoint_identity(size, seed)[
                "source_checkpoint_sha256"
            ]
            for size in A3C_SIZES for seed in MODEL_SEEDS
        },
        "five_shell_sha256": five_shell_identity()["source_checkpoint_sha256"],
        "five_panel_sha256": five_panel_identity()["source_instances_sha256"],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
        "test_panel_opened": False,
    }
    return {**payload, "contract_sha256": digest(payload)}


def gate_manifest() -> dict:
    value = _read(DEFAULT_GATE_OUTPUT / GATE_NAME)
    expected = gate_contract()
    if (
        value.get("status") != "pass"
        or value.get("protocol") != GATE_PROTOCOL
        or value.get("gate_contract_sha256") != expected["contract_sha256"]
        or value.get("test_panel_opened") is not False
    ):
        raise RuntimeError("E270 development gate identity or verdict differs")
    core = dict(value)
    recorded = core.pop("manifest_sha256", None)
    if recorded != digest(core):
        raise RuntimeError("E270 gate manifest hash is invalid")
    return value


def method_declaration(gate: dict) -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E270 is a historical conditioned-VCG certification mechanism "
                "ablation, not a canonical law-condition representation claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    evidence = {
        "reproducibility.method_revision": f"E270 records {method_revision()}.",
        "reproducibility.content_hashes": (
            "The generators, verifier, eight live routes, VCG and A3C checkpoints, "
            "shared schedulers, gate, and frozen state manifests are hashed."
        ),
        "reproducibility.oracle_declarations": (
            "Exact recovery is used to screen starts and filter certified arms; "
            "in unfiltered arms it is read-only telemetry after initialization."
        ),
        "reproducibility.failed_gates_retained": (
            "Rejected candidate starts, UNKNOWN counts, route failures, incomplete "
            "episodes, and verifier events are retained."
        ),
        "reproducibility.preprotected_integration_gate": (
            f"Development gate {gate['manifest_sha256']} exercised VCG, Dynamic, "
            "and paired A3C routes at both sizes before evaluation."
        ),
    }
    for identifier, text in evidence.items():
        audit[identifier] = {"status": "pass", "evidence": text}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": [
            "E270 evaluates historical conditioned VCG and a fixed Dynamic-style "
            "ranker, not the canonical law-condition representation.",
            "Prefilled states are conditioned on a SAFE initial witness; rejection "
            "counts quantify the resulting selection pressure.",
        ],
        "oracle_inputs": [
            "exact recoverability for initial-state screening, certified filtering, "
            "and non-authoritative monitoring"
        ],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def contract() -> dict:
    gate = gate_manifest()
    source = source_panel_identity()
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **method_declaration(gate),
        "scope": "frozen paired 8x8 prefilled certification curve",
        "scientific_question": (
            "with ranking fixed, does exact certification prevent failures as "
            "recoverable initial occupancy rises"
        ),
        "training_runs": 0,
        "model_seeds": list(MODEL_SEEDS),
        "source_panel": source,
        "five_by_five_panel": five_panel_identity(),
        "five_by_five_frontier_shell": five_shell_identity(),
        "a3c_checkpoints": {
            f"{size}/seed-{seed}": a3c_checkpoint_identity(size, seed)
            for size in A3C_SIZES for seed in MODEL_SEEDS
        },
        "initial_fills": list(scenario.INITIAL_FILLS),
        "initial_fill_fractions": [
            fill / scenario.CAPACITY for fill in scenario.INITIAL_FILLS
        ],
        "cases_per_fill": scenario.CASES_PER_FILL,
        "instance_seeds": list(scenario.SEEDS),
        "future_arrival_gap_steps": scenario.FUTURE_ARRIVAL_GAP,
        "initial_state_screen": {
            "status_required": "SAFE",
            "concrete_witness_required": True,
            "max_nodes": 20_000,
            "search_order": "goal_directed",
            "max_candidate_attempts_per_state": scenario.MAX_CANDIDATE_ATTEMPTS,
            "unknown_discards_reported": True,
            "unsafe_discards_reported": True,
        },
        "arms": {
            "vcg_certified": "each frozen critic over SAFE-only frontier",
            "vcg_physical_unfiltered": (
                "same critic over physical frontier; exact labels are telemetry"
            ),
            "dynamic_certified": "shared fixed Dynamic ranker over SAFE-only frontier",
            "dynamic_physical_unfiltered": (
                "identical Dynamic ranker over physical frontier; labels are telemetry"
            ),
            "a3c_certified": (
                "frozen stochastic E247 placement ranker plus the common scheduler; "
                "placements and scheduler rehandles take the first SAFE action in "
                "their unchanged order"
            ),
            "a3c_physical_unfiltered": (
                "same frozen stochastic ranking draw and scheduler order over the "
                "physical frontier; exact labels are passive telemetry"
            ),
        },
        "dynamic_pair_only_difference": "candidate certification filter",
        "dynamic_ranker_sha256": dynamic_ranker_sha256(),
        "guard_contract": (
            "unchanged E215/E218 progress guard in VCG arms; no VCG guard in "
            "Dynamic or A3C placement arms"
        ),
        "a3c_contract": {
            "learned_scope": "placement cell only",
            "scheduler": "same fixed Dynamic-style macro scheduler in both twins",
            "whole_controller_certification": (
                "Accept and Reconfigure are filtered; Deliver is removal-safe"
            ),
            "paired_randomness": "one stateless Gumbel placement order per state",
            "bounded_deferral": "same 10-step/16-consecutive frontier rule in both twins",
            "vcg_guard_used": False,
            "failure_classes": ["unsafe_selection", "step_limit_stall", "other"],
        },
        "expected_vcg_rows": EXPECTED_VCG_ROWS,
        "expected_dynamic_rows": EXPECTED_DYNAMIC_ROWS,
        "expected_a3c_rows": EXPECTED_A3C_ROWS,
        "expected_rows": EXPECTED_ROWS,
        "cost_analysis": "paired certified-minus-unfiltered J on joint completers",
        "development_gate": {
            "manifest": str(
                (DEFAULT_GATE_OUTPUT / GATE_NAME).relative_to(ROOT)
            ),
            "manifest_sha256": gate["manifest_sha256"],
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
        "test_panel_opened": False,
        "paper_evidence": True,
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "a3c_checkpoint_identity", "checkpoint_identity", "contract", "digest", "dynamic_ranker_sha256",
    "five_panel_identity", "five_shell_identity",
    "gate_contract", "gate_manifest", "method_declaration", "method_revision",
    "sha256", "source_panel_identity",
]
