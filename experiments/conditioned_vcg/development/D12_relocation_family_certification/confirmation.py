#!/usr/bin/env python3
"""Integrated online confirmation of relocation-family certification."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
from time import perf_counter
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    confirm as e14_confirm,
    reuse,
)
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_final86_four_method as final86
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from PSLAP.viability_candidates import RELOCATION_FAMILY_CERTIFICATION


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = "vcg_d12_integrated_relocation_family_confirmation_95k_v2"
SCENARIOS = ("size_10x10_occ_medium", "size_10x10_occ_high")
INSTANCE_SEED = 95_100
MODEL_SEED = 0
PREFERENCE_LAMBDA = 0.10
WALL_LIMIT_SECONDS = 1_800
PARENT_E14 = ROOT / "results/vcg-conditioned-e14-certificate-reuse-95k"
PARENT_D10 = ROOT / "results/vcg-d10-occupancy-extension-95k"
PARENT_LATENCY = ROOT / "results/vcg-d10-10x10-latency-diagnostic-95k"
PRIOR_INTEGRATED = (
    ROOT / "results/vcg-d12-relocation-family-integrated-confirmation-95k"
)
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-d12-relocation-family-integrated-confirmation-95k-v2"
)
CONTRACT_NAME = "contract.json"
REPORT_NAME = "report.json"
CONDITIONED_TERMINAL = (
    ROOT
    / "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/"
    "terminal.pth"
)


class ConfirmationError(RuntimeError):
    pass


class WallLimit(RuntimeError):
    pass


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ConfirmationError(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ConfirmationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ConfirmationError(f"{label} must contain an object")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise ConfirmationError(f"{label} self-hash mismatch")
    return value


def _conditioned_seed0(d10_contract: Mapping) -> dict:
    """Authenticate only the frozen seed-0 inputs used by this confirmation.

    The broader three-seed training contract contains historical source hashes
    unrelated to this frozen rollout and is intentionally not reopened here.
    """

    arms, _records = final86._authenticate_v11()
    selected = [arm for arm in arms if int(arm.model_seed) == MODEL_SEED]
    if len(selected) != 1:
        raise ConfirmationError("could not resolve frozen VCG seed-0 base")
    terminal_sha = _sha(CONDITIONED_TERMINAL)
    if terminal_sha != d10_contract["conditioned_terminal_sha256"]:
        raise ConfirmationError("conditioned checkpoint differs from D10 parent")
    terminal = torch.load(
        CONDITIONED_TERMINAL, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("fixed_terminal_checkpoint") is not True
        or not isinstance(terminal.get("agent_checkpoint"), Mapping)
    ):
        raise ConfirmationError("conditioned terminal checkpoint is invalid")
    return {
        "inputs": {"arms": {MODEL_SEED: selected[0]}},
        "terminals": {MODEL_SEED: terminal},
        "terminal_sha256": {MODEL_SEED: terminal_sha},
    }


def _historical_inputs() -> dict:
    e14_contract = _self_hashed(
        PARENT_E14 / "e14-contract.json",
        "contract_sha256",
        "historical E14 contract",
    )
    medium = _self_hashed(
        PARENT_E14
        / "online-ledgers/size_10x10_occ_medium/path_enumeration_cleanup.json",
        "ledger_sha256",
        "historical E14 medium ledger",
    )
    if medium.get("contract_sha256") != e14_contract["contract_sha256"]:
        raise ConfirmationError("historical E14 medium ledger contract mismatch")
    latency_contract = _self_hashed(
        PARENT_LATENCY / "latency-diagnostic-contract.json",
        "contract_sha256",
        "historical latency contract",
    )
    high = _self_hashed(
        PARENT_LATENCY
        / "run-ledger/size_10x10_occ_high/seed-95100.json",
        "ledger_sha256",
        "historical high-occupancy ledger",
    )
    if high.get("contract_sha256") != latency_contract["contract_sha256"]:
        raise ConfirmationError("historical high ledger contract mismatch")
    d10_contract = _self_hashed(
        PARENT_D10 / "d10-occupancy-contract.json",
        "contract_sha256",
        "D10 occupancy contract",
    )
    manifest = _self_hashed(
        PARENT_D10 / "episode-instance-manifest.json",
        "manifest_sha256",
        "D10 instance manifest",
    )
    if manifest.get("contract_sha256") != d10_contract["contract_sha256"]:
        raise ConfirmationError("D10 instance manifest contract mismatch")
    records = {}
    for record in manifest["records"]:
        if (
            record["scenario_id"] in SCENARIOS
            and int(record["seed"]) == INSTANCE_SEED
        ):
            path = PARENT_D10 / record["relative_path"]
            if _sha(path) != record["raw_sha256"]:
                raise ConfirmationError(
                    f"instance hash mismatch: {record['scenario_id']}"
                )
            records[record["scenario_id"]] = dict(record)
    if set(records) != set(SCENARIOS):
        raise ConfirmationError("D10 manifest lacks confirmation instances")
    conditioned = _conditioned_seed0(d10_contract)
    terminal_sha = conditioned["terminal_sha256"][MODEL_SEED]
    historical_occupancy_source = d10_contract["source_sha256"][
        "experiments/conditioned_vcg/development/"
        "D10_scalability_support_screen/occupancy_extension.py"
    ]
    if _sha(Path(occupancy.__file__).resolve()) != historical_occupancy_source:
        raise ConfirmationError("scenario implementation changed from D10")
    prior_integrated_contract = _self_hashed(
        PRIOR_INTEGRATED / CONTRACT_NAME,
        "contract_sha256",
        "prior integrated confirmation contract",
    )
    prior_integrated_medium = _self_hashed(
        PRIOR_INTEGRATED / "ledgers/size_10x10_occ_medium.json",
        "ledger_sha256",
        "prior integrated medium ledger",
    )
    if (
        prior_integrated_contract.get("protocol")
        != "vcg_d12_integrated_relocation_family_confirmation_95k_v1"
        or prior_integrated_medium.get("contract_sha256")
        != prior_integrated_contract["contract_sha256"]
        or prior_integrated_medium.get("scenario_id") != SCENARIOS[0]
        or prior_integrated_medium.get("failure_class") != "completed"
        or prior_integrated_medium.get("strict_safe_complete") is not True
    ):
        raise ConfirmationError("prior integrated medium evidence is invalid")
    prior_comparison = prior_integrated_medium.get("reference_comparison", {})
    if (
        prior_comparison.get("candidate_key_or_epoch_mismatches") != []
        or prior_comparison.get("selected_action_or_epoch_mismatches") != []
        or prior_comparison.get("complete_reference_length_match") is not True
        or prior_comparison.get("complete_decision_length_match") is not True
    ):
        raise ConfirmationError(
            "prior integrated medium behavior comparison is invalid"
        )
    return {
        "e14_contract": e14_contract,
        "medium_reference": medium,
        "latency_contract": latency_contract,
        "high_reference": high,
        "d10_contract": d10_contract,
        "manifest": manifest,
        "records": records,
        "conditioned": conditioned,
        "prior_integrated_contract": prior_integrated_contract,
        "prior_integrated_medium": prior_integrated_medium,
    }


def _source_hashes() -> dict:
    paths = (
        Path(__file__),
        ROOT / "PSLAP/relocation_family_certification.py",
        ROOT / "PSLAP/viability.py",
        ROOT / "PSLAP/viability_candidates.py",
        ROOT / "PSLAP/viability_filter.py",
        ROOT / "benchmark_viability_critic_priority.py",
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def expected_contract() -> dict:
    inputs = _historical_inputs()
    return _with_hash(
        {
            "protocol": PROTOCOL,
            "question": (
                "does_integrated_constructive_relocation_family_certification_"
                "preserve_behavior_and_reduce_online_cost"
            ),
            "scenario_ids": list(SCENARIOS),
            "instance_seed": INSTANCE_SEED,
            "model_seed": MODEL_SEED,
            "preference_lambda": PREFERENCE_LAMBDA,
            "wall_limit_seconds_per_episode": WALL_LIMIT_SECONDS,
            "device": "cpu",
            "training": False,
            "recovery_certification_strategy": (
                RELOCATION_FAMILY_CERTIFICATION
            ),
            "current_state_certificate": "native_exact_or_native_cache",
            "family_proofs_stored_in_outcome_cache": False,
            "unresolved_family_attempt": "native_exact_fallback",
            "path_enumeration_cleanup": "retained_from_e14",
            "timing_invariant_outcome_cache": "retained_from_e14",
            "historical_e14_contract_sha256": inputs["e14_contract"][
                "contract_sha256"
            ],
            "medium_reference_ledger_sha256": inputs["medium_reference"][
                "ledger_sha256"
            ],
            "high_reference_ledger_sha256": inputs["high_reference"][
                "ledger_sha256"
            ],
            "d10_contract_sha256": inputs["d10_contract"][
                "contract_sha256"
            ],
            "d10_manifest_sha256": inputs["manifest"]["manifest_sha256"],
            "instances": [
                inputs["records"][scenario] for scenario in SCENARIOS
            ],
            "conditioned_terminal_sha256": inputs["conditioned"][
                "terminal_sha256"
            ][MODEL_SEED],
            "prior_integrated_contract_sha256": inputs[
                "prior_integrated_contract"
            ]["contract_sha256"],
            "prior_integrated_medium_ledger_sha256": inputs[
                "prior_integrated_medium"
            ]["ledger_sha256"],
            "source_sha256": _source_hashes(),
            "comparison_scope": {
                "medium": (
                    "authenticated_completed_v1_integrated_confirmation_"
                    "against_complete_51_decision_historical_e14_reference"
                ),
                "high": "censored_historical_d10_frontier_prefix",
                "latency": "historical_not_fresh_paired_repeat",
            },
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    contract = expected_contract()
    path = output / CONTRACT_NAME
    if path.exists():
        if _load(path, "confirmation contract") != contract:
            raise ConfirmationError(
                "confirmation contract, sources, or inputs changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ConfirmationError("nonempty output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, contract)
    return contract


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = _load(output / CONTRACT_NAME, "confirmation contract")
    expected = expected_contract()
    if contract != expected:
        raise ConfirmationError(
            "confirmation contract, sources, or inputs changed"
        )
    return contract, _historical_inputs()


def _instance(inputs: Mapping, scenario_id: str) -> EpisodeInstance:
    record = inputs["records"][scenario_id]
    path = PARENT_D10 / record["relative_path"]
    instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    instance.validate_for(scenario.make_base_env())
    if instance.instance_id != record["episode_instance_id"]:
        raise ConfirmationError("instance identity mismatch")
    return instance


def _reference_frontiers(inputs: Mapping, scenario_id: str) -> tuple:
    if scenario_id == "size_10x10_occ_medium":
        return tuple(inputs["medium_reference"]["frontiers"])
    return tuple(inputs["high_reference"]["row"]["completed_frontiers"])


def _compare_reference(
    inputs: Mapping,
    scenario_id: str,
    frontiers: Sequence[Mapping],
    decisions: Sequence[Mapping],
    *,
    completed: bool,
) -> dict:
    reference = _reference_frontiers(inputs, scenario_id)
    common = min(len(reference), len(frontiers))
    frontier_mismatches = [
        index
        for index in range(common)
        if (
            list(reference[index]["candidate_keys"])
            != list(frontiers[index]["candidate_keys"])
            or int(reference[index]["decision_epoch"])
            != int(frontiers[index]["decision_epoch"])
        )
    ]
    reference_is_complete = scenario_id == "size_10x10_occ_medium"
    result = {
        "reference_frontiers": len(reference),
        "observed_frontiers": len(frontiers),
        "common_frontiers": common,
        "candidate_key_or_epoch_mismatches": frontier_mismatches,
        "identical_common_frontier_prefix": not frontier_mismatches,
        "reference_is_complete": reference_is_complete,
        "reference_prefix_fully_covered": len(frontiers) >= len(reference),
        "complete_reference_length_match": (
            len(reference) == len(frontiers)
            if completed and reference_is_complete
            else None
        ),
    }
    if scenario_id == "size_10x10_occ_medium":
        old = inputs["medium_reference"]["decisions"]
        decision_common = min(len(old), len(decisions))
        decision_mismatches = [
            index
            for index in range(decision_common)
            if (
                old[index]["selected_key"],
                int(old[index]["decision_epoch"]),
            )
            != (
                decisions[index]["selected_key"],
                int(decisions[index]["decision_epoch"]),
            )
        ]
        result.update(
            {
                "reference_decisions": len(old),
                "observed_decisions": len(decisions),
                "common_decisions": decision_common,
                "selected_action_or_epoch_mismatches": decision_mismatches,
                "complete_decision_length_match": (
                    len(old) == len(decisions) if completed else None
                ),
            }
        )
    return result


def _validate_reference_comparison(
    scenario_id: str, comparison: Mapping, *, strict: bool
) -> None:
    if comparison["candidate_key_or_epoch_mismatches"]:
        raise ConfirmationError("integrated strategy changed reference frontier")
    if strict and not comparison["reference_prefix_fully_covered"]:
        raise ConfirmationError(
            "completed run did not cover the full historical reference prefix"
        )
    if scenario_id == "size_10x10_occ_medium":
        if comparison["selected_action_or_epoch_mismatches"]:
            raise ConfirmationError("integrated strategy changed selected action")
        if strict and comparison["complete_reference_length_match"] is False:
            raise ConfirmationError("completed run has different frontier length")
        if strict and comparison["complete_decision_length_match"] is False:
            raise ConfirmationError("completed run has different decision length")


class _WallClock:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if not hasattr(signal, "setitimer"):
            return self
        self.previous = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise WallLimit(
                f"predeclared {self.seconds}s confirmation limit reached"
            )

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *_args):
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous)


def _ledger_path(output: Path, scenario_id: str) -> Path:
    return output / "ledgers" / f"{scenario_id}.json"


def _authenticate_ledger(
    output: Path, contract: Mapping, scenario_id: str
) -> dict:
    ledger = _self_hashed(
        _ledger_path(output, scenario_id),
        "ledger_sha256",
        f"{scenario_id} confirmation ledger",
    )
    if (
        ledger.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("scenario_id") != scenario_id
    ):
        raise ConfirmationError("confirmation ledger coordinate mismatch")
    return ledger


def run_one(output: Path, scenario_id: str) -> dict:
    if scenario_id not in SCENARIOS:
        raise ConfirmationError(f"unsupported scenario: {scenario_id}")
    contract, inputs = authenticate(output)
    path = _ledger_path(output, scenario_id)
    if path.exists():
        return _authenticate_ledger(output, contract, scenario_id)
    if scenario_id == "size_10x10_occ_medium":
        return dict(inputs["prior_integrated_medium"])
    instance = _instance(inputs, scenario_id)
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    conditioned = inputs["conditioned"]
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    cache = reuse.TimingInvariantCertificateCache()
    envs = []
    frontiers = []
    stream_path = output / "frontiers" / f"{scenario_id}.jsonl"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    original_enumerate = benchmark._enumerate_frontier

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            ROOT,
            {"conditioned": conditioned},
            model_seed=MODEL_SEED,
            base=base,
            device=torch.device("cpu"),
        )
        agent.set_epsilon(0.0)
        return conditioned_seed0._FixedLambdaAgent(
            agent, PREFERENCE_LAMBDA
        )

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        envs.append(env)
        return env

    def record_enumerate(*args, **kwargs):
        snapshot, record = original_enumerate(*args, **kwargs)
        projection = e14_confirm._frontier_projection(record, len(frontiers))
        projection.update(
            {
                "recovery_certification_strategy": record[
                    "recovery_certification_strategy"
                ],
                "relocation_family_attempt_count": record[
                    "relocation_family_attempt_count"
                ],
                "relocation_family_proof_count": record[
                    "relocation_family_proof_count"
                ],
                "relocation_family_miss_count": record[
                    "relocation_family_miss_count"
                ],
                "relocation_family_setup_seconds": record[
                    "relocation_family_setup_seconds"
                ],
                "relocation_family_connection_seconds": record[
                    "relocation_family_connection_seconds"
                ],
                "native_recovery_search_count": record[
                    "native_recovery_search_count"
                ],
            }
        )
        frontiers.append(projection)
        with stream_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(projection, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "scenario": scenario_id,
                    "frontier": len(frontiers),
                    "family_proofs": sum(
                        item["relocation_family_proof_count"]
                        for item in frontiers
                    ),
                    "frontier_seconds": projection["total_frontier_seconds"],
                }
            ),
            flush=True,
        )
        return snapshot, record

    raw = None
    error = None
    failure = None
    started = perf_counter()
    try:
        with _WallClock(WALL_LIMIT_SECONDS), reuse.path_cleanup_active(), ExitStack() as stack:
            stack.enter_context(patch.object(benchmark, "_make_env", env_factory))
            stack.enter_context(pilot._agent_factory(agent_factory))
            stack.enter_context(
                patch.object(
                    benchmark, "ViabilityCertificateCache", lambda: cache
                )
            )
            stack.enter_context(
                patch.object(benchmark, "_enumerate_frontier", record_enumerate)
            )
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL,
                controller_payload=arm.payload,
                instance=instance,
                instance_seed=int(instance.seed),
                search_config=benchmark._search_config(arm.payload),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None,
                max_steps=scenario.max_steps,
                device=torch.device("cpu"),
                recovery_certification_strategy=(
                    RELOCATION_FAMILY_CERTIFICATION
                ),
            )
    except WallLimit as caught:
        failure, error = "censored_wall_clock", str(caught)
    except Exception as caught:
        failure = "unsupported_or_implementation_error"
        error = f"{type(caught).__name__}: {caught}"
    wall_seconds = perf_counter() - started
    decisions = () if raw is None else raw["decisions"]
    strict = bool(
        raw is not None
        and raw["strict_method_success"]
        and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and raw["illegal_drops"] == 0
        and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == scenario.total_jobs
    )
    if failure is None:
        failure = "completed" if strict else "operationally_incomplete"
    if raw is not None:
        # Preserve a returned episode even if a post-run comparison gate fails.
        _atomic_json(output / "episodes" / f"{scenario_id}.json", raw)
    comparison = _compare_reference(
        inputs, scenario_id, frontiers, decisions, completed=strict
    )
    _validate_reference_comparison(scenario_id, comparison, strict=strict)
    ledger = _with_hash(
        {
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "scenario_id": scenario_id,
            "instance_id": instance.instance_id,
            "failure_class": failure,
            "error": error,
            "strict_safe_complete": strict,
            "wall_seconds": wall_seconds,
            "completed_frontiers": len(frontiers),
            "completed_decisions": len(decisions),
            "reference_comparison": comparison,
            "recovery_certification_strategy": (
                RELOCATION_FAMILY_CERTIFICATION
            ),
            "relocation_family_attempt_count": (
                None if raw is None else raw["relocation_family_attempt_count"]
            ),
            "relocation_family_proof_count": (
                sum(item["relocation_family_proof_count"] for item in frontiers)
            ),
            "relocation_family_miss_count": (
                sum(item["relocation_family_miss_count"] for item in frontiers)
            ),
            "relocation_family_setup_seconds": sum(
                item["relocation_family_setup_seconds"] for item in frontiers
            ),
            "relocation_family_connection_seconds": sum(
                item["relocation_family_connection_seconds"]
                for item in frontiers
            ),
            "native_recovery_search_count": sum(
                item["native_recovery_search_count"] for item in frontiers
            ),
            "exact_search_seconds": sum(
                item["exact_search_seconds"] for item in frontiers
            ),
            "total_frontier_seconds": sum(
                item["total_frontier_seconds"] for item in frontiers
            ),
            "steps": None if raw is None else raw["steps"],
            "deliveries": (
                None if raw is None else len(raw["delivery_deviations"])
            ),
            "physical_relocations": (
                None if raw is None else raw["relocations"]
            ),
            "macro_failures": None if raw is None else raw["macro_failures"],
            "illegal_drops": None if raw is None else raw["illegal_drops"],
            "occupancy_at_stop": envs[0]._measurement() if envs else None,
            "historical_timing_not_matched_repeat": True,
        },
        "ledger_sha256",
    )
    _atomic_json(path, ledger)
    return ledger


def analyze(output: Path, *, allow_partial: bool = False) -> dict:
    contract, inputs = authenticate(output)
    ledgers = []
    missing = []
    evidence_sources = {}
    for scenario_id in SCENARIOS:
        if _ledger_path(output, scenario_id).exists():
            ledgers.append(_authenticate_ledger(output, contract, scenario_id))
            evidence_sources[scenario_id] = "current_v2_confirmation"
        elif scenario_id == "size_10x10_occ_medium":
            ledgers.append(dict(inputs["prior_integrated_medium"]))
            evidence_sources[scenario_id] = (
                "authenticated_prior_v1_integrated_confirmation"
            )
        else:
            missing.append(scenario_id)
    if missing and not allow_partial:
        raise ConfirmationError(f"missing confirmation scenarios: {missing}")
    report = _with_hash(
        {
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "observed_scenarios": [item["scenario_id"] for item in ledgers],
            "missing_scenarios": missing,
            "scenario_evidence_sources": evidence_sources,
            "all_observed_strict_safe_complete": bool(ledgers)
            and all(item["strict_safe_complete"] for item in ledgers),
            "all_reference_prefixes_identical": bool(ledgers)
            and all(
                item["reference_comparison"][
                    "identical_common_frontier_prefix"
                ]
                for item in ledgers
            ),
            "scenario_results": ledgers,
            "interpretation": (
                "integrated exact constructive reuse; medium has a complete "
                "authenticated v1 integrated result against its complete "
                "historical behavior reference; high has a censored frontier "
                "reference whose full prefix must be covered; latency "
                "comparisons remain historical"
            ),
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "authenticate-inputs",
            "prepare",
            "run-medium",
            "run-high",
            "run-both",
            "analyze",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.command == "authenticate-inputs":
        inputs = _historical_inputs()
        result = {
            "status": "authenticated",
            "e14_contract_sha256": inputs["e14_contract"]["contract_sha256"],
            "d10_manifest_sha256": inputs["manifest"]["manifest_sha256"],
            "prior_integrated_medium_ledger_sha256": inputs[
                "prior_integrated_medium"
            ]["ledger_sha256"],
        }
    elif args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-medium":
        prepare(output)
        result = run_one(output, SCENARIOS[0])
    elif args.command == "run-high":
        prepare(output)
        result = run_one(output, SCENARIOS[1])
    elif args.command == "run-both":
        prepare(output)
        result = [run_one(output, scenario) for scenario in SCENARIOS]
    else:
        result = analyze(output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
