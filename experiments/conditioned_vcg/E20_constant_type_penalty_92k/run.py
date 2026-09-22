#!/usr/bin/env python3
"""E20: fixed action-type penalty versus learned handling signals.

Four per-seed constants are distilled on non-test training-probe frontiers.
The only new evaluation arm uses those constants at lambda=.10 on the frozen
92k EpisodeInstances.  The state-dependent action-type and full-future arms
are authenticated, reused E16-D controls rather than new replications.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import sys
from typing import Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scipy.stats import t as student_t

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    component_rollout_ablation as e16d,
)
from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    continuation_bank as e16b,
)
import diagnose_vcg_v11_conditioned_handling_fixed_bank as fixed_bank
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import train_vcg_v11_conditioned_handling_iterative as training
import run_vcg_v11_nested_handling_pilot as pilot
from PSLAP.viability_candidates import ViabilityMode
from vcg_v11_conditioned_handling import (
    ConditionedHandlingAgent,
    ConditionedHandlingDecision,
)
from vcg_v11_nested_handling import detached_v11_features
from viability_graph_hierarchy import (
    ID_TO_MODE,
    NoCertifiedViableAction,
    ViabilityGraphDecision,
)
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e20_constant_type_penalty_92k_v1"
CALIBRATION_PROTOCOL = "vcg_conditioned_e20_non_test_type_calibration_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = e16d.MODEL_SEEDS
INSTANCE_SEEDS = e16d.INSTANCE_SEEDS
DEPLOYMENT_LAMBDA = 0.10
PREDICTOR_INPUT_LAMBDA = 0.10
PILOT_MODEL_SEED = 0
PILOT_INSTANCE_COUNT = 5
ACTION_TYPES = ("accept", "deliver", "reconfigure", "defer")

CONSTANT_TYPE = "constant_type"
STATE_DEPENDENT_TYPE = "learned_state_dependent_type"
FULL_FUTURE = "full_future"
ARMS = (CONSTANT_TYPE, STATE_DEPENDENT_TYPE, FULL_FUTURE)
CONTROL_COMPONENTS = {
    STATE_DEPENDENT_TYPE: e16d.ACTION_TYPE_ONLY,
    FULL_FUTURE: e16d.FULL_FUTURE,
}

# Practical-equivalence units are deliberately interpretable on an 8-job
# episode: one time step of MAE, one relocation per 40 required deliveries,
# five primitive steps, and one within-window delivery per 40 deliveries.
EQUIVALENCE_MARGINS = {
    "mean_absolute_error": 1.0,
    "physical_rehandles_per_100_required_deliveries": 2.5,
    "steps": 5.0,
    "within_target_window_rate": 0.025,
}

CONTRACT_NAME = "e20-contract.json"
CALIBRATION_NAME = "e20-type-penalty-calibration.json"
PILOT_REPORT_NAME = "e20-pilot-report.json"
REPORT_NAME = "e20-report.json"
TABLE_NAME = "e20-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e20-constant-type-penalty-92k"
E16_OUTPUT = e16d.DEFAULT_OUTPUT


class E20Error(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E20Error(f"missing regular input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E20Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E20Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise E20Error(f"{label} self-hash mismatch")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "vcg_v11_conditioned_handling.py",
        PROJECT_ROOT / "vcg_v11_nested_handling.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "viability_graph_preference_conditioned.py",
        PROJECT_ROOT / "train_viability_graph_smdp.py",
        PROJECT_ROOT
        / "experiments/conditioned_vcg/E16_prediction_ranking_audit/component_rollout_ablation.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _calibration_provenance() -> dict:
    arms = e16b._frozen_arms()
    checkpoints = {}
    for seed in MODEL_SEEDS:
        terminal = e16b._terminal(seed)["agent_checkpoint"]
        arm = arms[seed]
        if (
            terminal.get("base_checkpoint_sha256") != arm.checkpoint_sha256
            or terminal.get("base_policy_digest") != arm.deployment_policy_digest
        ):
            raise E20Error(f"conditioned seed-{seed} does not bind its base arm")
        checkpoints[str(seed)] = {
            "operational_checkpoint_sha256": arm.checkpoint_sha256,
            "operational_policy_digest": arm.deployment_policy_digest,
            "conditioned_terminal_sha256": e16b.TERMINAL_SHA256[seed],
        }
    return {
        "checkpoints": checkpoints,
        "probe_seed_base": int(training.PROBE_SEED_BASE),
        "probe_lambdas": [float(value) for value in training.PROBE_LAMBDAS],
        "source_sha256": _source_hashes(),
    }


def _new_conditioned_agent(seed: int, base, arms: Mapping[int, object]):
    arm = arms[int(seed)]
    checkpoint = e16b._terminal(seed)["agent_checkpoint"]
    agent = ConditionedHandlingAgent.from_checkpoint(
        checkpoint,
        base_agent=base,
        expected_base_checkpoint_sha256=arm.checkpoint_sha256,
        expected_base_policy_digest=arm.deployment_policy_digest,
        expected_source_cost_sha256=checkpoint["source_cost_sha256"],
        seed=int(seed),
    )
    agent.handling_network = e16b.ClampedPreferenceNetwork(
        agent.handling_network, PREDICTOR_INPUT_LAMBDA
    ).to(base.device)
    agent.handling_network.requires_grad_(False).eval()
    agent.set_epsilon(0.0)
    return agent


def _collect_calibration_frontiers(device: torch.device):
    """Collect all unique frontiers from predeclared training probes only."""

    arms = e16b._frozen_arms()
    arm = arms[0]
    by_digest = {}
    probe_runs = []
    for probe_index, behavior_lambda in enumerate(training.PROBE_LAMBDAS):
        instance_seed = int(training.PROBE_SEED_BASE + probe_index)
        instance = benchmark._make_env(arm.payload).sample_episode_instance(
            instance_seed
        )
        holder = []

        def factory(base):
            agent = _new_conditioned_agent(0, base, arms)
            captured = fixed_bank._CaptureAgent(agent, limit=10_000)
            holder.append(captured)
            return conditioned_seed0._FixedLambdaAgent(captured, behavior_lambda)

        raw = pilot._run_raw(arm, instance, device=device, wrapper_factory=factory)
        if not raw.get("strict_method_success"):
            raise E20Error(
                f"training probe {instance_seed} failed: "
                f"{raw.get('method_failure_reason')}"
            )
        if len(holder) != 1:
            raise E20Error("calibration capture wrapper was not constructed once")
        probe_runs.append(
            {
                "probe_index": int(probe_index),
                "instance_seed": instance_seed,
                "behavior_lambda": float(behavior_lambda),
                "episode_instance_id": instance.instance_id,
                "captured_frontiers": len(holder[0].frontiers),
                "macro_decisions": int(raw["macro_decisions"]),
            }
        )
        for frontier in holder[0].frontiers:
            records = tuple(frontier["records"])
            digest = fixed_bank._frontier_digest(records)
            by_digest.setdefault(digest, records)
    if not by_digest:
        raise E20Error("non-test calibration bank is empty")
    return list(by_digest.items()), probe_runs, arms


def calibrate(output: Path, *, device_name: str) -> dict:
    """Distill four fixed constants without opening the 92k test panel."""

    output = output.resolve()
    path = output / CALIBRATION_NAME
    provenance = _calibration_provenance()
    if path.is_file():
        artifact = _load_json(path, "E20 calibration")
        _verify_hash(artifact, "calibration_sha256", "E20 calibration")
        if artifact.get("provenance") != provenance:
            raise E20Error("E20 calibration sources or frozen checkpoints changed")
        return {
            "status": "calibrated",
            "reused_existing_artifact": True,
            "evaluation_panel_opened": False,
            "calibration": str(path),
            "constants": artifact["constants"],
        }

    if output.exists() and any(output.iterdir()):
        raise E20Error("nonempty E20 output has no calibration artifact")
    output.mkdir(parents=True, exist_ok=True)
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E20Error("E20 calibration is frozen to CPU execution")
    torch.set_num_threads(1)
    frontiers, probe_runs, arms = _collect_calibration_frontiers(device)
    values = {seed: defaultdict(list) for seed in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        base = pilot._fresh_base(arms[seed], device)
        agent = _new_conditioned_agent(seed, base, arms)
        for _digest_value, records in frontiers:
            features = detached_v11_features(agent.base_agent.Q_local, records)
            with torch.no_grad():
                future = agent.handling_network(
                    features, PREDICTOR_INPUT_LAMBDA
                ).detach().cpu()
            for record, prediction in zip(records, future.tolist()):
                values[seed][str(record.action_type)].append(float(prediction))

    constants = {}
    distributions = {}
    for seed in MODEL_SEEDS:
        observed = set(values[seed])
        if observed != set(ACTION_TYPES):
            missing = sorted(set(ACTION_TYPES) - observed)
            extra = sorted(observed - set(ACTION_TYPES))
            raise E20Error(
                f"seed-{seed} calibration action types changed; "
                f"missing={missing}, extra={extra}"
            )
        constants[str(seed)] = {
            name: fmean(values[seed][name]) for name in ACTION_TYPES
        }
        distributions[str(seed)] = {
            name: {
                "n": len(values[seed][name]),
                "mean": fmean(values[seed][name]),
                "sd": (
                    stdev(values[seed][name])
                    if len(values[seed][name]) > 1
                    else 0.0
                ),
                "minimum": min(values[seed][name]),
                "maximum": max(values[seed][name]),
            }
            for name in ACTION_TYPES
        }
    artifact = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": CALIBRATION_PROTOCOL,
            "status": "complete",
            "evaluation_panel_opened": False,
            "checkpoint_selection_performed": False,
            "training_runs": 0,
            "predictor_input_lambda": PREDICTOR_INPUT_LAMBDA,
            "procedure": (
                "candidate_weighted_mean_frozen_future_prediction_by_model_seed_"
                "and_action_type_over_all_unique_frontiers_from_the_nine_"
                "predeclared_training_probes"
            ),
            "state_source_policy": (
                "frozen_seed0_controller_at_the_nine_predeclared_probe_lambdas"
            ),
            "unique_frontiers": len(frontiers),
            "candidate_observations_per_seed": sum(
                len(items) for items in values[0].values()
            ),
            "probe_runs": probe_runs,
            "frontier_digests": [item[0] for item in frontiers],
            "constants": constants,
            "distributions": distributions,
            "provenance": provenance,
        },
        "calibration_sha256",
    )
    _atomic_json(path, artifact)
    return {
        "status": "calibrated",
        "reused_existing_artifact": False,
        "evaluation_panel_opened": False,
        "unique_frontiers": len(frontiers),
        "candidate_observations_per_seed": artifact[
            "candidate_observations_per_seed"
        ],
        "calibration": str(path),
        "constants": constants,
    }


def _load_calibration(output: Path) -> dict:
    artifact = _load_json(output / CALIBRATION_NAME, "E20 calibration")
    _verify_hash(artifact, "calibration_sha256", "E20 calibration")
    if artifact.get("provenance") != _calibration_provenance():
        raise E20Error("E20 calibration sources or frozen checkpoints changed")
    if artifact.get("evaluation_panel_opened") is not False:
        raise E20Error("E20 calibration did not remain outside the test panel")
    for seed in MODEL_SEEDS:
        values = artifact.get("constants", {}).get(str(seed), {})
        if set(values) != set(ACTION_TYPES) or not all(
            math.isfinite(float(value)) for value in values.values()
        ):
            raise E20Error(f"seed-{seed} constants are incomplete or invalid")
    return artifact


def _control_bundle() -> dict:
    e16_contract, _ = e16d.authenticate(E16_OUTPUT)
    report = _load_json(E16_OUTPUT / e16d.REPORT_NAME, "E16-D report")
    _verify_hash(report, "report_sha256", "E16-D report")
    if (
        report.get("contract_sha256") != e16_contract["contract_sha256"]
        or report.get("status") != "complete"
        or report.get("scope") != "full_90_pair_confirmation"
    ):
        raise E20Error("E16-D control report is not a complete confirmation")
    controls = {name: {} for name in CONTROL_COMPONENTS}
    files = []
    for e20_name, component in CONTROL_COMPONENTS.items():
        for seed in MODEL_SEEDS:
            for instance_seed in INSTANCE_SEEDS:
                path = (
                    E16_OUTPUT
                    / "run-ledger"
                    / component
                    / f"model-{seed}"
                    / f"instance-{instance_seed}.json"
                )
                ledger = _load_json(path, f"E16-D {e20_name} control")
                _verify_hash(ledger, "ledger_sha256", "E16-D control ledger")
                if (
                    ledger.get("contract_sha256")
                    != e16_contract["contract_sha256"]
                    or ledger.get("spec", {}).get("component_arm") != component
                ):
                    raise E20Error("E16-D control ledger binding changed")
                row = ledger.get("run")
                key = (int(row["model_seed"]), int(row["instance_seed"]))
                if key != (seed, instance_seed):
                    raise E20Error("E16-D control identity changed")
                controls[e20_name][key] = row
                files.append(
                    {
                        "relative_path": str(path.relative_to(PROJECT_ROOT)),
                        "raw_sha256": _sha(path),
                    }
                )
    return {
        "contract": e16_contract,
        "report": report,
        "controls": controls,
        "files": files,
    }


def _contract(output: Path) -> dict:
    calibration = _load_calibration(output)
    controls = _control_bundle()
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "does_a_fixed_per_action_type_handling_penalty_reproduce_the_"
                "benefit_of_state_dependent_type_or_full_future_predictions"
            ),
            "model_seeds": list(MODEL_SEEDS),
            "instance_seeds": list(INSTANCE_SEEDS),
            "deployment_lambda": DEPLOYMENT_LAMBDA,
            "arms": list(ARMS),
            "new_rollout_arm": CONSTANT_TYPE,
            "reused_controls": {
                STATE_DEPENDENT_TYPE: "E16-D action_type_only",
                FULL_FUTURE: "E16-D full_future",
            },
            "new_rollout_rows": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
            "reused_control_rows": 2 * len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_values": calibration["constants"],
            "calibration_excludes_test_panel": True,
            "equivalence_rule": (
                "paired_two_sided_90_percent_student_t_interval_must_lie_"
                "strictly_within_the_predeclared_symmetric_margin"
            ),
            "equivalence_alpha": 0.05,
            "equivalence_margins": EQUIVALENCE_MARGINS,
            "e16d_contract_sha256": controls["contract"]["contract_sha256"],
            "e16d_report_sha256": controls["report"]["report_sha256"],
            "e16d_control_set_sha256": hashlib.sha256(
                json.dumps(
                    controls["files"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "held_fixed": [
                "frozen_operational_and_handling_checkpoints",
                "exact_SAFE_candidate_frontier",
                "recovery_witness_liveness_guard",
                "hierarchical_mode_aggregation_and_tie_breaking",
                "EpisodeInstance",
                "current_corrected_macro_executor",
                "lambda_point_0.10",
            ],
            "training_or_checkpoint_selection": False,
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = _contract(output)
    path = output / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, "E20 contract")
        _verify_hash(observed, "contract_sha256", "E20 contract")
        if observed != expected:
            raise E20Error("E20 contract, calibration, controls, or sources changed")
    else:
        allowed = {CALIBRATION_NAME}
        unexpected = {item.name for item in output.iterdir()} - allowed
        if unexpected:
            raise E20Error(f"unexpected files before E20 freeze: {sorted(unexpected)}")
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "training_runs": 0,
        "calibration_uses_test_panel": False,
        "new_pilot_rollouts": PILOT_INSTANCE_COUNT,
        "new_confirmation_rollouts": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "reused_control_rows": expected["reused_control_rows"],
        "contract": str(path),
        "constants": expected["calibration_values"],
    }


def authenticate(output: Path):
    contract = _load_json(output / CONTRACT_NAME, "E20 contract")
    _verify_hash(contract, "contract_sha256", "E20 contract")
    if contract != _contract(output):
        raise E20Error("E20 contract, calibration, controls, or sources changed")
    return contract, _load_calibration(output), _control_bundle()


class ConstantTypeAgent:
    """Replace the learned future signal by four deployment-fixed constants."""

    def __init__(self, agent: ConditionedHandlingAgent, constants: Mapping) -> None:
        if set(constants) != set(ACTION_TYPES):
            raise ValueError("constant-type adapter needs exactly four action types")
        self.agent = agent
        self.base_agent = agent.base_agent
        self.config = agent.config
        self.device = agent.device
        self.constants = {name: float(constants[name]) for name in ACTION_TYPES}

    def reset_episode_state(self):
        return self.agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if training or float(epsilon) != 0.0:
            raise E20Error("E20 permits deterministic frozen deployment only")
        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE E20 candidate")
        features = detached_v11_features(self.base_agent.Q_local, prepared.records)
        action_types = tuple(str(record.action_type) for record in prepared.records)
        unknown = sorted(set(action_types) - set(self.constants))
        if unknown:
            raise E20Error(f"unknown deployment action types: {unknown}")
        with torch.no_grad():
            operational = self.base_agent.Q_local.q_head(features).squeeze(-1)
            immediate = torch.as_tensor(
                [1.0 if name == "reconfigure" else 0.0 for name in action_types],
                dtype=features.dtype,
                device=features.device,
            )
            fixed_future = torch.as_tensor(
                [self.constants[name] for name in action_types],
                dtype=features.dtype,
                device=features.device,
            )
            handling = immediate + fixed_future
            merit = operational - DEPLOYMENT_LAMBDA * handling

        hierarchy = select_hierarchical_index(
            merit,
            prepared.mode_ids,
            self.base_agent.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        selected_index = 0 if liveness_forced else hierarchy.selected_index
        source = (
            "exact_recovery_witness_guard"
            if liveness_forced
            else "regularized_mode_map_candidate_map_constant_type_handling"
        )
        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        base_decision = ViabilityGraphDecision(
            candidate=candidate,
            record=prepared.records[selected_index],
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merit.detach().cpu()),
            mode_values=tuple(
                (ID_TO_MODE[int(mode)], float(value))
                for mode, value in hierarchy.mode_values
            ),
            explored=False,
            selection_source=source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[source] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        self.agent.decision_count += 1
        self.agent.preference_decisions[str(DEPLOYMENT_LAMBDA)] += 1
        return ConditionedHandlingDecision(
            base_decision=base_decision,
            preference_lambda=DEPLOYMENT_LAMBDA,
            operational_values=tuple(float(value) for value in operational.cpu()),
            handling_values=tuple(float(value) for value in handling.cpu()),
            chosen_feature=features[selected_index].detach().cpu().clone(),
            chosen_immediate=int(immediate[selected_index].item()),
        )

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _factory(seed: int, constants: Mapping, arms: Mapping[int, object]):
    def factory(base):
        agent = _new_conditioned_agent(seed, base, arms)
        return ConstantTypeAgent(agent, constants)

    return factory


def _spec(seed: int) -> dict:
    return {
        "component_arm": CONSTANT_TYPE,
        "model_seed": int(seed),
        "deployment_lambda": DEPLOYMENT_LAMBDA,
        "predictor_input_lambda": None,
        "constant_calibration_seed": int(seed),
    }


def _ledger_path(output: Path, seed: int, instance_seed: int) -> Path:
    return (
        output
        / "run-ledger"
        / CONSTANT_TYPE
        / f"model-{seed}"
        / f"instance-{instance_seed}.json"
    )


def run(
    output: Path,
    *,
    selected_seed: Optional[int],
    instance_limit: Optional[int],
    device_name: str,
) -> dict:
    output = output.resolve()
    contract, _calibration, controls = authenticate(output)
    seeds = MODEL_SEEDS if selected_seed is None else (int(selected_seed),)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise E20Error("model seed must be 0, 1, or 2")
    records = list(controls["contract"]["instance_seeds"])
    if records != list(INSTANCE_SEEDS):
        raise E20Error("E16-D instance grid changed")
    bundle = e16d._artifact_bundle()
    manifest_records = list(bundle["records"])
    if instance_limit is not None:
        if not 1 <= int(instance_limit) <= len(manifest_records):
            raise E20Error("invalid E20 instance limit")
        manifest_records = manifest_records[: int(instance_limit)]
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E20Error("E20 is frozen to CPU execution")
    torch.set_num_threads(1)
    arms = e16b._frozen_arms()
    expected = len(seeds) * len(manifest_records)
    completed = safe = 0
    for seed in seeds:
        spec = _spec(seed)
        constants = contract["calibration_values"][str(seed)]
        for record in manifest_records:
            identity = e16d.e04._identity(record)
            instance_seed = int(identity["instance_seed"])
            path = _ledger_path(output, seed, instance_seed)
            if path.is_file():
                ledger = _load_json(path, "E20 rollout ledger")
                _verify_hash(ledger, "ledger_sha256", "E20 rollout ledger")
                if (
                    ledger.get("contract_sha256") != contract["contract_sha256"]
                    or ledger.get("spec") != spec
                ):
                    raise E20Error("E20 rollout ledger binding changed")
                row = ledger["run"]
            else:
                instance = e16d._load_instance(record)
                try:
                    raw = pilot._run_raw(
                        arms[seed],
                        instance,
                        device=device,
                        wrapper_factory=_factory(seed, constants, arms),
                    )
                    compact = pilot._compact_row(raw, instance)
                    row = e16d._row(raw, compact, identity, spec)
                except Exception as error:
                    row = e16d._failed_row(error, identity, spec)
                row.update(
                    {
                        "protocol": PROTOCOL,
                        "e20_arm": CONSTANT_TYPE,
                        "control_provenance": "new_e20_rollout",
                        "constant_penalties": dict(constants),
                    }
                )
                ledger = _with_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "spec": spec,
                        "run": row,
                    },
                    "ledger_sha256",
                )
                _atomic_json(path, ledger)
            completed += 1
            safe += int(row["strict_safe_complete"])
            print(
                f"E20 {completed}/{expected} | seed={seed} | constant_type | "
                f"instance={instance_seed} | safe={int(row['strict_safe_complete'])}",
                flush=True,
            )
    return {
        "status": "complete",
        "training_runs": 0,
        "new_rollouts": completed,
        "strict_safe_complete_rows": safe,
        "model_seeds": list(seeds),
        "instances_per_seed": len(manifest_records),
        "reused_controls_run": False,
    }


def _new_rows(output: Path) -> list[dict]:
    rows = []
    root = output / "run-ledger" / CONSTANT_TYPE
    for path in sorted(root.glob("model-*/instance-*.json")):
        ledger = _load_json(path, "E20 rollout ledger")
        _verify_hash(ledger, "ledger_sha256", "E20 rollout ledger")
        rows.append(ledger["run"])
    return rows


def _paired_summary(
    constant_rows: Sequence[Mapping],
    control_rows: Mapping[tuple[int, int], Mapping],
) -> dict:
    pairs = []
    for row in constant_rows:
        key = (int(row["model_seed"]), int(row["instance_seed"]))
        control = control_rows.get(key)
        if control is not None:
            pairs.append((row, control))
    result = {
        "pairs": len(pairs),
        "difference_direction": "constant_type_minus_control",
        "both_strict_safe_complete": sum(
            left["strict_safe_complete"] and right["strict_safe_complete"]
            for left, right in pairs
        ),
        "identical_behavior_digest": sum(
            left.get("behavior_digest") == right.get("behavior_digest")
            for left, right in pairs
        ),
        "metrics": {},
    }
    for metric in e16d.METRICS:
        values = [
            float(left[metric]) - float(right[metric])
            for left, right in pairs
            if left["strict_safe_complete"] and right["strict_safe_complete"]
        ]
        mean = fmean(values) if values else None
        sd = stdev(values) if len(values) > 1 else None
        if len(values) > 1:
            standard_error = sd / math.sqrt(len(values))
            ci95_half = float(student_t.ppf(0.975, len(values) - 1)) * standard_error
            ci90_half = float(student_t.ppf(0.95, len(values) - 1)) * standard_error
            ci95 = [mean - ci95_half, mean + ci95_half]
            ci90 = [mean - ci90_half, mean + ci90_half]
        else:
            ci95 = ci90 = None
        margin = EQUIVALENCE_MARGINS.get(metric)
        equivalence = (
            None
            if margin is None or ci90 is None
            else ci90[0] > -margin and ci90[1] < margin
        )
        result["metrics"][metric] = {
            "n": len(values),
            "mean": mean,
            "sd": sd,
            "ci95_student_t": ci95,
            "equivalence_margin": margin,
            "equivalence_ci90_student_t": ci90 if margin is not None else None,
            "practical_equivalence_supported": equivalence,
        }
    return result


def _table(report: Mapping) -> str:
    lines = [
        "# E20 fixed action-type penalty ablation",
        "",
        "| Arm | Provenance | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = report["aggregate"][arm]

        def value(name: str, digits: int = 2) -> str:
            observed = item[name]
            return "—" if observed is None else f"{observed:.{digits}f}"

        provenance = "new E20" if arm == CONSTANT_TYPE else "reused E16-D"
        lines.append(
            f"| {arm} | {provenance} | "
            f"{item['strict_safe_complete']}/{item['rows']} | "
            f"{value('dense_return')} | {value('mean_absolute_error')} | "
            f"{value('within_target_window_rate', 3)} | {value('steps')} | "
            f"{value('physical_rehandles_per_100_required_deliveries')} |"
        )
    lines.extend(
        [
            "",
            "The constants were fixed from non-test training-probe frontiers before the 92k panel was evaluated.",
            "State-dependent-type and full-future rows are authenticated E16-D controls, not independent replications.",
            "",
            "## Practical-equivalence checks (constant minus control)",
            "",
            "| Control | Metric | Mean difference | 90% t CI | Margin | Equivalent |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for control in (STATE_DEPENDENT_TYPE, FULL_FUTURE):
        for metric, item in report["paired_constant_minus_control"][control][
            "metrics"
        ].items():
            if item["equivalence_margin"] is None:
                continue
            interval = item["equivalence_ci90_student_t"]
            rendered = "—" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
            lines.append(
                f"| {control} | {metric} | {item['mean']:.3f} | {rendered} | "
                f"±{item['equivalence_margin']:.3f} | "
                f"{'yes' if item['practical_equivalence_supported'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, pilot_only: bool) -> dict:
    output = output.resolve()
    contract, calibration, control_bundle = authenticate(output)
    rows = _new_rows(output)
    if pilot_only:
        allowed = set(INSTANCE_SEEDS[:PILOT_INSTANCE_COUNT])
        rows = [
            row
            for row in rows
            if int(row["model_seed"]) == PILOT_MODEL_SEED
            and int(row["instance_seed"]) in allowed
        ]
        expected = PILOT_INSTANCE_COUNT
    else:
        expected = len(MODEL_SEEDS) * len(INSTANCE_SEEDS)
    keys = [(int(row["model_seed"]), int(row["instance_seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise E20Error("duplicate E20 rollout coordinates")
    if len(rows) != expected:
        raise E20Error(f"E20 grid is incomplete: {len(rows)}/{expected}")

    allowed_keys = set(keys)
    controls = {
        name: {
            key: row
            for key, row in control_bundle["controls"][name].items()
            if key in allowed_keys
        }
        for name in CONTROL_COMPONENTS
    }
    if any(len(items) != expected for items in controls.values()):
        raise E20Error("matched E16-D controls are incomplete")
    by_arm = {
        CONSTANT_TYPE: rows,
        STATE_DEPENDENT_TYPE: list(controls[STATE_DEPENDENT_TYPE].values()),
        FULL_FUTURE: list(controls[FULL_FUTURE].values()),
    }
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete",
            "paper_evidence": not pilot_only,
            "scope": (
                "seed0_five_instance_pilot"
                if pilot_only
                else "full_90_pair_confirmation"
            ),
            "contract_sha256": contract["contract_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
            "training_runs": 0,
            "new_rollout_rows": len(rows),
            "reused_control_rows": 2 * len(rows),
            "controls_are_reanalyses_not_independent_replications": True,
            "aggregate": {arm: e16d._aggregate(by_arm[arm]) for arm in ARMS},
            "paired_constant_minus_control": {
                name: _paired_summary(rows, controls[name])
                for name in (STATE_DEPENDENT_TYPE, FULL_FUTURE)
            },
            "equivalence_margins": EQUIVALENCE_MARGINS,
            "interpretation": {
                "constant_equivalent_to_both": (
                    "static per-type deployment penalties reproduce the tested benefit; "
                    "state-dependent inference is not justified by this panel"
                ),
                "state_dependent_type_better": (
                    "context-dependent action-type adjustment adds deployment value"
                ),
                "full_future_better_than_state_dependent_type": (
                    "candidate-specific future predictions add value beyond type adjustment"
                ),
                "scope": (
                    "deployment utility at lambda=.10; not predictor accuracy and not "
                    "evidence that preference-input conditioning is useful"
                ),
            },
        },
        "report_sha256",
    )
    report_path = output / (PILOT_REPORT_NAME if pilot_only else REPORT_NAME)
    _atomic_json(report_path, report)
    if not pilot_only:
        _atomic_text(output / TABLE_NAME, _table(report))
    return {
        "status": "complete",
        "scope": report["scope"],
        "training_runs": 0,
        "new_rollout_rows": len(rows),
        "reused_control_rows": report["reused_control_rows"],
        "strict_safe_complete_new_rows": sum(
            int(row["strict_safe_complete"]) for row in rows
        ),
        "report": str(report_path),
        "table": None if pilot_only else str(output / TABLE_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "calibrate",
            "prepare",
            "run-pilot",
            "analyze-pilot",
            "run-confirmation",
            "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "calibrate":
        result = calibrate(args.output, device_name=args.device)
    elif args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run-pilot":
        result = run(
            args.output,
            selected_seed=PILOT_MODEL_SEED,
            instance_limit=PILOT_INSTANCE_COUNT,
            device_name=args.device,
        )
    elif args.command == "analyze-pilot":
        result = analyze(args.output, pilot_only=True)
    elif args.command == "run-confirmation":
        result = run(
            args.output,
            selected_seed=None,
            instance_limit=None,
            device_name=args.device,
        )
    else:
        result = analyze(args.output, pilot_only=False)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
