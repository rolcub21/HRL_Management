#!/usr/bin/env python3
"""E16-D: matched rollout ablation of future-signal components.

The experiment uses the frozen E5(c) EpisodeInstances and lambda=.10. It
replays the current exact-SAFE deployment stack with four merits: immediate
cost only, the action-type component of the learned future estimate, the
within-action-type candidate residual, and the complete learned estimate.
No network is trained or selected here.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    run_future_consequence as e05c,
)
from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    continuation_bank as e16b,
)
from PSLAP.viability_candidates import ViabilityMode
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
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


PROTOCOL = "vcg_conditioned_e16_component_rollout_ablation_92k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = tuple(range(92_000, 92_030))
DEPLOYMENT_LAMBDA = 0.10
PREDICTOR_INPUT_LAMBDA = 0.10
PILOT_MODEL_SEED = 0
PILOT_INSTANCE_COUNT = 5

IMMEDIATE_ONLY = "immediate_only"
ACTION_TYPE_ONLY = "action_type_only"
CANDIDATE_RESIDUAL_ONLY = "candidate_residual_only"
FULL_FUTURE = "full_future"
ARMS = (
    IMMEDIATE_ONLY,
    ACTION_TYPE_ONLY,
    CANDIDATE_RESIDUAL_ONLY,
    FULL_FUTURE,
)

CONTRACT_NAME = "e16d-contract.json"
PILOT_REPORT_NAME = "e16d-pilot-report.json"
REPORT_NAME = "e16d-report.json"
TABLE_NAME = "e16d-results-table.md"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e16-component-rollout-ablation-92k"
)
E04_OUTPUT = e04.DEFAULT_OUTPUT
E05C_OUTPUT = e05c.DEFAULT_OUTPUT


class E16DError(RuntimeError):
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
        raise E16DError(f"missing regular input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E16DError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E16DError(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise E16DError(f"{label} self-hash mismatch")


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


def component_future_values(
    future: torch.Tensor,
    action_types: Sequence[str],
    component: str,
) -> torch.Tensor:
    """Return the future estimate retained by one E16-D component arm."""

    if future.ndim != 1 or len(action_types) != future.numel():
        raise ValueError("future predictions and action types must align")
    if not bool(torch.isfinite(future).all()):
        raise ValueError("future predictions must be finite")
    if component == IMMEDIATE_ONLY:
        return torch.zeros_like(future)
    if component == FULL_FUTURE:
        return future
    if component not in (ACTION_TYPE_ONLY, CANDIDATE_RESIDUAL_ONLY):
        raise ValueError(f"unknown E16-D component: {component}")

    state_mean = future.mean()
    type_means = {}
    for name in sorted(set(str(value) for value in action_types)):
        indices = [index for index, value in enumerate(action_types) if value == name]
        type_means[name] = future[indices].mean()
    if component == ACTION_TYPE_ONLY:
        return torch.stack([type_means[str(name)] for name in action_types])
    # mu(s) + r(s,c), where r is the deviation from the action-type mean.
    return torch.stack(
        [
            state_mean + future[index] - type_means[str(name)]
            for index, name in enumerate(action_types)
        ]
    )


class ComponentHandlingAgent:
    """Change only which component of the frozen future head enters merit."""

    def __init__(self, agent: ConditionedHandlingAgent, component: str) -> None:
        if component not in (ACTION_TYPE_ONLY, CANDIDATE_RESIDUAL_ONLY):
            raise ValueError("component adapter requires a partial future signal")
        self.agent = agent
        self.base_agent = agent.base_agent
        self.config = agent.config
        self.device = agent.device
        self.component = component

    def reset_episode_state(self):
        return self.agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if training or float(epsilon) != 0.0:
            raise E16DError("E16-D permits deterministic frozen deployment only")
        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE E16-D candidate")
        features = detached_v11_features(self.base_agent.Q_local, prepared.records)
        action_types = tuple(str(record.action_type) for record in prepared.records)
        with torch.no_grad():
            operational = self.base_agent.Q_local.q_head(features).squeeze(-1)
            immediate = torch.as_tensor(
                [1.0 if name == "reconfigure" else 0.0 for name in action_types],
                dtype=features.dtype,
                device=features.device,
            )
            full_future = self.agent.handling_network(
                features, PREDICTOR_INPUT_LAMBDA
            )
            retained_future = component_future_values(
                full_future, action_types, self.component
            )
            handling = immediate + retained_future
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
            else "regularized_mode_map_candidate_map_handling_augmented"
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


def _artifact_bundle() -> dict:
    e04_contract = _load_json(E04_OUTPUT / e04.CONTRACT_NAME, "frozen E4 contract")
    manifest = _load_json(E04_OUTPUT / e04.MANIFEST_NAME, "frozen E4 manifest")
    e04_report = _load_json(E04_OUTPUT / e04.REPORT_NAME, "frozen E4 report")
    e05c_contract = _load_json(
        E05C_OUTPUT / e05c.CONTRACT_NAME, "frozen E5(c) contract"
    )
    e05c_report = _load_json(
        E05C_OUTPUT / e05c.REPORT_NAME, "frozen E5(c) report"
    )
    for value, field, label in (
        (e04_contract, "contract_sha256", "E4 contract"),
        (manifest, "manifest_sha256", "E4 manifest"),
        (e04_report, "report_sha256", "E4 report"),
        (e05c_contract, "contract_sha256", "E5(c) contract"),
        (e05c_report, "report_sha256", "E5(c) report"),
    ):
        _verify_hash(value, field, label)
    if (
        manifest.get("contract_sha256") != e04_contract["contract_sha256"]
        or e04_report.get("contract_sha256") != e04_contract["contract_sha256"]
        or e05c_contract.get("e04_contract_sha256")
        != e04_contract["contract_sha256"]
        or e05c_contract.get("e04_manifest_sha256")
        != manifest["manifest_sha256"]
        or e05c_report.get("contract_sha256")
        != e05c_contract["contract_sha256"]
        or e04_report.get("status") != "complete"
        or e05c_report.get("status") != "complete"
    ):
        raise E16DError("frozen E4/E5(c) artifact bindings are invalid")
    records = manifest.get("instances")
    if not isinstance(records, list) or [item.get("seed") for item in records] != list(
        INSTANCE_SEEDS
    ):
        raise E16DError("frozen E4 EpisodeInstance grid changed")

    inputs = []
    controls = {IMMEDIATE_ONLY: {}, FULL_FUTURE: {}}
    for record in records:
        instance_path = E04_OUTPUT / str(record["relative_path"])
        if _sha(instance_path) != record["raw_sha256"]:
            raise E16DError("frozen E4 EpisodeInstance bytes changed")
        inputs.append(
            {
                "relative_path": str(instance_path.relative_to(PROJECT_ROOT)),
                "raw_sha256": record["raw_sha256"],
                "canonical_sha256": record["canonical_sha256"],
                "instance_seed": int(record["seed"]),
                "episode_instance_id": record["episode_instance_id"],
                "schedule_id": record["schedule_id"],
            }
        )
        for model_seed in MODEL_SEEDS:
            instance_seed = int(record["seed"])
            full_path = (
                E04_OUTPUT
                / "run-ledger"
                / f"conditioned_vcg__model-{model_seed}__lambda-0.100"
                / f"instance-{instance_seed}.json"
            )
            immediate_path = (
                E05C_OUTPUT
                / "run-ledger"
                / f"model-{model_seed}"
                / "lambda-0.100"
                / f"instance-{instance_seed}.json"
            )
            for name, path, expected_contract in (
                (FULL_FUTURE, full_path, e04_contract["contract_sha256"]),
                (IMMEDIATE_ONLY, immediate_path, e05c_contract["contract_sha256"]),
            ):
                ledger = _load_json(path, f"historical {name} ledger")
                _verify_hash(ledger, "ledger_sha256", f"historical {name} ledger")
                if ledger.get("contract_sha256") != expected_contract:
                    raise E16DError(f"historical {name} ledger binding changed")
                row = ledger.get("run")
                if (
                    not isinstance(row, dict)
                    or int(row.get("model_seed", -1)) != model_seed
                    or int(row.get("instance_seed", -1)) != instance_seed
                    or float(row.get("preference_lambda", -1.0))
                    != DEPLOYMENT_LAMBDA
                ):
                    raise E16DError(f"historical {name} row identity changed")
                controls[name][(model_seed, instance_seed)] = row
                inputs.append(
                    {
                        "relative_path": str(path.relative_to(PROJECT_ROOT)),
                        "raw_sha256": _sha(path),
                    }
                )
    return {
        "e04_contract": e04_contract,
        "manifest": manifest,
        "e04_report": e04_report,
        "e05c_contract": e05c_contract,
        "e05c_report": e05c_report,
        "records": records,
        "controls": controls,
        "input_files": inputs,
    }


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "benchmark_viability_critic_priority.py",
        PROJECT_ROOT / "vcg_v11_conditioned_handling.py",
        PROJECT_ROOT / "vcg_v11_nested_handling.py",
        PROJECT_ROOT / "viability_graph_hierarchy.py",
        PROJECT_ROOT / "viability_graph_preference_conditioned.py",
        PROJECT_ROOT / "train_viability_graph_smdp.py",
        PROJECT_ROOT / "example/Options/DirectDeliverOption.py",
        PROJECT_ROOT / "example/Options/ReconfigureOption.py",
        PROJECT_ROOT / "example/Options/certified_path.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}


def _contract() -> dict:
    bundle = _artifact_bundle()
    arms = e16b._frozen_arms()
    checkpoints = {}
    for seed in MODEL_SEEDS:
        terminal = e16b._terminal(seed)["agent_checkpoint"]
        if (
            terminal.get("base_checkpoint_sha256") != arms[seed].checkpoint_sha256
            or terminal.get("base_policy_digest")
            != arms[seed].deployment_policy_digest
        ):
            raise E16DError(f"conditioned seed-{seed} does not bind its base arm")
        checkpoints[str(seed)] = {
            "operational_checkpoint_sha256": arms[seed].checkpoint_sha256,
            "operational_policy_digest": arms[seed].deployment_policy_digest,
            "conditioned_terminal_sha256": e16b.TERMINAL_SHA256[seed],
        }
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "does_the_action_type_or_within_type_candidate_component_of_the_"
                "frozen_future_signal_reproduce_its_episode_level_benefit"
            ),
            "model_seeds": list(MODEL_SEEDS),
            "instance_seeds": list(INSTANCE_SEEDS),
            "deployment_lambda": DEPLOYMENT_LAMBDA,
            "predictor_input_lambda": PREDICTOR_INPUT_LAMBDA,
            "arms": list(ARMS),
            "merits": {
                IMMEDIATE_ONLY: "Qop-lambda*I_reconfigure",
                ACTION_TYPE_ONLY: (
                    "Qop-lambda*(I_reconfigure+mu(s)+beta_action_type(s))"
                ),
                CANDIDATE_RESIDUAL_ONLY: (
                    "Qop-lambda*(I_reconfigure+mu(s)+r(s,c))"
                ),
                FULL_FUTURE: "Qop-lambda*(I_reconfigure+N_future(s,c,.10))",
            },
            "decomposition": (
                "N_future=mu_state+beta_action_type+within_type_candidate_residual"
            ),
            "held_fixed": [
                "frozen_operational_and_handling_weights",
                "exact_SAFE_candidate_frontier",
                "recovery_witness_liveness_guard",
                "hierarchical_mode_aggregation_and_tie_breaking",
                "EpisodeInstance",
                "current_corrected_macro_executor",
                "lambda_point_0.10",
            ],
            "training_or_checkpoint_selection": False,
            "pilot": {
                "model_seed": PILOT_MODEL_SEED,
                "first_instances": PILOT_INSTANCE_COUNT,
                "rows": PILOT_INSTANCE_COUNT * len(ARMS),
                "includes_current_control_replays": True,
                "purpose": "runtime_semantics_and_historical_control_parity",
            },
            "confirmation_rows": len(MODEL_SEEDS)
            * len(INSTANCE_SEEDS)
            * len(ARMS),
            "historical_controls_are_comparability_checks_not_substitutes": True,
            "e04_contract_sha256": bundle["e04_contract"]["contract_sha256"],
            "e04_manifest_sha256": bundle["manifest"]["manifest_sha256"],
            "e04_report_sha256": bundle["e04_report"]["report_sha256"],
            "e05c_contract_sha256": bundle["e05c_contract"]["contract_sha256"],
            "e05c_report_sha256": bundle["e05c_report"]["report_sha256"],
            "frozen_input_set_sha256": hashlib.sha256(
                json.dumps(
                    bundle["input_files"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "checkpoints": checkpoints,
            "source_sha256": _source_hashes(),
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = _contract()
    path = output / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, "E16-D contract")
        _verify_hash(observed, "contract_sha256", "E16-D contract")
        if observed != expected:
            raise E16DError("E16-D contract, sources, or frozen artifacts changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E16DError("nonempty E16-D output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "training_runs": 0,
        "pilot_rows": expected["pilot"]["rows"],
        "confirmation_rows": expected["confirmation_rows"],
        "contract": str(path),
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    contract = _load_json(output / CONTRACT_NAME, "E16-D contract")
    _verify_hash(contract, "contract_sha256", "E16-D contract")
    if contract != _contract():
        raise E16DError("E16-D contract, sources, or frozen artifacts changed")
    return contract, _artifact_bundle()


def _load_instance(record: Mapping):
    path = E04_OUTPUT / str(record["relative_path"])
    if _sha(path) != record["raw_sha256"]:
        raise E16DError("E16-D EpisodeInstance changed")
    instance = e04.final90.final86.EpisodeInstance.from_json(
        path.read_text(encoding="utf-8")
    )
    if (
        int(instance.seed) != int(record["seed"])
        or instance.instance_id != record["episode_instance_id"]
        or instance.schedule_id != record["schedule_id"]
    ):
        raise E16DError("E16-D EpisodeInstance identity changed")
    return instance


def _factory(model_seed: int, component: str, arms: Mapping[int, object]):
    arm = arms[int(model_seed)]

    def factory(base):
        checkpoint = e16b._terminal(model_seed)["agent_checkpoint"]
        agent = ConditionedHandlingAgent.from_checkpoint(
            checkpoint,
            base_agent=base,
            expected_base_checkpoint_sha256=arm.checkpoint_sha256,
            expected_base_policy_digest=arm.deployment_policy_digest,
            expected_source_cost_sha256=checkpoint["source_cost_sha256"],
            seed=int(model_seed),
        )
        agent.handling_network = e16b.ClampedPreferenceNetwork(
            agent.handling_network, PREDICTOR_INPUT_LAMBDA
        ).to(base.device)
        agent.handling_network.requires_grad_(False).eval()
        agent.set_epsilon(0.0)
        if component == IMMEDIATE_ONLY:
            agent.handling_network = e05c.ZeroFutureHandlingNetwork().to(
                base.device
            ).eval()
            return conditioned_seed0._FixedLambdaAgent(agent, DEPLOYMENT_LAMBDA)
        if component == FULL_FUTURE:
            return conditioned_seed0._FixedLambdaAgent(agent, DEPLOYMENT_LAMBDA)
        return ComponentHandlingAgent(agent, component)

    return factory


def _spec(model_seed: int, component: str) -> dict:
    return {
        "component_arm": component,
        "model_seed": int(model_seed),
        "deployment_lambda": DEPLOYMENT_LAMBDA,
        "predictor_input_lambda": PREDICTOR_INPUT_LAMBDA,
    }


def _ledger_path(output: Path, spec: Mapping, instance_seed: int) -> Path:
    return (
        output
        / "run-ledger"
        / str(spec["component_arm"])
        / f"model-{int(spec['model_seed'])}"
        / f"instance-{int(instance_seed)}.json"
    )


def _row(raw: Mapping, compact: Mapping, identity: Mapping, spec: Mapping) -> dict:
    proxy = {
        "ranking_signal": spec["component_arm"],
        "model_seed": spec["model_seed"],
        "preference_lambda": spec["deployment_lambda"],
        "ranking_seed": None,
    }
    row = e04._row(raw, compact, identity=identity, spec=proxy)
    counts = Counter(str(item["selected_action_type"]) for item in raw["decisions"])
    row.update(
        {
            "protocol": PROTOCOL,
            **dict(spec),
            "macro_decisions": int(raw["macro_decisions"]),
            "action_decision_counts": dict(sorted(counts.items())),
            "decision_trace": [
                {
                    "decision_index": int(item["decision_index"]),
                    "selected_key": item["selected_key"],
                    "selected_mode": item["selected_mode"],
                    "selected_action_type": item["selected_action_type"],
                    "liveness_forced": bool(item["liveness_forced"]),
                    "duration": int(item["duration"]),
                    "physical_storage_relocations": int(
                        item["physical_storage_relocations"]
                    ),
                    "delivery_deviations": list(item["delivery_deviations"]),
                }
                for item in raw["decisions"]
            ],
            "current_corrected_executor": True,
        }
    )
    return row


def _failed_row(error: Exception, identity: Mapping, spec: Mapping) -> dict:
    proxy = {
        "ranking_signal": spec["component_arm"],
        "model_seed": spec["model_seed"],
        "preference_lambda": spec["deployment_lambda"],
        "ranking_seed": None,
    }
    row = e04._failed_row(error, identity=identity, spec=proxy)
    row.update(
        {
            "protocol": PROTOCOL,
            **dict(spec),
            "macro_decisions": None,
            "action_decision_counts": None,
            "decision_trace": None,
            "current_corrected_executor": True,
        }
    )
    return row


def run(
    output: Path,
    *,
    selected_seed: Optional[int],
    instance_limit: Optional[int],
    device_name: str,
) -> dict:
    output = output.resolve()
    contract, bundle = authenticate(output)
    seeds = MODEL_SEEDS if selected_seed is None else (int(selected_seed),)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise E16DError("model seed must be 0, 1, or 2")
    records = list(bundle["records"])
    if instance_limit is not None:
        if not 1 <= int(instance_limit) <= len(records):
            raise E16DError("invalid E16-D instance limit")
        records = records[: int(instance_limit)]
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E16DError("E16-D is frozen to CPU execution")
    arms = e16b._frozen_arms()
    expected = len(seeds) * len(records) * len(ARMS)
    completed = safe = 0
    for model_seed in seeds:
        arm = arms[model_seed]
        for component in ARMS:
            spec = _spec(model_seed, component)
            for record in records:
                identity = e04._identity(record)
                path = _ledger_path(output, spec, identity["instance_seed"])
                if path.is_file():
                    ledger = _load_json(path, "E16-D rollout ledger")
                    _verify_hash(ledger, "ledger_sha256", "E16-D rollout ledger")
                    if (
                        ledger.get("contract_sha256") != contract["contract_sha256"]
                        or ledger.get("spec") != spec
                    ):
                        raise E16DError("E16-D rollout ledger binding changed")
                    row = ledger["run"]
                else:
                    instance = _load_instance(record)
                    try:
                        raw = pilot._run_raw(
                            arm,
                            instance,
                            device=device,
                            wrapper_factory=_factory(model_seed, component, arms),
                        )
                        compact = pilot._compact_row(raw, instance)
                        row = _row(raw, compact, identity, spec)
                    except Exception as error:
                        row = _failed_row(error, identity, spec)
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
                    f"E16-D {completed}/{expected} | seed={model_seed} | "
                    f"{component} | instance={identity['instance_seed']} | "
                    f"safe={int(row['strict_safe_complete'])}",
                    flush=True,
                )
    return {
        "status": "complete",
        "training_runs": 0,
        "rows": completed,
        "strict_safe_complete_rows": safe,
        "model_seeds": list(seeds),
        "instances_per_arm": len(records),
        "arms": list(ARMS),
    }


METRICS = (
    "dense_return",
    "mean_absolute_error",
    "mean_earliness",
    "mean_tardiness",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


def _rows(output: Path) -> list[dict]:
    result = []
    for path in sorted((output / "run-ledger").glob("**/instance-*.json")):
        ledger = _load_json(path, "E16-D rollout ledger")
        _verify_hash(ledger, "ledger_sha256", "E16-D rollout ledger")
        result.append(ledger["run"])
    return result


def _aggregate(rows: Sequence[Mapping]) -> dict:
    safe = [row for row in rows if row["strict_safe_complete"]]
    result = {
        "rows": len(rows),
        "strict_safe_complete": len(safe),
        "strict_completion_rate": len(safe) / len(rows) if rows else None,
        "complete_case_metrics_suppressed": len(safe) != len(rows),
    }
    for metric in METRICS:
        result[metric] = (
            fmean(float(row[metric]) for row in safe)
            if rows and len(safe) == len(rows)
            else None
        )
    return result


def _paired_delta(rows: Sequence[Mapping], arm: str) -> dict:
    by_key = {
        (
            str(row["component_arm"]),
            int(row["model_seed"]),
            int(row["instance_seed"]),
        ): row
        for row in rows
    }
    pairs = []
    for key, immediate in by_key.items():
        if key[0] != IMMEDIATE_ONLY:
            continue
        treated = by_key.get((arm, key[1], key[2]))
        if treated is not None:
            pairs.append((immediate, treated))
    result = {
        "pairs": len(pairs),
        "both_strict_safe_complete": sum(
            left["strict_safe_complete"] and right["strict_safe_complete"]
            for left, right in pairs
        ),
        "identical_behavior_digest": sum(
            left.get("behavior_digest") == right.get("behavior_digest")
            for left, right in pairs
        ),
    }
    for metric in METRICS:
        values = [
            float(right[metric]) - float(left[metric])
            for left, right in pairs
            if left["strict_safe_complete"] and right["strict_safe_complete"]
        ]
        result[f"{metric}_arm_minus_immediate"] = {
            "n": len(values),
            "mean": fmean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else None,
            "ci95_normal": (
                [
                    fmean(values) - 1.96 * stdev(values) / math.sqrt(len(values)),
                    fmean(values) + 1.96 * stdev(values) / math.sqrt(len(values)),
                ]
                if len(values) > 1
                else None
            ),
        }
    result["instances_with_lower_rehandles"] = sum(
        right["strict_safe_complete"]
        and left["strict_safe_complete"]
        and float(right["physical_rehandles_per_100_required_deliveries"])
        < float(left["physical_rehandles_per_100_required_deliveries"])
        for left, right in pairs
    )
    result["instances_with_lower_mae"] = sum(
        right["strict_safe_complete"]
        and left["strict_safe_complete"]
        and float(right["mean_absolute_error"])
        < float(left["mean_absolute_error"])
        for left, right in pairs
    )
    return result


def _historical_parity(rows: Sequence[Mapping], bundle: Mapping) -> dict:
    checks = {}
    comparison_fields = (
        "strict_safe_complete",
        "dense_return",
        "mean_absolute_error",
        "mean_earliness",
        "mean_tardiness",
        "within_target_window_rate",
        "steps",
        "physical_rehandles_per_100_required_deliveries",
    )
    for arm in (IMMEDIATE_ONLY, FULL_FUTURE):
        current = [row for row in rows if row["component_arm"] == arm]
        records = []
        for row in current:
            historical = bundle["controls"][arm][
                (int(row["model_seed"]), int(row["instance_seed"]))
            ]
            records.append(
                {
                    "model_seed": int(row["model_seed"]),
                    "instance_seed": int(row["instance_seed"]),
                    "behavior_digest_equal": (
                        row.get("behavior_digest") == historical.get("behavior_digest")
                    ),
                    "all_outcome_fields_equal": all(
                        row.get(field) == historical.get(field)
                        for field in comparison_fields
                    ),
                }
            )
        checks[arm] = {
            "rows": len(records),
            "behavior_digest_equal": sum(
                item["behavior_digest_equal"] for item in records
            ),
            "all_outcome_fields_equal": sum(
                item["all_outcome_fields_equal"] for item in records
            ),
            "records": records,
        }
    return checks


def _table(report: Mapping) -> str:
    lines = [
        "# E16-D matched component rollout ablation",
        "",
        "| Arm | Strict complete | Return | MAE | Earliness | Tardiness | Within ±20 | Steps | Rehandles/100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = report["aggregate"][arm]

        def value(name: str, digits: int = 2) -> str:
            observed = item[name]
            return "—" if observed is None else f"{observed:.{digits}f}"

        lines.append(
            f"| {arm} | {item['strict_safe_complete']}/{item['rows']} | "
            f"{value('dense_return')} | {value('mean_absolute_error')} | "
            f"{value('mean_earliness')} | {value('mean_tardiness')} | "
            f"{value('within_target_window_rate', 3)} | {value('steps')} | "
            f"{value('physical_rehandles_per_100_required_deliveries')} |"
        )
    lines.extend(
        [
            "",
            "All four arms use the current corrected executor, identical frozen checkpoints, exact-SAFE frontiers, lambda=.10, and matched EpisodeInstances.",
            "Historical E5(c) control rows are an execution-parity diagnostic; they are not substituted for current control runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, pilot_only: bool) -> dict:
    output = output.resolve()
    contract, bundle = authenticate(output)
    rows = _rows(output)
    if pilot_only:
        allowed_instances = set(INSTANCE_SEEDS[:PILOT_INSTANCE_COUNT])
        rows = [
            row
            for row in rows
            if int(row["model_seed"]) == PILOT_MODEL_SEED
            and int(row["instance_seed"]) in allowed_instances
        ]
        expected = PILOT_INSTANCE_COUNT * len(ARMS)
    else:
        expected = len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(ARMS)
    identities = [
        (row["component_arm"], int(row["model_seed"]), int(row["instance_seed"]))
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise E16DError("duplicate E16-D rollout coordinates")
    if len(rows) != expected:
        raise E16DError(f"E16-D grid is incomplete: {len(rows)}/{expected}")
    by_arm = {
        arm: [row for row in rows if row["component_arm"] == arm] for arm in ARMS
    }
    if any(not values for values in by_arm.values()):
        raise E16DError("E16-D is missing a component arm")
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
            "training_runs": 0,
            "current_rollout_rows": len(rows),
            "matched_coordinates": len(rows) // len(ARMS),
            "aggregate": {arm: _aggregate(by_arm[arm]) for arm in ARMS},
            "paired_arm_minus_immediate": {
                arm: _paired_delta(rows, arm)
                for arm in (
                    ACTION_TYPE_ONLY,
                    CANDIDATE_RESIDUAL_ONLY,
                    FULL_FUTURE,
                )
            },
            "historical_control_parity": _historical_parity(rows, bundle),
            "causal_scope": (
                "effect_of_retaining_action_type_or_within_type_candidate_"
                "components_of_the_same_frozen_future_signal_at_lambda_0.10"
            ),
        },
        "report_sha256",
    )
    name = PILOT_REPORT_NAME if pilot_only else REPORT_NAME
    _atomic_json(output / name, report)
    if not pilot_only:
        _atomic_text(output / TABLE_NAME, _table(report))
    return {
        "status": "complete",
        "scope": report["scope"],
        "training_runs": 0,
        "rows": len(rows),
        "strict_safe_complete_rows": sum(
            row["strict_safe_complete"] for row in rows
        ),
        "report": str(output / name),
        "table": None if pilot_only else str(output / TABLE_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "run-pilot",
            "run-confirmation",
            "analyze-pilot",
            "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run-pilot":
        result = run(
            args.output,
            selected_seed=PILOT_MODEL_SEED,
            instance_limit=PILOT_INSTANCE_COUNT,
            device_name=args.device,
        )
    elif args.command == "run-confirmation":
        result = run(
            args.output,
            selected_seed=None,
            instance_limit=None,
            device_name=args.device,
        )
    elif args.command == "analyze-pilot":
        result = analyze(args.output, pilot_only=True)
    else:
        result = analyze(args.output, pilot_only=False)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
