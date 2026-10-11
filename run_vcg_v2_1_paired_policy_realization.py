#!/usr/bin/env python3
"""Frozen paired MAP-versus-induced-soft realization audit for VCG V2.1.

This is an audit of one already-completed development checkpoint, not a new
training or checkpoint-selection protocol.  It reuses exactly the three
EpisodeInstances opened by the V2.1 development validation and crosses each
one with a predeclared set of controller-only RNG seeds.  The deterministic
MAP realization is executed once per instance; the induced nested regularized
policy is executed 32 times per instance.

The soft arm deliberately separates high-level policy sampling from learning:
``select(..., training=True)`` realizes the authenticated induced policy, but
the environment options remain in evaluation mode and replay/learning methods
are fail-fast forbidden.  Every row uses a fresh weight-only agent clone and a
fresh exact-certificate cache.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
from statistics import fmean, stdev
import tempfile
from typing import Callable, Mapping, Optional, Sequence

import torch

from example.episode_instance import EpisodeInstance
import viability_graph_constrained_v2 as v2
from viability_graph_constrained_v2_1 import (
    ConstrainedV21DevelopmentRuntime,
    ConstrainedV21HierarchyAgent,
    V2_1_DIAGNOSTIC_DISTRIBUTION,
)
from train_vcg_constrained_v2_1 import (
    CHECKPOINT_FAMILY,
    CHECKPOINT_SCHEMA_VERSION,
    CONTROLLER_ARCHITECTURE,
    METHOD_VERSION,
    POLICY_SCHEDULE_PROTOCOL,
    TOTAL_EPISODES,
    contract_hash,
    schedule_for_episode,
)
from vcg_v2_1_policy_realization_analysis import analyze_policy_realization


AUDIT_PROTOCOL = "vcg_v2_1_frozen_paired_policy_realization_audit_v1"
AUDIT_SCHEMA_VERSION = 1
PANEL_SEEDS = (84_000, 84_001, 84_002)
SOFT_REPLICATES = 32
SOFT_RNG_BASE = 910_000
SOFT_RNG_INSTANCE_STRIDE = 100
POLICIES = ("map", "induced_soft")
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "9364d16ab28610d7074619920e0d53961bfe3b94790d5a5f60a0d0f44b636e8a"
)
EXPECTED_SOURCE_CONTRACT_FILE_SHA256 = (
    "4b017d2ca68cdfadf2b22d635d60bc20f222064f40ccd7042450dc792e489cd4"
)
EXPECTED_SOURCE_SUMMARY_FILE_SHA256 = (
    "6fe06b15bbf8f62f89ab78d20b3dca67afc54e127a8414a6bb537bb2797305bd"
)
EXPECTED_EMBEDDED_CONTRACT_SHA256 = (
    "87c9d4804bf9bc0219a8b515202d3a193936c4c9a2232ad84b243fb7db85fc7a"
)
EXPECTED_LAMBDA = 0.147
EXPECTED_WITHIN_TEMPERATURES = (0.01, 0.01, 0.01, 0.01)
EXPECTED_GROUP_TEMPERATURE = 0.05
TWO_SIDED_T_975_DF31 = 2.0395134463964077
ONE_SIDED_T_95_DF31 = 1.695518782545865
PRIMARY_METRICS = (
    "dense_return",
    "mean_absolute_error",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
    "hold_decision_share",
    "reconfigure_decision_share",
)


class PairedPolicyAuditError(ValueError):
    """Raised when the frozen source or paired-audit contract is violated."""


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PairedPolicyAuditError("audit JSON cannot contain NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise PairedPolicyAuditError(
        f"unsupported audit JSON value {type(value).__name__}"
    )


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise PairedPolicyAuditError("Q-local tensor state is missing or empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise PairedPolicyAuditError(f"Q-local state {name!r} is not a tensor")
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedPolicyAuditError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise PairedPolicyAuditError(f"JSON artifact {path} must contain an object")
    return value


def _require_equal(name: str, observed, expected) -> None:
    if observed != expected:
        raise PairedPolicyAuditError(
            f"{name} mismatch: observed {observed!r}, expected {expected!r}"
        )


def _require_exact_float(name: str, observed, expected: float) -> float:
    if isinstance(observed, bool):
        raise PairedPolicyAuditError(f"{name} must be numeric")
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise PairedPolicyAuditError(f"{name} must be numeric") from error
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=0.0, abs_tol=0.0
    ):
        raise PairedPolicyAuditError(
            f"{name} mismatch: observed {value!r}, expected {expected!r}"
        )
    return value


@dataclass(frozen=True)
class AuthenticatedSource:
    checkpoint_path: Path
    contract_path: Path
    summary_path: Path
    checkpoint: Mapping
    contract: Mapping
    summary: Mapping
    checkpoint_sha256: str
    contract_file_sha256: str
    summary_file_sha256: str
    q_local_sha256: str
    q_local_state: Mapping[str, torch.Tensor]
    dual_lambda: float
    map_schedule: Mapping
    soft_schedule: Mapping


def _validate_loaded_source(
    checkpoint: Mapping,
    contract: Mapping,
    summary: Mapping,
) -> tuple[float, dict, dict, Mapping[str, torch.Tensor]]:
    """Authenticate semantic identity after the three files are hash-locked."""

    if contract_hash(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    ) != contract.get("contract_sha256"):
        raise PairedPolicyAuditError("source training-contract canonical hash mismatch")
    _require_equal(
        "embedded training contract SHA",
        contract.get("contract_sha256"),
        EXPECTED_EMBEDDED_CONTRACT_SHA256,
    )
    expected_contract = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "controller": CONTROLLER_ARCHITECTURE,
        "method_version": METHOD_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "episodes": TOTAL_EPISODES,
        "validation_seeds": PANEL_SEEDS,
        "development_only": True,
        "performance_claim_authorized": False,
        "prospective_83xxx_panel_opened": False,
        "resumable": False,
    }
    for key, expected in expected_contract.items():
        observed = contract.get(key)
        if key == "validation_seeds":
            observed = tuple(observed or ())
        _require_equal(f"source contract {key}", observed, expected)

    expected_checkpoint = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "latest_development_state",
        "completed_episodes": TOTAL_EPISODES,
        "training_contract_sha256": EXPECTED_EMBEDDED_CONTRACT_SHA256,
        "method_version": METHOD_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "resumable": False,
        "development_candidate_eligible": False,
        "deployment_checkpoint_eligible": False,
        "validation_authenticates_checkpoint_lambda": True,
        "validation_map_only": True,
        "same_temperatures_for_behavior_and_backup": True,
        "exact_safe_frontier_authoritative": True,
        "scalarized_reward_stored_in_replay": False,
    }
    for key, expected in expected_checkpoint.items():
        _require_equal(f"source checkpoint {key}", checkpoint.get(key), expected)

    expected_summary = {
        "status": "complete",
        "method_version": METHOD_VERSION,
        "completed_training_episodes": TOTAL_EPISODES,
        "completed_blocks": 10,
        "development_candidate_eligible": False,
        "deployment_checkpoint_eligible": False,
        "best_development_candidate": None,
        "prospective_83xxx_panel_opened": False,
    }
    for key, expected in expected_summary.items():
        _require_equal(f"source summary {key}", summary.get(key), expected)

    dual_lambda = _require_exact_float(
        "source runtime lambda", checkpoint.get("runtime_lambda"), EXPECTED_LAMBDA
    )
    for name in ("validated_block_lambda", "validated_policy_lambda"):
        _require_exact_float(f"source {name}", checkpoint.get(name), dual_lambda)
    _require_exact_float(
        "source dual state lambda",
        checkpoint.get("dual_state", {}).get("lambda_value"),
        dual_lambda,
    )

    map_schedule = schedule_for_episode(TOTAL_EPISODES, validation=True).to_dict()
    soft_schedule = schedule_for_episode(TOTAL_EPISODES, validation=False).to_dict()
    _require_equal("source MAP schedule", dict(checkpoint.get("schedule_state", {})), map_schedule)
    _require_equal(
        "source within temperatures",
        tuple(map_schedule["within_group_temperatures"]),
        EXPECTED_WITHIN_TEMPERATURES,
    )
    _require_exact_float(
        "source group temperature",
        map_schedule["group_temperature"],
        EXPECTED_GROUP_TEMPERATURE,
    )
    _require_equal(
        "soft/MAP within temperature identity",
        tuple(soft_schedule["within_group_temperatures"]),
        tuple(map_schedule["within_group_temperatures"]),
    )
    _require_exact_float(
        "soft/MAP group temperature identity",
        soft_schedule["group_temperature"],
        map_schedule["group_temperature"],
    )

    nested = checkpoint.get("agent_state")
    if not isinstance(nested, Mapping):
        raise PairedPolicyAuditError("source checkpoint has no nested agent checkpoint")
    for key, expected in {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "controller_architecture": CONTROLLER_ARCHITECTURE,
        "method_version": METHOD_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "same_temperatures_for_behavior_and_backup": True,
        "scalarized_reward_stored_in_replay": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "exact_safe_mask_authoritative": True,
    }.items():
        _require_equal(f"nested agent {key}", nested.get(key), expected)
    _require_equal("nested agent config", nested.get("config"), contract.get("agent_config"))
    _require_equal("nested agent MAP schedule", nested.get("schedule_state"), map_schedule)
    agent_state = nested.get("agent_state")
    if not isinstance(agent_state, Mapping):
        raise PairedPolicyAuditError("nested agent state is missing")
    _require_exact_float(
        "nested agent lambda", agent_state.get("dual_lambda"), dual_lambda
    )
    _require_equal("nested state MAP schedule", agent_state.get("schedule_state"), map_schedule)
    q_local = agent_state.get("Q_local")
    if not isinstance(q_local, Mapping):
        raise PairedPolicyAuditError("nested agent Q_local weights are missing")

    validation = checkpoint.get("validation_summary")
    if not isinstance(validation, Mapping):
        raise PairedPolicyAuditError("source checkpoint validation summary is missing")
    _require_equal("source validation episode", validation.get("checkpoint_episode"), 100)
    _require_equal("source validation MAP schedule", validation.get("schedule_state"), map_schedule)
    _require_exact_float(
        "source validation return", validation.get("mean_dense_return"), -109.33
    )
    _require_exact_float(
        "source validation MAE", validation.get("mean_absolute_error"), 32.0
    )
    _require_exact_float(
        "source validation rehandles/100",
        validation.get("physical_rehandles_per_100_required_deliveries"),
        0.0,
    )
    return dual_lambda, map_schedule, soft_schedule, q_local


def authenticate_source(checkpoint_path: Path) -> AuthenticatedSource:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise PairedPolicyAuditError(f"source checkpoint does not exist: {checkpoint_path}")
    contract_path = checkpoint_path.parent / "training-contract.json"
    summary_path = checkpoint_path.parent / "training-summary.json"
    for path in (contract_path, summary_path):
        if not path.is_file():
            raise PairedPolicyAuditError(f"required source artifact is missing: {path}")
    hashes = (
        _sha256_file(checkpoint_path),
        _sha256_file(contract_path),
        _sha256_file(summary_path),
    )
    expected_hashes = (
        EXPECTED_SOURCE_CHECKPOINT_SHA256,
        EXPECTED_SOURCE_CONTRACT_FILE_SHA256,
        EXPECTED_SOURCE_SUMMARY_FILE_SHA256,
    )
    for name, observed, expected in zip(
        ("checkpoint", "training contract file", "training summary file"),
        hashes,
        expected_hashes,
    ):
        _require_equal(f"source {name} SHA256", observed, expected)
    contract = _load_json(contract_path)
    summary = _load_json(summary_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise PairedPolicyAuditError(f"cannot load source checkpoint: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise PairedPolicyAuditError("source checkpoint must contain a mapping")
    dual, map_schedule, soft_schedule, q_local = _validate_loaded_source(
        checkpoint, contract, summary
    )
    return AuthenticatedSource(
        checkpoint_path=checkpoint_path,
        contract_path=contract_path,
        summary_path=summary_path,
        checkpoint=checkpoint,
        contract=contract,
        summary=summary,
        checkpoint_sha256=hashes[0],
        contract_file_sha256=hashes[1],
        summary_file_sha256=hashes[2],
        q_local_sha256=_tensor_state_sha256(q_local),
        q_local_state=q_local,
        dual_lambda=dual,
        map_schedule=map_schedule,
        soft_schedule=soft_schedule,
    )


def soft_policy_rng_seed(instance_index: int, replicate_index: int) -> int:
    if isinstance(instance_index, bool) or not isinstance(instance_index, int):
        raise PairedPolicyAuditError("instance_index must be an integer")
    if isinstance(replicate_index, bool) or not isinstance(replicate_index, int):
        raise PairedPolicyAuditError("replicate_index must be an integer")
    if not 0 <= instance_index < len(PANEL_SEEDS):
        raise PairedPolicyAuditError("instance_index lies outside the frozen panel")
    if not 0 <= replicate_index < SOFT_REPLICATES:
        raise PairedPolicyAuditError("replicate_index lies outside the frozen audit")
    return SOFT_RNG_BASE + SOFT_RNG_INSTANCE_STRIDE * instance_index + replicate_index


def build_audit_contract(source: AuthenticatedSource, *, device: str) -> dict:
    requested_device = torch.device(device)
    cuda_name = None
    if requested_device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(requested_device)
    contract = {
        "audit_protocol": AUDIT_PROTOCOL,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_only": True,
        "development_only": True,
        "cannot_create_or_select_checkpoint": True,
        "source_checkpoint": str(source.checkpoint_path),
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "source_contract_file_sha256": source.contract_file_sha256,
        "source_summary_file_sha256": source.summary_file_sha256,
        "source_training_contract_sha256": EXPECTED_EMBEDDED_CONTRACT_SHA256,
        "source_q_local_sha256": source.q_local_sha256,
        "source_checkpoint_role": "latest_development_state",
        "source_checkpoint_eligible": False,
        "source_completed_episodes": TOTAL_EPISODES,
        "source_method_version": METHOD_VERSION,
        "panel": {
            "episode_instance_seeds": PANEL_SEEDS,
            "reused_already_open_v2_1_development_validation_panel": True,
            "prospective_83xxx_panel_opened": False,
            "sealed_or_new_panel_opened": False,
            "pairing_key": "EpisodeInstance.instance_id_and_schedule_id_and_sha256",
        },
        "arms": {
            "map": {
                "runs_per_instance": 1,
                "controller_policy": "deterministic_nested_lagrangian_map",
                "policy_rng_used": False,
            },
            "induced_soft": {
                "runs_per_instance": SOFT_REPLICATES,
                "controller_policy": "induced_nested_regularized_lagrangian_sample",
                "rng_seed_formula": "910000 + 100 * instance_index + replicate_index",
                "replicate_indices": tuple(range(SOFT_REPLICATES)),
            },
        },
        "frozen_policy": {
            "dual_lambda": source.dual_lambda,
            "within_group_temperatures": tuple(
                source.map_schedule["within_group_temperatures"]
            ),
            "group_temperature": source.map_schedule["group_temperature"],
            "epsilon": 0.0,
            "latest_q_local_weights_only": True,
            "same_weights_lambda_and_temperatures_across_arms": True,
            "option_evaluation_mode": True,
            "evaluation_learning": False,
            "fresh_agent_clone_per_row": True,
            "fresh_certificate_cache_per_row": True,
            "replay_forbidden": True,
            "optimizer_step_forbidden": True,
            "target_update_forbidden": True,
        },
        "aggregation": {
            "soft_within_instance_first": True,
            "equal_weight_episode_instances": True,
            "soft_rows_are_not_independent_workload_samples": True,
            "conditional_mc_bundles": (
                "replicate_index crossed over three disjoint controller RNG seeds"
            ),
            "conditional_mc_replicates": SOFT_REPLICATES,
            "conditional_mc_uncertainty_only": True,
            "environment_generalization_interval": False,
            "two_sided_t_975_df31": TWO_SIDED_T_975_DF31,
            "one_sided_budget_t_95_df31": ONE_SIDED_T_95_DF31,
        },
        "endpoints": {
            "primary": "dense_return",
            "key_secondary": "mean_absolute_error",
            "additional_operational": ("steps",),
            "mechanism": (
                "physical_rehandles_per_100_required_deliveries",
                "pooled_action_shares",
            ),
        },
        "budget_per_100_required_deliveries": 20.0,
        "complete_case_filtering_used": False,
        "device": str(requested_device),
        "compute_implementation": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "policy_sampler": "random.Random.choices",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": cuda_name,
            "requested_device": str(requested_device),
            "map_reproduction_is_compute_authentication_gate": True,
        },
    }
    contract["audit_contract_sha256"] = _sha256_bytes(_canonical_json_bytes(contract))
    return contract


def _make_environment(contract: Mapping):
    from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec

    environment = dict(contract["environment"])
    return ObjectiveAuditSmallRoomsEnv(
        timing_objective=TimingObjectiveSpec.from_dict(
            contract["dense_objective_spec"]
        ),
        grid_rows=int(environment["grid_rows"]),
        grid_cols=int(environment["grid_cols"]),
        number_blocks=int(environment["number_blocks"]),
        choose_storage=False,
        arrival_rate=float(environment["arrival_rate"]),
        proc_mean=float(environment["proc_mean"]),
    )


def materialize_instances(
    source: AuthenticatedSource, destination: Path
) -> tuple[dict[int, EpisodeInstance], dict]:
    destination.mkdir(parents=True, exist_ok=False)
    environment = _make_environment(source.contract)
    instances: dict[int, EpisodeInstance] = {}
    records = {}
    for index, seed in enumerate(PANEL_SEEDS):
        instance = environment.sample_episode_instance(seed)
        instance.validate_for(environment)
        if instance.seed != seed:
            raise PairedPolicyAuditError("sampled EpisodeInstance seed mismatch")
        text = instance.to_json() + "\n"
        path = destination / f"seed-{seed}.json"
        path.write_text(text, encoding="utf-8")
        restored = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        if restored != instance:
            raise PairedPolicyAuditError("EpisodeInstance JSON round trip changed data")
        repeated = environment.sample_episode_instance(seed)
        if repeated != instance:
            raise PairedPolicyAuditError("EpisodeInstance sampler is not deterministic")
        digest = _sha256_file(path)
        instances[seed] = restored
        records[str(seed)] = {
            "instance_index": index,
            "instance_seed": seed,
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "number_blocks": instance.number_blocks,
            "relative_path": f"instances/{path.name}",
            "serialized_sha256": digest,
        }
    manifest = {
        "schema_version": 1,
        "audit_protocol": AUDIT_PROTOCOL,
        "source_training_contract_sha256": EXPECTED_EMBEDDED_CONTRACT_SHA256,
        "episode_instance_seeds": PANEL_SEEDS,
        "reused_development_panel": True,
        "prospective_83xxx_panel_opened": False,
        "instances": records,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    return instances, manifest


class _FrozenRealizationAgent(ConstrainedV21HierarchyAgent):
    """Weight-only evaluation clone whose learning interface is inaccessible."""

    def __init__(self, *args, audit_policy: str, **kwargs) -> None:
        if audit_policy not in POLICIES:
            raise PairedPolicyAuditError("unknown frozen audit policy")
        self.audit_policy = audit_policy
        super().__init__(*args, **kwargs)

    def select(self, snapshot, *, training: bool = False, epsilon=None):
        if training:
            raise RuntimeError("frozen audit caller cannot request training execution")
        # The internal flag selects the high-level realization only.  The
        # enclosing environment loop remains evaluation-only.
        induced_sample = self.audit_policy == "induced_soft"
        return super().select(
            snapshot, training=induced_sample, epsilon=0.0 if epsilon is None else epsilon
        )

    def remember(self, *args, **kwargs):  # pragma: no cover - must stay unreachable
        raise RuntimeError("frozen paired audit forbids replay mutation")

    def learn(self, *args, **kwargs):  # pragma: no cover - must stay unreachable
        raise RuntimeError("frozen paired audit forbids learning")


def _learning_state_signature(agent: ConstrainedV21HierarchyAgent) -> dict:
    return {
        "q_local_sha256": _tensor_state_sha256(agent.Q_local.state_dict()),
        "q_target_sha256": _tensor_state_sha256(agent.Q_target.state_dict()),
        "optimizer_state_entries": len(agent.optimizer.state),
        "replay_size": len(agent.replay),
        "transition_count": int(agent.transition_count),
        "gradient_steps": int(agent.gradient_steps),
        "target_updates": int(agent.target_updates),
        "dual_lambda": float(agent.dual_lambda),
        "schedule_state": dict(agent.schedule_state),
        "q_local_training_mode": bool(agent.Q_local.training),
        "q_target_training_mode": bool(agent.Q_target.training),
        "any_parameter_requires_grad": any(
            parameter.requires_grad for parameter in agent.Q_local.parameters()
        ),
    }


class FrozenPairedPolicyRuntime(ConstrainedV21DevelopmentRuntime):
    """V2.1 executor with decoupled policy sampling and evaluation semantics."""

    def __init__(
        self, *, source: AuthenticatedSource, device: str = "cpu"
    ) -> None:
        runtime_contract = deepcopy(dict(source.contract))
        runtime_contract["device"] = str(device)
        args = argparse.Namespace(device=str(device))
        super().__init__(args=args, contract=runtime_contract)
        self.source = source
        self._audit_policy = "map"
        self._audit_policy_rng_seed = PANEL_SEEDS[0]
        self._frozen_instance: Optional[EpisodeInstance] = None
        self._last_clone_signature_before: Optional[dict] = None
        self._last_clone: Optional[_FrozenRealizationAgent] = None
        self._macro_trace: list[dict] = []
        self._source_checkpoint_stat = self.source.checkpoint_path.stat()
        # Authenticate an immutable source template.  It is never selected by
        # the environment loop; every row gets a fresh clone below.
        self.agent = self._new_frozen_agent(
            policy="map", policy_rng_seed=int(source.contract["model_seed"])
        )
        self.set_schedule_state(source.map_schedule)
        self.set_dual_lambda(source.dual_lambda)

    def _new_frozen_agent(
        self, *, policy: str, policy_rng_seed: int
    ) -> _FrozenRealizationAgent:
        schedule = (
            self.source.map_schedule
            if policy == "map"
            else self.source.soft_schedule
        )
        # The nuisance seed belongs exclusively to action sampling.  Network
        # construction therefore always uses the authenticated model seed;
        # after the exact weights are loaded, only the agent-local Python RNG
        # is replaced by the policy realization stream.
        agent = _FrozenRealizationAgent(
            config=self.config,
            seed=int(self.source.contract["model_seed"]),
            device=self.device,
            epsilon=0.0,
            dual_lambda=self.source.dual_lambda,
            audit_policy=policy,
        )
        agent.Q_local.load_state_dict(self.source.q_local_state)
        agent.Q_target.load_state_dict(self.source.q_local_state)
        agent.Q_local.eval()
        agent.Q_target.eval()
        agent.Q_local.requires_grad_(False)
        agent.Q_target.requires_grad_(False)
        agent.rng = random.Random(int(policy_rng_seed))
        agent.audit_policy_rng_seed = int(policy_rng_seed)
        agent.set_schedule_state(schedule)
        if _tensor_state_sha256(agent.Q_local.state_dict()) != self.source.q_local_sha256:
            raise PairedPolicyAuditError("fresh clone did not receive exact source weights")
        return agent

    def _evaluation_agent(self, *, seed: int) -> _FrozenRealizationAgent:
        del seed  # controller RNG is explicitly separated from instance seed
        clone = self._new_frozen_agent(
            policy=self._audit_policy,
            policy_rng_seed=self._audit_policy_rng_seed,
        )
        self._last_clone = clone
        self._last_clone_signature_before = _learning_state_signature(clone)
        return clone

    def _execute_macro(self, candidate, *, remaining_steps: int, evaluation: bool):
        if evaluation is not True:
            raise RuntimeError("paired policy audit requires option evaluation mode")
        result = super()._execute_macro(
            candidate, remaining_steps=remaining_steps, evaluation=True
        )
        self._macro_trace.append(
            {
                "candidate_key": candidate.key,
                "action_type": candidate.action_type.value,
                "duration": int(result.duration),
                "physical_rehandles": int(result.relocations),
                "raw_return": float(result.raw_return),
            }
        )
        return result

    def run_frozen_instance(
        self,
        *,
        instance: EpisodeInstance,
        instance_index: int,
        policy: str,
        policy_rng_index: Optional[int],
        policy_rng_seed: int,
    ) -> dict:
        if policy not in POLICIES:
            raise PairedPolicyAuditError("unknown policy realization")
        if (
            isinstance(instance_index, bool)
            or not isinstance(instance_index, int)
            or not 0 <= instance_index < len(PANEL_SEEDS)
        ):
            raise PairedPolicyAuditError("instance index lies outside frozen panel")
        if instance.seed != PANEL_SEEDS[instance_index]:
            raise PairedPolicyAuditError("frozen instance/index mismatch")
        if policy == "map" and policy_rng_index is not None:
            raise PairedPolicyAuditError("MAP realization cannot have an RNG index")
        if policy == "induced_soft" and (
            isinstance(policy_rng_index, bool)
            or not isinstance(policy_rng_index, int)
            or not 0 <= policy_rng_index < SOFT_REPLICATES
        ):
            raise PairedPolicyAuditError("soft realization RNG index is invalid")
        expected_rng = (
            instance.seed
            if policy == "map"
            else soft_policy_rng_seed(instance_index, int(policy_rng_index))
        )
        if int(policy_rng_seed) != int(expected_rng):
            raise PairedPolicyAuditError("controller RNG seed violates frozen namespace")
        schedule = (
            self.source.map_schedule
            if policy == "map"
            else self.source.soft_schedule
        )
        self.set_schedule_state(schedule)
        self.set_dual_lambda(self.source.dual_lambda)
        self._audit_policy = policy
        self._audit_policy_rng_seed = int(policy_rng_seed)
        self._frozen_instance = instance
        self._macro_trace = []
        source_before = _learning_state_signature(self.agent)
        source_file_before = self.source.checkpoint_path.stat()
        original_sampler = self.env.sample_episode_instance

        def frozen_sampler(seed=None):
            if int(seed) != int(instance.seed):
                raise PairedPolicyAuditError("executor requested a different instance seed")
            return instance

        self.env.sample_episode_instance = frozen_sampler
        try:
            with torch.inference_mode():
                raw = v2.ConstrainedV2SmokeRuntime.run_episode(
                    self,
                    instance_seed=int(instance.seed),
                    training=False,
                    max_steps=int(self.max_steps),
                )
        finally:
            self.env.sample_episode_instance = original_sampler
            self._frozen_instance = None
        if self._last_clone is None or self._last_clone_signature_before is None:
            raise RuntimeError("frozen evaluator did not retain its row clone")
        clone_after = _learning_state_signature(self._last_clone)
        source_after = _learning_state_signature(self.agent)
        source_file_after = self.source.checkpoint_path.stat()
        clone_before = self._last_clone_signature_before
        immutable_fields = (
            "q_local_sha256",
            "q_target_sha256",
            "optimizer_state_entries",
            "replay_size",
            "transition_count",
            "gradient_steps",
            "target_updates",
            "dual_lambda",
            "schedule_state",
            "any_parameter_requires_grad",
        )
        clone_unchanged = all(
            clone_before[field] == clone_after[field] for field in immutable_fields
        )
        source_unchanged = source_before == source_after
        # Exact content hashes are checked once before and once after the
        # complete transaction.  The per-row stat identity avoids rereading a
        # large checkpoint 198 times while still failing immediately on an
        # incidental replacement or write during any realization.
        checkpoint_unchanged = (
            source_file_before
            == source_file_after
            == self._source_checkpoint_stat
        )
        selected = dict(raw.get("selected_action_counts", {}))
        macros = int(raw.get("macro_decisions", 0))
        holds = int(selected.get("defer", 0))
        reconfigures = int(selected.get("reconfigure", 0))
        required = int(raw["required_deliveries"])
        physical = int(raw["physical_rehandles"])
        diagnostics = self._last_clone.policy_diagnostic_state()
        hold_trace = [item for item in self._macro_trace if item["action_type"] == "defer"]
        row = {
            "audit_protocol": AUDIT_PROTOCOL,
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "policy": policy,
            "policy_realization": (
                "deterministic_nested_lagrangian_map"
                if policy == "map"
                else "induced_nested_regularized_lagrangian_sample"
            ),
            "instance_index": int(instance_index),
            "instance_seed": int(instance.seed),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "policy_rng_index": (
                None if policy_rng_index is None else int(policy_rng_index)
            ),
            "policy_rng_seed": int(policy_rng_seed),
            "policy_rng_used": policy == "induced_soft",
            "policy_rng_scope": "action_sampling_only",
            "source_checkpoint_sha256": self.source.checkpoint_sha256,
            "source_checkpoint_role": "latest_development_state",
            "source_q_local_sha256": self.source.q_local_sha256,
            "source_training_contract_sha256": EXPECTED_EMBEDDED_CONTRACT_SHA256,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "schedule_episode_number": int(schedule["episode_number"]),
            "schedule_block_number": int(schedule["block_number"]),
            "schedule_phase": str(schedule["phase"]),
            "schedule_policy_mode": str(schedule["policy_mode"]),
            "controller_select_internal_training_flag": policy == "induced_soft",
            "stochastic_selection_only": policy == "induced_soft",
            "dual_lambda": self.source.dual_lambda,
            "within_group_temperatures": tuple(
                schedule["within_group_temperatures"]
            ),
            "group_temperature": float(schedule["group_temperature"]),
            "epsilon": 0.0,
            "strict_method_success": bool(raw["strict_method_success"]),
            "success": bool(raw["success"]),
            "completion_rate": float(raw["completion_rate"]),
            "delivery_count": int(raw["delivery_count"]),
            "required_deliveries": required,
            "dense_return": float(raw["dense_return"]),
            "mean_absolute_error": raw["mean_absolute_error"],
            "delivery_deviations": tuple(float(x) for x in raw["delivery_deviations"]),
            "steps": int(raw["steps"]),
            "macro_decisions": macros,
            "physical_rehandles": physical,
            "physical_rehandles_per_100_required_deliveries": (
                100.0 * physical / required
            ),
            "selected_action_counts": selected,
            "hold_decisions": holds,
            "hold_decision_share": 0.0 if macros == 0 else holds / macros,
            "hold_primitive_steps": sum(item["duration"] for item in hold_trace),
            "hold_horizon_counts": {
                str(duration): sum(item["duration"] == duration for item in hold_trace)
                for duration in sorted({item["duration"] for item in hold_trace})
            },
            "hold_outcome_counts": dict(raw.get("hold_outcome_counts", {})),
            "reconfigure_decisions": reconfigures,
            "reconfigure_decision_share": (
                0.0 if macros == 0 else reconfigures / macros
            ),
            "method_failure_reason": raw.get("method_failure_reason"),
            "illegal_drops": int(raw.get("illegal_drops", 0)),
            "fallbacks": int(raw.get("fallbacks", 0)),
            "witness_mismatches": int(raw.get("witness_mismatches", 0)),
            "all_selected_candidates_exact_safe": bool(
                raw.get("all_selected_candidates_exact_safe", False)
            ),
            "policy_diagnostic_distribution": V2_1_DIAGNOSTIC_DISTRIBUTION,
            "policy_diagnostic_decisions": int(diagnostics["count"]),
            "mean_outer_policy_entropy": diagnostics["mean_outer_entropy"],
            "mean_outer_map_probability": diagnostics["mean_outer_map_probability"],
            "mean_selected_within_policy_entropy": diagnostics[
                "mean_selected_within_entropy"
            ],
            "mean_selected_within_map_probability": diagnostics[
                "mean_selected_within_map_probability"
            ],
            "option_evaluation_mode": True,
            "evaluation_learning": False,
            "q_local_training_mode": bool(self._last_clone.Q_local.training),
            "loss_updates": int(raw.get("loss_updates", 0)),
            "replay_size_before": clone_before["replay_size"],
            "replay_size_after": clone_after["replay_size"],
            "gradient_steps_before": clone_before["gradient_steps"],
            "gradient_steps_after": clone_after["gradient_steps"],
            "target_updates_before": clone_before["target_updates"],
            "target_updates_after": clone_after["target_updates"],
            "optimizer_state_entries_before": clone_before[
                "optimizer_state_entries"
            ],
            "optimizer_state_entries_after": clone_after[
                "optimizer_state_entries"
            ],
            "clone_learning_state_unchanged": clone_unchanged,
            "source_agent_unchanged": source_unchanged,
            "training_agent_unchanged": source_unchanged,
            "source_checkpoint_file_unchanged": checkpoint_unchanged,
            "policy_frozen": bool(
                clone_unchanged and source_unchanged and checkpoint_unchanged
            ),
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "fresh_certificate_cache": True,
            "exact_safe_frontier_authoritative": True,
        }
        return row


def _row_safety_issues(row: Mapping) -> tuple[str, ...]:
    issues = []
    boolean_requirements = {
        "strict_method_success": True,
        "success": True,
        "all_selected_candidates_exact_safe": True,
        "option_evaluation_mode": True,
        "evaluation_learning": False,
        "q_local_training_mode": False,
        "clone_learning_state_unchanged": True,
        "source_agent_unchanged": True,
        "source_checkpoint_file_unchanged": True,
        "policy_frozen": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "fresh_certificate_cache": True,
        "exact_safe_frontier_authoritative": True,
    }
    for key, expected in boolean_requirements.items():
        if row.get(key) is not expected:
            issues.append(f"{key}={row.get(key)!r}")
    for key in ("illegal_drops", "fallbacks", "witness_mismatches", "loss_updates"):
        if row.get(key) != 0:
            issues.append(f"{key}={row.get(key)!r}")
    for before, after in (
        ("replay_size_before", "replay_size_after"),
        ("gradient_steps_before", "gradient_steps_after"),
        ("target_updates_before", "target_updates_after"),
        ("optimizer_state_entries_before", "optimizer_state_entries_after"),
    ):
        if row.get(before) != 0 or row.get(after) != 0:
            issues.append(f"{before}/{after} changed or nonzero")
    if row.get("method_failure_reason") is not None:
        issues.append(f"method_failure_reason={row.get('method_failure_reason')!r}")
    if row.get("delivery_count") != row.get("required_deliveries"):
        issues.append("incomplete required deliveries")
    for key in PRIMARY_METRICS:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            issues.append(f"nonfinite_or_missing:{key}")
    return tuple(issues)


def _validate_paired_row_grid(
    rows: Sequence[Mapping],
    *,
    source_audit: Mapping,
    instance_manifest: Mapping,
) -> None:
    """Authenticate every row's source, workload, and nuisance-stream identity."""

    required_source = {
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "source_training_contract_sha256": EXPECTED_EMBEDDED_CONTRACT_SHA256,
        "dual_lambda": EXPECTED_LAMBDA,
        "within_group_temperatures": EXPECTED_WITHIN_TEMPERATURES,
        "group_temperature": EXPECTED_GROUP_TEMPERATURE,
    }
    for key, frozen in required_source.items():
        observed = source_audit.get(key)
        if key == "within_group_temperatures":
            observed = tuple(observed or ())
        if observed != frozen:
            raise PairedPolicyAuditError(
                f"source audit {key} mismatch: {observed!r} != {frozen!r}"
            )
    q_sha = source_audit.get("source_q_local_sha256")
    if not isinstance(q_sha, str) or len(q_sha) != 64:
        raise PairedPolicyAuditError("source audit Q-local SHA256 is invalid")
    if tuple(instance_manifest.get("episode_instance_seeds", ())) != PANEL_SEEDS:
        raise PairedPolicyAuditError("instance manifest panel differs from frozen panel")
    if instance_manifest.get("reused_development_panel") is not True:
        raise PairedPolicyAuditError("instance manifest is not the reused development panel")
    if instance_manifest.get("prospective_83xxx_panel_opened") is not False:
        raise PairedPolicyAuditError("instance manifest opened the prospective panel")
    records = instance_manifest.get("instances")
    if not isinstance(records, Mapping) or set(records) != {
        str(seed) for seed in PANEL_SEEDS
    }:
        raise PairedPolicyAuditError("instance manifest records are incomplete")

    map_schedule = schedule_for_episode(TOTAL_EPISODES, validation=True).to_dict()
    soft_schedule = schedule_for_episode(TOTAL_EPISODES, validation=False).to_dict()
    seen = set()
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PairedPolicyAuditError(f"row {row_number} is not a mapping")
        if row.get("audit_protocol") != AUDIT_PROTOCOL:
            raise PairedPolicyAuditError(f"row {row_number} audit protocol mismatch")
        if row.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            raise PairedPolicyAuditError(f"row {row_number} schema mismatch")
        policy = row.get("policy")
        if policy not in POLICIES:
            raise PairedPolicyAuditError(f"row {row_number} policy is invalid")
        instance_index = row.get("instance_index")
        if (
            isinstance(instance_index, bool)
            or not isinstance(instance_index, int)
            or not 0 <= instance_index < len(PANEL_SEEDS)
        ):
            raise PairedPolicyAuditError(f"row {row_number} instance index is invalid")
        seed = PANEL_SEEDS[instance_index]
        if row.get("instance_seed") != seed:
            raise PairedPolicyAuditError(f"row {row_number} instance seed/index mismatch")
        record = records[str(seed)]
        if not isinstance(record, Mapping):
            raise PairedPolicyAuditError(f"manifest record {seed} is invalid")
        for row_key, manifest_key in (
            ("instance_id", "instance_id"),
            ("schedule_id", "schedule_id"),
            ("episode_instance_serialized_sha256", "serialized_sha256"),
        ):
            if row.get(row_key) != record.get(manifest_key):
                raise PairedPolicyAuditError(
                    f"row {row_number} {row_key} differs from manifest"
                )
        if record.get("instance_index") != instance_index:
            raise PairedPolicyAuditError(f"manifest record {seed} index mismatch")

        if policy == "map":
            expected_index = None
            expected_rng = seed
            expected_used = False
            expected_realization = "deterministic_nested_lagrangian_map"
            schedule = map_schedule
        else:
            expected_index = row.get("policy_rng_index")
            if (
                isinstance(expected_index, bool)
                or not isinstance(expected_index, int)
                or not 0 <= expected_index < SOFT_REPLICATES
            ):
                raise PairedPolicyAuditError(
                    f"row {row_number} soft RNG index is invalid"
                )
            expected_rng = soft_policy_rng_seed(instance_index, expected_index)
            expected_used = True
            expected_realization = "induced_nested_regularized_lagrangian_sample"
            schedule = soft_schedule
        if row.get("policy_rng_index") != expected_index:
            raise PairedPolicyAuditError(f"row {row_number} MAP RNG index is not null")
        if row.get("policy_rng_seed") != expected_rng:
            raise PairedPolicyAuditError(
                f"row {row_number} RNG seed does not match its index"
            )
        if row.get("policy_rng_used") is not expected_used:
            raise PairedPolicyAuditError(f"row {row_number} RNG-use flag mismatch")
        if row.get("policy_rng_scope") != "action_sampling_only":
            raise PairedPolicyAuditError(f"row {row_number} RNG scope mismatch")
        if row.get("policy_realization") != expected_realization:
            raise PairedPolicyAuditError(f"row {row_number} realization mismatch")
        key = (policy, seed, expected_index)
        if key in seen:
            raise PairedPolicyAuditError(f"duplicate paired row key {key!r}")
        seen.add(key)

        expected_identity = {
            "source_checkpoint_sha256": source_audit["source_checkpoint_sha256"],
            "source_q_local_sha256": q_sha,
            "source_training_contract_sha256": source_audit[
                "source_training_contract_sha256"
            ],
            "source_checkpoint_role": "latest_development_state",
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "schedule_episode_number": TOTAL_EPISODES,
            "schedule_block_number": int(schedule["block_number"]),
            "schedule_phase": str(schedule["phase"]),
            "schedule_policy_mode": str(schedule["policy_mode"]),
            "controller_select_internal_training_flag": policy == "induced_soft",
            "stochastic_selection_only": policy == "induced_soft",
            "dual_lambda": EXPECTED_LAMBDA,
            "group_temperature": EXPECTED_GROUP_TEMPERATURE,
            "epsilon": 0.0,
            "policy_diagnostic_distribution": V2_1_DIAGNOSTIC_DISTRIBUTION,
            "option_evaluation_mode": True,
            "evaluation_learning": False,
            "q_local_training_mode": False,
            "exact_safe_frontier_authoritative": True,
        }
        for name, expected in expected_identity.items():
            if row.get(name) != expected:
                raise PairedPolicyAuditError(
                    f"row {row_number} {name} mismatch: "
                    f"{row.get(name)!r} != {expected!r}"
                )
        if tuple(row.get("within_group_temperatures", ())) != tuple(
            schedule["within_group_temperatures"]
        ):
            raise PairedPolicyAuditError(
                f"row {row_number} within-group temperatures mismatch"
            )
        if row.get("training_agent_unchanged") is not True:
            raise PairedPolicyAuditError(f"row {row_number} training source mutated")
        macros = row.get("macro_decisions")
        diagnostics = row.get("policy_diagnostic_decisions")
        if macros != diagnostics:
            raise PairedPolicyAuditError(
                f"row {row_number} diagnostic/decision count mismatch"
            )
        selected = row.get("selected_action_counts")
        if not isinstance(selected, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in selected.values()
        ):
            raise PairedPolicyAuditError(f"row {row_number} action counts invalid")
        if sum(selected.values()) != macros:
            raise PairedPolicyAuditError(f"row {row_number} action counts do not sum")
        if row.get("hold_decisions") != int(selected.get("defer", 0)):
            raise PairedPolicyAuditError(f"row {row_number} Hold count mismatch")
        if row.get("reconfigure_decisions") != int(
            selected.get("reconfigure", 0)
        ):
            raise PairedPolicyAuditError(
                f"row {row_number} reconfigure count mismatch"
            )
        required = row.get("required_deliveries")
        physical = row.get("physical_rehandles")
        if (
            isinstance(required, bool)
            or not isinstance(required, int)
            or required <= 0
            or isinstance(physical, bool)
            or not isinstance(physical, int)
            or physical < 0
        ):
            raise PairedPolicyAuditError(f"row {row_number} workload cost invalid")
        expected_rate = 100.0 * physical / required
        if not math.isclose(
            float(row.get("physical_rehandles_per_100_required_deliveries")),
            expected_rate,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise PairedPolicyAuditError(f"row {row_number} rehandle rate mismatch")


def _sample_summary(values: Sequence[float], *, t_value: float) -> dict:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise PairedPolicyAuditError("cannot summarize empty/nonfinite values")
    mean = float(fmean(values))
    sample_sd = 0.0 if len(values) == 1 else float(stdev(values))
    se = sample_sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_sd,
        "standard_error": se,
        "conditional_mc_95_ci": (mean - t_value * se, mean + t_value * se),
    }


def _map_reproduction_gate(map_rows: Sequence[Mapping]) -> tuple[bool, dict]:
    actions = {
        action: sum(int(row["selected_action_counts"].get(action, 0)) for row in map_rows)
        for action in ("accept", "deliver", "defer", "reconfigure")
    }
    observed = {
        "mean_dense_return": fmean(float(row["dense_return"]) for row in map_rows),
        "mean_absolute_error": fmean(
            float(row["mean_absolute_error"]) for row in map_rows
        ),
        "physical_rehandles_per_100_required_deliveries": (
            100.0
            * sum(int(row["physical_rehandles"]) for row in map_rows)
            / sum(int(row["required_deliveries"]) for row in map_rows)
        ),
        "selected_action_counts": actions,
        "strict_count": sum(bool(row["strict_method_success"]) for row in map_rows),
    }
    expected = {
        "mean_dense_return": -109.33,
        "mean_absolute_error": 32.0,
        "physical_rehandles_per_100_required_deliveries": 0.0,
        "selected_action_counts": {
            "accept": 24,
            "deliver": 24,
            "defer": 37,
            "reconfigure": 0,
        },
        "strict_count": 3,
    }
    numeric_ok = all(
        math.isclose(
            float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=1.0e-9
        )
        for key in (
            "mean_dense_return",
            "mean_absolute_error",
            "physical_rehandles_per_100_required_deliveries",
        )
    )
    return bool(
        numeric_ok
        and observed["selected_action_counts"] == expected["selected_action_counts"]
        and observed["strict_count"] == expected["strict_count"]
    ), {"observed": observed, "expected": expected, "absolute_tolerance": 1.0e-9}


def summarize_paired_rows(rows: Sequence[Mapping]) -> dict:
    rows = tuple(dict(row) for row in rows)
    expected_count = len(PANEL_SEEDS) * (SOFT_REPLICATES + 1)
    if len(rows) != expected_count:
        raise PairedPolicyAuditError(
            f"paired audit requires {expected_count} rows, found {len(rows)}"
        )
    keys = set()
    safety_issues = []
    for index, row in enumerate(rows):
        key = (row.get("policy"), row.get("instance_seed"), row.get("policy_rng_index"))
        if key in keys:
            raise PairedPolicyAuditError(f"duplicate paired row key {key!r}")
        keys.add(key)
        issues = _row_safety_issues(row)
        if issues:
            safety_issues.append({"row_index": index, "key": key, "issues": issues})
    map_rows = [row for row in rows if row.get("policy") == "map"]
    soft_rows = [row for row in rows if row.get("policy") == "induced_soft"]
    if len(map_rows) != len(PANEL_SEEDS) or len(soft_rows) != len(PANEL_SEEDS) * SOFT_REPLICATES:
        raise PairedPolicyAuditError("paired audit arm counts are inconsistent")
    for instance_index, seed in enumerate(PANEL_SEEDS):
        maps = [row for row in map_rows if row["instance_seed"] == seed]
        soft = [row for row in soft_rows if row["instance_seed"] == seed]
        if len(maps) != 1 or len(soft) != SOFT_REPLICATES:
            raise PairedPolicyAuditError(f"instance {seed} has incomplete paired arms")
        expected_rngs = {
            soft_policy_rng_seed(instance_index, replicate)
            for replicate in range(SOFT_REPLICATES)
        }
        if {int(row["policy_rng_seed"]) for row in soft} != expected_rngs:
            raise PairedPolicyAuditError(f"instance {seed} controller RNG panel mismatch")

    map_reproduced, reproduction = _map_reproduction_gate(map_rows)
    strict_integrity = not safety_issues and map_reproduced
    if not strict_integrity:
        return {
            "audit_protocol": AUDIT_PROTOCOL,
            "authentic_performance_estimands": False,
            "strict_integrity_gate": False,
            "map_source_reproduction_gate": map_reproduced,
            "map_source_reproduction": reproduction,
            "safety_issues": safety_issues,
            "complete_case_filtering_used": False,
            "per_instance": None,
            "aggregate": None,
            "conditional_mc_uncertainty_only": True,
            "deployment_checkpoint_eligible": False,
            "checkpoint_selection_performed": False,
        }

    per_instance = {}
    map_by_seed = {int(row["instance_seed"]): row for row in map_rows}
    soft_by_key = {
        (int(row["instance_seed"]), int(row["policy_rng_index"])): row
        for row in soft_rows
    }
    for seed in PANEL_SEEDS:
        map_row = map_by_seed[seed]
        instance_soft = [soft_by_key[(seed, replicate)] for replicate in range(SOFT_REPLICATES)]
        metrics = {}
        for metric in PRIMARY_METRICS:
            map_value = float(map_row[metric])
            values = [float(row[metric]) for row in instance_soft]
            soft_summary = _sample_summary(values, t_value=TWO_SIDED_T_975_DF31)
            metrics[metric] = {
                "map": map_value,
                "soft": soft_summary,
                "soft_minus_map": float(soft_summary["mean"] - map_value),
            }
        per_instance[str(seed)] = {
            "instance_id": map_row["instance_id"],
            "schedule_id": map_row["schedule_id"],
            "metrics": metrics,
        }

    bundle_metrics = {}
    for metric in PRIMARY_METRICS:
        bundle_deltas = []
        bundle_soft = []
        for replicate in range(SOFT_REPLICATES):
            soft_values = [
                float(soft_by_key[(seed, replicate)][metric]) for seed in PANEL_SEEDS
            ]
            map_values = [float(map_by_seed[seed][metric]) for seed in PANEL_SEEDS]
            bundle_soft.append(float(fmean(soft_values)))
            bundle_deltas.append(
                float(fmean(soft - mapped for soft, mapped in zip(soft_values, map_values)))
            )
        soft_summary = _sample_summary(bundle_soft, t_value=TWO_SIDED_T_975_DF31)
        delta_summary = _sample_summary(bundle_deltas, t_value=TWO_SIDED_T_975_DF31)
        bundle_metrics[metric] = {
            "map_equal_instance_mean": float(
                fmean(float(map_by_seed[seed][metric]) for seed in PANEL_SEEDS)
            ),
            "soft_equal_instance_mean": soft_summary,
            "soft_minus_map_equal_instance_mean": delta_summary,
            "bundle_values": tuple(bundle_deltas),
        }

    rehandle_bundles = []
    for replicate in range(SOFT_REPLICATES):
        rehandle_bundles.append(
            float(
                fmean(
                    float(
                        soft_by_key[(seed, replicate)][
                            "physical_rehandles_per_100_required_deliveries"
                        ]
                    )
                    for seed in PANEL_SEEDS
                )
            )
        )
    cost_mean = float(fmean(rehandle_bundles))
    cost_sd = float(stdev(rehandle_bundles))
    cost_se = cost_sd / math.sqrt(SOFT_REPLICATES)
    cost_ucb = cost_mean + ONE_SIDED_T_95_DF31 * cost_se
    map_cost = float(
        100.0
        * sum(int(row["physical_rehandles"]) for row in map_rows)
        / sum(int(row["required_deliveries"]) for row in map_rows)
    )
    budget = {
        "budget_per_100_required_deliveries": 20.0,
        "map_aggregate_rate": map_cost,
        "map_budget_gate": map_cost <= 20.0,
        "soft_equal_instance_point_rate": cost_mean,
        "soft_point_budget_gate": cost_mean <= 20.0,
        "soft_conditional_mc_standard_error": cost_se,
        "soft_one_sided_95_ucb": cost_ucb,
        "soft_ucb_budget_gate": cost_ucb <= 20.0,
        "one_sided_t_95_df31": ONE_SIDED_T_95_DF31,
    }
    return {
        "audit_protocol": AUDIT_PROTOCOL,
        "authentic_performance_estimands": True,
        "strict_integrity_gate": True,
        "map_source_reproduction_gate": True,
        "map_source_reproduction": reproduction,
        "safety_issues": [],
        "complete_case_filtering_used": False,
        "row_count": len(rows),
        "map_row_count": len(map_rows),
        "soft_row_count": len(soft_rows),
        "per_instance": per_instance,
        "aggregate": bundle_metrics,
        "budget": budget,
        "overall_budget_feasible": bool(
            budget["map_budget_gate"] and budget["soft_ucb_budget_gate"]
        ),
        "conditional_mc_uncertainty_only": True,
        "fixed_episode_instance_count": len(PANEL_SEEDS),
        "environment_generalization_interval": False,
        "deployment_checkpoint_eligible": False,
        "checkpoint_selection_performed": False,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _transactional_directory(output_dir: Path, writer: Callable[[Path], None]) -> None:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise PairedPolicyAuditError(
            "paired audit refuses to overwrite an existing output path"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        writer(staging)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def execute_audit(
    *,
    source: AuthenticatedSource,
    output_dir: Path,
    device: str,
) -> dict:
    contract = build_audit_contract(source, device=device)
    source_hashes_before = {
        "checkpoint": _sha256_file(source.checkpoint_path),
        "contract": _sha256_file(source.contract_path),
        "summary": _sha256_file(source.summary_path),
    }
    returned_summary = {}

    def write(staging: Path) -> None:
        nonlocal returned_summary
        _write_json(staging / "audit-contract.json", contract)
        instances, manifest = materialize_instances(source, staging / "instances")
        _write_json(staging / "instance-manifest.json", manifest)
        runtime = FrozenPairedPolicyRuntime(source=source, device=device)
        rows = []
        total_rows = len(PANEL_SEEDS) * (SOFT_REPLICATES + 1)

        def report_progress(row: Mapping) -> None:
            print(
                f"[{len(rows):3d}/{total_rows}] "
                f"{row['policy']:>12s} seed={row['instance_seed']} "
                f"rng={row['policy_rng_seed']} "
                f"R={row['dense_return']:.2f} "
                f"MAE={row['mean_absolute_error']:.3f} "
                f"rehandles/100={row['physical_rehandles_per_100_required_deliveries']:.3f} "
                f"strict={int(row['strict_method_success'])}",
                flush=True,
            )

        for instance_index, seed in enumerate(PANEL_SEEDS):
            instance = instances[seed]
            instance_hash = manifest["instances"][str(seed)]["serialized_sha256"]
            map_row = runtime.run_frozen_instance(
                instance=instance,
                instance_index=instance_index,
                policy="map",
                policy_rng_index=None,
                policy_rng_seed=seed,
            )
            map_row["episode_instance_serialized_sha256"] = instance_hash
            rows.append(map_row)
            report_progress(map_row)
            for replicate in range(SOFT_REPLICATES):
                row = runtime.run_frozen_instance(
                    instance=instance,
                    instance_index=instance_index,
                    policy="induced_soft",
                    policy_rng_index=replicate,
                    policy_rng_seed=soft_policy_rng_seed(instance_index, replicate),
                )
                row["episode_instance_serialized_sha256"] = instance_hash
                rows.append(row)
                report_progress(row)
        summary = analyze_policy_realization(
            rows,
            source_audit=contract,
            instance_manifest=manifest,
        )
        common_identity = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "audit_contract_sha256": contract["audit_contract_sha256"],
            "source_checkpoint_sha256": source.checkpoint_sha256,
            "instance_manifest_sha256": manifest["manifest_sha256"],
        }
        summary.update(common_identity)
        source_hashes_after = {
            "checkpoint": _sha256_file(source.checkpoint_path),
            "contract": _sha256_file(source.contract_path),
            "summary": _sha256_file(source.summary_path),
        }
        source_unchanged = source_hashes_before == source_hashes_after
        if not source_unchanged:
            raise RuntimeError("source artifacts changed during frozen audit")
        audit = {
            "audit_protocol": AUDIT_PROTOCOL,
            **common_identity,
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "source_artifacts_unchanged": source_unchanged,
            "source_checkpoint_eligible": False,
            "checkpoint_selection_performed": False,
            "panel_reused": True,
            "prospective_83xxx_panel_opened": False,
            "sealed_or_new_panel_opened": False,
            "conditional_mc_uncertainty_only": True,
            "strict_integrity_gate": summary["integrity"][
                "strict_integrity_gate"
            ],
            "performance_estimands_authentic": summary["conclusion"][
                "authenticated_comparison"
            ],
        }
        runs_payload = {
            "audit_protocol": AUDIT_PROTOCOL,
            **common_identity,
            "expected_row_count": total_rows,
            "observed_row_count": len(rows),
            "complete_case_filtering_used": False,
            "runs": rows,
        }
        runs_path = staging / "paired-runs.json"
        summary_path = staging / "paired-summary.json"
        _write_json(runs_path, runs_payload)
        _write_json(summary_path, summary)
        audit.update(
            {
                "paired_runs_file_sha256": _sha256_file(runs_path),
                "paired_summary_file_sha256": _sha256_file(summary_path),
                "audit_contract_file_sha256": _sha256_file(
                    staging / "audit-contract.json"
                ),
                "instance_manifest_file_sha256": _sha256_file(
                    staging / "instance-manifest.json"
                ),
            }
        )
        _write_json(staging / "audit.json", audit)
        returned_summary = summary

    _transactional_directory(Path(output_dir), write)
    return returned_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen V2.1 MAP versus induced-soft policy realizations"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the predeclared 99-row audit; omitted means authenticate only",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    source = authenticate_source(args.checkpoint)
    contract = build_audit_contract(source, device=str(args.device))
    if not args.execute:
        print(
            json.dumps(
                _json_safe(
                    {
                        "status": "authenticated_protocol_only",
                        "rows_executed": 0,
                        "audit_contract": contract,
                        "explicit_execute_required": True,
                    }
                ),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    summary = execute_audit(
        source=source,
        output_dir=args.output_dir,
        device=str(args.device),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_PROTOCOL",
    "AuthenticatedSource",
    "FrozenPairedPolicyRuntime",
    "PANEL_SEEDS",
    "PairedPolicyAuditError",
    "SOFT_REPLICATES",
    "authenticate_source",
    "build_audit_contract",
    "build_parser",
    "execute_audit",
    "materialize_instances",
    "soft_policy_rng_seed",
    "summarize_paired_rows",
]
