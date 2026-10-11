"""Frozen analysis for the V2.1 MAP-versus-induced-policy audit.

This module is intentionally independent of the environment runner.  It consumes
only authenticated rollout rows, verifies the complete predeclared grid, and
computes conditional Monte Carlo summaries without treating policy RNG draws as
additional EpisodeInstance observations.

The audit is development-only.  It reuses the three V2.1 validation instances
and cannot create a checkpoint, select a policy, or support a workload-
generalization claim.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import fsum, isclose, isfinite, sqrt
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence


PROTOCOL = "vcg_v2_1_frozen_paired_policy_realization_audit_v1"
SCOPE = "development_only_reused_v2_1_validation_panel"

MAP_POLICY = "map"
INDUCED_SOFT_POLICY = "induced_soft"
POLICIES = (MAP_POLICY, INDUCED_SOFT_POLICY)

FROZEN_INSTANCE_SEEDS = (84_000, 84_001, 84_002)
SOFT_REPLICATES_PER_INSTANCE = 32
EXPECTED_MAP_ROWS = len(FROZEN_INSTANCE_SEEDS)
EXPECTED_SOFT_ROWS = len(FROZEN_INSTANCE_SEEDS) * SOFT_REPLICATES_PER_INSTANCE
EXPECTED_TOTAL_ROWS = EXPECTED_MAP_ROWS + EXPECTED_SOFT_ROWS

POLICY_RNG_BASE = 910_000
POLICY_RNG_INSTANCE_STRIDE = 100

REHANDLE_BUDGET_PER_100_REQUIRED_DELIVERIES = 20.0
TWO_SIDED_95_T_DF31 = 2.0395134463964077
ONE_SIDED_95_T_DF31 = 1.695518782545865

FROZEN_CHECKPOINT_SHA256 = (
    "9364d16ab28610d7074619920e0d53961bfe3b94790d5a5f60a0d0f44b636e8a"
)
FROZEN_TRAINING_CONTRACT_SHA256 = (
    "87c9d4804bf9bc0219a8b515202d3a193936c4c9a2232ad84b243fb7db85fc7a"
)
FROZEN_CHECKPOINT_FAMILY = "vcg_constrained_vector_smdp_v2_1"
FROZEN_CHECKPOINT_ROLE = "latest_development_state"
FROZEN_DUAL_LAMBDA = 0.147
FROZEN_WITHIN_GROUP_TEMPERATURES = (0.01, 0.01, 0.01, 0.01)
FROZEN_GROUP_TEMPERATURE = 0.05
FROZEN_POLICY_SCHEDULE_PROTOCOL = (
    "block_constant_shared_behavior_backup_temperature_to_map_v1"
)
FROZEN_POLICY_DIAGNOSTIC_DISTRIBUTION = (
    "regularized_operator_distribution_not_executed_map_randomness"
)

ACTION_TYPES = ("accept", "deliver", "defer", "reconfigure")
RUN_METRICS = (
    "dense_return",
    "mean_absolute_error",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


class PolicyRealizationAnalysisError(ValueError):
    """Raised when rows do not authenticate the frozen audit contract."""


def expected_soft_policy_rng_seed(instance_index: int, replicate_index: int) -> int:
    """Return the predeclared, instance-disjoint nuisance RNG seed."""

    if isinstance(instance_index, bool) or not isinstance(instance_index, Integral):
        raise PolicyRealizationAnalysisError("instance_index must be an integer")
    if isinstance(replicate_index, bool) or not isinstance(replicate_index, Integral):
        raise PolicyRealizationAnalysisError("replicate_index must be an integer")
    instance_index = int(instance_index)
    replicate_index = int(replicate_index)
    if not 0 <= instance_index < len(FROZEN_INSTANCE_SEEDS):
        raise PolicyRealizationAnalysisError("instance_index lies outside frozen panel")
    if not 0 <= replicate_index < SOFT_REPLICATES_PER_INSTANCE:
        raise PolicyRealizationAnalysisError("replicate_index lies outside frozen grid")
    return (
        POLICY_RNG_BASE
        + POLICY_RNG_INSTANCE_STRIDE * instance_index
        + replicate_index
    )


def _require_mapping(value: Any, *, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise PolicyRealizationAnalysisError(f"{name} must be a mapping")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyRealizationAnalysisError(f"{name} must be boolean")
    return value


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PolicyRealizationAnalysisError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise PolicyRealizationAnalysisError(f"{name} must be >= {minimum}")
    return value


def _require_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PolicyRealizationAnalysisError(f"{name} must be numeric")
    value = float(value)
    if not isfinite(value):
        raise PolicyRealizationAnalysisError(f"{name} must be finite")
    return value


def _optional_finite(value: Any, *, name: str) -> Optional[float]:
    if value is None:
        return None
    return _require_finite(value, name=name)


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyRealizationAnalysisError(f"{name} must be non-empty text")
    return value


def _exact_float(observed: Any, expected: float, *, name: str) -> float:
    value = _require_finite(observed, name=name)
    if not isclose(value, expected, rel_tol=0.0, abs_tol=0.0):
        raise PolicyRealizationAnalysisError(
            f"{name} differs from frozen value: {value!r} != {expected!r}"
        )
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise PolicyRealizationAnalysisError("cannot average an empty sequence")
    return float(fsum(float(value) for value in values) / len(values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise PolicyRealizationAnalysisError("JSON authentication saw nonfinite data")
        return value
    raise PolicyRealizationAnalysisError(
        f"unsupported authentication value {type(value).__name__}"
    )


def _canonical_sha256(value: Mapping) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_audit(source_audit: Mapping) -> dict:
    source = _require_mapping(source_audit, name="source_audit")
    observed_contract_sha = _require_text(
        source.get("audit_contract_sha256"),
        name="source_audit.audit_contract_sha256",
    )
    unhashed_source = {
        key: value
        for key, value in source.items()
        if key != "audit_contract_sha256"
    }
    if _canonical_sha256(unhashed_source) != observed_contract_sha:
        raise PolicyRealizationAnalysisError("source audit contract SHA mismatch")
    expected = {
        "audit_protocol": PROTOCOL,
        "development_only": True,
        "cannot_create_or_select_checkpoint": True,
        "source_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "source_training_contract_sha256": FROZEN_TRAINING_CONTRACT_SHA256,
        "source_checkpoint_role": FROZEN_CHECKPOINT_ROLE,
        "source_checkpoint_eligible": False,
        "source_completed_episodes": 100,
        "complete_case_filtering_used": False,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise PolicyRealizationAnalysisError(
                f"source_audit.{key} mismatch: {source.get(key)!r} != {value!r}"
            )
    q_sha = _require_text(
        source.get("source_q_local_sha256"), name="source_audit.source_q_local_sha256"
    )
    if len(q_sha) != 64:
        raise PolicyRealizationAnalysisError("source Q-local SHA must have 64 characters")
    panel = _require_mapping(source.get("panel"), name="source_audit.panel")
    if tuple(panel.get("episode_instance_seeds", ())) != FROZEN_INSTANCE_SEEDS:
        raise PolicyRealizationAnalysisError("source audit panel seeds changed")
    for key, expected_value in (
        ("reused_already_open_v2_1_development_validation_panel", True),
        ("prospective_83xxx_panel_opened", False),
        ("sealed_or_new_panel_opened", False),
    ):
        if panel.get(key) is not expected_value:
            raise PolicyRealizationAnalysisError(f"source audit panel {key} mismatch")
    arms = _require_mapping(source.get("arms"), name="source_audit.arms")
    map_arm = _require_mapping(arms.get(MAP_POLICY), name="source_audit.arms.map")
    soft_arm = _require_mapping(
        arms.get(INDUCED_SOFT_POLICY), name="source_audit.arms.induced_soft"
    )
    if map_arm.get("runs_per_instance") != 1 or map_arm.get("policy_rng_used") is not False:
        raise PolicyRealizationAnalysisError("source MAP arm contract changed")
    if soft_arm.get("runs_per_instance") != SOFT_REPLICATES_PER_INSTANCE:
        raise PolicyRealizationAnalysisError("source soft replicate count changed")
    if tuple(soft_arm.get("replicate_indices", ())) != tuple(
        range(SOFT_REPLICATES_PER_INSTANCE)
    ):
        raise PolicyRealizationAnalysisError("source soft replicate indices changed")
    frozen = _require_mapping(
        source.get("frozen_policy"), name="source_audit.frozen_policy"
    )
    _exact_float(
        frozen.get("dual_lambda"), FROZEN_DUAL_LAMBDA, name="source frozen lambda"
    )
    if tuple(frozen.get("within_group_temperatures", ())) != (
        FROZEN_WITHIN_GROUP_TEMPERATURES
    ):
        raise PolicyRealizationAnalysisError("source frozen within temperatures changed")
    _exact_float(
        frozen.get("group_temperature"),
        FROZEN_GROUP_TEMPERATURE,
        name="source frozen group temperature",
    )
    _exact_float(frozen.get("epsilon"), 0.0, name="source frozen epsilon")
    for key in (
        "latest_q_local_weights_only",
        "same_weights_lambda_and_temperatures_across_arms",
        "option_evaluation_mode",
        "fresh_agent_clone_per_row",
        "fresh_certificate_cache_per_row",
        "replay_forbidden",
        "optimizer_step_forbidden",
        "target_update_forbidden",
    ):
        if frozen.get(key) is not True:
            raise PolicyRealizationAnalysisError(f"source frozen policy {key} mismatch")
    if frozen.get("evaluation_learning") is not False:
        raise PolicyRealizationAnalysisError("source audit permits evaluation learning")
    return {
        "source_q_local_sha256": q_sha,
        "audit_contract_sha256": observed_contract_sha,
    }


def _validate_instance_manifest(instance_manifest: Mapping) -> dict[int, dict]:
    manifest = _require_mapping(instance_manifest, name="instance_manifest")
    if manifest.get("audit_protocol") != PROTOCOL:
        raise PolicyRealizationAnalysisError("instance manifest protocol mismatch")
    if tuple(manifest.get("episode_instance_seeds", ())) != FROZEN_INSTANCE_SEEDS:
        raise PolicyRealizationAnalysisError("instance manifest panel changed")
    if manifest.get("reused_development_panel") is not True:
        raise PolicyRealizationAnalysisError("instance manifest is not reused development")
    if manifest.get("prospective_83xxx_panel_opened") is not False:
        raise PolicyRealizationAnalysisError("instance manifest opened protected panel")
    observed_digest = _require_text(
        manifest.get("manifest_sha256"), name="instance_manifest.manifest_sha256"
    )
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _canonical_sha256(unhashed) != observed_digest:
        raise PolicyRealizationAnalysisError("instance manifest SHA mismatch")
    records = _require_mapping(manifest.get("instances"), name="instance_manifest.instances")
    if set(records) != {str(seed) for seed in FROZEN_INSTANCE_SEEDS}:
        raise PolicyRealizationAnalysisError("instance manifest keys are not exact")
    normalized = {}
    ids = set()
    hashes = set()
    for index, seed in enumerate(FROZEN_INSTANCE_SEEDS):
        record = _require_mapping(records[str(seed)], name=f"manifest instance {seed}")
        if record.get("instance_index") != index or record.get("instance_seed") != seed:
            raise PolicyRealizationAnalysisError("instance manifest index/seed mismatch")
        instance_id = _require_text(record.get("instance_id"), name="manifest instance_id")
        schedule_id = _require_text(record.get("schedule_id"), name="manifest schedule_id")
        serialized_sha = _require_text(
            record.get("serialized_sha256"), name="manifest serialized_sha256"
        )
        required = _require_int(
            record.get("number_blocks"), name="manifest number_blocks", minimum=1
        )
        if instance_id in ids or serialized_sha in hashes:
            raise PolicyRealizationAnalysisError("manifest instances are not unique")
        ids.add(instance_id)
        hashes.add(serialized_sha)
        normalized[seed] = {
            "instance_index": index,
            "instance_seed": seed,
            "instance_id": instance_id,
            "schedule_id": schedule_id,
            "serialized_sha256": serialized_sha,
            "required_deliveries": required,
        }
    return normalized


def _sample_standard_error(values: Sequence[float]) -> float:
    if len(values) != SOFT_REPLICATES_PER_INSTANCE:
        raise PolicyRealizationAnalysisError("Monte Carlo bundle must contain 32 values")
    center = _mean(values)
    variance = fsum((float(value) - center) ** 2 for value in values) / (
        len(values) - 1
    )
    return float(sqrt(max(variance, 0.0) / len(values)))


def _normalize_actions(value: Any, *, row_name: str, macro_decisions: int) -> dict:
    mapping = _require_mapping(value, name=f"{row_name}.selected_action_counts")
    unknown = set(mapping).difference(ACTION_TYPES)
    if unknown:
        raise PolicyRealizationAnalysisError(
            f"{row_name} has unknown action types: {sorted(unknown)!r}"
        )
    counts = {
        action: _require_int(
            mapping.get(action, 0), name=f"{row_name}.selected_action_counts[{action}]"
        )
        for action in ACTION_TYPES
    }
    if sum(counts.values()) != macro_decisions:
        raise PolicyRealizationAnalysisError(
            f"{row_name} action counts do not sum to macro_decisions"
        )
    return counts


def _row_key(row: Mapping) -> tuple[str, int, Optional[int]]:
    return (row["policy"], row["instance_seed"], row["policy_rng_index"])


def _normalize_row(raw: Mapping, index: int) -> dict:
    row = _require_mapping(raw, name=f"rows[{index}]")
    name = f"rows[{index}]"
    if row.get("audit_protocol") != PROTOCOL or row.get("audit_schema_version") != 1:
        raise PolicyRealizationAnalysisError(f"{name} audit protocol/schema mismatch")
    policy = row.get("policy")
    if policy not in POLICIES:
        raise PolicyRealizationAnalysisError(f"{name}.policy is not frozen")

    instance_index = _require_int(row.get("instance_index"), name=f"{name}.instance_index")
    if instance_index >= len(FROZEN_INSTANCE_SEEDS):
        raise PolicyRealizationAnalysisError(f"{name}.instance_index is outside panel")
    instance_seed = _require_int(row.get("instance_seed"), name=f"{name}.instance_seed")
    expected_instance_seed = FROZEN_INSTANCE_SEEDS[instance_index]
    if instance_seed != expected_instance_seed:
        raise PolicyRealizationAnalysisError(
            f"{name} instance index/seed mismatch: {instance_seed} != {expected_instance_seed}"
        )

    rng_index_raw = row.get("policy_rng_index")
    rng_seed_raw = row.get("policy_rng_seed")
    if policy == MAP_POLICY:
        if rng_index_raw is not None:
            raise PolicyRealizationAnalysisError(f"{name} MAP RNG index must be null")
        rng_index = None
        rng_seed = _require_int(rng_seed_raw, name=f"{name}.policy_rng_seed")
        if rng_seed != instance_seed:
            raise PolicyRealizationAnalysisError(
                f"{name} MAP must replay with policy seed equal to instance seed"
            )
        if _require_bool(row.get("policy_rng_used"), name=f"{name}.policy_rng_used"):
            raise PolicyRealizationAnalysisError(f"{name} MAP cannot use policy RNG")
    else:
        rng_index = _require_int(rng_index_raw, name=f"{name}.policy_rng_index")
        if rng_index >= SOFT_REPLICATES_PER_INSTANCE:
            raise PolicyRealizationAnalysisError(f"{name} soft RNG index is outside grid")
        rng_seed = _require_int(rng_seed_raw, name=f"{name}.policy_rng_seed")
        expected_rng_seed = expected_soft_policy_rng_seed(instance_index, rng_index)
        if rng_seed != expected_rng_seed:
            raise PolicyRealizationAnalysisError(
                f"{name} soft RNG seed mismatch: {rng_seed} != {expected_rng_seed}"
            )
        if not _require_bool(row.get("policy_rng_used"), name=f"{name}.policy_rng_used"):
            raise PolicyRealizationAnalysisError(f"{name} soft policy must use policy RNG")

    expected_realization = (
        "deterministic_nested_lagrangian_map"
        if policy == MAP_POLICY
        else "induced_nested_regularized_lagrangian_sample"
    )
    if row.get("policy_realization") != expected_realization:
        raise PolicyRealizationAnalysisError(f"{name} policy realization mismatch")
    if row.get("policy_rng_scope") != "action_sampling_only":
        raise PolicyRealizationAnalysisError(f"{name} policy RNG scope mismatch")

    checkpoint_sha = _require_text(
        row.get("source_checkpoint_sha256"), name=f"{name}.source_checkpoint_sha256"
    )
    if checkpoint_sha != FROZEN_CHECKPOINT_SHA256:
        raise PolicyRealizationAnalysisError(f"{name} source checkpoint SHA mismatch")
    contract_sha = _require_text(
        row.get("source_training_contract_sha256"),
        name=f"{name}.source_training_contract_sha256"
    )
    if contract_sha != FROZEN_TRAINING_CONTRACT_SHA256:
        raise PolicyRealizationAnalysisError(f"{name} training contract SHA mismatch")
    if row.get("source_checkpoint_role") != FROZEN_CHECKPOINT_ROLE:
        raise PolicyRealizationAnalysisError(f"{name} checkpoint role mismatch")
    source_q_sha = _require_text(
        row.get("source_q_local_sha256"), name=f"{name}.source_q_local_sha256"
    )
    if len(source_q_sha) != 64:
        raise PolicyRealizationAnalysisError(f"{name} source Q-local SHA is malformed")
    if row.get("policy_schedule_protocol") != FROZEN_POLICY_SCHEDULE_PROTOCOL:
        raise PolicyRealizationAnalysisError(f"{name} schedule protocol mismatch")
    if (
        row.get("schedule_episode_number") != 100
        or row.get("schedule_block_number") != 10
        or row.get("schedule_phase") != "low_temperature_stabilization"
    ):
        raise PolicyRealizationAnalysisError(f"{name} schedule clock mismatch")
    expected_policy_mode = (
        "map" if policy == MAP_POLICY else "regularized_sample"
    )
    expected_stochastic = policy == INDUCED_SOFT_POLICY
    if row.get("schedule_policy_mode") != expected_policy_mode:
        raise PolicyRealizationAnalysisError(f"{name} schedule policy mode mismatch")
    if _require_bool(
        row.get("controller_select_internal_training_flag"),
        name=f"{name}.controller_select_internal_training_flag",
    ) is not expected_stochastic:
        raise PolicyRealizationAnalysisError(
            f"{name} controller selection realization mismatch"
        )
    if _require_bool(
        row.get("stochastic_selection_only"),
        name=f"{name}.stochastic_selection_only",
    ) is not expected_stochastic:
        raise PolicyRealizationAnalysisError(
            f"{name} stochastic-selection flag mismatch"
        )

    dual_lambda = _exact_float(
        row.get("dual_lambda"), FROZEN_DUAL_LAMBDA, name=f"{name}.dual_lambda"
    )
    temperatures = row.get("within_group_temperatures")
    if isinstance(temperatures, (str, bytes)) or not isinstance(temperatures, Sequence):
        raise PolicyRealizationAnalysisError(
            f"{name}.within_group_temperatures must be a sequence"
        )
    temperatures = tuple(
        _require_finite(value, name=f"{name}.within_group_temperatures")
        for value in temperatures
    )
    if temperatures != FROZEN_WITHIN_GROUP_TEMPERATURES:
        raise PolicyRealizationAnalysisError(f"{name} within temperatures mismatch")
    group_temperature = _exact_float(
        row.get("group_temperature"),
        FROZEN_GROUP_TEMPERATURE,
        name=f"{name}.group_temperature",
    )
    _exact_float(row.get("epsilon"), 0.0, name=f"{name}.epsilon")

    required_deliveries = _require_int(
        row.get("required_deliveries"), name=f"{name}.required_deliveries", minimum=1
    )
    delivery_count = _require_int(
        row.get("delivery_count"), name=f"{name}.delivery_count"
    )
    completion_rate = _require_finite(
        row.get("completion_rate"), name=f"{name}.completion_rate"
    )
    physical_rehandles = _require_int(
        row.get("physical_rehandles"), name=f"{name}.physical_rehandles"
    )
    reported_rehandle_rate = _require_finite(
        row.get("physical_rehandles_per_100_required_deliveries"),
        name=f"{name}.physical_rehandles_per_100_required_deliveries",
    )
    derived_rehandle_rate = 100.0 * physical_rehandles / required_deliveries
    if not isclose(
        reported_rehandle_rate, derived_rehandle_rate, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise PolicyRealizationAnalysisError(f"{name} rehandle rate is inconsistent")
    dense_return = _require_finite(row.get("dense_return"), name=f"{name}.dense_return")
    mean_absolute_error = _optional_finite(
        row.get("mean_absolute_error"), name=f"{name}.mean_absolute_error"
    )
    steps = _require_int(row.get("steps"), name=f"{name}.steps")
    macro_decisions = _require_int(
        row.get("macro_decisions"), name=f"{name}.macro_decisions"
    )
    actions = _normalize_actions(
        row.get("selected_action_counts"),
        row_name=name,
        macro_decisions=macro_decisions,
    )
    for action, count_field, share_field in (
        ("defer", "hold_decisions", "hold_decision_share"),
        ("reconfigure", "reconfigure_decisions", "reconfigure_decision_share"),
    ):
        if _require_int(row.get(count_field), name=f"{name}.{count_field}") != actions[action]:
            raise PolicyRealizationAnalysisError(f"{name}.{count_field} is inconsistent")
        expected_share = 0.0 if macro_decisions == 0 else actions[action] / macro_decisions
        share = _require_finite(row.get(share_field), name=f"{name}.{share_field}")
        if not isclose(share, expected_share, rel_tol=0.0, abs_tol=1.0e-12):
            raise PolicyRealizationAnalysisError(f"{name}.{share_field} is inconsistent")

    deviations_raw = row.get("delivery_deviations")
    if isinstance(deviations_raw, (str, bytes)) or not isinstance(
        deviations_raw, Sequence
    ):
        raise PolicyRealizationAnalysisError(f"{name}.delivery_deviations must be a sequence")
    deviations = tuple(
        _require_finite(value, name=f"{name}.delivery_deviations")
        for value in deviations_raw
    )
    if len(deviations) != delivery_count:
        raise PolicyRealizationAnalysisError(
            f"{name} delivery deviation count does not match delivery_count"
        )
    derived_mae = None if not deviations else _mean([abs(value) for value in deviations])
    if mean_absolute_error is None:
        if derived_mae is not None:
            raise PolicyRealizationAnalysisError(f"{name} omits an available MAE")
    elif derived_mae is None or not isclose(
        mean_absolute_error, derived_mae, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise PolicyRealizationAnalysisError(f"{name} MAE disagrees with deviations")

    diagnostic_count = _require_int(
        row.get("policy_diagnostic_decisions"),
        name=f"{name}.policy_diagnostic_decisions",
    )
    if diagnostic_count != macro_decisions:
        raise PolicyRealizationAnalysisError(
            f"{name} policy diagnostic count differs from macro decisions"
        )
    if row.get("policy_diagnostic_distribution") != FROZEN_POLICY_DIAGNOSTIC_DISTRIBUTION:
        raise PolicyRealizationAnalysisError(f"{name} diagnostic distribution mismatch")
    diagnostics = {}
    for field in (
        "mean_outer_map_probability",
        "mean_selected_within_map_probability",
        "mean_outer_policy_entropy",
        "mean_selected_within_policy_entropy",
    ):
        value = _optional_finite(row.get(field), name=f"{name}.{field}")
        if diagnostic_count > 0 and value is None:
            raise PolicyRealizationAnalysisError(f"{name}.{field} is required")
        if value is not None:
            if "probability" in field and not 0.0 <= value <= 1.0:
                raise PolicyRealizationAnalysisError(f"{name}.{field} lies outside [0,1]")
            if "entropy" in field and value < 0.0:
                raise PolicyRealizationAnalysisError(f"{name}.{field} is negative")
        diagnostics[field] = value

    normalized = {
        "policy": policy,
        "instance_index": instance_index,
        "instance_seed": instance_seed,
        "instance_id": _require_text(
            row.get("instance_id"), name=f"{name}.instance_id"
        ),
        "episode_instance_serialized_sha256": _require_text(
            row.get("episode_instance_serialized_sha256"),
            name=f"{name}.episode_instance_serialized_sha256"
        ),
        "schedule_id": _require_text(row.get("schedule_id"), name=f"{name}.schedule_id"),
        "policy_rng_index": rng_index,
        "policy_rng_seed": rng_seed,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_training_contract_sha256": contract_sha,
        "source_q_local_sha256": source_q_sha,
        "dual_lambda": dual_lambda,
        "within_group_temperatures": temperatures,
        "group_temperature": group_temperature,
        "strict_method_success": _require_bool(
            row.get("strict_method_success"), name=f"{name}.strict_method_success"
        ),
        "success": _require_bool(row.get("success"), name=f"{name}.success"),
        "completion_rate": completion_rate,
        "delivery_count": delivery_count,
        "required_deliveries": required_deliveries,
        "physical_rehandles": physical_rehandles,
        "physical_rehandles_per_100_required_deliveries": derived_rehandle_rate,
        "dense_return": dense_return,
        "mean_absolute_error": mean_absolute_error,
        "delivery_deviations": deviations,
        "steps": steps,
        "macro_decisions": macro_decisions,
        "selected_action_counts": actions,
        "method_failure_reason": row.get("method_failure_reason"),
        "illegal_drops": _require_int(row.get("illegal_drops"), name=f"{name}.illegal_drops"),
        "fallbacks": _require_int(row.get("fallbacks"), name=f"{name}.fallbacks"),
        "witness_mismatches": _require_int(
            row.get("witness_mismatches"), name=f"{name}.witness_mismatches"
        ),
        "all_selected_candidates_exact_safe": _require_bool(
            row.get("all_selected_candidates_exact_safe"),
            name=f"{name}.all_selected_candidates_exact_safe",
        ),
        "evaluation_learning": _require_bool(
            row.get("evaluation_learning"), name=f"{name}.evaluation_learning"
        ),
        "policy_frozen": _require_bool(
            row.get("policy_frozen"), name=f"{name}.policy_frozen"
        ),
        "option_evaluation_mode": _require_bool(
            row.get("option_evaluation_mode"), name=f"{name}.option_evaluation_mode"
        ),
        "q_local_training_mode": _require_bool(
            row.get("q_local_training_mode"), name=f"{name}.q_local_training_mode"
        ),
        "loss_updates": _require_int(row.get("loss_updates"), name=f"{name}.loss_updates"),
        "source_agent_unchanged": _require_bool(
            row.get("source_agent_unchanged"), name=f"{name}.source_agent_unchanged"
        ),
        "training_agent_unchanged": _require_bool(
            row.get("training_agent_unchanged"),
            name=f"{name}.training_agent_unchanged",
        ),
        "clone_learning_state_unchanged": _require_bool(
            row.get("clone_learning_state_unchanged"),
            name=f"{name}.clone_learning_state_unchanged",
        ),
        "source_checkpoint_file_unchanged": _require_bool(
            row.get("source_checkpoint_file_unchanged"),
            name=f"{name}.source_checkpoint_file_unchanged",
        ),
        "fresh_certificate_cache": _require_bool(
            row.get("fresh_certificate_cache"), name=f"{name}.fresh_certificate_cache"
        ),
        "exact_safe_frontier_authoritative": _require_bool(
            row.get("exact_safe_frontier_authoritative"),
            name=f"{name}.exact_safe_frontier_authoritative",
        ),
        "baseline_teacher": _require_bool(
            row.get("baseline_teacher"), name=f"{name}.baseline_teacher"
        ),
        "baseline_policy_query": _require_bool(
            row.get("baseline_policy_query"), name=f"{name}.baseline_policy_query"
        ),
        "policy_rng_used": bool(row["policy_rng_used"]),
        "policy_diagnostic_decisions": diagnostic_count,
        **diagnostics,
    }
    for field in (
        "replay_size_before",
        "replay_size_after",
        "gradient_steps_before",
        "gradient_steps_after",
        "target_updates_before",
        "target_updates_after",
        "optimizer_state_entries_before",
        "optimizer_state_entries_after",
    ):
        normalized[field] = _require_int(row.get(field), name=f"{name}.{field}")
    physical_alias = row.get("physical_storage_relocations")
    if physical_alias is not None and _require_int(
        physical_alias, name=f"{name}.physical_storage_relocations"
    ) != physical_rehandles:
        raise PolicyRealizationAnalysisError(
            f"{name} physical relocation aliases disagree"
        )
    return normalized


def _safety_reasons(row: Mapping) -> list[str]:
    reasons = []
    checks = (
        (row["strict_method_success"], "strict_method_success_false"),
        (row["success"], "success_false"),
        (isclose(row["completion_rate"], 1.0, rel_tol=0.0, abs_tol=1.0e-12), "incomplete"),
        (row["delivery_count"] == row["required_deliveries"], "delivery_count_mismatch"),
        (row["method_failure_reason"] is None, "method_failure"),
        (row["illegal_drops"] == 0, "illegal_drops"),
        (row["fallbacks"] == 0, "fallbacks"),
        (row["witness_mismatches"] == 0, "witness_mismatches"),
        (row["all_selected_candidates_exact_safe"], "non_exact_safe_selection"),
        (row["exact_safe_frontier_authoritative"], "exact_frontier_not_authoritative"),
        (not row["evaluation_learning"], "evaluation_learning_enabled"),
        (row["policy_frozen"], "policy_not_frozen"),
        (row["option_evaluation_mode"], "options_not_in_evaluation_mode"),
        (not row["q_local_training_mode"], "q_local_training_mode"),
        (row["loss_updates"] == 0, "loss_update"),
        (row["replay_size_before"] == row["replay_size_after"] == 0, "evaluation_replay_nonempty"),
        (row["gradient_steps_before"] == row["gradient_steps_after"] == 0, "gradient_update"),
        (row["target_updates_before"] == row["target_updates_after"] == 0, "target_update"),
        (
            row["optimizer_state_entries_before"]
            == row["optimizer_state_entries_after"]
            == 0,
            "optimizer_state_nonempty",
        ),
        (row["clone_learning_state_unchanged"], "clone_learning_state_changed"),
        (row["source_agent_unchanged"], "source_agent_changed"),
        (row["training_agent_unchanged"], "training_agent_changed"),
        (row["source_checkpoint_file_unchanged"], "source_checkpoint_file_changed"),
        (row["fresh_certificate_cache"], "certificate_cache_not_fresh"),
        (not row["baseline_teacher"], "baseline_teacher_used"),
        (not row["baseline_policy_query"], "baseline_policy_queried"),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    if row["mean_absolute_error"] is None:
        reasons.append("mean_absolute_error_unavailable")
    return reasons


def _metric_summary(
    map_rows: Mapping[int, Mapping],
    soft_rows: Mapping[tuple[int, int], Mapping],
    field: str,
    *,
    higher_is_better: bool,
) -> dict:
    all_values = [row[field] for row in map_rows.values()] + [
        row[field] for row in soft_rows.values()
    ]
    if any(value is None for value in all_values):
        return {
            "estimable": False,
            "reason": "at_least_one_predeclared_row_has_no_metric; no complete-case filtering",
            "higher_is_better": bool(higher_is_better),
            "conditional_mc_only": True,
        }

    map_values = [float(map_rows[seed][field]) for seed in FROZEN_INSTANCE_SEEDS]
    map_mean = _mean(map_values)
    bundles = []
    for replicate in range(SOFT_REPLICATES_PER_INSTANCE):
        bundles.append(
            _mean(
                [
                    float(soft_rows[(seed, replicate)][field])
                    for seed in FROZEN_INSTANCE_SEEDS
                ]
            )
        )
    soft_mean = _mean(bundles)
    contrasts = [value - map_mean for value in bundles]
    contrast = _mean(contrasts)
    standard_error = _sample_standard_error(contrasts)
    half_width = TWO_SIDED_95_T_DF31 * standard_error
    per_instance = {}
    for seed in FROZEN_INSTANCE_SEEDS:
        local_soft = _mean(
            [
                float(soft_rows[(seed, replicate)][field])
                for replicate in range(SOFT_REPLICATES_PER_INSTANCE)
            ]
        )
        local_map = float(map_rows[seed][field])
        per_instance[str(seed)] = {
            "map": local_map,
            "induced_soft_conditional_mean": local_soft,
            "soft_minus_map": local_soft - local_map,
        }
    return {
        "estimable": True,
        "higher_is_better": bool(higher_is_better),
        "map_equal_instance_mean": map_mean,
        "induced_soft_equal_instance_conditional_mean": soft_mean,
        "soft_minus_map": contrast,
        "soft_minus_map_mc_standard_error": standard_error,
        "soft_minus_map_mc_95_interval": [
            contrast - half_width,
            contrast + half_width,
        ],
        "mc_degrees_of_freedom": SOFT_REPLICATES_PER_INSTANCE - 1,
        "mc_t_critical_two_sided_95": TWO_SIDED_95_T_DF31,
        "conditional_mc_only": True,
        "per_instance": per_instance,
    }


def _action_summary(rows: Sequence[Mapping]) -> dict:
    output = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        counts = Counter()
        decisions = 0
        for row in selected:
            counts.update(row["selected_action_counts"])
            decisions += row["macro_decisions"]
        output[policy] = {
            "counts": {action: int(counts[action]) for action in ACTION_TYPES},
            "macro_decisions": int(decisions),
            "decision_weighted_shares": {
                action: (None if decisions == 0 else float(counts[action] / decisions))
                for action in ACTION_TYPES
            },
        }
    output["share_convention"] = (
        "pooled_action_count_divided_by_pooled_macro_decisions; descriptive_only"
    )
    return output


def _map_reproduction_summary(map_rows: Mapping[int, Mapping]) -> dict:
    rows = [map_rows[seed] for seed in FROZEN_INSTANCE_SEEDS]
    actions = {
        action: sum(row["selected_action_counts"][action] for row in rows)
        for action in ACTION_TYPES
    }
    total_required = sum(row["required_deliveries"] for row in rows)
    observed = {
        "mean_dense_return": _mean([row["dense_return"] for row in rows]),
        "mean_absolute_error": (
            None
            if any(row["mean_absolute_error"] is None for row in rows)
            else _mean([row["mean_absolute_error"] for row in rows])
        ),
        "physical_rehandles_per_100_required_deliveries": (
            100.0 * sum(row["physical_rehandles"] for row in rows) / total_required
        ),
        "selected_action_counts": actions,
        "strict_count": sum(row["strict_method_success"] for row in rows),
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
    numeric = all(
        observed[key] is not None
        and isclose(
            float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=1.0e-9
        )
        for key in (
            "mean_dense_return",
            "mean_absolute_error",
            "physical_rehandles_per_100_required_deliveries",
        )
    )
    passed = bool(
        numeric
        and observed["selected_action_counts"] == expected["selected_action_counts"]
        and observed["strict_count"] == expected["strict_count"]
    )
    return {
        "gate": passed,
        "observed": observed,
        "expected": expected,
        "absolute_tolerance": 1.0e-9,
    }


def analyze_policy_realization_rows(
    rows: Sequence[Mapping], *, source_audit: Mapping, instance_manifest: Mapping
) -> dict:
    """Validate and summarize the complete frozen policy-realization audit.

    Structural authentication failures raise :class:`PolicyRealizationAnalysisError`.
    Operational or mutation failures remain in the result and make all
    authenticated gates false; no failed row is filtered from an estimand.
    """

    source_identity = _validate_source_audit(source_audit)
    authenticated_manifest = _validate_instance_manifest(instance_manifest)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise PolicyRealizationAnalysisError("rows must be a sequence")
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise PolicyRealizationAnalysisError(
            f"frozen grid requires {EXPECTED_TOTAL_ROWS} rows; found {len(rows)}"
        )
    normalized = [_normalize_row(row, index) for index, row in enumerate(rows)]
    for row in normalized:
        if row["source_q_local_sha256"] != source_identity["source_q_local_sha256"]:
            raise PolicyRealizationAnalysisError("row/source Q-local SHA mismatch")
        record = authenticated_manifest[row["instance_seed"]]
        observed = (
            row["instance_index"],
            row["instance_id"],
            row["schedule_id"],
            row["episode_instance_serialized_sha256"],
            row["required_deliveries"],
        )
        expected = (
            record["instance_index"],
            record["instance_id"],
            record["schedule_id"],
            record["serialized_sha256"],
            record["required_deliveries"],
        )
        if observed != expected:
            raise PolicyRealizationAnalysisError(
                f"row/manifest identity mismatch for seed {row['instance_seed']}"
            )

    keyed = {}
    for row in normalized:
        key = _row_key(row)
        if key in keyed:
            raise PolicyRealizationAnalysisError(f"duplicate frozen row key: {key!r}")
        keyed[key] = row

    map_rows = {}
    soft_rows = {}
    for instance_index, seed in enumerate(FROZEN_INSTANCE_SEEDS):
        map_key = (MAP_POLICY, seed, None)
        if map_key not in keyed:
            raise PolicyRealizationAnalysisError(f"missing MAP row for instance {seed}")
        map_rows[seed] = keyed[map_key]
        for replicate in range(SOFT_REPLICATES_PER_INSTANCE):
            soft_key = (INDUCED_SOFT_POLICY, seed, replicate)
            if soft_key not in keyed:
                raise PolicyRealizationAnalysisError(
                    f"missing induced-soft row for instance {seed}, replicate {replicate}"
                )
            soft_rows[(seed, replicate)] = keyed[soft_key]
    if len(map_rows) != EXPECTED_MAP_ROWS or len(soft_rows) != EXPECTED_SOFT_ROWS:
        raise PolicyRealizationAnalysisError("frozen policy grid is incomplete")

    manifest_summary = []
    for index, seed in enumerate(FROZEN_INSTANCE_SEEDS):
        reference = map_rows[seed]
        identity = (
            reference["instance_id"],
            reference["episode_instance_serialized_sha256"],
            reference["schedule_id"],
            reference["required_deliveries"],
        )
        for replicate in range(SOFT_REPLICATES_PER_INSTANCE):
            row = soft_rows[(seed, replicate)]
            observed = (
                row["instance_id"],
                row["episode_instance_serialized_sha256"],
                row["schedule_id"],
                row["required_deliveries"],
            )
            if observed != identity:
                raise PolicyRealizationAnalysisError(
                    f"instance manifest changed across policies for seed {seed}"
                )
        manifest_summary.append(
            {
                "instance_index": index,
                "instance_seed": seed,
                "instance_id": identity[0],
                "episode_instance_serialized_sha256": identity[1],
                "schedule_id": identity[2],
                "required_deliveries": identity[3],
            }
        )

    failures = []
    for row in normalized:
        reasons = _safety_reasons(row)
        if reasons:
            failures.append(
                {
                    "policy": row["policy"],
                    "instance_seed": row["instance_seed"],
                    "policy_rng_index": row["policy_rng_index"],
                    "reasons": reasons,
                }
            )
    map_reproduction = _map_reproduction_summary(map_rows)
    strict_integrity_gate = bool(not failures and map_reproduction["gate"])

    metrics = {
        "dense_return": _metric_summary(
            map_rows, soft_rows, "dense_return", higher_is_better=True
        ),
        "mean_absolute_error": _metric_summary(
            map_rows, soft_rows, "mean_absolute_error", higher_is_better=False
        ),
        "steps": _metric_summary(
            map_rows, soft_rows, "steps", higher_is_better=False
        ),
        "physical_rehandles_per_100_required_deliveries": _metric_summary(
            map_rows,
            soft_rows,
            "physical_rehandles_per_100_required_deliveries",
            higher_is_better=False,
        ),
    }

    budget_metric = metrics["physical_rehandles_per_100_required_deliveries"]
    if not budget_metric["estimable"]:
        raise PolicyRealizationAnalysisError("physical-rehandle budget is not estimable")
    map_budget = float(budget_metric["map_equal_instance_mean"])
    soft_budget = float(budget_metric["induced_soft_equal_instance_conditional_mean"])
    budget_bundles = [
        _mean(
            [
                soft_rows[(seed, replicate)][
                    "physical_rehandles_per_100_required_deliveries"
                ]
                for seed in FROZEN_INSTANCE_SEEDS
            ]
        )
        for replicate in range(SOFT_REPLICATES_PER_INSTANCE)
    ]
    budget_se = _sample_standard_error(budget_bundles)
    budget_ucb = soft_budget + ONE_SIDED_95_T_DF31 * budget_se
    map_budget_gate = map_budget <= REHANDLE_BUDGET_PER_100_REQUIRED_DELIVERIES
    soft_budget_point_gate = soft_budget <= REHANDLE_BUDGET_PER_100_REQUIRED_DELIVERIES
    soft_budget_ucb_gate = budget_ucb <= REHANDLE_BUDGET_PER_100_REQUIRED_DELIVERIES
    budget_feasibility_gate = bool(
        strict_integrity_gate
        and map_budget_gate
        and soft_budget_point_gate
        and soft_budget_ucb_gate
    )

    return_metric = metrics["dense_return"]
    mae_metric = metrics["mean_absolute_error"]
    performance_estimable = bool(return_metric["estimable"] and mae_metric["estimable"])
    authenticated_comparison = bool(
        strict_integrity_gate and budget_feasibility_gate and performance_estimable
    )
    return_status = "not_authenticated"
    mae_status = "not_authenticated"
    if authenticated_comparison:
        return_low, return_high = return_metric["soft_minus_map_mc_95_interval"]
        mae_low, mae_high = mae_metric["soft_minus_map_mc_95_interval"]
        return_status = (
            "conditional_soft_improvement_supported"
            if return_low > 0.0
            else (
                "conditional_soft_disadvantage_supported"
                if return_high < 0.0
                else "conditional_difference_inconclusive"
            )
        )
        mae_status = (
            "conditional_soft_improvement_supported"
            if mae_high < 0.0
            else (
                "conditional_soft_disadvantage_supported"
                if mae_low > 0.0
                else "conditional_difference_inconclusive"
            )
        )

    return {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "development_only": True,
        "panel_reused": True,
        "prospective_or_sealed_panel_opened": False,
        "source_checkpoint": {
            "sha256": FROZEN_CHECKPOINT_SHA256,
            "training_contract_sha256": FROZEN_TRAINING_CONTRACT_SHA256,
            "checkpoint_family": FROZEN_CHECKPOINT_FAMILY,
            "checkpoint_role": FROZEN_CHECKPOINT_ROLE,
            "development_candidate_eligible": False,
            "deployment_checkpoint_eligible": False,
            "dual_lambda": FROZEN_DUAL_LAMBDA,
            "within_group_temperatures": list(FROZEN_WITHIN_GROUP_TEMPERATURES),
            "group_temperature": FROZEN_GROUP_TEMPERATURE,
        },
        "design": {
            "instance_seeds": list(FROZEN_INSTANCE_SEEDS),
            "instance_count": len(FROZEN_INSTANCE_SEEDS),
            "map_rows": EXPECTED_MAP_ROWS,
            "soft_nuisance_replicates_per_instance": SOFT_REPLICATES_PER_INSTANCE,
            "soft_rows": EXPECTED_SOFT_ROWS,
            "total_rows": EXPECTED_TOTAL_ROWS,
            "policy_rng_seed_formula": "910000 + 100 * instance_index + replicate_index",
            "aggregation": "soft-within-instance_then_equal-weight-instances",
            "conditional_mc_uncertainty_only": True,
            "environment_population_uncertainty_estimated": False,
            "no_complete_case_filtering": True,
        },
        "instance_manifest": manifest_summary,
        "integrity": {
            "exact_grid_authenticated": True,
            "strict_integrity_gate": strict_integrity_gate,
            "map_source_reproduction_gate": map_reproduction["gate"],
            "map_source_reproduction": map_reproduction,
            "failure_count": len(failures),
            "failures": failures,
            "all_99_rows_retained": True,
        },
        "metrics": metrics,
        "action_realization": _action_summary(normalized),
        "budget": {
            "definition": "expected_physical_rehandles_per_100_required_deliveries",
            "limit": REHANDLE_BUDGET_PER_100_REQUIRED_DELIVERIES,
            "map_point_estimate": map_budget,
            "map_point_gate": map_budget_gate,
            "soft_conditional_point_estimate": soft_budget,
            "soft_point_gate": soft_budget_point_gate,
            "soft_mc_standard_error": budget_se,
            "soft_one_sided_95_upper_bound": budget_ucb,
            "soft_one_sided_95_t_critical": ONE_SIDED_95_T_DF31,
            "soft_upper_bound_gate": soft_budget_ucb_gate,
            "strict_integrity_required": True,
            "authenticated_budget_feasibility_gate": budget_feasibility_gate,
        },
        "conclusion": {
            "authenticated_comparison": authenticated_comparison,
            "dense_return": return_status,
            "mean_absolute_error": mae_status,
            "allowed": (
                "mechanistic conditional evidence about MAP versus the induced soft "
                "policy on the reused three-instance V2.1 development panel"
            ),
            "disallowed": [
                "deployment eligibility",
                "checkpoint selection or rescue",
                "workload or layout generalization",
                "superiority over baselines",
                "prospective or confirmatory evidence",
                "causal benefit of reconfiguration",
            ],
        },
    }


# Concise public alias for runner integration.
analyze_policy_realization = analyze_policy_realization_rows


__all__ = [
    "ACTION_TYPES",
    "EXPECTED_TOTAL_ROWS",
    "FROZEN_INSTANCE_SEEDS",
    "INDUCED_SOFT_POLICY",
    "MAP_POLICY",
    "ONE_SIDED_95_T_DF31",
    "POLICY_RNG_BASE",
    "PROTOCOL",
    "PolicyRealizationAnalysisError",
    "REHANDLE_BUDGET_PER_100_REQUIRED_DELIVERIES",
    "SOFT_REPLICATES_PER_INSTANCE",
    "TWO_SIDED_95_T_DF31",
    "analyze_policy_realization",
    "analyze_policy_realization_rows",
    "expected_soft_policy_rng_seed",
]
