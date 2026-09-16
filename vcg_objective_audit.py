#!/usr/bin/env python3
"""Versioned timing-objective arms for the VCG objective audit.

This module is deliberately isolated from :mod:`example.small_rooms_env`.
``SmallRoomsEnv`` remains the legacy environment used by VCG-v1.  The
subclass below changes only the delivery reward and emits a dual-objective
component record on every primitive step, making a fixed trajectory exactly
rescorable under either audit arm.

The dense arm is

    B - lambda_abs * |e| - lambda_outside * (|e| - window)_+.

The screen defaults ``B=40``, ``lambda_abs=1.5``,
``lambda_outside=0.5``, and ``window=20`` coincide with the legacy triangular
delivery reward throughout the target window.  Unlike the legacy reward,
they retain a nonzero gradient outside the window.  Dense rewards are not
clipped and can therefore be negative for sufficiently large timing errors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Mapping, Optional, Sequence

from example.small_rooms_env import SmallRoomsEnv


TIMING_OBJECTIVE_SCHEMA_VERSION = 1
TIMING_OBJECTIVE_CONTRACT = "dual_rescorable_timing_objective_components_v1"
LEGACY_CLIPPED = "legacy_clipped_v1"
DENSE_PIECEWISE = "dense_piecewise_v1"
TIMING_OBJECTIVES = (LEGACY_CLIPPED, DENSE_PIECEWISE)
TIMING_OBJECTIVE_INFO_KEY = "timing_objective_components"

LEGACY_DELIVERY_BASE = 10.0
LEGACY_MAX_TIMING_BONUS = 30.0
LEGACY_TARGET_WINDOW = 20.0
DEFAULT_DENSE_B = 40.0
DEFAULT_LAMBDA_ABS = 1.5
DEFAULT_LAMBDA_OUTSIDE = 0.5
DEFAULT_DENSE_WINDOW = 20.0
DEFAULT_STEP_COST = -0.09
DEFAULT_PLACEMENT_REWARD = 5.0

_OBJECTIVE_ALIASES = {
    "legacy": LEGACY_CLIPPED,
    "legacy_clipped": LEGACY_CLIPPED,
    LEGACY_CLIPPED: LEGACY_CLIPPED,
    "dense": DENSE_PIECEWISE,
    "dense_piecewise": DENSE_PIECEWISE,
    DENSE_PIECEWISE: DENSE_PIECEWISE,
}


def canonical_timing_objective(value: str) -> str:
    """Normalize a CLI-friendly objective name to its versioned identity."""

    try:
        return _OBJECTIVE_ALIASES[str(value).strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown timing objective {value!r}; expected legacy or dense"
        ) from error


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class TimingObjectiveSpec:
    """Serializable contract for one objective-audit environment arm.

    The legacy coefficients are recorded explicitly so every artifact is
    self-describing.  ``exact_legacy`` is true only for the unmodified
    ``SmallRoomsEnv`` profile.  The dense coefficients remain fully explicit
    and configurable for screened and confirmatory runs.
    """

    objective: str
    dense_b: float = DEFAULT_DENSE_B
    lambda_abs: float = DEFAULT_LAMBDA_ABS
    lambda_outside: float = DEFAULT_LAMBDA_OUTSIDE
    window: float = DEFAULT_DENSE_WINDOW
    legacy_base: float = LEGACY_DELIVERY_BASE
    legacy_max_bonus: float = LEGACY_MAX_TIMING_BONUS
    legacy_window: float = LEGACY_TARGET_WINDOW
    schema_version: int = field(
        default=TIMING_OBJECTIVE_SCHEMA_VERSION, init=False
    )
    contract: str = field(default=TIMING_OBJECTIVE_CONTRACT, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "objective", canonical_timing_objective(self.objective)
        )
        for name in (
            "dense_b",
            "lambda_abs",
            "lambda_outside",
            "window",
            "legacy_base",
            "legacy_max_bonus",
            "legacy_window",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name=name)
            )
        if self.lambda_abs < 0.0 or self.lambda_outside < 0.0:
            raise ValueError("dense timing slopes must be non-negative")
        if self.window <= 0.0 or self.legacy_window <= 0.0:
            raise ValueError("timing windows must be positive")
        if self.legacy_max_bonus < 0.0:
            raise ValueError("legacy_max_bonus must be non-negative")

    @property
    def exact_legacy(self) -> bool:
        """Whether the recorded legacy profile exactly matches VCG-v1."""

        return bool(
            self.legacy_base == LEGACY_DELIVERY_BASE
            and self.legacy_max_bonus == LEGACY_MAX_TIMING_BONUS
            and self.legacy_window == LEGACY_TARGET_WINDOW
        )

    @property
    def dense_matches_legacy_inside_window(self) -> bool:
        """Whether dense and legacy rewards agree for ``|e| <= 20``."""

        return bool(
            self.exact_legacy
            and self.window == self.legacy_window
            and self.dense_b
            == self.legacy_base + self.legacy_max_bonus
            and self.lambda_abs
            == self.legacy_max_bonus / self.legacy_window
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["exact_legacy"] = self.exact_legacy
        result["dense_matches_legacy_inside_window"] = (
            self.dense_matches_legacy_inside_window
        )
        return result

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TimingObjectiveSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("timing objective spec must be a mapping")
        schema = payload.get(
            "schema_version", TIMING_OBJECTIVE_SCHEMA_VERSION
        )
        if int(schema) != TIMING_OBJECTIVE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported timing objective schema version: {schema}"
            )
        contract = payload.get("contract", TIMING_OBJECTIVE_CONTRACT)
        if str(contract) != TIMING_OBJECTIVE_CONTRACT:
            raise ValueError(f"unsupported timing objective contract: {contract}")
        names = (
            "objective",
            "dense_b",
            "lambda_abs",
            "lambda_outside",
            "window",
            "legacy_base",
            "legacy_max_bonus",
            "legacy_window",
        )
        values = {name: payload[name] for name in names if name in payload}
        if "objective" not in values:
            raise ValueError("timing objective spec is missing objective")
        return cls(**values)

    @classmethod
    def legacy(cls, **dense_counterfactual_coefficients):
        """Construct the exact legacy arm with optional dense rescore values."""

        return cls(
            objective=LEGACY_CLIPPED,
            **dense_counterfactual_coefficients,
        )

    @classmethod
    def dense(
        cls,
        *,
        dense_b: float = DEFAULT_DENSE_B,
        lambda_abs: float = DEFAULT_LAMBDA_ABS,
        lambda_outside: float = DEFAULT_LAMBDA_OUTSIDE,
        window: float = DEFAULT_DENSE_WINDOW,
    ):
        """Construct a fully recorded dense piecewise arm."""

        return cls(
            objective=DENSE_PIECEWISE,
            dense_b=dense_b,
            lambda_abs=lambda_abs,
            lambda_outside=lambda_outside,
            window=window,
        )


@dataclass(frozen=True)
class DeliveryRewardComponents:
    """One delivery's primitive terms under both timing objectives."""

    signed_error: float
    absolute_error: float
    dense_excess_error: float
    legacy_timing_bonus: float
    legacy_delivery_reward: float
    dense_absolute_penalty: float
    dense_outside_penalty: float
    dense_delivery_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


def delivery_reward_components(
    error: float,
    spec: TimingObjectiveSpec,
) -> DeliveryRewardComponents:
    """Compute transparent delivery components under both audit arms."""

    if not isinstance(spec, TimingObjectiveSpec):
        raise TypeError("spec must be a TimingObjectiveSpec")
    error = _finite(error, name="delivery error")
    absolute = abs(error)
    legacy_bonus = max(
        0.0,
        spec.legacy_max_bonus
        * (1.0 - absolute / spec.legacy_window),
    )
    excess = max(0.0, absolute - spec.window)
    dense_absolute_penalty = spec.lambda_abs * absolute
    dense_outside_penalty = spec.lambda_outside * excess
    return DeliveryRewardComponents(
        signed_error=error,
        absolute_error=absolute,
        dense_excess_error=excess,
        legacy_timing_bonus=legacy_bonus,
        legacy_delivery_reward=spec.legacy_base + legacy_bonus,
        dense_absolute_penalty=dense_absolute_penalty,
        dense_outside_penalty=dense_outside_penalty,
        dense_delivery_reward=(
            spec.dense_b
            - dense_absolute_penalty
            - dense_outside_penalty
        ),
    )


def delivery_reward(
    error: float,
    spec: TimingObjectiveSpec,
    objective: Optional[str] = None,
) -> float:
    """Return one delivery reward for the selected or named objective."""

    objective = canonical_timing_objective(objective or spec.objective)
    components = delivery_reward_components(error, spec)
    if objective == LEGACY_CLIPPED:
        return components.legacy_delivery_reward
    return components.dense_delivery_reward


def _component_record(value: Mapping) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError("objective component record must be a mapping")
    record = value.get(TIMING_OBJECTIVE_INFO_KEY, value)
    if not isinstance(record, Mapping):
        raise TypeError("timing objective info entry must be a mapping")
    if int(record.get("schema_version", -1)) != TIMING_OBJECTIVE_SCHEMA_VERSION:
        raise ValueError("timing objective component schema mismatch")
    if record.get("contract") != TIMING_OBJECTIVE_CONTRACT:
        raise ValueError("timing objective component contract mismatch")
    return record


def rescore_step_reward(value: Mapping, objective: str) -> float:
    """Read an already logged step reward for either objective arm."""

    record = _component_record(value)
    objective = canonical_timing_objective(objective)
    try:
        result = float(record["step_reward_by_objective"][objective])
    except (KeyError, TypeError) as error:
        raise ValueError("component record lacks dual step rewards") from error
    return _finite(result, name="rescored step reward")


def rescore_step_reward_with_spec(
    value: Mapping,
    spec: TimingObjectiveSpec,
    objective: Optional[str] = None,
) -> float:
    """Recompute a logged step under a new explicit coefficient contract."""

    record = _component_record(value)
    non_delivery = _finite(
        record["non_delivery_step_reward"],
        name="non-delivery step reward",
    )
    if not bool(record["delivery_event"]):
        return non_delivery
    delivery = record.get("delivery")
    if not isinstance(delivery, Mapping) or "signed_error" not in delivery:
        raise ValueError("delivery component record lacks signed error")
    return non_delivery + delivery_reward(
        float(delivery["signed_error"]), spec, objective
    )


def rescore_logged_return(
    records: Iterable[Mapping], objective: str
) -> float:
    """Sum recorded primitive-step rewards for one objective arm."""

    return float(sum(rescore_step_reward(item, objective) for item in records))


def rescore_logged_return_with_spec(
    records: Iterable[Mapping],
    spec: TimingObjectiveSpec,
    objective: Optional[str] = None,
) -> float:
    """Rescore a trajectory under arbitrary explicit dense coefficients."""

    return float(
        sum(
            rescore_step_reward_with_spec(item, spec, objective)
            for item in records
        )
    )


def rescore_episode_return(
    delivery_deviations: Sequence[float],
    steps: int,
    delivery_count: int,
    placement_count: int,
    spec: TimingObjectiveSpec,
    objective: Optional[str] = None,
    *,
    step_cost: float = DEFAULT_STEP_COST,
    placement_reward: float = DEFAULT_PLACEMENT_REWARD,
    other_reward: float = 0.0,
) -> float:
    """Rescore an episode from sufficient aggregate environment components.

    This exact aggregate form applies to the current ``SmallRoomsEnv`` reward:
    primitive step cost, one placement reward per observed storage event, and
    one delivery reward per deviation.  ``other_reward`` is explicit so a
    future environment cannot silently omit a new reward component.
    """

    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 0:
        raise ValueError("steps must be a non-negative integer")
    if (
        isinstance(delivery_count, bool)
        or int(delivery_count) != delivery_count
        or int(delivery_count) < 0
    ):
        raise ValueError("delivery_count must be a non-negative integer")
    if (
        isinstance(placement_count, bool)
        or int(placement_count) != placement_count
        or int(placement_count) < 0
    ):
        raise ValueError("placement_count must be a non-negative integer")
    deviations = tuple(
        _finite(value, name="delivery deviation")
        for value in delivery_deviations
    )
    if len(deviations) != int(delivery_count):
        raise ValueError(
            "delivery_count must equal the number of delivery deviations"
        )
    step_cost = _finite(step_cost, name="step_cost")
    placement_reward = _finite(
        placement_reward, name="placement_reward"
    )
    other_reward = _finite(other_reward, name="other_reward")
    return float(
        int(steps) * step_cost
        + int(placement_count) * placement_reward
        + sum(delivery_reward(error, spec, objective) for error in deviations)
        + other_reward
    )


class ObjectiveAuditSmallRoomsEnv(SmallRoomsEnv):
    """Isolated ``SmallRoomsEnv`` with a selected, fully logged objective.

    No V1 class or global constant is mutated.  ``info`` contains
    ``TIMING_OBJECTIVE_INFO_KEY`` on every primitive step.  The returned reward
    is exactly the selected arm's entry in ``step_reward_by_objective``.
    """

    def __init__(
        self,
        *,
        timing_objective: TimingObjectiveSpec,
        **environment_kwargs,
    ) -> None:
        if not isinstance(timing_objective, TimingObjectiveSpec):
            raise TypeError(
                "timing_objective must be a TimingObjectiveSpec"
            )
        self.timing_objective_spec = timing_objective
        self._pending_delivery_components: Optional[
            DeliveryRewardComponents
        ] = None
        self._objective_audit_records: list[dict] = []
        self._objective_audit_placement_count = 0
        self._objective_audit_delivery_count = 0
        super().__init__(**environment_kwargs)
        # Instance attributes keep existing evaluation helpers consistent with
        # the selected experiment's declared target window, without modifying
        # SmallRoomsEnv's class-level V1 constants.
        self.DELIVERY_TARGET_WINDOW = float(
            timing_objective.legacy_window
            if timing_objective.objective == LEGACY_CLIPPED
            else timing_objective.window
        )

    def reset(self, instance=None):
        state = super().reset(instance=instance)
        self._pending_delivery_components = None
        self._objective_audit_records.clear()
        self._objective_audit_placement_count = 0
        self._objective_audit_delivery_count = 0
        return state

    def _delivery_reward(self, storage_loc, error_time):
        # ``storage_loc`` is retained for signature compatibility.  Neither V1
        # nor the dense audit arm currently uses a distance term.
        del storage_loc
        components = delivery_reward_components(
            error_time, self.timing_objective_spec
        )
        self._pending_delivery_components = components
        if self.timing_objective_spec.objective == LEGACY_CLIPPED:
            return components.legacy_delivery_reward
        return components.dense_delivery_reward

    def step(self, action):
        self._pending_delivery_components = None
        next_state, selected_reward, terminal, info = super().step(action)
        info = dict(info)
        selected_reward = float(selected_reward)
        delivery_event = bool(info.get("delivered_block"))
        components = self._pending_delivery_components
        if delivery_event != (components is not None):
            raise RuntimeError(
                "delivery event and timing reward components disagree"
            )

        if components is None:
            non_delivery_reward = selected_reward
            delivery_record = None
            legacy_step_reward = selected_reward
            dense_step_reward = selected_reward
        else:
            selected_delivery_reward = (
                components.legacy_delivery_reward
                if self.timing_objective_spec.objective == LEGACY_CLIPPED
                else components.dense_delivery_reward
            )
            non_delivery_reward = selected_reward - selected_delivery_reward
            delivery_record = components.to_dict()
            legacy_step_reward = (
                non_delivery_reward + components.legacy_delivery_reward
            )
            dense_step_reward = (
                non_delivery_reward + components.dense_delivery_reward
            )
            # Preserve the established key, now explicitly selected-arm data.
            info["delivery_reward"] = selected_delivery_reward
            self._objective_audit_delivery_count += 1

        if bool(info.get("stored_block")):
            self._objective_audit_placement_count += 1
        step_rewards = {
            LEGACY_CLIPPED: float(legacy_step_reward),
            DENSE_PIECEWISE: float(dense_step_reward),
        }
        record = {
            "schema_version": TIMING_OBJECTIVE_SCHEMA_VERSION,
            "contract": TIMING_OBJECTIVE_CONTRACT,
            "selected_objective": self.timing_objective_spec.objective,
            "selected_step_reward": selected_reward,
            "delivery_event": delivery_event,
            "non_delivery_step_reward": float(non_delivery_reward),
            "delivery": delivery_record,
            "step_reward_by_objective": step_rewards,
            "spec": self.timing_objective_spec.to_dict(),
        }
        if not math.isclose(
            selected_reward,
            step_rewards[self.timing_objective_spec.objective],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("selected timing objective reward mismatch")
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records.append(record)
        return next_state, selected_reward, terminal, info

    @property
    def objective_audit_records(self) -> tuple[dict, ...]:
        """Immutable view of primitive-step component records this episode."""

        return tuple(self._objective_audit_records)

    def objective_audit_summary(self) -> dict:
        """Return exact dual totals and observed event counts."""

        return {
            "schema_version": TIMING_OBJECTIVE_SCHEMA_VERSION,
            "contract": TIMING_OBJECTIVE_CONTRACT,
            "spec": self.timing_objective_spec.to_dict(),
            "steps": len(self._objective_audit_records),
            "placement_count": self._objective_audit_placement_count,
            "delivery_count": self._objective_audit_delivery_count,
            "return_by_objective": {
                objective: rescore_logged_return(
                    self._objective_audit_records, objective
                )
                for objective in TIMING_OBJECTIVES
            },
        }


__all__ = [
    "DEFAULT_DENSE_B",
    "DEFAULT_DENSE_WINDOW",
    "DEFAULT_LAMBDA_ABS",
    "DEFAULT_LAMBDA_OUTSIDE",
    "DENSE_PIECEWISE",
    "DeliveryRewardComponents",
    "LEGACY_CLIPPED",
    "ObjectiveAuditSmallRoomsEnv",
    "TIMING_OBJECTIVES",
    "TIMING_OBJECTIVE_CONTRACT",
    "TIMING_OBJECTIVE_INFO_KEY",
    "TIMING_OBJECTIVE_SCHEMA_VERSION",
    "TimingObjectiveSpec",
    "canonical_timing_objective",
    "delivery_reward",
    "delivery_reward_components",
    "rescore_episode_return",
    "rescore_logged_return",
    "rescore_logged_return_with_spec",
    "rescore_step_reward",
    "rescore_step_reward_with_spec",
]
