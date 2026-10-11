"""Stable names and construction for PSLAP-family baselines."""

from PSLAP.dynamic_policy import DynamicPSLAPPolicy
from PSLAP.ga_policy import OfflinePSLAPGAPolicy, RollingPSLAPGAPolicy
from PSLAP.legacy import LegacyAdaptedPSLAPPolicy


DYNAMIC_PSLAP = "dynamic_pslap"
LEGACY_ADAPTED_PSLAP = "legacy_adapted_pslap"
PSLAP_GA_2009 = "pslap_ga_2009"  # historical alias for the offline reference
PSLAP_GA_2009_OFFLINE = "pslap_ga_2009_offline"
PSLAP_GA_2009_ROLLING = "pslap_ga_2009_rolling"

RUNNABLE_BASELINES = (
    DYNAMIC_PSLAP,
    PSLAP_GA_2009_OFFLINE,
    PSLAP_GA_2009_ROLLING,
    LEGACY_ADAPTED_PSLAP,
)
BASELINE_ALIASES = {
    "pslap": DYNAMIC_PSLAP,
    PSLAP_GA_2009: PSLAP_GA_2009_OFFLINE,
}
ACCEPTED_BASELINES = tuple(BASELINE_ALIASES) + RUNNABLE_BASELINES
BASELINE_STATUS = {
    DYNAMIC_PSLAP: "current",
    LEGACY_ADAPTED_PSLAP: "internal_ablation",
    PSLAP_GA_2009_OFFLINE: "information_advantaged_offline_reference",
    PSLAP_GA_2009_ROLLING: "online_reoptimization_baseline",
}
BASELINE_INFORMATION = {
    DYNAMIC_PSLAP: "online",
    PSLAP_GA_2009_OFFLINE: "offline_full_schedule",
    PSLAP_GA_2009_ROLLING: "online_arrived_only",
    LEGACY_ADAPTED_PSLAP: "internal_ablation",
}


def normalize_baseline(name: str) -> str:
    if name in BASELINE_ALIASES:
        return BASELINE_ALIASES[name]
    if name in RUNNABLE_BASELINES:
        return name
    raise ValueError(
        f"Unknown PSLAP baseline {name!r}; choose one of {ACCEPTED_BASELINES}"
    )


def make_policy(env, baseline: str):
    baseline = normalize_baseline(baseline)
    if baseline == DYNAMIC_PSLAP:
        return DynamicPSLAPPolicy(env)
    if baseline == PSLAP_GA_2009_OFFLINE:
        return OfflinePSLAPGAPolicy(env)
    if baseline == PSLAP_GA_2009_ROLLING:
        return RollingPSLAPGAPolicy(env)
    return LegacyAdaptedPSLAPPolicy(env)


__all__ = [
    "ACCEPTED_BASELINES",
    "BASELINE_ALIASES",
    "BASELINE_INFORMATION",
    "BASELINE_STATUS",
    "DYNAMIC_PSLAP",
    "LEGACY_ADAPTED_PSLAP",
    "PSLAP_GA_2009",
    "PSLAP_GA_2009_OFFLINE",
    "PSLAP_GA_2009_ROLLING",
    "RUNNABLE_BASELINES",
    "make_policy",
    "normalize_baseline",
]
