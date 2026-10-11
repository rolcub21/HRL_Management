"""Stable public import surface for the final conditioned VCG controller.

The authenticated implementation remains in ``vcg_v11_conditioned_handling``
because completed training and final-evaluation contracts bind that exact path
and byte hash. This module gives new code a clean package import without
mutating the frozen evidence source.
"""

from vcg_v11_conditioned_handling import (
    CHECKPOINT_SCHEMA_VERSION,
    CONTROLLER_ARCHITECTURE,
    NETWORK_ARCHITECTURE,
    TARGET_CONTRACT,
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
    ConditionedHandlingConfig,
    FutureHandlingSample,
    fit_conditioned_future_handling,
    future_samples_from_episode,
    handling_calibration,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONTROLLER_ARCHITECTURE",
    "NETWORK_ARCHITECTURE",
    "TARGET_CONTRACT",
    "ConditionedFutureHandlingNetwork",
    "ConditionedHandlingAgent",
    "ConditionedHandlingConfig",
    "FutureHandlingSample",
    "fit_conditioned_future_handling",
    "future_samples_from_episode",
    "handling_calibration",
]
