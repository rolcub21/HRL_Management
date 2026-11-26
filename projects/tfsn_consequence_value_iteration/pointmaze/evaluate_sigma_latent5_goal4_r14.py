"""Five-command fixed-sigma versus latent-radius four-goal screen."""

from pathlib import Path

from . import evaluate_factorial_r12 as evaluation


HERE = Path(__file__).resolve().parent
ARMS = (
    "sigma_kernel__nearest_recorded",
    "latent_radius__nearest_recorded",
)
GOALS = (
    "spatial_diagnostic_native_goal_r01_c06",
    "spatial_diagnostic_native_goal_r06_c06",
    "spatial_diagnostic_native_goal_r03_c03",
    "spatial_diagnostic_native_goal_r05_c03",
)


def main():
    evaluation.PROTOCOL = HERE / "sigma_latent5_goal4_protocol_r14.json"
    evaluation.REFERENCE = HERE / "R14_SIGMA_LATENT5_GOAL4_REFERENCE.md"
    evaluation.METHOD_REVISION = (
        "pointmaze_sigma_vs_latent_passage_gain_goal4_r14"
    )
    evaluation.CLAIM_LEVEL = (
        "consumed_development_truncated_horizon_memory_comparison_only"
    )
    evaluation.REQUIRED_ARM = None
    evaluation.ALLOWED_ARMS = ARMS
    evaluation.REQUIRED_COMMAND_CAP = 5
    evaluation.GOAL_NAMES = GOALS
    evaluation.main()


if __name__ == "__main__":
    main()
