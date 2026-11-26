"""Isolated five-command PointMaze latent-memory behavior check."""

from pathlib import Path

from . import evaluate_factorial_r12 as evaluation


HERE = Path(__file__).resolve().parent


def main():
    evaluation.PROTOCOL = HERE / "latent5_protocol_r12.json"
    evaluation.REFERENCE = HERE / "R12_LATENT5_SMOKE_REFERENCE.md"
    evaluation.METHOD_REVISION = (
        "pointmaze_latent_radius_nearest_recorded_five_command_smoke_r12"
    )
    evaluation.CLAIM_LEVEL = "exploratory_truncated_horizon_smoke_only"
    evaluation.REQUIRED_ARM = "latent_radius__nearest_recorded"
    evaluation.REQUIRED_COMMAND_CAP = 5
    evaluation.main()


if __name__ == "__main__":
    main()
