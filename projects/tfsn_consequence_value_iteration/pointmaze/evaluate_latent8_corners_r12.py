"""Eight-episode PointMaze corner extension for latent-radius memory."""

from pathlib import Path

from . import evaluate_factorial_r12 as evaluation


HERE = Path(__file__).resolve().parent


def main():
    evaluation.PROTOCOL = HERE / "latent8_corner_protocol_r12.json"
    evaluation.REFERENCE = HERE / "R12_LATENT8_CORNER_REFERENCE.md"
    evaluation.METHOD_REVISION = (
        "pointmaze_latent_radius_nearest_recorded_corner8_r12"
    )
    evaluation.CLAIM_LEVEL = "consumed_development_corner_extension_only"
    evaluation.REQUIRED_ARM = "latent_radius__nearest_recorded"
    evaluation.REQUIRED_COMMAND_CAP = 20
    evaluation.main()


if __name__ == "__main__":
    main()
