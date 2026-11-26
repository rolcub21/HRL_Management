"""Eight-episode PointMaze command-region development screen."""

from pathlib import Path

from . import evaluate_factorial_r12 as evaluation
from .command_region_planner_r13 import (
    REGION_MANIFEST,
    load_command_region_planner,
)


HERE = Path(__file__).resolve().parent


def main():
    evaluation.PROTOCOL = HERE / "command_region_protocol_r13.json"
    evaluation.REFERENCE = HERE / "R13_COMMAND_REGION_REFERENCE.md"
    evaluation.METHOD_REVISION = "pointmaze_command_region_acquisition_r13"
    evaluation.CLAIM_LEVEL = "consumed_development_command_region_screen_only"
    evaluation.REQUIRED_ARM = "latent_radius__command_region"
    evaluation.REQUIRED_COMMAND_CAP = 20
    evaluation.PLANNER_LOADER = load_command_region_planner
    evaluation.METHOD_MANIFEST = REGION_MANIFEST
    evaluation.main()


if __name__ == "__main__":
    main()
