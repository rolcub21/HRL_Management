"""Eight-episode AntMaze extension of the latent-radius nearest-passage arm."""

from pathlib import Path

from . import r49_latent_passage_factorial as evaluation


HERE = Path(__file__).resolve().parent


def main():
    evaluation.PROTOCOL = HERE / "r50_latent8_miss_protocol.json"
    evaluation.REFERENCE = HERE / "R50_LATENT8_MISS_REFERENCE.md"
    evaluation.METHOD_REVISION = "antmaze_latent_radius_nearest_miss8_r50"
    evaluation.CLAIM_LEVEL = "consumed_development_miss_extension_only"
    evaluation.RUN_LABEL = "r50"
    evaluation.TASK_IDS = (0, 2)
    evaluation.REQUIRED_EPISODES = 8
    evaluation.REQUIRE_BASELINE_MISSES = True
    evaluation.FactorialAntPlanner.method_revision = evaluation.METHOD_REVISION
    evaluation.FactorialAntPlanner.planner_name = "r50_latent_nearest_ant_adapter"
    evaluation.main()


if __name__ == "__main__":
    main()
