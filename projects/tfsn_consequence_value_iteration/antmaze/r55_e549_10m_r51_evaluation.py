"""Run the R53 evaluation contract with R54's E549-10M preparation."""

from __future__ import annotations

from pathlib import Path

from projects.tfsn_consequence_value_iteration.antmaze import (
    r53_e549_r51_evaluation as implementation,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r54_e549_10m_preparation as preparation,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_10M_CHECKPOINT,
    E549_10M_CHECKPOINT_SHA256,
    E54910MSACExecutor,
    inspect_e549_10m_checkpoint,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r55_e549_10m_r51_protocol.json"
REFERENCE = HERE / "R55_E549_10M_R51_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_10m_destination_arrival_r55"
EVALUATION_ARM = "R51_destination_arrival_with_E549_R54_10M_preparation"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_10m_destination_arrival_r55/stalled_pair"
)


def configure_implementation() -> None:
    preparation.configure_implementation()
    implementation.r52 = preparation
    implementation.PROTOCOL = PROTOCOL
    implementation.REFERENCE = REFERENCE
    implementation.METHOD_REVISION = METHOD_REVISION
    implementation.EVALUATION_ARM = EVALUATION_ARM
    implementation.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    implementation.E549_CHECKPOINT = E549_10M_CHECKPOINT
    implementation.E549_CHECKPOINT_SHA256 = E549_10M_CHECKPOINT_SHA256
    implementation.E549SACExecutor = E54910MSACExecutor
    implementation.inspect_e549_checkpoint = inspect_e549_10m_checkpoint
    implementation.E549DestinationArrivalPlanner.method_revision = METHOD_REVISION
    implementation.E549DestinationArrivalPlanner.planner_name = EVALUATION_ARM
    implementation.PROTOCOL_VALUE = implementation.load_protocol()


configure_implementation()


def main() -> None:
    configure_implementation()
    args = implementation.parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
