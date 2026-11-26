"""Run the R52 preparation contract with the frozen E549 10M executor."""

from __future__ import annotations

from pathlib import Path

from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as implementation,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_10M_CHECKPOINT,
    E549_10M_CHECKPOINT_SHA256,
    E54910MSACExecutor,
    inspect_e549_10m_checkpoint,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
PROTOCOL = HERE / "r54_e549_10m_preparation_protocol.json"
REFERENCE = HERE / "R54_E549_10M_PREPARATION_REFERENCE.md"
METHOD_REVISION = "antmaze_e549_10m_preparation_r54"
CLAIM_LEVEL = "reward_free_executor_matched_preparation_only"
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_10m_preparation_r54"
)


def configure_implementation() -> None:
    implementation.PROTOCOL = PROTOCOL
    implementation.REFERENCE = REFERENCE
    implementation.METHOD_REVISION = METHOD_REVISION
    implementation.CLAIM_LEVEL = CLAIM_LEVEL
    implementation.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    implementation.E549_CHECKPOINT = E549_10M_CHECKPOINT
    implementation.E549_CHECKPOINT_SHA256 = E549_10M_CHECKPOINT_SHA256
    implementation.E549SACExecutor = E54910MSACExecutor
    implementation.inspect_e549_checkpoint = inspect_e549_10m_checkpoint
    implementation.PROTOCOL_VALUE = implementation.load_protocol()


configure_implementation()
sha256 = implementation.sha256
output_paths = implementation.output_paths


def configure_r52_runtime(output: Path):
    configure_implementation()
    return implementation.configure_r52_runtime(output)


def main() -> None:
    configure_implementation()
    args = implementation.parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
