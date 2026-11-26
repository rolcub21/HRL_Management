"""Declared global settings shared by all continuous TF-SN realizations."""

from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("method_manifest.json")
METHOD_MANIFEST = json.loads(MANIFEST_PATH.read_text())
if METHOD_MANIFEST.get("schema_version") != 1:
    raise ValueError("unsupported TF-SN method manifest")
GLOBAL = METHOD_MANIFEST["categories"]["global_computational"]


COMMAND_STEPS = int(GLOBAL["command_steps"])
MIXTURE_COMPONENTS = int(GLOBAL["mixture_components"])
ENSEMBLE_SIZE = int(GLOBAL["ensemble_size"])
HIDDEN_DIMENSION = int(GLOBAL["hidden_dimension"])
HIDDEN_LAYERS = int(GLOBAL["hidden_layers"])
BATCH_SIZE = int(GLOBAL["batch_size"])
EPOCHS = int(GLOBAL["epochs"])
LEARNING_RATE = float(GLOBAL["learning_rate"])
VALIDATION_FRACTION = float(GLOBAL["validation_fraction"])
STD_FLOOR = float(GLOBAL["std_floor"])
BRIDGE_CANDIDATES = int(GLOBAL["bridge_candidates"])
MAX_ALIGNMENT_ROUNDS = int(GLOBAL["maximum_alignment_rounds"])
ALIGNMENT_RELATIVE_TOLERANCE = float(GLOBAL["alignment_relative_tolerance"])
ALIGNMENT_TASKS_PER_PHASE = int(GLOBAL["alignment_tasks_per_phase"])
ALIGNMENT_COMMANDS_PER_TASK = int(GLOBAL["alignment_commands_per_task"])
GAUSSIAN_CIRCLE_QUADRATURE_POINTS = int(
    GLOBAL["gaussian_circle_quadrature_points"]
)
PASSAGE_BINS = int(GLOBAL["passage_bins"])
CONSUMPTIVE_LOOKAHEAD_DEPTH = int(GLOBAL["consumptive_lookahead_depth"])
CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH = int(
    GLOBAL["consumptive_lookahead_beam_width"]
)
RECORDED_RELIABILITY_NEIGHBORS = int(
    GLOBAL["recorded_reliability_neighbors"]
)
RECORDED_EVIDENCE_PRIOR_WEIGHT = float(
    GLOBAL["recorded_evidence_prior_weight"]
)
SITUATION_DENSITY_TRUNCATION_SIGMAS = float(
    GLOBAL["situation_density_truncation_sigmas"]
)
PAIR_INFERENCE_BATCH = int(GLOBAL["pair_inference_batch"])
ARTIFACT_NODE_SHARD = int(GLOBAL["artifact_node_shard"])
FIELD_EVALUATION_BATCH = int(GLOBAL["field_evaluation_batch"])


def finite_horizon(episode_primitive_steps):
    """Derive the command horizon from task time and the shared command time."""
    steps = int(episode_primitive_steps)
    if steps <= 0 or steps % COMMAND_STEPS:
        raise ValueError("episode length must be positive and divisible by tau")
    return steps // COMMAND_STEPS

