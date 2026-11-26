"""Immutable settings accessor for the R12 development revision."""

from __future__ import annotations

import json
from pathlib import Path


R12_MANIFEST_PATH = Path(__file__).with_name("method_manifest_r12.json")
R12_MANIFEST = json.loads(R12_MANIFEST_PATH.read_text())
if R12_MANIFEST.get("method_revision") != "tfsn_latent_detection_passage_r12_development":
    raise ValueError("R12 development manifest revision changed")
R12_GLOBAL = R12_MANIFEST["categories"]["global_computational"]
R12_COMMAND_STEPS = int(R12_GLOBAL["command_steps"])
R12_CANDIDATE_BUDGET = int(R12_GLOBAL["bridge_candidates"])
R12_LOOKAHEAD_DEPTH = int(R12_GLOBAL["consumptive_lookahead_depth"])
R12_LOOKAHEAD_BEAM = int(R12_GLOBAL["consumptive_lookahead_beam_width"])
