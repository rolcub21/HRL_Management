#!/usr/bin/env python3
"""Rerun E239's simple dial with the three corrected-temperature controllers."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import representation as e195_representation
from experiments.conditioned_vcg.E201_e195_matched_5x5_test_panel_260k_v1 import scenario
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import run as e204
from experiments.conditioned_vcg.E239_e195_simple_dial_curve_298k_v1 import run as core

from . import protocol as p


def _read(path: Path) -> dict:
    return core._read(path)


def _load_base(contract: dict, model_seed: int, device):
    frozen = next(row for row in contract["source"]["frozen_stage_a"] if int(row["model_seed"]) == int(model_seed))
    checkpoint = Path(frozen["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expected = {
        "training_protocol": frozen["training_protocol"],
        "model_seed": int(model_seed),
        "selected_checkpoint_episode": int(frozen["selected_episode"]),
        "completed_training_episodes": 300,
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    core.require(not mismatch, f"E262 corrected checkpoint identity differs: {mismatch}")
    base = e195_representation.FreshScaleMatched5x5Agent.from_checkpoint(
        payload, device=device, resumable=False, seed=int(model_seed)
    )
    base.set_epsilon(0.0)
    for network in (base.Q_local, base.Q_target):
        network.requires_grad_(False).eval()
    search = e204._search(payload)
    liveness = e204._liveness(payload)
    base_digest = e204.state_digest(base.Q_local.state_dict())
    return base, search, liveness, base_digest, frozen


def _source_vcg_row(model_seed: int, instance_seed: int) -> dict:
    return _read(p.current_panel_row(model_seed, instance_seed))


def _all_rows(output: Path, contract: dict) -> list[dict]:
    rows = []
    for model_seed in p.MODEL_SEEDS:
        for instance_seed in scenario.TEST_SEEDS:
            row = dict(_source_vcg_row(model_seed, instance_seed))
            row["method"] = "e195"
            rows.append(core._flat_source(row))
    for relative in sorted(contract["source"]["e201_rows"]):
        row = _read(p.SOURCE_OUTPUT / relative)
        if row.get("method") in ("dynamic_pslap", "ga"):
            rows.append(core._flat_source(row))
    for value in p.NEW_LAMBDAS:
        for model_seed in p.MODEL_SEEDS:
            for instance_seed in scenario.TEST_SEEDS:
                path = core._row_path(output, model_seed, instance_seed, value)
                if path.is_file():
                    rows.append(core._flat_new(_read(path)))
    return rows


def main() -> None:
    with ExitStack() as stack:
        stack.enter_context(patch.object(core, "p", p))
        stack.enter_context(patch.object(core, "_load_base", _load_base))
        stack.enter_context(patch.object(core, "_source_vcg_row", _source_vcg_row))
        stack.enter_context(patch.object(core, "_all_rows", _all_rows))
        core.main()


if __name__ == "__main__":
    main()
