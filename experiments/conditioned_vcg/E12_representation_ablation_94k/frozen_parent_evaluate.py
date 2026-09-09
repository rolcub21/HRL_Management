#!/usr/bin/env python3
"""Run E12 confirmation against the byte-frozen E11 parent artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    evaluate as core,
    program,
)
from methods.conditioned_vcg.representation_ablation import (
    REPRESENTATION_VARIANTS,
)


def authenticate_frozen_e11_parent(
    e12_output: Path, project_root: Path, e11_output: Path
) -> tuple[dict, dict]:
    """Authenticate saved E11 identity without rebuilding historical inputs."""
    expected_root = program.PROJECT_ROOT.resolve()
    expected_e11 = e11.DEFAULT_OUTPUT.resolve()
    if (
        project_root.resolve() != expected_root
        or e11_output.resolve() != expected_e11
    ):
        raise program.E12Error("unexpected E11 parent location")

    e12_contract, _development_manifest = program.authenticate(e12_output)
    contract = program.load_json(
        expected_e11 / e11.CONTRACT_NAME, label="frozen E11 contract"
    )
    manifest_path = expected_e11 / e11.MANIFEST_NAME
    manifest = program.load_json(manifest_path, label="frozen E11 manifest")
    e11._verify_hash(contract, "contract_sha256", label="frozen E11 contract")
    e11._verify_hash(manifest, "manifest_sha256", label="frozen E11 manifest")

    if (
        contract.get("protocol") != e11.PROTOCOL
        or manifest.get("protocol") != e11.PROTOCOL
        or manifest.get("contract_sha256") != contract.get("contract_sha256")
        or program.sha256(manifest_path) != e12_contract["e11_manifest_sha256"]
        or len(manifest.get("records", ()))
        != len(e11.REGIMES) * len(e11.INSTANCE_SEEDS)
    ):
        raise program.E12Error("frozen E11 parent binding changed")

    expected_coordinates = {
        (regime.regime_id, seed)
        for regime in e11.REGIMES
        for seed in e11.INSTANCE_SEEDS
    }
    observed_coordinates = {
        (record.get("regime_id"), int(record.get("seed", -1)))
        for record in manifest["records"]
    }
    if observed_coordinates != expected_coordinates:
        raise program.E12Error("frozen E11 coordinate panel changed")
    return contract, manifest


def run_confirmation(
    output_dir: Path,
    *,
    model_seeds: Sequence[int],
    variants: Sequence[str],
    regimes: Optional[Sequence[str]],
    device_name: str,
) -> dict:
    output_dir = output_dir.resolve()

    def frozen_auth(project_root: Path, e11_output: Path):
        return authenticate_frozen_e11_parent(
            output_dir, project_root, e11_output
        )

    with patch.object(core.e11, "authenticate", frozen_auth):
        return core.evaluate(
            output_dir,
            panel=core.CONFIRMATION_PANEL,
            model_seeds=tuple(model_seeds),
            variants=tuple(variants),
            regimes=None if regimes is None else tuple(regimes),
            device_name=device_name,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument(
        "--variant", choices=REPRESENTATION_VARIANTS, action="append"
    )
    parser.add_argument("--regime", action="append")
    args = parser.parse_args(argv)
    result = run_confirmation(
        args.output_dir,
        model_seeds=tuple(args.model_seed or program.MODEL_SEEDS),
        variants=tuple(args.variant or REPRESENTATION_VARIANTS),
        regimes=args.regime,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
