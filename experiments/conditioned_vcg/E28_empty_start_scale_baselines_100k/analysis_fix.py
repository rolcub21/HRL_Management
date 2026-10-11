"""Reporting-only repair for E28-v2's omitted VCG replicate field.

The frozen runner stored the VCG preference in each authenticated row but
omitted the redundant numeric replicate index. Reconstruct that index from
the authenticated coordinate filename for analysis, without editing a
ledger, checkpoint, contract, or rollout source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import (
    render_presentation,
    run,
)


ERRATUM_NAME = "e28-analysis-erratum.json"
LEDGER_NAME = re.compile(r"seed-(\d+)-rep-(\d+)\.json\Z")


def corrected_ledger(ledger: dict, path: Path, output: Path) -> tuple[dict, bool]:
    """Return a separate in-memory copy with the verified VCG coordinate."""
    row = ledger.get("row")
    if not isinstance(row, dict) or row.get("method") != run.VCG:
        return ledger, False
    match = LEDGER_NAME.fullmatch(path.name)
    if match is None:
        raise run.E28Error(f"unrecognized VCG ledger name: {path}")
    seed, replicate = map(int, match.groups())
    if replicate not in run.REPLICATES[run.VCG]:
        raise run.E28Error(f"VCG replicate outside frozen preference grid: {path}")
    expected = run._ledger_path(
        output, "confirmation", run.VCG, row["scenario_id"], seed, replicate
    )
    if (path.resolve() != expected.resolve()
        or row.get("instance_seed") != seed
        or row.get("panel") != "confirmation"
        or row.get("preference_lambda") != run.LAMBDAS[replicate]
        or row.get("model_seed") != 0):
        raise run.E28Error(f"VCG coordinate and saved λ disagree: {path}")
    if "replicate" in row:
        if row["replicate"] != replicate:
            raise run.E28Error(f"saved VCG replicate disagrees with coordinate: {path}")
        return ledger, False
    return {**ledger, "row": {**row, "replicate": replicate}}, True


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract, _ = run._contract(output)
    run._manifest(output, "confirmation", contract)
    native_read = run._read
    corrected_paths = []

    def read_with_redundant_coordinate(path, field):
        # native_read validates the saved JSON self-hash before any correction.
        ledger = native_read(path, field)
        if field != "ledger_sha256":
            return ledger
        repaired, changed = corrected_ledger(ledger, Path(path), output)
        if changed:
            corrected_paths.append(str(Path(path).relative_to(output)))
        return repaired

    # Replace only the rendering callable, not the module or its frozen source.
    # The original source-contract authentication therefore remains unchanged.
    with patch.object(run, "_read", side_effect=read_with_redundant_coordinate), \
         patch.object(run.render, "render", side_effect=render_presentation.render):
        result = run.analyze(output)
    if len(corrected_paths) != len(run.SCENARIOS) * len(run.SEEDS) * len(run.LAMBDAS):
        raise run.E28Error("expected to repair exactly the 90 frozen VCG ledger views")
    report = native_read(output / run.REPORT_NAME, "report_sha256")
    if report["rows"] != 270 or report["new_episode_runs"] != 270:
        raise run.E28Error("corrected analysis did not include all 270 episodes")
    erratum = run._hashed({
        "schema_version": 1,
        "protocol": run.PROTOCOL,
        "status": "analysis_only_correction",
        "original_contract_sha256": contract["contract_sha256"],
        "report_sha256": report["report_sha256"],
        "correction": (
            "VCG row omitted redundant replicate; derive index from authenticated "
            "seed-N-rep-K ledger path and assert saved lambda equals frozen grid[K]"
        ),
        "corrected_in_memory_rows": len(corrected_paths),
        "saved_ledgers_modified": False,
        "rollouts_rerun": 0,
        "analysis_fix_source_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "affected_ledger_paths": sorted(corrected_paths),
    }, "erratum_sha256")
    run._write(output / ERRATUM_NAME, erratum)
    return {
        **result,
        "analysis_erratum": str(output / ERRATUM_NAME),
        "corrected_in_memory_vcg_rows": len(corrected_paths),
        "saved_ledgers_modified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=run.OUTPUT)
    args = parser.parse_args(argv)
    result = analyze(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
