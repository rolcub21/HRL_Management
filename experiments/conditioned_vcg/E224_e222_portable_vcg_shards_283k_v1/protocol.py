"""Pinned identities for portable E222 VCG seed-1/2 compute shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import (
    protocol as e222,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e224_e222_portable_vcg_shards_283k_v1"
ROW_PROTOCOL = e222.PROTOCOL
MODEL_SEEDS = (1, 2)
ANCHOR_CONTRACT_SHA256 = (
    "4de8015958ff78bb3e10dfe00f3cdc12f18a4a62821dfd4be17a350b28e51773"
)
ANCHOR_INSTANCE_MANIFEST_SHA256 = (
    "06b75b0765169e042c69093eab72b05af8eac0d6010bc16f0a9fa75c911f3a9a"
)
E218_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e218-e215-long-exploration-replication-277k-v1"
)
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e224-e222-portable-vcg-shards-283k-v1"
)

# These hashes are copied from the E222 contract before the protected panel was
# opened.  E224 is a compute relocation of that contract, not a new panel.
SOURCE_SHA256 = {
    "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py": "0aeae72eb32074f8be503bb6f46b532a5d0764837d91d72a3389dd0dba154c7b",
    "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py": "4d20a171f9213460ff8354dcdcc8c14743bf9a4767ebb2bedd96dc467d88c85f",
    "experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/roundoff.py": "7d0518c0f20c2de4f9abe66e059ebebe374a6ccff1830f21941e380f30cdccfe",
    "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/instrumentation.py": "f786e83ff69dbd722354980acdc21702f43299e4be698c7c1f10aee82b19b669",
    "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py": "db205f1be6c82451b9c74de8358f327887712914d45befc044d0edcb9a25e3b3",
    "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py": "1e0a65f0f403c6be4c74221222bd7bd2f9dddfc68964e02f3c717d627dded79a",
    "experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.py": "890ee66755bedc76a3698eeb5dded32b820ee8045f844debb9101d3d3cf5ed3f",
    "experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/plot.py": "3b0b2876e3025baf503b07780f98d21d84df7c76e522dcaae27170d56293335d",
    "experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/run.py": "74e2575c3115c1ee3fb0731b67c1e79bf99882c9ffa91c560e3161fe4d5a3add",
    "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/README.md": "7ed7d68e8c07a75488559c62d824589c18b1e456335026c35d43be847a02da53",
    "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/plot.py": "55ab1f2eebf0513ca2c5d6fe72a0c5dfc9cfe85a74535111387e3e626b344f35",
    "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/protocol.py": "ad8c03fcc222a3581c27f15ebc6ecb0af9c4958768f7a7ed775fc5d6f0bc7d04",
    "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.py": "422c0ea686dd1a2ff5c4efd5afb20f78bf7e2860e79e1c056879bb949cb8659e",
    "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh": "07f2623d54cee8cb0ed1e8455720ccf00752e83553927bfbfdb87356f2e46b61",
    "experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/scenario.py": "1085ea5c73ea368c4d079bcf56267a9ccedb4740f95b8e43a2b55024622fbe83",
    "experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.py": "1717e2ed782ba7e128d68b697e2ebe115543a55d42b10d41e750fc69372a8cfe",
    "projects/temporal_consequence_realization/METHOD_REFERENCE.md": "78c4356f2c424366a7167fa9f6b15f0d75f1c32a30382da294cbfa65d17e9a26",
    "projects/temporal_consequence_realization/method_conformance.py": "6ff8f4ab7bc3b40c949ae2d7ff83d899b82c1560505f7120f313bf9ec3ee10d3",
}

CHECKPOINTS = {
    1: {
        "selected_episode": 280,
        "checkpoint_sha256": "9b0f4950e49ad2480e4adcd365bc5f3985e3eaaa91b255e768a66f75ba0e3713",
        "summary_sha256": "d626d72d801df3ec3b7b600567b43d93d61bd367e28b941cd2c8397f6b7efade",
    },
    2: {
        "selected_episode": 220,
        "checkpoint_sha256": "77e8ce5cd1959205559d427ace80e428b9be400a9fbd47ca3df8d9e21359a189",
        "summary_sha256": "b2d3d0fa7f3af89fcc000ed95e326abc7e1d81e1594bb00af378380e38429bbd",
    },
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def checkpoint_paths(seed: int) -> tuple[Path, Path]:
    if int(seed) not in MODEL_SEEDS:
        raise ValueError("E224 model seed must be 1 or 2")
    directory = (
        E218_OUTPUT / "training/operational" / e191.SCENARIO / f"seed-{int(seed)}"
    )
    return directory / "training-summary.json", directory / "best.pth"

