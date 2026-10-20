"""Pin E119's nominal replay, not a changed controller or another training run."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E119_tolerance_defer_comparison_8x8_183k_v1 import protocol as parent

ROOT = parent.ROOT
SOURCE = parent.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e120-accept-frontier-attribution-8x8-184k-v1"
PROTOCOL = "vcg_conditioned_e120_accept_frontier_attribution_8x8_184k_v1"
INSTANCE_SEEDS = parent.INSTANCE_SEEDS


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_hash(handle)


def _stream_hash(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def contract():
    parent_contract = json.loads((SOURCE / "contract.json").read_text())
    manifest = json.loads((SOURCE / "frozen-source.json").read_text())
    if manifest["contract_sha256"] != parent_contract["contract_sha256"]:
        raise RuntimeError("E120 E119 manifest contract differs")
    if sha256(SOURCE / "frozen-model.pth") != manifest["model_sha256"]:
        raise RuntimeError("E120 E119 frozen model differs")
    source_hashes = dict(parent_contract["source_sha256"])
    for name, expected in source_hashes.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"E120 E119 implementation changed: {name}")
    for path in Path(__file__).resolve().parent.iterdir():
        if path.suffix in (".py", ".sh"):
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    for name in ("PSLAP/viability_filter.py", "PSLAP/dynamic_yard.py",
                 "PSLAP/neutral_protocol.py", "example/Options/ExplicitAcceptOption.py"):
        source_hashes[name] = sha256(ROOT / name)
    inputs = {name: sha256(SOURCE / name) for name in ("contract.json", "frozen-source.json")}
    for seed in INSTANCE_SEEDS:
        name = f"rows/nominal_due/instance-{seed}.json"
        row = json.loads((SOURCE / name).read_text())
        if not row["strict_complete"] or not row["native_parity_passed"]:
            raise RuntimeError("E120 requires all five completed native-parity rows")
        inputs[name] = sha256(SOURCE / name)
    value = {
        "protocol": PROTOCOL, "source_contract_sha256": parent_contract["contract_sha256"],
        "source_model_sha256": manifest["model_sha256"], "source_episode": 120,
        "source_output": str(SOURCE.relative_to(ROOT)), "source_inputs_sha256": inputs,
        "source_sha256": source_hashes, "instance_seeds": list(INSTANCE_SEEDS),
        "controller": "unchanged nominal due cutoff, inbound Defer exclusion and guard",
        "parity": "every selected key, time, delivery error, aggregate cost and strict completion",
        "attribution": "physical mask -> executor -> certificate -> admissible ranker frontier",
        "wall_seconds_per_episode": 300, "training_runs": 0, "paper_evidence": False,
        "test_panel_opened": False, "scope": "one frozen model; five reused development instances",
    }
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**value, "contract_sha256": digest}
