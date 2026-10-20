"""Predeclare the gate and fresh development panels; no training authorization."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E121_blocked_pickup_wait_8x8_185k_v1 import protocol as parent

ROOT, SOURCE = parent.ROOT, parent.SOURCE
PROTOCOL = "vcg_conditioned_e122_joint_flow_contract_8x8_186k_v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e122-joint-flow-contract-8x8-186k-v1"
PANELS = {"pilot": (186990, 186991), "mechanism": tuple(range(186000, 186010))}
ARMS = {
    "A_outflow_bounded": {"joint_flow": False, "progress_only": False},
    "B_joint_bounded": {"joint_flow": True, "progress_only": False},
    "C_outflow_progress": {"joint_flow": False, "progress_only": True},
    "D_joint_progress": {"joint_flow": True, "progress_only": True},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integers(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, dict):
        return set().union(*(integers(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(integers(item) for item in value)) if value else set()
    return set()


def verify_freshness(payload):
    excluded = set()
    for key in ("train_instance_seeds", "validation_instance_seeds", "sealed_in_regime_test_seeds", "sealed_stress_test_seeds"):
        excluded.update(integers(payload.get(key, [])))
    planned = [seed for seeds in PANELS.values() for seed in seeds]
    if len(planned) != len(set(planned)) or set(planned).intersection(excluded):
        raise RuntimeError("E122 fresh panel overlaps source training/validation/test seeds")
    return True


@lru_cache(maxsize=1)
def contract():
    expected = parent.contract()
    actual = json.loads((parent.DEFAULT_OUTPUT / "contract.json").read_text())
    if expected != actual:
        raise RuntimeError("E122 source E121 implementation changed")
    hashes = dict(expected["source_sha256"])
    for name, wanted in hashes.items():
        if parent.attribution.sha256(ROOT / name) != wanted:
            raise RuntimeError(f"E122 inherited implementation changed: {name}")
    directory = Path(__file__).resolve().parent
    for path in directory.iterdir():
        if path.suffix in (".py", ".sh"):
            hashes[str(path.relative_to(ROOT))] = parent.attribution.sha256(path)
    for name in ("PSLAP/viability.py", "PSLAP/viability_candidates.py", "viability_graph_hierarchy.py",
                 "tests/test_vcg_e122_joint_flow.py"):
        hashes[name] = parent.attribution.sha256(ROOT / name)
    return {**(value := {
        "protocol": PROTOCOL, "architecture_name": "Joint-Flow VCG",
        "scope": "gated, frozen-controller fresh development mechanism comparison",
        "source_contract_sha256": actual["contract_sha256"],
        "source_model_sha256": actual["source_model_sha256"], "source_episode": 120,
        "model_seed": 0, "arms": ARMS, "panels": {key: list(value) for key, value in PANELS.items()},
        "objective": "J=2T+E+0.1L+10R; release-fixed due dates; +/-20 tolerance; gamma=1",
        "regime": "empty 8x8, 36 jobs, release gap 30, Poisson mean 80, same E118 generator",
        "inflow": "when usable free storage exists, pickup-to-free-cell and unladen approach remain possible",
        "full_yard": "no Accept; bounded wait/recovery until space; first freeing move must restore access",
        "joint_witness": "every macro-boundary prefix of the committed recovery witness satisfies joint access",
        "proof_boundary": "macro boundaries, not every primitive step and not arbitrary future arrival viability",
        "waiting": {"bounded": "E121 physically blocked pickup; release age <=80, nominal due cutoff retained",
                    "progress_only": "SAFE current state; <=10 steps per Defer; <=16 consecutive no-event Defers; no due/inbound/age preference"},
        "guard": "unchanged: after 2 nonprogress recovery decisions; due trigger disabled; Defer does not reset recovery counter",
        "certificate_caches": "fresh per episode, disjoint native/joint contracts",
        "recovery_shortcuts": "relocation-family disabled in all four arms; no candidate sampling",
        "wall_seconds_per_episode": 300, "max_steps": 3600,
        "primary": "paired J and completion; report every arm including failures",
        "secondary": "ET components, wait, storage/release MAE, work, rehandles, occupancy, blocked admission, latency, guard use",
        "training_runs": 0, "training_authorized": False, "paper_evidence": False, "test_panel_opened": False,
        "source_sha256": hashes,
    }), "contract_sha256": digest(value)}
