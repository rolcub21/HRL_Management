"""Require completed exact-parity attribution before changing action availability."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E120_accept_frontier_attribution_8x8_184k_v1 import protocol as attribution

ROOT = attribution.ROOT
SOURCE = attribution.SOURCE
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e121-blocked-pickup-wait-8x8-185k-v1"
PROTOCOL = "vcg_conditioned_e121_blocked_pickup_wait_8x8_185k_v1"
INSTANCE_SEEDS = attribution.INSTANCE_SEEDS


@lru_cache(maxsize=1)
def contract():
    parent = attribution.contract()
    report_path = attribution.DEFAULT_OUTPUT / "e120-report.json"
    report = json.loads(report_path.read_text())
    if (report["status"] != "complete" or report["exact_parity_rows"] != 5
            or report["contract_sha256"] != parent["contract_sha256"]):
        raise RuntimeError("E121 requires completed, authenticated E120 attribution")
    hashes = dict(parent["source_sha256"])
    for path in Path(__file__).resolve().parent.iterdir():
        if path.suffix in (".py", ".sh"):
            hashes[str(path.relative_to(ROOT))] = attribution.sha256(path)
    value = {
        "protocol": PROTOCOL, "source_attribution_contract_sha256": parent["contract_sha256"],
        "source_attribution_report_sha256": attribution.sha256(report_path),
        "source_model_sha256": parent["source_model_sha256"],
        "source_inputs_sha256": parent["source_inputs_sha256"], "source_sha256": hashes,
        "instance_seeds": list(INSTANCE_SEEDS), "source_episode": 120,
        "intervention": "offer Defer only when inbound present and physical Accept mask empty",
        "bounds": {"inbound_release_age": 80, "option_steps": 10, "consecutive_defer": 16,
                   "stored_deadline": "nominal due, not due+20"},
        "retained": "exact SAFE certificate gate, recovery guard, critic, objective and hierarchical selector",
        "evaluation": "epsilon zero; no optimizer/replay mutation; compare with nominal E119 rows",
        "scope": "one frozen model on five reused development instances; no formal liveness extension claimed",
        "decision": "report J, its components, early tail, outside wait, completion and guard forcing",
        "training_runs": 0, "paper_evidence": False, "test_panel_opened": False,
    }
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**value, "contract_sha256": digest}
