"""Pin the completed episode-120 model, its panel, and E119 code."""

from functools import lru_cache
import json
from pathlib import Path

from experiments.conditioned_vcg.E118_short_exploration_earliness_8x8_182k_v1 import protocol as e118

ROOT = e118.ROOT
PROTOCOL = "vcg_conditioned_e119_tolerance_defer_comparison_8x8_183k_v1_repair1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e119-tolerance-defer-comparison-8x8-183k-v1-repair1"
SOURCE_OUTPUT = e118.DEFAULT_OUTPUT / "schedule_only"
SOURCE_DIRECTORY = SOURCE_OUTPUT / "training/operational" / e118.SCENARIO / "seed-0"
SOURCE_CHECKPOINT = SOURCE_DIRECTORY / "latest.pth"
SOURCE_SUMMARY = SOURCE_DIRECTORY / "training-summary.json"
ARMS = ("nominal_due", "due_plus_20")
INSTANCE_SEEDS = tuple(e118.contract()["validation_seeds"])
SCENARIO = e118.SCENARIO
MAX_STEPS = e118.base.e107.e105.e78.SCENARIOS[SCENARIO].max_steps


@lru_cache(maxsize=1)
def contract():
    for path in (SOURCE_CHECKPOINT, SOURCE_SUMMARY, SOURCE_OUTPUT / "contract.json"):
        if not path.is_file():
            raise RuntimeError(f"E119 missing completed E118 source: {path}")
    summary = json.loads(SOURCE_SUMMARY.read_text())
    parent = e118.arm_contract("schedule_only")
    if (summary.get("completed_training_episodes") != 120
            or summary.get("status") != "complete"
            or summary.get("e115_contract_sha256") != parent["contract_sha256"]
            or json.loads((SOURCE_OUTPUT / "contract.json").read_text()) != parent):
        raise RuntimeError("E119 requires authenticated completed E118 schedule_only episode 120")
    directory = Path(__file__).resolve().parent
    hashes = dict(e118.contract()["source_sha256"])
    for name in ("protocol.py", "policy.py", "run.py", "run.sh"):
        path = directory / name
        hashes[str(path.relative_to(ROOT))] = e118.base.sha256(path)
    for name in ("PSLAP/viability_candidates.py", "example/Options/ExplicitDeferOption.py"):
        hashes[name] = e118.base.sha256(ROOT / name)
    value = {
        "schema_version": 1, "protocol": PROTOCOL,
        "scope": "frozen policy development ablation; no training",
        "source_arm": "schedule_only", "source_episode": 120,
        "source_checkpoint_sha256": e118.base.sha256(SOURCE_CHECKPOINT),
        "source_summary_sha256": e118.base.sha256(SOURCE_SUMMARY),
        "source_arm_contract_sha256": parent["contract_sha256"],
        "arms": list(ARMS), "instance_seeds": list(INSTANCE_SEEDS),
        "scenario": SCENARIO, "max_steps": MAX_STEPS,
        "wall_seconds_per_episode": 300,
        "nominal_rule": "forbid Defer when min(d_i-t) <= 0",
        "changed_rule": "forbid Defer when min(d_i+20-t) <= 0; bound wait horizon by same quantity",
        "other_rules": "retain pickup exclusion, SAFE requirement, 10-step cap, 16-Defer bound, recovery guard",
        "critic_clock": "unchanged nominal release-relative due slack",
        "objective": "Jref=2T+E+0.1L+10R, T/E beyond +/-20, release-fixed due dates",
        "native_parity_required": "same instance, delivery deviations, steps, actions, guard counts and J as episode-120 validation",
        "evaluation": "epsilon zero; no replay insertion, optimizer step, or target update",
        "incomplete_episodes": "report failures; exclude costs from comparable quality averages",
        "causal_scope": "runtime effect on one frozen checkpoint; not retrained performance or a new liveness theorem",
        "training_runs": 0, "paper_evidence": False, "test_panel_opened": False,
        "source_sha256": hashes,
    }
    return {**value, "contract_sha256": e118.base.digest(value)}
