"""Isolated online D12 confirmation; native current-state witness remains authoritative."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import json

import torch
import benchmark_viability_critic_priority as benchmark
import PSLAP.viability_candidates as candidates
from PSLAP import viability as v
from experiments.conditioned_vcg.E14_certification_scalability_95k import capture, confirm, program as e14, reuse
from experiments.conditioned_vcg.development.D10_scalability_support_screen import occupancy_extension as occupancy
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from .proofs import ValidatedAnchor, targeted_action, try_relocation
from .run import ROOT, SCENARIO


class FamilyVerifier:
    """Per-frontier adapter. Family proofs never enter the native outcome cache.

    This preserves native current-state witnesses used by the retained liveness
    guard. All family misses delegate to the original exact cached checker.
    """

    def __init__(self, original):
        self.original = original
        self.counts = Counter()
        self.setup_seconds = 0.0
        self.connection_seconds = 0.0
        self.begin()

    def begin(self):
        self.first = True
        self.anchor = None

    def __call__(self, state, config, cache):
        if self.first:
            self.first = False
            result = self.original(state, config, cache)
            start = perf_counter()
            if result[0].is_safe and result[0].witness:
                self.anchor = ValidatedAnchor(state, result[0])
            self.setup_seconds += perf_counter() - start
            self.counts["native_current_state_checks"] += 1
            return result
        # Respect preexisting exact outcomes before attempting a family proof.
        cached = cache.get(candidates._certificate_key(state, config))
        if cached is not None:
            self.counts["native_outcome_hits"] += 1
            return cached, True, 0.0
        start = perf_counter()
        physical = reuse.timing_erased_state(state)
        anchor = self.anchor
        attempt = None
        if anchor is not None and len(physical.blocks) == len(anchor.state.blocks):
            changes = [(old, physical.block(old.label)) for old in anchor.state.blocks
                       if physical.block(old.label) != old]
            if len(changes) == 1 and changes[0][1] is not None:
                old, new = changes[0]
                action = targeted_action(anchor.state, v.RecoveryActionKind.RELOCATION, old.label, new.position)
                if action is not None:
                    attempt = try_relocation(anchor, action, state, config)
        elapsed = perf_counter() - start
        self.connection_seconds += elapsed
        if attempt is not None and attempt.certificate is not None:
            self.counts["family_proofs"] += 1
            return attempt.certificate, False, elapsed
        self.counts["native_recovery_fallbacks"] += 1
        if attempt is not None:
            self.counts[f"miss:{attempt.reason}"] += 1
        return self.original(state, config, cache)


def compare_prefix(reference, frontiers, decisions):
    """Compare actions independently of expected certificate metadata changes."""
    refs = reference["frontiers"]
    length = min(len(refs), len(frontiers))
    frontier_mismatches = [i for i in range(length)
                          if refs[i]["candidate_keys"] != frontiers[i]["candidate_keys"]
                          or refs[i]["decision_epoch"] != frontiers[i]["decision_epoch"]]
    old_decisions = reference["decisions"]
    common = min(len(old_decisions), len(decisions))
    action_mismatches = [i for i in range(common)
                         if (old_decisions[i]["selected_key"], old_decisions[i]["decision_epoch"])
                         != (decisions[i]["selected_key"], decisions[i]["decision_epoch"])]
    return {"common_frontiers": length, "candidate_key_or_epoch_mismatches": frontier_mismatches,
            "common_decisions": common, "selected_action_or_epoch_mismatches": action_mismatches,
            "certificate_digest_identity_required": False}


def run(source, output, max_seconds=1200):
    contract, manifest = e14.authenticate(source)
    capture.authenticate_capture(source, contract, SCENARIO)
    reference = confirm._authenticate_ledger(source, contract, SCENARIO, reuse.STAGE_PATH_CLEANUP)
    parent = capture._parent_record(manifest, SCENARIO)
    instance = occupancy._load_instance(e14.PARENT_OUTPUT, parent)
    conditioned = occupancy._conditioned_auth()
    arm = conditioned["inputs"]["arms"][e14.MODEL_SEED]
    scenario = occupancy.SCENARIO_BY_ID[SCENARIO]
    output.mkdir(parents=True, exist_ok=False)
    online_contract = e14.with_hash({
        "protocol": "d12_online_family_v1", "max_seconds": max_seconds,
        "e14_contract_sha256": contract["contract_sha256"],
        "baseline_ledger_sha256": reference["ledger_sha256"],
        "source_sha256": {str(p.relative_to(ROOT)): e14.sha256(p)
                          for p in (Path(__file__), Path(__file__).with_name("proofs.py"))},
        "current_state_and_liveness_witness": "native_exact;family_proofs_not_cached",
        "model_seed": e14.MODEL_SEED, "lambda": e14.PREFERENCE_LAMBDA,
        "scenario": SCENARIO, "instance_seed": int(instance.seed),
    }, "contract_sha256")
    e14.atomic_json(output / "contract.json", online_contract)
    verifier = FamilyVerifier(candidates._analyze_cached)
    native_enumerate = benchmark._enumerate_frontier
    active_cache = reuse.TimingInvariantCertificateCache()
    frontiers = []
    envs = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(ROOT, {"conditioned": conditioned},
            model_seed=e14.MODEL_SEED, base=base, device=torch.device("cpu"))
        agent.set_epsilon(0.0)
        return conditioned_seed0._FixedLambdaAgent(agent, e14.PREFERENCE_LAMBDA)

    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        envs.append(env)
        return env

    def enumerate_frontier(*args, **kwargs):
        verifier.begin()
        result = native_enumerate(*args, **kwargs)
        projection = confirm._frontier_projection(result[1], len(frontiers))
        frontiers.append(projection)
        with (output / "frontiers.jsonl").open("a") as stream:
            stream.write(json.dumps(projection) + "\n")
        print(json.dumps({"frontier": len(frontiers), "family_proofs": verifier.counts["family_proofs"],
                          "frontier_seconds": projection["total_frontier_seconds"]}), flush=True)
        return result

    raw = None
    error = None
    failure = None
    start = perf_counter()
    try:
        with capture.wall_limit(max_seconds), reuse.path_cleanup_active(), ExitStack() as stack:
            stack.enter_context(patch.object(candidates, "_analyze_cached", verifier))
            stack.enter_context(patch.object(benchmark, "_enumerate_frontier", enumerate_frontier))
            stack.enter_context(patch.object(benchmark, "_make_env", env_factory))
            stack.enter_context(patch.object(benchmark, "ViabilityCertificateCache", lambda: active_cache))
            stack.enter_context(pilot._agent_factory(agent_factory))
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL, controller_payload=arm.payload,
                instance=instance, instance_seed=int(instance.seed),
                search_config=benchmark._search_config(arm.payload),
                liveness_rule=benchmark._liveness_rule(arm.payload), prioritizer=None,
                max_steps=scenario.max_steps, device=torch.device("cpu"))
    except capture.CaptureLimit as caught:
        failure, error = "censored_wall_clock", str(caught)
    except Exception as caught:
        failure, error = "implementation_error", f"{type(caught).__name__}: {caught}"
    elapsed = perf_counter() - start
    strict = bool(raw is not None and raw["strict_method_success"] and raw["terminal"]
                  and raw["method_failure_reason"] is None and raw["illegal_drops"] == 0
                  and raw["macro_failures"] == 0 and len(raw["delivery_deviations"]) == scenario.total_jobs)
    decisions = raw["decisions"] if raw else []
    report = {
        "protocol": online_contract["protocol"], "contract_sha256": online_contract["contract_sha256"],
        "failure_class": failure or ("completed" if strict else "operationally_incomplete"),
        "error": error, "strict_safe_complete": strict, "wall_seconds": elapsed,
        "baseline_historical_wall_seconds": reference["wall_seconds"],
        "comparison": compare_prefix(reference, frontiers, decisions),
        "verifier_counts": dict(verifier.counts), "anchor_validation_seconds": verifier.setup_seconds,
        "connection_seconds": verifier.connection_seconds, "completed_frontiers": len(frontiers),
        "completed_decisions": len(decisions), "steps": raw["steps"] if raw else None,
        "deliveries": len(raw["delivery_deviations"]) if raw else None,
        "physical_relocations": raw["relocations"] if raw else None,
        "forced_liveness_decisions": sum(d["liveness_forced"] for d in decisions),
        "macro_failures": raw["macro_failures"] if raw else None,
        "illegal_drops": raw["illegal_drops"] if raw else None,
        "occupancy_at_stop": envs[0]._measurement() if envs else None,
        "interpretation": "one online medium instance; historical timing comparison, not a matched repeat",
    }
    if raw is not None:
        e14.atomic_json(output / "episode.json", raw)
    e14.atomic_json(output / "report.json", e14.with_hash(report, "report_sha256"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/vcg-conditioned-e14-certificate-reuse-95k")
    parser.add_argument("--output", type=Path, default=ROOT / "results/vcg-d12-relocation-family-online-95k")
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("time budget must be positive")
    print(json.dumps(run(args.source, args.output, args.max_seconds), indent=2))


if __name__ == "__main__":
    main()
