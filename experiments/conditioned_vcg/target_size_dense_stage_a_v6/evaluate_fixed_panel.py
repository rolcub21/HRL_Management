"""Evaluate the terminal E79-v6 checkpoint on its fixed development panel.

This evaluator deliberately preserves the policy used by the continuation:
direct executable-legal candidates, the observable overdue-progress filter,
greedy mode-regularized ranking, and no recoverability certification.  It
loads a deployment-only clone and never mutates replay, optimizer, or the
source checkpoint.

The five seeds are the prespecified E79 development instances.  They are not
a sealed test panel and this report is not safety or deployment evidence.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from PSLAP.viability_candidates import BoundedEventDeferRule
from PSLAP.viability_filter import ViabilitySearchConfig
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E79_direct_legal_overdue_progress_full_10x10_v1 import (
    run as continuation,
)
from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v2 import (
    roundoff,
)
from experiments.conditioned_vcg.e79_direct_legal_frontier import (
    NO_CERTIFICATION_CONTRACT,
    DirectLegalRanker,
    update_direct_frontier_audit,
)
from experiments.conditioned_vcg.e79_direct_legal_overdue_progress import (
    OVERDUE_PROGRESS_CONTRACT,
    OVERDUE_PROGRESS_INTERFACE,
    OverdueProgressAudit,
    enumerate_overdue_progress_candidates,
)
PROTOCOL = "vcg_e79_v6_terminal_fixed_development_evaluation_v1"
FIXED_DEVELOPMENT_SEEDS = (112100, 112101, 112102, 112103, 112104)


def _load_resumable():
    path = Path(__file__).resolve().with_name("resumable_continuation.py")
    spec = importlib.util.spec_from_file_location(
        "e79_v6_resumable_for_evaluation", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V6 continuation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resumable = _load_resumable()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_agent(checkpoint_output: Path, device: torch.device):
    checkpoint_output = checkpoint_output.resolve()
    contract = json.loads(
        (checkpoint_output / "contract.json").read_text(encoding="utf-8")
    )
    source_sha = str(contract["source_checkpoint_sha256"])
    if contract != resumable._contract(source_sha):
        raise RuntimeError("E79-v6 continuation contract differs")

    checkpoint = checkpoint_output / "latest.pth"
    checkpoint_sha = _sha256(checkpoint)
    summary = json.loads(
        (checkpoint_output / "training-summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("status") != "complete"
        or int(summary.get("through_episode", -1)) != resumable.TOTAL_EPISODES
        or summary.get("latest_checkpoint_sha256") != checkpoint_sha
    ):
        raise RuntimeError("terminal summary does not authenticate the checkpoint")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumable._validate(payload, contract)
    if int(payload["through_episode"]) != resumable.TOTAL_EPISODES:
        raise RuntimeError("evaluation requires the terminal episode-63 checkpoint")

    config = EpisodicViabilityGraphConfig.from_dict(payload["config"])
    agent = EpisodicViabilityGraphHierarchyAgent(
        config=config,
        seed=int(getattr(continuation.p, "MODEL_SEED", 0)),
        device=device,
        epsilon=0.0,
    )
    # Deployment-only loading intentionally ignores optimizer, replay, and the
    # training RNG stream even though they remain present in latest.pth.
    agent.load_checkpoint_state(payload["agent_state"], resumable=False)
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    return checkpoint, checkpoint_sha, payload, agent


def _run_one(agent, resume: dict, seed: int) -> tuple[dict, dict]:
    wrapper = DirectLegalRanker(agent)
    env = continuation.original.TargetDenseEnv(
        continuation.p.SCENARIO, continuation.p.SPEC
    )
    env.current_episode = 1
    search_config = ViabilitySearchConfig(**dict(resume["search_config"]))
    liveness = dict(resume["liveness_rule"])
    liveness_rule = BoundedEventDeferRule(
        max_option_steps=int(liveness["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            liveness["max_consecutive_defer_decisions"]
        ),
    )
    progress = OverdueProgressAudit()
    started = perf_counter()
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    with ExitStack() as stack:
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(
            patch.object(
                audit,
                "enumerate_viability_candidates",
                lambda *args, **kwargs: enumerate_overdue_progress_candidates(
                    *args, progress_audit=progress, **kwargs
                ),
            )
        )
        stack.enter_context(
            patch.object(
                audit, "_update_frontier_audit", update_direct_frontier_audit
            )
        )
        stack.enter_context(roundoff.repaired_audit())
        run = audit.run_objective_audit_episode(
            wrapper,
            env,
            objective_spec=continuation.p.SPEC,
            instance_seed=int(seed),
            training=False,
            max_steps=int(resume["environment"]["max_steps"]),
            search_config=search_config,
            liveness_rule=liveness_rule,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=0,
            run_protocol=PROTOCOL,
        )
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    wall_seconds = perf_counter() - started

    if run.get("method_failure_reason") == "no_exact_safe_candidate":
        run["method_failure_reason"] = "no_overdue_progress_candidate"
    operational_strict = bool(run.pop("strict_method_success"))
    run.update(
        {
            "wall_seconds": wall_seconds,
            "operational_strict_success": operational_strict,
            "physical_completion_success": bool(run["success"]),
            "strict_method_success": None,
        }
    )
    run["audit"].update(
        {
            "candidate_interface": OVERDUE_PROGRESS_INTERFACE,
            "certification_contract": NO_CERTIFICATION_CONTRACT,
            "overdue_progress_contract": OVERDUE_PROGRESS_CONTRACT,
            "overdue_progress": progress.to_dict(),
            "exact_verifier_authoritative": False,
            "complete_frontier_exactly_verified": False,
            "eligibility_proxy_is_safety_evidence": False,
            "recovery_witness_guard_enabled": False,
        }
    )
    metadata = {
        "direct_candidates_scored": int(wrapper.direct_candidates_scored),
        "overdue_progress_audit": progress.to_dict(),
    }
    return run, metadata


def _summary(raw_runs: list[dict]) -> dict:
    # summarize_objective_runs expects a Boolean strict field.  Supply the
    # explicitly operational definition temporarily, then rename its result.
    compatible = []
    for item in raw_runs:
        value = dict(item)
        value["strict_method_success"] = bool(
            item["operational_strict_success"]
        )
        compatible.append(value)
    summary = audit.summarize_objective_runs(compatible)
    summary["operational_strict_success_rate"] = summary.pop(
        "strict_method_success_rate"
    )
    summary.update(
        {
            "all_64_delivered_rate": sum(
                int(run["delivery_count"] == continuation.p.JOBS_PER_EPISODE)
                for run in raw_runs
            )
            / len(raw_runs),
            "development_panel_only": True,
            "sealed_test_panel": False,
            "evaluation_learning": False,
            "evaluation_epsilon": 0.0,
            "recoverability_certification": False,
            "safety_guarantee": False,
            "deployment_checkpoint_eligible": False,
            "paper_evidence": False,
        }
    )
    return summary


def evaluate(checkpoint_output: Path, output: Path, device_name: str) -> dict:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    global_rng = audit._capture_global_rng_state()
    device = torch.device(device_name)
    checkpoint, checkpoint_sha, payload, agent = _load_frozen_agent(
        checkpoint_output, device
    )
    gradient_steps_before = int(agent.gradient_steps)
    transition_count_before = int(agent.transition_count)
    runs = []
    metadata = []
    started = perf_counter()
    try:
        for index, seed in enumerate(FIXED_DEVELOPMENT_SEEDS, start=1):
            run, row_metadata = _run_one(
                agent, payload["resume_contract"], seed
            )
            runs.append(run)
            metadata.append(row_metadata)
            print(
                f"Eval {index}/{len(FIXED_DEVELOPMENT_SEEDS)}"
                f" | Seed {seed}"
                f" | DenseR {float(run['dense_rescored_return']):9.2f}"
                f" | Done {int(run['delivery_count']):2d}/"
                f"{continuation.p.JOBS_PER_EPISODE}"
                f" | Dec {int(run['macro_decisions']):4d}"
                f" | MAE {float(run['mean_absolute_error']):.3f}"
                f" | Time {float(run['wall_seconds']) / 60.0:.1f}m",
                flush=True,
            )
            partial = {
                "status": "partial",
                "protocol": PROTOCOL,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "completed_seeds": index,
                "fixed_development_seeds": FIXED_DEVELOPMENT_SEEDS,
                "runs": tuple(runs),
            }
            audit._atomic_json_save(
                audit._json_safe(partial), output.with_suffix(".partial.json")
            )
    finally:
        audit._restore_global_rng_state(global_rng)

    final_sha = _sha256(checkpoint)
    if final_sha != checkpoint_sha:
        raise RuntimeError("source checkpoint changed during evaluation")
    report = {
        "status": "complete",
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_through_episode": int(payload["through_episode"]),
        "fixed_development_seeds": FIXED_DEVELOPMENT_SEEDS,
        "evaluation_wall_seconds": perf_counter() - started,
        "source_checkpoint_unchanged": True,
        "gradient_steps_unchanged": int(agent.gradient_steps)
        == gradient_steps_before,
        "transition_count_unchanged": int(agent.transition_count)
        == transition_count_before,
        "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
        "summary": _summary(runs),
        "runs": tuple(runs),
        "run_metadata": tuple(metadata),
    }
    audit._atomic_json_save(audit._json_safe(report), output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate(args.checkpoint_output, args.output, args.device)
    print(json.dumps(audit._json_safe(report["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
