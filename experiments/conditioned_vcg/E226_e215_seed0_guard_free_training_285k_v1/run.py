#!/usr/bin/env python3
"""Prepare and smoke-test E226; train only when explicitly requested."""

from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
)
from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1.roundoff import (
    repaired_audit,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    run as e191_run,
    scenario as e191_scenario,
    stage as e191_stage,
)
from experiments.conditioned_vcg.E215_e191_long_exploration_drift_pilot_274k_v1 import (
    run as e215_run,
)
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import (
    protocol as e222,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import instrumentation, protocol as p, representation


class E226Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_e215_seed0_guard_free_ablation"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E226Error(message)


@contextmanager
def implementation():
    """Install E226 only at E191's explicit trainer extension seams."""

    with (
        patch.object(e191_run, "p", p),
        patch.object(e191_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e191_run.stage, "p", p),
        patch.object(
            e191_run.representation,
            "FreshChoicePreservingQueueAgent",
            representation.GuardFreeTrainingQueueAgent,
        ),
        patch.object(e191_run, "instrumentation", instrumentation),
    ):
        yield


def prepare(output: Path) -> dict:
    conformance = validate_protocol(p.contract(), source=p.PROTOCOL)
    output = output.resolve()
    contract_path = output / "contract.json"
    if not contract_path.is_file() and output.exists():
        # A shell pipeline creates its tee log before Python enters prepare.
        # Permit only those inert logs; never infer recovery from training data.
        existing = tuple(path for path in output.rglob("*") if path.is_file())
        allowed = {
            output / "seed-0-run.log",
            output / "seed-0-run.log~",
        }
        require(
            set(existing).issubset(allowed),
            "E226 refuses non-log artifacts in an uncontracted output",
        )
        e187_run._write(contract_path, p.contract())
    with implementation():
        value = e191_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "model_seeds": [0],
        "single_behavior_delta": p.contract()["single_behavior_delta"],
        "validation_every": p.VALIDATION_EVERY,
        "checkpoint_every": p.CHECKPOINT_EVERY,
        "smoke_required_before_training": True,
        "method_revision": conformance["method_revision"],
    }


def authenticate(output: Path) -> dict:
    validate_protocol(p.contract(), source=p.PROTOCOL)
    with implementation():
        return e191_run.authenticate(output)


def _state_equal(first: dict, second: dict) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract = authenticate(output)
    require(p.MODEL_SEEDS == (0,), "E226 must remain a seed-0 ablation")
    require(
        p.TOTAL_EPISODES == 300
        and p.VALIDATION_EVERY == 20
        and p.CHECKPOINT_EVERY == 1
        and p.EPSILON_DECAY_DECISIONS == 57_300,
        "E226 training schedule differs from E215",
    )
    require(
        contract["defer_contract"] == {
            "identical_to_e215": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
        },
        "E226 Defer rule differs from E215",
    )
    config = EpisodicViabilityGraphConfig(
        gamma=p.GAMMA,
        episode_horizon_steps=p.MAX_STEPS,
    )
    guard_free = representation.GuardFreeTrainingQueueAgent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    reference = e191_run.representation.FreshChoicePreservingQueueAgent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    require(
        representation.architecture_signature(guard_free)
        == representation.architecture_signature(reference),
        "E226 neural architecture differs from E215",
    )
    payload = guard_free.checkpoint(
        include_replay=True,
        training_protocol=p.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    restored = representation.GuardFreeTrainingQueueAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    require(
        _state_equal(guard_free.Q_local.state_dict(), restored.Q_local.state_dict()),
        "E226 checkpoint round-trip differs",
    )
    metadata = guard_free.checkpoint_metadata()
    require(
        metadata["e226_full_certified_frontier_in_selection"]
        and metadata["e226_full_certified_frontier_in_td_targets"]
        and not metadata["e226_guard_intervention_enabled"]
        and not metadata["e226_reconfigure_mask_enabled"]
        and not metadata["e226_witness_fallback_enabled"]
        and not metadata["e226_critic_bypass_enabled"],
        "E226 checkpoint metadata does not authenticate guard removal",
    )
    require(
        representation.QUEUE_CONTEXT_NAMES
        == e191_run.representation.QUEUE_CONTEXT_NAMES,
        "E226 input feature names differ from E215",
    )
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "controller_architecture": representation.CONTROLLER_ARCHITECTURE,
        "network_architecture": representation.NETWORK_ARCHITECTURE,
        "neural_architecture_identical_to_e215": True,
        "queue_inputs_identical_to_e215": True,
        "checkpoint_roundtrip": True,
        "guard_intervention_removed": True,
        "full_frontier_used_by_selection_and_td": True,
        "episode_runs": 0,
        "training_runs": 0,
    }


def _search(payload: dict):
    from PSLAP.viability_filter import ViabilitySearchConfig

    value = payload["viability_search"]
    return ViabilitySearchConfig(
        max_depth=value["max_depth"],
        max_nodes=int(value["max_nodes"]),
        max_primitive_steps=value["max_primitive_steps"],
        reserve_queue_cells=bool(value["reserve_queue_cells"]),
        search_order=str(value["search_order"]),
    )


def _liveness(payload: dict):
    from PSLAP.viability_candidates import BoundedEventDeferRule

    value = payload["liveness_rule"]
    require(
        int(value["max_option_steps"]) == 10
        and int(value["max_consecutive_defer_decisions"]) == 16,
        "E226 smoke checkpoint has another Defer contract",
    )
    return BoundedEventDeferRule(
        max_option_steps=int(value["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            value["max_consecutive_defer_decisions"]
        ),
    )


def _smoke_path(output: Path) -> Path:
    return output.resolve() / "smoke/complete-episode.json"


def smoke(output: Path, device: str) -> dict:
    contract = authenticate(output)
    require(device in ("cpu", "cuda"), "E226 smoke device must be cpu or cuda")
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    path = _smoke_path(output)
    if path.is_file():
        value = e187_run._read(path)
        require(
            value.get("contract_sha256") == contract["contract_sha256"]
            and value.get("strict_complete") is True,
            "E226 saved smoke artifact differs",
        )
        return {**value, "reused": True, "episode_runs": 0}

    identity = e222.checkpoint_identity(0)
    source = torch.load(
        identity["source_checkpoint"], map_location="cpu", weights_only=False
    )
    require(
        int(source.get("selected_checkpoint_episode", -1)) == 300,
        "E226 smoke requires the finalized E215 episode-300 checkpoint",
    )
    agent = representation.GuardFreeTrainingQueueAgent(
        config=EpisodicViabilityGraphConfig.from_dict(source["config"]),
        seed=0,
        device=device,
        epsilon=0.0,
    )
    agent.load_checkpoint_state(source["agent_state"], resumable=False)
    agent.set_epsilon(0.0)
    seed = int(e191_scenario.VALIDATION_SEEDS[0])
    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    original_print = builtins.print

    def renamed_print(*values, **keywords):
        if values and isinstance(values[0], str) and values[0].startswith("E191 "):
            values = ("E226 " + values[0][len("E191 "):], *values[1:])
        return original_print(*values, **keywords)

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            audit,
            "enumerate_viability_candidates",
            e191_stage._accelerated_enumerator(
                audit.enumerate_viability_candidates
            ),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", instrumentation.single_objective_rescore
        ))
        stack.enter_context(patch.object(builtins, "print", renamed_print))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        stack.enter_context(repaired_audit())
        run = instrumentation.run_episode(
            agent,
            env,
            objective_spec=p.SPEC,
            instance_seed=seed,
            training=False,
            max_steps=p.MAX_STEPS,
            search_config=_search(source),
            liveness_rule=_liveness(source),
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
            run_protocol=p.PROTOCOL,
        )
    require(
        bool(run["strict_method_success"])
        and not bool(run["horizon_cap_hit"])
        and int(run["e226_guard_free"]["forced_decisions"]) == 0
        and int(run["e226_guard_free"]["masked_frontiers"]) == 0,
        "E226 complete smoke episode failed",
    )
    row = {
        "status": "passed",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "source": "E215 selected episode-300 weights; execution-path smoke only",
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "instance_seed": seed,
        "strict_complete": True,
        "steps": int(run["steps"]),
        "J": float(run["e187_metrics"]["J"]),
        "physical_rehandles": float(
            run["e187_metrics"].get(
                "physical_rehandles",
                run["e187_metrics"].get("components", {}).get(
                    "physical_rehandles", 0.0
                ),
            )
        ),
        "telemetry": run["e226_guard_free"],
        "episode_runs": 1,
        "training_runs": 0,
        "test_panel_opened": False,
    }
    e187_run._write(path, row)
    return row


def _smoke_gate(output: Path) -> dict:
    path = _smoke_path(output)
    require(path.is_file(), "run E226 smoke before launching official training")
    value = e187_run._read(path)
    require(
        value.get("contract_sha256") == p.contract()["contract_sha256"]
        and value.get("strict_complete") is True,
        "E226 smoke gate is stale or failed",
    )
    return value


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    authenticate(output)
    _smoke_gate(output)
    require(
        1 <= through_episode <= p.TOTAL_EPISODES,
        "E226 through-episode must be between 1 and 300",
    )
    with implementation(), repaired_audit():
        # Enter through E191 so its adapter installs the E226 protocol at the
        # E187 trainer seams (agent class, instrumentation, workload, and
        # accelerations). Calling E187 directly would authenticate E187's own
        # contract instead of this experiment's contract.
        result = e191_run.train_seed(output, 0, through_episode, device)
    result.pop("e187_protocol", None)
    result.pop("e191_protocol", None)
    result.pop("e191_contract_sha256", None)
    return {
        **result,
        "e226_protocol": p.PROTOCOL,
        "e226_contract_sha256": p.contract()["contract_sha256"],
        "model_seed": 0,
        "guard_intervention_removed": True,
        "paper_evidence": False,
    }


def status(output: Path) -> dict:
    authenticate(output)
    with implementation():
        value = e191_run.status(output)
    return {**value, "protocol": p.PROTOCOL}


def _summary_path(output: Path) -> Path:
    return (
        output.resolve() / "training/operational" / p.SCENARIO
        / "seed-0/training-summary.json"
    )


def _validation_rows(summary: dict) -> list[dict]:
    training = list(summary["training"]["runs"])
    rows = []
    for record in summary.get("validation_history", ()):
        row = e187_run._validation_row(record, training)
        row["guard_free"] = record["summary"].get("e226_guard_free", {})
        rows.append(row)
    return rows


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    current_path = _summary_path(output)
    reference_path = e215_run._summary_path(p.REFERENCE_OUTPUT)
    require(current_path.is_file(), "E226 training summary is missing")
    require(reference_path.is_file(), "matched E215 seed-0 summary is missing")
    current = e187_run._read(current_path)
    reference = e187_run._read(reference_path)
    current_rows = _validation_rows(current)
    reference_rows = e215_run._validation_rows(reference)
    current_by_ep = {int(row["checkpoint_episode"]): row for row in current_rows}
    reference_by_ep = {int(row["checkpoint_episode"]): row for row in reference_rows}
    paired = []
    for episode in sorted(set(current_by_ep) & set(reference_by_ep)):
        new = current_by_ep[episode]
        old = reference_by_ep[episode]
        telemetry = new["guard_free"]
        strict_rate = float(new["strict_completion_rate"])
        max_run = int(telemetry.get(
            "maximum_longest_reconfigure_run_without_accept_or_deliver", 0
        ))
        cap_hits = int(telemetry.get("step_limit_hits", 0))
        j_regression = (
            float(new["J"]) / float(old["J"]) - 1.0
            if float(old["J"]) else None
        )
        rehandle_regression = (
            float(new["rehandles_per_100"])
            - float(old["rehandles_per_100"])
        )
        paired.append({
            "checkpoint_episode": episode,
            "guard_free_J": float(new["J"]),
            "guard_on_J": float(old["J"]),
            "J_regression_fraction": j_regression,
            "guard_free_rehandles_per_100": float(new["rehandles_per_100"]),
            "guard_on_rehandles_per_100": float(old["rehandles_per_100"]),
            "rehandles_per_100_regression": rehandle_regression,
            "strict_completion_rate": strict_rate,
            "step_limit_hits": cap_hits,
            "maximum_nonprogress_rehandle_run": max_run,
            "liveness_pass": bool(
                strict_rate == 1.0
                and cap_hits == 0
                and max_run <= p.MAX_ALLOWED_NONPROGRESS_REHANDLE_RUN
            ),
            "tracking_pass": bool(
                j_regression is not None
                and j_regression <= p.MAX_J_REGRESSION_FRACTION
                and rehandle_regression <= p.MAX_REHANDLE_PER_100_REGRESSION
            ),
        })
    complete = int(current["completed_training_episodes"]) == p.TOTAL_EPISODES
    liveness_pass = bool(paired) and all(row["liveness_pass"] for row in paired)
    tracking_pass = bool(paired) and all(row["tracking_pass"] for row in paired)
    report = {
        "method_conformance": validate_protocol(contract, source=p.PROTOCOL),
        "status": "complete" if complete else "partial",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "completed_training_episodes": int(current["completed_training_episodes"]),
        "paired_checkpoints": paired,
        "early_read": [
            row for row in paired
            if row["checkpoint_episode"] in p.EARLY_READ_CHECKPOINTS
        ],
        "all_observed_liveness_gates_passed": liveness_pass,
        "all_observed_tracking_gates_passed": tracking_pass,
        "verdict": (
            "guard_not_needed" if complete and liveness_pass and tracking_pass
            else "guard_needed" if paired and not liveness_pass
            else "continue_or_inconclusive"
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    report_path = output.resolve() / "e226-report.json"
    e187_run._write(report_path, report)
    text = render_validated_conformance_audit(contract)
    text += "\nRESULTS\n" + "=" * 72 + "\n"
    text += json.dumps({
        "status": report["status"],
        "verdict": report["verdict"],
        "completed_training_episodes": report["completed_training_episodes"],
        "early_read": report["early_read"],
        "paired_checkpoints": report["paired_checkpoints"],
    }, indent=2, sort_keys=True) + "\n"
    report_text = output.resolve() / "REPORT.txt"
    temporary = report_text.with_suffix(".txt.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(report_text)
    return {
        "status": report["status"],
        "verdict": report["verdict"],
        "completed_training_episodes": report["completed_training_episodes"],
        "paired_checkpoints": len(paired),
        "report": str(report_path),
        "report_text": str(report_text),
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "self-check", "smoke", "train-seed", "status", "analyze"
        ),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--through-episode", type=int, default=60)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "smoke":
        result = smoke(args.output, args.device)
    elif args.command == "train-seed":
        require(args.model_seed == 0, "E226 is predeclared for seed 0 only")
        result = train_seed(args.output, args.through_episode, args.device)
    elif args.command == "status":
        result = status(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

