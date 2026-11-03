#!/usr/bin/env python3
"""Prepare, smoke, train, and compare E227 model seeds 1 and 2."""

from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
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
from experiments.conditioned_vcg.E226_e215_seed0_guard_free_training_285k_v1 import (
    instrumentation,
    representation,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p


class E227Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_e218_seeds12_guard_free_replication"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E227Error(message)


@contextmanager
def implementation():
    """Install E227 only at E191's authenticated trainer seams."""

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


def _known_precontract_logs(output: Path) -> set[Path]:
    return {
        output / "seed-1-run.log",
        output / "seed-1-run.log~",
        output / "seed-2-run.log",
        output / "seed-2-run.log~",
    }


def prepare(output: Path) -> dict:
    conformance = validate_protocol(p.contract(), source=p.PROTOCOL)
    output = output.resolve()
    contract_path = output / "contract.json"
    if not contract_path.is_file() and output.exists():
        existing = tuple(path for path in output.rglob("*") if path.is_file())
        require(
            set(existing).issubset(_known_precontract_logs(output)),
            "E227 refuses non-log artifacts in an uncontracted output",
        )
        e187_run._write(contract_path, p.contract())
    with implementation():
        value = e191_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "model_seeds": list(p.MODEL_SEEDS),
        "single_behavior_delta": p.contract()["single_behavior_delta"],
        "validation_every": p.VALIDATION_EVERY,
        "checkpoint_every": p.CHECKPOINT_EVERY,
        "smoke_required_per_seed": True,
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
    require(p.MODEL_SEEDS == (1, 2), "E227 seeds must remain 1 and 2")
    require(
        p.TOTAL_EPISODES == 300
        and p.VALIDATION_EVERY == 20
        and p.CHECKPOINT_EVERY == 1
        and p.EPSILON_DECAY_DECISIONS == 57_300,
        "E227 schedule differs from E218",
    )
    require(
        contract["defer_contract"] == {
            "identical_to_e218": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
        },
        "E227 Defer rule differs from E218",
    )
    checked = []
    for seed in p.MODEL_SEEDS:
        config = EpisodicViabilityGraphConfig(
            gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
        )
        agent = representation.GuardFreeTrainingQueueAgent(
            config=config,
            seed=seed,
            device="cpu",
            epsilon=p.EPSILON_START,
        )
        reference = e191_run.representation.FreshChoicePreservingQueueAgent(
            config=config,
            seed=seed,
            device="cpu",
            epsilon=p.EPSILON_START,
        )
        require(
            representation.architecture_signature(agent)
            == representation.architecture_signature(reference),
            f"E227 seed {seed} neural architecture differs from E218",
        )
        payload = agent.checkpoint(
            include_replay=True,
            training_protocol=p.PROTOCOL,
            future_schedule_visible_to_policy=False,
        )
        restored = representation.GuardFreeTrainingQueueAgent.from_checkpoint(
            payload, device="cpu", resumable=True, seed=seed
        )
        require(
            _state_equal(agent.Q_local.state_dict(), restored.Q_local.state_dict()),
            f"E227 seed {seed} checkpoint round-trip differs",
        )
        metadata = agent.checkpoint_metadata()
        require(
            metadata["e226_full_certified_frontier_in_selection"]
            and metadata["e226_full_certified_frontier_in_td_targets"]
            and not metadata["e226_guard_intervention_enabled"]
            and not metadata["e226_reconfigure_mask_enabled"]
            and not metadata["e226_witness_fallback_enabled"]
            and not metadata["e226_critic_bypass_enabled"],
            f"E227 seed {seed} metadata does not authenticate guard removal",
        )
        checked.append(seed)
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "model_seeds": checked,
        "controller_architecture": representation.CONTROLLER_ARCHITECTURE,
        "network_architecture": representation.NETWORK_ARCHITECTURE,
        "neural_architecture_identical_to_e218": True,
        "queue_inputs_identical_to_e218": True,
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
        "E227 source checkpoint has another Defer contract",
    )
    return BoundedEventDeferRule(
        max_option_steps=int(value["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            value["max_consecutive_defer_decisions"]
        ),
    )


def _smoke_path(output: Path, seed: int) -> Path:
    return output.resolve() / "smoke" / f"seed-{int(seed)}.json"


def _load_reference(seed: int) -> tuple[dict, Path]:
    path = p.reference_checkpoint_path(seed)
    require(path.is_file(), f"E227 seed {seed} smoke checkpoint is missing: {path}")
    expected = p.REFERENCE_CHECKPOINTS[seed]
    require(
        p.sha256(path) == expected["sha256"],
        f"E227 seed {seed} smoke checkpoint SHA-256 differs",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(
        int(payload.get("model_seed", -1)) == seed
        and int(payload.get("selected_checkpoint_episode", -1))
        == int(expected["selected_episode"])
        and payload.get("checkpoint_role") == "best_deployment_finalized",
        f"E227 seed {seed} smoke checkpoint identity differs",
    )
    return payload, path


def smoke(output: Path, seed: int, device: str) -> dict:
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E227 smoke seed must be 1 or 2")
    require(device in ("cpu", "cuda"), "E227 smoke device must be cpu or cuda")
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    path = _smoke_path(output, seed)
    if path.is_file():
        value = e187_run._read(path)
        require(
            value.get("contract_sha256") == contract["contract_sha256"]
            and value.get("strict_complete") is True,
            f"E227 seed {seed} saved smoke artifact differs",
        )
        return {**value, "reused": True, "episode_runs": 0}

    source, source_path = _load_reference(seed)
    agent = representation.GuardFreeTrainingQueueAgent(
        config=EpisodicViabilityGraphConfig.from_dict(source["config"]),
        seed=seed,
        device=device,
        epsilon=0.0,
    )
    agent.load_checkpoint_state(source["agent_state"], resumable=False)
    agent.set_epsilon(0.0)
    instance_seed = int(e191_scenario.VALIDATION_SEEDS[seed - 1])
    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    inherited_print = builtins.print

    def renamed_print(*values, **keywords):
        if values and isinstance(values[0], str) and values[0].startswith("E191 "):
            values = ("E227 " + values[0][len("E191 "):], *values[1:])
        return inherited_print(*values, **keywords)

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
            instance_seed=instance_seed,
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
        f"E227 seed {seed} complete smoke episode failed",
    )
    metrics = run["e187_metrics"]
    row = {
        "status": "passed",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seed": seed,
        "source": (
            "E218 selected weights; execution-path smoke only, never training warm start"
        ),
        "source_checkpoint": str(source_path.relative_to(p.ROOT)),
        "source_checkpoint_sha256": p.REFERENCE_CHECKPOINTS[seed]["sha256"],
        "instance_seed": instance_seed,
        "strict_complete": True,
        "steps": int(run["steps"]),
        "J": float(metrics["J"]),
        "physical_rehandles": float(
            metrics.get(
                "physical_rehandles",
                metrics.get("components", {}).get("physical_rehandles", 0.0),
            )
        ),
        "telemetry": run["e226_guard_free"],
        "episode_runs": 1,
        "training_runs": 0,
        "test_panel_opened": False,
    }
    e187_run._write(path, row)
    return row


def _smoke_gate(output: Path, seed: int) -> dict:
    path = _smoke_path(output, seed)
    require(path.is_file(), f"run E227 seed {seed} smoke before training it")
    value = e187_run._read(path)
    require(
        value.get("contract_sha256") == p.contract()["contract_sha256"]
        and value.get("model_seed") == seed
        and value.get("strict_complete") is True,
        f"E227 seed {seed} smoke gate is stale or failed",
    )
    return value


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    authenticate(output)
    require(seed in p.MODEL_SEEDS, "E227 model seed must be 1 or 2")
    _smoke_gate(output, seed)
    require(
        1 <= through_episode <= p.TOTAL_EPISODES,
        "E227 through-episode must be between 1 and 300",
    )
    with implementation(), repaired_audit():
        result = e191_run.train_seed(output, seed, through_episode, device)
    result.pop("e191_protocol", None)
    result.pop("e191_contract_sha256", None)
    return {
        **result,
        "e227_protocol": p.PROTOCOL,
        "e227_contract_sha256": p.contract()["contract_sha256"],
        "model_seed": seed,
        "guard_intervention_removed": True,
        "paper_evidence": False,
    }


def _summary_path(output: Path, seed: int) -> Path:
    return (
        output.resolve() / "training/operational" / p.SCENARIO
        / f"seed-{seed}/training-summary.json"
    )


def status(output: Path) -> dict:
    contract = authenticate(output)
    seeds = {}
    for seed in p.MODEL_SEEDS:
        path = _summary_path(output, seed)
        if path.is_file():
            summary = e187_run._read(path)
            seeds[str(seed)] = {
                "completed_training_episodes": int(
                    summary.get("completed_training_episodes", 0)
                ),
                "latest_checkpoint": summary.get("latest_checkpoint"),
            }
        else:
            latest = path.with_name("latest.pth")
            seeds[str(seed)] = {
                "completed_training_episodes": 0,
                "latest_checkpoint_exists": latest.is_file(),
            }
    return {
        "status": "available",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "seeds": seeds,
        "test_panel_opened": False,
    }


def _validation_rows(summary: dict, *, guard_free: bool) -> list[dict]:
    training = list(summary["training"]["runs"])
    rows = []
    for record in summary.get("validation_history", ()):
        row = e187_run._validation_row(record, training)
        if guard_free:
            row["guard_free"] = record["summary"].get("e226_guard_free", {})
        rows.append(row)
    return rows


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    seeds = []
    for seed in p.MODEL_SEEDS:
        current_path = _summary_path(output, seed)
        reference_path = _summary_path(p.REFERENCE_OUTPUT, seed)
        if not current_path.is_file():
            seeds.append({
                "model_seed": seed,
                "status": "not_started",
                "paired_checkpoints": [],
            })
            continue
        require(reference_path.is_file(), f"E218 seed {seed} summary is missing")
        current = e187_run._read(current_path)
        reference = e187_run._read(reference_path)
        new_by_ep = {
            int(row["checkpoint_episode"]): row
            for row in _validation_rows(current, guard_free=True)
        }
        old_by_ep = {
            int(row["checkpoint_episode"]): row
            for row in _validation_rows(reference, guard_free=False)
        }
        paired = []
        for episode in sorted(set(new_by_ep) & set(old_by_ep)):
            new = new_by_ep[episode]
            old = old_by_ep[episode]
            telemetry = new["guard_free"]
            strict_rate = float(new["strict_completion_rate"])
            max_run = int(telemetry.get(
                "maximum_longest_reconfigure_run_without_accept_or_deliver", 0
            ))
            cap_hits = int(telemetry.get("step_limit_hits", 0))
            j_regression = float(new["J"]) / float(old["J"]) - 1.0
            r_regression = (
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
                "rehandles_per_100_regression": r_regression,
                "strict_completion_rate": strict_rate,
                "step_limit_hits": cap_hits,
                "maximum_nonprogress_rehandle_run": max_run,
                "liveness_pass": bool(
                    strict_rate == 1.0
                    and cap_hits == 0
                    and max_run <= p.MAX_ALLOWED_NONPROGRESS_REHANDLE_RUN
                ),
                "tracking_pass": bool(
                    j_regression <= p.MAX_J_REGRESSION_FRACTION
                    and r_regression <= p.MAX_REHANDLE_PER_100_REGRESSION
                ),
            })
        completed = int(current["completed_training_episodes"])
        seeds.append({
            "model_seed": seed,
            "status": "complete" if completed == p.TOTAL_EPISODES else "partial",
            "completed_training_episodes": completed,
            "paired_checkpoints": paired,
            "early_read": [
                row for row in paired
                if row["checkpoint_episode"] in p.EARLY_READ_CHECKPOINTS
            ],
            "all_observed_liveness_gates_passed": bool(paired) and all(
                row["liveness_pass"] for row in paired
            ),
            "all_observed_tracking_gates_passed": bool(paired) and all(
                row["tracking_pass"] for row in paired
            ),
        })
    report = {
        "method_conformance": validate_protocol(contract, source=p.PROTOCOL),
        "status": (
            "complete" if all(row["status"] == "complete" for row in seeds)
            else "partial"
        ),
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "seeds": seeds,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    report_path = output.resolve() / "e227-report.json"
    e187_run._write(report_path, report)
    text = render_validated_conformance_audit(contract)
    text += "\nRESULTS\n" + "=" * 72 + "\n"
    text += json.dumps({
        "status": report["status"],
        "seeds": seeds,
    }, indent=2, sort_keys=True) + "\n"
    report_text = output.resolve() / "REPORT.txt"
    temporary = report_text.with_suffix(".txt.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(report_text)
    return {
        "status": report["status"],
        "seeds": [
            {
                "model_seed": row["model_seed"],
                "status": row["status"],
                "paired_checkpoints": len(row["paired_checkpoints"]),
            }
            for row in seeds
        ],
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
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=60)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "smoke":
        require(args.model_seed is not None, "smoke requires --model-seed 1 or 2")
        result = smoke(args.output, args.model_seed, args.device)
    elif args.command == "train-seed":
        require(
            args.model_seed is not None,
            "train-seed requires --model-seed 1 or 2",
        )
        result = train_seed(
            args.output, args.model_seed, args.through_episode, args.device
        )
    elif args.command == "status":
        result = status(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

