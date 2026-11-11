#!/usr/bin/env python3
"""Prepare and run E270's paired 8x8 prefilled certification curve."""

from __future__ import annotations

import argparse
import builtins
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from PSLAP import viability_candidates as vc
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import BoundedEventDeferRule
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import legal
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1.roundoff import (
    repaired_audit,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    instrumentation as e191_instrumentation,
    protocol as e191,
    stage as e191_stage,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    instrumentation as e187_instrumentation,
)
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import (
    instrumentation as e195_instrumentation,
    protocol as e195,
    stage as e195_stage,
)
from experiments.conditioned_vcg.E198_e195_shadow_certification_ablation_257k_v1 import (
    frontier as e198_frontier,
)
from experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1 import (
    scenario as e243_scenario,
)
from experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1.tracker import (
    TrackedE195Agent,
)
from experiments.conditioned_vcg.E247_kim2020_objective_matched_a3c_306k_v1.source import (
    ObjectiveMatchedKimSource,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
)

from . import protocol as p, scenario
from .a3c_ranker import MatchedA3CPlacementAgent
from .tracker import MatchedDynamicE191Agent, PrefillTracker, TrackedE191Agent


class E270Error(RuntimeError):
    pass


MAX_UNKNOWN_DISCARD_FRACTION = 0.25


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E270Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E270 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E270 JSON object: {path}")
    return value


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _payload(model_seed: int) -> dict:
    identity = p.checkpoint_identity(model_seed)
    return torch.load(
        identity["source_checkpoint"], map_location="cpu", weights_only=False
    )


def _a3c_payload(size: str, model_seed: int) -> dict:
    identity = p.a3c_checkpoint_identity(size, model_seed)
    return torch.load(
        identity["source_checkpoint"], map_location="cpu", weights_only=False
    )


def _five_shell_payload() -> dict:
    return torch.load(
        p.five_shell_identity()["source_checkpoint"],
        map_location="cpu", weights_only=False,
    )


def _search_and_liveness_payload(payload: dict):
    search = payload["viability_search"]
    liveness = payload["liveness_rule"]
    require(
        int(search["max_nodes"]) == 20_000
        and str(search["search_order"]) == "goal_directed",
        "E270 requires the finalized B=20000 goal-directed verifier",
    )
    return (
        ViabilitySearchConfig(
            max_depth=search["max_depth"],
            max_nodes=int(search["max_nodes"]),
            max_primitive_steps=search["max_primitive_steps"],
            reserve_queue_cells=bool(search["reserve_queue_cells"]),
            search_order=str(search["search_order"]),
        ),
        BoundedEventDeferRule(
            max_option_steps=int(liveness["max_option_steps"]),
            max_consecutive_defer_decisions=int(
                liveness["max_consecutive_defer_decisions"]
            ),
        ),
    )


def _search_and_liveness(model_seed: int = 0):
    return _search_and_liveness_payload(_payload(model_seed))


def _screen_search() -> ViabilitySearchConfig:
    return ViabilitySearchConfig(
        max_depth=None,
        max_nodes=20_000,
        max_primitive_steps=None,
        reserve_queue_cells=True,
        search_order="goal_directed",
    )


def _certificate_row(certificate, elapsed: float | None = None) -> dict:
    row = {
        "status": certificate.status.value,
        "exhaustive": bool(certificate.exhaustive),
        "reason": str(certificate.reason),
        "witness_length": len(certificate.witness),
        "witness_primitive_steps": certificate.witness_primitive_steps,
        "explored_nodes": int(certificate.explored_nodes),
        "generated_states": int(certificate.generated_states),
        "frontier_states": int(certificate.frontier_states),
        "search_order": str(certificate.search_order),
    }
    if elapsed is not None:
        row["analysis_seconds"] = float(elapsed)
    return row


def _screen_target(arguments: tuple[int, int]) -> dict:
    """Process-safe deterministic search for one witnessed panel start."""

    seed, fill = (int(arguments[0]), int(arguments[1]))
    counts = Counter()
    total_seconds = 0.0
    for attempt in range(scenario.MAX_CANDIDATE_ATTEMPTS):
        record = scenario.candidate_record(seed, fill, attempt)
        env = scenario.Prefilled8x8Env(scenario.SCENARIO, e191.SPEC, record)
        instance = env.sample_episode_instance(seed=seed)
        env.reset(instance=instance)
        state = legal.current_state_from_env(env, _screen_search())
        certificate, _cache_hit, elapsed = vc._analyze_cached(
            state, _screen_search(), vc.ViabilityCertificateCache()
        )
        counts[certificate.status.value] += 1
        total_seconds += float(elapsed)
        if certificate.status is not ViabilityStatus.SAFE:
            continue
        require(bool(certificate.witness), "E270 nonempty start lacks a witness")
        record.update({
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "instance_json": instance.to_json(),
            "storage_steps_needed": list(instance.storage_steps_needed),
            "storage_steps_elapsed": [0] * fill,
            "initial_certificate": _certificate_row(certificate, elapsed),
            "candidate_screen": {
                "attempts_through_acceptance": attempt + 1,
                "status_counts": dict(counts),
                "unsafe_discards": int(counts[ViabilityStatus.UNSAFE.value]),
                "unknown_discards": int(counts[ViabilityStatus.UNKNOWN.value]),
                "analysis_seconds": total_seconds,
            },
        })
        return record
    raise E270Error(
        f"E270 could not find SAFE start seed={seed} fill={fill} within "
        f"{scenario.MAX_CANDIDATE_ATTEMPTS} candidates; counts={dict(counts)}"
    )


def _aggregate_screening(records: list[dict]) -> dict:
    by_fill = {}
    authorized = True
    for fill in scenario.INITIAL_FILLS:
        subset = [row for row in records if int(row["initial_fill"]) == fill]
        counts = Counter()
        seconds = 0.0
        for row in subset:
            counts.update(row["candidate_screen"]["status_counts"])
            seconds += float(row["candidate_screen"]["analysis_seconds"])
        total = sum(counts.values())
        unknown_fraction = (
            0.0 if total == 0 else counts[ViabilityStatus.UNKNOWN.value] / total
        )
        within_bound = unknown_fraction <= MAX_UNKNOWN_DISCARD_FRACTION
        authorized = authorized and within_bound
        by_fill[str(fill)] = {
            "accepted_states": len(subset),
            "candidate_status_counts": dict(counts),
            "candidate_attempts": total,
            "unsafe_discards": int(counts[ViabilityStatus.UNSAFE.value]),
            "unknown_discards": int(counts[ViabilityStatus.UNKNOWN.value]),
            "unknown_discard_fraction": unknown_fraction,
            "unknown_bias_bound": MAX_UNKNOWN_DISCARD_FRACTION,
            "within_predeclared_unknown_bias_bound": within_bound,
            "screen_seconds": seconds,
        }
    return {
        "by_fill": by_fill,
        "evaluation_authorized": authorized,
        "authorization_rule": (
            "UNKNOWN/(SAFE+UNSAFE+UNKNOWN) <= 0.25 at every fill; otherwise "
            "do not evaluate and redesign the high-fill endpoint near 85%"
        ),
    }


def _panel_records(workers: int) -> list[dict]:
    targets = [(seed, scenario.initial_fill(seed)) for seed in scenario.SEEDS]
    if int(workers) <= 1:
        return [_screen_target(item) for item in targets]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(_screen_target, targets))


def _manifest(contract: dict, records: list[dict]) -> dict:
    screening = _aggregate_screening(records)
    core = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "records": records,
        "state_screening": screening,
        "all_initial_states_have_completion_witness": all(
            row["initial_certificate"]["status"] == ViabilityStatus.SAFE.value
            and int(row["initial_certificate"]["witness_length"]) > 0
            for row in records
        ),
        "test_panel_opened": False,
    }
    return {**core, "manifest_sha256": p.digest(core)}


def _validate_manifest(value: dict, contract: dict) -> None:
    core = dict(value)
    recorded = core.pop("manifest_sha256", None)
    require(recorded == p.digest(core), "E270 state manifest hash is invalid")
    require(
        value.get("contract_sha256") == contract["contract_sha256"]
        and value.get("all_initial_states_have_completion_witness") is True
        and len(value.get("records", ())) == len(scenario.SEEDS),
        "E270 state manifest is incomplete",
    )
    require(
        value.get("state_screening", {}).get("evaluation_authorized") is True,
        "E270 evaluation is blocked by the predeclared UNKNOWN-discard bias rule",
    )


def prepare(output: Path, workers: int = 1) -> dict:
    output = output.resolve()
    expected = p.contract()
    render_validated_conformance_audit(expected)
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    if contract_path.is_file():
        require(_read(contract_path) == expected, "E270 contract or sources changed")
        manifest = _read(manifest_path)
        _validate_manifest(manifest, expected)
    else:
        nonlog = [] if not output.exists() else [
            item for item in output.iterdir() if item.name != "logs"
        ]
        require(not nonlog, "E270 refuses a nonempty uncontracted output")
        records = _panel_records(workers)
        manifest = _manifest(expected, records)
        _write(contract_path, expected)
        _write(manifest_path, manifest)
        _validate_manifest(manifest, expected)
    screening = manifest["state_screening"]
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "initial_states_verified": len(manifest["records"]),
        "screening": screening,
        "expected_rows": p.EXPECTED_ROWS,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    expected = p.contract()
    require(
        _read(output.resolve() / "contract.json") == expected,
        "E270 contract, checkpoints, ranker, or sources changed",
    )
    manifest = _read(output.resolve() / "instances.json")
    _validate_manifest(manifest, expected)
    return expected, manifest


def _record(manifest: dict, seed: int) -> dict:
    matches = [row for row in manifest["records"] if int(row["seed"]) == int(seed)]
    require(len(matches) == 1, "E270 state record is not unique")
    return matches[0]


def _load_agent(model_seed: int, device: str, arm: str, tracker: PrefillTracker):
    cls = MatchedDynamicE191Agent if arm in p.DYNAMIC_ARMS else TrackedE191Agent
    agent = cls.from_checkpoint(
        _payload(model_seed), device=device, resumable=False, seed=model_seed
    )
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    if arm in p.DYNAMIC_ARMS:
        agent.configure_e270(tracker)
    else:
        agent.configure_e233(tracker)
    return agent


def _frontier_wrapper(
    original, tracker: PrefillTracker, arm: str, *, stage_module=e191_stage,
):
    accelerated = stage_module._accelerated_enumerator(original)

    def wrapped(env, **kwargs):
        certified = accelerated(env, **kwargs)
        cache = kwargs.get("cache")
        require(cache is not None, "E270 requires a per-episode certificate cache")
        physical = e198_frontier.expose_physical_snapshot(
            env,
            certified,
            search_config=kwargs["search_config"],
            liveness_rule=kwargs["liveness_rule"],
            consecutive_defer_decisions=kwargs["consecutive_defer_decisions"],
            cache=cache,
            reserved_cells=kwargs.get("reserved_cells", ()),
            max_replans=kwargs.get("max_replans", 8),
        )
        tracker.record_frontier(certified, physical)
        return physical if arm in p.UNFILTERED_ARMS else certified

    return wrapped


def _failure_class(strict: bool, reason, steps: int, monitor: dict) -> str:
    if strict:
        return "completed"
    text = "" if reason is None else str(reason).lower()
    if monitor.get("guard_witness_unavailable", 0):
        return "guard_witness_unavailable"
    if int(steps) >= int(scenario.MAX_STEPS):
        return "step_limit"
    if monitor.get("first_exhaustive_unsafe_current") is not None:
        return "stranded_block"
    if any(word in text for word in ("no viable", "empty", "infeasible", "deadlock")):
        return "deadlock"
    return "other_method_failure"


def _run_one(*, arm: str, model_seed: int, record: dict, device: str, protocol: str) -> dict:
    require(arm in p.ARMS, "invalid E270 arm")
    tracker = PrefillTracker(arm)
    agent_seed = 0 if arm in p.DYNAMIC_ARMS else int(model_seed)
    agent = _load_agent(agent_seed, device, arm, tracker)
    env = scenario.Prefilled8x8Env(scenario.SCENARIO, e191.SPEC, record)
    env.current_episode = scenario.SEEDS.index(int(record["seed"])) + 1 if int(record["seed"]) in scenario.SEEDS else 0
    search, liveness = _search_and_liveness(agent_seed)
    original_print = builtins.print

    def e270_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith("E191 "):
            values = ("E270 " + values[0][len("E191 "):], *values[1:])
        return original_print(*values, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            audit, "enumerate_viability_candidates",
            _frontier_wrapper(audit.enumerate_viability_candidates, tracker, arm),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", e191_instrumentation.single_objective_rescore
        ))
        stack.enter_context(patch.object(builtins, "print", e270_print))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        stack.enter_context(repaired_audit())
        run = e191_instrumentation.run_episode(
            agent,
            env,
            objective_spec=e191.SPEC,
            instance_seed=int(record["seed"]),
            training=False,
            max_steps=scenario.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
            run_protocol=protocol,
        )
    monitor = tracker.audit()
    initial = monitor.get("initial_current_certificate") or {}
    require(
        initial.get("status") == ViabilityStatus.SAFE.value
        and int(initial.get("witness_length", 0)) > 0,
        f"E270 seed {record['seed']} did not begin with a concrete witness",
    )
    strict = bool(run.get("strict_method_success"))
    metrics = dict(run.get("e187_metrics", {}))
    if metrics:
        metrics["strict_complete"] = strict
    steps = int(run.get("steps", env.time_steps))
    dynamic_audit = None
    if arm in p.DYNAMIC_ARMS:
        dynamic_audit = {
            "matched_policy": agent.MATCHED_POLICY,
            "ranker_sha256": p.dynamic_ranker_sha256(),
            "choices": list(agent.e270_dynamic.choice_audit),
        }
    return {
        "status": "complete",
        "protocol": protocol,
        "arm": arm,
        "model_seed": None if arm in p.DYNAMIC_ARMS else int(model_seed),
        "adapter_checkpoint_seed": int(agent_seed),
        "instance_seed": int(record["seed"]),
        "initial_fill": int(record["initial_fill"]),
        "initial_fill_fraction": int(record["initial_fill"]) / scenario.CAPACITY,
        "initial_layout": list(record["initial_layout"]),
        "initial_completion_witness": initial,
        "strict_complete": strict,
        "failure_class": _failure_class(
            strict, run.get("method_failure_reason"), steps, monitor
        ),
        "method_failure_reason": run.get("method_failure_reason"),
        "steps_at_stop": steps,
        "delivered_at_stop": int(run.get("delivery_count", 0)),
        "metrics": metrics,
        "run": run,
        "certification_monitor": monitor,
        "dynamic_ranker_audit": dynamic_audit,
        "test_panel_opened": False,
    }


def _a3c_environment(size: str, record: dict):
    if size == "8x8":
        return scenario.Prefilled8x8Env(scenario.SCENARIO, e191.SPEC, record)
    if size == "5x5":
        return e243_scenario.Prefilled5x5Env(
            e243_scenario.SCENARIO, e195.SPEC
        )
    raise E270Error(f"invalid A3C size: {size}")


def _load_a3c_agent(
    size: str, model_seed: int, device: str, tracker: PrefillTracker, env,
):
    if size == "8x8":
        shell_payload = _payload(0)
        base = TrackedE191Agent.from_checkpoint(
            shell_payload, device=device, resumable=False, seed=0
        )
    else:
        shell_payload = _five_shell_payload()
        base = TrackedE195Agent.from_checkpoint(
            shell_payload, device=device, resumable=False, seed=0
        )
    base.set_epsilon(0.0)
    base.Q_local.eval()
    base.Q_target.eval()
    base.configure_e233(tracker)
    identity = p.a3c_checkpoint_identity(size, model_seed)
    source = ObjectiveMatchedKimSource.from_checkpoint(
        env,
        _a3c_payload(size, model_seed),
        learning_enabled=False,
        device=device,
        seed=int(model_seed),
        policy_seed=329_270_000 + int(model_seed),
    )
    return MatchedA3CPlacementAgent(
        base, source, tracker,
        policy_seed=329_270_000 + int(model_seed),
        checkpoint_sha256=identity["source_checkpoint_sha256"],
    ), shell_payload


def _a3c_failure_class(strict: bool, reason, steps: int, monitor: dict, max_steps: int) -> str:
    if strict:
        return "completed"
    selected = monitor.get("selected_successor_status_counts", {})
    if int(selected.get("UNSAFE", 0)) + int(selected.get("UNKNOWN", 0)) > 0:
        return "unsafe_selection"
    if int(steps) >= int(max_steps):
        return "step_limit_stall"
    return _failure_class(strict, reason, steps, monitor)


def _run_a3c_one(
    *, size: str, arm: str, model_seed: int, instance_seed: int,
    record: dict, device: str, protocol: str,
) -> dict:
    require(size in p.A3C_SIZES and arm in p.A3C_ARMS, "invalid E270 A3C route")
    tracker = PrefillTracker(arm)
    env = _a3c_environment(size, record)
    env.current_episode = 0
    agent, shell_payload = _load_a3c_agent(
        size, model_seed, device, tracker, env
    )
    search, liveness = _search_and_liveness_payload(shell_payload)
    stage_module = e191_stage if size == "8x8" else e195_stage
    instrumentation = (
        e191_instrumentation if size == "8x8" else e195_instrumentation
    )
    objective = e191.SPEC if size == "8x8" else e195.SPEC
    max_steps = scenario.MAX_STEPS if size == "8x8" else e243_scenario.MAX_STEPS
    original_print = builtins.print

    def e270_print(*values, **kwargs):
        if values and isinstance(values[0], str):
            first = values[0]
            for prefix in ("E191 ", "E195 "):
                if first.startswith(prefix):
                    values = ("E270-A3C " + first[len(prefix):], *values[1:])
                    break
        return original_print(*values, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            audit, "enumerate_viability_candidates",
            _frontier_wrapper(
                audit.enumerate_viability_candidates, tracker, arm,
                stage_module=stage_module,
            ),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", instrumentation.single_objective_rescore
        ))
        # E187's instrumentation deliberately rejects composition wrappers.
        # Admit this authenticated wrapper without weakening the environment
        # type check or changing any episode semantics.
        stack.enter_context(patch.object(
            e187_instrumentation,
            "WaitCorrectedQueueAgent",
            (e187_instrumentation.WaitCorrectedQueueAgent,
             MatchedA3CPlacementAgent),
        ))
        stack.enter_context(patch.object(builtins, "print", e270_print))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        stack.enter_context(repaired_audit())
        run = instrumentation.run_episode(
            agent,
            env,
            objective_spec=objective,
            instance_seed=int(instance_seed),
            training=False,
            max_steps=max_steps,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
            run_protocol=protocol,
        )
    monitor = tracker.audit()
    initial = monitor.get("initial_current_certificate") or {}
    require(
        initial.get("status") == ViabilityStatus.SAFE.value
        and int(initial.get("witness_length", 0)) > 0,
        f"E270 A3C {size}/{instance_seed} lacks an initial witness",
    )
    strict = bool(run.get("strict_method_success"))
    metrics = dict(run.get("e187_metrics", {}))
    if metrics:
        metrics["strict_complete"] = strict
    steps = int(run.get("steps", env.time_steps))
    fill = (
        int(record["initial_fill"])
        if size == "8x8" else e243_scenario.initial_fill(instance_seed)
    )
    capacity = scenario.CAPACITY if size == "8x8" else e243_scenario.CAPACITY
    return {
        "status": "complete",
        "protocol": protocol,
        "size": size,
        "arm": arm,
        "model_seed": int(model_seed),
        "instance_seed": int(instance_seed),
        "initial_fill": fill,
        "initial_fill_fraction": fill / capacity,
        "initial_completion_witness": initial,
        "strict_complete": strict,
        "failure_class": _a3c_failure_class(
            strict, run.get("method_failure_reason"), steps, monitor, max_steps
        ),
        "method_failure_reason": run.get("method_failure_reason"),
        "steps_at_stop": steps,
        "delivered_at_stop": int(run.get("delivery_count", 0)),
        "metrics": metrics,
        "run": run,
        "certification_monitor": monitor,
        "a3c_ranker_audit": agent.audit(),
        "test_panel_opened": False,
    }


def _row_path(output: Path, arm: str, seed: int, model_seed=None) -> Path:
    if arm in p.VCG_ARMS:
        return (
            output.resolve() / "rows" / arm / f"model-seed-{int(model_seed)}"
            / f"instance-{int(seed)}.json"
        )
    return output.resolve() / "rows" / arm / f"instance-{int(seed)}.json"


def _validate_row(row: dict, contract: dict, arm: str, seed: int, model_seed=None) -> None:
    expected = {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "arm": arm,
        "model_seed": None if arm in p.DYNAMIC_ARMS else int(model_seed),
        "instance_seed": int(seed),
        "initial_fill": scenario.initial_fill(seed),
        "test_panel_opened": False,
    }
    mismatches = {
        key: (row.get(key), value) for key, value in expected.items()
        if row.get(key) != value
    }
    require(not mismatches, f"E270 saved row differs: {mismatches!r}")
    require(
        row.get("initial_completion_witness", {}).get("status") == "SAFE",
        "E270 row lacks its initial completion witness",
    )
    if arm in p.DYNAMIC_ARMS:
        require(
            row.get("dynamic_ranker_audit", {}).get("ranker_sha256")
            == contract["dynamic_ranker_sha256"],
            "E270 Dynamic row used another ranker",
        )
    if arm in p.UNFILTERED_ARMS:
        require(
            row.get("certification_monitor", {}).get("monitor_is_non_authoritative") is True,
            "E270 unfiltered row lacks its passive monitor declaration",
        )


def _evaluate_group(output: Path, arms: tuple[str, ...], model_seed, device: str) -> dict:
    contract, manifest = authenticate(output)
    new = reused = 0
    for arm in arms:
        for seed in scenario.SEEDS:
            path = _row_path(output, arm, seed, model_seed)
            if path.is_file():
                _validate_row(_read(path), contract, arm, seed, model_seed)
                reused += 1
                continue
            record = _record(manifest, seed)
            label = "dynamic" if model_seed is None else f"model_seed={model_seed}"
            print(
                f"E270 start arm={arm} {label} fill={record['initial_fill']}/35 "
                f"instance={seed}", flush=True,
            )
            row = _run_one(
                arm=arm,
                model_seed=0 if model_seed is None else int(model_seed),
                record=record,
                device=device,
                protocol=p.PROTOCOL,
            )
            _validate_row(row, contract, arm, seed, model_seed)
            _write(path, row)
            new += 1
            print(
                f"E270 done arm={arm} {label} fill={record['initial_fill']}/35 "
                f"instance={seed} strict={int(row['strict_complete'])} "
                f"failure={row['failure_class']}", flush=True,
            )
    return {
        "status": "complete",
        "model_seed": model_seed,
        "arms": list(arms),
        "new_episode_runs": new,
        "reused_rows": reused,
        "rows": new + reused,
        "test_panel_opened": False,
    }


def evaluate_vcg(output: Path, model_seed: int, device: str) -> dict:
    require(int(model_seed) in p.MODEL_SEEDS, "invalid E270 VCG model seed")
    return _evaluate_group(output, p.VCG_ARMS, int(model_seed), device)


def evaluate_dynamic(output: Path, device: str) -> dict:
    return _evaluate_group(output, p.DYNAMIC_ARMS, None, device)


def _a3c_row_path(
    output: Path, size: str, arm: str, model_seed: int, instance_seed: int,
) -> Path:
    return (
        output.resolve() / "rows" / "a3c" / size / arm
        / f"model-seed-{int(model_seed)}" / f"instance-{int(instance_seed)}.json"
    )


def _validate_a3c_row(
    row: dict, contract: dict, size: str, arm: str,
    model_seed: int, instance_seed: int,
) -> None:
    expected = {
        "status": "complete", "protocol": p.PROTOCOL, "size": size,
        "arm": arm, "model_seed": int(model_seed),
        "instance_seed": int(instance_seed), "test_panel_opened": False,
    }
    mismatch = {
        key: (row.get(key), value) for key, value in expected.items()
        if row.get(key) != value
    }
    require(not mismatch, f"E270 A3C saved row differs: {mismatch}")
    audit_row = row.get("a3c_ranker_audit", {})
    require(
        audit_row.get("whole_controller_filter_contract") is True
        and audit_row.get("vcg_guard_used") is False,
        "E270 A3C route violated its controller contract",
    )
    require(
        audit_row.get("certification_filter_enabled")
        is (arm == "a3c_certified"),
        "E270 A3C certification authority differs from its arm",
    )
    expected_sha = contract["a3c_checkpoints"][
        f"{size}/seed-{int(model_seed)}"
    ]["source_checkpoint_sha256"]
    require(
        audit_row.get("checkpoint_sha256") == expected_sha,
        "E270 A3C row used another checkpoint",
    )
    if arm == "a3c_physical_unfiltered":
        require(
            row.get("certification_monitor", {}).get(
                "monitor_is_non_authoritative"
            ) is True,
            "E270 unfiltered A3C row lacks its passive monitor declaration",
        )


def evaluate_a3c(output: Path, size: str, model_seed: int, device: str) -> dict:
    require(size in p.A3C_SIZES, "invalid E270 A3C size")
    require(int(model_seed) in p.MODEL_SEEDS, "invalid E270 A3C model seed")
    contract, manifest = authenticate(output)
    if size == "8x8":
        targets = [
            (int(seed), _record(manifest, seed)) for seed in scenario.SEEDS
        ]
    else:
        # Authenticate the exact E263 state set even though the deterministic
        # scenario can reconstruct every state from its seed.
        source = _read(Path(contract["five_by_five_panel"]["source_instances"]))
        require(
            p.sha256(Path(contract["five_by_five_panel"]["source_instances"]))
            == contract["five_by_five_panel"]["source_instances_sha256"],
            "E270 5x5 source manifest changed",
        )
        source_seeds = {int(row["seed"]) for row in source["records"]}
        require(
            source_seeds == set(e243_scenario.SEEDS),
            "E270 5x5 source panel seeds differ",
        )
        targets = [(int(seed), {"initial_fill": e243_scenario.initial_fill(seed)})
                   for seed in e243_scenario.SEEDS]
    new = reused = 0
    for arm in p.A3C_ARMS:
        for instance_seed, record in targets:
            path = _a3c_row_path(
                output, size, arm, model_seed, instance_seed
            )
            if path.is_file():
                _validate_a3c_row(
                    _read(path), contract, size, arm, model_seed, instance_seed
                )
                reused += 1
                continue
            print(
                f"E270 A3C start size={size} arm={arm} model_seed={model_seed} "
                f"instance={instance_seed}", flush=True,
            )
            row = _run_a3c_one(
                size=size, arm=arm, model_seed=model_seed,
                instance_seed=instance_seed, record=record,
                device=device, protocol=p.PROTOCOL,
            )
            _validate_a3c_row(
                row, contract, size, arm, model_seed, instance_seed
            )
            _write(path, row)
            new += 1
            print(
                f"E270 A3C done size={size} arm={arm} model_seed={model_seed} "
                f"instance={instance_seed} strict={int(row['strict_complete'])} "
                f"failure={row['failure_class']}", flush=True,
            )
    return {
        "status": "complete", "size": size, "model_seed": int(model_seed),
        "arms": list(p.A3C_ARMS), "new_episode_runs": new,
        "reused_rows": reused, "rows": new + reused,
        "test_panel_opened": False,
    }


def _metric(row: dict, name: str):
    value = row.get("metrics", {}).get(name)
    return None if value is None else float(value)


def _rows(output: Path, contract: dict) -> list[dict]:
    rows = []
    for arm in p.VCG_ARMS:
        for model_seed in p.MODEL_SEEDS:
            for seed in scenario.SEEDS:
                path = _row_path(output, arm, seed, model_seed)
                if path.is_file():
                    row = _read(path)
                    _validate_row(row, contract, arm, seed, model_seed)
                    rows.append(row)
    for arm in p.DYNAMIC_ARMS:
        for seed in scenario.SEEDS:
            path = _row_path(output, arm, seed)
            if path.is_file():
                row = _read(path)
                _validate_row(row, contract, arm, seed)
                rows.append(row)
    for size in p.A3C_SIZES:
        seeds = scenario.SEEDS if size == "8x8" else e243_scenario.SEEDS
        for arm in p.A3C_ARMS:
            for model_seed in p.MODEL_SEEDS:
                for instance_seed in seeds:
                    path = _a3c_row_path(
                        output, size, arm, model_seed, instance_seed
                    )
                    if path.is_file():
                        row = _read(path)
                        _validate_a3c_row(
                            row, contract, size, arm, model_seed, instance_seed
                        )
                        rows.append(row)
    return rows


def _summary(subset: list[dict]) -> dict:
    completed = [row for row in subset if row["strict_complete"]]
    j_values = [_metric(row, "J") for row in completed if _metric(row, "J") is not None]
    return {
        "cases": len(subset),
        "strict_completions": len(completed),
        "strict_completion_rate": None if not subset else len(completed) / len(subset),
        "failure_classes": dict(Counter(
            row["failure_class"] for row in subset if not row["strict_complete"]
        )),
        "unsafe_selections": sum(
            int(row["certification_monitor"]["selected_successor_status_counts"].get("UNSAFE", 0))
            for row in subset
        ),
        "unknown_selections": sum(
            int(row["certification_monitor"]["selected_successor_status_counts"].get("UNKNOWN", 0))
            for row in subset
        ),
        "mean_J_completed_only": None if not j_values else fmean(j_values),
    }


def _paired(
    rows: list[dict], certified_arm: str, unfiltered_arm: str,
    model_seed=None, fill=None,
) -> dict:
    def keep(row):
        seed_matches = model_seed is None or row.get("model_seed") == int(model_seed)
        fill_matches = fill is None or int(row["initial_fill"]) == int(fill)
        return seed_matches and fill_matches
    certified = {
        int(row["instance_seed"]): row for row in rows
        if row["arm"] == certified_arm and keep(row)
    }
    unfiltered = {
        int(row["instance_seed"]): row for row in rows
        if row["arm"] == unfiltered_arm and keep(row)
    }
    common = sorted(set(certified).intersection(unfiltered))
    joint = [
        seed for seed in common
        if certified[seed]["strict_complete"] and unfiltered[seed]["strict_complete"]
        and _metric(certified[seed], "J") is not None
        and _metric(unfiltered[seed], "J") is not None
    ]
    differences = [
        _metric(certified[seed], "J") - _metric(unfiltered[seed], "J")
        for seed in joint
    ]
    return {
        "paired_instances_present": len(common),
        "joint_completers": len(joint),
        "mean_certified_minus_unfiltered_J": (
            None if not differences else fmean(differences)
        ),
        "certified_lower_J": sum(value < 0 for value in differences),
        "equal_J": sum(value == 0 for value in differences),
        "certified_higher_J": sum(value > 0 for value in differences),
    }


def analyze(output: Path) -> dict:
    contract, manifest = authenticate(output)
    rows = _rows(output, contract)
    by_fill = {}
    for arm in p.ARMS:
        by_fill[arm] = {}
        model_seeds = p.MODEL_SEEDS if arm in p.VCG_ARMS else (None,)
        for model_seed in model_seeds:
            key = "dynamic" if model_seed is None else str(model_seed)
            by_fill[arm][key] = {}
            for fill in scenario.INITIAL_FILLS:
                subset = [
                    row for row in rows
                    if row["arm"] == arm
                    and row.get("model_seed") == model_seed
                    and int(row["initial_fill"]) == fill
                ]
                by_fill[arm][key][str(fill)] = _summary(subset)
    paired = {"vcg": {}, "dynamic": {}}
    for seed in p.MODEL_SEEDS:
        paired["vcg"][str(seed)] = {
            "overall": _paired(
                rows, "vcg_certified", "vcg_physical_unfiltered", seed
            ),
            "by_fill": {
                str(fill): _paired(
                    rows, "vcg_certified", "vcg_physical_unfiltered",
                    seed, fill,
                ) for fill in scenario.INITIAL_FILLS
            },
        }
    paired["dynamic"] = {
        "overall": _paired(
            rows, "dynamic_certified", "dynamic_physical_unfiltered"
        ),
        "by_fill": {
            str(fill): _paired(
                rows, "dynamic_certified", "dynamic_physical_unfiltered",
                fill=fill,
            ) for fill in scenario.INITIAL_FILLS
        },
    }
    unfiltered_seed_rates = {}
    for seed in p.MODEL_SEEDS:
        subset = [
            row for row in rows
            if row["arm"] == "vcg_physical_unfiltered" and row["model_seed"] == seed
        ]
        unfiltered_seed_rates[str(seed)] = (
            None if not subset else sum(row["strict_complete"] for row in subset) / len(subset)
        )
    a3c = {}
    for size in p.A3C_SIZES:
        fills = (
            scenario.INITIAL_FILLS if size == "8x8"
            else e243_scenario.INITIAL_FILLS
        )
        a3c[size] = {}
        for model_seed in p.MODEL_SEEDS:
            a3c[size][str(model_seed)] = {
                "overall": {
                    arm: _summary([
                        row for row in rows
                        if row.get("size") == size and row["arm"] == arm
                        and row["model_seed"] == model_seed
                    ]) for arm in p.A3C_ARMS
                },
                "paired_J_joint_completers": _paired(
                    [row for row in rows if row.get("size") == size],
                    "a3c_certified", "a3c_physical_unfiltered", model_seed,
                ),
                "by_fill": {
                    str(fill): {
                        arm: _summary([
                            row for row in rows
                            if row.get("size") == size and row["arm"] == arm
                            and row["model_seed"] == model_seed
                            and int(row["initial_fill"]) == int(fill)
                        ]) for arm in p.A3C_ARMS
                    } for fill in fills
                },
            }
    complete = len(rows) == p.EXPECTED_ROWS
    report = {
        "status": "complete" if complete else "partial",
        "protocol": p.PROTOCOL,
        "rows": len(rows),
        "expected_rows": p.EXPECTED_ROWS,
        "all_initial_states_witnessed": all(
            row["initial_certificate"]["status"] == "SAFE"
            for row in manifest["records"]
        ),
        "initial_state_screening": manifest["state_screening"],
        "dynamic_pair_one_ranker": {
            row["dynamic_ranker_audit"]["ranker_sha256"]
            for row in rows if row["arm"] in p.DYNAMIC_ARMS
        } in (set(), {contract["dynamic_ranker_sha256"]}),
        "by_arm_seed_fill": by_fill,
        "paired_J_joint_completers": paired,
        "vcg_unfiltered_completion_by_seed": unfiltered_seed_rates,
        "a3c_by_size_seed_fill": a3c,
        "test_panel_opened": False,
        "paper_evidence": complete,
    }
    _write(output.resolve() / p.REPORT_JSON, report)
    lines = [
        "# E270 8x8 prefilled certification curve", "",
        "| arm | seed | fill | cases | completion | failures | unsafe selected | mean J (completers) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in p.ARMS:
        model_seeds = p.MODEL_SEEDS if arm in p.VCG_ARMS else (None,)
        for model_seed in model_seeds:
            key = "dynamic" if model_seed is None else str(model_seed)
            for fill in scenario.INITIAL_FILLS:
                item = by_fill[arm][key][str(fill)]
                rate = item["strict_completion_rate"]
                mean_j = item["mean_J_completed_only"]
                lines.append(
                    f"| {arm} | {key} | {fill}/35 | {item['cases']} | "
                    f"{'n/a' if rate is None else f'{rate:.3f}'} | "
                    f"{sum(item['failure_classes'].values())} | "
                    f"{item['unsafe_selections']} | "
                    f"{'n/a' if mean_j is None else f'{mean_j:.1f}'} |"
                )
    lines.extend(["", "## A3C placement whole-controller certification", ""])
    lines.extend([
        "| size | arm | seed | fill | cases | completion | failures | unsafe selected | mean J (completers) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for size in p.A3C_SIZES:
        fills = scenario.INITIAL_FILLS if size == "8x8" else e243_scenario.INITIAL_FILLS
        for model_seed in p.MODEL_SEEDS:
            for fill in fills:
                for arm in p.A3C_ARMS:
                    item = a3c[size][str(model_seed)]["by_fill"][str(fill)][arm]
                    rate = item["strict_completion_rate"]
                    mean_j = item["mean_J_completed_only"]
                    lines.append(
                        f"| {size} | {arm} | {model_seed} | {fill} | {item['cases']} | "
                        f"{'n/a' if rate is None else f'{rate:.3f}'} | "
                        f"{sum(item['failure_classes'].values())} | "
                        f"{item['unsafe_selections']} | "
                        f"{'n/a' if mean_j is None else f'{mean_j:.1f}'} |"
                    )
    table = "\n".join(lines) + "\n"
    (output.resolve() / p.TABLE_NAME).write_text(table, encoding="utf-8")
    conformance = render_validated_conformance_audit(contract)
    (output.resolve() / p.REPORT_NAME).write_text(
        conformance + "\n\n" + table, encoding="utf-8"
    )
    return report


def status(output: Path) -> dict:
    root = output.resolve() / "rows"
    rows = list(root.rglob("*.json")) if root.is_dir() else []
    return {
        "status": "complete" if len(rows) == p.EXPECTED_ROWS else "partial",
        "rows": len(rows),
        "expected_rows": p.EXPECTED_ROWS,
        "percent_complete": 100.0 * len(rows) / p.EXPECTED_ROWS,
        "test_panel_opened": False,
    }


def gate(device: str) -> dict:
    contract = p.gate_contract()
    path = p.DEFAULT_GATE_OUTPUT / p.GATE_NAME
    if path.is_file():
        value = _read(path)
        core = dict(value)
        recorded = core.pop("manifest_sha256", None)
        require(recorded == p.digest(core), "E270 gate manifest hash is invalid")
        require(
            value.get("gate_contract_sha256") == contract["contract_sha256"],
            "E270 gate source identity changed",
        )
        return value
    record = _screen_target((scenario.GATE_SEED, scenario.GATE_FILL))
    routes = {}
    for arm in p.ARMS:
        row = _run_one(
            arm=arm, model_seed=0, record=record, device=device,
            protocol=p.GATE_PROTOCOL,
        )
        routes[arm] = {
            "route_executed": True,
            "strict_complete": bool(row["strict_complete"]),
            "failure_class": row["failure_class"],
            "initial_witness_present": (
                row["initial_completion_witness"]["status"] == "SAFE"
            ),
            "monitor_present": bool(row["certification_monitor"]),
            "dynamic_ranker_sha256": (
                None if arm not in p.DYNAMIC_ARMS
                else row["dynamic_ranker_audit"]["ranker_sha256"]
            ),
        }
    for size in p.A3C_SIZES:
        if size == "8x8":
            instance_seed = scenario.GATE_SEED
            a3c_record = record
        else:
            instance_seed = e243_scenario.SEEDS[0]
            a3c_record = {"initial_fill": e243_scenario.initial_fill(instance_seed)}
        for arm in p.A3C_ARMS:
            key = f"{size}/{arm}"
            row = _run_a3c_one(
                size=size, arm=arm, model_seed=0,
                instance_seed=instance_seed, record=a3c_record,
                device=device, protocol=p.GATE_PROTOCOL,
            )
            routes[key] = {
                "route_executed": True,
                "strict_complete": bool(row["strict_complete"]),
                "failure_class": row["failure_class"],
                "initial_witness_present": (
                    row["initial_completion_witness"]["status"] == "SAFE"
                ),
                "monitor_present": bool(row["certification_monitor"]),
                "whole_controller_certification": row["a3c_ranker_audit"][
                    "whole_controller_certification"
                ],
                "whole_controller_filter_contract": row["a3c_ranker_audit"][
                    "whole_controller_filter_contract"
                ],
                "vcg_guard_used": row["a3c_ranker_audit"]["vcg_guard_used"],
                "checkpoint_sha256": row["a3c_ranker_audit"][
                    "checkpoint_sha256"
                ],
            }
    passed = bool(
        all(item["route_executed"] and item["initial_witness_present"] and item["monitor_present"] for item in routes.values())
        and routes["dynamic_certified"]["dynamic_ranker_sha256"]
        == routes["dynamic_physical_unfiltered"]["dynamic_ranker_sha256"]
        == contract["dynamic_ranker_sha256"]
        and all(
            routes[f"{size}/{arm}"]["whole_controller_filter_contract"] is True
            and routes[f"{size}/{arm}"]["whole_controller_certification"]
            is (arm == "a3c_certified")
            and routes[f"{size}/{arm}"]["vcg_guard_used"] is False
            and routes[f"{size}/{arm}"]["checkpoint_sha256"]
            == contract["a3c_checkpoint_sha256"][f"{size}/seed-0"]
            for size in p.A3C_SIZES for arm in p.A3C_ARMS
        )
    )
    result = {
        "status": "pass" if passed else "fail",
        "protocol": p.GATE_PROTOCOL,
        "gate_contract_sha256": contract["contract_sha256"],
        "development_seed": scenario.GATE_SEED,
        "development_fill": scenario.GATE_FILL,
        "initial_state_screen": record["candidate_screen"],
        "routes": routes,
        "episode_runs": len(routes),
        "test_panel_opened": False,
    }
    result["manifest_sha256"] = p.digest(result)
    _write(path, result)
    require(passed, "E270 development integration gate failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "gate", "prepare", "evaluate-vcg", "evaluate-dynamic", "evaluate-a3c",
            "analyze", "status",
        ),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--size", choices=p.A3C_SIZES)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.command == "gate":
        value = gate(args.device)
    elif args.command == "prepare":
        value = prepare(args.output, args.workers)
    elif args.command == "evaluate-vcg":
        require(args.model_seed is not None, "evaluate-vcg requires --model-seed")
        value = evaluate_vcg(args.output, args.model_seed, args.device)
    elif args.command == "evaluate-dynamic":
        value = evaluate_dynamic(args.output, args.device)
    elif args.command == "evaluate-a3c":
        require(args.size is not None, "evaluate-a3c requires --size")
        require(args.model_seed is not None, "evaluate-a3c requires --model-seed")
        value = evaluate_a3c(
            args.output, args.size, args.model_seed, args.device
        )
    elif args.command == "analyze":
        value = analyze(args.output)
    else:
        value = status(args.output)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
