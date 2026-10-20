"""Readable objective-component and service instrumentation for E115."""

from __future__ import annotations

from statistics import fmean
from unittest.mock import patch

import run_vcg_objective_gamma_audit as audit
from train_viability_graph_smdp import epsilon_at

from .model import FactorizedConservativeAgent, learning_rate_at
from .rewards import COMPONENT_NAMES, episode_service_metrics, objective_cost


BASE_RUN_EPISODE = audit.run_objective_audit_episode
BASE_SUMMARIZE_RUNS = audit.summarize_objective_runs


def e115_rescore_run(run: dict, _objective_spec) -> None:
    """Adapt the inherited two-arm integrity hook to E115's third objective.

    The base runner's offline rescorer reconstructs legacy/base-dense delivery
    rewards from the realized timing deviations.  E115 intentionally retired
    both rewards, so its two required compatibility fields are the logged E115
    return.  Exactness is then checked independently from the four primitive
    component streams and, on completion, the terminal ET formula.
    """

    value = float(run["return"])
    run.update({
        "selected_objective": "release_clock_et_cost_v1",
        "selected_objective_return": value,
        # Retained only because older result consumers require the field. E115
        # does not define the old delivery/non-delivery reward decomposition.
        "non_delivery_return": value,
        "legacy_rescored_return": value,
        "dense_rescored_return": value,
        "dual_rescore_uses_fixed_realized_trajectory": False,
        "e115_compatibility_rescore": (
            "legacy summary fields contain the release-ET operator return"
        ),
    })


def run_episode(*args, **kwargs) -> dict:
    agent = args[0] if args else kwargs["agent"]
    env = args[1] if len(args) >= 2 else kwargs["env"]
    if not isinstance(agent, FactorizedConservativeAgent):
        raise TypeError("E115 episode received another agent architecture")
    agent.bind_context_environment(
        env,
        wait_cap=80,
        required_jobs=36,
        max_steps=int(kwargs["max_steps"]),
    )
    decisions_before = int(agent.decision_count)
    component_losses_before = len(agent.component_loss_history)
    anchor_pairs_before = int(agent.anchor_pairs_used)
    with patch.object(audit, "_rescore_run", e115_rescore_run):
        run = BASE_RUN_EPISODE(*args, **kwargs)
    decisions_after = int(agent.decision_count)
    service = episode_service_metrics(env)
    component_totals = {
        name: float(sum(record[index] for record in env.factorized_component_records))
        for index, name in enumerate(COMPONENT_NAMES)
    }
    reconstructed_return = -objective_cost(component_totals.values())
    if abs(reconstructed_return - float(run["return"])) > 1.0e-6:
        raise RuntimeError("E115 episode components do not reconstruct return")
    if service["strict_complete"] and not service[
        "terminal_costs_match_dense_accrual"
    ]:
        raise RuntimeError("E115 dense timing costs differ from terminal formula")

    training = bool(kwargs.get("training"))
    scheduled_epsilon = (
        epsilon_at(
            decisions_after,
            start=float(kwargs["epsilon_start"]),
            end=float(kwargs["epsilon_end"]),
            decay_decisions=int(kwargs["epsilon_decay_decisions"]),
            warmup_decisions=int(kwargs.get("epsilon_warmup_decisions", 0)),
        )
        if training else 0.0
    )
    new_component_losses = agent.component_loss_history[component_losses_before:]
    component_loss_mean = {
        name: (
            float(fmean(row[index] for row in new_component_losses))
            if new_component_losses else None
        )
        for index, name in enumerate(COMPONENT_NAMES)
    }
    run.update({
        "service_metrics": service,
        "factorized_component_totals": component_totals,
        "factorized_component_loss_mean": component_loss_mean,
        "operator_cost": float(service["objective_cost"]),
        "training_decision_count_before": decisions_before,
        "training_decision_count_after": decisions_after,
        "scheduled_epsilon_after": float(scheduled_epsilon),
        "learning_rate_after": float(learning_rate_at(decisions_after)),
        "anchor_pairs_used_in_episode": (
            int(agent.anchor_pairs_used) - anchor_pairs_before
        ),
    })

    def shown(value) -> str:
        return "n/a" if value is None else f"{float(value):.1f}"

    phase = "train" if training else "validation"
    episode = getattr(env, "current_episode", "?")
    print(
        f"E115 {phase} ep={episode} | J={service['objective_cost']:.1f} "
        f"return={run['return']:.1f} strict={int(run['strict_method_success'])} "
        f"steps={run['steps']} work={service['equipment_work']:.0f} "
        f"wait={shown(service['mean_wait_before_storage'])} "
        f"releaseMAE={shown(service['release_clock_mae'])} "
        f"T={shown(service['mean_release_tardiness_beyond_tolerance'])} "
        f"E={shown(service['mean_release_earliness_beyond_tolerance'])} "
        f"window={shown(service['within_tolerance_rate'])} "
        f"rehandles={service['physical_rehandles']:.0f} "
        f"decisions={decisions_after} eps={scheduled_epsilon:.3f} "
        f"lr={learning_rate_at(decisions_after):.1e}",
        flush=True,
    )
    return run


def _mean_service(runs, key: str):
    values = [run["service_metrics"].get(key) for run in runs]
    values = [float(value) for value in values if value is not None]
    return float(fmean(values)) if values else None


def summarize_runs(runs) -> dict:
    runs = tuple(runs)
    summary = BASE_SUMMARIZE_RUNS(runs)
    for source, destination in (
        ("mean_wait_before_storage", "mean_wait_before_storage"),
        ("max_wait_before_storage", "mean_max_wait_before_storage"),
        ("release_clock_mae", "mean_release_clock_mae"),
        ("release_clock_mean_signed_error", "mean_release_clock_signed_error"),
        ("mean_release_tardiness_beyond_tolerance", "mean_release_tardiness"),
        ("mean_release_earliness_beyond_tolerance", "mean_release_earliness"),
        ("within_tolerance_rate", "mean_within_tolerance_rate"),
        ("storage_clock_mae", "mean_storage_clock_mae"),
        ("equipment_work", "mean_equipment_work"),
        ("equipment_work_per_delivery", "mean_equipment_work_per_delivery"),
        ("physical_rehandles", "mean_physical_rehandles"),
        ("objective_cost", "mean_operator_cost"),
    ):
        summary[destination] = _mean_service(runs, source)
    summary["component_accounting_exact"] = all(
        int(run["service_metrics"]["component_record_count"])
        == int(run["steps"])
        and (
            not run["service_metrics"]["strict_complete"]
            or run["service_metrics"]["terminal_costs_match_dense_accrual"]
        )
        for run in runs
    )
    for name in COMPONENT_NAMES:
        summary[f"mean_component_{name}"] = float(fmean(
            run["factorized_component_totals"][name] for run in runs
        ))
    return summary


__all__ = ["e115_rescore_run", "run_episode", "summarize_runs"]
